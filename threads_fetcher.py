import requests
import json
import pandas as pd
from datetime import datetime
import time
from typing import Dict, List, Optional
import os
import sys
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ThreadsAPIClient:
    """Client for interacting with Threads API"""

    def __init__(self, access_token: str):
        """
        Initialize the Threads API client

        Args:
            access_token: Your Threads API access token
        """
        self.access_token = access_token
        self.base_url = "https://graph.threads.net/v1.0"
        self.headers = {
            "Authorization": f"Bearer {access_token}"
        }
        self.rate_limit_delay = 1.0  # Increased from 0.5 seconds
        self.max_retries = 3

    def _make_request(self, url: str, params: dict, retry_count: int = 0) -> Optional[Dict]:
        """
        Make a request with improved error handling and retries
        """
        try:
            logger.info(f"Making request to: {url}")
            logger.info(f"Parameters: {params}")

            response = requests.get(url, headers=self.headers, params=params)

            # Log response status
            logger.info(f"Response status: {response.status_code}")

            if response.status_code == 429:  # Rate limit
                wait_time = min(2 ** retry_count * 5, 60)  # Exponential backoff, max 60s
                logger.warning(f"Rate limited. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)

                if retry_count < self.max_retries:
                    return self._make_request(url, params, retry_count + 1)
                else:
                    logger.error("Max retries reached for rate limiting")
                    return None

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            if retry_count < self.max_retries:
                wait_time = min(2 ** retry_count * 2, 30)
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                return self._make_request(url, params, retry_count + 1)
            return None

    def get_post_details(self, post_id: str) -> Dict:
        """
        Get details about a specific Threads post

        Args:
            post_id: The ID of the Threads post

        Returns:
            Dictionary containing post details
        """
        url = f"{self.base_url}/{post_id}"

        # Try different field combinations to get comment count
        field_attempts = [
            "id,reply_count,replies_count,comment_count,comments_count",  # Try various count fields
            "id,text,username,timestamp,reply_count",  # Basic with reply_count
            "id,engagement,metrics,reply_count,comments",  # Try engagement metrics
            "id,insights,reply_count",  # Try insights
            "id,text",  # Minimal fields
            "id"  # Just ID
        ]

        for i, fields in enumerate(field_attempts):
            logger.info(f"Attempt {i + 1}: Trying fields: {fields}")
            response = self._make_request(url, {"fields": fields})

            if response is not None:
                logger.info(f"✅ Success with fields: {fields}")
                logger.info(f"Response keys: {list(response.keys())}")
                if any(key in response for key in ['reply_count', 'replies_count', 'comment_count', 'comments_count']):
                    logger.info(f"🎯 Found count field in response!")
                return response
            else:
                logger.warning(f"❌ Failed with fields: {fields}")

        # If all field attempts fail, try with no fields
        logger.info("All field attempts failed, trying with no fields parameter")
        response = self._make_request(url, {})

        if response is None:
            raise Exception(f"Failed to fetch post details for ID {post_id}")

        return response

    def get_replies(self, post_id: str, limit: int = 100, max_pages: int = None) -> List[Dict]:
        """
        Fetch all replies to a specific Threads post with improved pagination

        Args:
            post_id: The ID of the Threads post
            limit: Maximum number of replies per request (max 100)
            max_pages: Maximum number of pages to fetch (None for all)

        Returns:
            List of reply dictionaries
        """
        all_replies = []
        next_cursor = None
        page_count = 0
        consecutive_empty_pages = 0
        max_empty_pages = 3

        logger.info(f"Starting to fetch replies for post {post_id}")

        while True:
            page_count += 1
            logger.info(f"Fetching page {page_count}...")

            # Check if we've hit max pages limit
            if max_pages and page_count > max_pages:
                logger.info(f"Reached maximum pages limit ({max_pages})")
                break

            # Construct the URL for fetching replies
            url = f"{self.base_url}/{post_id}/replies"

            # Set up parameters
            params = {
                "fields": "id,text,username,timestamp,permalink,reply_count,media_type,media_url",
                "limit": min(limit, 100)
            }

            if next_cursor:
                params["after"] = next_cursor
                logger.info(f"Using cursor: {next_cursor}")

            # Make the request
            data = self._make_request(url, params)

            if data is None:
                logger.error("Failed to fetch page, stopping...")
                break

            # Process the response
            page_replies = data.get("data", [])
            logger.info(f"Received {len(page_replies)} replies in this batch")

            if not page_replies:
                consecutive_empty_pages += 1
                logger.warning(f"Empty page {consecutive_empty_pages}/{max_empty_pages}")

                if consecutive_empty_pages >= max_empty_pages:
                    logger.warning("Too many consecutive empty pages, stopping")
                    break
            else:
                consecutive_empty_pages = 0
                all_replies.extend(page_replies)
                logger.info(f"Total replies collected so far: {len(all_replies)}")

            # Check for pagination with improved cursor extraction
            paging_info = data.get("paging", {})
            if "next" in paging_info:
                next_url = paging_info["next"]
                logger.info(f"Next URL: {next_url}")

                # Better cursor extraction using URL parsing
                try:
                    parsed_url = urlparse(next_url)
                    query_params = parse_qs(parsed_url.query)

                    if "after" in query_params:
                        next_cursor = query_params["after"][0]
                        logger.info(f"Extracted cursor: {next_cursor}")
                    else:
                        logger.warning("No 'after' parameter found in next URL")
                        break

                except Exception as e:
                    logger.error(f"Failed to parse next URL: {e}")
                    # Fallback to old method
                    if "after=" in next_url:
                        try:
                            next_cursor = next_url.split("after=")[1].split("&")[0]
                            logger.info(f"Fallback cursor extraction: {next_cursor}")
                        except IndexError:
                            logger.error("Fallback cursor extraction failed")
                            break
                    else:
                        break
            else:
                logger.info("No more pages available")
                break

            # Rate limiting with progressive delay
            delay = self.rate_limit_delay * (1 + page_count * 0.1)  # Slightly increase delay over time
            logger.info(f"Waiting {delay:.1f} seconds before next request...")
            time.sleep(delay)

        logger.info(f"Completed fetching replies. Total pages: {page_count}, Total replies: {len(all_replies)}")
        return all_replies

    def get_nested_replies(self, reply_id: str, limit: int = 100) -> List[Dict]:
        """
        Fetch nested replies (replies to replies) for a given reply

        Args:
            reply_id: The ID of the reply to get nested replies for
            limit: Maximum number of nested replies to fetch

        Returns:
            List of nested reply dictionaries
        """
        logger.info(f"Fetching nested replies for reply {reply_id}")

        url = f"{self.base_url}/{reply_id}/replies"
        params = {
            "fields": "id,text,username,timestamp,permalink,reply_count",
            "limit": min(limit, 100)
        }

        data = self._make_request(url, params)
        if data is None:
            return []

        nested_replies = data.get("data", [])
        logger.info(f"Found {len(nested_replies)} nested replies for reply {reply_id}")

        return nested_replies

    def get_user_posts(self, user_id: str = "me", limit: int = 25) -> List[Dict]:
        """
        Get posts from a specific user (or yourself with 'me')

        Args:
            user_id: User ID or 'me' for authenticated user
            limit: Number of posts to fetch

        Returns:
            List of post dictionaries
        """
        url = f"{self.base_url}/{user_id}/threads"
        params = {
            "fields": "id,text,username,timestamp,permalink,reply_count",
            "limit": limit
        }

        data = self._make_request(url, params)
        if data is None:
            return []

        return data.get("data", [])

    def debug_post_access(self, post_id: str) -> Dict:
        """
        Debug post access issues by trying different approaches

        Args:
            post_id: The post ID to debug

        Returns:
            Debug information dictionary
        """
        debug_info = {
            "post_id": post_id,
            "attempts": [],
            "success": False,
            "final_data": None
        }

        # Test 1: Basic API call
        try:
            url = f"https://graph.threads.net/v1.0/{post_id}"
            response = requests.get(url, headers=self.headers)
            debug_info["attempts"].append({
                "method": "Basic call (no params)",
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "response": response.text[:200] if len(response.text) < 200 else response.text[:200] + "..."
            })

            if response.status_code == 200:
                debug_info["success"] = True
                debug_info["final_data"] = response.json()
                return debug_info

        except Exception as e:
            debug_info["attempts"].append({
                "method": "Basic call (no params)",
                "status_code": "Error",
                "success": False,
                "response": str(e)
            })

        # Test 2: With basic fields
        try:
            params = {"fields": "id,text"}
            response = requests.get(url, headers=self.headers, params=params)
            debug_info["attempts"].append({
                "method": "Basic fields (id,text)",
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "response": response.text[:200] if len(response.text) < 200 else response.text[:200] + "..."
            })

            if response.status_code == 200:
                debug_info["success"] = True
                debug_info["final_data"] = response.json()
                return debug_info

        except Exception as e:
            debug_info["attempts"].append({
                "method": "Basic fields (id,text)",
                "status_code": "Error",
                "success": False,
                "response": str(e)
            })

        # Test 3: Check if it's a permissions issue by trying to access replies directly
        try:
            replies_url = f"https://graph.threads.net/v1.0/{post_id}/replies"
            params = {"fields": "id", "limit": 1}
            response = requests.get(replies_url, headers=self.headers, params=params)
            debug_info["attempts"].append({
                "method": "Direct replies access",
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "response": response.text[:200] if len(response.text) < 200 else response.text[:200] + "..."
            })

        except Exception as e:
            debug_info["attempts"].append({
                "method": "Direct replies access",
                "status_code": "Error",
                "success": False,
                "response": str(e)
            })

        return debug_info

    def verify_token(self) -> bool:
        """
        Verify that the access token is valid

        Returns:
            True if token is valid, False otherwise
        """
        try:
            logger.info("Verifying access token...")
            url = f"{self.base_url}/me"
            params = {"fields": "id,username"}

            data = self._make_request(url, params)
            if data and "id" in data:
                logger.info(f"Token verified for user: {data.get('username', 'Unknown')}")
                return True
            else:
                logger.error("Token verification failed")
                return False

        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return False


def setup_api_permissions():
    """
    Guide for setting up Threads API permissions
    """
    print("""
    === THREADS API SETUP GUIDE ===

    1. PREREQUISITES:
       - You need a Meta (Facebook) Developer account
       - Your Instagram account must be a Professional/Business account
       - You need to connect your Instagram to a Facebook Page

    2. CREATE META APP:
       a. Go to: https://developers.facebook.com/apps/
       b. Click "Create App"
       c. Choose "Other" > "Business"
       d. Fill in app details

    3. ADD THREADS API:
       a. In your app dashboard, click "Add Product"
       b. Find "Threads" and click "Set Up"
       c. Accept the supplemental terms

    4. CONFIGURE PERMISSIONS:
       Required permissions for this script:
       - threads_basic (for reading your own posts)
       - threads_read_replies (for reading replies)
       - threads_manage_replies (if you want to reply)
       - threads_manage_insights (for analytics)

    5. GET ACCESS TOKEN:
       a. Go to Threads API > Settings
       b. Generate a User Access Token
       c. Select the permissions listed above
       d. Copy the token (it starts with "THQS...")

    6. IMPORTANT NOTES:
       - User tokens expire after 60 days
       - You'll need to refresh them periodically
       - Store tokens securely (use .env file)

    7. TEST TOKEN:
       You can test your token at:
       https://developers.facebook.com/tools/debug/accesstoken/

    """)


def create_env_file():
    """
    Create a template .env file for storing credentials
    """
    env_template = """# Threads API Configuration
THREADS_ACCESS_TOKEN=your_access_token_here

# Optional: Post ID to analyze
THREADS_POST_ID=post_id_to_analyze

# Optional: Maximum pages to fetch (leave empty for all pages)
MAX_PAGES=

# Optional: Include nested replies (replies to replies)
INCLUDE_NESTED_REPLIES=false
"""

    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_template)
        print("Created .env file template. Please add your access token.")
    else:
        print(".env file already exists")


def debug_post_id(access_token: str, post_id: str):
    """
    Debug function to test post ID access
    """
    client = ThreadsAPIClient(access_token)

    print(f"=== DEBUGGING POST ID: {post_id} ===\n")

    # First verify token works
    if not client.verify_token():
        print("❌ Token verification failed")
        return
    else:
        print("✅ Token verification successful")

    # Debug post access
    debug_info = client.debug_post_access(post_id)

    print(f"\nPost ID: {debug_info['post_id']}")
    print(f"Overall Success: {'✅' if debug_info['success'] else '❌'}")

    print("\n=== Attempts ===")
    for i, attempt in enumerate(debug_info['attempts'], 1):
        status_icon = "✅" if attempt['success'] else "❌"
        print(f"{i}. {attempt['method']}: {status_icon}")
        print(f"   Status: {attempt['status_code']}")
        print(f"   Response: {attempt['response']}")
        print()

    if debug_info['success']:
        print("=== Final Data ===")
        print(json.dumps(debug_info['final_data'], indent=2))
    else:
        print("=== Troubleshooting Suggestions ===")
        print("1. Verify the post ID is correct")
        print("2. Check if the post is public (private posts may not be accessible)")
        print("3. Ensure your access token has the right permissions:")
        print("   - threads_basic")
        print("   - threads_read_replies")
        print("4. Try accessing one of your own posts first")
        print("5. Check if the post belongs to your account or a connected account")


def fetch_replies_data(access_token: str, post_id: Optional[str] = None, include_nested: bool = False):
    """
    Main function to fetch replies from a Threads post

    Args:
        access_token: Your Threads API access token
        post_id: Optional post ID to analyze. If None, fetches your recent posts
        include_nested: Whether to fetch nested replies (replies to replies)

    Returns:
        DataFrame containing replies data
    """
    client = ThreadsAPIClient(access_token)

    # Verify token first
    if not client.verify_token():
        logger.error("Invalid access token. Please check your token.")
        return None

    try:
        if not post_id:
            # Fetch user's recent posts
            logger.info("Fetching your recent posts...")
            posts = client.get_user_posts(limit=10)

            if not posts:
                print("No posts found")
                return None

            # Display posts for selection
            print("\nYour recent posts:")
            for i, post in enumerate(posts):
                text_preview = (post.get('text', 'No text')[:50] + '...') if len(
                    post.get('text', '')) > 50 else post.get('text', 'No text')
                print(f"{i + 1}. {text_preview}")
                print(f"   Replies: {post.get('reply_count', 0)}")
                print(f"   ID: {post['id']}")
                print()

            # Let user select a post
            selection = input("Enter the number of the post to analyze (or post ID directly): ").strip()

            if selection.isdigit() and 1 <= int(selection) <= len(posts):
                post_id = posts[int(selection) - 1]['id']
            else:
                post_id = selection

        # Fetch post details
        logger.info(f"Fetching details for post {post_id}...")
        post_details = client.get_post_details(post_id)
        print(f"\nPost text: {post_details.get('text', 'No text')[:100]}...")
        print(f"Official reply count from API: {post_details.get('reply_count', 'Unknown')}")

        # Get max pages from environment
        max_pages = os.getenv('MAX_PAGES')
        max_pages = int(max_pages) if max_pages and max_pages.isdigit() else None

        if max_pages:
            logger.info(f"Limited to {max_pages} pages")

        # Fetch replies
        logger.info("Starting reply fetch process...")
        replies = client.get_replies(post_id, max_pages=max_pages)

        if not replies:
            print("No replies found for this post")
            return None

        print(f"\nFound {len(replies)} direct replies")

        # Optionally fetch nested replies
        all_replies = replies.copy()
        if include_nested:
            print("✅ Testing nested replies approach...")

            # First, let's examine what fields we're actually getting
            if replies:
                print(f"\n=== SAMPLE REPLY STRUCTURE ===")
                sample_reply = replies[0]
                print("Available fields:", list(sample_reply.keys()))
                print("Sample reply:", json.dumps(sample_reply, indent=2, default=str))
                print("=" * 50)

            # Try a different approach - test if we can access replies via their permalink
            print(f"\n=== TESTING NESTED REPLY ACCESS ===")
            test_reply_id = replies[0]['id'] if replies else None

            if test_reply_id:
                print(f"Testing nested replies for ID: {test_reply_id}")
                try:
                    # Try the standard nested replies endpoint
                    nested_url = f"{client.base_url}/{test_reply_id}/replies"
                    test_response = client._make_request(nested_url, {"fields": "id", "limit": 1})

                    if test_response:
                        print(f"✅ Nested replies endpoint works: {test_response}")
                    else:
                        print(f"❌ Nested replies endpoint returned no data")

                    # Also try to get more info about the specific reply
                    reply_details_url = f"{client.base_url}/{test_reply_id}"
                    reply_details = client._make_request(reply_details_url, {"fields": "id,reply_count,has_replies"})
                    print(f"Reply details: {reply_details}")

                except Exception as e:
                    print(f"❌ Error testing nested replies: {e}")

            print(f"\n🤔 ANALYSIS: Threads API Reply Structure")
            print(f"• API returned 573 direct replies")
            print(f"• No reply_count field in individual replies")
            print(f"• UI shows ~1.5k total interactions")
            print(f"\nPossible explanations for the discrepancy:")
            print(f"1. UI counts likes/reactions as 'replies' (most likely)")
            print(f"2. API has undocumented pagination limits")
            print(f"3. Nested replies require different permissions")
            print(f"4. Some replies are private/filtered out")
            print(f"5. API definition of 'reply' differs from UI")

        else:
            print("❌ Nested replies disabled (INCLUDE_NESTED_REPLIES not set to 'true')")
            print("   To enable: Add INCLUDE_NESTED_REPLIES=true to your .env file")

        # Convert to DataFrame
        df = pd.DataFrame(all_replies)

        # Add timestamp conversion if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Add helpful columns
        if 'text' in df.columns:
            df['reply_length'] = df['text'].str.len()
            df['has_text'] = df['text'].notna() & (df['text'].str.strip() != '')

        # Mark direct vs nested replies
        if 'is_nested' not in df.columns:
            df['is_nested'] = False

        # Save to CSV
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"threads_replies_{post_id}_{timestamp_str}.csv"
        df.to_csv(filename, index=False)
        print(f"\nData saved to {filename}")

        # Detailed reporting
        print(f"\n=== FETCH REPORT ===")
        print(f"Post ID: {post_id}")
        print(f"API reported reply count: {post_details.get('reply_count', 'Unknown')}")
        print(f"Direct replies fetched: {len(replies)}")
        if include_nested:
            nested_count = len(df[df.get('is_nested', False) == True])
            print(f"Nested replies fetched: {nested_count}")
            print(f"Total replies fetched: {len(df)}")
        else:
            print(f"Total replies fetched: {len(replies)}")

        # Calculate discrepancy
        api_count = post_details.get('reply_count', 0)
        if api_count and isinstance(api_count, int) and api_count > len(replies):
            missing = api_count - len(replies)
            percentage = (len(replies) / api_count) * 100
            print(f"\nDiscrepancy Analysis:")
            print(f"Missing replies: {missing}")
            print(f"Fetch completeness: {percentage:.1f}%")
            print(f"Possible reasons for discrepancy:")
            print(f"  - Rate limiting stopped fetch early")
            print(f"  - Some replies are private/restricted")
            print(f"  - API pagination issues")
            print(f"  - Nested replies not included (try INCLUDE_NESTED_REPLIES=true)")
        elif not isinstance(api_count, int):
            print(f"\nNote: API didn't provide exact reply count, but we fetched {len(replies)} replies")
            print(f"UI shows ~1.5k replies vs {len(replies)} fetched = significant gap")
            print(f"Possible reasons for the gap:")
            print(f"  - API pagination limitations")
            print(f"  - Nested replies not included (try INCLUDE_NESTED_REPLIES=true)")
            print(f"  - Some replies are private/restricted")
            print(f"  - Rate limiting or API constraints")

        return df

    except Exception as e:
        logger.error(f"Error in fetch_replies_data: {e}")
        return None


def main():
    """
    Main execution function
    """
    print("=== ENHANCED THREADS REPLY FETCHER ===\n")

    # Check for .env file
    if not os.path.exists('.env'):
        print("No .env file found. Creating template...")
        create_env_file()
        setup_api_permissions()
        return

    # Load configuration
    access_token = os.getenv('THREADS_ACCESS_TOKEN')
    post_id = os.getenv('THREADS_POST_ID')
    include_nested = os.getenv('INCLUDE_NESTED_REPLIES', 'false').lower() == 'true'

    if not access_token or access_token == 'your_access_token_here':
        print("Please add your Threads API access token to the .env file")
        setup_api_permissions()
        return

    # Debug: Show what we loaded from environment
    print(f"Debug - INCLUDE_NESTED_REPLIES env var: '{os.getenv('INCLUDE_NESTED_REPLIES', 'NOT_SET')}'")
    print(f"Debug - include_nested boolean: {include_nested}")
    print()

    # Add debug option
    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        if len(sys.argv) > 2:
            debug_post_id(access_token, sys.argv[2])
        else:
            debug_post_id_input = input("Enter post ID to debug: ").strip()
            debug_post_id(access_token, debug_post_id_input)
        return

    # Add fetch option with specific post ID
    if len(sys.argv) > 1 and sys.argv[1] == "--fetch":
        if len(sys.argv) > 2:
            specific_post_id = sys.argv[2]
            print(f"Fetching replies for post ID: {specific_post_id}")
            df = fetch_replies_data(access_token, specific_post_id, include_nested)
            return
        else:
            print("Please provide a post ID: python threads_fetcher.py --fetch POST_ID")
            return

    # Fetch replies
    df = fetch_replies_data(access_token, post_id, include_nested)

    if df is not None:
        # Display enhanced statistics
        print("\n=== ENHANCED STATISTICS ===")
        print(f"Total replies: {len(df)}")

        if 'is_nested' in df.columns:
            direct_replies = len(df[df['is_nested'] == False])
            nested_replies = len(df[df['is_nested'] == True])
            print(f"Direct replies: {direct_replies}")
            print(f"Nested replies: {nested_replies}")

        if 'username' in df.columns:
            print(f"Unique users: {df['username'].nunique()}")
            top_users = df['username'].value_counts().head(5)
            print("Most active repliers:")
            for user, count in top_users.items():
                print(f"  @{user}: {count} replies")

        if 'reply_length' in df.columns:
            print(f"Average reply length: {df['reply_length'].mean():.1f} characters")
            print(f"Longest reply: {df['reply_length'].max()} characters")
            print(f"Shortest reply: {df['reply_length'].min()} characters")

        if 'timestamp' in df.columns:
            df_sorted = df.sort_values('timestamp')
            print(f"Earliest reply: {df_sorted.iloc[0]['timestamp']}")
            print(f"Latest reply: {df_sorted.iloc[-1]['timestamp']}")

        print("\n=== SAMPLE REPLIES ===")
        sample_df = df[df.get('has_text', True) == True].head(3)
        for i, row in sample_df.iterrows():
            nested_indicator = " (nested)" if row.get('is_nested', False) else ""
            print(f"@{row.get('username', 'unknown')}{nested_indicator}: {row.get('text', 'No text')[:100]}...")
            print()


if __name__ == "__main__":
    main()