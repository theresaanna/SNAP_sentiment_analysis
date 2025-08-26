import requests
import json
import pandas as pd
from datetime import datetime
import time
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
import urllib.parse

# Load environment variables
load_dotenv()


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
            "Authorization": f"Bearer {access_token}",
            "User-Agent": "ThreadsAPIClient/1.0"
        }

    def _make_request(self, url: str, params: Dict = None, max_retries: int = 3) -> Dict:
        """
        Make a request with error handling and retries

        Args:
            url: Request URL
            params: Request parameters
            max_retries: Maximum number of retries

        Returns:
            Response JSON data
        """
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=30)

                # Handle different error codes
                if response.status_code == 429:
                    print(f"Rate limited. Waiting 60 seconds...")
                    time.sleep(60)
                    continue
                elif response.status_code == 500:
                    print(f"Server error (500). Retrying in {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)
                    continue
                elif response.status_code == 400:
                    print(f"Bad request (400): {response.text}")
                    response.raise_for_status()
                elif response.status_code == 403:
                    print(f"Forbidden (403): Check your permissions and token validity")
                    response.raise_for_status()

                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

        raise Exception(f"Failed after {max_retries} attempts")

    def get_user_id(self) -> str:
        """
        Get the authenticated user's Threads user ID

        Returns:
            User ID string
        """
        url = f"{self.base_url}/me"
        params = {"fields": "id,username"}

        try:
            data = self._make_request(url, params)
            print(f"Authenticated as: @{data.get('username', 'unknown')} (ID: {data.get('id', 'unknown')})")
            return data.get('id')
        except Exception as e:
            print(f"Error getting user ID: {e}")
            raise

    def get_post_details(self, post_id: str) -> Dict:
        """
        Get details about a specific Threads post

        Args:
            post_id: The ID of the Threads post

        Returns:
            Dictionary containing post details
        """
        # Use only basic fields that are most likely to be available
        fields = "id,text,username,timestamp,permalink"
        url = f"{self.base_url}/{post_id}"
        params = {"fields": fields}

        return self._make_request(url, params)

    def get_replies(self, post_id: str, limit: int = 100) -> List[Dict]:
        """
        Fetch all replies to a specific Threads post

        Args:
            post_id: The ID of the Threads post
            limit: Maximum number of replies per request (max 100)

        Returns:
            List of reply dictionaries
        """
        all_replies = []
        next_cursor = None
        request_count = 0
        max_requests = 50  # Safety limit

        print(f"Fetching replies for post {post_id}...")

        while request_count < max_requests:
            # Construct the URL for fetching replies
            url = f"{self.base_url}/{post_id}/replies"

            # Set up parameters with basic fields
            params = {
                "fields": "id,text,username,timestamp,permalink",
                "limit": min(limit, 25)  # Start with smaller batches
            }

            if next_cursor:
                params["after"] = next_cursor

            try:
                data = self._make_request(url, params)
                request_count += 1

                # Add replies to our list
                if "data" in data:
                    replies_batch = data["data"]
                    all_replies.extend(replies_batch)
                    print(f"Fetched {len(replies_batch)} replies (total: {len(all_replies)})")
                else:
                    print("No data in response")
                    break

                # Check for pagination
                if "paging" in data and "cursors" in data["paging"] and "after" in data["paging"]["cursors"]:
                    next_cursor = data["paging"]["cursors"]["after"]
                elif "paging" in data and "next" in data["paging"]:
                    # Parse cursor from next URL if needed
                    next_url = data["paging"]["next"]
                    if "after=" in next_url:
                        next_cursor = next_url.split("after=")[1].split("&")[0]
                        next_cursor = urllib.parse.unquote(next_cursor)
                    else:
                        break
                else:
                    print("No more pages")
                    break

                # Rate limiting - be more conservative
                time.sleep(2)

            except Exception as e:
                print(f"Error fetching replies (batch {request_count}): {e}")
                break

        print(f"Completed fetching replies. Total: {len(all_replies)}")
        return all_replies

    def get_user_posts(self, user_id: str = None, limit: int = 25) -> List[Dict]:
        """
        Get posts from a specific user (or yourself with None/me)

        Args:
            user_id: User ID or None for authenticated user
            limit: Number of posts to fetch

        Returns:
            List of post dictionaries
        """
        if user_id is None:
            user_id = self.get_user_id()

        url = f"{self.base_url}/{user_id}/threads"
        params = {
            "fields": "id,text,username,timestamp,permalink",
            "limit": min(limit, 25)  # Be conservative with limits
        }

        data = self._make_request(url, params)
        return data.get("data", [])


def setup_api_permissions():
    """
    Guide for setting up Threads API permissions
    """
    print("""
    === THREADS API SETUP GUIDE ===

    1. PREREQUISITES:
       - Meta (Facebook) Developer account
       - Instagram Professional/Business account
       - Instagram connected to a Facebook Page

    2. CREATE META APP:
       a. Go to: https://developers.facebook.com/apps/
       b. Click "Create App" > "Business" type
       c. Fill in app details

    3. ADD THREADS API:
       a. In app dashboard: Products > "Add a Product"
       b. Find "Threads API" and click "Set Up"
       c. Accept supplemental terms

    4. CONFIGURE PERMISSIONS:
       Required scopes:
       - threads_basic (read your posts)
       - threads_read_replies (read replies)
       - threads_manage_replies (optional - for posting replies)

    5. GET ACCESS TOKEN:
       a. Threads API > Threads API Settings
       b. Generate User Access Token
       c. Select required permissions
       d. Copy token (starts with "THQS...")

    6. IMPORTANT NOTES:
       - Tokens expire after 60 days
       - Test at: https://developers.facebook.com/tools/debug/accesstoken/
       - API has rate limits - be patient!

    7. COMMON ISSUES:
       - 500 errors: Often due to invalid post IDs or permission issues
       - 403 errors: Check token validity and permissions
       - 400 errors: Check request format and field names
    """)


def create_env_file():
    """
    Create a template .env file for storing credentials
    """
    env_template = """# Threads API Configuration
THREADS_ACCESS_TOKEN=your_access_token_here

# Optional: Post ID to analyze (leave empty to select from your posts)
THREADS_POST_ID=
"""

    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_template)
        print("Created .env file template. Please add your access token.")
    else:
        print(".env file already exists")


def validate_token(access_token: str) -> bool:
    """
    Validate the access token by making a simple API call

    Args:
        access_token: Access token to validate

    Returns:
        True if token is valid, False otherwise
    """
    try:
        client = ThreadsAPIClient(access_token)
        client.get_user_id()
        return True
    except Exception as e:
        print(f"Token validation failed: {e}")
        return False


def fetch_replies_data(access_token: str, post_id: Optional[str] = None):
    """
    Main function to fetch replies from a Threads post

    Args:
        access_token: Your Threads API access token
        post_id: Optional post ID to analyze. If None, fetches your recent posts

    Returns:
        DataFrame containing replies data
    """
    if not validate_token(access_token):
        print("Invalid or expired access token. Please check your token.")
        return None

    client = ThreadsAPIClient(access_token)

    try:
        if not post_id:
            # Fetch user's recent posts
            print("Fetching your recent posts...")
            posts = client.get_user_posts(limit=10)

            if not posts:
                print("No posts found")
                return None

            # Display posts for selection
            print("\nYour recent posts:")
            for i, post in enumerate(posts):
                text_preview = (post.get('text', 'No text')[:50] + '...') if post.get('text') and len(
                    post.get('text', '')) > 50 else post.get('text', 'No text')
                timestamp = post.get('timestamp', 'No date')
                print(f"{i + 1}. {text_preview}")
                print(f"   Posted: {timestamp}")
                print(f"   ID: {post['id']}")
                print()

            # Let user select a post
            while True:
                selection = input(
                    "Enter the number of the post to analyze (1-{}) or post ID directly: ".format(len(posts))).strip()

                if selection.isdigit() and 1 <= int(selection) <= len(posts):
                    post_id = posts[int(selection) - 1]['id']
                    break
                elif len(selection) > 10:  # Likely a post ID
                    post_id = selection
                    break
                else:
                    print("Invalid selection. Please try again.")

        # Fetch post details
        print(f"\nAnalyzing post {post_id}...")
        try:
            post_details = client.get_post_details(post_id)
            post_text = post_details.get('text', 'No text')
            print(f"Post preview: {post_text[:100]}{'...' if len(post_text) > 100 else ''}")
        except Exception as e:
            print(f"Warning: Could not fetch post details: {e}")
            print("Proceeding with reply fetching...")

        # Fetch replies
        print(f"\nFetching replies...")
        replies = client.get_replies(post_id)

        if not replies:
            print("No replies found for this post")
            return None

        print(f"Successfully fetched {len(replies)} replies")

        # Convert to DataFrame
        df = pd.DataFrame(replies)

        # Add timestamp conversion if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Add some basic analysis columns
        if 'text' in df.columns:
            df['reply_length'] = df['text'].fillna('').str.len()
            df['has_text'] = df['text'].notna() & (df['text'] != '')

        # Save to CSV
        filename = f"threads_replies_{post_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        print(f"\nData saved to {filename}")

        return df

    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check that your access token is valid and not expired")
        print("2. Verify you have the correct permissions (threads_basic, threads_read_replies)")
        print("3. Make sure the post ID exists and you have access to it")
        print("4. Try with a smaller batch size if you're hitting rate limits")
        return None


def main():
    """
    Main execution function
    """
    print("=== THREADS REPLY FETCHER (FIXED VERSION) ===\n")

    # Check for .env file
    if not os.path.exists('.env'):
        print("No .env file found. Creating template...")
        create_env_file()
        setup_api_permissions()
        return

    # Load access token
    access_token = os.getenv('THREADS_ACCESS_TOKEN')
    post_id = os.getenv('THREADS_POST_ID')

    # Clean up post_id if it's empty
    if post_id and post_id.strip() == '':
        post_id = None

    if not access_token or access_token == 'your_access_token_here':
        print("Please add your Threads API access token to the .env file")
        setup_api_permissions()
        return

    # Fetch replies
    df = fetch_replies_data(access_token, post_id)

    if df is not None:
        # Display basic statistics
        print("\n=== BASIC STATISTICS ===")
        print(f"Total replies: {len(df)}")

        if 'username' in df.columns:
            print(f"Unique users: {df['username'].nunique()}")

        if 'reply_length' in df.columns:
            print(f"Average reply length: {df['reply_length'].mean():.1f} characters")
            print(f"Longest reply: {df['reply_length'].max()} characters")
            print(f"Shortest reply: {df['reply_length'].min()} characters")

        print("\n=== SAMPLE REPLIES ===")
        sample_size = min(3, len(df))
        for i, (idx, row) in enumerate(df.head(sample_size).iterrows()):
            username = row.get('username', 'unknown')
            text = row.get('text', 'No text')
            preview = text[:100] + '...' if len(text) > 100 else text
            print(f"@{username}: {preview}")
            if i < sample_size - 1:
                print()

        print(f"\nData exported successfully! You can now analyze the CSV file.")


if __name__ == "__main__":
    main()