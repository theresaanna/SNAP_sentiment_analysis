return True
else:
print("❌ Token validation failed!")
if 'error' in data:
    print(f"   Error: {data['error'].get('message', 'Unknown error')}")
return False

except Exception as e:
print(f"❌ Error validating token: {e}")
return False


def get_user_id(self):
    """Get the user's Threads user ID"""
    print("\n🔍 Getting User ID...")

    url = f"{self.base_url}/me"
    params = {"fields": "id,username"}

    try:
        response = requests.get(url, headers=self.headers, params=params)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ User ID: {data.get('id')}")
            print(f"   Username: {data.get('username')}")
            return data.get('id')
        else:
            print(f"❌ Failed to get user ID: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def convert_post_url_to_id(self, url_or_shortcode: str) -> Optional[str]:
    """
    Convert a Threads post URL or shortcode to the actual post ID

    Args:
        url_or_shortcode: Either a full Threads URL or just the shortcode

    Returns:
        The numeric post ID if found, None otherwise
    """
    print(f"\n🔄 Converting post identifier: {url_or_shortcode}")

    # Extract shortcode from URL if full URL provided
    shortcode = None

    # Pattern for Threads URLs
    patterns = [
        r'threads\.net/[@\w]+/post/([A-Za-z0-9_-]+)',  # New format
        r'threads\.net/t/([A-Za-z0-9_-]+)',  # Old format
        r'^([A-Za-z0-9_-]+)import requests
    import json
    import os
    from dotenv import load_dotenv
    import re
    from urllib.parse import urlparse, parse_qs
    from typing import Optional

    # Load environment variables
    load_dotenv()


class ThreadsAPIDiagnostic:
    """Diagnostic tool for Threads API issues"""

    def __init__(self, access_token: str):
        self.access_token = access_token
        self.base_url = "https://graph.threads.net/v1.0"
        self.graph_url = "https://graph.facebook.com/v18.0"
        self.headers = {"Authorization": f"Bearer {access_token}"}

    def test_token(self):
        """Test if the access token is valid"""
        print("\n🔍 Testing Access Token...")

        # Test token with debug endpoint
        debug_url = f"{self.graph_url}/debug_token"
        params = {
            "input_token": self.access_token,
            "access_token": self.access_token
        }

        try:
            response = requests.get(debug_url, params=params)
            data = response.json()

            if 'data' in data:
                token_data = data['data']
                print("✅ Token is valid!")
                print(f"   App ID: {token_data.get('app_id', 'N/A')}")
                print(f"   Type: {token_data.get('type', 'N/A')}")
                print(f"   Expires: {token_data.get('expires_at', 'Never')}")

                # Check scopes
                scopes = token_data.get('scopes', [])
                print(f"\n📋 Permissions granted:")
                for scope in scopes:
                    print(f"   - {scope}")

                # Check for required permissions
                required = ['threads_basic', 'threads_read_replies']
                missing = [p for p in required if p not in scopes]

                if missing:
                    print(f"\n⚠️  Missing required permissions: {', '.join(missing)}")
                    print("   Please add these permissions in the Meta Developer Dashboard")
                    return False

                return True
                # Just the code
        ]

        for pattern in patterns:
            match = re.search(pattern, url_or_shortcode)
        if match:
            shortcode = match.group(1)
        break

        if not shortcode:
            print("❌ Could not extract shortcode from input")
            return None

        print(f"   Shortcode extracted: {shortcode}")

        # Now we need to get the numeric ID
        # First, get your recent posts to find the matching one
        user_id = self.get_user_id()
        if not user_id:
            return None

        print(f"\n📋 Fetching your recent posts to find matching ID...")

        url = f"{self.base_url}/{user_id}/threads"
        params = {
            "fields": "id,text,permalink",
            "limit": 100  # Get more posts to increase chance of finding it
        }

        try:
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                data = response.json()
                posts = data.get('data', [])

                # Look for matching post
                for post in posts:
                    permalink = post.get('permalink', '')
                    if shortcode in permalink:
                        post_id = post.get('id')
                        print(f"✅ Found matching post!")
                        print(f"   Numeric ID: {post_id}")
                        print(f"   Text preview: {post.get('text', '')[:50]}...")
                        return post_id

                print(f"❌ No matching post found in your recent {len(posts)} posts")
                print("   The post might be too old or not yours")

                # Try alternative method - construct media ID
                print("\n🔄 Trying alternative ID format...")
                # Threads media IDs often follow a pattern, but this is less reliable
                return None

            else:
                print(f"❌ Failed to fetch posts: {response.status_code}")
                print(f"   Response: {response.text}")
                return None

        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def get_post_id_from_reply(self, reply_url: str) -> Optional[str]:
        """
        If you have a reply URL, get the parent post ID
        """
        print(f"\n🔄 Getting parent post from reply URL: {reply_url}")

        # Extract reply shortcode
        shortcode = None
        patterns = [
            r'threads\.net/[@\w]+/post/([A-Za-z0-9_-]+)',
            r'threads\.net/t/([A-Za-z0-9_-]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, reply_url)
            if match:
                shortcode = match.group(1)
                break

        if not shortcode:
            print("❌ Could not extract shortcode from reply URL")
            return None

        # Get the reply's numeric ID first
        reply_id = self.convert_post_url_to_id(reply_url)
        if not reply_id:
            return None

        # Now get the parent post
        url = f"{self.base_url}/{reply_id}"
        params = {"fields": "replied_to"}

        try:
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                data = response.json()
                parent = data.get('replied_to')
                if parent:
                    parent_id = parent.get('id')
                    print(f"✅ Found parent post ID: {parent_id}")
                    return parent_id
                else:
                    print("ℹ️  This appears to be a top-level post, not a reply")
                    return reply_id
            else:
                print(f"❌ Failed to get parent post: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def test_post_access(self, post_id: str):
        """Test if we can access a specific post"""
        print(f"\n🔍 Testing access to post ID: {post_id}")

        url = f"{self.base_url}/{post_id}"
        params = {"fields": "id,text,username,timestamp,reply_count"}

        try:
            response = requests.get(url, headers=self.headers, params=params)

            if response.status_code == 200:
                data = response.json()
                print("✅ Successfully accessed post!")
                print(f"   Username: @{data.get('username', 'N/A')}")
                print(f"   Text: {data.get('text', '')[:100]}...")
                print(f"   Reply count: {data.get('reply_count', 0)}")
                return True
            else:
                print(f"❌ Cannot access post: {response.status_code}")
                error_data = response.json()
                if 'error' in error_data:
                    error = error_data['error']
                    print(f"   Error: {error.get('message', 'Unknown error')}")
                    print(f"   Type: {error.get('type', 'N/A')}")
                    print(f"   Code: {error.get('code', 'N/A')}")

                    # Provide specific guidance based on error
                    if error.get('code') == 100:
                        print("\n💡 This usually means:")
                        print("   - The post ID is incorrect")
                        print("   - The post is private/deleted")
                        print("   - You don't have permission to access it")
                return False

        except Exception as e:
            print(f"❌ Error: {e}")
            return False


def main():
    """Main diagnostic function"""
    print("=" * 60)
    print("THREADS API DIAGNOSTIC TOOL")
    print("=" * 60)

    # Load token
    access_token = os.getenv('THREADS_ACCESS_TOKEN')

    if not access_token or access_token == 'your_access_token_here':
        print("\n❌ No valid access token found in .env file")
        print("Please add your Threads API access token to the .env file")
        return

    diag = ThreadsAPIDiagnostic(access_token)

    # Test token
    if not diag.test_token():
        print("\n⚠️  Fix token issues before continuing")
        return

    printimport
    requests


import json
import os
from dotenv import load_dotenv
import re
from urllib.parse import urlparse, parse_qs
from typing import Optional

# Load environment variables
load_dotenv()


class ThreadsAPIDiagnostic:
    """Diagnostic tool for Threads API issues"""

    def __init__(self, access_token: str):
        self.access_token = access_token
        self.base_url = "https://graph.threads.net/v1.0"
        self.graph_url = "https://graph.facebook.com/v18.0"
        self.headers = {"Authorization": f"Bearer {access_token}"}

    def test_token(self):
        """Test if the access token is valid"""
        print("\n🔍 Testing Access Token...")

        # Test token with debug endpoint
        debug_url = f"{self.graph_url}/debug_token"
        params = {
            "input_token": self.access_token,
            "access_token": self.access_token
        }

        try:
            response = requests.get(debug_url, params=params)
            data = response.json()

            if 'data' in data:
                token_data = data['data']
                print("✅ Token is valid!")
                print(f"   App ID: {token_data.get('app_id', 'N/A')}")
                print(f"   Type: {token_data.get('type', 'N/A')}")
                print(f"   Expires: {token_data.get('expires_at', 'Never')}")

                # Check scopes
                scopes = token_data.get('scopes', [])
                print(f"\n📋 Permissions granted:")
                for scope in scopes:
                    print(f"   - {scope}")

                # Check for required permissions
                required = ['threads_basic', 'threads_read_replies']
                missing = [p for p in required if p not in scopes]

                if missing:
                    print(f"\n⚠️  Missing required permissions: {', '.join(missing)}")
                    print("   Please add these permissions in the Meta Developer Dashboard")
                    return False

                return True
