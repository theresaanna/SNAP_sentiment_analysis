import requests
import json
import pandas as pd
from datetime import datetime
import time
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import numpy as np

# Load environment variables
load_dotenv()


class FixedThreadsAnalyzer:
    """Fixed Threads API client with properly working sentiment analysis"""

    def __init__(self, access_token: str):
        """
        Initialize the Threads API client with fixed sentiment analysis

        Args:
            access_token: Your Threads API access token
        """
        self.access_token = access_token
        self.base_url = "https://graph.threads.net/v1.0"
        self.headers = {"Authorization": f"Bearer {access_token}"}

        # Initialize sentiment analyzers
        self.vader_analyzer = SentimentIntensityAnalyzer()

        # Enhanced keyword lists for context-aware analysis
        self.positive_keywords = [
            'love', 'great', 'awesome', 'amazing', 'perfect', 'excellent', 'fantastic',
            'wonderful', 'brilliant', 'outstanding', 'superb', 'fabulous', 'incredible',
            'thank', 'thanks', 'grateful', 'appreciate', 'helpful', 'useful', 'valuable',
            'agree', 'exactly', 'correct', 'right', 'yes', 'absolutely', 'definitely',
            'smart', 'clever', 'wise', 'insightful', 'thoughtful', 'good', 'nice',
            'well done', 'congrats', 'congratulations', 'proud', 'impressed', 'respect'
        ]

        self.negative_keywords = [
            'hate', 'terrible', 'awful', 'horrible', 'disgusting', 'pathetic', 'stupid',
            'dumb', 'idiotic', 'moronic', 'ridiculous', 'absurd', 'nonsense', 'crazy',
            'wrong', 'false', 'lie', 'lies', 'fake', 'fraud', 'scam', 'bullshit', 'bs',
            'disagree', 'no', 'never', 'impossible', 'ridiculous', 'outrageous',
            'disappointed', 'angry', 'mad', 'furious', 'upset', 'annoyed', 'frustrated',
            'waste', 'useless', 'pointless', 'meaningless', 'worthless', 'fail', 'failed'
        ]

    def get_post_details(self, post_id: str) -> Dict:
        """Get details about a specific Threads post"""
        fields = "id,text,username,timestamp,media_type,media_url,permalink"
        url = f"{self.base_url}/{post_id}"
        params = {"fields": fields}

        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

    def get_replies(self, post_id: str, limit: int = 100) -> List[Dict]:
        """Fetch all replies to a specific Threads post"""
        all_replies = []
        next_page = None

        while True:
            url = f"{self.base_url}/{post_id}/replies"
            params = {
                "fields": "id,text,username,timestamp,permalink,reply_count",
                "limit": min(limit, 100)
            }

            if next_page:
                params["after"] = next_page

            try:
                response = requests.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                data = response.json()

                if "data" in data:
                    all_replies.extend(data["data"])

                if "paging" in data and "next" in data["paging"]:
                    next_url = data["paging"]["next"]
                    if "after=" in next_url:
                        next_page = next_url.split("after=")[1].split("&")[0]
                    else:
                        break
                else:
                    break

                time.sleep(0.5)  # Rate limiting

            except requests.exceptions.RequestException as e:
                print(f"Error fetching replies: {e}")
                break

        return all_replies

    def extract_features(self, text) -> Dict:
        """Extract enhanced features from text for sentiment analysis"""
        # Handle NaN/null values from pandas
        if pd.isna(text) or not text or not str(text).strip():
            return {
                'cleaned_text': '',
                'emojis': '',
                'emoji_count': 0,
                'has_question': False,
                'has_exclamation': False,
                'has_caps': False,
                'word_count': 0,
                'char_count': 0,
                'positive_keywords': 0,
                'negative_keywords': 0
            }

        # Convert to string to handle any remaining edge cases
        text = str(text).strip()
        if not text:
            return {
                'cleaned_text': '',
                'emojis': '',
                'emoji_count': 0,
                'has_question': False,
                'has_exclamation': False,
                'has_caps': False,
                'word_count': 0,
                'char_count': 0,
                'positive_keywords': 0,
                'negative_keywords': 0
            }

        # Clean text and extract emojis
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002500-\U00002BEF"  # chinese char
            "\U00002702-\U000027B0"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001f926-\U0001f937"
            "\U00010000-\U0010ffff"
            "\u2640-\u2642"
            "\u2600-\u2B55"
            "\u200d"
            "\u23cf"
            "\u23e9"
            "\u231a"
            "\ufe0f"
            "\u3030"
            "]+", flags=re.UNICODE)

        emojis = emoji_pattern.findall(text)
        cleaned_text = emoji_pattern.sub('', text).strip()

        # Count keyword matches (case insensitive)
        text_lower = text.lower()
        positive_count = sum(1 for keyword in self.positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in self.negative_keywords if keyword in text_lower)

        return {
            'cleaned_text': cleaned_text,
            'emojis': ''.join(emojis),
            'emoji_count': len(emojis),
            'has_question': '?' in text,
            'has_exclamation': '!' in text,
            'has_caps': any(word.isupper() and len(word) > 2 for word in text.split()),
            'word_count': len(cleaned_text.split()) if cleaned_text else 0,
            'char_count': len(text),
            'positive_keywords': positive_count,
            'negative_keywords': negative_count
        }

    def analyze_sentiment_fixed(self, text, features: Dict) -> Dict:
        """
        Fixed sentiment analysis with proper classification logic

        Key fixes:
        1. Removed overly complex optimization logic that wasn't working
        2. Used more reasonable thresholds
        3. Proper handling of edge cases
        4. Clear final sentiment assignment
        """
        # Handle NaN/null values from pandas
        if pd.isna(text) or not text or not str(text).strip():
            return {
                'vader_compound': 0.0,
                'vader_positive': 0.0,
                'vader_negative': 0.0,
                'vader_neutral': 1.0,
                'vader_sentiment': 'neutral',
                'textblob_polarity': 0.0,
                'textblob_subjectivity': 0.0,
                'textblob_sentiment': 'neutral',
                'combined_sentiment_score': 0.0,
                'final_sentiment': 'neutral',
                'confidence': 'low'
            }

        # Convert to string to handle any remaining edge cases
        text = str(text).strip()
        if not text:
            return {
                'vader_compound': 0.0,
                'vader_positive': 0.0,
                'vader_negative': 0.0,
                'vader_neutral': 1.0,
                'vader_sentiment': 'neutral',
                'textblob_polarity': 0.0,
                'textblob_subjectivity': 0.0,
                'textblob_sentiment': 'neutral',
                'combined_sentiment_score': 0.0,
                'final_sentiment': 'neutral',
                'confidence': 'low'
            }

        # VADER analysis
        vader_scores = self.vader_analyzer.polarity_scores(text)
        vader_compound = vader_scores['compound']

        # VADER classification with reasonable thresholds
        if vader_compound >= 0.1:
            vader_sentiment = 'positive'
        elif vader_compound <= -0.1:
            vader_sentiment = 'negative'
        else:
            vader_sentiment = 'neutral'

        # TextBlob analysis
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity

        # TextBlob classification
        if textblob_polarity > 0.1:
            textblob_sentiment = 'positive'
        elif textblob_polarity < -0.1:
            textblob_sentiment = 'negative'
        else:
            textblob_sentiment = 'neutral'

        # Combined score (weighted average)
        combined_score = (vader_compound * 0.6) + (textblob_polarity * 0.4)

        # FINAL CLASSIFICATION - This is the key fix!
        # Use more lenient thresholds and context clues

        # Base classification from combined score
        if combined_score > 0.05:
            final_sentiment = 'positive'
        elif combined_score < -0.05:
            final_sentiment = 'negative'
        else:
            final_sentiment = 'neutral'

        # Context-aware adjustments for borderline cases
        if final_sentiment == 'neutral' and abs(combined_score) > 0.02:
            # Check for strong positive indicators
            if (features['positive_keywords'] > features['negative_keywords'] and
                    features['positive_keywords'] > 0):
                final_sentiment = 'positive'
            # Check for strong negative indicators
            elif (features['negative_keywords'] > features['positive_keywords'] and
                  features['negative_keywords'] > 0):
                final_sentiment = 'negative'
            # Emoji context
            elif features['emoji_count'] > 0 and combined_score > 0:
                final_sentiment = 'positive'

        # Confidence scoring
        abs_score = abs(combined_score)
        if abs_score > 0.3:
            confidence = 'high'
        elif abs_score > 0.1:
            confidence = 'medium'
        else:
            confidence = 'low'

        return {
            'vader_compound': vader_compound,
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'vader_sentiment': vader_sentiment,
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'textblob_sentiment': textblob_sentiment,
            'combined_sentiment_score': combined_score,
            'final_sentiment': final_sentiment,  # This is the key column for visualization
            'confidence': confidence
        }

    def analyze_replies_fixed(self, replies: List[Dict]) -> pd.DataFrame:
        """Analyze replies with fixed sentiment classification"""
        analysis_results = []

        print(f"Analyzing {len(replies)} replies with fixed sentiment classification...")

        for i, reply in enumerate(replies):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(replies)} replies analyzed")

            # Extract features - handle text safely
            text = reply.get('text', '')
            if pd.isna(text):
                text = ''
            else:
                text = str(text)  # Ensure it's a string

            features = self.extract_features(text)

            # Perform fixed sentiment analysis
            sentiment_analysis = self.analyze_sentiment_fixed(text, features)

            # Combine all data
            result = {
                **reply,
                'reply_length': len(text),
                'has_text': bool(text.strip()),
                'is_nested': False,
                **features,
                **sentiment_analysis,
                'hour': datetime.fromisoformat(reply['timestamp'].replace('Z', '+00:00')).hour if reply.get(
                    'timestamp') else None
            }

            analysis_results.append(result)

        df = pd.DataFrame(analysis_results)

        # Convert timestamp if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        print(f"Analysis complete! Processed {len(df)} replies.")
        return df

    def generate_analysis_report(self, df: pd.DataFrame) -> Dict:
        """Generate a comprehensive analysis report"""

        # Sentiment distribution
        sentiment_counts = df['final_sentiment'].value_counts()

        # Confidence distribution
        confidence_dist = df['confidence'].value_counts()

        # Score statistics
        score_stats = df['combined_sentiment_score'].describe()

        report = {
            'total_replies': len(df),
            'sentiment_distribution': sentiment_counts.to_dict(),
            'confidence_distribution': confidence_dist.to_dict(),
            'score_statistics': score_stats.to_dict(),
            'positive_percentage': (sentiment_counts.get('positive', 0) / len(df)) * 100,
            'negative_percentage': (sentiment_counts.get('negative', 0) / len(df)) * 100,
            'neutral_percentage': (sentiment_counts.get('neutral', 0) / len(df)) * 100
        }

        return report


def main_fixed():
    """Main execution function with fixed analysis"""
    print("=== FIXED THREADS SENTIMENT ANALYZER ===\n")

    # Load access token
    access_token = os.getenv('THREADS_ACCESS_TOKEN')
    post_id = os.getenv('THREADS_POST_ID')

    if not access_token or access_token == 'your_access_token_here':
        print("Please add your Threads API access token to the .env file")
        return

    # Initialize the fixed analyzer
    analyzer = FixedThreadsAnalyzer(access_token)

    try:
        if not post_id:
            print("No specific post ID provided. Please set THREADS_POST_ID in your .env file")
            return

        # Fetch post details
        print(f"Fetching details for post {post_id}...")
        post_details = analyzer.get_post_details(post_id)
        print(f"Post text: {post_details.get('text', 'No text')[:100]}...")

        # Fetch replies
        print(f"Fetching replies...")
        replies = analyzer.get_replies(post_id)

        if not replies:
            print("No replies found for this post")
            return

        print(f"Found {len(replies)} replies")

        # Analyze with fixed sentiment classification
        df = analyzer.analyze_replies_fixed(replies)

        # Generate analysis report
        report = analyzer.generate_analysis_report(df)

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"threads_fixed_analysis_{post_id}_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"\nFixed analysis saved to {filename}")

        # Display analysis report
        print("\n=== ANALYSIS REPORT ===")
        print(f"Total replies analyzed: {report['total_replies']}")

        print(f"\nSentiment distribution:")
        for sentiment, count in report['sentiment_distribution'].items():
            pct = (count / report['total_replies']) * 100
            print(f"  {sentiment}: {count} ({pct:.1f}%)")

        print(f"\nConfidence distribution:")
        for conf, count in report['confidence_distribution'].items():
            pct = (count / report['total_replies']) * 100
            print(f"  {conf}: {count} ({pct:.1f}%)")

        print(f"\nScore statistics:")
        print(f"  Mean: {report['score_statistics']['mean']:.3f}")
        print(f"  Std: {report['score_statistics']['std']:.3f}")
        print(f"  Min: {report['score_statistics']['min']:.3f}")
        print(f"  Max: {report['score_statistics']['max']:.3f}")

        # Show sample classifications by sentiment
        for sentiment in ['positive', 'negative', 'neutral']:
            sample = df[df['final_sentiment'] == sentiment].head(3)
            if not sample.empty:
                print(f"\n=== SAMPLE {sentiment.upper()} REPLIES ===")
                for _, row in sample.iterrows():
                    print(f"Score: {row['combined_sentiment_score']:.3f} ({row['confidence']} confidence)")
                    print(f"Text: \"{row['text'][:80]}...\"")
                    print()

    except Exception as e:
        print(f"Error in fixed analysis: {e}")
        return


def analyze_existing_csv(csv_file_path: str):
    """
    Analyze an existing CSV file with the fixed sentiment analyzer
    This is useful for re-analyzing data you already have
    """
    print(f"=== ANALYZING EXISTING CSV: {csv_file_path} ===\n")

    try:
        # Load the CSV
        df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(df)} replies from {csv_file_path}")

        # Initialize analyzer (no need for API token for this)
        analyzer = FixedThreadsAnalyzer("dummy_token")

        # Re-analyze each reply
        analysis_results = []
        for i, row in df.iterrows():
            if i % 50 == 0:
                print(f"Progress: {i}/{len(df)} replies re-analyzed")

            # Handle text field safely
            text = row.get('text', '')
            if pd.isna(text):
                text = ''
            else:
                text = str(text)  # Ensure it's a string

            features = analyzer.extract_features(text)
            sentiment_analysis = analyzer.analyze_sentiment_fixed(text, features)

            # Combine original data with new analysis
            result = {
                **row.to_dict(),
                **features,
                **sentiment_analysis
            }
            analysis_results.append(result)

        # Create new DataFrame
        new_df = pd.DataFrame(analysis_results)

        # Generate report
        report = analyzer.generate_analysis_report(new_df)

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_filename = csv_file_path.replace('.csv', f'_fixed_{timestamp}.csv')
        new_df.to_csv(new_filename, index=False)
        print(f"\nFixed analysis saved to {new_filename}")

        # Display report
        print("\n=== FIXED ANALYSIS REPORT ===")
        print(f"Total replies: {report['total_replies']}")

        print(f"\nSentiment distribution:")
        for sentiment, count in report['sentiment_distribution'].items():
            pct = (count / report['total_replies']) * 100
            print(f"  {sentiment}: {count} ({pct:.1f}%)")

        return new_df, report

    except Exception as e:
        print(f"Error analyzing existing CSV: {e}")
        return None, None


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # If a CSV file path is provided as argument, analyze that
        csv_file = sys.argv[1]
        analyze_existing_csv(csv_file)
    else:
        # Otherwise run the main function
        main_fixed()