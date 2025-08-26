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


class OptimizedThreadsAnalyzer:
    """Enhanced Threads API client with optimized sentiment analysis"""

    def __init__(self, access_token: str):
        """
        Initialize the Threads API client with optimized sentiment analysis

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

    def extract_features(self, text: str) -> Dict:
        """Extract enhanced features from text for sentiment analysis"""
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

    def analyze_sentiment_optimized(self, text: str, features: Dict) -> Dict:
        """
        Optimized sentiment analysis with tighter thresholds and context awareness

        Key improvements:
        1. Tighter neutral zone (±0.05 instead of ±0.1)
        2. Context-aware signals (emojis, punctuation, keywords)
        3. Confidence scoring
        4. More decisive classification
        """
        if not text or not text.strip():
            return {
                'vader_compound': 0.0,
                'vader_sentiment': 'neutral',
                'textblob_polarity': 0.0,
                'textblob_sentiment': 'neutral',
                'combined_sentiment_score': 0.0,
                'optimized_sentiment': 'neutral',
                'confidence': 'low',
                'optimization_applied': False
            }

        # VADER analysis
        vader_scores = self.vader_analyzer.polarity_scores(text)
        vader_compound = vader_scores['compound']

        if vader_compound >= 0.05:
            vader_sentiment = 'positive'
        elif vader_compound <= -0.05:
            vader_sentiment = 'negative'
        else:
            vader_sentiment = 'neutral'

        # TextBlob analysis
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity

        if textblob_polarity > 0.1:
            textblob_sentiment = 'positive'
        elif textblob_polarity < -0.1:
            textblob_sentiment = 'negative'
        else:
            textblob_sentiment = 'neutral'

        # Combined score (weighted average)
        combined_score = (vader_compound * 0.6) + (textblob_polarity * 0.4)

        # OPTIMIZED CLASSIFICATION with tighter thresholds
        # Phase 1: Tighter neutral zone
        if combined_score > 0.05:
            optimized_sentiment = 'positive'
        elif combined_score < -0.05:
            optimized_sentiment = 'negative'
        else:
            optimized_sentiment = 'neutral'

        # Phase 2: Context-aware enhancement
        optimization_applied = False

        # If we're in the tight neutral zone but have contextual signals, be more decisive
        if optimized_sentiment == 'neutral' and abs(combined_score) > 0.02:

            # Positive context signals
            positive_signals = (
                    (features['positive_keywords'] > 0) or
                    (features['emoji_count'] > 0 and combined_score > 0) or
                    (features['has_exclamation'] and combined_score > 0)
            )

            # Negative context signals
            negative_signals = (
                    (features['negative_keywords'] > 0) or
                    (features['has_caps'] and combined_score < 0) or
                    (combined_score < -0.03)
            )

            if positive_signals and not negative_signals:
                optimized_sentiment = 'positive'
                optimization_applied = True
            elif negative_signals and not positive_signals:
                optimized_sentiment = 'negative'
                optimization_applied = True

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
            'optimized_sentiment': optimized_sentiment,
            'confidence': confidence,
            'optimization_applied': optimization_applied
        }

    def analyze_replies_optimized(self, replies: List[Dict]) -> pd.DataFrame:
        """Analyze replies with optimized sentiment classification"""
        analysis_results = []

        print(f"Analyzing {len(replies)} replies with optimized sentiment classification...")

        for i, reply in enumerate(replies):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(replies)} replies analyzed")

            # Extract features
            text = reply.get('text', '')
            features = self.extract_features(text)

            # Perform optimized sentiment analysis
            sentiment_analysis = self.analyze_sentiment_optimized(text, features)

            # Combine all data
            result = {
                **reply,
                'reply_length': len(text),
                'has_text': bool(text.strip()),
                'is_nested': False,  # Could be enhanced to detect nested replies
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

    def generate_optimization_report(self, df: pd.DataFrame) -> Dict:
        """Generate a report on the optimization improvements"""

        # Compare original vs optimized classifications
        original_counts = df['vader_sentiment'].value_counts()
        optimized_counts = df['optimized_sentiment'].value_counts()

        # Calculate improvements
        neutral_reduction = original_counts.get('neutral', 0) - optimized_counts.get('neutral', 0)
        neutral_reduction_pct = (neutral_reduction / original_counts.get('neutral', 1)) * 100

        reclassified = df[df['vader_sentiment'] != df['optimized_sentiment']]
        optimization_applied_count = df['optimization_applied'].sum()

        confidence_dist = df['confidence'].value_counts()

        report = {
            'total_replies': len(df),
            'original_distribution': original_counts.to_dict(),
            'optimized_distribution': optimized_counts.to_dict(),
            'neutral_reduction': neutral_reduction,
            'neutral_reduction_percentage': neutral_reduction_pct,
            'reclassified_count': len(reclassified),
            'reclassified_percentage': (len(reclassified) / len(df)) * 100,
            'optimization_applied_count': optimization_applied_count,
            'confidence_distribution': confidence_dist.to_dict(),
            'average_confidence_score': df['combined_sentiment_score'].abs().mean()
        }

        return report


def main_optimized():
    """Main execution function with optimized analysis"""
    print("=== OPTIMIZED THREADS SENTIMENT ANALYZER ===\n")

    # Load access token
    access_token = os.getenv('THREADS_ACCESS_TOKEN')
    post_id = os.getenv('THREADS_POST_ID')

    if not access_token or access_token == 'your_access_token_here':
        print("Please add your Threads API access token to the .env file")
        return

    # Initialize the optimized analyzer
    analyzer = OptimizedThreadsAnalyzer(access_token)

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

        # Analyze with optimized sentiment classification
        df = analyzer.analyze_replies_optimized(replies)

        # Generate optimization report
        report = analyzer.generate_optimization_report(df)

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"threads_optimized_analysis_{post_id}_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"\nOptimized analysis saved to {filename}")

        # Display optimization report
        print("\n=== OPTIMIZATION REPORT ===")
        print(f"Total replies analyzed: {report['total_replies']}")
        print(f"Reclassified replies: {report['reclassified_count']} ({report['reclassified_percentage']:.1f}%)")
        print(
            f"Neutral reduction: {report['neutral_reduction']} replies ({report['neutral_reduction_percentage']:.1f}%)")
        print(f"Optimization signals applied: {report['optimization_applied_count']}")

        print(f"\nOriginal distribution:")
        for sentiment, count in report['original_distribution'].items():
            pct = (count / report['total_replies']) * 100
            print(f"  {sentiment}: {count} ({pct:.1f}%)")

        print(f"\nOptimized distribution:")
        for sentiment, count in report['optimized_distribution'].items():
            pct = (count / report['total_replies']) * 100
            print(f"  {sentiment}: {count} ({pct:.1f}%)")

        print(f"\nConfidence distribution:")
        for conf, count in report['confidence_distribution'].items():
            pct = (count / report['total_replies']) * 100
            print(f"  {conf}: {count} ({pct:.1f}%)")

        print(f"\nAverage sentiment score magnitude: {report['average_confidence_score']:.3f}")

        # Show sample reclassifications
        reclassified_sample = df[df['vader_sentiment'] != df['optimized_sentiment']].head(5)
        if not reclassified_sample.empty:
            print(f"\n=== SAMPLE RECLASSIFICATIONS ===")
            for _, row in reclassified_sample.iterrows():
                print(f"\nScore: {row['combined_sentiment_score']:.3f}")
                print(
                    f"Change: {row['vader_sentiment']} → {row['optimized_sentiment']} ({row['confidence']} confidence)")
                if row['optimization_applied']:
                    print("✓ Context optimization applied")
                print(f"Text: \"{row['text'][:80]}...\"")

    except Exception as e:
        print(f"Error in optimized analysis: {e}")
        return


if __name__ == "__main__":
    main_optimized()