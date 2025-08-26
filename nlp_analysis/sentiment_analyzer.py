#!/usr/bin/env python3
"""
Simple script to fix sentiment analysis on existing CSV files
Run this on your existing analyzed CSV to get proper sentiment classifications
"""

import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from datetime import datetime


def fix_sentiment_analysis(csv_file_path: str):
    """Fix sentiment analysis on an existing CSV file"""

    print(f"Loading CSV file: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    print(f"Found {len(df)} replies to analyze")

    # Initialize sentiment analyzer
    vader_analyzer = SentimentIntensityAnalyzer()

    # Enhanced keyword lists
    positive_keywords = [
        'love', 'great', 'awesome', 'amazing', 'perfect', 'excellent', 'fantastic',
        'wonderful', 'brilliant', 'outstanding', 'superb', 'fabulous', 'incredible',
        'thank', 'thanks', 'grateful', 'appreciate', 'helpful', 'useful', 'valuable',
        'agree', 'exactly', 'correct', 'right', 'yes', 'absolutely', 'definitely',
        'smart', 'clever', 'wise', 'insightful', 'thoughtful', 'good', 'nice',
        'well done', 'congrats', 'congratulations', 'proud', 'impressed', 'respect'
    ]

    negative_keywords = [
        'hate', 'terrible', 'awful', 'horrible', 'disgusting', 'pathetic', 'stupid',
        'dumb', 'idiotic', 'moronic', 'ridiculous', 'absurd', 'nonsense', 'crazy',
        'wrong', 'false', 'lie', 'lies', 'fake', 'fraud', 'scam', 'bullshit', 'bs',
        'disagree', 'no', 'never', 'impossible', 'ridiculous', 'outrageous',
        'disappointed', 'angry', 'mad', 'furious', 'upset', 'annoyed', 'frustrated',
        'waste', 'useless', 'pointless', 'meaningless', 'worthless', 'fail', 'failed'
    ]

    def count_keywords(text, keywords):
        """Count keyword matches in text"""
        if not text:
            return 0
        text_lower = text.lower()
        return sum(1 for keyword in keywords if keyword in text_lower)

    def extract_emoji_count(text):
        """Count emojis in text"""
        if not text:
            return 0
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002500-\U00002BEF"  # chinese char
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
        return len(emoji_pattern.findall(text))

    def classify_sentiment_fixed(text):
        """Fixed sentiment classification logic"""
        if not text or not text.strip():
            return {
                'vader_compound_fixed': 0.0,
                'textblob_polarity_fixed': 0.0,
                'combined_score_fixed': 0.0,
                'final_sentiment_fixed': 'neutral',
                'confidence_fixed': 'low',
                'positive_keywords_fixed': 0,
                'negative_keywords_fixed': 0,
                'emoji_count_fixed': 0
            }

        # VADER analysis
        vader_scores = vader_analyzer.polarity_scores(text)
        vader_compound = vader_scores['compound']

        # TextBlob analysis
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity

        # Combined score
        combined_score = (vader_compound * 0.6) + (textblob_polarity * 0.4)

        # Count features
        positive_count = count_keywords(text, positive_keywords)
        negative_count = count_keywords(text, negative_keywords)
        emoji_count = extract_emoji_count(text)

        # Base classification
        if combined_score > 0.05:
            final_sentiment = 'positive'
        elif combined_score < -0.05:
            final_sentiment = 'negative'
        else:
            final_sentiment = 'neutral'

        # Context-aware adjustments for borderline cases
        if final_sentiment == 'neutral' and abs(combined_score) > 0.02:
            if positive_count > negative_count and positive_count > 0:
                final_sentiment = 'positive'
            elif negative_count > positive_count and negative_count > 0:
                final_sentiment = 'negative'
            elif emoji_count > 0 and combined_score > 0:
                final_sentiment = 'positive'

        # Confidence
        abs_score = abs(combined_score)
        if abs_score > 0.3:
            confidence = 'high'
        elif abs_score > 0.1:
            confidence = 'medium'
        else:
            confidence = 'low'

        return {
            'vader_compound_fixed': vader_compound,
            'textblob_polarity_fixed': textblob_polarity,
            'combined_score_fixed': combined_score,
            'final_sentiment_fixed': final_sentiment,
            'confidence_fixed': confidence,
            'positive_keywords_fixed': positive_count,
            'negative_keywords_fixed': negative_count,
            'emoji_count_fixed': emoji_count
        }

    # Apply fixed sentiment analysis
    print("Applying fixed sentiment analysis...")

    sentiment_results = []
    for i, row in df.iterrows():
        if i % 100 == 0:
            print(f"Progress: {i}/{len(df)}")

        text = row.get('text', '')
        result = classify_sentiment_fixed(text)
        sentiment_results.append(result)

    # Add results to dataframe
    sentiment_df = pd.DataFrame(sentiment_results)

    # Combine with original data
    df_fixed = pd.concat([df, sentiment_df], axis=1)

    # Save fixed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = csv_file_path.replace('.csv', f'_FIXED_{timestamp}.csv')
    df_fixed.to_csv(output_file, index=False)

    print(f"\nFixed analysis saved to: {output_file}")

    # Generate report
    sentiment_counts = df_fixed['final_sentiment_fixed'].value_counts()
    print(f"\n=== FIXED SENTIMENT ANALYSIS REPORT ===")
    print(f"Total replies: {len(df_fixed)}")

    print(f"\nSentiment distribution:")
    for sentiment, count in sentiment_counts.items():
        pct = (count / len(df_fixed)) * 100
        print(f"  {sentiment}: {count} ({pct:.1f}%)")

    # Compare with original if available
    if 'final_sentiment' in df.columns:
        original_counts = df['final_sentiment'].value_counts()
        print(f"\nComparison with original:")
        print(
            f"Original neutral: {original_counts.get('neutral', 0)} -> Fixed neutral: {sentiment_counts.get('neutral', 0)}")
        print(
            f"Original positive: {original_counts.get('positive', 0)} -> Fixed positive: {sentiment_counts.get('positive', 0)}")
        print(
            f"Original negative: {original_counts.get('negative', 0)} -> Fixed negative: {sentiment_counts.get('negative', 0)}")

    # Show sample improvements
    print(f"\n=== SAMPLE CLASSIFICATIONS ===")
    for sentiment in ['positive', 'negative', 'neutral']:
        sample = df_fixed[df_fixed['final_sentiment_fixed'] == sentiment].head(2)
        if not sample.empty:
            print(f"\nSample {sentiment.upper()} replies:")
            for _, row in sample.iterrows():
                score = row.get('combined_score_fixed', 0)
                confidence = row.get('confidence_fixed', 'unknown')
                text = str(row.get('text', ''))[:80]
                print(f"  Score: {score:.3f} ({confidence}) - \"{text}...\"")

    return df_fixed


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python csv_sentiment_fixer.py <path_to_csv_file>")
        print("Example: python csv_sentiment_fixer.py threads_replies_18087086674818268_20250825_191726_analyzed.csv")
        sys.exit(1)

    csv_file = sys.argv[1]
    try:
        fixed_df = fix_sentiment_analysis(csv_file)
        print(f"\n✅ Successfully fixed sentiment analysis!")
        print(f"Use the new CSV file ending in '_FIXED_' for your visualizations.")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)