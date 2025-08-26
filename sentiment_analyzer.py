import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter
import emoji
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


class ThreadsSentimentAnalyzer:
    """Sentiment analysis for Threads replies"""

    def __init__(self):
        """Initialize sentiment analyzers"""
        self.vader = SentimentIntensityAnalyzer()

    def clean_text(self, text: str) -> str:
        """
        Clean text for analysis

        Args:
            text: Raw text from Threads

        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""

        # Convert to string if not already
        text = str(text)

        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)

        # Remove mentions but keep the context
        text = re.sub(r'@\w+', '', text)

        # Remove hashtags but keep the word
        text = re.sub(r'#(\w+)', r'\1', text)

        # Keep emojis for sentiment analysis but extract them separately
        return text.strip()

    def extract_emojis(self, text: str) -> List[str]:
        """Extract emojis from text"""
        if pd.isna(text):
            return []
        return [c for c in str(text) if c in emoji.EMOJI_DATA]

    def analyze_sentiment_vader(self, text: str) -> Dict:
        """
        Analyze sentiment using VADER

        Args:
            text: Cleaned text

        Returns:
            Dictionary with sentiment scores
        """
        scores = self.vader.polarity_scores(text)

        # Determine overall sentiment
        if scores['compound'] >= 0.05:
            sentiment = 'positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        return {
            'vader_compound': scores['compound'],
            'vader_positive': scores['pos'],
            'vader_negative': scores['neg'],
            'vader_neutral': scores['neu'],
            'vader_sentiment': sentiment
        }

    def analyze_sentiment_textblob(self, text: str) -> Dict:
        """
        Analyze sentiment using TextBlob

        Args:
            text: Cleaned text

        Returns:
            Dictionary with polarity and subjectivity
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity

            # Determine sentiment category
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'

            return {
                'textblob_polarity': polarity,
                'textblob_subjectivity': blob.sentiment.subjectivity,
                'textblob_sentiment': sentiment
            }
        except:
            return {
                'textblob_polarity': 0,
                'textblob_subjectivity': 0,
                'textblob_sentiment': 'neutral'
            }

    def analyze_engagement_indicators(self, text: str) -> Dict:
        """
        Analyze engagement indicators in text

        Args:
            text: Reply text

        Returns:
            Dictionary with engagement metrics
        """
        if pd.isna(text):
            text = ""
        else:
            text = str(text).lower()

        indicators = {
            'has_question': '?' in text,
            'has_exclamation': '!' in text,
            'has_caps': bool(re.search(r'[A-Z]{3,}', str(text))),
            'word_count': len(text.split()),
            'char_count': len(text),
            'emoji_count': len(self.extract_emojis(text))
        }

        # Engagement keywords
        positive_engagement = ['love', 'amazing', 'awesome', 'great', 'excellent', 'best', 'perfect', 'wonderful']
        negative_engagement = ['hate', 'terrible', 'worst', 'awful', 'horrible', 'disgusting', 'trash']

        indicators['positive_keywords'] = sum(1 for word in positive_engagement if word in text)
        indicators['negative_keywords'] = sum(1 for word in negative_engagement if word in text)

        return indicators

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sentiment for entire DataFrame of replies

        Args:
            df: DataFrame with 'text' column

        Returns:
            DataFrame with added sentiment columns
        """
        if 'text' not in df.columns:
            raise ValueError("DataFrame must have a 'text' column")

        # Clean text
        df['cleaned_text'] = df['text'].apply(self.clean_text)

        # Extract emojis
        df['emojis'] = df['text'].apply(self.extract_emojis)

        # VADER sentiment
        vader_results = df['cleaned_text'].apply(self.analyze_sentiment_vader)
        for key in ['vader_compound', 'vader_positive', 'vader_negative', 'vader_neutral', 'vader_sentiment']:
            df[key] = vader_results.apply(lambda x: x[key])

        # TextBlob sentiment
        textblob_results = df['cleaned_text'].apply(self.analyze_sentiment_textblob)
        for key in ['textblob_polarity', 'textblob_subjectivity', 'textblob_sentiment']:
            df[key] = textblob_results.apply(lambda x: x[key])

        # Engagement indicators
        engagement_results = df['text'].apply(self.analyze_engagement_indicators)
        for key in ['has_question', 'has_exclamation', 'has_caps', 'word_count',
                    'char_count', 'emoji_count', 'positive_keywords', 'negative_keywords']:
            df[key] = engagement_results.apply(lambda x: x[key])

        # Combined sentiment (average of both methods)
        df['combined_sentiment_score'] = (df['vader_compound'] + df['textblob_polarity']) / 2

        # Final sentiment classification
        df['final_sentiment'] = df.apply(
            lambda row: 'positive' if row['combined_sentiment_score'] > 0.1
            else 'negative' if row['combined_sentiment_score'] < -0.1
            else 'neutral',
            axis=1
        )

        return df


def generate_report(df: pd.DataFrame) -> Dict:
    """
    Generate comprehensive sentiment analysis report

    Args:
        df: DataFrame with sentiment analysis results

    Returns:
        Dictionary with report metrics
    """
    report = {}

    # Overall sentiment distribution
    sentiment_dist = df['final_sentiment'].value_counts(normalize=True) * 100
    report['sentiment_distribution'] = sentiment_dist.to_dict()

    # Average scores
    report['average_scores'] = {
        'vader_compound': df['vader_compound'].mean(),
        'textblob_polarity': df['textblob_polarity'].mean(),
        'textblob_subjectivity': df['textblob_subjectivity'].mean(),
        'combined_sentiment': df['combined_sentiment_score'].mean()
    }

    # Engagement metrics
    report['engagement'] = {
        'avg_word_count': df['word_count'].mean(),
        'avg_char_count': df['char_count'].mean(),
        'pct_with_questions': (df['has_question'].sum() / len(df)) * 100,
        'pct_with_exclamations': (df['has_exclamation'].sum() / len(df)) * 100,
        'pct_with_emojis': (df['emoji_count'] > 0).sum() / len(df) * 100,
        'avg_emoji_count': df['emoji_count'].mean()
    }

    # Most common emojis
    all_emojis = [e for emojis in df['emojis'] for e in emojis]
    if all_emojis:
        emoji_counts = Counter(all_emojis)
        report['top_emojis'] = dict(emoji_counts.most_common(10))

    # Temporal analysis if timestamp available
    if 'timestamp' in df.columns:
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        report['sentiment_by_hour'] = df.groupby('hour')['combined_sentiment_score'].mean().to_dict()

    # Most positive and negative replies
    report['most_positive'] = df.nlargest(3, 'combined_sentiment_score')[['text', 'combined_sentiment_score']].to_dict(
        'records')
    report['most_negative'] = df.nsmallest(3, 'combined_sentiment_score')[['text', 'combined_sentiment_score']].to_dict(
        'records')

    return report


def visualize_sentiment(df: pd.DataFrame, save_path: str = None):
    """
    Create visualizations for sentiment analysis

    Args:
        df: DataFrame with sentiment analysis results
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Threads Reply Sentiment Analysis', fontsize=16)

    # 1. Sentiment distribution pie chart
    sentiment_counts = df['final_sentiment'].value_counts()
    colors = {'positive': '#2ecc71', 'neutral': '#95a5a6', 'negative': '#e74c3c'}
    axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index,
                   autopct='%1.1f%%', colors=[colors[s] for s in sentiment_counts.index])
    axes[0, 0].set_title('Sentiment Distribution')

    # 2. Sentiment scores distribution
    axes[0, 1].hist(df['combined_sentiment_score'], bins=30, color='skyblue', edgecolor='black')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Combined Sentiment Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Sentiment Score Distribution')

    # 3. Comparison of sentiment methods
    comparison_data = pd.DataFrame({
        'VADER': df['vader_compound'],
        'TextBlob': df['textblob_polarity']
    })
    axes[0, 2].boxplot([comparison_data['VADER'], comparison_data['TextBlob']],
                       labels=['VADER', 'TextBlob'])
    axes[0, 2].set_ylabel('Sentiment Score')
    axes[0, 2].set_title('Sentiment Method Comparison')
    axes[0, 2].axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # 4. Word count vs sentiment
    axes[1, 0].scatter(df['word_count'], df['combined_sentiment_score'], alpha=0.5)
    axes[1, 0].set_xlabel('Word Count')
    axes[1, 0].set_ylabel('Sentiment Score')
    axes[1, 0].set_title('Word Count vs Sentiment')
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # 5. Emoji usage by sentiment
    emoji_sentiment = df.groupby('final_sentiment')['emoji_count'].mean()
    axes[1, 1].bar(emoji_sentiment.index, emoji_sentiment.values,
                   color=[colors[s] for s in emoji_sentiment.index])
    axes[1, 1].set_xlabel('Sentiment')
    axes[1, 1].set_ylabel('Average Emoji Count')
    axes[1, 1].set_title('Emoji Usage by Sentiment')

    # 6. Subjectivity distribution
    axes[1, 2].hist(df['textblob_subjectivity'], bins=20, color='lightcoral', edgecolor='black')
    axes[1, 2].set_xlabel('Subjectivity Score')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Reply Subjectivity Distribution')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    plt.show()


def create_wordcloud(df: pd.DataFrame, sentiment: str = None, save_path: str = None):
    """
    Create word cloud from replies

    Args:
        df: DataFrame with text data
        sentiment: Optional - filter by sentiment ('positive', 'negative', 'neutral')
        save_path: Optional path to save the word cloud
    """
    # Filter by sentiment if specified
    if sentiment:
        text_data = df[df['final_sentiment'] == sentiment]['cleaned_text']
        title = f'{sentiment.capitalize()} Sentiment Word Cloud'
    else:
        text_data = df['cleaned_text']
        title = 'All Replies Word Cloud'

    # Combine all text
    all_text = ' '.join(text_data.dropna())

    if not all_text:
        print(f"No text data available for {sentiment or 'all'} sentiment")
        return

    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400,
                          background_color='white',
                          colormap='viridis',
                          max_words=100).generate(all_text)

    # Display
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Word cloud saved to {save_path}")

    plt.show()


def main():
    """
    Main function to run sentiment analysis on Threads replies
    """
    import sys
    import os

    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python sentiment_analysis.py <csv_file>")
        print("Where <csv_file> is the output from the threads_fetcher.py script")
        return

    csv_file = sys.argv[1]

    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return

    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    print(f"Loaded {len(df)} replies")

    # Initialize analyzer
    analyzer = ThreadsSentimentAnalyzer()

    # Perform sentiment analysis
    print("\nPerforming sentiment analysis...")
    df_analyzed = analyzer.analyze_dataframe(df)

    # Generate report
    print("\nGenerating report...")
    report = generate_report(df_analyzed)

    # Print report
    print("\n" + "=" * 50)
    print("SENTIMENT ANALYSIS REPORT")
    print("=" * 50)

    print("\n📊 SENTIMENT DISTRIBUTION:")
    for sentiment, percentage in report['sentiment_distribution'].items():
        print(f"  {sentiment.capitalize()}: {percentage:.1f}%")

    print("\n📈 AVERAGE SCORES:")
    print(f"  VADER Compound: {report['average_scores']['vader_compound']:.3f}")
    print(f"  TextBlob Polarity: {report['average_scores']['textblob_polarity']:.3f}")
    print(f"  TextBlob Subjectivity: {report['average_scores']['textblob_subjectivity']:.3f}")
    print(f"  Combined Sentiment: {report['average_scores']['combined_sentiment']:.3f}")

    print("\n💬 ENGAGEMENT METRICS:")
    print(f"  Average word count: {report['engagement']['avg_word_count']:.1f}")
    print(f"  Average character count: {report['engagement']['avg_char_count']:.1f}")
    print(f"  Replies with questions: {report['engagement']['pct_with_questions']:.1f}%")
    print(f"  Replies with exclamations: {report['engagement']['pct_with_exclamations']:.1f}%")
    print(f"  Replies with emojis: {report['engagement']['pct_with_emojis']:.1f}%")

    if 'top_emojis' in report and report['top_emojis']:
        print("\n😊 TOP EMOJIS:")
        for emoji_char, count in list(report['top_emojis'].items())[:5]:
            print(f"  {emoji_char}: {count}")

    print("\n✨ MOST POSITIVE REPLIES:")
    for i, reply in enumerate(report['most_positive'], 1):
        text_preview = reply['text'][:100] + '...' if len(reply['text']) > 100 else reply['text']
        print(f"  {i}. {text_preview}")
        print(f"     Score: {reply['combined_sentiment_score']:.3f}")

    print("\n😔 MOST NEGATIVE REPLIES:")
    for i, reply in enumerate(report['most_negative'], 1):
        text_preview = reply['text'][:100] + '...' if len(reply['text']) > 100 else reply['text']
        print(f"  {i}. {text_preview}")
        print(f"     Score: {reply['combined_sentiment_score']:.3f}")

    # Save analyzed data
    output_file = csv_file.replace('.csv', '_analyzed.csv')
    df_analyzed.to_csv(output_file, index=False)
    print(f"\n✅ Analyzed data saved to {output_file}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    viz_file = csv_file.replace('.csv', '_sentiment_viz.png')
    visualize_sentiment(df_analyzed, save_path=viz_file)

    # Generate word clouds
    print("\nGenerating word clouds...")
    for sentiment in ['positive', 'negative', 'neutral']:
        wc_file = csv_file.replace('.csv', f'_wordcloud_{sentiment}.png')
        create_wordcloud(df_analyzed, sentiment=sentiment, save_path=wc_file)


if __name__ == "__main__":
    main()