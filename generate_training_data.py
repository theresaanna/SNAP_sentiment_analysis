import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def create_skeleton_csv(filename="manually_tagged_replies.csv", num_examples=50):
    """
    Create a skeleton CSV file with sample data for testing the sentiment training pipeline
    """

    # Sample reply texts with obvious sentiments for testing
    sample_replies = {
        "positive": [
            "This is absolutely amazing! Love it so much! 🎉",
            "Best product ever! Highly recommend! 👍",
            "Fantastic quality and great service! Thank you!",
            "I'm so happy with this purchase! Perfect!",
            "Excellent work! Keep it up! 💪",
            "Amazing experience! Will definitely buy again!",
            "Love love love this! So good! ❤️",
            "Perfect! Exactly what I was looking for!",
            "Outstanding quality! Impressed! 🌟",
            "Brilliant! This exceeded my expectations!",
            "Wonderful product! Great value for money!",
            "Awesome! This made my day! 😊",
            "Incredible! Best decision I made!",
            "Superb quality! Highly satisfied!",
            "Amazing customer service! Thank you!",
            "Great job! This is fantastic! 🔥",
            "Perfect timing! Exactly what I needed!"
        ],

        "negative": [
            "This is terrible! Worst purchase ever! 😠",
            "Completely disappointed. Waste of money!",
            "Poor quality. Broke after one day. Awful!",
            "Terrible customer service! Never again!",
            "This sucks! Total garbage! 👎",
            "Worst experience of my life! Horrible!",
            "Disgusting quality! Want my money back!",
            "Pathetic! This is completely useless!",
            "Awful! Doesn't work at all!",
            "Terrible! Very disappointed! 😡",
            "Worst product ever! Complete waste!",
            "Horrible experience! Will not recommend!",
            "Useless! Broke immediately!",
            "Disappointing! Not worth it at all!",
            "Bad quality! Very upset!",
            "Terrible service! Frustrated!",
            "Worst decision ever! Regret buying!"
        ],

        "neutral": [
            "It's okay. Nothing special but works fine.",
            "Average product. Does what it says.",
            "Decent quality. Not great, not terrible.",
            "It's alright. Could be better.",
            "Okay product. As expected.",
            "Fine. Does the job.",
            "It works. Nothing to complain about.",
            "Standard quality. Normal experience.",
            "Average. What you'd expect.",
            "It's fine. No issues.",
            "Decent. Works as described.",
            "Okay quality. Fair price.",
            "Normal product. No surprises.",
            "It's adequate. Meets basic needs.",
            "Standard. Nothing remarkable.",
            "Fair quality. Reasonable price.",
            "Acceptable. Does what it should."
        ]
    }

    # Create sample data
    data = []

    # Generate sample entries for each sentiment
    sentiments = list(sample_replies.keys())
    replies_per_sentiment = num_examples // len(sentiments)

    base_timestamp = datetime.now() - timedelta(days=30)

    for i, sentiment in enumerate(sentiments):
        texts = sample_replies[sentiment]

        for j in range(replies_per_sentiment):
            # Cycle through texts if we need more examples
            text = texts[j % len(texts)]

            # Generate synthetic data that matches your original CSV structure
            reply_id = 18000000000000000 + (i * 1000) + j
            username = f"user_{random.randint(1000, 9999)}"
            timestamp = base_timestamp + timedelta(hours=random.randint(0, 720))

            # Basic text features
            words = text.split()
            word_count = len(words)
            char_count = len(text)
            emoji_count = len([c for c in text if ord(c) > 127])  # Rough emoji detection

            # Punctuation features
            has_question = 1 if '?' in text else 0
            has_exclamation = 1 if '!' in text else 0
            has_caps = 1 if any(c.isupper() for c in text) else 0

            # Synthetic sentiment scores (simulate VADER/TextBlob)
            if sentiment == "positive":
                vader_compound = random.uniform(0.1, 0.9)
                textblob_polarity = random.uniform(0.1, 0.8)
                vader_sentiment = "positive"
                textblob_sentiment = "positive"
                final_sentiment = "positive"
            elif sentiment == "negative":
                vader_compound = random.uniform(-0.9, -0.1)
                textblob_polarity = random.uniform(-0.8, -0.1)
                vader_sentiment = "negative"
                textblob_sentiment = "negative"
                final_sentiment = "negative"
            else:  # neutral
                vader_compound = random.uniform(-0.1, 0.1)
                textblob_polarity = random.uniform(-0.1, 0.1)
                vader_sentiment = "neutral"
                textblob_sentiment = "neutral"
                final_sentiment = "neutral"

            # Sometimes add conflicts for more realistic data
            if random.random() < 0.2:  # 20% chance of model disagreement
                conflicting_sentiments = ["positive", "negative", "neutral"]
                conflicting_sentiments.remove(sentiment)
                textblob_sentiment = random.choice(conflicting_sentiments)

            # Keyword counts (simple)
            positive_words = ["love", "great", "amazing", "excellent", "perfect", "best", "awesome", "fantastic"]
            negative_words = ["terrible", "awful", "worst", "bad", "horrible", "disappointing", "useless"]

            text_lower = text.lower()
            positive_keywords = sum(1 for word in positive_words if word in text_lower)
            negative_keywords = sum(1 for word in negative_words if word in text_lower)

            combined_sentiment_score = (vader_compound + textblob_polarity) / 2

            row = {
                'id': reply_id,
                'text': text,
                'username': username,
                'timestamp': timestamp.isoformat(),
                'permalink': f"https://threads.net/@{username}/post/{reply_id}",
                'media_type': '',
                'media_url': '',
                'reply_length': char_count,
                'has_text': True,
                'is_nested': False,
                'cleaned_text': text,
                'emojis': ''.join([c for c in text if ord(c) > 127]),
                'vader_compound': round(vader_compound, 4),
                'vader_positive': max(0, vader_compound) if vader_compound > 0 else 0,
                'vader_negative': abs(min(0, vader_compound)) if vader_compound < 0 else 0,
                'vader_neutral': 1 - abs(vader_compound) if abs(vader_compound) < 1 else 0,
                'vader_sentiment': vader_sentiment,
                'textblob_polarity': round(textblob_polarity, 4),
                'textblob_subjectivity': round(random.uniform(0.1, 0.9), 4),
                'textblob_sentiment': textblob_sentiment,
                'has_question': has_question,
                'has_exclamation': has_exclamation,
                'has_caps': has_caps,
                'word_count': word_count,
                'char_count': char_count,
                'emoji_count': emoji_count,
                'positive_keywords': positive_keywords,
                'negative_keywords': negative_keywords,
                'combined_sentiment_score': round(combined_sentiment_score, 4),
                'final_sentiment': final_sentiment,
                'hour': timestamp.hour,
                'manual_sentiment': sentiment,  # This is the key column for training!
                'is_manually_tagged': True,
                'has_conflict': vader_sentiment != textblob_sentiment
            }

            data.append(row)

    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

    print(f"Created skeleton CSV: {filename}")
    print(f"Total rows: {len(df)}")
    print(f"Manual tag distribution:")
    print(df['manual_sentiment'].value_counts())
    print(f"Conflicted rows: {df['has_conflict'].sum()}")
    print("\nColumn names:")
    print(list(df.columns))

    return df


def create_empty_template(filename="manual_tagging_template.csv"):
    """
    Create an empty template with just the column headers for manual population
    """

    # Define all required columns
    columns = [
        'id', 'text', 'username', 'timestamp', 'permalink', 'media_type', 'media_url',
        'reply_length', 'has_text', 'is_nested', 'cleaned_text', 'emojis',
        'vader_compound', 'vader_positive', 'vader_negative', 'vader_neutral', 'vader_sentiment',
        'textblob_polarity', 'textblob_subjectivity', 'textblob_sentiment',
        'has_question', 'has_exclamation', 'has_caps',
        'word_count', 'char_count', 'emoji_count',
        'positive_keywords', 'negative_keywords',
        'combined_sentiment_score', 'final_sentiment', 'hour',
        'manual_sentiment', 'is_manually_tagged', 'has_conflict'
    ]

    # Create empty DataFrame with columns
    df = pd.DataFrame(columns=columns)
    df.to_csv(filename, index=False)

    print(f"Created empty template: {filename}")
    print("Required columns:")
    for i, col in enumerate(columns, 1):
        marker = " <- MANUAL TAG COLUMN" if col == 'manual_sentiment' else ""
        print(f"  {i:2d}. {col}{marker}")

    return df


def main():
    """
    Create skeleton files for testing the sentiment training pipeline
    """
    print("=== CREATING SKELETON CSV FILES ===\n")

    # Create sample data for testing
    print("1. Creating sample data with manual tags...")
    sample_df = create_skeleton_csv("manually_tagged_replies.csv", num_examples=150)

    print("\n" + "=" * 50 + "\n")

    # Create empty template for real data
    print("2. Creating empty template for your data...")
    template_df = create_empty_template("manual_tagging_template.csv")

    print("\n" + "=" * 50)
    print("NEXT STEPS:")
    print("=" * 50)
    print("FOR TESTING:")
    print("  - Use 'manually_tagged_replies.csv' to test the training pipeline")
    print("  - Run: python sentiment_model_trainer.py")
    print()
    print("FOR REAL DATA:")
    print("  1. Export your real data from the manual tagging tool")
    print("  2. Ensure it has the 'manual_sentiment' column filled")
    print("  3. Make sure column names match the template")
    print("  4. Run the training script on your real tagged data")
    print()
    print("IMPORTANT: The 'manual_sentiment' column should contain:")
    print("  - 'positive' for positive sentiment")
    print("  - 'negative' for negative sentiment")
    print("  - 'neutral' for neutral sentiment")


if __name__ == "__main__":
    main()