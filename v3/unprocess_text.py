#!/usr/bin/env python3

"""
Data Merger: Combine unprocessed text with manual sentiment tags
Merges threads_comments_full_analyzed.csv with threads_comments_training.csv
"""

import pandas as pd
from datetime import datetime
import os
import sys


def merge_training_data(analyzed_file, training_file, output_file=None):
    """
    Merge the unprocessed text data with manual sentiment tags

    Args:
        analyzed_file: Path to threads_comments_full_analyzed.csv (has clean text)
        training_file: Path to threads_comments_training.csv (has manual tags)
        output_file: Optional output filename
    """

    print("=== THREADS DATA MERGER ===")
    print(f"📁 Loading analyzed file: {analyzed_file}")
    print(f"📁 Loading training file: {training_file}")
    print("=" * 50)

    try:
        # Load both files
        analyzed_df = pd.read_csv(analyzed_file)
        training_df = pd.read_csv(training_file)

        print(f"✅ Analyzed file: {len(analyzed_df)} rows, {len(analyzed_df.columns)} columns")
        print(f"✅ Training file: {len(training_df)} rows, {len(training_df.columns)} columns")

        # Check for key columns
        if 'id' not in analyzed_df.columns or 'id' not in training_df.columns:
            print("❌ Both files must have 'id' column for merging")
            return None

        if 'text' not in analyzed_df.columns:
            print("❌ Analyzed file missing 'text' column")
            return None

        if 'manual_sentiment' not in training_df.columns:
            print("❌ Training file missing 'manual_sentiment' column")
            return None

        print("\n📊 Analyzing manual sentiment tags in training file...")
        manual_tagged = training_df[training_df['manual_sentiment'].notna() & (training_df['manual_sentiment'] != '')]
        print(f"✅ Found {len(manual_tagged)} manually tagged samples")

        if len(manual_tagged) > 0:
            sentiment_dist = manual_tagged['manual_sentiment'].value_counts()
            print("📈 Manual sentiment distribution:")
            for sentiment, count in sentiment_dist.items():
                print(f"  {sentiment}: {count}")

        # Select essential columns from analyzed file (unprocessed data)
        essential_columns = ['id', 'text', 'username', 'timestamp', 'permalink', 'reply_length']

        # Add any other useful columns that exist in analyzed file but not processed versions
        optional_columns = ['media_type', 'media_url', 'has_text', 'is_nested']

        analyzed_columns = []
        for col in essential_columns:
            if col in analyzed_df.columns:
                analyzed_columns.append(col)

        for col in optional_columns:
            if col in analyzed_df.columns:
                analyzed_columns.append(col)

        print(f"\n📋 Using columns from analyzed file: {analyzed_columns}")

        # Select manual tagging columns from training file
        training_columns = ['id', 'manual_sentiment']
        if 'is_tagged' in training_df.columns:
            training_columns.append('is_tagged')

        print(f"📋 Using columns from training file: {training_columns}")

        # Perform the merge
        print(f"\n🔄 Merging data on 'id' column...")

        # Start with analyzed data (clean text)
        merged_df = analyzed_df[analyzed_columns].copy()

        # Clean up text column - ensure it's string type and handle NaN values
        merged_df['text'] = merged_df['text'].fillna('').astype(str)

        # Merge with training data (manual tags)
        merged_df = merged_df.merge(
            training_df[training_columns],
            on='id',
            how='left'  # Keep all records from analyzed file
        )

        print(f"✅ Merged successfully: {len(merged_df)} total rows")

        # Analyze the merged result
        print(f"\n📊 Merged data analysis:")
        print(f"  Total rows: {len(merged_df)}")
        print(
            f"  Rows with manual tags: {len(merged_df[merged_df['manual_sentiment'].notna() & (merged_df['manual_sentiment'] != '')])}")
        print(
            f"  Rows without manual tags: {len(merged_df[merged_df['manual_sentiment'].isna() | (merged_df['manual_sentiment'] == '')])}")

        # Check for any manual tags that weren't merged
        training_ids = set(training_df['id'])
        analyzed_ids = set(analyzed_df['id'])

        missing_from_analyzed = training_ids - analyzed_ids
        missing_from_training = analyzed_ids - training_ids

        if missing_from_analyzed:
            print(f"⚠️ Warning: {len(missing_from_analyzed)} IDs from training file not found in analyzed file")

        if missing_from_training:
            print(f"ℹ️ Info: {len(missing_from_training)} IDs from analyzed file not in training file (expected)")

        # Show manual sentiment distribution in merged data
        merged_tagged = merged_df[merged_df['manual_sentiment'].notna() & (merged_df['manual_sentiment'] != '')]
        if len(merged_tagged) > 0:
            print(f"\n📈 Final manual sentiment distribution in merged data:")
            sentiment_dist = merged_tagged['manual_sentiment'].value_counts()
            for sentiment, count in sentiment_dist.items():
                pct = (count / len(merged_tagged)) * 100
                print(f"  {sentiment}: {count} ({pct:.1f}%)")

        # Create output filename if not provided
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"threads_merged_training_{timestamp}.csv"

        # Save merged data
        merged_df.to_csv(output_file, index=False)
        print(f"\n💾 Saved merged data: {output_file}")

        # Show sample of merged data
        print(f"\n📋 Sample of merged data:")
        print("=" * 80)

        # Show some tagged examples
        tagged_samples = merged_df[merged_df['manual_sentiment'].notna() & (merged_df['manual_sentiment'] != '')].head(
            3)
        for _, row in tagged_samples.iterrows():
            print(f"ID: {row['id']}")
            # Handle NaN/missing text safely
            text_value = row['text']
            if pd.isna(text_value) or text_value == '':
                display_text = "[Empty or missing text]"
            else:
                text_str = str(text_value)
                display_text = f"\"{text_str[:100]}{'...' if len(text_str) > 100 else ''}\""
            print(f"Text: {display_text}")
            print(f"Manual Sentiment: {row['manual_sentiment']}")
            print(f"Username: {row.get('username', 'N/A')}")
            print("-" * 40)

        # Show some untagged examples
        print(f"\nUntagged samples (for prediction):")
        untagged_samples = merged_df[merged_df['manual_sentiment'].isna() | (merged_df['manual_sentiment'] == '')].head(
            2)
        for _, row in untagged_samples.iterrows():
            print(f"ID: {row['id']}")
            # Handle NaN/missing text safely
            text_value = row['text']
            if pd.isna(text_value) or text_value == '':
                display_text = "[Empty or missing text]"
            else:
                text_str = str(text_value)
                display_text = f"\"{text_str[:100]}{'...' if len(text_str) > 100 else ''}\""
            print(f"Text: {display_text}")
            print(f"Manual Sentiment: [Not Tagged]")
            print("-" * 40)

        print(f"\n🎉 SUCCESS! Ready for ML training with {len(merged_tagged)} tagged samples")
        print(f"📄 Use this file with: python comprehensive_ml_analyzer.py {output_file}")

        return merged_df, output_file

    except Exception as e:
        print(f"❌ Error during merge: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def validate_merged_data(merged_df):
    """Validate the merged data quality"""
    print(f"\n🔍 Data Quality Validation:")
    print("=" * 40)

    # Check for duplicates
    duplicates = merged_df.duplicated(subset=['id']).sum()
    print(f"Duplicate IDs: {duplicates}")

    # Check for empty text
    empty_text = merged_df[merged_df['text'].isna() | (merged_df['text'] == '')].shape[0]
    print(f"Empty text entries: {empty_text}")

    # Check manual sentiment tag quality
    tagged_data = merged_df[merged_df['manual_sentiment'].notna() & (merged_df['manual_sentiment'] != '')]

    if len(tagged_data) > 0:
        unique_sentiments = tagged_data['manual_sentiment'].unique()
        print(f"Unique sentiment labels: {list(unique_sentiments)}")

        # Check for potential labeling issues
        potential_issues = tagged_data[~tagged_data['manual_sentiment'].isin(['positive', 'negative', 'neutral'])]
        if len(potential_issues) > 0:
            print(f"⚠️ Unusual sentiment labels found: {potential_issues['manual_sentiment'].unique()}")

    # Text length analysis
    text_lengths = merged_df['text'].fillna('').astype(str).str.len()
    print(f"Text length stats:")
    print(f"  Mean: {text_lengths.mean():.1f} characters")
    print(f"  Min: {text_lengths.min()}")
    print(f"  Max: {text_lengths.max()}")

    # Check for missing/empty text in tagged data
    tagged_data = merged_df[merged_df['manual_sentiment'].notna() & (merged_df['manual_sentiment'] != '')]
    if len(tagged_data) > 0:
        empty_tagged_text = tagged_data[tagged_data['text'].isna() | (tagged_data['text'] == '')].shape[0]
        print(f"Tagged samples with empty text: {empty_tagged_text}")
        if empty_tagged_text > 0:
            print("⚠️ Warning: Some manually tagged samples have empty text - these should be reviewed")

    print(f"✅ Data validation complete")


def main():
    # Default file paths (can be overridden by command line args)
    analyzed_file = "threads_comments_full_unanalyzed.csv"
    training_file = "threads_comments_training.csv"
    output_file = None

    # Parse command line arguments
    if len(sys.argv) > 1:
        analyzed_file = sys.argv[1]
    if len(sys.argv) > 2:
        training_file = sys.argv[2]
    if len(sys.argv) > 3:
        output_file = sys.argv[3]

    print(f"Using files:")
    print(f"  Analyzed: {analyzed_file}")
    print(f"  Training: {training_file}")
    if output_file:
        print(f"  Output: {output_file}")

    # Check if files exist
    if not os.path.exists(analyzed_file):
        print(f"❌ Analyzed file not found: {analyzed_file}")
        return

    if not os.path.exists(training_file):
        print(f"❌ Training file not found: {training_file}")
        return

    # Perform the merge
    merged_df, output_filename = merge_training_data(analyzed_file, training_file, output_file)

    if merged_df is not None:
        # Validate the merged data
        validate_merged_data(merged_df)

        print(f"\n🚀 Next Steps:")
        print(f"1. Review the merged data: {output_filename}")
        print(f"2. Run ML analysis: python comprehensive_ml_analyzer.py {output_filename}")
        print(f"3. Compare results in the dashboard!")
    else:
        print(f"❌ Merge failed. Please check the error messages above.")


if __name__ == "__main__":
    main()