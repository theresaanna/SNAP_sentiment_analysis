#!/usr/bin/env python3

"""
Fixed CSV Sentiment Analyzer with Real Multi-Model Support
This script trains multiple ML algorithms and saves individual CSV files for each.
"""

import pandas as pd
from datetime import datetime
import os
import sys
import re
import numpy as np

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


class SimpleMLAnalyzer:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'naive_bayes': MultinomialNB(),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.label_encoder = LabelEncoder()
        self.trained_models = {}

    def preprocess_text(self, text):
        """Simple text preprocessing"""
        if not text or pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def train_models(self, df):
        """Train all models on manually tagged data"""
        print("🤖 Training ML models...")

        # Get manually tagged data
        tagged_data = df[df['manual_sentiment'].notna() & (df['manual_sentiment'] != '')].copy()
        print(f"📊 Training on {len(tagged_data)} manually tagged samples")

        # Preprocess texts
        texts = [self.preprocess_text(text) for text in tagged_data['text']]
        labels = tagged_data['manual_sentiment'].tolist()

        # Create features
        X = self.vectorizer.fit_transform(texts)
        y = self.label_encoder.fit_transform(labels)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train all models
        results = {}
        for name, model in self.models.items():
            print(f"  Training {name}...")
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            results[name] = score
            self.trained_models[name] = model
            print(f"    Accuracy: {score:.3f}")

        # Find best model
        best_model_name = max(results, key=results.get)
        print(f"🏆 Best model: {best_model_name} (accuracy: {results[best_model_name]:.3f})")

        return results, best_model_name

    def predict_all_models(self, df):
        """Generate predictions from all trained models"""
        print("🔮 Generating predictions from all models...")

        # Preprocess all texts
        all_texts = [self.preprocess_text(text) for text in df['text']]
        X_all = self.vectorizer.transform(all_texts)

        # Generate predictions for each model
        for model_name, model in self.trained_models.items():
            print(f"  Predicting with {model_name}...")

            predictions = model.predict(X_all)
            probabilities = model.predict_proba(X_all)
            confidences = np.max(probabilities, axis=1)

            # Convert predictions back to labels
            predicted_labels = self.label_encoder.inverse_transform(predictions)

            # Add to dataframe
            df[f'{model_name}_sentiment'] = predicted_labels
            df[f'{model_name}_confidence'] = confidences
            df[f'{model_name}_type'] = model_name

            # Add individual probabilities
            for i, class_name in enumerate(self.label_encoder.classes_):
                df[f'{model_name}_prob_{class_name}'] = probabilities[:, i]

        return df


def main():
    if len(sys.argv) < 2:
        print("Usage: python fixed_csv_analyzer.py your_file.csv")
        return

    csv_file = sys.argv[1]

    print("=== REAL ML CSV SENTIMENT ANALYZER ===")
    print(f"Processing: {csv_file}")
    print("=" * 50)

    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"❌ Error: File '{csv_file}' not found")
        return

    try:
        # Load the CSV
        print(f"📁 Loading CSV file: {csv_file}")
        df = pd.read_csv(csv_file)
        print(f"✅ Successfully loaded {len(df)} rows with {len(df.columns)} columns")

        # Check for required columns
        if 'text' not in df.columns:
            print(f"❌ Missing 'text' column. Available: {list(df.columns)}")
            return

        if 'manual_sentiment' not in df.columns:
            print(f"❌ Missing 'manual_sentiment' column. Available: {list(df.columns)}")
            return

        # Check manual sentiment data
        manual_data = df[df['manual_sentiment'].notna() & (df['manual_sentiment'] != '')]
        print(f"✅ Found {len(manual_data)} manually tagged samples")

        if len(manual_data) < 10:
            print("❌ Need at least 10 manual sentiment tags for training!")
            return

        # Show distribution
        sentiment_dist = manual_data['manual_sentiment'].value_counts()
        print("📊 Manual sentiment distribution:")
        for sentiment, count in sentiment_dist.items():
            print(f"  {sentiment}: {count}")

        # Initialize ML analyzer
        analyzer = SimpleMLAnalyzer()

        # Train models
        results, best_model = analyzer.train_models(df)

        # Generate predictions for all rows
        df = analyzer.predict_all_models(df)

        # Add winning model predictions to main columns
        df['ml_sentiment'] = df[f'{best_model}_sentiment']
        df['ml_confidence'] = df[f'{best_model}_confidence']
        df['model_type'] = best_model
        df['ml_prob_positive'] = df[f'{best_model}_prob_positive']
        df['ml_prob_negative'] = df[f'{best_model}_prob_negative']
        df['ml_prob_neutral'] = df[f'{best_model}_prob_neutral']

        # Save files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create main analysis file (all models combined)
        main_output = f"ml_analysis_{timestamp}.csv"
        df.to_csv(main_output, index=False)
        print(f"\n✅ Saved main analysis: {main_output}")

        # Create individual model files
        saved_files = []
        for model_name in analyzer.trained_models.keys():
            # Create individual dataframe
            individual_df = df[[
                'id', 'text', 'username', 'timestamp', 'manual_sentiment',
                f'{model_name}_sentiment', f'{model_name}_confidence', f'{model_name}_type',
                f'{model_name}_prob_positive', f'{model_name}_prob_negative', f'{model_name}_prob_neutral'
            ]].copy()

            # Rename columns for consistency
            individual_df = individual_df.rename(columns={
                f'{model_name}_sentiment': 'ml_sentiment',
                f'{model_name}_confidence': 'ml_confidence',
                f'{model_name}_type': 'model_type',
                f'{model_name}_prob_positive': 'ml_prob_positive',
                f'{model_name}_prob_negative': 'ml_prob_negative',
                f'{model_name}_prob_neutral': 'ml_prob_neutral'
            })

            # Add metadata
            individual_df['algorithm_name'] = model_name
            individual_df['test_accuracy'] = results[model_name]

            # Save individual file
            algo_output = f"ml_individual_{timestamp}_{model_name}.csv"
            individual_df.to_csv(algo_output, index=False)
            saved_files.append(algo_output)
            print(f"📁 Saved {model_name}: {algo_output}")

        # Display results summary
        print(f"\n🎉 SUCCESS! Analysis Complete")
        print(f"📊 Main file: {main_output} ({len(df)} rows)")
        print(f"🤖 Individual files: {len(saved_files)} algorithm-specific CSVs")
        print(f"🏆 Best model: {best_model}")

        # Show prediction distribution
        pred_dist = df['ml_sentiment'].value_counts()
        print(f"\n📈 Final predictions ({best_model}):")
        for sentiment, count in pred_dist.items():
            pct = (count / len(df)) * 100
            print(f"  {sentiment}: {count} ({pct:.1f}%)")

        # Calculate training accuracy
        if len(manual_data) > 0:
            training_predictions = df[df['manual_sentiment'].notna()]
            if len(training_predictions) > 0:
                correct = (training_predictions['manual_sentiment'] == training_predictions['ml_sentiment']).sum()
                accuracy = correct / len(training_predictions)
                print(f"🎯 Training accuracy: {accuracy:.1%} ({correct}/{len(training_predictions)})")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()