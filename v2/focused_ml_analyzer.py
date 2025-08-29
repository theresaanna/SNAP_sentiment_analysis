#!/usr/bin/env python3

"""
Focused CSV Sentiment Analyzer - Gradient Boosting vs Logistic Regression
This script trains only gradient boosting and logistic regression algorithms
and saves individual CSV files for each.

Meant to add gradient boosting and logistic regression algorithms to an existing
main analysis file and add individual csv results for each algorithm.
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')


class FocusedMLAnalyzer:
    def __init__(self):
        self.models = {
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0,
                solver='liblinear'
            )
        }
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams for better context
            min_df=2,
            max_df=0.95
        )
        self.label_encoder = LabelEncoder()
        self.trained_models = {}
        self.training_results = {}

    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        if not text or pd.isna(text):
            return ""
        text = str(text).lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove mentions and hashtags but keep the text
        text = re.sub(r'[@#]\S+', '', text)
        # Keep only letters, spaces, and some punctuation
        text = re.sub(r'[^a-zA-Z\s!?.,]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def train_models(self, df):
        """Train both models on manually tagged data with detailed metrics"""
        print("🤖 Training ML models (Gradient Boosting vs Logistic Regression)...")
        print("=" * 60)

        # Get manually tagged data
        tagged_data = df[df['manual_sentiment'].notna() & (df['manual_sentiment'] != '')].copy()
        print(f"📊 Training on {len(tagged_data)} manually tagged samples")

        # Show class distribution
        class_dist = tagged_data['manual_sentiment'].value_counts()
        print(f"📈 Class distribution:")
        for sentiment, count in class_dist.items():
            pct = (count / len(tagged_data)) * 100
            print(f"   {sentiment}: {count} ({pct:.1f}%)")

        # Preprocess texts
        print("\n🔤 Preprocessing text data...")
        texts = [self.preprocess_text(text) for text in tagged_data['text']]
        labels = tagged_data['manual_sentiment'].tolist()

        # Create features
        print("🔢 Creating TF-IDF features...")
        X = self.vectorizer.fit_transform(texts)
        y = self.label_encoder.fit_transform(labels)

        print(f"   Feature matrix shape: {X.shape}")
        print(f"   Vocabulary size: {len(self.vectorizer.vocabulary_)}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Testing samples: {X_test.shape[0]}")

        # Train both models
        results = {}
        detailed_results = {}

        for name, model in self.models.items():
            print(f"\n🏋️ Training {name.replace('_', ' ').title()}...")

            # Train model
            model.fit(X_train, y_train)

            # Evaluate on test set
            test_score = model.score(X_test, y_test)
            train_score = model.score(X_train, y_train)

            # Get predictions for detailed metrics
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            # Calculate confidence metrics
            confidences = np.max(y_pred_proba, axis=1)
            avg_confidence = np.mean(confidences)
            high_conf_pct = (confidences > 0.8).mean() * 100

            # Store results
            results[name] = test_score
            detailed_results[name] = {
                'test_accuracy': test_score,
                'train_accuracy': train_score,
                'avg_confidence': avg_confidence,
                'high_confidence_pct': high_conf_pct,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'confidences': confidences
            }
            self.trained_models[name] = model

            # Print results
            print(f"   ✅ Training accuracy: {train_score:.3f}")
            print(f"   ✅ Test accuracy: {test_score:.3f}")
            print(f"   ✅ Average confidence: {avg_confidence:.3f}")
            print(f"   ✅ High confidence (>0.8): {high_conf_pct:.1f}%")

            # Check for overfitting
            if train_score - test_score > 0.1:
                print(f"   ⚠️  Potential overfitting detected")

        # Find best model
        best_model_name = max(results, key=results.get)
        best_score = results[best_model_name]

        print(f"\n🏆 WINNER: {best_model_name.replace('_', ' ').title()}")
        print(f"   Test accuracy: {best_score:.3f}")

        # Compare models
        gb_score = results['gradient_boosting']
        lr_score = results['logistic_regression']
        diff = abs(gb_score - lr_score)

        print(f"\n📊 MODEL COMPARISON:")
        print(f"   Gradient Boosting: {gb_score:.3f}")
        print(f"   Logistic Regression: {lr_score:.3f}")
        print(f"   Difference: {diff:.3f}")

        if diff < 0.02:
            print("   📌 Models perform very similarly!")
        elif diff < 0.05:
            print("   📌 Small performance difference")
        else:
            print("   📌 Significant performance difference")

        self.training_results = detailed_results
        return results, best_model_name, detailed_results

    def predict_all_models(self, df):
        """Generate predictions from both trained models"""
        print("\n🔮 Generating predictions from both models...")

        # Preprocess all texts
        all_texts = [self.preprocess_text(text) for text in df['text']]
        X_all = self.vectorizer.transform(all_texts)

        # Generate predictions for each model
        for model_name, model in self.trained_models.items():
            print(f"   📝 Predicting with {model_name.replace('_', ' ').title()}...")

            # Get predictions and probabilities
            predictions = model.predict(X_all)
            probabilities = model.predict_proba(X_all)
            confidences = np.max(probabilities, axis=1)

            # Convert predictions back to labels
            predicted_labels = self.label_encoder.inverse_transform(predictions)

            # Add to dataframe
            df[f'{model_name}_sentiment'] = predicted_labels
            df[f'{model_name}_confidence'] = confidences
            df[f'{model_name}_model_type'] = 'ensemble' if 'gradient' in model_name else 'linear'
            df[f'{model_name}_algorithm'] = model_name

            # Add individual probabilities for each class
            for i, class_name in enumerate(self.label_encoder.classes_):
                df[f'{model_name}_prob_{class_name}'] = probabilities[:, i]

            # Add training metrics
            if model_name in self.training_results:
                df[f'{model_name}_test_accuracy'] = self.training_results[model_name]['test_accuracy']
                df[f'{model_name}_avg_confidence'] = self.training_results[model_name]['avg_confidence']

        return df

    def generate_comparison_report(self, df):
        """Generate a detailed comparison report"""
        print("\n📋 GENERATING COMPARISON REPORT...")

        # Calculate agreement between models
        gb_preds = df['gradient_boosting_sentiment'].fillna('unknown')
        lr_preds = df['logistic_regression_sentiment'].fillna('unknown')

        # Agreement analysis
        agreement = (gb_preds == lr_preds).sum()
        total = len(df)
        agreement_pct = (agreement / total) * 100

        print(f"🤝 Model Agreement: {agreement_pct:.1f}% ({agreement}/{total})")

        # Disagreement analysis
        disagreements = df[gb_preds != lr_preds].copy()
        if len(disagreements) > 0:
            print(f"❌ Disagreements: {len(disagreements)} cases")

            # Show some examples
            print("\n📝 Sample disagreements:")
            for i, row in disagreements.head(3).iterrows():
                text_preview = str(row['text'])[:100] + "..." if len(str(row['text'])) > 100 else str(row['text'])
                print(f"   Text: \"{text_preview}\"")
                print(f"   GB: {row['gradient_boosting_sentiment']} (conf: {row['gradient_boosting_confidence']:.3f})")
                print(
                    f"   LR: {row['logistic_regression_sentiment']} (conf: {row['logistic_regression_confidence']:.3f})")
                print()

        # Confidence comparison
        gb_conf_avg = df['gradient_boosting_confidence'].mean()
        lr_conf_avg = df['logistic_regression_confidence'].mean()

        print(f"🎯 Average Confidence:")
        print(f"   Gradient Boosting: {gb_conf_avg:.3f}")
        print(f"   Logistic Regression: {lr_conf_avg:.3f}")

        # Sentiment distribution comparison
        print(f"\n📊 Sentiment Distribution:")
        gb_dist = df['gradient_boosting_sentiment'].value_counts(normalize=True) * 100
        lr_dist = df['logistic_regression_sentiment'].value_counts(normalize=True) * 100

        for sentiment in ['positive', 'negative', 'neutral']:
            gb_pct = gb_dist.get(sentiment, 0)
            lr_pct = lr_dist.get(sentiment, 0)
            print(f"   {sentiment.capitalize()}:")
            print(f"      GB: {gb_pct:.1f}% | LR: {lr_pct:.1f}%")


def main():
    if len(sys.argv) < 2:
        print("Usage: python focused_ml_analyzer.py your_file.csv")
        print("\nThis script compares Gradient Boosting vs Logistic Regression")
        print("for sentiment analysis on your Threads data.")
        return

    csv_file = sys.argv[1]

    print("=" * 60)
    print("🎯 FOCUSED ML SENTIMENT ANALYZER")
    print("   Gradient Boosting vs Logistic Regression")
    print("=" * 60)
    print(f"📁 Processing: {csv_file}")

    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"❌ Error: File '{csv_file}' not found")
        return

    try:
        # Load the CSV
        print(f"\n📂 Loading CSV file...")
        df = pd.read_csv(csv_file)
        print(f"✅ Successfully loaded {len(df)} rows with {len(df.columns)} columns")

        # Check for required columns
        required_columns = ['text', 'manual_sentiment']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            print(f"   Available columns: {list(df.columns)}")
            return

        # Check manual sentiment data
        manual_data = df[df['manual_sentiment'].notna() & (df['manual_sentiment'] != '')]
        print(f"📊 Found {len(manual_data)} manually tagged samples")

        if len(manual_data) < 20:
            print("⚠️  Warning: Less than 20 manual tags. Results may be unreliable.")
            if len(manual_data) < 10:
                print("❌ Need at least 10 manual sentiment tags for training!")
                return

        # Show distribution
        print(f"\n📈 Manual sentiment distribution:")
        sentiment_dist = manual_data['manual_sentiment'].value_counts()
        for sentiment, count in sentiment_dist.items():
            pct = (count / len(manual_data)) * 100
            print(f"   {sentiment}: {count} ({pct:.1f}%)")

        # Initialize ML analyzer
        analyzer = FocusedMLAnalyzer()

        # Train models
        results, best_model, detailed_results = analyzer.train_models(df)

        # Generate predictions for all rows
        df = analyzer.predict_all_models(df)

        # Generate comparison report
        analyzer.generate_comparison_report(df)

        # Add winning model predictions to main columns
        df['ml_sentiment'] = df[f'{best_model}_sentiment']
        df['ml_confidence'] = df[f'{best_model}_confidence']
        df['winning_model'] = best_model
        df['model_type'] = df[f'{best_model}_model_type']

        # Look for existing main analysis file to extend
        existing_main_file = None
        for filename in os.listdir('.'):
            if filename.startswith('ml_analysis_') and filename.endswith('.csv'):
                existing_main_file = filename
                break

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Either extend existing file or create new one
        if existing_main_file:
            print(f"\n📂 Found existing analysis file: {existing_main_file}")
            print("🔄 Adding new algorithms to existing analysis...")

            try:
                # Load existing data
                existing_df = pd.read_csv(existing_main_file)

                # Add new columns to existing dataframe
                for model_name in ['gradient_boosting', 'logistic_regression']:
                    existing_df[f'{model_name}_sentiment'] = df[f'{model_name}_sentiment']
                    existing_df[f'{model_name}_confidence'] = df[f'{model_name}_confidence']
                    existing_df[f'{model_name}_model_type'] = df[f'{model_name}_model_type']
                    existing_df[f'{model_name}_algorithm'] = df[f'{model_name}_algorithm']
                    existing_df[f'{model_name}_prob_positive'] = df[f'{model_name}_prob_positive']
                    existing_df[f'{model_name}_prob_negative'] = df[f'{model_name}_prob_negative']
                    existing_df[f'{model_name}_prob_neutral'] = df[f'{model_name}_prob_neutral']
                    existing_df[f'{model_name}_test_accuracy'] = df[f'{model_name}_test_accuracy']
                    existing_df[f'{model_name}_avg_confidence'] = df[f'{model_name}_avg_confidence']

                # Save updated file
                existing_df.to_csv(existing_main_file, index=False)
                main_output = existing_main_file
                print(f"✅ Updated existing file: {existing_main_file}")

            except Exception as e:
                print(f"⚠️  Could not update existing file ({e}), creating new one...")
                main_output = f"focused_ml_analysis_{timestamp}.csv"
                df.to_csv(main_output, index=False)
                print(f"✅ Created new analysis: {main_output}")
        else:
            # Create new main analysis file
            main_output = f"focused_ml_analysis_{timestamp}.csv"
            df.to_csv(main_output, index=False)
            print(f"\n✅ Created new analysis: {main_output}")

        # Create individual model files using the same naming structure as original
        saved_files = []
        for model_name in ['gradient_boosting', 'logistic_regression']:
            # Create individual dataframe with all necessary columns
            base_columns = ['id', 'text', 'username', 'timestamp', 'permalink',
                            'media_type', 'media_url', 'reply_length', 'has_text',
                            'is_nested', 'manual_sentiment', 'is_tagged']

            # Get base columns that exist in the dataframe
            available_base_columns = [col for col in base_columns if col in df.columns]

            individual_columns = available_base_columns + [
                f'{model_name}_sentiment', f'{model_name}_confidence',
                f'{model_name}_model_type', f'{model_name}_algorithm',
                f'{model_name}_prob_positive', f'{model_name}_prob_negative',
                f'{model_name}_prob_neutral', f'{model_name}_test_accuracy',
                f'{model_name}_avg_confidence'
            ]

            individual_df = df[individual_columns].copy()

            # Rename columns for dashboard compatibility (using original structure)
            individual_df = individual_df.rename(columns={
                f'{model_name}_sentiment': 'ml_sentiment',
                f'{model_name}_confidence': 'ml_confidence',
                f'{model_name}_model_type': 'model_type',
                f'{model_name}_algorithm': 'model_name',
                f'{model_name}_prob_positive': 'ml_prob_positive',
                f'{model_name}_prob_negative': 'ml_prob_negative',
                f'{model_name}_prob_neutral': 'ml_prob_neutral',
                f'{model_name}_test_accuracy': 'test_accuracy',
                f'{model_name}_avg_confidence': 'avg_training_confidence'
            })

            # Save individual file using the same naming structure as the original script
            algo_output = f"ml_individual_{timestamp}_{model_name}.csv"
            individual_df.to_csv(algo_output, index=False)
            saved_files.append(algo_output)
            print(f"📄 Saved {model_name.replace('_', ' ').title()}: {algo_output}")

        # Display final results summary
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print("=" * 40)
        print(f"📊 Main file: {main_output} ({len(df)} rows)")
        print(f"🤖 Individual files: {len(saved_files)} algorithm-specific CSVs")
        print(f"🏆 Best model: {best_model.replace('_', ' ').title()}")

        gb_acc = results['gradient_boosting']
        lr_acc = results['logistic_regression']
        print(f"📈 Final test accuracies:")
        print(f"   Gradient Boosting: {gb_acc:.1%}")
        print(f"   Logistic Regression: {lr_acc:.1%}")

        # Show prediction distribution for winning model
        pred_dist = df['ml_sentiment'].value_counts()
        print(f"\n📊 Final predictions ({best_model.replace('_', ' ').title()}):")
        for sentiment, count in pred_dist.items():
            pct = (count / len(df)) * 100
            print(f"   {sentiment}: {count} ({pct:.1f}%)")

        print(f"\n💡 Next steps:")
        print(f"   1. Load {main_output} in the algorithm comparison dashboard")
        print(f"   2. Compare the two models' performance and agreement")
        print(f"   3. Use the winning model ({best_model}) for production")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()