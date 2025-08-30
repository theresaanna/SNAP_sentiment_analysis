import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
from datetime import datetime
import os

warnings.filterwarnings('ignore')


class ThreadsMLAnalyzer:
    """Machine Learning analyzer for Threads sentiment data"""

    def __init__(self, data_path: str):
        """Initialize with data path"""
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.models = {}
        self.results = {}
        self.scalers = {}

    def load_and_prepare_data(self):
        """Load and prepare the analyzed Threads data"""
        print("Loading analyzed Threads data...")
        self.df = pd.read_csv(self.data_path)

        print(f"Loaded {len(self.df)} records with {len(self.df.columns)} features")

        # Handle missing values
        self.df = self.df.fillna({
            'text': '',
            'cleaned_text': '',
            'emojis': '',
            'emoji_count': 0,
            'word_count': 0,
            'char_count': 0,
            'positive_keywords': 0,
            'negative_keywords': 0,
            'combined_sentiment_score': 0.0
        })

        # Convert boolean columns
        boolean_cols = ['has_text', 'is_nested', 'has_question', 'has_exclamation', 'has_caps']
        for col in boolean_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.lower().isin(['true', '1', 'yes'])

        print("Data preparation complete!")
        return self.df

    def prepare_features(self, use_text_features=True, max_tfidf_features=1000):
        """Prepare feature matrix and target variable"""
        print("Preparing features for ML models...")

        # Numerical features
        numerical_features = [
            'reply_length', 'emoji_count', 'word_count', 'char_count',
            'positive_keywords', 'negative_keywords', 'combined_sentiment_score',
            'vader_compound', 'vader_positive', 'vader_negative', 'vader_neutral',
            'textblob_polarity', 'textblob_subjectivity'
        ]

        # Boolean features
        boolean_features = ['has_text', 'is_nested', 'has_question', 'has_exclamation', 'has_caps']

        # Filter features that exist in the dataset
        available_numerical = [f for f in numerical_features if f in self.df.columns]
        available_boolean = [f for f in boolean_features if f in self.df.columns]

        # Create feature matrix
        feature_df = pd.DataFrame()

        # Add numerical features
        for feature in available_numerical:
            feature_df[feature] = pd.to_numeric(self.df[feature], errors='coerce').fillna(0)

        # Add boolean features
        for feature in available_boolean:
            feature_df[feature] = self.df[feature].astype(int)

        # Add time-based features if timestamp exists
        if 'timestamp' in self.df.columns:
            try:
                timestamps = pd.to_datetime(self.df['timestamp'])
                feature_df['hour'] = timestamps.dt.hour
                feature_df['day_of_week'] = timestamps.dt.dayofweek
                feature_df['is_weekend'] = (timestamps.dt.dayofweek >= 5).astype(int)
            except:
                print("Could not parse timestamps, skipping time features")

        # Add text features using TF-IDF
        if use_text_features and 'cleaned_text' in self.df.columns:
            print(f"Adding TF-IDF features (max {max_tfidf_features} features)...")

            # Use cleaned_text if available, otherwise fall back to text
            text_column = 'cleaned_text' if self.df['cleaned_text'].notna().sum() > 0 else 'text'
            text_data = self.df[text_column].fillna('').astype(str)

            # Create TF-IDF features
            tfidf = TfidfVectorizer(
                max_features=max_tfidf_features,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )

            tfidf_features = tfidf.fit_transform(text_data)
            tfidf_df = pd.DataFrame(
                tfidf_features.toarray(),
                columns=[f'tfidf_{word}' for word in tfidf.get_feature_names_out()]
            )

            # Combine with other features
            feature_df = pd.concat([feature_df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

        self.X = feature_df

        # Prepare target variable - use final_sentiment if available, otherwise vader_sentiment
        target_col = 'final_sentiment' if 'final_sentiment' in self.df.columns else 'vader_sentiment'
        self.y = self.df[target_col].fillna('neutral')

        print(f"Feature matrix shape: {self.X.shape}")
        print(f"Target distribution: {self.y.value_counts().to_dict()}")

        return self.X, self.y

    def initialize_models(self):
        """Initialize all ML models to test"""
        print("Initializing ML models...")

        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42,
                learning_rate=0.1
            ),
            'svm': SVC(
                random_state=42,
                class_weight='balanced',
                probability=True,
                kernel='rbf'
            ),
            'naive_bayes': MultinomialNB(
                alpha=1.0
            ),
            'knn': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                random_state=42,
                max_iter=500,
                early_stopping=True
            )
        }

        print(f"Initialized {len(self.models)} models")
        return self.models

    def train_and_evaluate_models(self, test_size=0.2, cv_folds=5):
        """Train and evaluate all models"""
        print(f"Training and evaluating models with {cv_folds}-fold cross-validation...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=42,
            stratify=self.y
        )

        # Scale features for models that need it
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        results = {}
        predictions = {}

        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")

            try:
                # Choose scaled or unscaled features based on model type
                if model_name in ['svm', 'neural_network', 'knn', 'logistic_regression']:
                    X_train_model = X_train_scaled
                    X_test_model = X_test_scaled
                else:
                    X_train_model = X_train
                    X_test_model = X_test

                # Handle Naive Bayes (needs non-negative features)
                if model_name == 'naive_bayes':
                    # Use only non-negative features for Naive Bayes
                    X_train_model = np.abs(X_train_model)
                    X_test_model = np.abs(X_test_model)

                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_train_model, y_train,
                    cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                    scoring='accuracy',
                    n_jobs=-1
                )

                # Train final model
                model.fit(X_train_model, y_train)

                # Predictions
                y_pred = model.predict(X_test_model)
                y_prob = None

                # Get probabilities if available
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X_test_model)

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)

                # Store results
                results[model_name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'test_accuracy': accuracy,
                    'cv_scores': cv_scores,
                    'classification_report': classification_report(y_test, y_pred, output_dict=True),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
                }

                # Store predictions for saving
                predictions[model_name] = {
                    'predictions': y_pred,
                    'probabilities': y_prob,
                    'test_indices': X_test.index.tolist()
                }

                print(f"{model_name} - CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                print(f"{model_name} - Test Accuracy: {accuracy:.4f}")

            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}

        self.results = results
        self.predictions = predictions

        return results, predictions

    def save_predictions(self, output_dir='./'):
        """Save predictions from each model to separate CSV files"""
        print("\nSaving model predictions...")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        for model_name, pred_data in self.predictions.items():
            if 'predictions' not in pred_data:
                continue

            # Create output dataframe with original data + predictions
            output_df = self.df.copy()

            # Add prediction columns
            output_df['ml_prediction'] = None
            output_df['ml_confidence'] = None

            # Fill in predictions for test indices
            test_indices = pred_data['test_indices']
            predictions = pred_data['predictions']

            output_df.loc[test_indices, 'ml_prediction'] = predictions

            # Add confidence scores if available
            if pred_data['probabilities'] is not None:
                # Use max probability as confidence
                confidences = np.max(pred_data['probabilities'], axis=1)
                output_df.loc[test_indices, 'ml_confidence'] = confidences

            # Add model metadata
            output_df['ml_model'] = model_name
            output_df['ml_timestamp'] = timestamp

            # Save to CSV
            filename = f"threads_comments_full_{model_name}_analyzed.csv"
            filepath = os.path.join(output_dir, filename)
            output_df.to_csv(filepath, index=False)
            print(f"Saved {model_name} predictions to {filename}")

        # Save results summary
        summary_filename = f"threads_ml_results_summary_{timestamp}.txt"
        with open(summary_filename, 'w') as f:
            f.write("=== THREADS ML SENTIMENT ANALYSIS RESULTS ===\n\n")

            # Model comparison
            f.write("MODEL PERFORMANCE COMPARISON:\n")
            f.write("-" * 50 + "\n")

            # Sort models by test accuracy
            sorted_models = sorted(
                [(name, res) for name, res in self.results.items() if 'test_accuracy' in res],
                key=lambda x: x[1]['test_accuracy'],
                reverse=True
            )

            for model_name, results in sorted_models:
                f.write(f"\n{model_name.upper()}:\n")
                f.write(f"  Cross-validation: {results['cv_mean']:.4f} (±{results['cv_std']:.4f})\n")
                f.write(f"  Test accuracy: {results['test_accuracy']:.4f}\n")

                # Add classification report summary
                if 'classification_report' in results:
                    cr = results['classification_report']
                    f.write(f"  Precision (macro): {cr['macro avg']['precision']:.4f}\n")
                    f.write(f"  Recall (macro): {cr['macro avg']['recall']:.4f}\n")
                    f.write(f"  F1-score (macro): {cr['macro avg']['f1-score']:.4f}\n")

        print(f"Saved results summary to {summary_filename}")

    def display_results(self):
        """Display formatted results"""
        print("\n" + "=" * 60)
        print("MACHINE LEARNING RESULTS SUMMARY")
        print("=" * 60)

        # Sort by test accuracy
        sorted_results = sorted(
            [(name, res) for name, res in self.results.items() if 'test_accuracy' in res],
            key=lambda x: x[1]['test_accuracy'],
            reverse=True
        )

        print(f"\n{'Model':<20} {'CV Accuracy':<15} {'Test Accuracy':<15} {'F1-Score':<10}")
        print("-" * 65)

        for model_name, results in sorted_results:
            cv_acc = f"{results['cv_mean']:.4f}±{results['cv_std']:.4f}"
            test_acc = f"{results['test_accuracy']:.4f}"

            # Get macro F1 score
            f1_score = "N/A"
            if 'classification_report' in results:
                f1_score = f"{results['classification_report']['macro avg']['f1-score']:.4f}"

            print(f"{model_name:<20} {cv_acc:<15} {test_acc:<15} {f1_score:<10}")

        # Show best model details
        if sorted_results:
            best_model_name, best_results = sorted_results[0]
            print(f"\n🏆 BEST MODEL: {best_model_name}")
            print(f"Test Accuracy: {best_results['test_accuracy']:.4f}")

            if 'classification_report' in best_results:
                print(f"\nDetailed Classification Report for {best_model_name}:")
                cr = best_results['classification_report']
                for class_name in ['negative', 'neutral', 'positive']:
                    if class_name in cr:
                        print(
                            f"  {class_name}: P={cr[class_name]['precision']:.3f}, R={cr[class_name]['recall']:.3f}, F1={cr[class_name]['f1-score']:.3f}")


def main():
    """Main execution function"""
    # Configuration
    data_file = "nlp_analysis/threads_comments_full_analyzed.csv"

    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found!")
        print("Please ensure the analyzed CSV file is in the current directory.")
        return

    print("=== THREADS ML SENTIMENT ANALYSIS ===\n")

    # Initialize analyzer
    analyzer = ThreadsMLAnalyzer(data_file)

    # Load and prepare data
    df = analyzer.load_and_prepare_data()

    # Prepare features
    X, y = analyzer.prepare_features(use_text_features=True, max_tfidf_features=500)

    # Initialize models
    models = analyzer.initialize_models()

    # Train and evaluate
    results, predictions = analyzer.train_and_evaluate_models()

    # Display results
    analyzer.display_results()

    # Save predictions
    analyzer.save_predictions()

    print("\n✅ Analysis complete! Check the generated CSV files for model predictions.")


if __name__ == "__main__":
    main()