import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import List, Dict, Union, Tuple


class SentimentPredictor:
    """
    Production-ready sentiment predictor using trained custom models
    """

    def __init__(self, model_path: str):
        """
        Load a trained sentiment model

        Args:
            model_path: Path to the saved model pickle file
        """
        self.model_path = model_path
        self.model_package = None
        self.load_model()

    def load_model(self):
        """Load the trained model package"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model_package = pickle.load(f)

            print(f"Loaded model: {self.model_package['model_name']}")
            print(f"Accuracy: {self.model_package['accuracy']:.4f}")
            print(f"Feature type: {self.model_package['feature_type']}")
            print(f"Training date: {self.model_package['training_date']}")

        except Exception as e:
            raise ValueError(f"Error loading model: {e}")

    def extract_features(self, texts: List[str]) -> pd.DataFrame:
        """
        Extract numeric features from text data

        Args:
            texts: List of text strings

        Returns:
            DataFrame with extracted features
        """
        features = []

        for text in texts:
            text_str = str(text) if text else ""

            # Basic text features
            word_count = len(text_str.split())
            char_count = len(text_str)
            reply_length = char_count

            # Emoji features
            emoji_pattern = re.compile(
                r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+')
            emojis = emoji_pattern.findall(text_str)
            emoji_count = len(emojis)

            # Punctuation features
            has_question = 1 if '?' in text_str else 0
            has_exclamation = 1 if '!' in text_str else 0
            has_caps = 1 if any(c.isupper() for c in text_str) else 0

            question_count = text_str.count('?')
            exclamation_count = text_str.count('!')
            caps_ratio = sum(1 for c in text_str if c.isupper()) / max(len(text_str), 1)

            # Text structure features
            sentence_count = len(re.findall(r'[.!?]+', text_str)) + 1 if text_str else 1
            words = text_str.split()
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            text_length = len(text_str)

            # Keyword-based sentiment indicators (basic)
            positive_words = ['good', 'great', 'awesome', 'amazing', 'love', 'excellent', 'perfect', 'wonderful',
                              'fantastic', 'best']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointing', 'useless',
                              'pathetic', 'disgusting']

            text_lower = text_str.lower()
            positive_keywords = sum(1 for word in positive_words if word in text_lower)
            negative_keywords = sum(1 for word in negative_words if word in text_lower)

            # Combine features
            feature_dict = {
                'word_count': word_count,
                'char_count': char_count,
                'emoji_count': emoji_count,
                'reply_length': reply_length,
                'has_question': has_question,
                'has_exclamation': has_exclamation,
                'has_caps': has_caps,
                'positive_keywords': positive_keywords,
                'negative_keywords': negative_keywords,
                'question_count': question_count,
                'exclamation_count': exclamation_count,
                'caps_ratio': caps_ratio,
                'sentence_count': sentence_count,
                'avg_word_length': avg_word_length,
                'text_length': text_length,
                # Placeholder values for sentiment scores (would need VADER/TextBlob)
                'vader_compound': 0.0,
                'textblob_polarity': 0.0,
                'textblob_subjectivity': 0.5,
                'vader_abs_compound': 0.0,
                'textblob_abs_polarity': 0.0,
                'models_agree': 1
            }

            features.append(feature_dict)

        return pd.DataFrame(features)

    def predict(self, texts: Union[str, List[str]], return_probabilities: bool = True) -> Dict:
        """
        Predict sentiment for given texts

        Args:
            texts: Single text string or list of texts
            return_probabilities: Whether to return prediction probabilities

        Returns:
            Dictionary with predictions and optional probabilities
        """
        if isinstance(texts, str):
            texts = [texts]

        model = self.model_package['model']
        feature_type = self.model_package['feature_type']

        try:
            if feature_type == 'text':
                # Text-only model
                vectorizer = self.model_package['vectorizer']
                X = vectorizer.transform(texts)
                predictions = model.predict(X)
                if return_probabilities:
                    probabilities = model.predict_proba(X)

            elif feature_type == 'numeric':
                # Numeric-only model
                features_df = self.extract_features(texts)
                required_features = self.model_package['features']
                X = features_df[required_features]

                scaler = self.model_package['scaler']
                X_scaled = scaler.transform(X)

                predictions = model.predict(X_scaled)
                if return_probabilities:
                    probabilities = model.predict_proba(X_scaled)

            elif feature_type == 'hybrid':
                # Hybrid model (text + numeric)
                vectorizer = self.model_package['vectorizer']
                X_text = vectorizer.transform(texts)

                features_df = self.extract_features(texts)
                required_features = self.model_package['features']
                X_numeric = features_df[required_features]

                scaler = self.model_package['scaler']
                X_numeric_scaled = scaler.transform(X_numeric)

                # Combine features
                X_combined = np.hstack([X_text.toarray(), X_numeric_scaled])

                predictions = model.predict(X_combined)
                if return_probabilities:
                    probabilities = model.predict_proba(X_combined)

            # Convert predictions back to labels
            label_encoder = self.model_package['label_encoder']
            predicted_labels = label_encoder.inverse_transform(predictions)

            results = {
                'predictions': predicted_labels.tolist(),
                'model_name': self.model_package['model_name'],
                'feature_type': feature_type,
                'accuracy': self.model_package['accuracy']
            }

            if return_probabilities:
                # Get class names and probabilities
                classes = label_encoder.classes_
                prob_dict = []
                for i, text in enumerate(texts):
                    text_probs = {}
                    for j, class_name in enumerate(classes):
                        text_probs[class_name] = probabilities[i][j]
                    prob_dict.append(text_probs)

                results['probabilities'] = prob_dict
                results['confidence'] = [max(prob_dict[i].values()) for i in range(len(texts))]

            return results

        except Exception as e:
            raise ValueError(f"Error during prediction: {e}")

    def predict_single(self, text: str) -> Dict:
        """
        Predict sentiment for a single text with detailed output

        Args:
            text: Text string to analyze

        Returns:
            Dictionary with detailed prediction results
        """
        result = self.predict([text], return_probabilities=True)

        return {
            'text': text,
            'prediction': result['predictions'][0],
            'confidence': result['confidence'][0],
            'probabilities': result['probabilities'][0],
            'model_info': {
                'name': result['model_name'],
                'accuracy': result['accuracy'],
                'feature_type': result['feature_type']
            }
        }

    def batch_predict(self, csv_file: str, text_column: str, output_file: str = None) -> pd.DataFrame:
        """
        Predict sentiment for a batch of texts from CSV

        Args:
            csv_file: Path to CSV file
            text_column: Name of column containing text
            output_file: Optional output file path

        Returns:
            DataFrame with predictions added
        """
        print(f"Loading data from {csv_file}...")
        df = pd.read_csv(csv_file)

        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV")

        print(f"Predicting sentiment for {len(df)} texts...")
        texts = df[text_column].fillna('').astype(str).tolist()

        # Predict in batches to avoid memory issues
        batch_size = 1000
        all_predictions = []
        all_probabilities = []
        all_confidences = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = self.predict(batch_texts, return_probabilities=True)

            all_predictions.extend(batch_results['predictions'])
            all_probabilities.extend(batch_results['probabilities'])
            all_confidences.extend(batch_results['confidence'])

            print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")

        # Add results to dataframe
        df['custom_prediction'] = all_predictions
        df['custom_confidence'] = all_confidences

        # Add probability columns
        if all_probabilities:
            classes = list(all_probabilities[0].keys())
            for class_name in classes:
                df[f'custom_prob_{class_name}'] = [probs[class_name] for probs in all_probabilities]

        if output_file:
            df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")

        return df

    def compare_with_baselines(self, csv_file: str, text_column: str,
                               vader_col: str = None, textblob_col: str = None) -> Dict:
        """
        Compare custom model performance with baseline models

        Args:
            csv_file: Path to CSV with baseline predictions
            text_column: Column with text data
            vader_col: Column with VADER predictions
            textblob_col: Column with TextBlob predictions

        Returns:
            Comparison statistics
        """
        df = pd.read_csv(csv_file)

        # Get custom predictions
        texts = df[text_column].fillna('').astype(str).tolist()
        custom_results = self.predict(texts, return_probabilities=True)

        comparison = {
            'total_texts': len(texts),
            'custom_model': {
                'name': self.model_package['model_name'],
                'accuracy': self.model_package['accuracy'],
                'predictions': custom_results['predictions'],
                'avg_confidence': np.mean(custom_results['confidence'])
            }
        }

        # Compare with VADER if available
        if vader_col and vader_col in df.columns:
            vader_preds = df[vader_col].tolist()
            agreement = sum(1 for c, v in zip(custom_results['predictions'], vader_preds) if c == v)
            comparison['vader_agreement'] = {
                'total_agreement': agreement,
                'agreement_rate': agreement / len(texts),
                'disagreements': len(texts) - agreement
            }

        # Compare with TextBlob if available
        if textblob_col and textblob_col in df.columns:
            textblob_preds = df[textblob_col].tolist()
            agreement = sum(1 for c, t in zip(custom_results['predictions'], textblob_preds) if c == t)
            comparison['textblob_agreement'] = {
                'total_agreement': agreement,
                'agreement_rate': agreement / len(texts),
                'disagreements': len(texts) - agreement
            }

        return comparison


class SentimentAnalyzer:
    """
    High-level sentiment analysis interface that can use multiple models
    """

    def __init__(self):
        self.models = {}
        self.default_model = None

    def add_model(self, name: str, model_path: str, set_default: bool = False):
        """Add a trained model to the analyzer"""
        self.models[name] = SentimentPredictor(model_path)
        if set_default or not self.default_model:
            self.default_model = name
            print(f"Set {name} as default model")

    def analyze(self, text: str, model_name: str = None) -> Dict:
        """Analyze sentiment using specified or default model"""
        model_name = model_name or self.default_model
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        return self.models[model_name].predict_single(text)

    def compare_models(self, text: str) -> Dict:
        """Compare predictions from all loaded models"""
        results = {}
        for name, model in self.models.items():
            try:
                results[name] = model.predict_single(text)
            except Exception as e:
                results[name] = {'error': str(e)}

        return {
            'text': text,
            'model_comparisons': results,
            'consensus': self._find_consensus(results)
        }

    def _find_consensus(self, results: Dict) -> Dict:
        """Find consensus prediction across models"""
        predictions = []
        confidences = []

        for name, result in results.items():
            if 'error' not in result:
                predictions.append(result['prediction'])
                confidences.append(result['confidence'])

        if not predictions:
            return {'consensus': None, 'confidence': 0.0}

        # Find most common prediction
        from collections import Counter
        pred_counts = Counter(predictions)
        consensus_pred = pred_counts.most_common(1)[0][0]
        consensus_count = pred_counts.most_common(1)[0][1]

        # Average confidence of models that agree with consensus
        consensus_confidences = [conf for pred, conf in zip(predictions, confidences)
                                 if pred == consensus_pred]
        avg_confidence = np.mean(consensus_confidences) if consensus_confidences else 0.0

        return {
            'consensus': consensus_pred,
            'confidence': avg_confidence,
            'agreement_rate': consensus_count / len(predictions),
            'agreeing_models': consensus_count,
            'total_models': len(predictions)
        }


def demo_usage():
    """Demonstrate how to use the sentiment predictor"""

    print("=== SENTIMENT PREDICTOR DEMO ===\n")

    # Example usage - you'll need to update the model path
    model_path = "best_sentiment_model_20250825_123456.pkl"  # Update with actual path

    try:
        # Initialize predictor
        predictor = SentimentPredictor(model_path)

        # Test single prediction
        test_text = "This is absolutely amazing! I love it so much! 🎉"
        result = predictor.predict_single(test_text)

        print("Single Text Analysis:")
        print(f"Text: {result['text']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Probabilities: {result['probabilities']}")
        print(f"Model: {result['model_info']['name']}")
        print()

        # Test batch prediction
        test_texts = [
            "I absolutely love this product!",
            "This is terrible, worst purchase ever",
            "It's okay, nothing special",
            "Amazing quality and fast shipping! 😊",
            "Disappointed with the service"
        ]

        batch_results = predictor.predict(test_texts, return_probabilities=True)

        print("Batch Analysis:")
        for i, text in enumerate(test_texts):
            print(f"Text: {text}")
            print(f"Prediction: {batch_results['predictions'][i]} "
                  f"(confidence: {batch_results['confidence'][i]:.3f})")
            print()

        # Demonstrate analyzer with multiple models
        analyzer = SentimentAnalyzer()
        analyzer.add_model("custom_model", model_path, set_default=True)

        comparison = analyzer.compare_models("This product is absolutely fantastic!")
        print("Model Comparison:")
        print(f"Text: {comparison['text']}")
        print(f"Consensus: {comparison['consensus']}")
        print()

    except Exception as e:
        print(f"Demo error: {e}")
        print("\nTo use this predictor:")
        print("1. Train a model using the sentiment_model_trainer.py script")
        print("2. Update the model_path variable with your trained model file")
        print("3. Run this script to make predictions")


def main():
    """Main execution for command-line usage"""
    import sys

    if len(sys.argv) < 3:
        print("Usage:")
        print("  python sentiment_predictor.py <model_path> <text>")
        print("  python sentiment_predictor.py <model_path> --batch <csv_file> <text_column>")
        print("  python sentiment_predictor.py --demo")
        return

    if sys.argv[1] == "--demo":
        demo_usage()
        return

    model_path = sys.argv[1]

    if sys.argv[2] == "--batch":
        if len(sys.argv) < 5:
            print("Batch usage: python sentiment_predictor.py <model_path> --batch <csv_file> <text_column>")
            return

        csv_file = sys.argv[3]
        text_column = sys.argv[4]
        output_file = sys.argv[5] if len(sys.argv) > 5 else None

        predictor = SentimentPredictor(model_path)
        results_df = predictor.batch_predict(csv_file, text_column, output_file)

        print(f"\nPrediction Summary:")
        print(results_df['custom_prediction'].value_counts())
        print(f"Average confidence: {results_df['custom_confidence'].mean():.3f}")

    else:
        # Single text prediction
        text = " ".join(sys.argv[2:])

        predictor = SentimentPredictor(model_path)
        result = predictor.predict_single(text)

        print(f"Text: {result['text']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Model: {result['model_info']['name']}")


if __name__ == "__main__":
    main()