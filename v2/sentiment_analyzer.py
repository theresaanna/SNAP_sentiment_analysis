import pandas as pd
from datetime import datetime
import time
from typing import Dict, List, Optional
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle

# Neural Network imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, GlobalMaxPooling1D, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings

warnings.filterwarnings('ignore')

# Download required NLTK data with SSL fix
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# Fallback text processing without NLTK
def simple_tokenize(text):
    """Simple tokenizer fallback if NLTK fails"""
    return re.findall(r'\b\w+\b', text.lower())


def simple_lemmatize(word):
    """Simple lemmatizer fallback"""
    # Basic stemming - remove common suffixes
    suffixes = ['ing', 'ed', 'er', 'est', 'ly', 's']
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word


# Try to download NLTK data, use fallbacks if it fails
nltk_available = False
try:
    import nltk

    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer

    nltk_available = True
    print("✓ NLTK loaded successfully")
except Exception as e:
    print(f"⚠️  NLTK failed to load: {e}")
    print("Using simple text processing fallbacks...")
    nltk_available = False


class CSVSentimentAnalyzer:
    """CSV-only sentiment analyzer with machine learning"""

    def __init__(self):
        """Initialize the CSV-based sentiment analyzer"""

        # Traditional ML models
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', probability=True, random_state=42),
            'naive_bayes': MultinomialNB(),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }

        # ML components
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            min_df=2,
            max_df=0.8
        )

        # Neural Network components
        self.tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
        self.max_sequence_length = 100
        self.embedding_dim = 100
        self.neural_models = {}

        self.label_encoder = LabelEncoder()
        self.nltk_available = nltk_available
        if nltk_available:
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.lemmatizer = None
        self.trained_model = None
        self.model_name = None
        self.model_type = 'traditional'

        # Enhanced keyword lists for feature engineering
        self.positive_keywords = [
            'love', 'great', 'awesome', 'amazing', 'perfect', 'excellent', 'fantastic',
            'wonderful', 'brilliant', 'outstanding', 'superb', 'fabulous', 'incredible',
            'thank', 'thanks', 'grateful', 'appreciate', 'helpful', 'useful', 'valuable',
            'agree', 'exactly', 'correct', 'right', 'yes', 'absolutely', 'definitely',
            'smart', 'clever', 'wise', 'insightful', 'thoughtful', 'good', 'nice',
            'congrats', 'congratulations', 'proud', 'impressed', 'respect', 'beautiful'
        ]

        self.negative_keywords = [
            'hate', 'terrible', 'awful', 'horrible', 'disgusting', 'pathetic', 'stupid',
            'dumb', 'idiotic', 'moronic', 'ridiculous', 'absurd', 'nonsense', 'crazy',
            'wrong', 'false', 'lie', 'lies', 'fake', 'fraud', 'scam', 'bullshit',
            'disagree', 'no', 'never', 'impossible', 'outrageous', 'disappointed',
            'angry', 'mad', 'furious', 'upset', 'annoyed', 'frustrated', 'waste',
            'useless', 'pointless', 'meaningless', 'worthless', 'fail', 'failed'
        ]

    def preprocess_text(self, text: str) -> str:
        """Advanced text preprocessing for ML models"""
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove mentions and hashtags (but keep the text)
        text = re.sub(r'[@#]\w+', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep emoticons
        text = re.sub(r'[^\w\s\!\?\.\,\:\;\(\)\-\'\"]+', ' ', text)

        # Tokenize and lemmatize
        if self.nltk_available and self.lemmatizer:
            try:
                tokens = word_tokenize(text)
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens if len(token) > 2]
                return ' '.join(tokens)
            except:
                pass

        # Fallback processing
        tokens = simple_tokenize(text)
        tokens = [simple_lemmatize(token) for token in tokens if len(token) > 2]
        return ' '.join(tokens)

    def extract_ml_features(self, text: str) -> Dict:
        """Extract comprehensive features for ML models"""
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
                'negative_keywords': 0,
                'punctuation_count': 0,
                'avg_word_length': 0,
                'sentence_count': 0
            }

        # Extract emojis
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

        emojis = emoji_pattern.findall(text)
        cleaned_text = emoji_pattern.sub('', text).strip()
        processed_text = self.preprocess_text(cleaned_text)

        # Count features
        words = cleaned_text.split()
        text_lower = text.lower()

        positive_count = sum(1 for keyword in self.positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in self.negative_keywords if keyword in text_lower)

        punctuation_count = len([c for c in text if c in '!?.,;:'])
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        sentence_count = len([s for s in re.split(r'[.!?]+', text) if s.strip()])

        return {
            'cleaned_text': processed_text,
            'emojis': ''.join(emojis),
            'emoji_count': len(emojis),
            'has_question': '?' in text,
            'has_exclamation': '!' in text,
            'has_caps': any(word.isupper() and len(word) > 2 for word in words),
            'word_count': len(words),
            'char_count': len(text),
            'positive_keywords': positive_count,
            'negative_keywords': negative_count,
            'punctuation_count': punctuation_count,
            'avg_word_length': avg_word_length,
            'sentence_count': max(1, sentence_count)
        }

    def create_neural_models(self, num_classes: int) -> dict:
        """Create various neural network architectures"""
        models = {}

        # Simple Dense Neural Network
        models['dense_nn'] = Sequential([
            Dense(128, activation='relu', input_shape=(self.max_sequence_length,)),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])

        # LSTM Neural Network
        models['lstm'] = Sequential([
            Embedding(10000, self.embedding_dim, input_length=self.max_sequence_length),
            LSTM(64, dropout=0.5, recurrent_dropout=0.5),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])

        # CNN Neural Network
        models['cnn'] = Sequential([
            Embedding(10000, self.embedding_dim, input_length=self.max_sequence_length),
            Conv1D(128, 5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])

        # Hybrid CNN-LSTM
        models['cnn_lstm'] = Sequential([
            Embedding(10000, self.embedding_dim, input_length=self.max_sequence_length),
            Conv1D(64, 3, activation='relu'),
            LSTM(32, dropout=0.3, recurrent_dropout=0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])

        # Compile all models
        for name, model in models.items():
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

        return models

    def load_and_process_csv(self, csv_file: str) -> pd.DataFrame:
        """Load CSV and extract features"""
        print(f"Loading CSV file: {csv_file}")
        df = pd.read_csv(csv_file)

        if 'text' not in df.columns:
            raise ValueError("CSV file must contain a 'text' column")

        print(f"Loaded {len(df)} rows from CSV")

        # Extract features if not already present
        if 'cleaned_text' not in df.columns:
            print("Extracting features from text data...")
            feature_data = []

            for i, row in df.iterrows():
                if i % 100 == 0:
                    print(f"Feature extraction progress: {i}/{len(df)}")

                text = row.get('text', '')
                features = self.extract_ml_features(text)

                # Combine original row with features
                combined = {**row.to_dict(), **features}
                combined['reply_length'] = len(text)
                combined['has_text'] = bool(text.strip())

                # Add hour if timestamp exists
                if 'timestamp' in row and pd.notna(row['timestamp']):
                    try:
                        combined['hour'] = pd.to_datetime(row['timestamp']).hour
                    except:
                        combined['hour'] = None

                feature_data.append(combined)

            df = pd.DataFrame(feature_data)

        return df

    def create_training_data_from_manual_tags(self, df: pd.DataFrame) -> tuple:
        """Create training data from manually tagged sentiment labels"""
        print("Creating training data from manual sentiment tags...")

        # Look for manual sentiment columns in order of preference
        sentiment_columns = ['manual_sentiment', 'tagged_sentiment', 'sentiment', 'label']
        sentiment_column = None

        for col in sentiment_columns:
            if col in df.columns:
                sentiment_column = col
                break

        if not sentiment_column:
            raise ValueError(f"No sentiment column found. Expected one of: {sentiment_columns}")

        print(f"Using sentiment column: {sentiment_column}")

        # Filter for manually tagged data
        valid_data = df[df[sentiment_column].notna() &
                        df['text'].notna() &
                        (df[sentiment_column] != '')].copy()

        # Only include valid sentiment labels
        valid_sentiments = ['positive', 'negative', 'neutral']
        valid_data = valid_data[valid_data[sentiment_column].str.lower().isin(valid_sentiments)]

        if len(valid_data) == 0:
            raise ValueError(
                "No valid manually tagged data found. Make sure sentiment values are 'positive', 'negative', or 'neutral'")

        print(f"Found {len(valid_data)} manually tagged samples")

        # Show distribution
        sentiment_dist = valid_data[sentiment_column].value_counts()
        print("Sentiment distribution:")
        for sentiment, count in sentiment_dist.items():
            print(f"  {sentiment}: {count}")

        # Extract features for all texts
        features_list = []
        texts = []
        labels = []

        for _, row in valid_data.iterrows():
            if row['cleaned_text'] and row['cleaned_text'].strip():
                features_list.append({
                    'word_count': row['word_count'],
                    'char_count': row['char_count'],
                    'emoji_count': row['emoji_count'],
                    'positive_keywords': row['positive_keywords'],
                    'negative_keywords': row['negative_keywords'],
                    'punctuation_count': row['punctuation_count'],
                    'avg_word_length': row['avg_word_length'],
                    'sentence_count': row['sentence_count'],
                    'has_question': row['has_question'],
                    'has_exclamation': row['has_exclamation'],
                    'has_caps': row['has_caps']
                })
                texts.append(row['cleaned_text'])
                labels.append(row[sentiment_column].lower())

        print(f"Extracted {len(texts)} training samples after preprocessing")

        # Create feature matrix
        X_text = self.vectorizer.fit_transform(texts)

        # Add numerical features
        numerical_features = []
        for features in features_list:
            num_features = [
                features['word_count'],
                features['char_count'],
                features['emoji_count'],
                features['positive_keywords'],
                features['negative_keywords'],
                features['punctuation_count'],
                features['avg_word_length'],
                features['sentence_count'],
                int(features['has_question']),
                int(features['has_exclamation']),
                int(features['has_caps'])
            ]
            numerical_features.append(num_features)

        X_numerical = np.array(numerical_features)

        # Combine text and numerical features
        from scipy.sparse import hstack
        X_combined = hstack([X_text, X_numerical])

        # Encode labels
        y = self.label_encoder.fit_transform(labels)

        return X_combined, y, labels

    def prepare_neural_data(self, texts: list, labels: list) -> tuple:
        """Prepare data for neural network training"""
        # Tokenize texts
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)

        # Pad sequences
        X = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)
        y_categorical = to_categorical(y_encoded)

        return X, y_categorical, y_encoded

    def train_neural_models(self, X, y_categorical, y_encoded, texts):
        """Train neural network models"""
        print("\nTraining Neural Network models...")

        # Split data
        X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded = train_test_split(
            X, y_categorical, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Create neural models
        num_classes = y_categorical.shape[1]
        neural_models = self.create_neural_models(num_classes)

        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)

        results = {}

        for name, model in neural_models.items():
            print(f"\nTraining {name}...")

            try:
                if name == 'dense_nn':
                    # For dense NN, use TF-IDF features
                    tfidf_X = self.vectorizer.transform([' '.join(map(str, seq)) for seq in X])
                    tfidf_X_train, tfidf_X_test = train_test_split(
                        tfidf_X.toarray(), test_size=0.2, random_state=42, stratify=y_encoded
                    )

                    # Recreate dense model with correct input shape
                    dense_model = Sequential([
                        Dense(128, activation='relu', input_shape=(tfidf_X_train.shape[1],)),
                        Dropout(0.5),
                        Dense(64, activation='relu'),
                        Dropout(0.3),
                        Dense(32, activation='relu'),
                        Dense(num_classes, activation='softmax')
                    ])
                    dense_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

                    history = dense_model.fit(
                        tfidf_X_train, y_train,
                        epochs=50,
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=[early_stopping, reduce_lr],
                        verbose=0
                    )

                    test_loss, test_acc = dense_model.evaluate(tfidf_X_test, y_test, verbose=0)
                    neural_models[name] = dense_model

                else:
                    # For sequence models
                    history = model.fit(
                        X_train, y_train,
                        epochs=50,
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=[early_stopping, reduce_lr],
                        verbose=0
                    )

                    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

                results[name] = {
                    'test_accuracy': test_acc,
                    'test_loss': test_loss,
                    'model': model,
                    'history': history
                }

                print(f"Test Accuracy: {test_acc:.3f}")

            except Exception as e:
                print(f"Error training {name}: {e}")
                continue

        self.neural_models = neural_models
        return results

    def train_all_models(self, X, y, labels, texts):
        """Train all types of models"""
        print("\n" + "=" * 50)
        print("COMPREHENSIVE MODEL TRAINING")
        print("=" * 50)

        # Traditional ML Models
        print("\n1. TRADITIONAL MACHINE LEARNING MODELS")
        print("-" * 40)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        best_score = 0
        best_model = None
        best_model_name = None
        best_model_type = None

        traditional_results = {}

        for name, model in self.models.items():
            print(f"\nTraining {name}...")

            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

            # Train on full training set
            model.fit(X_train, y_train)

            # Test set performance
            test_score = model.score(X_test, y_test)

            traditional_results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_score': test_score,
                'model': model,
                'type': 'traditional'
            }

            print(f"CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            print(f"Test Score: {test_score:.3f}")

            if test_score > best_score:
                best_score = test_score
                best_model = model
                best_model_name = name
                best_model_type = 'traditional'

        # Neural Network Models
        print("\n2. NEURAL NETWORK MODELS")
        print("-" * 40)

        neural_results = {}
        try:
            # Prepare neural data
            X_neural, y_categorical, y_encoded = self.prepare_neural_data(texts, labels)
            neural_results = self.train_neural_models(X_neural, y_categorical, y_encoded, texts)

            # Check if any neural model is better
            for name, result in neural_results.items():
                if result['test_accuracy'] > best_score:
                    best_score = result['test_accuracy']
                    best_model = result['model']
                    best_model_name = name
                    best_model_type = 'neural'
        except Exception as e:
            print(f"Neural network training failed: {e}")

        # Set the best model
        self.trained_model = best_model
        self.model_name = best_model_name
        self.model_type = best_model_type

        # Summary
        print("\n" + "=" * 50)
        print("FINAL RESULTS SUMMARY")
        print("=" * 50)
        print(f"Best Model: {best_model_name} ({best_model_type})")
        print(f"Best Score: {best_score:.3f}")

        print(f"\nTraditional ML Results:")
        for name, result in traditional_results.items():
            print(f"  {name}: {result['test_score']:.3f}")

        if neural_results:
            print(f"\nNeural Network Results:")
            for name, result in neural_results.items():
                print(f"  {name}: {result['test_accuracy']:.3f}")

        # Detailed evaluation of best model
        if best_model_type == 'traditional':
            y_pred = best_model.predict(X_test)
            print(f"\nDetailed evaluation of {best_model_name}:")
            print(classification_report(y_test, y_pred,
                                        target_names=self.label_encoder.classes_))

        return {
            'traditional': traditional_results,
            'neural': neural_results,
            'best_model': best_model_name,
            'best_score': best_score,
            'best_type': best_model_type
        }

    def predict_sentiment(self, text: str, features: Dict) -> Dict:
        """Predict sentiment using the trained model"""
        if not self.trained_model:
            return {
                'ml_sentiment': 'neutral',
                'ml_confidence': 0.0,
                'ml_probabilities': {},
                'model_type': 'none'
            }

        if not text.strip():
            return {
                'ml_sentiment': 'neutral',
                'ml_confidence': 0.0,
                'ml_probabilities': {'positive': 0.33, 'neutral': 0.34, 'negative': 0.33},
                'model_type': self.model_type
            }

        try:
            if self.model_type == 'traditional':
                return self._predict_traditional(text, features)
            elif self.model_type == 'neural':
                return self._predict_neural(text, features)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

        except Exception as e:
            print(f"Error in prediction: {e}")
            return {
                'ml_sentiment': 'neutral',
                'ml_confidence': 0.0,
                'ml_probabilities': {'positive': 0.33, 'neutral': 0.34, 'negative': 0.33},
                'model_type': self.model_type
            }

    def _predict_traditional(self, text: str, features: Dict) -> Dict:
        """Predict using traditional ML models"""
        # Transform text
        X_text = self.vectorizer.transform([features['cleaned_text']])

        # Add numerical features
        numerical_features = np.array([[
            features['word_count'], features['char_count'], features['emoji_count'],
            features['positive_keywords'], features['negative_keywords'],
            features['punctuation_count'], features['avg_word_length'],
            features['sentence_count'], int(features['has_question']),
            int(features['has_exclamation']), int(features['has_caps'])
        ]])

        # Combine features
        from scipy.sparse import hstack
        X_combined = hstack([X_text, numerical_features])

        # Predict
        prediction = self.trained_model.predict(X_combined)[0]
        probabilities = self.trained_model.predict_proba(X_combined)[0]

        sentiment = self.label_encoder.inverse_transform([prediction])[0]
        confidence = float(np.max(probabilities))

        prob_dict = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            prob_dict[class_name] = float(probabilities[i])

        return {
            'ml_sentiment': sentiment,
            'ml_confidence': confidence,
            'ml_probabilities': prob_dict,
            'model_type': 'traditional'
        }

    def _predict_neural(self, text: str, features: Dict) -> Dict:
        """Predict using neural network models"""
        if self.model_name == 'dense_nn':
            # Use TF-IDF for dense NN
            X_tfidf = self.vectorizer.transform([features['cleaned_text']])
            probabilities = self.trained_model.predict(X_tfidf.toarray(), verbose=0)[0]
        else:
            # Use sequences for LSTM/CNN models
            sequences = self.tokenizer.texts_to_sequences([features['cleaned_text']])
            X_padded = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
            probabilities = self.trained_model.predict(X_padded, verbose=0)[0]

        prediction = np.argmax(probabilities)
        sentiment = self.label_encoder.inverse_transform([prediction])[0]
        confidence = float(np.max(probabilities))

        prob_dict = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            prob_dict[class_name] = float(probabilities[i])

        return {
            'ml_sentiment': sentiment,
            'ml_confidence': confidence,
            'ml_probabilities': prob_dict,
            'model_type': 'neural'
        }

    def analyze_csv(self, csv_file: str) -> pd.DataFrame:
        """Main analysis function for CSV files"""
        # Load and process CSV
        df = self.load_and_process_csv(csv_file)

        # Check for manual tags and train models
        sentiment_columns = ['manual_sentiment', 'tagged_sentiment', 'sentiment', 'label']
        has_manual_tags = any(col in df.columns and df[col].notna().sum() > 0 for col in sentiment_columns)

        if has_manual_tags:
            print("\nFound manual sentiment tags - training models...")
            X, y, labels = self.create_training_data_from_manual_tags(df)

            # Extract texts for neural training
            texts = []
            manual_sentiment_col = None
            for col in sentiment_columns:
                if col in df.columns and df[col].notna().sum() > 0:
                    manual_sentiment_col = col
                    break

            for _, row in df.iterrows():
                if pd.notna(row.get(manual_sentiment_col)) and row.get('text'):
                    if row['cleaned_text'] and row['cleaned_text'].strip():
                        texts.append(row['cleaned_text'])

            print(f"Training models with {len(texts)} manually tagged samples...")
            model_results = self.train_all_models(X, y, labels, texts)

        else:
            print("No manual sentiment tags found. Models cannot be trained.")

        # Predict sentiment for all rows
        print("\nPredicting sentiment for all rows...")
        ml_predictions = []

        for i, row in df.iterrows():
            if i % 100 == 0:
                print(f"Prediction progress: {i}/{len(df)}")

            features = {
                'cleaned_text': row.get('cleaned_text', ''),
                'emoji_count': row.get('emoji_count', 0),
                'has_question': row.get('has_question', False),
                'has_exclamation': row.get('has_exclamation', False),
                'has_caps': row.get('has_caps', False),
                'word_count': row.get('word_count', 0),
                'char_count': row.get('char_count', 0),
                'positive_keywords': row.get('positive_keywords', 0),
                'negative_keywords': row.get('negative_keywords', 0),
                'punctuation_count': row.get('punctuation_count', 0),
                'avg_word_length': row.get('avg_word_length', 0),
                'sentence_count': row.get('sentence_count', 1)
            }

            if self.trained_model:
                prediction = self.predict_sentiment(row.get('text', ''), features)
                ml_predictions.append(prediction)
            else:
                ml_predictions.append({
                    'ml_sentiment': 'neutral',
                    'ml_confidence': 0.0,
                    'ml_probabilities': {'positive': 0.33, 'neutral': 0.34, 'negative': 0.33},
                    'model_type': 'none'
                })

        # Add predictions to dataframe
        for i, pred in enumerate(ml_predictions):
            for key, value in pred.items():
                if key == 'ml_probabilities':
                    # Convert dict to string for storage, or skip complex objects
                    if isinstance(value, dict):
                        df.at[i, key] = str(value)  # Store as string
                        # Also add individual probability columns
                        for sentiment, prob in value.items():
                            df.at[i, f'ml_prob_{sentiment}'] = prob
                    else:
                        df.at[i, key] = value
                else:
                    df.at[i, key] = value

        # Convert timestamp if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        print(f"\nML analysis complete! Processed {len(df)} rows.")

        if hasattr(self, 'trained_model') and self.trained_model:
            print(f"Best model: {self.model_name} ({self.model_type})")

        return df

    def save_model(self, filepath: str):
        """Save the trained model and all components"""
        if self.trained_model:
            model_data = {
                'model': self.trained_model,
                'vectorizer': self.vectorizer,
                'label_encoder': self.label_encoder,
                'model_name': self.model_name,
                'model_type': self.model_type,
                'tokenizer': self.tokenizer if hasattr(self, 'tokenizer') else None,
                'max_sequence_length': self.max_sequence_length,
                'embedding_dim': self.embedding_dim
            }

            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)

            print(f"Model saved to {filepath}")

    def generate_report(self, df: pd.DataFrame) -> Dict:
        """Generate analysis report"""
        report = {
            'total_rows': len(df),
            'model_info': {
                'name': self.model_name,
                'type': self.model_type
            }
        }

        if 'ml_sentiment' in df.columns:
            sentiment_dist = df['ml_sentiment'].value_counts()
            report['sentiment_distribution'] = sentiment_dist.to_dict()

            if 'ml_confidence' in df.columns:
                report['confidence_stats'] = {
                    'mean': df['ml_confidence'].mean(),
                    'median': df['ml_confidence'].median(),
                    'std': df['ml_confidence'].std()
                }

                high_conf = df[df['ml_confidence'] > 0.8]
                report['high_confidence'] = {
                    'count': len(high_conf),
                    'percentage': len(high_conf) / len(df) * 100,
                    'sentiment_distribution': high_conf['ml_sentiment'].value_counts().to_dict()
                }

        if 'word_count' in df.columns:
            report['text_stats'] = {
                'avg_word_count': df['word_count'].mean(),
                'avg_char_count': df['char_count'].mean() if 'char_count' in df.columns else None,
                'emoji_usage_rate': (df['emoji_count'] > 0).sum() / len(
                    df) * 100 if 'emoji_count' in df.columns else None
            }

        return report


def main():
    """Main execution function"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python csv_sentiment_analyzer.py your_tagged_file.csv")
        return

    csv_file = sys.argv[1]

    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found.")
        return

    print("=== CSV SENTIMENT ANALYZER ===")
    print(f"Processing: {csv_file}")
    print("=" * 50)

    # Initialize analyzer
    analyzer = CSVSentimentAnalyzer()

    try:
        # Analyze the CSV
        df = analyzer.analyze_csv(csv_file)

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"ml_analysis_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

        # Save model if trained
        if analyzer.trained_model:
            model_file = f"ml_model_{timestamp}.pkl"
            analyzer.save_model(model_file)
            print(f"Model saved to: {model_file}")

        # Generate report
        report = analyzer.generate_report(df)

        # Display results
        print("\n" + "=" * 60)
        print("ANALYSIS RESULTS")
        print("=" * 60)

        print(f"Total rows processed: {report['total_rows']}")
        if analyzer.trained_model:
            print(f"Best model: {report['model_info']['name']} ({report['model_info']['type']})")

        if 'sentiment_distribution' in report:
            print(f"\nPredicted sentiment distribution:")
            for sentiment, count in report['sentiment_distribution'].items():
                pct = (count / report['total_rows']) * 100
                print(f"  {sentiment}: {count} ({pct:.1f}%)")

        if 'confidence_stats' in report:
            print(f"\nConfidence statistics:")
            print(f"  Mean: {report['confidence_stats']['mean']:.3f}")
            print(f"  Median: {report['confidence_stats']['median']:.3f}")

            if 'high_confidence' in report:
                hc = report['high_confidence']
                print(f"  High confidence (>0.8): {hc['count']} ({hc['percentage']:.1f}%)")

        if 'text_stats' in report:
            ts = report['text_stats']
            print(f"\nText statistics:")
            print(f"  Average words per text: {ts['avg_word_count']:.1f}")
            if ts['avg_char_count']:
                print(f"  Average characters per text: {ts['avg_char_count']:.1f}")
            if ts['emoji_usage_rate']:
                print(f"  Emoji usage rate: {ts['emoji_usage_rate']:.1f}%")

        # Show manual vs ML comparison if manual tags exist
        sentiment_columns = ['manual_sentiment', 'tagged_sentiment', 'sentiment', 'label']
        manual_col = None
        for col in sentiment_columns:
            if col in df.columns and df[col].notna().sum() > 0:
                manual_col = col
                break

        if manual_col:
            print(f"\n" + "=" * 60)
            print("MANUAL vs ML COMPARISON")
            print("=" * 60)

            comparison_df = df[df[manual_col].notna()].copy()
            if len(comparison_df) > 0:
                manual_dist = comparison_df[manual_col].value_counts()

                print(f"Manual tags distribution (training data):")
                for sentiment, count in manual_dist.items():
                    pct = (count / len(comparison_df)) * 100
                    print(f"  {sentiment}: {count} ({pct:.1f}%)")

                # Calculate accuracy on training data
                if 'ml_sentiment' in comparison_df.columns:
                    correct = (comparison_df[manual_col].str.lower() == comparison_df['ml_sentiment']).sum()
                    accuracy = correct / len(comparison_df)
                    print(f"\nTraining accuracy: {accuracy:.3f} ({correct}/{len(comparison_df)})")

        # Show sample predictions
        sample_size = min(5, len(df))
        sample_predictions = df[['text', 'ml_sentiment', 'ml_confidence']].head(sample_size)

        print(f"\n" + "=" * 60)
        print("SAMPLE PREDICTIONS")
        print("=" * 60)

        for i, (idx, row) in enumerate(sample_predictions.iterrows()):
            manual_tag = ""
            if manual_col and idx < len(df) and pd.notna(df.iloc[idx].get(manual_col)):
                manual_tag = f" (Manual: {df.iloc[idx][manual_col]})"

            print(f"\nSample {i + 1}:")
            print(f"Text: \"{row['text'][:80]}...\"")
            print(f"ML Prediction: {row['ml_sentiment']}{manual_tag}")
            print(f"Confidence: {row['ml_confidence']:.3f}")

        print(f"\n✅ Analysis complete!")
        print(f"📁 Results: {output_file}")
        if analyzer.trained_model:
            print(f"🤖 Model: {model_file}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()