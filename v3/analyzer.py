#!/usr/bin/env python3

"""
Extended Comprehensive ML Sentiment Analyzer
Runs RoBERTa, DistilBERT, SVM, Naive Bayes, Random Forest, Gradient Boosting, Logistic Regression
with optimized hyperparameters and comprehensive metrics tracking.
"""

import pandas as pd
from datetime import datetime
import os
import sys
import re
import numpy as np
import warnings
from typing import Dict, List, Tuple
import logging
import json
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_auc_score
)

# Traditional ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


# Test imports individually to identify the issue
def check_individual_imports():
    """Check each import individually to identify the specific problem"""
    import_status = {}

    try:
        import torch
        import_status['torch'] = f"✅ {torch.__version__}"
    except ImportError as e:
        import_status['torch'] = f"❌ {e}"

    try:
        import transformers
        import_status['transformers'] = f"✅ {transformers.__version__}"
    except ImportError as e:
        import_status['transformers'] = f"❌ {e}"

    try:
        from transformers import AutoTokenizer
        import_status['AutoTokenizer'] = "✅"
    except ImportError as e:
        import_status['AutoTokenizer'] = f"❌ {e}"

    try:
        from transformers import AutoModelForSequenceClassification
        import_status['AutoModelForSequenceClassification'] = "✅"
    except ImportError as e:
        import_status['AutoModelForSequenceClassification'] = f"❌ {e}"

    try:
        from transformers import pipeline
        import_status['pipeline'] = "✅"
    except ImportError as e:
        import_status['pipeline'] = f"❌ {e}"

    try:
        from torch.utils.data import Dataset
        import_status['Dataset'] = "✅"
    except ImportError as e:
        import_status['Dataset'] = f"❌ {e}"

    return import_status


# Neural network imports with detailed error reporting and TF avoidance
TRANSFORMERS_AVAILABLE = False
try:
    # Check individual imports first
    import_status = check_individual_imports()
    print("🔍 Import Status Check:")
    for component, status in import_status.items():
        print(f"  {component}: {status}")

    # Check if all critical components are available
    critical_failures = [k for k, v in import_status.items() if v.startswith("❌")]

    if critical_failures:
        print(f"\n❌ Critical import failures: {critical_failures}")
        TRANSFORMERS_AVAILABLE = False

        # Try to provide specific fix for TensorFlow conflict
        if any("TFPreTrainedModel" in str(v) for v in import_status.values()):
            print("\n🔧 TensorFlow Conflict Detected!")
            print("Try this fix:")
            print("pip uninstall tensorflow tensorflow-cpu")
            print("pip install transformers --force-reinstall --no-deps")
            print("pip install torch transformers datasets accelerate tokenizers")
    else:
        # If individual imports work, try the minimal PyTorch-only import
        print("\n🔧 Attempting PyTorch-only transformers import...")

        import torch
        from torch.utils.data import Dataset

        # Import transformers components one by one to avoid TF conflicts
        from transformers import AutoTokenizer
        from transformers import AutoModelForSequenceClassification
        from transformers import pipeline

        # Only import training components if we need them (optional)
        try:
            from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, DataCollatorWithPadding

            print("✅ Training components available")
        except ImportError as e:
            print(f"⚠️ Training components not available: {e}")
            print("Fine-tuning will be disabled, but pre-trained models will work")

        TRANSFORMERS_AVAILABLE = True
        print("✅ Core transformers components imported successfully (PyTorch-only)")

except Exception as e:
    print(f"❌ Import failed: {e}")
    print(f"Error type: {type(e).__name__}")

    if "TFPreTrainedModel" in str(e) or "tensorflow" in str(e).lower():
        print("\n🔧 TensorFlow Conflict Resolution:")
        print("1. Uninstall conflicting packages:")
        print("   pip uninstall tensorflow tensorflow-cpu tf-keras")
        print("2. Reinstall transformers for PyTorch only:")
        print("   pip install transformers[torch] --force-reinstall")
        print("3. Or use a clean environment:")
        print("   pip install torch transformers datasets accelerate --no-deps --force-reinstall")

    TRANSFORMERS_AVAILABLE = False

# Create dummy classes if libraries not available
if not TRANSFORMERS_AVAILABLE:
    print("⚠️ Creating dummy classes for missing components...")


    class Dataset:
        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return 0

        def __getitem__(self, idx):
            return {}


    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return None


    class AutoModelForSequenceClassification:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return None

# Suppress warnings
warnings.filterwarnings('ignore')
if TRANSFORMERS_AVAILABLE:
    logging.getLogger("transformers").setLevel(logging.ERROR)

# Optimized hyperparameters for all models
OPTIMIZED_CONFIGS = {
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'bootstrap': True,
        'random_state': 42,
        'n_jobs': -1
    },
    'gradient_boosting': {
        'n_estimators': 150,
        'learning_rate': 0.1,
        'max_depth': 7,
        'min_samples_split': 5,
        'subsample': 0.8,
        'random_state': 42
    },
    'svm': {
        'C': 10,
        'kernel': 'rbf',
        'gamma': 'scale',
        'probability': True,
        'random_state': 42
    },
    'logistic_regression': {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'liblinear',
        'random_state': 42,
        'max_iter': 2000
    },
    'naive_bayes': {
        'alpha': 0.5,
        'fit_prior': True
    }
}

# Neural model configurations
NEURAL_CONFIGS = {
    'roberta_social': {
        'model_path': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
        'learning_rate': 2e-5,
        'batch_size': 16,
        'epochs': 5,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'max_length': 128,
        'dropout': 0.2
    },
    'distilbert_sentiment': {
        'model_path': 'distilbert-base-uncased-finetuned-sst-2-english',
        'learning_rate': 5e-5,
        'batch_size': 16,
        'epochs': 6,
        'warmup_ratio': 0.1,
        'weight_decay': 0.005,
        'max_length': 128,
        'dropout': 0.3
    }
}


class SentimentDataset(Dataset):
    """Dataset for neural model fine-tuning"""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and torch are required for neural models")

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        import torch  # Import here in case it's available

        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class ExtendedMLAnalyzer:
    def __init__(self):
        # Initialize all models
        self.traditional_models = {
            'random_forest': RandomForestClassifier(**OPTIMIZED_CONFIGS['random_forest']),
            'gradient_boosting': GradientBoostingClassifier(**OPTIMIZED_CONFIGS['gradient_boosting']),
            'svm': SVC(**OPTIMIZED_CONFIGS['svm']),
            'logistic_regression': LogisticRegression(**OPTIMIZED_CONFIGS['logistic_regression']),
            'naive_bayes': MultinomialNB(**OPTIMIZED_CONFIGS['naive_bayes'])
        }

        self.neural_models = {
            'roberta_social': NEURAL_CONFIGS['roberta_social']['model_path'],
            'distilbert_sentiment': NEURAL_CONFIGS['distilbert_sentiment']['model_path']
        }

        # Initialize components
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.label_encoder = LabelEncoder()

        # Storage for results
        self.trained_models = {}
        self.neural_pipelines = {}
        self.fine_tuned_models = {}
        self.all_results = {}

        # Device setup
        self.device = 'cpu'  # Default fallback
        if TRANSFORMERS_AVAILABLE:
            try:
                import torch
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print(f"🔧 Device: {self.device}")
                if torch.cuda.is_available():
                    print(f"🔧 GPU: {torch.cuda.get_device_name(0)}")
            except Exception as e:
                print(f"⚠️ Device setup failed: {e}")
                self.device = 'cpu'
        else:
            print(f"🔧 Device: {self.device} (neural models disabled)")

    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        if not text or pd.isna(text):
            return ""

        text = str(text).lower()
        # Keep punctuation that might be meaningful for sentiment
        text = re.sub(r'[^\w\s!?.,@#$%]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def calculate_comprehensive_metrics(self, y_true, y_pred, y_prob=None, model_name=""):
        """Calculate comprehensive metrics for a model"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

        # Add AUC if probabilities available and we have binary/multiclass
        try:
            if y_prob is not None and len(np.unique(y_true)) > 1:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    # Multiclass
                    metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
        except Exception as e:
            print(f"⚠️ Could not calculate AUC for {model_name}: {e}")
            metrics['auc'] = None

        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

        # Per-class metrics
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics['per_class_metrics'] = class_report

        return metrics

    def train_traditional_models(self, df):
        """Train all traditional ML models with comprehensive evaluation"""
        print("🤖 Training traditional ML models with optimized hyperparameters...")

        # Get manually tagged data
        tagged_data = df[df['manual_sentiment'].notna() & (df['manual_sentiment'] != '')].copy()
        print(f"📊 Training on {len(tagged_data)} manually tagged samples")

        if len(tagged_data) < 20:
            raise ValueError("Need at least 20 manually tagged samples for training!")

        # Preprocess texts
        texts = [self.preprocess_text(text) for text in tagged_data['text']]
        labels = tagged_data['manual_sentiment'].tolist()

        # Create features
        X = self.vectorizer.fit_transform(texts)
        y = self.label_encoder.fit_transform(labels)

        # Stratified split to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"📈 Training set: {X_train.shape[0]} samples")
        print(f"📈 Test set: {X_test.shape[0]} samples")

        results = {}

        # Train each traditional model
        for name, model in self.traditional_models.items():
            print(f"\n🔧 Training {name}...")

            try:
                # Train the model
                model.fit(X_train, y_train)
                self.trained_models[name] = model

                # Predictions on test set
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

                # Calculate comprehensive metrics
                metrics = self.calculate_comprehensive_metrics(
                    y_test, y_pred, y_prob, name
                )

                # Cross-validation for more robust evaluation
                cv_scores = cross_val_score(
                    model, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                    scoring='f1_weighted'
                )
                metrics['cv_f1_mean'] = cv_scores.mean()
                metrics['cv_f1_std'] = cv_scores.std()

                results[name] = metrics

                print(f"    ✅ Accuracy: {metrics['accuracy']:.3f}")
                print(f"    ✅ F1 (weighted): {metrics['f1_weighted']:.3f}")
                print(f"    ✅ F1 (macro): {metrics['f1_macro']:.3f}")
                print(f"    ✅ CV F1: {metrics['cv_f1_mean']:.3f} ± {metrics['cv_f1_std']:.3f}")

            except Exception as e:
                print(f"    ❌ Failed to train {name}: {e}")
                continue

        return results

    def compute_neural_metrics(self, eval_pred):
        """Compute metrics for neural model evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1_weighted': f1_score(labels, predictions, average='weighted', zero_division=0),
            'f1_macro': f1_score(labels, predictions, average='macro', zero_division=0),
            'precision_weighted': precision_score(labels, predictions, average='weighted', zero_division=0),
            'recall_weighted': recall_score(labels, predictions, average='weighted', zero_division=0)
        }

    def fine_tune_neural_model(self, model_name, tagged_data):
        """Fine-tune a neural model with comprehensive evaluation"""
        if not TRANSFORMERS_AVAILABLE:
            return None

        print(f"🧠 Fine-tuning {model_name}...")

        try:
            config = NEURAL_CONFIGS[model_name]
            model_path = config['model_path']

            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=len(self.label_encoder.classes_),
                ignore_mismatched_sizes=True
            )

            # Prepare data
            texts = [str(text) for text in tagged_data['text']]
            labels = self.label_encoder.transform(tagged_data['manual_sentiment'])

            # Split data
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=0.2, random_state=42, stratify=labels
            )

            # Create datasets
            train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, config['max_length'])
            val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, config['max_length'])

            # Training arguments - Updated for newer transformers versions
            training_args = TrainingArguments(
                output_dir=f'./results_{model_name}',
                num_train_epochs=config['epochs'],
                per_device_train_batch_size=config['batch_size'],
                per_device_eval_batch_size=config['batch_size'],
                learning_rate=config['learning_rate'],
                weight_decay=config['weight_decay'],
                warmup_ratio=config['warmup_ratio'],
                eval_strategy='steps',  # Changed from evaluation_strategy
                eval_steps=50,
                save_strategy='steps',
                save_steps=50,
                logging_steps=25,
                load_best_model_at_end=True,
                metric_for_best_model='f1_weighted',
                greater_is_better=True,
                save_total_limit=2,
                fp16=torch.cuda.is_available(),
                dataloader_pin_memory=False,
                report_to=[]
            )

            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=self.compute_neural_metrics,
                data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )

            # Train
            trainer.train()

            # Final evaluation
            eval_results = trainer.evaluate()

            # Test set evaluation for comprehensive metrics
            test_predictions = trainer.predict(val_dataset)
            y_pred = np.argmax(test_predictions.predictions, axis=1)
            y_prob = torch.softmax(torch.tensor(test_predictions.predictions), dim=1).numpy()

            comprehensive_metrics = self.calculate_comprehensive_metrics(
                val_labels, y_pred, y_prob, model_name
            )

            # Combine trainer metrics with comprehensive metrics
            final_metrics = {**eval_results, **comprehensive_metrics}

            # Store fine-tuned model
            self.fine_tuned_models[model_name] = {
                'model': trainer.model,
                'tokenizer': tokenizer,
                'metrics': final_metrics
            }

            print(f"    ✅ Accuracy: {final_metrics['accuracy']:.3f}")
            print(f"    ✅ F1 (weighted): {final_metrics['f1_weighted']:.3f}")
            print(f"    ✅ F1 (macro): {final_metrics['f1_macro']:.3f}")

            return final_metrics

        except Exception as e:
            print(f"    ❌ Failed to fine-tune {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def train_neural_models(self, df):
        """Train/fine-tune neural models"""
        if not TRANSFORMERS_AVAILABLE:
            print("⚠️ Transformers not available, loading pre-trained models...")
            return self.load_pretrained_neural_models()

        # Get tagged data
        tagged_data = df[df['manual_sentiment'].notna() & (df['manual_sentiment'] != '')].copy()

        if len(tagged_data) < 50:
            print(f"⚠️ Insufficient data for fine-tuning ({len(tagged_data)} samples, need ≥50)")
            print("⚠️ Loading pre-trained models instead...")
            return self.load_pretrained_neural_models()

        print("🧠 Fine-tuning neural models...")

        results = {}
        for model_name in self.neural_models.keys():
            try:
                metrics = self.fine_tune_neural_model(model_name, tagged_data)
                if metrics:
                    results[model_name] = metrics
            except Exception as e:
                print(f"❌ Fine-tuning {model_name} failed: {e}")
                continue

        # If fine-tuning failed, fall back to pre-trained
        if not results:
            print("⚠️ Fine-tuning failed, falling back to pre-trained models...")
            return self.load_pretrained_neural_models()

        return results

    def load_pretrained_neural_models(self):
        """Load pre-trained neural models for inference"""
        if not TRANSFORMERS_AVAILABLE:
            print("❌ Cannot load neural models: transformers not available")
            return {}

        print("🧠 Loading pre-trained neural models...")

        results = {}
        for model_name, model_path in self.neural_models.items():
            try:
                print(f"  Loading {model_name} from {model_path}...")

                # Import here to catch any remaining import issues
                from transformers import pipeline

                pipeline_model = pipeline(
                    "sentiment-analysis",
                    model=model_path,
                    tokenizer=model_path,
                    return_all_scores=True,
                    device=0 if (hasattr(self, 'device') and 'cuda' in str(self.device)) else -1
                )

                self.neural_pipelines[model_name] = pipeline_model
                print(f"    ✅ {model_name} loaded successfully")

                # We'll evaluate these later when we have predictions
                results[model_name] = {'loaded': True}

            except Exception as e:
                print(f"    ❌ Failed to load {model_name}: {e}")
                continue

        if not results:
            print("❌ No neural models could be loaded")

        return results

    def predict_traditional_models(self, df):
        """Generate predictions from traditional models"""
        print("🔮 Generating traditional model predictions...")

        # Preprocess all texts
        all_texts = [self.preprocess_text(text) for text in df['text']]
        X_all = self.vectorizer.transform(all_texts)

        for model_name, model in self.trained_models.items():
            try:
                print(f"  Predicting with {model_name}...")

                # Predictions
                predictions = model.predict(X_all)
                probabilities = model.predict_proba(X_all)
                confidences = np.max(probabilities, axis=1)

                # Convert predictions back to labels
                predicted_labels = self.label_encoder.inverse_transform(predictions)

                # Add to dataframe
                df[f'{model_name}_sentiment'] = predicted_labels
                df[f'{model_name}_confidence'] = confidences
                df[f'{model_name}_model_type'] = 'traditional'

                # Add probability distributions
                for i, class_name in enumerate(self.label_encoder.classes_):
                    df[f'{model_name}_prob_{class_name}'] = probabilities[:, i]

                print(f"    ✅ {model_name} predictions complete")

            except Exception as e:
                print(f"    ❌ {model_name} failed: {e}")
                continue

        return df

    def predict_neural_models(self, df):
        """Generate predictions from neural models"""
        print("🧠 Generating neural model predictions...")

        texts = [str(text) for text in df['text']]

        # Fine-tuned models
        for model_name, model_data in self.fine_tuned_models.items():
            try:
                print(f"  Predicting with fine-tuned {model_name}...")

                model = model_data['model']
                tokenizer = model_data['tokenizer']
                model.eval()

                predictions = []
                confidences = []
                prob_distributions = []

                # Process in batches
                batch_size = 16
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]

                    inputs = tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=128,
                        return_tensors='pt'
                    )

                    inputs = {k: v.to(model.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = model(**inputs)
                        probs = torch.softmax(outputs.logits, dim=-1)

                        batch_predictions = torch.argmax(probs, dim=-1).cpu().numpy()
                        batch_confidences = torch.max(probs, dim=-1)[0].cpu().numpy()
                        batch_probs = probs.cpu().numpy()

                        predictions.extend(batch_predictions)
                        confidences.extend(batch_confidences)
                        prob_distributions.extend(batch_probs)

                # Map predictions to sentiment labels
                predicted_sentiments = self.label_encoder.inverse_transform(predictions)

                df[f'{model_name}_sentiment'] = predicted_sentiments
                df[f'{model_name}_confidence'] = confidences
                df[f'{model_name}_model_type'] = 'neural_finetuned'

                # Add probability distributions
                prob_array = np.array(prob_distributions)
                for i, class_name in enumerate(self.label_encoder.classes_):
                    df[f'{model_name}_prob_{class_name}'] = prob_array[:, i]

                print(f"    ✅ {model_name} predictions complete")

            except Exception as e:
                print(f"    ❌ {model_name} failed: {e}")
                continue

        # Pre-trained models
        for model_name, pipeline_model in self.neural_pipelines.items():
            try:
                print(f"  Predicting with pre-trained {model_name}...")

                # Process in batches
                batch_size = 32
                all_predictions = []

                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_predictions = pipeline_model(batch_texts)
                    all_predictions.extend(batch_predictions)

                # Map to standard sentiment format
                mapped_results = self.map_neural_predictions(model_name, all_predictions)

                df[f'{model_name}_sentiment'] = [r['sentiment'] for r in mapped_results]
                df[f'{model_name}_confidence'] = [r['confidence'] for r in mapped_results]
                df[f'{model_name}_model_type'] = 'neural_pretrained'

                print(f"    ✅ {model_name} predictions complete")

            except Exception as e:
                print(f"    ❌ {model_name} failed: {e}")
                continue

        return df

    def map_neural_predictions(self, model_name, predictions):
        """Map neural model outputs to standard sentiment format"""
        mapped_results = []

        label_mappings = {
            'roberta_social': {'LABEL_0': 'negative', 'LABEL_1': 'neutral', 'LABEL_2': 'positive'},
            'distilbert_sentiment': {'LABEL_0': 'negative', 'LABEL_1': 'positive'}  # Updated mapping
        }

        for pred_list in predictions:
            if model_name == 'distilbert_sentiment' and len(pred_list) == 2:
                # Handle binary output by adding neutral threshold
                scores = {p['label']: p['score'] for p in pred_list}
                neg_score = scores.get('LABEL_0', 0)  # Negative
                pos_score = scores.get('LABEL_1', 0)  # Positive

                if abs(pos_score - neg_score) < 0.15:  # Close scores = neutral
                    sentiment = 'neutral'
                    confidence = 0.5
                elif pos_score > neg_score:
                    sentiment = 'positive'
                    confidence = pos_score
                else:
                    sentiment = 'negative'
                    confidence = neg_score
            else:
                # Standard mapping
                best_pred = max(pred_list, key=lambda x: x['score'])
                raw_label = best_pred['label']

                mapping = label_mappings.get(model_name, {})
                sentiment = mapping.get(raw_label, raw_label.lower())

                if sentiment not in ['positive', 'negative', 'neutral']:
                    sentiment = 'neutral'

                confidence = best_pred['score']

            mapped_results.append({
                'sentiment': sentiment,
                'confidence': confidence
            })

        return mapped_results

    def evaluate_pretrained_neural_models(self, df):
        """Evaluate pre-trained neural models against manual tags"""
        results = {}

        tagged_data = df[df['manual_sentiment'].notna() & (df['manual_sentiment'] != '')].copy()
        if len(tagged_data) == 0:
            return results

        print("📊 Evaluating pre-trained neural models...")

        for model_name in self.neural_pipelines.keys():
            if f'{model_name}_sentiment' not in tagged_data.columns:
                continue

            try:
                y_true = tagged_data['manual_sentiment']
                y_pred = tagged_data[f'{model_name}_sentiment']

                # Calculate comprehensive metrics
                metrics = self.calculate_comprehensive_metrics(
                    y_true, y_pred, model_name=model_name
                )

                results[model_name] = metrics

                print(f"  {model_name}:")
                print(f"    Accuracy: {metrics['accuracy']:.3f}")
                print(f"    F1 (weighted): {metrics['f1_weighted']:.3f}")
                print(f"    F1 (macro): {metrics['f1_macro']:.3f}")

            except Exception as e:
                print(f"    ❌ {model_name} evaluation failed: {e}")
                continue

        return results

    def generate_individual_analyses(self, df, timestamp):
        """Generate comprehensive individual analysis files for each model"""
        print("\n💾 Generating individual analysis files...")

        saved_files = []

        # Get all model names from dataframe columns
        model_names = set()
        for col in df.columns:
            if col.endswith('_sentiment'):
                model_name = col.replace('_sentiment', '')
                model_names.add(model_name)

        for model_name in model_names:
            try:
                print(f"  Creating analysis for {model_name}...")

                # Base columns
                base_cols = ['id', 'text', 'username', 'timestamp']
                if 'manual_sentiment' in df.columns:
                    base_cols.append('manual_sentiment')

                # Model-specific columns
                model_cols = [
                    f'{model_name}_sentiment',
                    f'{model_name}_confidence',
                    f'{model_name}_model_type'
                ]

                # Probability columns
                prob_cols = [col for col in df.columns if col.startswith(f'{model_name}_prob_')]

                # Combine all columns
                all_cols = base_cols + model_cols + prob_cols
                available_cols = [col for col in all_cols if col in df.columns]

                # Create individual dataframe
                individual_df = df[available_cols].copy()

                # Rename for consistency
                individual_df = individual_df.rename(columns={
                    f'{model_name}_sentiment': 'ml_sentiment',
                    f'{model_name}_confidence': 'ml_confidence',
                    f'{model_name}_model_type': 'model_type'
                })

                # Rename probability columns
                for col in prob_cols:
                    if col in individual_df.columns:
                        new_col = col.replace(f'{model_name}_prob_', 'ml_prob_')
                        individual_df = individual_df.rename(columns={col: new_col})

                # Add metadata
                individual_df['model_name'] = model_name
                individual_df['algorithm_name'] = model_name
                individual_df['timestamp'] = datetime.now().isoformat()

                # Add performance metrics if available
                if model_name in self.all_results:
                    metrics = self.all_results[model_name]
                    individual_df['test_accuracy'] = metrics.get('accuracy', None)
                    individual_df['test_f1_weighted'] = metrics.get('f1_weighted', None)
                    individual_df['test_f1_macro'] = metrics.get('f1_macro', None)
                    individual_df['test_precision_weighted'] = metrics.get('precision_weighted', None)
                    individual_df['test_recall_weighted'] = metrics.get('recall_weighted', None)

                    if 'cv_f1_mean' in metrics:
                        individual_df['cv_f1_mean'] = metrics['cv_f1_mean']
                        individual_df['cv_f1_std'] = metrics['cv_f1_std']

                # Calculate prediction distribution
                if 'ml_sentiment' in individual_df.columns:
                    pred_dist = individual_df['ml_sentiment'].value_counts()
                    total_preds = len(individual_df)

                    for sentiment in ['positive', 'negative', 'neutral']:
                        count = pred_dist.get(sentiment, 0)
                        pct = (count / total_preds) * 100 if total_preds > 0 else 0
                        individual_df[f'prediction_dist_{sentiment}_count'] = count
                        individual_df[f'prediction_dist_{sentiment}_pct'] = pct

                # Save individual file
                filename = f"ml_individual_{timestamp}_{model_name}.csv"
                individual_df.to_csv(filename, index=False)
                saved_files.append(filename)

                print(f"    ✅ Saved: {filename}")

                # Generate summary statistics file
                summary_stats = self.generate_model_summary(individual_df, model_name)
                summary_file = f"ml_summary_{timestamp}_{model_name}.json"

                with open(summary_file, 'w') as f:
                    json.dump(summary_stats, f, indent=2, default=str)

                print(f"    ✅ Summary: {summary_file}")

            except Exception as e:
                print(f"    ❌ Failed to create analysis for {model_name}: {e}")
                continue

        return saved_files

    def generate_model_summary(self, df, model_name):
        """Generate comprehensive summary statistics for a model"""
        summary = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'total_predictions': len(df),
            'data_quality': {},
            'prediction_analysis': {},
            'performance_metrics': {}
        }

        # Data quality metrics
        summary['data_quality'] = {
            'total_samples': len(df),
            'samples_with_text': len(df[df['text'].notna() & (df['text'] != '')]),
            'samples_with_manual_tags': len(df[df['manual_sentiment'].notna() & (
                        df['manual_sentiment'] != '')]) if 'manual_sentiment' in df.columns else 0,
            'average_text_length': df['text'].str.len().mean() if 'text' in df.columns else None,
            'missing_predictions': len(df[df['ml_sentiment'].isna()]) if 'ml_sentiment' in df.columns else 0
        }

        # Prediction analysis
        if 'ml_sentiment' in df.columns:
            pred_dist = df['ml_sentiment'].value_counts().to_dict()
            total = len(df)

            summary['prediction_analysis'] = {
                'sentiment_distribution': pred_dist,
                'sentiment_percentages': {k: (v / total) * 100 for k, v in pred_dist.items()},
                'confidence_stats': {
                    'mean': df['ml_confidence'].mean() if 'ml_confidence' in df.columns else None,
                    'std': df['ml_confidence'].std() if 'ml_confidence' in df.columns else None,
                    'min': df['ml_confidence'].min() if 'ml_confidence' in df.columns else None,
                    'max': df['ml_confidence'].max() if 'ml_confidence' in df.columns else None,
                    'high_confidence_count': len(
                        df[df['ml_confidence'] > 0.8]) if 'ml_confidence' in df.columns else None
                }
            }

        # Performance metrics from stored results
        if model_name in self.all_results:
            metrics = self.all_results[model_name]
            summary['performance_metrics'] = {
                'accuracy': metrics.get('accuracy'),
                'f1_weighted': metrics.get('f1_weighted'),
                'f1_macro': metrics.get('f1_macro'),
                'precision_weighted': metrics.get('precision_weighted'),
                'recall_weighted': metrics.get('recall_weighted'),
                'cross_validation': {
                    'cv_f1_mean': metrics.get('cv_f1_mean'),
                    'cv_f1_std': metrics.get('cv_f1_std')
                } if 'cv_f1_mean' in metrics else None,
                'per_class_metrics': metrics.get('per_class_metrics'),
                'confusion_matrix': metrics.get('confusion_matrix')
            }

        return summary

    def save_master_report(self, timestamp):
        """Save comprehensive master report with all model comparisons"""
        print("\n📋 Generating master performance report...")

        master_report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'system_info': {
                'device': str(self.device) if hasattr(self, 'device') else 'cpu',
                'transformers_available': TRANSFORMERS_AVAILABLE,
                'torch_available': TRANSFORMERS_AVAILABLE and 'torch' in sys.modules
            },
            'model_configurations': {
                'traditional_models': OPTIMIZED_CONFIGS,
                'neural_models': NEURAL_CONFIGS
            },
            'performance_summary': {},
            'model_rankings': {},
            'recommendations': {}
        }

        # Organize results by model type
        traditional_results = {}
        neural_results = {}

        for model_name, metrics in self.all_results.items():
            if model_name in ['random_forest', 'gradient_boosting', 'svm', 'logistic_regression', 'naive_bayes']:
                traditional_results[model_name] = metrics
            else:
                neural_results[model_name] = metrics

        master_report['performance_summary'] = {
            'traditional_models': traditional_results,
            'neural_models': neural_results,
            'total_models_evaluated': len(self.all_results)
        }

        # Model rankings
        if self.all_results:
            # Sort by different metrics
            rankings = {
                'by_f1_weighted': sorted(self.all_results.keys(),
                                         key=lambda x: self.all_results[x].get('f1_weighted', 0),
                                         reverse=True),
                'by_accuracy': sorted(self.all_results.keys(),
                                      key=lambda x: self.all_results[x].get('accuracy', 0),
                                      reverse=True),
                'by_f1_macro': sorted(self.all_results.keys(),
                                      key=lambda x: self.all_results[x].get('f1_macro', 0),
                                      reverse=True)
            }

            master_report['model_rankings'] = rankings

            # Generate recommendations
            best_overall = rankings['by_f1_weighted'][0] if rankings['by_f1_weighted'] else None
            best_traditional = None
            best_neural = None

            for model in rankings['by_f1_weighted']:
                if model in traditional_results and best_traditional is None:
                    best_traditional = model
                elif model in neural_results and best_neural is None:
                    best_neural = model
                if best_traditional and best_neural:
                    break

            recommendations = []
            if best_overall:
                f1_score = self.all_results[best_overall].get('f1_weighted', 0)
                recommendations.append(f"Best overall model: {best_overall} (F1: {f1_score:.3f})")

            if best_traditional:
                f1_score = self.all_results[best_traditional].get('f1_weighted', 0)
                recommendations.append(f"Best traditional model: {best_traditional} (F1: {f1_score:.3f})")

            if best_neural:
                f1_score = self.all_results[best_neural].get('f1_weighted', 0)
                recommendations.append(f"Best neural model: {best_neural} (F1: {f1_score:.3f})")

            # Performance thresholds
            excellent_models = [k for k, v in self.all_results.items() if v.get('f1_weighted', 0) > 0.85]
            good_models = [k for k, v in self.all_results.items() if 0.75 < v.get('f1_weighted', 0) <= 0.85]

            if excellent_models:
                recommendations.append(f"Models with excellent performance (F1>0.85): {excellent_models}")
            if good_models:
                recommendations.append(f"Models with good performance (0.75<F1<=0.85): {good_models}")

            master_report['recommendations'] = recommendations

        # Save master report
        master_file = f"ml_master_report_{timestamp}.json"
        with open(master_file, 'w') as f:
            json.dump(master_report, f, indent=2, default=str)

        print(f"✅ Master report saved: {master_file}")
        return master_file


def check_transformers_installation():
    """Check if transformers and torch are properly installed"""
    print("\n🔍 Detailed Dependency Check:")

    try:
        import sys
        print(f"Python version: {sys.version}")
        print(f"Python executable: {sys.executable}")
    except Exception as e:
        print(f"Could not get Python info: {e}")

    all_good = True

    try:
        import torch
        print(f"✅ torch: {torch.__version__}")
        print(f"   - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   - CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ torch import failed: {e}")
        all_good = False

    try:
        import transformers
        print(f"✅ transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ transformers import failed: {e}")
        all_good = False

    try:
        from transformers import pipeline
        print(f"✅ transformers.pipeline: available")

        # Test a simple pipeline creation
        test_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                                 device=-1)
        print(f"✅ Test pipeline creation: successful")

        # Test a simple prediction
        result = test_pipeline("This is a test")
        print(f"✅ Test prediction: {result}")

    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        all_good = False

    if not all_good:
        print(f"\n🔧 Installation Commands:")
        print(f"# Install PyTorch (CPU version)")
        print(f"pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print(f"")
        print(f"# Install Transformers and dependencies")
        print(f"pip install transformers datasets accelerate")
        print(f"")
        print(f"# Or install everything at once")
        print(f"pip install torch transformers datasets accelerate --index-url https://download.pytorch.org/whl/cpu")

    return all_good


def main():
    print("=== EXTENDED ML SENTIMENT ANALYZER ===")
    print("Models: RoBERTa, DistilBERT, SVM, Naive Bayes, Random Forest, Gradient Boosting, Logistic Regression")
    print("=" * 80)

    # Check dependencies
    transformers_ok = check_transformers_installation()
    if not transformers_ok:
        print("\n⚠️ Neural models will be skipped. Only traditional ML models will run.")
        print("Continue anyway? (y/n): ", end="")
        response = input().lower().strip()
        if response != 'y' and response != 'yes':
            print("Exiting. Install dependencies and try again.")
            return

    if len(sys.argv) < 2:
        print("Usage: python extended_ml_analyzer.py <csv_file> [options]")
        print("\nOptions:")
        print("  --traditional-only    Run only traditional ML models")
        print("  --neural-only         Run only neural models")
        print("  --no-fine-tuning      Use pre-trained neural models only")
        return

    csv_file = sys.argv[1]
    traditional_only = '--traditional-only' in sys.argv
    neural_only = '--neural-only' in sys.argv
    no_fine_tuning = '--no-fine-tuning' in sys.argv

    if not os.path.exists(csv_file):
        print(f"❌ Error: File '{csv_file}' not found")
        return

    try:
        # Load data
        print(f"📁 Loading: {csv_file}")
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")

        # Validate required columns
        if 'text' not in df.columns:
            print(f"❌ Missing 'text' column")
            return

        # Check manual tags
        has_manual_tags = False
        manual_count = 0

        if 'manual_sentiment' in df.columns:
            manual_data = df[df['manual_sentiment'].notna() & (df['manual_sentiment'] != '')]
            manual_count = len(manual_data)
            has_manual_tags = manual_count > 0

            if has_manual_tags:
                print(f"✅ Found {manual_count} manual sentiment tags")
                sentiment_dist = manual_data['manual_sentiment'].value_counts()
                print("📊 Distribution:")
                for sentiment, count in sentiment_dist.items():
                    pct = (count / manual_count) * 100
                    print(f"  {sentiment}: {count} ({pct:.1f}%)")
            else:
                print("⚠️ No manual sentiment tags found")
        else:
            print("⚠️ No 'manual_sentiment' column found")
            df['manual_sentiment'] = None

        # Initialize analyzer
        analyzer = ExtendedMLAnalyzer()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Train models
        if not neural_only and has_manual_tags and manual_count >= 20:
            print(f"\n🤖 Training Traditional Models...")
            traditional_results = analyzer.train_traditional_models(df)
            analyzer.all_results.update(traditional_results)
        elif not neural_only:
            print("⚠️ Insufficient manual tags for traditional model training (need ≥20)")

        if not traditional_only:
            print(f"\n🧠 Processing Neural Models...")
            if no_fine_tuning or manual_count < 50:
                neural_results = analyzer.train_neural_models(df)
            else:
                neural_results = analyzer.train_neural_models(df)
            analyzer.all_results.update(neural_results)

        # Generate predictions
        print(f"\n🔮 Generating Predictions...")
        if analyzer.trained_models:
            df = analyzer.predict_traditional_models(df)

        if analyzer.fine_tuned_models or analyzer.neural_pipelines:
            df = analyzer.predict_neural_models(df)

        # Evaluate pre-trained neural models if we have manual tags
        if analyzer.neural_pipelines and has_manual_tags:
            pretrained_results = analyzer.evaluate_pretrained_neural_models(df)
            analyzer.all_results.update(pretrained_results)

        # Save main analysis file
        main_output = f"ml_extended_analysis_{timestamp}.csv"
        df.to_csv(main_output, index=False)
        print(f"\n✅ Main analysis saved: {main_output}")

        # Generate individual analyses
        individual_files = analyzer.generate_individual_analyses(df, timestamp)

        # Generate master report
        master_report = analyzer.save_master_report(timestamp)

        # Display results summary
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"📊 Main file: {main_output}")
        print(f"📁 Individual files: {len(individual_files)} model-specific analyses")
        print(f"📋 Master report: {master_report}")

        if analyzer.all_results:
            print(f"\n🏆 MODEL PERFORMANCE SUMMARY")
            print("=" * 80)
            print(f"{'Model':<25} {'Type':<15} {'Accuracy':<9} {'F1-W':<9} {'F1-M':<9} {'Prec-W':<9} {'Rec-W':<9}")
            print("-" * 80)

            # Sort by F1-weighted score
            sorted_models = sorted(analyzer.all_results.items(),
                                   key=lambda x: x[1].get('f1_weighted', 0),
                                   reverse=True)

            for model_name, metrics in sorted_models:
                model_type = 'Traditional' if model_name in analyzer.traditional_models else 'Neural'
                if model_name in analyzer.fine_tuned_models:
                    model_type += ' (Tuned)'
                elif model_name in analyzer.neural_pipelines:
                    model_type += ' (Pre-trained)'

                print(f"{model_name:<25} {model_type:<15} "
                      f"{metrics.get('accuracy', 0):<9.3f} "
                      f"{metrics.get('f1_weighted', 0):<9.3f} "
                      f"{metrics.get('f1_macro', 0):<9.3f} "
                      f"{metrics.get('precision_weighted', 0):<9.3f} "
                      f"{metrics.get('recall_weighted', 0):<9.3f}")

            # Best model recommendation
            best_model_name, best_metrics = sorted_models[0]
            print(f"\n🏆 BEST MODEL: {best_model_name}")
            print(f"   F1-Weighted: {best_metrics.get('f1_weighted', 0):.3f}")
            print(f"   Accuracy: {best_metrics.get('accuracy', 0):.3f}")

            if 'cv_f1_mean' in best_metrics:
                print(f"   Cross-Val F1: {best_metrics['cv_f1_mean']:.3f} ± {best_metrics['cv_f1_std']:.3f}")

        print(f"\n💡 Next Steps:")
        print(f"1. Review individual model files for detailed analysis")
        print(f"2. Load CSV files into your dashboard for visualization")
        print(f"3. Check master report for comprehensive comparison")
        print(f"4. Use the best performing model for production!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()