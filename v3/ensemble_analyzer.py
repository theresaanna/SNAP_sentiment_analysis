#!/usr/bin/env python3

"""
Ensemble Sentiment Analyzer
Combines RoBERTa Social and Gradient Boosting predictions for improved accuracy.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')


class EnsembleAnalyzer:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.ensemble_methods = {
            'majority_vote': self.majority_vote_ensemble,
            'weighted_average': self.weighted_average_ensemble,
            'confidence_weighted': self.confidence_weighted_ensemble,
            'stacked_ensemble': self.stacked_ensemble,
            'adaptive_ensemble': self.adaptive_ensemble
        }

    def load_predictions(self, roberta_file, gradient_boosting_file):
        """Load predictions from both models"""
        print("📊 Loading model predictions...")

        # Load RoBERTa predictions
        roberta_df = pd.read_csv(roberta_file)
        print(f"✅ RoBERTa data: {len(roberta_df)} rows")

        # Load Gradient Boosting predictions
        gb_df = pd.read_csv(gradient_boosting_file)
        print(f"✅ Gradient Boosting data: {len(gb_df)} rows")

        # Merge on common identifier (assuming 'id' or 'text' column)
        if 'id' in roberta_df.columns and 'id' in gb_df.columns:
            merged_df = pd.merge(roberta_df, gb_df, on='id', suffixes=('_roberta', '_gb'))
        elif 'text' in roberta_df.columns and 'text' in gb_df.columns:
            merged_df = pd.merge(roberta_df, gb_df, on='text', suffixes=('_roberta', '_gb'))
        else:
            print("⚠️  No common identifier found. Assuming same order...")
            # Ensure same length
            min_len = min(len(roberta_df), len(gb_df))
            roberta_df = roberta_df.iloc[:min_len].reset_index(drop=True)
            gb_df = gb_df.iloc[:min_len].reset_index(drop=True)

            # Add suffixes manually
            for col in roberta_df.columns:
                if col in gb_df.columns and col != 'text':
                    roberta_df = roberta_df.rename(columns={col: f"{col}_roberta"})
                    gb_df = gb_df.rename(columns={col: f"{col}_gb"})

            merged_df = pd.concat([roberta_df, gb_df], axis=1)

        print(f"🔗 Merged dataset: {len(merged_df)} rows")
        return merged_df

    def majority_vote_ensemble(self, roberta_pred, gb_pred, roberta_conf=None, gb_conf=None):
        """Simple majority vote between two models"""
        ensemble_pred = []

        for r_pred, g_pred in zip(roberta_pred, gb_pred):
            if r_pred == g_pred:
                # Agreement - use the prediction
                ensemble_pred.append(r_pred)
            else:
                # Disagreement - use confidence if available, otherwise RoBERTa (higher overall F1)
                if roberta_conf is not None and gb_conf is not None:
                    r_conf_val = roberta_conf[len(ensemble_pred)]
                    g_conf_val = gb_conf[len(ensemble_pred)]
                    ensemble_pred.append(r_pred if r_conf_val > g_conf_val else g_pred)
                else:
                    # Default to RoBERTa (2nd best model)
                    ensemble_pred.append(r_pred)

        return ensemble_pred

    def weighted_average_ensemble(self, roberta_pred, gb_pred, roberta_conf, gb_conf):
        """Weighted ensemble based on model performance"""
        # Weights based on F1-weighted scores from your results
        roberta_weight = 0.784  # RoBERTa F1-weighted
        gb_weight = 0.801  # Gradient Boosting F1-weighted

        # Normalize weights
        total_weight = roberta_weight + gb_weight
        roberta_weight = roberta_weight / total_weight
        gb_weight = gb_weight / total_weight

        ensemble_pred = []
        ensemble_conf = []

        for r_pred, g_pred, r_conf, g_conf in zip(roberta_pred, gb_pred, roberta_conf, gb_conf):
            # Weight the confidence scores
            weighted_r_conf = r_conf * roberta_weight
            weighted_g_conf = g_conf * gb_weight

            # Choose prediction with higher weighted confidence
            if weighted_r_conf > weighted_g_conf:
                ensemble_pred.append(r_pred)
                ensemble_conf.append(weighted_r_conf)
            else:
                ensemble_pred.append(g_pred)
                ensemble_conf.append(weighted_g_conf)

        return ensemble_pred, ensemble_conf

    def confidence_weighted_ensemble(self, roberta_pred, gb_pred, roberta_conf, gb_conf):
        """Ensemble based purely on confidence scores"""
        ensemble_pred = []
        ensemble_conf = []

        for r_pred, g_pred, r_conf, g_conf in zip(roberta_pred, gb_pred, roberta_conf, gb_conf):
            if r_conf > g_conf:
                ensemble_pred.append(r_pred)
                ensemble_conf.append(r_conf)
            else:
                ensemble_pred.append(g_pred)
                ensemble_conf.append(g_conf)

        return ensemble_pred, ensemble_conf

    def stacked_ensemble(self, roberta_pred, gb_pred, roberta_conf, gb_conf, manual_labels=None):
        """Stacked ensemble - uses agreement patterns to make decisions"""
        ensemble_pred = []

        for r_pred, g_pred, r_conf, g_conf in zip(roberta_pred, gb_pred, roberta_conf, gb_conf):
            # Level 1: Check agreement
            if r_pred == g_pred:
                # Both models agree - high confidence prediction
                ensemble_pred.append(r_pred)
            else:
                # Models disagree - use more sophisticated logic
                conf_diff = abs(r_conf - g_conf)

                if conf_diff > 0.3:
                    # Large confidence difference - trust the more confident model
                    ensemble_pred.append(r_pred if r_conf > g_conf else g_pred)
                else:
                    # Small confidence difference - use Gradient Boosting (best overall)
                    ensemble_pred.append(g_pred)

        return ensemble_pred

    def adaptive_ensemble(self, roberta_pred, gb_pred, roberta_conf, gb_conf):
        """Adaptive ensemble that changes strategy based on confidence patterns"""
        ensemble_pred = []

        for r_pred, g_pred, r_conf, g_conf in zip(roberta_pred, gb_pred, roberta_conf, gb_conf):
            avg_conf = (r_conf + g_conf) / 2

            if avg_conf > 0.8:
                # High confidence case - use majority vote with confidence tiebreaker
                if r_pred == g_pred:
                    ensemble_pred.append(r_pred)
                else:
                    ensemble_pred.append(r_pred if r_conf > g_conf else g_pred)
            elif avg_conf > 0.5:
                # Medium confidence - prefer Gradient Boosting (best model)
                ensemble_pred.append(g_pred)
            else:
                # Low confidence - use weighted approach
                gb_weight = 0.801 / (0.784 + 0.801)
                if g_conf * gb_weight > r_conf * (1 - gb_weight):
                    ensemble_pred.append(g_pred)
                else:
                    ensemble_pred.append(r_pred)

        return ensemble_pred

    def evaluate_ensemble(self, predictions, true_labels, method_name):
        """Evaluate ensemble performance"""
        if true_labels is None or len(true_labels) == 0:
            print(f"⚠️  No ground truth labels available for {method_name}")
            return None

        # Filter out None/NaN values
        valid_indices = [i for i, (pred, true) in enumerate(zip(predictions, true_labels))
                         if pred is not None and true is not None and str(true).strip() != '']

        if len(valid_indices) == 0:
            print(f"⚠️  No valid labels for evaluation of {method_name}")
            return None

        valid_predictions = [predictions[i] for i in valid_indices]
        valid_true = [true_labels[i] for i in valid_indices]

        # Calculate metrics
        accuracy = accuracy_score(valid_true, valid_predictions)
        report = classification_report(valid_true, valid_predictions, output_dict=True, zero_division=0)

        return {
            'method': method_name,
            'accuracy': accuracy,
            'precision_macro': report['macro avg']['precision'],
            'recall_macro': report['macro avg']['recall'],
            'f1_macro': report['macro avg']['f1-score'],
            'f1_weighted': report['weighted avg']['f1-score'],
            'valid_samples': len(valid_predictions)
        }

    def run_ensemble_analysis(self, merged_df):
        """Run all ensemble methods and compare results"""
        print("\n🤖 Running Ensemble Analysis...")
        print("=" * 60)

        # Extract predictions and confidences
        roberta_pred = merged_df['ml_sentiment_roberta'].tolist()
        gb_pred = merged_df['ml_sentiment_gb'].tolist()

        # Try to get confidence scores
        roberta_conf = None
        gb_conf = None

        if 'ml_confidence_roberta' in merged_df.columns:
            roberta_conf = merged_df['ml_confidence_roberta'].tolist()
        if 'ml_confidence_gb' in merged_df.columns:
            gb_conf = merged_df['ml_confidence_gb'].tolist()

        # Get ground truth if available
        manual_labels = None
        for col in ['manual_sentiment_roberta', 'manual_sentiment_gb', 'manual_sentiment']:
            if col in merged_df.columns:
                manual_labels = merged_df[col].tolist()
                break

        # Run ensemble methods
        results = []
        ensemble_predictions = {}

        # 1. Majority Vote
        print("🗳️  Running Majority Vote Ensemble...")
        majority_pred = self.majority_vote_ensemble(roberta_pred, gb_pred, roberta_conf, gb_conf)
        ensemble_predictions['majority_vote'] = majority_pred
        if manual_labels:
            result = self.evaluate_ensemble(majority_pred, manual_labels, 'Majority Vote')
            if result:
                results.append(result)

        # 2. Weighted Average (if confidence available)
        if roberta_conf and gb_conf:
            print("⚖️  Running Weighted Average Ensemble...")
            weighted_pred, weighted_conf = self.weighted_average_ensemble(
                roberta_pred, gb_pred, roberta_conf, gb_conf)
            ensemble_predictions['weighted_average'] = weighted_pred
            if manual_labels:
                result = self.evaluate_ensemble(weighted_pred, manual_labels, 'Weighted Average')
                if result:
                    results.append(result)

        # 3. Confidence Weighted (if confidence available)
        if roberta_conf and gb_conf:
            print("🎯 Running Confidence Weighted Ensemble...")
            conf_pred, conf_conf = self.confidence_weighted_ensemble(
                roberta_pred, gb_pred, roberta_conf, gb_conf)
            ensemble_predictions['confidence_weighted'] = conf_pred
            if manual_labels:
                result = self.evaluate_ensemble(conf_pred, manual_labels, 'Confidence Weighted')
                if result:
                    results.append(result)

        # 4. Stacked Ensemble
        if roberta_conf and gb_conf:
            print("🏗️  Running Stacked Ensemble...")
            stacked_pred = self.stacked_ensemble(roberta_pred, gb_pred, roberta_conf, gb_conf, manual_labels)
            ensemble_predictions['stacked'] = stacked_pred
            if manual_labels:
                result = self.evaluate_ensemble(stacked_pred, manual_labels, 'Stacked Ensemble')
                if result:
                    results.append(result)

        # 5. Adaptive Ensemble
        if roberta_conf and gb_conf:
            print("🧠 Running Adaptive Ensemble...")
            adaptive_pred = self.adaptive_ensemble(roberta_pred, gb_pred, roberta_conf, gb_conf)
            ensemble_predictions['adaptive'] = adaptive_pred
            if manual_labels:
                result = self.evaluate_ensemble(adaptive_pred, manual_labels, 'Adaptive Ensemble')
                if result:
                    results.append(result)

        return results, ensemble_predictions, merged_df

    def save_ensemble_results(self, merged_df, ensemble_predictions, results):
        """Save ensemble results to CSV"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Add ensemble predictions to dataframe
        for method, predictions in ensemble_predictions.items():
            merged_df[f'ensemble_{method}'] = predictions

        # Save combined results
        output_file = f"ensemble_analysis_{timestamp}.csv"
        merged_df.to_csv(output_file, index=False)
        print(f"💾 Saved ensemble results: {output_file}")

        # Save performance summary
        if results:
            summary_file = f"ensemble_performance_{timestamp}.csv"
            results_df = pd.DataFrame(results)
            results_df.to_csv(summary_file, index=False)
            print(f"📈 Saved performance summary: {summary_file}")

        return output_file

    def print_results(self, results):
        """Print formatted results"""
        if not results:
            print("⚠️  No evaluation results available (missing ground truth labels)")
            return

        print("\n🏆 ENSEMBLE PERFORMANCE RESULTS")
        print("=" * 80)
        print(f"{'Method':<20} {'Accuracy':<10} {'F1-Weighted':<12} {'F1-Macro':<10} {'Samples':<8}")
        print("-" * 80)

        # Sort by F1-weighted score
        results.sort(key=lambda x: x['f1_weighted'], reverse=True)

        for result in results:
            print(f"{result['method']:<20} {result['accuracy']:<10.3f} {result['f1_weighted']:<12.3f} "
                  f"{result['f1_macro']:<10.3f} {result['valid_samples']:<8}")

        # Show best method
        best = results[0]
        print(f"\n🥇 BEST ENSEMBLE: {best['method']}")
        print(f"   F1-Weighted: {best['f1_weighted']:.3f}")
        print(f"   Accuracy: {best['accuracy']:.3f}")
        print(f"   Evaluated on {best['valid_samples']} samples")


def main():
    if len(sys.argv) < 3:
        print("Usage: python ensemble_analyzer.py roberta_predictions.csv gradient_boosting_predictions.csv")
        print("\nExample:")
        print(
            "python ensemble_analyzer.py ml_individual_20240101_120000_roberta_social.csv ml_individual_20240101_120000_gradient_boosting.csv")
        return

    roberta_file = sys.argv[1]
    gb_file = sys.argv[2]

    print("🚀 ENSEMBLE SENTIMENT ANALYZER")
    print("Combining RoBERTa Social + Gradient Boosting")
    print("=" * 60)

    # Check files exist
    for file in [roberta_file, gb_file]:
        if not os.path.exists(file):
            print(f"❌ File not found: {file}")
            return

    # Initialize analyzer
    analyzer = EnsembleAnalyzer()

    try:
        # Load predictions
        merged_df = analyzer.load_predictions(roberta_file, gb_file)

        # Run ensemble analysis
        results, ensemble_predictions, final_df = analyzer.run_ensemble_analysis(merged_df)

        # Save results
        output_file = analyzer.save_ensemble_results(final_df, ensemble_predictions, results)

        # Print results
        analyzer.print_results(results)

        print(f"\n✅ Analysis complete!")
        print(f"📁 Results saved to: {output_file}")
        print("\n💡 Next steps:")
        print("   1. Load the ensemble CSV in your dashboard")
        print("   2. Compare ensemble methods with individual models")
        print("   3. Use the best ensemble method for production")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()