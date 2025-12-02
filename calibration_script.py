"""
Trust Calibration Script for Scarlet.AI
Offline script to learn empirical confidence calibration from interaction logs.

This script:
1. Reads interactions.jsonl and accuracy_evaluations.jsonl
2. Extracts avg_distance and accuracy labels (Yes/Partially/No)
3. Trains a calibrated mapping: avg_distance → P(correct)
4. Saves calibration model for use in explainability module
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import statistics
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class TrustCalibrator:
    """Learns empirical confidence calibration from logged interactions."""
    
    def __init__(self, log_dir: str = "./evaluation_logs"):
        self.log_dir = Path(log_dir)
        self.interactions_file = self.log_dir / "interactions.jsonl"
        self.accuracy_file = self.log_dir / "accuracy_evaluations.jsonl"
        self.calibration_file = self.log_dir / "calibration_model.pkl"
        
        self.model = None
        self.scaler = None
        self.distance_bins = None
        self.bin_stats = None
    
    def load_data(self) -> List[Dict]:
        """
        Load and merge interaction and accuracy data.
        
        Returns:
            List of merged records with distance and accuracy info
        """
        # Load interactions
        interactions = []
        if self.interactions_file.exists():
            with open(self.interactions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    interactions.append(json.loads(line))
        
        # Load accuracy evaluations
        accuracy_map = {}
        if self.accuracy_file.exists():
            with open(self.accuracy_file, 'r', encoding='utf-8') as f:
                for line in f:
                    acc = json.loads(line)
                    key = (acc['question'], acc['timestamp'][:19])  # Match by question + time
                    accuracy_map[key] = acc
        
        # Merge data
        merged_data = []
        for interaction in interactions:
            # Calculate average distance from sources
            sources = interaction.get('sources', [])
            if not sources:
                continue
            
            distances = [s.get('distance', 1.0) for s in sources if s.get('distance') is not None]
            if not distances:
                continue
            
            avg_distance = statistics.mean(distances)
            
            # Try to find matching accuracy evaluation
            key = (interaction['question'], interaction['timestamp'][:19])
            accuracy_data = accuracy_map.get(key)
            
            # If we have accuracy data, add to merged dataset
            if accuracy_data:
                is_correct = accuracy_data.get('is_correct', 'Unknown')
                
                # Convert to binary: Yes=1, Partially=0.5, No=0
                if is_correct == 'Yes':
                    correctness = 1.0
                elif is_correct == 'Partially':
                    correctness = 0.5
                elif is_correct == 'No':
                    correctness = 0.0
                else:
                    continue  # Skip unknown
                
                merged_data.append({
                    'avg_distance': avg_distance,
                    'correctness': correctness,
                    'confidence_level': interaction.get('confidence_level', 'Unknown'),
                    'num_sources': interaction.get('num_sources', 0),
                    'had_warning': interaction.get('had_warning', False),
                    'trust_rating': accuracy_data.get('trust_rating'),
                    'accuracy_rating': accuracy_data.get('accuracy_rating'),
                    # New fields from advanced features
                    'claim_checking_enabled': interaction.get('claim_checking_enabled', False),
                    'policy_aware_enabled': interaction.get('policy_aware_enabled', False),
                    'topic_category': interaction.get('topic_category'),
                    'risk_level': interaction.get('risk_level'),
                    'had_policy_disclaimer': interaction.get('had_policy_disclaimer', False),
                })
        
        return merged_data
    
    def train_logistic_model(self, data: List[Dict]) -> Dict:
        """
        Train logistic regression model: avg_distance → P(correct).
        
        Args:
            data: Merged interaction + accuracy data
            
        Returns:
            Dictionary with model performance stats
        """
        if len(data) < 10:
            print(f"⚠️  Warning: Only {len(data)} samples available. Need more data for reliable calibration.")
            return None
        
        # Prepare features and labels
        X = np.array([[d['avg_distance']] for d in data])
        y = np.array([1 if d['correctness'] >= 0.5 else 0 for d in data])  # Binary: correct or not
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train logistic regression
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(X_scaled, y)
        
        # Calculate statistics
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
        
        stats = {
            'n_samples': len(data),
            'accuracy': self.model.score(X_scaled, y),
            'mean_predicted_prob': float(np.mean(y_pred_proba)),
            'mean_actual_correctness': float(np.mean([d['correctness'] for d in data])),
        }
        
        return stats
    
    def create_distance_bins(self, data: List[Dict]) -> Dict:
        """
        Create binned lookup table: distance ranges → empirical P(correct).
        
        Args:
            data: Merged interaction + accuracy data
            
        Returns:
            Dictionary with bin statistics
        """
        # Define distance bins
        self.distance_bins = [
            (0.0, 0.3, "Very High"),
            (0.3, 0.5, "High"),
            (0.5, 0.8, "Medium"),
            (0.8, 1.2, "Low"),
            (1.2, float('inf'), "Very Low")
        ]
        
        # Calculate empirical P(correct) for each bin
        self.bin_stats = {}
        for min_dist, max_dist, label in self.distance_bins:
            bin_data = [d for d in data if min_dist <= d['avg_distance'] < max_dist]
            
            if bin_data:
                correctness_scores = [d['correctness'] for d in bin_data]
                n_correct = sum(1 for d in bin_data if d['correctness'] >= 0.5)
                
                self.bin_stats[label] = {
                    'range': (min_dist, max_dist),
                    'n_samples': len(bin_data),
                    'empirical_prob': n_correct / len(bin_data),
                    'avg_correctness': statistics.mean(correctness_scores),
                    'avg_distance': statistics.mean([d['avg_distance'] for d in bin_data]),
                }
            else:
                self.bin_stats[label] = {
                    'range': (min_dist, max_dist),
                    'n_samples': 0,
                    'empirical_prob': 0.0,
                    'avg_correctness': 0.0,
                    'avg_distance': (min_dist + max_dist) / 2,
                }
        
        return self.bin_stats
    
    def predict_confidence(self, avg_distance: float) -> Tuple[str, float, Dict]:
        """
        Predict calibrated confidence for a given avg_distance.
        
        Args:
            avg_distance: Average retrieval distance
            
        Returns:
            Tuple of (confidence_label, probability, bin_info)
        """
        # Method 1: Use logistic model if available
        if self.model is not None and self.scaler is not None:
            X = np.array([[avg_distance]])
            X_scaled = self.scaler.transform(X)
            prob = self.model.predict_proba(X_scaled)[0, 1]
            
            # Map probability to confidence level
            if prob >= 0.80:
                label = "Very High"
            elif prob >= 0.65:
                label = "High"
            elif prob >= 0.45:
                label = "Medium"
            elif prob >= 0.25:
                label = "Low"
            else:
                label = "Very Low"
            
            return label, prob, {'method': 'logistic', 'probability': prob}
        
        # Method 2: Use binned lookup table
        elif self.bin_stats is not None:
            for min_dist, max_dist, label in self.distance_bins:
                if min_dist <= avg_distance < max_dist:
                    bin_info = self.bin_stats[label]
                    return label, bin_info['empirical_prob'], bin_info
            
            # Fallback for out-of-range distances
            return "Very Low", 0.0, {'method': 'fallback'}
        
        # Fallback: no calibration available
        else:
            if avg_distance < 0.5:
                return "High", 0.75, {'method': 'uncalibrated'}
            elif avg_distance < 1.0:
                return "Medium", 0.5, {'method': 'uncalibrated'}
            else:
                return "Low", 0.25, {'method': 'uncalibrated'}
    
    def save_model(self):
        """Save calibration model to disk."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'distance_bins': self.distance_bins,
            'bin_stats': self.bin_stats,
        }
        
        with open(self.calibration_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Calibration model saved to {self.calibration_file}")
    
    def load_model(self):
        """Load calibration model from disk."""
        if not self.calibration_file.exists():
            print(f"⚠️  No calibration model found at {self.calibration_file}")
            return False
        
        with open(self.calibration_file, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.distance_bins = model_data['distance_bins']
        self.bin_stats = model_data['bin_stats']
        
        print(f"✅ Calibration model loaded from {self.calibration_file}")
        return True
    
    def generate_report(self, data: List[Dict], stats: Dict):
        """Generate calibration report."""
        print("\n" + "="*70)
        print("TRUST CALIBRATION REPORT")
        print("="*70)
        
        if not data:
            print("\n⚠️  No data available for calibration.")
            print("Please collect more interaction logs with accuracy evaluations.")
            return
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total samples: {len(data)}")
        print(f"   Fully correct: {sum(1 for d in data if d['correctness'] == 1.0)}")
        print(f"   Partially correct: {sum(1 for d in data if d['correctness'] == 0.5)}")
        print(f"   Incorrect: {sum(1 for d in data if d['correctness'] == 0.0)}")
        
        if stats:
            print(f"\n🤖 Logistic Model Performance:")
            print(f"   Accuracy: {stats['accuracy']:.2%}")
            print(f"   Mean predicted probability: {stats['mean_predicted_prob']:.2%}")
            print(f"   Mean actual correctness: {stats['mean_actual_correctness']:.2%}")
        
        if self.bin_stats:
            print(f"\n📈 Distance Bin Statistics:")
            for label, info in self.bin_stats.items():
                if info['n_samples'] > 0:
                    print(f"   {label:12} [{info['range'][0]:.2f}-{info['range'][1]:.2f}): "
                          f"{info['n_samples']:3} samples, "
                          f"{info['empirical_prob']:.1%} correct, "
                          f"avg dist={info['avg_distance']:.3f}")
                else:
                    print(f"   {label:12} [{info['range'][0]:.2f}-{info['range'][1]:.2f}): "
                          f"No samples")
        
        # Feature impact analysis
        self._analyze_feature_impact(data)
        
        print("\n" + "="*70)
    
    def _analyze_feature_impact(self, data: List[Dict]):
        """Analyze impact of claim checking and policy-aware features."""
        print(f"\n🔬 Feature Impact Analysis:")
        
        # Claim checking impact
        with_claims = [d for d in data if d.get('claim_checking_enabled')]
        without_claims = [d for d in data if not d.get('claim_checking_enabled')]
        
        if with_claims and without_claims:
            avg_correct_with = statistics.mean([d['correctness'] for d in with_claims])
            avg_correct_without = statistics.mean([d['correctness'] for d in without_claims])
            print(f"   Claim-Level Checking:")
            print(f"      With: {len(with_claims)} samples, {avg_correct_with:.1%} correct")
            print(f"      Without: {len(without_claims)} samples, {avg_correct_without:.1%} correct")
        
        # Policy-aware impact by risk level
        policy_data = [d for d in data if d.get('policy_aware_enabled')]
        if policy_data:
            print(f"   Policy-Aware Responses: {len(policy_data)} samples")
            
            # Group by risk level
            for risk in ['low', 'medium', 'high']:
                risk_samples = [d for d in policy_data if d.get('risk_level') == risk]
                if risk_samples:
                    avg_correct = statistics.mean([d['correctness'] for d in risk_samples])
                    with_disclaimer = sum(1 for d in risk_samples if d.get('had_policy_disclaimer'))
                    print(f"      {risk.capitalize()} risk: {len(risk_samples)} samples, "
                          f"{avg_correct:.1%} correct, {with_disclaimer} had disclaimers")
            
            # Topic category breakdown
            topics = set(d.get('topic_category') for d in policy_data if d.get('topic_category'))
            if topics:
                print(f"   Topic Categories:")
                for topic in sorted(topics):
                    topic_samples = [d for d in policy_data if d.get('topic_category') == topic]
                    if topic_samples:
                        avg_correct = statistics.mean([d['correctness'] for d in topic_samples])
                        print(f"      {topic}: {len(topic_samples)} samples, {avg_correct:.1%} correct")


def main():
    """Main calibration workflow."""
    print("🔧 Scarlet.AI Trust Calibration Script")
    print("="*70)
    
    calibrator = TrustCalibrator()
    
    # Load data
    print("\n1️⃣  Loading interaction and accuracy data...")
    data = calibrator.load_data()
    print(f"   Found {len(data)} interactions with accuracy labels")
    
    if not data:
        print("\n⚠️  No data available for calibration.")
        print("   Please ensure you have:")
        print("   - interactions.jsonl (logged interactions)")
        print("   - accuracy_evaluations.jsonl (accuracy ratings)")
        return
    
    # Train logistic model
    print("\n2️⃣  Training logistic regression model...")
    stats = calibrator.train_logistic_model(data)
    
    # Create distance bins
    print("\n3️⃣  Creating distance bins...")
    calibrator.create_distance_bins(data)
    
    # Save model
    print("\n4️⃣  Saving calibration model...")
    calibrator.save_model()
    
    # Generate report
    calibrator.generate_report(data, stats)
    
    # Test predictions
    print("\n5️⃣  Testing calibrated predictions:")
    test_distances = [0.2, 0.4, 0.7, 1.0, 1.5]
    for dist in test_distances:
        label, prob, info = calibrator.predict_confidence(dist)
        print(f"   Distance {dist:.2f} → {label:12} (P={prob:.1%})")
    
    print("\n✅ Calibration complete!")


if __name__ == "__main__":
    main()
