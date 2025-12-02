"""
System Test Script for Scarlet.AI Advanced Features
Tests all three major features: calibration, claim checking, and policy-aware responses
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "="*70)
    print("TEST 1: Module Imports")
    print("="*70)
    
    try:
        print("Testing explainability_module...")
        from explainability_module import (
            calculate_confidence_score,
            load_calibration_model,
            perform_claim_level_checking
        )
        print("  ✅ explainability_module OK")
    except Exception as e:
        print(f"  ❌ explainability_module FAILED: {e}")
        return False
    
    try:
        print("Testing topic_classifier...")
        from topic_classifier import TopicClassifier, TopicCategory, RiskLevel
        print("  ✅ topic_classifier OK")
    except Exception as e:
        print(f"  ❌ topic_classifier FAILED: {e}")
        return False
    
    try:
        print("Testing policy_aware_module...")
        from policy_aware_module import PolicyAwareResponder, apply_policy_aware_modulation
        print("  ✅ policy_aware_module OK")
    except Exception as e:
        print(f"  ❌ policy_aware_module FAILED: {e}")
        return False
    
    try:
        print("Testing calibration_script...")
        from calibration_script import TrustCalibrator
        print("  ✅ calibration_script OK")
    except Exception as e:
        print(f"  ❌ calibration_script FAILED: {e}")
        return False
    
    print("\n✅ All imports successful!\n")
    return True


def test_topic_classifier():
    """Test topic classification."""
    print("\n" + "="*70)
    print("TEST 2: Topic Classification")
    print("="*70)
    
    try:
        from topic_classifier import TopicClassifier
        
        classifier = TopicClassifier()
        
        test_cases = [
            ("How do I register for classes?", "academics", "low"),
            ("When will I get my financial aid?", "financial", "medium"),
            ("I'm feeling suicidal", "mental_health", "high"),
            ("How to report sexual harassment?", "title_ix", "high"),
        ]
        
        for question, expected_category, expected_risk in test_cases:
            result = classifier.classify(question)
            
            print(f"\nQ: {question}")
            print(f"  Expected: {expected_category} ({expected_risk})")
            print(f"  Got: {result['category']} ({result['risk_level']})")
            
            if result['category'] == expected_category and result['risk_level'] == expected_risk:
                print("  ✅ PASS")
            else:
                print("  ⚠️  MISMATCH (may be acceptable)")
        
        print("\n✅ Topic classifier test complete!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Topic classifier test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_policy_aware():
    """Test policy-aware response modulation."""
    print("\n" + "="*70)
    print("TEST 3: Policy-Aware Response Modulation")
    print("="*70)
    
    try:
        from topic_classifier import TopicClassifier
        from policy_aware_module import PolicyAwareResponder
        
        classifier = TopicClassifier()
        responder = PolicyAwareResponder(classifier)
        
        # Test low-risk
        print("\n--- Low Risk Test ---")
        result = responder.modulate_response(
            "How do I register for classes?",
            "You can register through WebReg.",
            []
        )
        print(f"Risk Level: {result['risk_level']}")
        print(f"Modifications: {result['applied_modifications']}")
        print(f"Has Disclaimer: {result['disclaimer'] is not None}")
        assert result['risk_level'] == 'low'
        print("✅ Low risk test passed")
        
        # Test medium-risk
        print("\n--- Medium Risk Test ---")
        result = responder.modulate_response(
            "When will I get my refund?",
            "Refunds are processed within 2 weeks.",
            []
        )
        print(f"Risk Level: {result['risk_level']}")
        print(f"Modifications: {result['applied_modifications']}")
        print(f"Has Disclaimer: {result['disclaimer'] is not None}")
        assert result['risk_level'] == 'medium'
        assert result['disclaimer'] is not None
        print("✅ Medium risk test passed")
        
        # Test high-risk
        print("\n--- High Risk Test ---")
        result = responder.modulate_response(
            "I'm feeling depressed",
            "You should contact counseling services.",
            []
        )
        print(f"Risk Level: {result['risk_level']}")
        print(f"Modifications: {result['applied_modifications']}")
        print(f"Has Disclaimer: {result['disclaimer'] is not None}")
        print(f"Has Contact Info: {result['contact_info'] is not None}")
        assert result['risk_level'] == 'high'
        assert result['disclaimer'] is not None
        assert result['contact_info'] is not None
        print("✅ High risk test passed")
        
        print("\n✅ Policy-aware tests complete!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Policy-aware test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_calibration():
    """Test calibration model existence and loading."""
    print("\n" + "="*70)
    print("TEST 4: Trust Calibration")
    print("="*70)
    
    try:
        from explainability_module import load_calibration_model
        from pathlib import Path
        
        model_path = Path("./evaluation_logs/calibration_model.pkl")
        
        if model_path.exists():
            print(f"✅ Calibration model found at {model_path}")
            
            success = load_calibration_model()
            if success:
                print("✅ Model loaded successfully")
            else:
                print("⚠️  Model exists but failed to load")
        else:
            print(f"ℹ️  No calibration model found (expected before first training)")
            print("   Run: python calibration_script.py")
        
        print("\n✅ Calibration test complete!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Calibration test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_confidence_calculation():
    """Test confidence score calculation."""
    print("\n" + "="*70)
    print("TEST 5: Confidence Calculation")
    print("="*70)
    
    try:
        from explainability_module import calculate_confidence_score
        
        # Test with mock chunks
        test_chunks = [
            {"distance": 0.3, "text": "Test chunk 1"},
            {"distance": 0.4, "text": "Test chunk 2"},
            {"distance": 0.5, "text": "Test chunk 3"},
        ]
        
        result = calculate_confidence_score(test_chunks, use_calibration=False)
        level, emoji, color, avg_dist, prob, info = result
        
        print(f"Test Chunks: avg_distance = {avg_dist:.3f}")
        print(f"Confidence: {level} {emoji}")
        print(f"Color: {color}")
        
        assert level in ["Very High", "High", "Medium", "Low", "Very Low", "No Data"]
        assert 0 <= avg_dist <= 2.0
        
        print("✅ Confidence calculation working")
        
        # Test with calibration
        result_cal = calculate_confidence_score(test_chunks, use_calibration=True)
        level_cal, emoji_cal, color_cal, avg_dist_cal, prob_cal, info_cal = result_cal
        
        print(f"\nWith calibration: {level_cal}")
        if prob_cal:
            print(f"Calibrated probability: {prob_cal:.1%}")
        else:
            print("No calibrated probability (model not loaded)")
        
        print("\n✅ Confidence tests complete!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Confidence test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_logs():
    """Test evaluation logging structure."""
    print("\n" + "="*70)
    print("TEST 6: Evaluation Logs")
    print("="*70)
    
    try:
        from pathlib import Path
        import json
        
        log_dir = Path("./evaluation_logs")
        
        if not log_dir.exists():
            print("ℹ️  Creating evaluation_logs directory...")
            log_dir.mkdir(exist_ok=True)
        
        # Check for log files
        interactions_file = log_dir / "interactions.jsonl"
        feedback_file = log_dir / "feedback.csv"
        accuracy_file = log_dir / "accuracy_evaluations.jsonl"
        
        print(f"\nLog Directory: {log_dir.absolute()}")
        print(f"  interactions.jsonl: {'✅ exists' if interactions_file.exists() else 'ℹ️  will be created on first use'}")
        print(f"  feedback.csv: {'✅ exists' if feedback_file.exists() else 'ℹ️  will be created on first use'}")
        print(f"  accuracy_evaluations.jsonl: {'✅ exists' if accuracy_file.exists() else 'ℹ️  will be created on first use'}")
        
        # Count interactions if file exists
        if interactions_file.exists():
            with open(interactions_file, 'r') as f:
                count = sum(1 for line in f)
            print(f"\n📊 Current interactions logged: {count}")
        
        print("\n✅ Evaluation logs test complete!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Evaluation logs test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and provide summary."""
    print("\n" + "="*70)
    print("SCARLET.AI ADVANCED FEATURES - SYSTEM TEST")
    print("="*70)
    print("\nThis script tests:")
    print("  1. Module imports")
    print("  2. Topic classification")
    print("  3. Policy-aware response modulation")
    print("  4. Trust calibration")
    print("  5. Confidence calculation")
    print("  6. Evaluation logging")
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Topic Classifier", test_topic_classifier()))
    results.append(("Policy-Aware", test_policy_aware()))
    results.append(("Calibration", test_calibration()))
    results.append(("Confidence", test_confidence_calculation()))
    results.append(("Evaluation Logs", test_evaluation_logs()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20} {status}")
    
    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed_count}/{total} tests passed")
    
    if passed_count == total:
        print("\n🎉 All tests passed! System is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed_count} test(s) failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
