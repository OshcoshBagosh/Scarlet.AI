"""
User Testing and Evaluation Module for Scarlet.AI
Collects feedback, logs interactions, and generates evaluation metrics.
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import streamlit as st


class EvaluationLogger:
    """Handles logging of user interactions and feedback."""
    
    def __init__(self, log_dir: str = "./evaluation_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.interactions_file = self.log_dir / "interactions.jsonl"
        self.feedback_file = self.log_dir / "feedback.csv"
        self.accuracy_file = self.log_dir / "accuracy_evaluations.jsonl"
        
        # Initialize CSV if it doesn't exist
        if not self.feedback_file.exists():
            self._init_feedback_csv()
    
    def _init_feedback_csv(self):
        """Initialize feedback CSV with headers."""
        with open(self.feedback_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'question', 'mode', 'helpful', 
                'trust_rating', 'accuracy_rating', 'clarity_rating',
                'confidence_level', 'num_sources', 'had_warning'
            ])
    
    def log_interaction(self, question: str, answer: str, sources: List[Dict], 
                       mode: str, confidence_level: str, had_warning: bool):
        """
        Log a complete interaction for analysis.
        
        Args:
            question: User's question
            answer: Generated answer
            sources: Retrieved sources
            mode: "black_box" or "explainable"
            confidence_level: Confidence level of answer
            had_warning: Whether hallucination warning was shown
        """
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer,
            'mode': mode,
            'confidence_level': confidence_level,
            'num_sources': len(sources),
            'had_warning': had_warning,
            'sources': [
                {
                    'title': s.get('title', ''),
                    'url': s.get('url', ''),
                    'distance': s.get('distance', 1.0)
                }
                for s in sources
            ]
        }
        
        with open(self.interactions_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(interaction) + '\n')
    
    def log_feedback(self, question: str, mode: str, helpful: str, 
                    trust_rating: int, accuracy_rating: int, clarity_rating: int,
                    confidence_level: str, num_sources: int, had_warning: bool):
        """
        Log user feedback for an answer.
        
        Args:
            question: Original question
            mode: "black_box" or "explainable"
            helpful: "Yes", "Somewhat", or "No"
            trust_rating: 1-5 rating
            accuracy_rating: 1-5 rating
            clarity_rating: 1-5 rating
            confidence_level: Confidence level shown
            num_sources: Number of sources provided
            had_warning: Whether warning was shown
        """
        with open(self.feedback_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                question,
                mode,
                helpful,
                trust_rating,
                accuracy_rating,
                clarity_rating,
                confidence_level,
                num_sources,
                had_warning
            ])
    
    def log_accuracy_evaluation(self, question: str, answer: str, 
                               is_correct: str, actual_answer: Optional[str],
                               evaluator_notes: Optional[str]):
        """
        Log accuracy evaluation from tester/expert.
        
        Args:
            question: Original question
            answer: AI-generated answer
            is_correct: "Yes", "No", or "Partially"
            actual_answer: What the correct answer should be
            evaluator_notes: Additional notes from evaluator
        """
        evaluation = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'ai_answer': answer,
            'is_correct': is_correct,
            'actual_answer': actual_answer,
            'evaluator_notes': evaluator_notes
        }
        
        with open(self.accuracy_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(evaluation) + '\n')
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics from logged data."""
        stats = {
            'total_interactions': 0,
            'total_feedback': 0,
            'avg_trust': 0,
            'avg_accuracy': 0,
            'avg_clarity': 0,
            'helpful_percent': 0,
            'black_box_count': 0,
            'explainable_count': 0
        }
        
        # Count interactions
        if self.interactions_file.exists():
            with open(self.interactions_file, 'r', encoding='utf-8') as f:
                stats['total_interactions'] = sum(1 for _ in f)
        
        # Analyze feedback
        if self.feedback_file.exists():
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                stats['total_feedback'] = len(rows)
                
                if rows:
                    trust_ratings = [int(r['trust_rating']) for r in rows if r['trust_rating'].isdigit()]
                    accuracy_ratings = [int(r['accuracy_rating']) for r in rows if r['accuracy_rating'].isdigit()]
                    clarity_ratings = [int(r['clarity_rating']) for r in rows if r['clarity_rating'].isdigit()]
                    
                    stats['avg_trust'] = sum(trust_ratings) / len(trust_ratings) if trust_ratings else 0
                    stats['avg_accuracy'] = sum(accuracy_ratings) / len(accuracy_ratings) if accuracy_ratings else 0
                    stats['avg_clarity'] = sum(clarity_ratings) / len(clarity_ratings) if clarity_ratings else 0
                    
                    helpful_count = sum(1 for r in rows if r['helpful'] == 'Yes')
                    stats['helpful_percent'] = (helpful_count / len(rows)) * 100
                    
                    stats['black_box_count'] = sum(1 for r in rows if r['mode'] == 'black_box')
                    stats['explainable_count'] = sum(1 for r in rows if r['mode'] == 'explainable')
        
        return stats


def render_feedback_widget(logger: EvaluationLogger, question: str, mode: str, 
                          confidence_level: str, num_sources: int, had_warning: bool):
    """
    Render Streamlit feedback collection widget.
    
    Args:
        logger: EvaluationLogger instance
        question: Original question
        mode: Current mode
        confidence_level: Confidence level of answer
        num_sources: Number of sources used
        had_warning: Whether warning was shown
    """
    st.markdown("---")
    st.subheader("📊 Help Us Improve")
    
    with st.form(key=f"feedback_form_{datetime.now().timestamp()}"):
        st.write("**How would you rate this answer?**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            helpful = st.radio(
                "Was this answer helpful?",
                ["Yes", "Somewhat", "No"],
                horizontal=True
            )
            
            trust_rating = st.slider(
                "How much do you trust this answer?",
                1, 5, 3,
                help="1 = Don't trust at all, 5 = Completely trust"
            )
        
        with col2:
            accuracy_rating = st.slider(
                "How accurate do you think this answer is?",
                1, 5, 3,
                help="1 = Not accurate, 5 = Very accurate"
            )
            
            clarity_rating = st.slider(
                "How clear was the answer?",
                1, 5, 3,
                help="1 = Very confusing, 5 = Very clear"
            )
        
        submitted = st.form_submit_button("Submit Feedback")
        
        if submitted:
            logger.log_feedback(
                question=question,
                mode=mode,
                helpful=helpful,
                trust_rating=trust_rating,
                accuracy_rating=accuracy_rating,
                clarity_rating=clarity_rating,
                confidence_level=confidence_level,
                num_sources=num_sources,
                had_warning=had_warning
            )
            st.success("✅ Thank you for your feedback!")
            return True
    
    return False


def render_accuracy_evaluation_widget(logger: EvaluationLogger, question: str, answer: str):
    """
    Render expert/tester accuracy evaluation widget.
    
    Args:
        logger: EvaluationLogger instance
        question: Original question
        answer: Generated answer
    """
    st.markdown("---")
    st.subheader("🔍 Accuracy Evaluation (Testers Only)")
    
    with st.expander("Click here if you're verifying accuracy"):
        with st.form(key=f"accuracy_form_{datetime.now().timestamp()}"):
            is_correct = st.radio(
                "Is this answer factually correct?",
                ["Yes", "Partially Correct", "No"],
                help="Compare with official Rutgers sources"
            )
            
            actual_answer = st.text_area(
                "What should the correct answer be? (if applicable)",
                help="Provide the accurate information or corrections"
            )
            
            evaluator_notes = st.text_area(
                "Additional notes",
                help="Any observations about sources, hallucinations, or improvements"
            )
            
            submitted = st.form_submit_button("Submit Evaluation")
            
            if submitted:
                logger.log_accuracy_evaluation(
                    question=question,
                    answer=answer,
                    is_correct=is_correct,
                    actual_answer=actual_answer if actual_answer else None,
                    evaluator_notes=evaluator_notes if evaluator_notes else None
                )
                st.success("✅ Evaluation recorded. Thank you!")
                return True
    
    return False


def render_evaluation_dashboard(logger: EvaluationLogger):
    """
    Render evaluation dashboard with statistics.
    
    Args:
        logger: EvaluationLogger instance
    """
    st.title("📈 Evaluation Dashboard")
    
    stats = logger.get_summary_stats()
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Interactions", stats['total_interactions'])
    
    with col2:
        st.metric("Total Feedback", stats['total_feedback'])
    
    with col3:
        st.metric("Helpful Answers", f"{stats['helpful_percent']:.1f}%")
    
    with col4:
        completion_rate = (stats['total_feedback'] / stats['total_interactions'] * 100) if stats['total_interactions'] > 0 else 0
        st.metric("Feedback Rate", f"{completion_rate:.1f}%")
    
    # Ratings
    st.subheader("Average Ratings (out of 5)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Trust", f"{stats['avg_trust']:.2f}")
    
    with col2:
        st.metric("Accuracy", f"{stats['avg_accuracy']:.2f}")
    
    with col3:
        st.metric("Clarity", f"{stats['avg_clarity']:.2f}")
    
    # Mode comparison
    st.subheader("Mode Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Black Box Mode Uses", stats['black_box_count'])
    
    with col2:
        st.metric("Explainable Mode Uses", stats['explainable_count'])
    
    # Data export
    st.subheader("Export Data")
    
    if st.button("Download Feedback CSV"):
        if logger.feedback_file.exists():
            with open(logger.feedback_file, 'r', encoding='utf-8') as f:
                st.download_button(
                    "📥 Download Feedback Data",
                    f.read(),
                    file_name=f"scarlet_ai_feedback_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )


# Export main components
__all__ = [
    'EvaluationLogger',
    'render_feedback_widget',
    'render_accuracy_evaluation_widget',
    'render_evaluation_dashboard'
]
