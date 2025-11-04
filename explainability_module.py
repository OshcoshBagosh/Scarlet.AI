"""
Explainability Module for Scarlet.AI
Provides confidence scoring, hallucination detection, and transparency features.
"""

from typing import List, Dict, Tuple
import statistics


def calculate_confidence_score(chunks: List[Dict]) -> Tuple[str, str, str, float]:
    """
    Calculate confidence level based on retrieval distance scores.
    
    Args:
        chunks: List of retrieved chunks with distance scores
        
    Returns:
        Tuple of (level, emoji, color, avg_distance)
        level: "High", "Medium", "Low", or "No Data"
        emoji: Visual indicator
        color: Hex color for UI
        avg_distance: Average distance score
    """
    if not chunks:
        return "No Data", "⚫", "#808080", 1.0
    
    # Get distances
    distances = [c.get('distance', 1.0) for c in chunks if c.get('distance') is not None]
    
    if not distances:
        return "No Data", "⚫", "#808080", 1.0
    
    avg_distance = statistics.mean(distances)
    
    # Define thresholds (lower distance = higher similarity = better)
    if avg_distance < 0.5:
        return "High", "🟢", "#28a745", avg_distance
    elif avg_distance < 1.0:
        return "Medium", "🟡", "#ffc107", avg_distance
    else:
        return "Low", "🔴", "#dc3545", avg_distance


def detect_hallucination(chunks: List[Dict], answer: str, confidence_level: str) -> Tuple[bool, str]:
    """
    Detect potential hallucinations in the generated answer.
    
    Args:
        chunks: Retrieved source chunks
        answer: Generated answer text
        confidence_level: Confidence level from calculate_confidence_score
        
    Returns:
        Tuple of (is_hallucination, reason)
    """
    reasons = []
    
    # Check 1: No sources found
    if not chunks or len(chunks) == 0:
        return True, "No relevant sources found in the knowledge base."
    
    # Check 2: Very low confidence
    if confidence_level == "Low":
        distances = [c.get('distance', 1.0) for c in chunks]
        avg_dist = statistics.mean(distances) if distances else 1.0
        if avg_dist > 1.2:
            reasons.append(f"Low confidence in retrieved sources (score: {avg_dist:.2f})")
    
    # Check 3: Answer is suspiciously long compared to context
    if chunks:
        total_context_length = sum(len(c.get('text', '')) for c in chunks)
        answer_length = len(answer)
        
        # If answer is more than 60% of the context length, it might be extrapolating
        if total_context_length > 0 and answer_length > total_context_length * 0.6:
            reasons.append("Answer may extrapolate beyond provided sources")
    
    # Check 4: Generic "I don't know" responses
    dont_know_phrases = [
        "i couldn't find",
        "i don't have information",
        "sorry, i",
        "no information available",
        "i'm not sure"
    ]
    
    answer_lower = answer.lower()
    if any(phrase in answer_lower for phrase in dont_know_phrases):
        return True, "AI acknowledged insufficient information."
    
    # Return result
    if reasons:
        return True, " • ".join(reasons)
    
    return False, ""


def generate_confidence_badge_html(level: str, emoji: str, color: str, avg_distance: float) -> str:
    """
    Generate HTML for confidence badge display.
    
    Args:
        level: Confidence level
        emoji: Emoji indicator
        color: Hex color
        avg_distance: Average distance score
        
    Returns:
        HTML string for display
    """
    return f"""
    <div style="background: {color}20; padding: 0.8rem; border-radius: 8px; 
                border-left: 4px solid {color}; margin: 1rem 0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <strong style="font-size: 1.1em;">Answer Confidence: {emoji} {level}</strong>
                <div style="font-size: 0.85em; color: #666; margin-top: 0.3rem;">
                    Relevance Score: {avg_distance:.3f}
                </div>
            </div>
        </div>
    </div>
    """


def generate_hallucination_warning_html(reason: str) -> str:
    """
    Generate HTML for hallucination warning banner.
    
    Args:
        reason: Reason for the warning
        
    Returns:
        HTML string for warning display
    """
    return f"""
    <div style="background: #fff3cd; border: 2px solid #ffc107; border-radius: 8px; 
                padding: 1rem; margin: 1rem 0;">
        <div style="display: flex; align-items: start;">
            <div style="font-size: 1.5em; margin-right: 0.5rem;">⚠️</div>
            <div>
                <strong style="color: #856404;">Verification Recommended</strong>
                <p style="margin: 0.5rem 0 0 0; color: #856404;">
                    This answer may not be fully reliable:
                </p>
                <ul style="margin: 0.3rem 0 0.5rem 1.2rem; color: #856404;">
                    <li>{reason}</li>
                </ul>
                <p style="margin: 0.5rem 0 0 0; color: #856404; font-size: 0.9em;">
                    <strong>Please double-check with official Rutgers resources or contact the relevant office.</strong>
                </p>
            </div>
        </div>
    </div>
    """


def get_source_confidence_indicator(distance: float) -> str:
    """
    Get a visual indicator for individual source confidence.
    
    Args:
        distance: Distance score for the source
        
    Returns:
        HTML string with confidence indicator
    """
    if distance < 0.5:
        return '<span style="color: #28a745;">●●●</span> <small style="color: #666;">(High relevance)</small>'
    elif distance < 1.0:
        return '<span style="color: #ffc107;">●●○</span> <small style="color: #666;">(Medium relevance)</small>'
    else:
        return '<span style="color: #dc3545;">●○○</span> <small style="color: #666;">(Low relevance)</small>'


def format_sources_with_confidence(sources: List[Dict]) -> str:
    """
    Format sources with confidence indicators for display.
    
    Args:
        sources: List of source dictionaries
        
    Returns:
        HTML string with formatted sources
    """
    if not sources:
        return "<p>No sources available.</p>"
    
    html_parts = []
    for i, source in enumerate(sources):
        distance = source.get('distance', 1.0)
        confidence_indicator = get_source_confidence_indicator(distance)
        title = source.get('title', 'Unknown Source')
        url = source.get('url', '#')
        
        html_parts.append(f"""
        <div class="source-box" style="background:#f9f9f9; padding:0.8rem; 
                                       border-radius:6px; margin:0.5rem 0; 
                                       border-left:3px solid #CC0000;">
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <strong style="color:#000;">{i+1}. {title}</strong>
                <span style="white-space: nowrap; margin-left: 1rem;">{confidence_indicator}</span>
            </div>
            <div style="font-size: 0.85em; color:#666; margin-top:0.3rem;">
                Relevance Score: {distance:.3f} | 
                <a href="{url}" target="_blank" style="color:#CC0000; text-decoration:none;">
                    🔗 View Source
                </a>
            </div>
        </div>
        """)
    
    return "".join(html_parts)


def generate_evaluation_prompt(question: str, answer: str, sources: List[Dict]) -> str:
    """
    Generate a prompt for user evaluation of answer quality.
    
    Args:
        question: Original question
        answer: Generated answer
        sources: Retrieved sources
        
    Returns:
        Formatted evaluation prompt
    """
    return f"""
    ### Evaluate This Answer
    
    **Question:** {question}
    
    **Answer:** {answer}
    
    **Sources Used:** {len(sources)}
    
    Please rate:
    - Accuracy (1-5): Was the answer correct?
    - Helpfulness (1-5): Did this answer help you?
    - Trust (1-5): How much do you trust this answer?
    """


# Export main functions
__all__ = [
    'calculate_confidence_score',
    'detect_hallucination',
    'generate_confidence_badge_html',
    'generate_hallucination_warning_html',
    'get_source_confidence_indicator',
    'format_sources_with_confidence',
    'generate_evaluation_prompt'
]
