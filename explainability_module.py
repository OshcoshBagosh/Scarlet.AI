"""
Explainability Module for Scarlet.AI
Provides confidence scoring, hallucination detection, and transparency features.
"""

from typing import List, Dict, Tuple, Optional
import statistics
import pickle
from pathlib import Path


# Global calibration model
_calibration_model = None


def load_calibration_model(model_path: str = "./evaluation_logs/calibration_model.pkl") -> bool:
    """
    Load calibration model for empirically-calibrated confidence scores.
    
    Args:
        model_path: Path to calibration model file
        
    Returns:
        True if model loaded successfully, False otherwise
    """
    global _calibration_model
    
    model_file = Path(model_path)
    if not model_file.exists():
        return False
    
    try:
        with open(model_file, 'rb') as f:
            _calibration_model = pickle.load(f)
        return True
    except Exception as e:
        print(f"Failed to load calibration model: {e}")
        return False


def get_calibrated_confidence(avg_distance: float) -> Tuple[str, float, Optional[Dict]]:
    """
    Get calibrated confidence using trained model.
    
    Args:
        avg_distance: Average retrieval distance
        
    Returns:
        Tuple of (confidence_label, probability, calibration_info)
    """
    if _calibration_model is None:
        # Fallback to uncalibrated thresholds
        if avg_distance < 0.5:
            return "High", 0.75, None
        elif avg_distance < 1.0:
            return "Medium", 0.5, None
        else:
            return "Low", 0.25, None
    
    # Use calibration model
    import numpy as np
    
    model = _calibration_model.get('model')
    scaler = _calibration_model.get('scaler')
    bin_stats = _calibration_model.get('bin_stats')
    distance_bins = _calibration_model.get('distance_bins')
    
    # Method 1: Logistic model
    if model is not None and scaler is not None:
        X = np.array([[avg_distance]])
        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[0, 1]
        
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
    
    # Method 2: Binned lookup
    elif bin_stats is not None and distance_bins is not None:
        for min_dist, max_dist, label in distance_bins:
            if min_dist <= avg_distance < max_dist:
                bin_info = bin_stats[label]
                return label, bin_info['empirical_prob'], bin_info
        
        # Out of range
        return "Very Low", 0.0, {'method': 'fallback'}
    
    # Fallback
    else:
        if avg_distance < 0.5:
            return "High", 0.75, None
        elif avg_distance < 1.0:
            return "Medium", 0.5, None
        else:
            return "Low", 0.25, None


def calculate_confidence_score(chunks: List[Dict], use_calibration: bool = True) -> Tuple[str, str, str, float, Optional[float], Optional[Dict]]:
    """
    Calculate confidence level based on retrieval distance scores.
    Now supports calibrated confidence using empirical data.
    
    Args:
        chunks: List of retrieved chunks with distance scores
        use_calibration: Whether to use calibrated confidence (default True)
        
    Returns:
        Tuple of (level, emoji, color, avg_distance, calibrated_prob, calibration_info)
        level: "Very High", "High", "Medium", "Low", "Very Low", or "No Data"
        emoji: Visual indicator
        color: Hex color for UI
        avg_distance: Average distance score
        calibrated_prob: Empirical probability of correctness (if calibrated)
        calibration_info: Additional calibration metadata
    """
    if not chunks:
        return "No Data", "⚫", "#808080", 1.0, None, None
    
    # Get distances
    distances = [c.get('distance', 1.0) for c in chunks if c.get('distance') is not None]
    
    if not distances:
        return "No Data", "⚫", "#808080", 1.0, None, None
    
    avg_distance = statistics.mean(distances)
    
    # Use calibrated confidence if available
    if use_calibration:
        level, prob, info = get_calibrated_confidence(avg_distance)
        
        # Map level to emoji and color
        if level == "Very High":
            emoji, color = "🟢", "#28a745"
        elif level == "High":
            emoji, color = "🟢", "#28a745"
        elif level == "Medium":
            emoji, color = "🟡", "#ffc107"
        elif level == "Low":
            emoji, color = "🟠", "#fd7e14"
        else:  # Very Low
            emoji, color = "🔴", "#dc3545"
        
        return level, emoji, color, avg_distance, prob, info
    
    # Fallback: uncalibrated thresholds
    if avg_distance < 0.5:
        return "High", "🟢", "#28a745", avg_distance, None, None
    elif avg_distance < 1.0:
        return "Medium", "🟡", "#ffc107", avg_distance, None, None
    else:
        return "Low", "🔴", "#dc3545", avg_distance, None, None


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


def generate_confidence_badge_html(level: str, emoji: str, color: str, avg_distance: float,
                                   calibrated_prob: Optional[float] = None,
                                   calibration_info: Optional[Dict] = None,
                                   answer_text: Optional[str] = None) -> str:
    """
    Generate HTML for confidence badge display with calibrated probability.
    
    Args:
        level: Confidence level
        emoji: Emoji indicator
        color: Hex color
        avg_distance: Average distance score
        calibrated_prob: Calibrated probability of correctness
        calibration_info: Additional calibration metadata
        
    Returns:
        HTML string for display
    """
    # Detect uncertainty / insufficient info phrases in answer
    uncertainty_phrases = [
        "i couldn't find",
        "i could not find",
        "i don't have information",
        "no relevant information",
        "no relevant info",
        "not sure",
        "i'm not sure",
        "cannot find",
        "can't find",
        "insufficient information",
        "no data",
    ]
    downgrade_for_uncertainty = False
    answer_lower = answer_text.lower() if answer_text else ""
    if any(p in answer_lower for p in uncertainty_phrases):
        downgrade_for_uncertainty = True

    # Dynamic historical accuracy phrase mapping
    accuracy_phrase = None
    if calibrated_prob is not None:
        if calibrated_prob >= 0.85:
            accuracy_phrase = "~85–95% historically accurate"
        elif calibrated_prob >= 0.70:
            accuracy_phrase = "~70–85% historically accurate"
        elif calibrated_prob >= 0.55:
            accuracy_phrase = "~55–70% historically accurate"
        else:
            accuracy_phrase = "Often inaccurate (<55%)"

    # If uncertainty detected, force Low confidence visual even if calibration is high
    if downgrade_for_uncertainty:
        level_display = "Low"
        emoji_display = "🔴"
        color_display = "#dc3545"
    else:
        level_display = level
        emoji_display = emoji
        color_display = color

    confidence_text = f"Answer Confidence: {emoji_display} {level_display}"
    if accuracy_phrase and not downgrade_for_uncertainty:
        confidence_text += f" ({accuracy_phrase})"
    elif downgrade_for_uncertainty:
        confidence_text += " (Model retrieved sources but could not answer confidently)"
    
    # Additional info based on calibration
    extra_info = f"Relevance Score: {avg_distance:.3f}"
    if downgrade_for_uncertainty:
        extra_info += " | Answer expressed insufficient information"
    if calibration_info and 'n_samples' in calibration_info:
        n_samples = calibration_info.get('n_samples', 0)
        if n_samples > 0:
            extra_info += f" | Based on {n_samples} past examples"
    
    return (
        f"<div style='background:{color_display}20; padding:0.8rem; border-radius:8px; "
        f"border-left:4px solid {color_display}; margin:1rem 0;'>"
        f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
        f"<div><strong style='font-size:1.1em;'>{confidence_text}</strong>"
        f"<div style='font-size:0.85em; color:#666; margin-top:0.3rem;'>{extra_info}</div>"
        f"</div></div></div>"
    )


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


# ============================================================================
# CLAIM-LEVEL EVIDENCE CHECKING (Micro-Explanations)
# ============================================================================

def extract_claims_from_answer(answer: str, llm_client=None, model: str = "llama3.2:3b") -> List[str]:
    """
    Extract atomic factual claims from an answer.
    
    Args:
        answer: Generated answer text
        llm_client: LLM client (e.g., ollama module)
        model: Model name to use
        
    Returns:
        List of atomic claims
    """
    if not llm_client:
        # Fallback: simple sentence splitting
        import re
        sentences = re.split(r'[.!?]+', answer)
        claims = [s.strip() for s in sentences if len(s.strip()) > 20]
        return claims[:6]  # Limit to 6 claims
    
    # Use LLM to extract structured claims
    prompt = f"""Extract 3-6 atomic factual claims from this answer. Each claim should be a single, verifiable statement.

Answer: {answer}

Format your response as a numbered list:
1. [First claim]
2. [Second claim]
...

Claims:"""
    
    try:
        response = llm_client.generate(model=model, prompt=prompt)
        response_text = response.get('response', '')
        
        # Parse numbered list
        import re
        claims = []
        for line in response_text.split('\n'):
            match = re.match(r'^\d+\.\s*(.+)$', line.strip())
            if match:
                claims.append(match.group(1).strip())
        
        return claims[:6]  # Limit to 6
    except Exception as e:
        print(f"Failed to extract claims with LLM: {e}")
        # Fallback to sentence splitting
        import re
        sentences = re.split(r'[.!?]+', answer)
        claims = [s.strip() for s in sentences if len(s.strip()) > 20]
        return claims[:6]


def verify_claim_support(claim: str, collection, n_results: int = 3) -> Tuple[str, float, List[Dict]]:
    """
    Verify if a claim is supported by the knowledge base.
    
    Args:
        claim: Claim to verify
        collection: ChromaDB collection to search
        n_results: Number of sources to retrieve
        
    Returns:
        Tuple of (support_level, avg_distance, sources)
        support_level: "Strong", "Moderate", "Weak", or "None"
    """
    try:
        # Search for claim in knowledge base
        results = collection.query(
            query_texts=[claim],
            n_results=n_results
        )
        
        if not results or not results.get('distances') or not results['distances'][0]:
            return "None", 1.0, []
        
        distances = results['distances'][0]
        avg_distance = statistics.mean(distances)
        
        # Extract source information
        sources = []
        if results.get('metadatas') and results['metadatas'][0]:
            for i, metadata in enumerate(results['metadatas'][0]):
                sources.append({
                    'title': metadata.get('title', 'Unknown'),
                    'url': metadata.get('url', '#'),
                    'distance': distances[i]
                })
        
        # Determine support level
        if avg_distance < 0.4:
            support = "Strong"
        elif avg_distance < 0.7:
            support = "Moderate"
        elif avg_distance < 1.0:
            support = "Weak"
        else:
            support = "None"
        
        return support, avg_distance, sources
        
    except Exception as e:
        print(f"Error verifying claim: {e}")
        return "None", 1.0, []


def generate_claim_audit_table_html(claim_verifications: List[Dict]) -> str:
    """
    Generate HTML table showing claim-level support.
    
    Args:
        claim_verifications: List of dicts with keys: claim, support, distance, sources
        
    Returns:
        HTML string with audit table
    """
    if not claim_verifications:
        return "<p>No claims to verify.</p>"
    
    # Build table
    rows = []
    for i, cv in enumerate(claim_verifications):
        claim = cv.get('claim', '')
        support = cv.get('support', 'None')
        distance = cv.get('distance', 1.0)
        sources = cv.get('sources', [])
        
        # Color code by support level
        if support == "Strong":
            color = "#28a745"
            emoji = "✅"
        elif support == "Moderate":
            color = "#ffc107"
            emoji = "⚠️"
        elif support == "Weak":
            color = "#fd7e14"
            emoji = "⚠️"
        else:  # None
            color = "#dc3545"
            emoji = "❌"
        
        # Format sources
        source_links = []
        for src in sources[:2]:  # Show top 2 sources
            title = src.get('title', 'Source')[:30]
            url = src.get('url', '#')
            source_links.append(f'<a href="{url}" target="_blank" style="color:{color}; text-decoration:none;">{title}...</a>')
        
        source_html = "<br>".join(source_links) if source_links else "No sources"
        
        rows.append(
            f"<tr>"
            f"<td style='padding:0.5rem; border-bottom:1px solid #ddd; font-size:0.9em;'>{claim}</td>"
            f"<td style='padding:0.5rem; border-bottom:1px solid #ddd; text-align:center;'>"
            f"<span style='color:{color}; font-weight:bold;'>{emoji} {support}</span>"
            f"<div style='font-size:0.75em; color:#666;'>Score: {distance:.3f}</div>"
            f"</td>"
            f"<td style='padding:0.5rem; border-bottom:1px solid #ddd; font-size:0.85em;'>{source_html}</td>"
            f"</tr>"
        )
    
    table_html = (
        "<style>"
        ".claim-audit-container{background:#1e1e1e;border:1px solid #333;border-radius:8px;margin:1rem 0;overflow:hidden;font-family:inherit;}"
        ".claim-audit-header{background:#262626;padding:0.8rem;border-bottom:1px solid #333;}"
        ".claim-audit-header strong{font-size:1.05em;color:#fafafa;}"
        ".claim-audit-sub{font-size:0.8em;color:#b0b0b0;margin-top:0.35rem;}"
        ".claim-audit-table{width:100%;border-collapse:collapse;}"
        ".claim-audit-table th{padding:0.55rem;text-align:left;border-bottom:1px solid #333;font-size:0.75em;color:#dcdcdc;font-weight:600;background:#262626;}"
        ".claim-audit-table td{padding:0.5rem;border-bottom:1px solid #2d2d2d;font-size:0.75em;color:#e0e0e0;vertical-align:top;}"
        ".claim-audit-table tr:nth-child(even){background:#222;}"
        ".claim-audit-table a{color:#5bc46b;text-decoration:none;}"
        ".claim-audit-table a:hover{text-decoration:underline;}"
        ".support-score{font-size:0.65em;color:#aaaaaa;margin-top:0.25rem;}"
        "</style>"
        "<div class='claim-audit-container'>"
        "<div class='claim-audit-header'>"
        "<strong>📋 Claim-Level Evidence Audit</strong>"
        "<div class='claim-audit-sub'>Each claim in the answer is independently verified against source documents.</div>"
        "</div>"
        "<table class='claim-audit-table'>"
        "<thead><tr>"
        "<th style='width:45%;'>Claim</th>"
        "<th style='width:20%;text-align:center;'>Support</th>"
        "<th style='width:35%;'>Top Sources</th>"
        "</tr></thead><tbody>"
        f"{''.join(rows)}"
        "</tbody></table></div>"
    )
    
    return table_html


def perform_claim_level_checking(answer: str, collection, llm_client=None, 
                                  model: str = "llama3.2:3b") -> Tuple[List[Dict], str]:
    """
    Full pipeline: extract claims, verify each, generate audit table.
    
    Args:
        answer: Generated answer
        collection: ChromaDB collection
        llm_client: LLM client for claim extraction
        model: Model name
        
    Returns:
        Tuple of (claim_verifications, audit_table_html)
    """
    # Step 1: Extract claims
    claims = extract_claims_from_answer(answer, llm_client, model)
    
    if not claims:
        return [], "<p>No claims extracted.</p>"
    
    # Step 2: Verify each claim
    claim_verifications = []
    for claim in claims:
        support, distance, sources = verify_claim_support(claim, collection, n_results=3)
        claim_verifications.append({
            'claim': claim,
            'support': support,
            'distance': distance,
            'sources': sources
        })
    
    # Step 3: Generate audit table
    audit_html = generate_claim_audit_table_html(claim_verifications)
    
    return claim_verifications, audit_html


# Export main functions
__all__ = [
    'calculate_confidence_score',
    'detect_hallucination',
    'generate_confidence_badge_html',
    'generate_hallucination_warning_html',
    'get_source_confidence_indicator',
    'format_sources_with_confidence',
    'generate_evaluation_prompt',
    'load_calibration_model',
    'get_calibrated_confidence',
    'extract_claims_from_answer',
    'verify_claim_support',
    'generate_claim_audit_table_html',
    'perform_claim_level_checking'
]
