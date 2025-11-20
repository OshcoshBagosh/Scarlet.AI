"""
Policy-Aware Response Module for Scarlet.AI
Modulates RAG behavior based on topic risk level.

Features:
- Risk-based disclaimers
- Verbatim quote requirements for high-risk topics
- Contact information enforcement
- Response filtering and safety checks
"""

from typing import Dict, List, Optional, Tuple
from topic_classifier import TopicClassifier, TopicCategory, RiskLevel


class PolicyAwareResponder:
    """Modulates RAG responses based on topic risk level."""
    
    def __init__(self, classifier: TopicClassifier):
        """
        Initialize policy-aware responder.
        
        Args:
            classifier: Topic classifier instance
        """
        self.classifier = classifier
    
    def modulate_response(self, question: str, answer: str, sources: List[Dict],
                          classification: Optional[Dict] = None) -> Dict:
        """
        Apply policy-aware modulation to response.
        
        Args:
            question: Original question
            answer: Generated answer
            sources: Retrieved sources
            classification: Pre-computed classification (optional)
            
        Returns:
            Dictionary with modulated response and metadata
        """
        # Classify question if not provided
        if classification is None:
            classification = self.classifier.classify(question)
        
        category = classification['category_enum']
        risk_level = classification['risk_level_enum']
        
        # Get guidelines for this category
        guidelines = self.classifier.get_risk_guidelines(category)
        
        # Initialize modulated response
        modulated = {
            'original_answer': answer,
            'modified_answer': answer,
            'category': classification['category'],
            'risk_level': classification['risk_level'],
            'applied_modifications': [],
            'disclaimer': None,
            'contact_info': None,
            'requires_verification': False
        }
        
        # Apply risk-based modifications
        if risk_level == RiskLevel.LOW:
            # No modifications needed
            pass
        
        elif risk_level == RiskLevel.MEDIUM:
            modulated['requires_verification'] = True
            
            # Add disclaimer if required
            if guidelines['requires_disclaimer']:
                modulated['disclaimer'] = guidelines['disclaimer_text']
                modulated['applied_modifications'].append('Added verification disclaimer')
            
            # Add contact info if required
            if guidelines['requires_contact_info']:
                modulated['contact_info'] = guidelines['contact_info']
                modulated['applied_modifications'].append('Added contact information')
        
        elif risk_level == RiskLevel.HIGH:
            modulated['requires_verification'] = True
            
            # Always add disclaimer for high-risk
            modulated['disclaimer'] = guidelines['disclaimer_text']
            modulated['contact_info'] = guidelines['contact_info']
            modulated['applied_modifications'].append('Added critical safety disclaimer')
            modulated['applied_modifications'].append('Added emergency contact information')
            
            # Filter speculative language
            modified_answer = self._filter_speculation(answer)
            if modified_answer != answer:
                modulated['modified_answer'] = modified_answer
                modulated['applied_modifications'].append('Removed speculative language')
            
            # Ensure no "legal advice" tone
            if self._contains_legal_advice_language(answer):
                warning = (
                    "\n\n[Note: This is general information only and does not constitute "
                    "legal or professional advice. Please consult with appropriate officials.]"
                )
                modulated['modified_answer'] += warning
                modulated['applied_modifications'].append('Added legal disclaimer')
        
        return modulated
    
    def _filter_speculation(self, text: str) -> str:
        """
        Remove or soften speculative language.
        
        Args:
            text: Original text
            
        Returns:
            Filtered text
        """
        speculative_phrases = [
            ('you should', 'you may want to consider'),
            ('you must', 'you may need to'),
            ('definitely', 'typically'),
            ('always', 'usually'),
            ('never', 'rarely'),
        ]
        
        modified = text
        for original, replacement in speculative_phrases:
            modified = modified.replace(original, replacement)
        
        return modified
    
    def _contains_legal_advice_language(self, text: str) -> bool:
        """
        Check if text contains language that sounds like legal advice.
        
        Args:
            text: Text to check
            
        Returns:
            True if legal advice language detected
        """
        legal_phrases = [
            'you should hire',
            'you need a lawyer',
            'your legal rights',
            'sue',
            'lawsuit',
            'legal action',
        ]
        
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in legal_phrases)
    
    def generate_system_prompt_modifier(self, classification: Dict) -> str:
        """
        Generate system prompt modifications based on risk level.
        
        Args:
            classification: Classification result
            
        Returns:
            Additional system prompt instructions
        """
        risk_level = classification['risk_level_enum']
        category = classification['category_enum']
        
        base_prompt = ""
        
        if risk_level == RiskLevel.HIGH:
            base_prompt += (
                "\nIMPORTANT SAFETY GUIDELINES:\n"
                "- Only provide information directly from official sources\n"
                "- Use verbatim quotes from source documents when possible\n"
                "- Do NOT speculate or provide personal opinions\n"
                "- Do NOT provide legal, medical, or counseling advice\n"
                "- Always emphasize consulting with appropriate officials\n"
            )
            
            if category == TopicCategory.MENTAL_HEALTH:
                base_prompt += (
                    "- For crisis situations, immediately provide crisis hotline information\n"
                    "- Emphasize that this chatbot cannot replace professional counseling\n"
                    "- Direct users to CAPS (Counseling and Psychological Services)\n"
                )
            
            elif category == TopicCategory.TITLE_IX:
                base_prompt += (
                    "- Emphasize 911 for immediate danger\n"
                    "- Provide confidential reporting resources\n"
                    "- Do NOT attempt to investigate or judge situations\n"
                    "- Direct users to Title IX Office and VAWC\n"
                )
            
            elif category == TopicCategory.CONDUCT:
                base_prompt += (
                    "- Emphasize right to due process\n"
                    "- Direct to Office of Student Conduct\n"
                    "- Note that conduct matters can be complex\n"
                    "- Suggest consulting with advisor or legal counsel\n"
                )
        
        elif risk_level == RiskLevel.MEDIUM:
            base_prompt += (
                "\nVERIFICATION REMINDER:\n"
                "- Remind users to verify information with relevant offices\n"
                "- Note that policies may change and individual circumstances vary\n"
                "- Provide relevant office contact information\n"
            )
        
        return base_prompt
    
    def format_response_with_safety_features(self, modulated: Dict, 
                                             sources_html: str = "") -> str:
        """
        Format final response with all safety features.
        
        Args:
            modulated: Modulated response dictionary
            sources_html: HTML for source display
            
        Returns:
            Complete formatted response HTML
        """
        parts = []
        
        # Critical disclaimer at top for high-risk
        if modulated['risk_level'] == 'high' and modulated['disclaimer']:
            parts.append(f"""
            <div style="background: #dc3545; color: white; padding: 1rem; 
                        border-radius: 8px; margin-bottom: 1rem; border: 3px solid #a02a2a;">
                <div style="font-size: 1.2em; font-weight: bold; margin-bottom: 0.5rem;">
                    ⚠️ IMPORTANT SAFETY INFORMATION
                </div>
                <div style="white-space: pre-line;">
                    {modulated['disclaimer']}
                </div>
                {f'<div style="margin-top: 0.8rem; padding-top: 0.8rem; border-top: 1px solid rgba(255,255,255,0.3); font-weight: bold;">{modulated["contact_info"]}</div>' if modulated['contact_info'] else ''}
            </div>
            """)
        
        # Answer
        parts.append(f"""
        <div style="padding: 1rem; background: white; border-radius: 8px; margin-bottom: 1rem;">
            {modulated['modified_answer']}
        </div>
        """)
        
        # Medium-risk disclaimer at bottom
        if modulated['risk_level'] == 'medium' and modulated['disclaimer']:
            parts.append(f"""
            <div style="background: #fff3cd; border: 2px solid #ffc107; padding: 0.8rem; 
                        border-radius: 8px; margin-bottom: 1rem;">
                <div style="color: #856404;">
                    <strong>⚠️ Please Verify:</strong> {modulated['disclaimer']}
                </div>
                {f'<div style="margin-top: 0.5rem; color: #856404;"><strong>Contact:</strong> {modulated["contact_info"]}</div>' if modulated['contact_info'] else ''}
            </div>
            """)
        
        # Sources
        if sources_html:
            parts.append(sources_html)
        
        # Applied modifications (debug info)
        if modulated['applied_modifications']:
            mods_list = ", ".join(modulated['applied_modifications'])
            parts.append(f"""
            <div style="font-size: 0.75em; color: #999; margin-top: 1rem; padding: 0.5rem; 
                        background: #f8f9fa; border-radius: 4px;">
                🛡️ Policy-aware modifications: {mods_list}
            </div>
            """)
        
        return "\n".join(parts)


def create_risk_aware_rag_prompt(question: str, context: str, 
                                  classification: Dict, guidelines: Dict) -> str:
    """
    Create RAG prompt with risk-appropriate instructions.
    
    Args:
        question: User question
        context: Retrieved context
        classification: Classification result
        guidelines: Risk guidelines
        
    Returns:
        Modified prompt with safety instructions
    """
    risk_level = classification['risk_level']
    
    base_prompt = f"""You are a helpful assistant for Rutgers University students.

Question: {question}

Context from official Rutgers sources:
{context}

"""
    
    if risk_level == 'high':
        base_prompt += """
CRITICAL SAFETY INSTRUCTIONS:
- Only provide information DIRECTLY from the context above
- Use verbatim quotes when possible
- Do NOT speculate, give opinions, or provide advice beyond what's in the sources
- If the question requires professional guidance (legal, medical, counseling), state that clearly
- Emphasize that users should contact the appropriate office for their specific situation

"""
    
    elif risk_level == 'medium':
        base_prompt += """
VERIFICATION REMINDER:
- Base your answer on the provided context
- Note that policies may vary based on individual circumstances
- Remind users to verify details with the relevant office

"""
    
    base_prompt += "Answer:"
    
    return base_prompt


# Convenience function for integration
def apply_policy_aware_modulation(question: str, answer: str, sources: List[Dict],
                                   classifier: TopicClassifier) -> Dict:
    """
    One-stop function to apply all policy-aware features.
    
    Args:
        question: User question
        answer: Generated answer
        sources: Retrieved sources
        classifier: Topic classifier
        
    Returns:
        Fully modulated response
    """
    responder = PolicyAwareResponder(classifier)
    return responder.modulate_response(question, answer, sources)


if __name__ == "__main__":
    # Test the policy-aware responder
    from topic_classifier import TopicClassifier
    
    classifier = TopicClassifier()
    responder = PolicyAwareResponder(classifier)
    
    test_cases = [
        {
            'question': "How do I register for classes?",
            'answer': "To register, log into WebReg and select your courses."
        },
        {
            'question': "When will I get my financial aid refund?",
            'answer': "Refunds are typically processed 2-3 weeks after aid disburses. You should check your student account."
        },
        {
            'question': "I'm feeling suicidal, what should I do?",
            'answer': "It's important to reach out for help. Contact the counseling center."
        },
        {
            'question': "How do I report sexual harassment?",
            'answer': "You should contact the Title IX office to file a report."
        }
    ]
    
    print("\n" + "="*70)
    print("POLICY-AWARE RESPONSE MODULATION TEST")
    print("="*70)
    
    for test in test_cases:
        question = test['question']
        answer = test['answer']
        
        modulated = responder.modulate_response(question, answer, [])
        
        print(f"\n{'='*70}")
        print(f"Q: {question}")
        print(f"Category: {modulated['category']} | Risk: {modulated['risk_level']}")
        print(f"Modifications: {', '.join(modulated['applied_modifications']) if modulated['applied_modifications'] else 'None'}")
        
        if modulated['disclaimer']:
            print(f"\n📋 DISCLAIMER:\n{modulated['disclaimer'][:150]}...")
        
        if modulated['contact_info']:
            print(f"\n📞 CONTACT:\n{modulated['contact_info'][:150]}...")
