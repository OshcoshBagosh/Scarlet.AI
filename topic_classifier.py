"""
Topic Classifier for Policy-Aware Routing in Scarlet.AI
Classifies questions into categories to enable risk-aware behavior.

Categories:
- academics: Course registration, schedules, requirements
- housing: Dorms, meal plans, housing policies
- financial: Tuition, financial aid, refunds
- conduct: Student conduct, disciplinary issues
- mental_health: Counseling, wellness, mental health resources
- title_ix: Title IX, sexual misconduct, safety
- health: Medical services, insurance, health policies
- general: General campus information
"""

from typing import Dict, List, Tuple
from enum import Enum


class TopicCategory(Enum):
    """Topic categories for question classification."""
    ACADEMICS = "academics"
    HOUSING = "housing"
    FINANCIAL = "financial"
    CONDUCT = "conduct"
    MENTAL_HEALTH = "mental_health"
    TITLE_IX = "title_ix"
    HEALTH = "health"
    GENERAL = "general"


class RiskLevel(Enum):
    """Risk levels for different topic categories."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Topic-specific keywords for rule-based classification
TOPIC_KEYWORDS = {
    TopicCategory.ACADEMICS: [
        'register', 'registration', 'course', 'class', 'schedule', 'professor',
        'major', 'minor', 'degree', 'credit', 'transcript', 'grade', 'gpa',
        'exam', 'final', 'midterm', 'assignment', 'syllabus', 'withdraw',
        'drop', 'add', 'enrollment', 'waitlist', 'prerequisite', 'graduation',
        'diploma', 'advising', 'advisor', 'academic', 'study abroad',
        'transfer credit', 'webreg', 'soc', 'calendar'
    ],
    TopicCategory.HOUSING: [
        'housing', 'dorm', 'residence', 'dormitory', 'room', 'roommate',
        'meal plan', 'dining', 'food', 'cafeteria', 'dining hall',
        'housing assignment', 'move in', 'move out', 'housing cancellation',
        'ru express', 'meal swipe', 'housing application', 'on campus',
        'off campus', 'apartment', 'housing lottery'
    ],
    TopicCategory.FINANCIAL: [
        'tuition', 'financial aid', 'fafsa', 'scholarship', 'grant', 'loan',
        'refund', 'payment', 'bill', 'fee', 'cost', 'budget', 'afford',
        'money', 'pay', 'debt', 'work study', 'student account', 'bursar',
        'direct deposit', 'payment plan', 'financial', 'aid package',
        'verification', 'disbursement', 'balance', 'charge'
    ],
    TopicCategory.CONDUCT: [
        'student conduct', 'discipline', 'violation', 'policy violation',
        'academic integrity', 'cheating', 'plagiarism', 'honor code',
        'misconduct', 'sanction', 'hearing', 'appeal', 'code of conduct',
        'judicial', 'disciplinary action', 'probation', 'suspension',
        'expulsion', 'citation', 'complaint', 'investigation'
    ],
    TopicCategory.MENTAL_HEALTH: [
        'mental health', 'counseling', 'therapy', 'therapist', 'depression',
        'anxiety', 'stress', 'suicide', 'crisis', 'caps', 'psychological',
        'wellness', 'emotional', 'mental crisis', 'feeling depressed',
        'self harm', 'suicidal', 'mental illness', 'psychiatrist',
        'counselor', 'support group', 'mental wellness', 'struggling'
    ],
    TopicCategory.TITLE_IX: [
        'title ix', 'sexual assault', 'sexual harassment', 'sexual misconduct',
        'rape', 'consent', 'dating violence', 'domestic violence',
        'stalking', 'gender discrimination', 'sex discrimination',
        'sexual violence', 'harassment', 'assault', 'survivor', 'victim',
        'title 9', 'sexual abuse', 'unwanted contact', 'restraining order',
        'inappropriate', 'unwelcome', 'report a concern', 'report concern',
        'report incident', 'touching', 'unwanted comment', 'inappropriate comment'
    ],
    TopicCategory.HEALTH: [
        'health', 'medical', 'doctor', 'clinic', 'insurance', 'ship',
        'student health', 'immunization', 'vaccine', 'medication',
        'prescription', 'physical', 'sick', 'illness', 'injury',
        'health services', 'pharmacy', 'medical records', 'health center',
        'urgent care', 'covid', 'flu shot', 'health insurance'
    ],
    TopicCategory.GENERAL: [
        'campus', 'location', 'hours', 'contact', 'phone', 'email',
        'office', 'building', 'map', 'directions', 'parking', 'bus',
        'transportation', 'rutgers', 'new brunswick', 'library',
        'gym', 'recreation', 'club', 'organization', 'event'
    ]
}


# Risk tier mapping
RISK_TIERS = {
    TopicCategory.ACADEMICS: RiskLevel.LOW,
    TopicCategory.HOUSING: RiskLevel.MEDIUM,
    TopicCategory.FINANCIAL: RiskLevel.MEDIUM,
    TopicCategory.CONDUCT: RiskLevel.HIGH,
    TopicCategory.MENTAL_HEALTH: RiskLevel.HIGH,
    TopicCategory.TITLE_IX: RiskLevel.HIGH,
    TopicCategory.HEALTH: RiskLevel.MEDIUM,
    TopicCategory.GENERAL: RiskLevel.LOW,
}


class TopicClassifier:
    """Classifies questions into topic categories for policy-aware routing."""
    
    def __init__(self, llm_client=None, model: str = "llama3.2:3b"):
        """
        Initialize topic classifier.
        
        Args:
            llm_client: Optional LLM client for advanced classification
            model: Model name to use for LLM-based classification
        """
        self.llm_client = llm_client
        self.model = model
    
    def classify_keyword_based(self, question: str) -> Tuple[TopicCategory, float]:
        """
        Classify question using keyword matching.
        
        Args:
            question: User's question
            
        Returns:
            Tuple of (category, confidence_score)
        """
        question_lower = question.lower()
        
        # Count keyword matches for each category
        scores = {}
        for category, keywords in TOPIC_KEYWORDS.items():
            matches = sum(1 for keyword in keywords if keyword in question_lower)
            scores[category] = matches
        
        # Heuristic boosts for high-risk indicators
        high_risk_triggers = [
            'harass', 'assault', 'rape', 'stalking', 'violence', 'suicide',
            'self harm', 'crisis', 'threat', 'bias', 'discrimination', 'inappropriate',
            'unwelcome', 'report a concern', 'report concern', 'complaint',
        ]

        if any(trig in question_lower for trig in high_risk_triggers):
            # Boost TITLE_IX, CONDUCT, MENTAL_HEALTH slightly so they win close ties
            for cat in [TopicCategory.TITLE_IX, TopicCategory.CONDUCT, TopicCategory.MENTAL_HEALTH]:
                scores[cat] = scores.get(cat, 0) + 2

        # Specific case: professor + inappropriate → likely Title IX or Conduct
        if 'professor' in question_lower and 'inappropriate' in question_lower:
            scores[TopicCategory.TITLE_IX] = scores.get(TopicCategory.TITLE_IX, 0) + 2
            scores[TopicCategory.CONDUCT] = scores.get(TopicCategory.CONDUCT, 0) + 1

        # Find category with highest score
        if not scores or max(scores.values()) == 0:
            return TopicCategory.GENERAL, 0.3
        
        max_score = max(scores.values())
        # If no positive matches, default to GENERAL (prevents high-risk false positives)
        if max_score == 0:
            return TopicCategory.GENERAL, 0.3

        # Only consider categories with the top positive score (strict tie set)
        top_candidates = [cat for cat, sc in scores.items() if sc == max_score]
        if len(top_candidates) == 1:
            best_category = top_candidates[0]
        else:
            # Tie: pick the candidate whose risk tier is highest ONLY if its score >1 (more semantic weight)
            # Otherwise prefer lower risk to avoid unnecessary warnings on ambiguous queries
            risk_rank = {RiskLevel.HIGH: 3, RiskLevel.MEDIUM: 2, RiskLevel.LOW: 1}
            if max_score > 1:
                top_candidates.sort(key=lambda c: risk_rank[RISK_TIERS.get(c, RiskLevel.LOW)], reverse=True)
            else:
                top_candidates.sort(key=lambda c: risk_rank[RISK_TIERS.get(c, RiskLevel.LOW)])  # prefer lower risk when weak signal
            best_category = top_candidates[0]

        # Calculate confidence: base on score with slight boost for high-risk trigger presence
        confidence_base = min(1.0, scores[best_category] / 3)  # 3+ matches = high confidence
        if scores[best_category] == 1 and RISK_TIERS.get(best_category) == RiskLevel.HIGH:
            # Single keyword match in high-risk domain: keep modest confidence
            confidence = min(confidence_base, 0.6)
        elif any(trig in question_lower for trig in high_risk_triggers) and RISK_TIERS.get(best_category) == RiskLevel.HIGH:
            confidence = min(1.0, confidence_base + 0.1)
        else:
            confidence = confidence_base
        
        return best_category, confidence
    
    def classify_llm_based(self, question: str) -> Tuple[TopicCategory, float]:
        """
        Classify question using LLM.
        
        Args:
            question: User's question
            
        Returns:
            Tuple of (category, confidence_score)
        """
        if not self.llm_client:
            return self.classify_keyword_based(question)
        
        # Create classification prompt
        categories_list = ", ".join([cat.value for cat in TopicCategory])
        
        prompt = f"""Classify this student question into ONE category: {categories_list}

Question: "{question}"

Respond with ONLY the category name, nothing else.
Category:"""
        
        try:
            response = self.llm_client.generate(model=self.model, prompt=prompt, 
                                               options={'temperature': 0.1})
            response_text = response.get('response', '').strip().lower()
            
            # Parse category from response
            for category in TopicCategory:
                if category.value in response_text:
                    return category, 0.85  # High confidence for LLM
            
            # Fallback to keyword-based if LLM fails
            return self.classify_keyword_based(question)
            
        except Exception as e:
            print(f"LLM classification failed: {e}")
            return self.classify_keyword_based(question)
    
    def classify(self, question: str, use_llm: bool = False) -> Dict:
        """
        Classify question and return full classification result.
        
        Args:
            question: User's question
            use_llm: Whether to use LLM-based classification
            
        Returns:
            Dictionary with classification results
        """
        # Get classification
        if use_llm and self.llm_client:
            category, confidence = self.classify_llm_based(question)
        else:
            category, confidence = self.classify_keyword_based(question)
        
        # Get risk level
        risk_level = RISK_TIERS.get(category, RiskLevel.LOW)
        
        return {
            'category': category.value,
            'category_enum': category,
            'confidence': confidence,
            'risk_level': risk_level.value,
            'risk_level_enum': risk_level,
        }
    
    def get_risk_guidelines(self, category: TopicCategory) -> Dict[str, str]:
        """
        Get risk-specific guidelines for a category.
        
        Args:
            category: Topic category
            
        Returns:
            Dictionary with guidelines
        """
        risk_level = RISK_TIERS.get(category, RiskLevel.LOW)
        
        guidelines = {
            'risk_level': risk_level.value,
            'requires_disclaimer': False,
            'requires_verbatim': False,
            'requires_contact_info': False,
            'disclaimer_text': '',
            'contact_info': ''
        }
        
        # Low risk: normal behavior
        if risk_level == RiskLevel.LOW:
            return guidelines
        
        # Medium risk: add extra verification
        elif risk_level == RiskLevel.MEDIUM:
            guidelines['requires_disclaimer'] = True
            
            if category == TopicCategory.HOUSING:
                guidelines['disclaimer_text'] = (
                    "Please verify housing policies and deadlines with the Housing Office. "
                    "Policies may change, and individual circumstances may affect your situation."
                )
                guidelines['contact_info'] = "Housing Office: https://ruoncampus.rutgers.edu/"
                
            elif category == TopicCategory.FINANCIAL:
                guidelines['disclaimer_text'] = (
                    "Financial aid information can be complex and varies by individual circumstance. "
                    "Please consult with the Financial Aid Office for personalized guidance."
                )
                guidelines['contact_info'] = "Financial Aid: https://scarlethub.rutgers.edu/financial-services/"
                guidelines['requires_contact_info'] = True
                
            elif category == TopicCategory.HEALTH:
                guidelines['disclaimer_text'] = (
                    "This is general information only and does not constitute medical advice. "
                    "Please contact Student Health Services for medical concerns."
                )
                guidelines['contact_info'] = "Student Health: https://health.rutgers.edu/"
                guidelines['requires_contact_info'] = True
        
        # High risk: strict controls
        elif risk_level == RiskLevel.HIGH:
            guidelines['requires_disclaimer'] = True
            guidelines['requires_verbatim'] = True
            guidelines['requires_contact_info'] = True
            
            if category == TopicCategory.CONDUCT:
                guidelines['disclaimer_text'] = (
                    "⚠️ IMPORTANT: This information is for general reference only and does not "
                    "constitute legal advice. Student conduct matters can have serious consequences. "
                    "For specific situations, please contact the Office of Student Conduct directly."
                )
                guidelines['contact_info'] = (
                    "Office of Student Conduct: https://studentconduct.rutgers.edu/ | "
                    "Phone: 848-932-9414"
                )
                
            elif category == TopicCategory.MENTAL_HEALTH:
                guidelines['disclaimer_text'] = (
                    "🆘 CRISIS RESOURCES: If you are in crisis or having thoughts of self-harm, "
                    "please reach out immediately:\n"
                    "• CAPS Crisis Services: 848-932-7884 (24/7)\n"
                    "• National Suicide Prevention Lifeline: 988\n"
                    "• Crisis Text Line: Text 'HELLO' to 741741\n\n"
                    "This chatbot cannot provide counseling or mental health treatment. "
                    "Please contact CAPS for professional support."
                )
                guidelines['contact_info'] = (
                    "CAPS (Counseling and Psychological Services): https://health.rutgers.edu/caps/ | "
                    "Phone: 848-932-7884"
                )
                
            elif category == TopicCategory.TITLE_IX:
                guidelines['disclaimer_text'] = (
                    "⚠️ IMPORTANT SAFETY INFORMATION: If you are in immediate danger, call 911. "
                    "Title IX and sexual misconduct matters require specialized support and resources.\n\n"
                    "This information is general and does not replace official guidance. "
                    "For confidential support and to report incidents, please contact:"
                )
                guidelines['contact_info'] = (
                    "Title IX Office: titleix@rutgers.edu | Phone: 848-932-8200\n"
                    "VAWC (Violence Prevention & Victim Assistance): 848-932-1181 (24/7)"
                )
        
        return guidelines


def test_classifier():
    """Test the topic classifier with sample questions."""
    classifier = TopicClassifier()
    
    test_questions = [
        "How do I register for classes?",
        "What are the housing cancellation policies?",
        "When will I get my financial aid refund?",
        "I was charged with plagiarism, what should I do?",
        "I'm feeling really depressed and need help",
        "How do I report sexual harassment?",
        "What dining options are on campus?",
        "How much does tuition cost?",
    ]
    
    print("\n" + "="*70)
    print("TOPIC CLASSIFIER TEST")
    print("="*70)
    
    for question in test_questions:
        result = classifier.classify(question)
        guidelines = classifier.get_risk_guidelines(result['category_enum'])
        
        print(f"\nQ: {question}")
        print(f"   Category: {result['category']} (confidence: {result['confidence']:.2f})")
        print(f"   Risk: {result['risk_level']}")
        
        if guidelines['requires_disclaimer']:
            print(f"   ⚠️  Requires disclaimer")
        if guidelines['requires_verbatim']:
            print(f"   📝 Requires verbatim quotes")
        if guidelines['requires_contact_info']:
            print(f"   📞 Requires contact info")


if __name__ == "__main__":
    test_classifier()
