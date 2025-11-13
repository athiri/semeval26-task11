"""Basic text features"""

def extract_basic_features(text: str) -> dict:
    """Extract basic statistical features"""
    words = text.split()
    sentences = [s for s in text.split('.') if s.strip()]
    
    return {
        'text_length': len(text),
        'num_words': len(words),
        'num_sentences': len(sentences),
        'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
        'num_uppercase': sum(1 for c in text if c.isupper()),
        'num_punctuation': sum(1 for c in text if c in '.,;:!?'),
    }
