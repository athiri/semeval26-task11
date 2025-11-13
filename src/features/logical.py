"""Logical structure features"""

def extract_logical_features(text: str) -> dict:
    """Extract logical structure features"""
    text_lower = text.lower()
    
    return {
        'count_all': text_lower.count('all '),
        'count_some': text_lower.count('some '),
        'count_no': text_lower.count('no '),
        'count_not': text_lower.count('not '),
        'has_therefore': 1 if 'therefore' in text_lower else 0,
        'has_thus': 1 if 'thus' in text_lower else 0,
        'count_is': text_lower.count(' is '),
        'count_are': text_lower.count(' are '),
        'has_negation': 1 if any(n in text_lower for n in ['not', 'no ', "n't"]) else 0,
    }
