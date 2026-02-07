"""
Data parser for CoNLL-U format files.

This module handles reading and parsing of Universal Dependencies CoNLL-U files.
"""

from typing import List, Tuple, Set


def parse_conllu(filepath: str) -> List[List[Tuple[str, str]]]:
    """
    Parse a CoNLL-U file and extract sentences with (word, UPOS tag) pairs.
    
    Args:
        filepath: Path to .conllu file
    
    Returns:
        List of sentences, where each sentence is a list of (word, tag) tuples
    """
    sentences = []
    current_sentence = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Empty line marks end of sentence
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue
            
            # Skip comment lines
            if line.startswith('#'):
                continue
            
            # Parse token line
            parts = line.split('\t')
            if len(parts) >= 4:
                word_id = parts[0]
                
                # Skip multiword tokens and empty nodes
                if '-' in word_id or '.' in word_id:
                    continue
                
                word = parts[1].lower()  # Lowercase for better generalization
                upos_tag = parts[3]
                
                current_sentence.append((word, upos_tag))
        
        # Add last sentence if exists
        if current_sentence:
            sentences.append(current_sentence)
    
    return sentences


def get_vocabulary(sentences: List[List[Tuple[str, str]]]) -> Set[str]:
    """Extract vocabulary from sentences."""
    vocab = set()
    for sentence in sentences:
        for word, _ in sentence:
            vocab.add(word)
    return vocab


def get_tagset(sentences: List[List[Tuple[str, str]]]) -> Set[str]:
    """Extract unique POS tags from sentences."""
    tags = set()
    for sentence in sentences:
        for _, tag in sentence:
            tags.add(tag)
    return tags


if __name__ == "__main__":
    # Test the parser
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        sentences = parse_conllu(filepath)
        print(f"Loaded {len(sentences)} sentences")
        print(f"Vocabulary: {len(get_vocabulary(sentences))} words")
        print(f"Tagset: {sorted(get_tagset(sentences))}")
        
        # Show first sentence
        if sentences:
            print("\nFirst sentence:")
            for word, tag in sentences[0]:
                print(f"  {word:<15} → {tag}")
