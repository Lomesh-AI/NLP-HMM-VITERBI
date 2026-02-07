"""
HMM Trainer for POS Tagging.

This module implements training of a Hidden Markov Model for POS tagging,
estimating initial, transition, and emission probabilities.
"""

import pickle
from collections import Counter
from typing import List, Tuple, Set, Dict


class HMMTagger:
    """
    Hidden Markov Model for POS tagging.
    """
    
    def __init__(self, smoothing=0.02):
        self.initial_probs = {}  # P(tag)
        self.transition_probs = {}  # P(tag_j | tag_i)
        self.emission_probs = {}  # P(word | tag)
        self.tagset = set()
        self.vocab = set()
        self.smoothing = smoothing  # Laplace smoothing parameter
        self.unk_threshold = 2  # Words appearing <= this are treated as rare
    
    def train(self, sentences: List[List[Tuple[str, str]]]):
        """
        Train HMM from annotated sentences.
        
        Args:
            sentences: List of sentences with (word, tag) pairs
        """
        print("Training HMM...")
        
        # Initialize counters
        initial_counts = Counter()
        transition_counts = Counter()
        emission_counts = Counter()
        tag_counts = Counter()
        word_counts = Counter()
        
        # Count statistics
        for sentence in sentences:
            if not sentence:
                continue
            
            # Initial tag (first word of sentence)
            _, first_tag = sentence[0]
            initial_counts[first_tag] += 1
            
            # Process each word in sentence
            for i, (word, tag) in enumerate(sentence):
                # Emission counts
                emission_counts[(word, tag)] += 1
                tag_counts[tag] += 1
                word_counts[word] += 1
                
                # Transition counts (except for last word)
                if i < len(sentence) - 1:
                    next_tag = sentence[i + 1][1]
                    transition_counts[(tag, next_tag)] += 1
        
        # Store tagset and vocabulary
        self.tagset = set(tag_counts.keys())
        self.vocab = set(word_counts.keys())
        
        # Handle rare words (for unknown word handling)
        rare_words = {word for word, count in word_counts.items() 
                      if count <= self.unk_threshold}
        
        print(f"  Found {len(rare_words)} rare words (will be treated as <UNK>)")
        
        # Calculate Initial Probabilities: P(tag)
        total_sentences = sum(initial_counts.values())
        for tag in self.tagset:
            count = initial_counts.get(tag, 0)
            self.initial_probs[tag] = (count + self.smoothing) / (total_sentences + self.smoothing * len(self.tagset))
        
        # Calculate Transition Probabilities: P(tag_j | tag_i)
        for tag_i in self.tagset:
            total_transitions = tag_counts[tag_i]
            for tag_j in self.tagset:
                count = transition_counts.get((tag_i, tag_j), 0)
                # Laplace smoothing
                self.transition_probs[(tag_i, tag_j)] = (
                    (count + self.smoothing) / 
                    (total_transitions + self.smoothing * len(self.tagset))
                )
        
        # Calculate Emission Probabilities: P(word | tag)
        # Replace rare words with <UNK>
        unk_emission_counts = Counter()
        for (word, tag), count in emission_counts.items():
            if word in rare_words:
                unk_emission_counts[('<UNK>', tag)] += count
            else:
                self.emission_probs[(word, tag)] = count
        
        # Add <UNK> emissions
        for (word, tag), count in unk_emission_counts.items():
            if (word, tag) in self.emission_probs:
                self.emission_probs[(word, tag)] += count
            else:
                self.emission_probs[(word, tag)] = count
        
        # Normalize emission probabilities
        for tag in self.tagset:
            tag_total = tag_counts[tag]
            vocab_size = len(self.vocab)
            
            # For each word in vocabulary
            for word in self.vocab:
                count = emission_counts.get((word, tag), 0)
                
                # Add smoothing
                prob = (count + self.smoothing) / (tag_total + self.smoothing * (vocab_size + 1))
                
                # Only store if non-negligible
                if prob > 1e-10:
                    self.emission_probs[(word, tag)] = prob
            
            # Add <UNK> for unknown words
            # Count rare words for this tag
            rare_count = sum(
                emission_counts.get((w, tag), 0) 
                for w in self.vocab 
                if word_counts[w] <= self.unk_threshold
            )
            
            unk_prob = (rare_count + self.smoothing) / (tag_total + self.smoothing * (vocab_size + 1))
            self.emission_probs[('<UNK>', tag)] = max(unk_prob, 1e-10)

        
        print(f"  Initial probabilities: {len(self.initial_probs)} tags")
        print(f"  Transition probabilities: {len(self.transition_probs)} pairs")
        print(f"  Emission probabilities: {len(self.emission_probs)} pairs")
        print("✓ Training complete")
    
    def get_emission_prob(self, word: str, tag: str) -> float:
        """Get emission probability with proper fallback."""
    
        # Try exact word
        if (word, tag) in self.emission_probs:
            return self.emission_probs[(word, tag)]
        
        # Unknown word - use <UNK>
        if ('<UNK>', tag) in self.emission_probs:
            return self.emission_probs[('<UNK>', tag)]
        
        # Ultimate fallback - uniform over all tags
        return self.smoothing / len(self.tagset)
        
    def save_model(self, filepath: str):
        """Save trained model to file."""
        model_data = {
            'initial_probs': self.initial_probs,
            'transition_probs': self.transition_probs,
            'emission_probs': self.emission_probs,
            'tagset': self.tagset,
            'vocab': self.vocab
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.initial_probs = model_data['initial_probs']
        self.transition_probs = model_data['transition_probs']
        self.emission_probs = model_data['emission_probs']
        self.tagset = model_data['tagset']
        self.vocab = model_data['vocab']
        print(f"✓ Model loaded from {filepath}")