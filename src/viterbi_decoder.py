"""
Viterbi Decoder for POS Tagging.

This module implements the Viterbi algorithm with explicit matrix construction
for decoding the most probable POS tag sequence.
"""

import numpy as np
import math
from typing import List, Tuple
from hmm_trainer import HMMTagger


def viterbi_decode(words: List[str], hmm: HMMTagger) -> List[str]:
    """
    Decode POS tag sequence using Viterbi algorithm.
    
    Args:
        words: List of words in the sentence
        hmm: Trained HMM model
    
    Returns:
        List of predicted POS tags
    """
    if not words:
        return []
    
    T = len(words)  # Sentence length
    tags = sorted(list(hmm.tagset))  # List of tags
    N = len(tags)  # Number of tags
    
    # Create tag to index mapping
    tag_to_idx = {tag: i for i, tag in enumerate(tags)}
    idx_to_tag = {i: tag for i, tag in enumerate(tags)}
    
    # Initialize Viterbi matrix V[tag_idx][time] - stores log probabilities
    V = np.full((N, T), -np.inf)
    
    # Initialize Backpointer matrix B[tag_idx][time] - stores previous tag index
    B = np.zeros((N, T), dtype=int)
    
    # === STEP 1: Initialization (t=0) ===
    first_word = words[0]
    for tag in tags:
        tag_idx = tag_to_idx[tag]
        
        # V[tag][0] = log(P(tag)) + log(P(word[0] | tag))
        init_prob = hmm.initial_probs.get(tag, 1e-10)
        emit_prob = hmm.get_emission_prob(first_word, tag)
        
        V[tag_idx][0] = math.log(init_prob) + math.log(emit_prob)
        B[tag_idx][0] = -1  # No previous state
    
    # === STEP 2: Recursion (t=1 to T-1) ===
    for t in range(1, T):
        word = words[t]
        
        for curr_tag in tags:
            curr_idx = tag_to_idx[curr_tag]
            
            max_prob = -np.inf
            best_prev_idx = 0
            
            # Find best previous tag
            for prev_tag in tags:
                prev_idx = tag_to_idx[prev_tag]
                
                # Calculate: V[prev][t-1] + log(P(curr|prev)) + log(P(word|curr))
                trans_prob = hmm.transition_probs.get((prev_tag, curr_tag), 1e-10)
                emit_prob = hmm.get_emission_prob(word, curr_tag)
                
                prob = V[prev_idx][t-1] + math.log(trans_prob) + math.log(emit_prob)
                
                if prob > max_prob:
                    max_prob = prob
                    best_prev_idx = prev_idx
            
            V[curr_idx][t] = max_prob
            B[curr_idx][t] = best_prev_idx
    
    # === STEP 3: Termination - Find best final tag ===
    best_last_idx = np.argmax(V[:, T-1])
    
    # === STEP 4: Backtracking ===
    best_path = [0] * T
    best_path[T-1] = best_last_idx
    
    for t in range(T-2, -1, -1):
        best_path[t] = B[best_path[t+1]][t+1]
    
    # Convert indices back to tags
    predicted_tags = [idx_to_tag[idx] for idx in best_path]
    
    return predicted_tags


def decode_sentences(sentences: List[List[Tuple[str, str]]], 
                     hmm: HMMTagger) -> List[List[str]]:
    """
    Decode all sentences.
    
    Args:
        sentences: List of sentences with (word, tag) pairs
        hmm: Trained HMM model
    
    Returns:
        List of predicted tag sequences
    """
    predictions = []
    
    for i, sentence in enumerate(sentences):
        if i % 100 == 0:
            print(f"  Decoding sentence {i+1}/{len(sentences)}...", end='\r')
        
        words = [word for word, _ in sentence]
        predicted_tags = viterbi_decode(words, hmm)
        predictions.append(predicted_tags)
    
    print(f"  Decoding sentence {len(sentences)}/{len(sentences)}... Done!")
    return predictions