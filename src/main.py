"""
Main script to run the complete HMM POS tagging pipeline.

Usage:
    python main.py --train data/en_ewt-ud-train.conllu --test data/en_ewt-ud-test.conllu
"""

import os
import argparse
from data_parser import parse_conllu, get_tagset
from hmm_trainer import HMMTagger
from viterbi_decoder import decode_sentences
from evaluator import (
    evaluate_predictions, 
    create_confusion_matrix, 
    per_tag_accuracy,
    print_evaluation_report,
    save_results,
    save_predictions,
    print_confusion_matrix
)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='HMM-based POS Tagger')
    parser.add_argument('--train', type=str, required=True, 
                       help='Path to training .conllu file')
    parser.add_argument('--test', type=str, required=True, 
                       help='Path to test .conllu file')
    parser.add_argument('--model', type=str, default='models/hmm_model.pkl', 
                       help='Path to save/load model')
    parser.add_argument('--output-dir', type=str, default='output', 
                       help='Directory for output files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    
    print("=" * 60)
    print(" " * 15 + "HMM POS TAGGER")
    print("=" * 60)
    
    # === STEP 1: Parse Training Data ===
    print("\n[1/5] Parsing training data...")
    train_sentences = parse_conllu(args.train)
    print(f"✓ Loaded {len(train_sentences)} training sentences")
    
    # === STEP 2: Train HMM ===
    print("\n[2/5] Training HMM...")
    hmm = HMMTagger()
    hmm.train(train_sentences)
    hmm.save_model(args.model)
    
    # === STEP 3: Parse Test Data ===
    print("\n[3/5] Parsing test data...")
    test_sentences = parse_conllu(args.test)
    print(f"✓ Loaded {len(test_sentences)} test sentences")
    
    # === STEP 4: Decode Test Set ===
    print("\n[4/5] Decoding test set with Viterbi...")
    predictions = decode_sentences(test_sentences, hmm)
    
    # === STEP 5: Evaluate ===
    print("\n[5/5] Evaluating predictions...")
    results = evaluate_predictions(test_sentences, predictions)
    confusion = create_confusion_matrix(results['gold_tags'], results['pred_tags'], hmm.tagset)
    tag_metrics = per_tag_accuracy(results['gold_tags'], results['pred_tags'])
    
    # Print comprehensive report
    print_evaluation_report(results, confusion, tag_metrics)
    print_confusion_matrix(confusion)
    
    # Save results
    results_file = os.path.join(args.output_dir, 'results.txt')
    predictions_file = os.path.join(args.output_dir, 'predictions.txt')
    
    save_results(results, results_file)
    save_predictions(test_sentences, predictions, predictions_file)
    
    print("\n" + "=" * 60)
    print("COMPLETED SUCCESSFULLY!")
    print(f"Model saved: {args.model}")
    print(f"Results saved: {results_file}")
    print(f"Predictions saved: {predictions_file}")
    print("=" * 60 + "\n")
    
    return results['accuracy']


if __name__ == "__main__":
    accuracy = main()
