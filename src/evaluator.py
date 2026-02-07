"""
Evaluator for POS Tagging.

This module evaluates predicted POS tags against gold standard annotations.
"""

from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Set


def evaluate_predictions(test_sentences: List[List[Tuple[str, str]]], 
                        predictions: List[List[str]]) -> Dict:
    """
    Evaluate predictions against gold standard.
    
    Args:
        test_sentences: Test sentences with gold tags
        predictions: Predicted tag sequences
    
    Returns:
        Dictionary with evaluation metrics
    """
    total_tokens = 0
    correct_tokens = 0
    
    # Tag-level counts for confusion matrix
    all_gold_tags = []
    all_pred_tags = []
    
    for sentence, pred_tags in zip(test_sentences, predictions):
        gold_tags = [tag for _, tag in sentence]
        
        for gold, pred in zip(gold_tags, pred_tags):
            total_tokens += 1
            all_gold_tags.append(gold)
            all_pred_tags.append(pred)
            
            if gold == pred:
                correct_tokens += 1
    
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    
    return {
        'accuracy': accuracy,
        'total_tokens': total_tokens,
        'correct_tokens': correct_tokens,
        'gold_tags': all_gold_tags,
        'pred_tags': all_pred_tags
    }


def create_confusion_matrix(gold_tags: List[str], 
                           pred_tags: List[str], 
                           tagset: Set[str]) -> Dict:
    """
    Create confusion matrix.
    
    Returns:
        Dictionary with confusion counts
    """
    tags = sorted(list(tagset))
    confusion = {tag: Counter() for tag in tags}
    
    for gold, pred in zip(gold_tags, pred_tags):
        confusion[gold][pred] += 1
    
    return confusion


def print_confusion_matrix(confusion: Dict, top_n: int = 10):
    """
    Print top confusion pairs.
    """
    print("\nTop Confusions (Gold → Predicted):")
    print("-" * 50)
    
    confusions = []
    for gold_tag, pred_counts in confusion.items():
        for pred_tag, count in pred_counts.items():
            if gold_tag != pred_tag:  # Only errors
                confusions.append((count, gold_tag, pred_tag))
    
    # Sort by count and show top N
    confusions.sort(reverse=True)
    
    for count, gold, pred in confusions[:top_n]:
        print(f"  {gold:<8} → {pred:<8}  {count:>5} times")


def per_tag_accuracy(gold_tags: List[str], pred_tags: List[str]) -> Dict[str, Dict]:
    """
    Calculate per-tag precision, recall, and F1.
    """
    tag_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    for gold, pred in zip(gold_tags, pred_tags):
        if gold == pred:
            tag_stats[gold]['tp'] += 1
        else:
            tag_stats[gold]['fn'] += 1
            tag_stats[pred]['fp'] += 1
    
    results = {}
    for tag, stats in tag_stats.items():
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results[tag] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'count': tp + fn
        }
    
    return results


def print_evaluation_report(results: Dict, confusion: Dict, tag_metrics: Dict):
    """
    Print comprehensive evaluation report.
    """
    print("\n" + "=" * 60)
    print(" " * 20 + "EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nTotal tokens:     {results['total_tokens']:,}")
    print(f"Correct tokens:   {results['correct_tokens']:,}")
    print(f"Incorrect tokens: {results['total_tokens'] - results['correct_tokens']:,}")
    print(f"\n{'★' * 20}")
    print(f"ACCURACY: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"{'★' * 20}\n")
    
    # Confusion matrix
    print_confusion_matrix(confusion, top_n=15)
    
    # Per-tag metrics
    print("\n\nPer-Tag Performance:")
    print("=" * 70)
    print(f"{'Tag':<8} {'Count':>7} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 70)
    
    sorted_tags = sorted(tag_metrics.items(), key=lambda x: x[1]['count'], reverse=True)
    
    for tag, metrics in sorted_tags:
        print(f"{tag:<8} {metrics['count']:>7} {metrics['precision']:>10.4f} "
              f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f}")
    
    print("=" * 70)


def save_results(results: Dict, output_file: str):
    """
    Save results to file.
    """
    with open(output_file, 'w') as f:
        f.write("HMM POS Tagger - Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total tokens:     {results['total_tokens']:,}\n")
        f.write(f"Correct tokens:   {results['correct_tokens']:,}\n")
        f.write(f"Accuracy:         {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"✓ Results saved to {output_file}")


def save_predictions(test_sentences: List[List[Tuple[str, str]]], 
                    predictions: List[List[str]], 
                    output_file: str):
    """
    Save predictions to file.
    """
    with open(output_file, 'w') as f:
        for i, (sentence, pred_tags) in enumerate(zip(test_sentences, predictions)):
            f.write(f"Sentence {i+1}:\n")
            for (word, gold_tag), pred_tag in zip(sentence, pred_tags):
                match = "✓" if gold_tag == pred_tag else "✗"
                f.write(f"{word}\t{gold_tag}\t{pred_tag}\t{match}\n")
            f.write("\n")
    
    print(f"✓ Predictions saved to {output_file}")


if __name__ == "__main__":
    # Test the evaluator
    from data_parser import parse_conllu
    from hmm_trainer import HMMTagger
    from viterbi_decoder import decode_sentences
    import sys
    
    if len(sys.argv) > 2:
        model_file = sys.argv[1]
        test_file = sys.argv[2]
        
        # Load model
        hmm = HMMTagger()
        hmm.load_model(model_file)
        
        # Load test data
        test_sentences = parse_conllu(test_file)
        
        # Decode
        print("Decoding test set...")
        predictions = decode_sentences(test_sentences, hmm)
        
        # Evaluate
        print("\nEvaluating...")
        results = evaluate_predictions(test_sentences, predictions)
        confusion = create_confusion_matrix(results['gold_tags'], results['pred_tags'], hmm.tagset)
        tag_metrics = per_tag_accuracy(results['gold_tags'], results['pred_tags'])
        
        # Print report
        print_evaluation_report(results, confusion, tag_metrics)
