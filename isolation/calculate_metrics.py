import json
from collections import defaultdict, Counter

def flatten_labels(data_dict):
    """Flatten all labels from all stories into a list of individual label instances."""
    all_labels = []
    for story_text, labels in data_dict.items():
        if labels:  # Skip empty label lists
            all_labels.extend(labels)
    return all_labels

def calculate_per_label_metrics(ground_truth, model_violations, label):
    """Calculate metrics for a specific label/criterion."""
    tp = fp = fn = 0
    
    all_stories = set(ground_truth.keys()) | set(model_violations.keys())
    
    for story_text in all_stories:
        gt_labels = set(ground_truth.get(story_text, []))
        pred_labels = set(model_violations.get(story_text, []))
        
        # Check for this specific label
        gt_has = label in gt_labels
        pred_has = label in pred_labels
        
        if gt_has and pred_has:
            tp += 1
        elif pred_has and not gt_has:
            fp += 1
        elif gt_has and not pred_has:
            fn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def calculate_model_metrics(ground_truth, model_violations):
    """Calculate overall metrics for a model by comparing predicted vs GT labels per story."""
    
    total_tp = total_fp = total_fn = 0
    story_details = []
    
    # Get all stories from both GT and predictions
    all_stories = set(ground_truth.keys()) | set(model_violations.keys())
    
    for story_text in all_stories:
        gt_labels = set(ground_truth.get(story_text, []))
        pred_labels = set(model_violations.get(story_text, []))
        
        # Calculate TP, FP, FN for this story
        tp = len(gt_labels & pred_labels)  # Labels in both GT and prediction
        fp = len(pred_labels - gt_labels)  # Predicted but not in GT
        fn = len(gt_labels - pred_labels)  # In GT but not predicted
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        story_details.append({
            'story': story_text,  # Keep full story text
            'gt_labels': list(gt_labels),
            'pred_labels': list(pred_labels),
            'tp': tp,
            'fp': fp,
            'fn': fn
        })
    
    # Overall metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    # Per-label metrics
    all_labels = set()
    for labels in ground_truth.values():
        all_labels.update(labels)
    for labels in model_violations.values():
        all_labels.update(labels)
    
    per_label_metrics = {}
    for label in all_labels:
        per_label_metrics[label] = calculate_per_label_metrics(ground_truth, model_violations, label)
    
    # Count total labels
    total_gt_labels = sum(len(labels) for labels in ground_truth.values())
    total_pred_labels = sum(len(labels) for labels in model_violations.values())
    
    return {
        'total_stories': len(all_stories),
        'total_gt_labels': total_gt_labels,
        'total_pred_labels': total_pred_labels,
        'total_tp': total_tp,
        'total_fp': total_fp, 
        'total_fn': total_fn,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1': overall_f1,
        'per_label_metrics': per_label_metrics,
        'story_details': story_details
    }

def main():
    # Load ground truth and collected violations
    with open('ground_truth_extracted.json', 'r') as f:
        ground_truth = json.load(f)
    
    with open('violations_collected.json', 'r') as f:
        violations_per_model = json.load(f)
    
    # Flatten ground truth labels
    gt_labels = flatten_labels(ground_truth)
    
    print("LABEL-BASED EVALUATION METRICS")
    print("=" * 60)
    print(f"Total Ground Truth Labels: {len(gt_labels)}")
    print(f"GT Label Counts: {dict(Counter(gt_labels))}")
    print()
    
    results = {}
    
    for model_name, model_violations in violations_per_model.items():
        print(f"\n{model_name.upper()}")
        print("-" * 40)
        
        # Flatten predicted labels
        pred_labels = flatten_labels(model_violations)
        
        # Calculate model metrics
        metrics = calculate_model_metrics(ground_truth, model_violations)
        results[model_name] = metrics
        
        print(f"Total Stories: {metrics['total_stories']}")
        print(f"Total GT Labels: {metrics['total_gt_labels']}")
        print(f"Total Predicted Labels: {metrics['total_pred_labels']}")
        print(f"Predicted Label Counts: {dict(Counter(pred_labels))}")
        print()
        print(f"Overall Precision: {metrics['overall_precision']:.3f}")
        print(f"Overall Recall: {metrics['overall_recall']:.3f}")
        print(f"Overall F1: {metrics['overall_f1']:.3f}")
        print(f"Total TP: {metrics['total_tp']}, FP: {metrics['total_fp']}, FN: {metrics['total_fn']}")
        
        # Per-label metrics
        print(f"\nPer-Label Metrics:")
        print(f"{'Label':<12} {'Precision':<10} {'Recall':<8} {'F1':<8} {'TP':<4} {'FP':<4} {'FN':<4}")
        print("-" * 50)
        for label, label_metrics in metrics['per_label_metrics'].items():
            print(f"{label:<12} {label_metrics['precision']:.3f}      {label_metrics['recall']:.3f}    "
                  f"{label_metrics['f1']:.3f}    {label_metrics['tp']:<4} {label_metrics['fp']:<4} {label_metrics['fn']:<4}")
        
        # Show some story examples
        print(f"\nExample story results:")
        for story_detail in metrics['story_details'][:3]:  # Show first 3 stories
            if story_detail['tp'] > 0 or story_detail['fp'] > 0 or story_detail['fn'] > 0:
                print(f"  Story: {story_detail['story'][:60]}...")
                print(f"    GT: {story_detail['gt_labels']}, Pred: {story_detail['pred_labels']}")
                print(f"    TP: {story_detail['tp']}, FP: {story_detail['fp']}, FN: {story_detail['fn']}")
    
    # Create clean JSON stats for each LLM
    llm_stats = {}
    for model_name, metrics in results.items():
        per_label_stats = {}
        for label, label_metrics in metrics['per_label_metrics'].items():
            per_label_stats[label] = {
                "precision": round(label_metrics['precision'], 3),
                "recall": round(label_metrics['recall'], 3),
                "f1_score": round(label_metrics['f1'], 3),
                "true_positives": label_metrics['tp'],
                "false_positives": label_metrics['fp'],
                "false_negatives": label_metrics['fn']
            }
        
        llm_stats[model_name] = {
            "total_stories": metrics['total_stories'],
            "total_gt_labels": metrics['total_gt_labels'],
            "total_pred_labels": metrics['total_pred_labels'],
            "true_positives": metrics['total_tp'],
            "false_positives": metrics['total_fp'],
            "false_negatives": metrics['total_fn'],
            "precision": round(metrics['overall_precision'], 3),
            "recall": round(metrics['overall_recall'], 3),
            "f1_score": round(metrics['overall_f1'], 3),
            "per_label_metrics": per_label_stats
        }
    
    # Save LLM stats as JSON
    with open('llm_stats.json', 'w') as f:
        json.dump(llm_stats, f, indent=2)
    
    # Save detailed results
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Model comparison summary
    print(f"\n{'MODEL COMPARISON'}")
    print("=" * 70)
    print(f"{'Model':<25} {'F1':<8} {'Precision':<10} {'Recall':<8} {'TP':<4} {'FP':<4} {'FN':<4}")
    print("-" * 70)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<25} {metrics['overall_f1']:.3f}    {metrics['overall_precision']:.3f}      "
              f"{metrics['overall_recall']:.3f}    {metrics['total_tp']:<4} {metrics['total_fp']:<4} {metrics['total_fn']:<4}")
    
    print(f"\nLLM stats saved to 'llm_stats.json'")
    print(f"Detailed results saved to 'evaluation_results.json'")

if __name__ == "__main__":
    main()