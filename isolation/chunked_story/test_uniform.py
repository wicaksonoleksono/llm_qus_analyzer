#!/usr/bin/env python3
"""Simple test of Uniform Analyzer with Llama data."""

import sys
import json
from pathlib import Path

# Add root directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_qus_analyzer.set import UniformAnalyzer
from llm_qus_analyzer.chunker.models import QUSComponent
from llm_qus_analyzer.chunker.parser import Template

def create_qus_component(raw_comp):
    """Convert raw dictionary component to QUSComponent object."""
    template_data = raw_comp.get('template', {})
    
    template = Template(
        text=template_data.get('text', ''),
        chunk=template_data.get('chunk', {}),
        tail=template_data.get('tail'),
        order=template_data.get('order', [])
    )
    
    return QUSComponent(
        text=raw_comp.get('text', ''),
        role=raw_comp.get('role', []),
        means=raw_comp.get('means'),
        ends=raw_comp.get('ends'),
        template=template,
        id=raw_comp.get('id'),
        original_text=raw_comp.get('original_text', raw_comp.get('text', ''))
    )

def main():
    # Load Llama data
    data_file = Path(__file__).parent / "Llama 17B-Instruct FP8.json"
    gt_file = Path(__file__).parent / "ground_truth_extracted.json"
    
    with open(data_file, 'r') as f:
        raw_data = json.load(f)
    
    with open(gt_file, 'r') as f:
        ground_truth = json.load(f)
    
    print(f"Loaded {len(raw_data)} stories from Llama data")
    print(f"Loaded ground truth for {len(ground_truth)} stories")
    
    # Convert to QUSComponent objects
    components = []
    for item in raw_data:
        if 'component' in item:
            comp = create_qus_component(item['component'])
            components.append(comp)
    
    print(f"Created {len(components)} QUSComponent objects")
    
    # Run UniformAnalyzer
    print("\nRunning UniformAnalyzer...")
    results = UniformAnalyzer.run(None, 0, components)
    
    # Calculate stats against ground truth
    tp = fp = fn = tn = 0
    detailed_results = []
    
    for i, (comp, (violations, _)) in enumerate(zip(components, results)):
        story_text = comp.original_text
        has_violations = len(violations) > 0
        
        # Check ground truth
        gt_tags = ground_truth.get(story_text, [])
        has_uniform_gt = "Uniform" in gt_tags
        
        # Classify result
        if has_uniform_gt and has_violations:
            result_type = "TRUE_POSITIVE"
            tp += 1
        elif not has_uniform_gt and not has_violations:
            result_type = "TRUE_NEGATIVE"
            tn += 1
        elif has_violations and not has_uniform_gt:
            result_type = "FALSE_POSITIVE"
            fp += 1
        else:  # has_uniform_gt and not has_violations
            result_type = "FALSE_NEGATIVE"
            fn += 1
        
        detailed_results.append({
            "story_id": i + 1,
            "original_text": story_text,
            "template": comp.template.text,
            "has_violations": has_violations,
            "ground_truth_uniform": has_uniform_gt,
            "result_type": result_type,
            "violations": [{"issue": v.issue, "suggestion": v.suggestion} for v in violations]
        })
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Show results
    print(f"\nUNIFORM ANALYSIS RESULTS")
    print("=" * 50)
    
    for result in detailed_results:
        if result["has_violations"]:
            print(f"\nStory {result['story_id']}: {result['original_text'][:60]}...")
            print(f"Template: {result['template']}")
            print(f"Result: {result['result_type']}")
            for violation in result["violations"]:
                print(f"VIOLATION: {violation['issue']}")
    
    print(f"\nSTATISTICS:")
    print("=" * 30)
    print(f"Total stories: {len(components)}")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Negatives: {tn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    # Save results to file
    output_data = {
        "total_stories": len(components),
        "statistics": {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        },
        "detailed_results": detailed_results
    }
    
    output_file = Path(__file__).parent / "uniform_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()