#!/usr/bin/env python3
"""Test Uniform Analyzer using the actual UniformAnalyzer class."""

import sys
import json
import csv
from pathlib import Path
from collections import Counter

# Add root directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_qus_analyzer.set import UniformAnalyzer
from llm_qus_analyzer import Settings, LLMClient
from llm_qus_analyzer.chunker.models import QUSComponent
from llm_qus_analyzer.chunker.parser import Template

def load_chunked_with_templates():
    """Load chunked data with built templates."""
    chunk_file = Path(__file__).parent / "chunked_story" / "chunked_with_built_templates.json"
    
    if not chunk_file.exists():
        raise FileNotFoundError(f"Template data not found at {chunk_file}")
    
    with open(chunk_file, 'r') as f:
        raw_chunks = json.load(f)
    
    # Extract components with real templates
    components = []
    for item in raw_chunks:
        if 'component' in item:
            comp_data = item['component']
            components.append(comp_data)
    
    print(f"Loaded {len(components)} components with built templates")
    return components

def load_ground_truth_violations():
    """Load ground truth violation labels from JSON file."""
    gt_file = Path(__file__).parent / "ground_truth_extracted.json"
    
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth file not found at {gt_file}")
    
    with open(gt_file, 'r') as f:
        gt_violations = json.load(f)
    
    print(f"Loaded ground truth violations for {len(gt_violations)} stories")
    return gt_violations

def create_qus_component(raw_comp):
    """Convert raw dictionary component to QUSComponent object with real Template."""
    # Extract template data from the raw component
    template_data = raw_comp.get('template', {})
    
    # Create a proper template with the actual parsed data
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

def test_uniform_analyzer():
    """Test the actual UniformAnalyzer class."""
    
    print("UNIFORM ANALYZER - USING ACTUAL UNIFORMANALYZER CLASS")
    print("=" * 60)
    
    # Load data
    raw_components = load_chunked_with_templates()
    gt_violations = load_ground_truth_violations()
    
    print(f"Raw components: {len(raw_components)}")
    print(f"Ground truth entries: {len(gt_violations)}")
    
    # Show some examples of both to debug matching
    print("\nFirst few raw component texts:")
    for i, comp in enumerate(raw_components[:3]):
        print(f"  {i+1}. {comp.get('original_text', comp.get('text', ''))[:100]}...")
    
    print("\nFirst few ground truth texts:")
    gt_texts = list(gt_violations.keys())
    for i, text in enumerate(gt_texts[:3]):
        print(f"  {i+1}. {text[:100]}...")
    
    # Convert all components to QUSComponent objects
    all_qus_components = []
    matched_count = 0
    unmatched_components = []
    unmatched_gt = list(gt_violations.keys())
    
    for raw_comp in raw_components:
        qus_comp = create_qus_component(raw_comp)
        original_text = qus_comp.original_text
        
        # Debug: Show what we're looking for
        # print(f"Looking for: {original_text[:100]}...")
        
        # Check if this component has ground truth data
        if original_text in gt_violations:
            all_qus_components.append(qus_comp)
            matched_count += 1
            if original_text in unmatched_gt:
                unmatched_gt.remove(original_text)
        else:
            unmatched_components.append(original_text)
    
    print(f"\nMatched components: {matched_count}")
    print(f"Unmatched components: {len(unmatched_components)}")
    print(f"Unmatched ground truth entries: {len(unmatched_gt)}")
    
    # Show some unmatched examples
    if unmatched_components:
        print("\nSome unmatched component texts:")
        for i, text in enumerate(unmatched_components[:3]):
            print(f"  {i+1}. {text[:100]}...")
    
    if unmatched_gt:
        print("\nSome unmatched ground truth texts:")
        for i, text in enumerate(unmatched_gt[:3]):
            print(f"  {i+1}. {text[:100]}...")
    
    print(f"\nFound {len(all_qus_components)} stories with both templates and ground truth")
    
    if len(all_qus_components) < 2:
        print("ERROR: Need at least 2 components for uniformity analysis")
        return
    
    # Use the actual UniformAnalyzer
    print(f"\nRunning UniformAnalyzer...")
    results = UniformAnalyzer.run(None, 0, all_qus_components)
    
    # Process results for evaluation
    violations_found = {}
    
    for i, (comp, (violations, _)) in enumerate(zip(all_qus_components, results)):
        story_text = comp.original_text
        has_violations = len(violations) > 0
        
        # Ground truth comparison
        gt_story_violations = gt_violations.get(story_text, [])
        has_uniform_gt = 'Uniform' in gt_story_violations
        has_uniform_detected = has_violations
        
        # Classify result
        if has_uniform_gt and has_uniform_detected:
            result_type = "TRUE POSITIVE"
        elif not has_uniform_gt and not has_uniform_detected:
            result_type = "TRUE NEGATIVE"
        elif has_uniform_detected and not has_uniform_gt:
            result_type = "FALSE POSITIVE"
        else:  # has_uniform_gt and not has_uniform_detected
            result_type = "FALSE NEGATIVE"
        
        violations_found[story_text] = {
            'violations': [v.issue for v in violations],
            'has_violations': has_violations,
            'ground_truth_has_uniform': has_uniform_gt,
            'detected_violations_count': len(violations),
            'result_type': result_type
        }
    
    # Calculate metrics
    tp = sum(1 for v in violations_found.values() if v['result_type'] == 'TRUE POSITIVE')
    fp = sum(1 for v in violations_found.values() if v['result_type'] == 'FALSE POSITIVE')
    fn = sum(1 for v in violations_found.values() if v['result_type'] == 'FALSE NEGATIVE')
    tn = sum(1 for v in violations_found.values() if v['result_type'] == 'TRUE NEGATIVE')
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Prepare results
    eval_results = {
        'analyzer_type': 'UniformAnalyzer_class',
        'total_stories': len(all_qus_components),
        'stories_flagged': sum(1 for v in violations_found.values() if v['has_violations']),
        'stories': violations_found,
        'summary': {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    }
    
    # Print detailed results
    print(f"\nVIOLATION ANALYSIS RESULTS:")
    stories_with_violations = [(story, info) for story, info in violations_found.items() if info['has_violations']]
    print(f"Stories with violations: {len(stories_with_violations)}")
    
    for story, info in stories_with_violations[:20]:  # Show first 20 for brevity
        print(f"\n--- {story[:60]}... ---")
        print(f"  Ground Truth: {'Uniform' if info['ground_truth_has_uniform'] else 'No Uniform'}")
        print(f"  Result: {info['result_type']}")
        for violation in info['violations']:
            print(f"  - {violation}")
    
    if len(stories_with_violations) > 20:
        print(f"\n... and {len(stories_with_violations) - 20} more stories with violations")
    
    # Print summary
    print(f"\nUNIFORMITY ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Analysis Method: Using actual UniformAnalyzer class")
    print(f"Stories tested: {len(all_qus_components)}")
    print(f"Stories flagged: {eval_results['stories_flagged']}")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Negatives: {tn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    # Save results
    output_dir = Path(__file__).parent / "results" / "uniform"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "uniform_analyzer_results.json"
    
    with open(output_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    try:
        test_uniform_analyzer()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the chunked data and ground truth files exist.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()