#!/usr/bin/env python3
"""Test Unique Analyzer violation detection only."""

import sys
import json
from pathlib import Path

# Add root directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_qus_analyzer.set import UniqueAnalyzer
from llm_qus_analyzer import Settings, LLMClient
from llm_qus_analyzer.chunker.models import QUSComponent
from llm_qus_analyzer.chunker.parser import Template

def load_chunk_best():
    """Load best quality chunked data."""
    chunk_file = Path("chunked_story/chunk_best.json")
    
    with open(chunk_file, 'r') as f:
        raw_chunks = json.load(f)
    
    # Extract components and convert to QUSComponent format
    components = []
    for item in raw_chunks:
        if 'component' in item:
            comp_data = item['component']
            # Convert to expected format for analyzer
            components.append(comp_data)
    
    print(f"Loaded {len(components)} components for violation testing")
    return components

def load_ground_truth_violations():
    """Load ground truth violation labels."""
    gt_file = Path("ground_truth_extracted.json")
    
    with open(gt_file, 'r') as f:
        gt_violations = json.load(f)
    
    print(f"Loaded ground truth violations for {len(gt_violations)} stories")
    return gt_violations

def create_qus_component(raw_comp):
    """Convert raw dictionary component to QUSComponent object."""
    # Create a minimal template 
    dummy_template = Template(
        text=raw_comp.get('text', ''),
        chunk={},
        tail=None,
        order=[]
    )
    
    return QUSComponent(
        text=raw_comp.get('text', ''),
        role=raw_comp.get('role', []),
        means=raw_comp.get('means'),
        ends=raw_comp.get('ends'),
        template=dummy_template,
        id=raw_comp.get('id'),
        original_text=raw_comp.get('original_text', raw_comp.get('text', ''))
    )

def test_unique_analyzer():
    """Test UniqueAnalyzer set-based violation detection."""
    
    print("UNIQUE ANALYZER - SET-BASED VIOLATION DETECTION TEST")
    print("=" * 60)
    
    # Load data
    components = load_chunk_best()
    gt_violations = load_ground_truth_violations()
    
    # Find stories that exist in both datasets
    all_qus_components = []
    
    for component in components:
        original_text = component.get('original_text', component.get('text', ''))
        
        # Only include components that have ground truth data
        if original_text in gt_violations:
            qus_comp = create_qus_component(component)
            all_qus_components.append(qus_comp)
    
    print(f"Found {len(all_qus_components)} stories with both components and ground truth")
    
    if len(all_qus_components) < 2:
        print("ERROR: Need at least 2 components for set-based uniqueness analysis")
        return
    
    # Run set-based analysis on all components at once
    print(f"\nRunning set-based uniqueness analysis on {len(all_qus_components)} components...")
    
    # Note: UniqueAnalyzer is rule-based for duplicate detection, no LLM client needed
    analyzer = UniqueAnalyzer()
    
    try:
        # Run analyzer on entire set - this will identify duplicate stories
        violations_by_story, usage = analyzer.run(None, 0, all_qus_components)
        
        print(f"Uniqueness analysis completed.")
        
    except Exception as e:
        print(f"Error running UniqueAnalyzer: {e}")
        return
    
    # Map detected violations back to individual stories
    violations_found = {}
    
    for i, component in enumerate(all_qus_components):
        story_text = component.original_text
        component_violations = violations_by_story[i] if i < len(violations_by_story) else []
        
        # Ground truth comparison
        gt_story_violations = gt_violations.get(story_text, [])
        has_unique_gt = 'Unique' in gt_story_violations
        has_unique_detected = len(component_violations) > 0
        
        # Classify result
        if has_unique_gt and has_unique_detected:
            result_type = "TRUE POSITIVE"
        elif not has_unique_gt and not has_unique_detected:
            result_type = "TRUE NEGATIVE"
        elif has_unique_detected and not has_unique_gt:
            result_type = "FALSE POSITIVE"
        else:  # has_unique_gt and not has_unique_detected
            result_type = "FALSE NEGATIVE"
        
        violations_found[story_text] = {
            'violations': component_violations,
            'ground_truth_has_unique': has_unique_gt,
            'detected_violations_count': len(component_violations),
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
    results = {
        'analyzer_type': 'set_based_uniqueness_analysis',
        'total_stories': len(all_qus_components),
        'stories_flagged': sum(1 for v in violations_found.values() if len(v['violations']) > 0),
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
    stories_with_violations = [(story, info) for story, info in violations_found.items() if info['violations']]
    print(f"Stories with violations: {len(stories_with_violations)}")
    
    for story, info in stories_with_violations:
        print(f"\n--- {story[:60]}... ---")
        print(f"  Ground Truth: {'Unique' if info['ground_truth_has_unique'] else 'No Unique'}")
        print(f"  Result: {info['result_type']}")
        for violation in info['violations']:
            print(f"  - {violation.issue}")
    
    # Print summary
    print(f"\nUNIQUENESS ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Analysis Method: Set-based duplicate detection")
    print(f"Stories tested: {len(all_qus_components)}")
    print(f"Stories flagged: {results['stories_flagged']}")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Negatives: {tn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    # Save results
    output_dir = Path("results/unique")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "unique_set_analysis_results.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    return results

if __name__ == "__main__":
    try:
        test_unique_analyzer()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run 'python extract_gt.py' first and that env/models are configured.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()