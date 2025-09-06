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

_end = {
    "deepseek",
    "gpt", 
    "llama",
    "mistral"
}

def load_chunked_with_templates_per_model(model_name):
    """Load chunked data with built templates for a specific model."""
    chunk_file = Path(__file__).parent / "chunked_story" / f"chunked_with_built_templates_{model_name}.json"
    
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
    
    print(f"Loaded {len(components)} components with built templates from {model_name}")
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
    """Test the actual UniformAnalyzer class per model."""
    
    print("UNIFORM ANALYZER - USING ACTUAL UNIFORMANALYZER CLASS")
    print("=" * 60)
    
    # Load ground truth
    gt_violations = load_ground_truth_violations()
    
    # Create results directory
    output_dir = Path(__file__).parent / "results" / "uniform"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test each model separately
    for model_name in _end:
        print(f"\n" + "="*80)
        print(f"TESTING MODEL: {model_name.upper()}")
        print(f"="*80)
        
        # Load data for this model
        try:
            raw_components = load_chunked_with_templates_per_model(model_name)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            continue
        
        print(f"Raw components: {len(raw_components)}")
        print(f"Ground truth entries: {len(gt_violations)}")
        
        # Convert all components to QUSComponent objects
        all_qus_components = []
        matched_count = 0
        
        for raw_comp in raw_components:
            qus_comp = create_qus_component(raw_comp)
            original_text = qus_comp.original_text
            
            # Check if this component has ground truth data
            if original_text in gt_violations:
                all_qus_components.append(qus_comp)
                matched_count += 1
        
        print(f"Found {len(all_qus_components)} stories with templates and ground truth for {model_name}")
        
        if len(all_qus_components) < 2:
            print(f"Skipping {model_name} - need at least 2 components for uniformity analysis")
            continue
        
        # Use the actual UniformAnalyzer
        print(f"\nRunning UniformAnalyzer for {model_name}...")
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
            'analyzer_info': {
                'analyzer_name': 'UniformAnalyzer',
                'analyzer_type': 'rule-based',
                'model': model_name
            },
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
        
        # Print summary for this model
        print(f"\nUNIFORMITY ANALYSIS SUMMARY - {model_name.upper()}")
        print("=" * 60)
        print(f"Stories tested: {len(all_qus_components)}")
        print(f"Stories flagged: {eval_results['stories_flagged']}")
        print(f"True Positives: {tp}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Negatives: {tn}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")
        
        # Save results for this model
        output_file = output_dir / f"uniform_{model_name}_results.json"
        
        with open(output_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        print(f"Results saved to {output_file}")
    
    print(f"\n{'='*80}")
    print("UNIFORM ANALYZER TEST SUMMARY")
    print(f"{'='*80}")
    print("UniformAnalyzer tested for each model separately using existing template files:")
    print("  - Rule-based uniformity analysis")
    print(f"\nResults saved in: {output_dir}/")
    for model in _end:
        print(f"  - uniform_{model}_results.json")

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