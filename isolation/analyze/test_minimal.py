#!/usr/bin/env python3
"""Test Minimal Analyzer violation detection only."""

import sys
import json
from pathlib import Path

# Add root directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_qus_analyzer.individual.minimal import MinimalAnalyzer
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

def test_minimal_analyzer():
    """Test MinimalAnalyzer violation detection."""
    
    print("MINIMAL ANALYZER - VIOLATION DETECTION TEST")
    print("=" * 60)
    
    # Load data
    components = load_chunk_best()
    gt_violations = load_ground_truth_violations()
    
    # Setup LLM client and analyzer
    setting = Settings()
    setting.configure_paths_and_load(
        env_path=Path('../.env'),
        model_config_path=Path('../models.yaml'),
    )
    clients = LLMClient(from_settings=setting)
    
    # Find stories that exist in both datasets
    matched_stories = []
    for component in components:
        original_text = component.get('original_text', component.get('text', ''))
        if original_text in gt_violations:
            matched_stories.append((original_text, component))
    
    print(f"Found {len(matched_stories)} stories in both datasets")
    
    # Create results directory
    output_dir = Path("results/minimal")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test with all available models
    for model_idx in range(clients.n_models):
        model_name = clients.names[model_idx]
        print(f"\n{'='*60}")
        print(f"Testing MinimalAnalyzer with Model {model_idx}: {model_name}")
        print(f"{'='*60}")
        
        results = {
            'analyzer_info': {
                'analyzer_name': 'MinimalAnalyzer',
                'analyzer_type': 'hybrid (rule-based + LLM abbreviation checking)',
                'model_name': model_name,
                'model_idx': model_idx
            },
            'total_stories': len(matched_stories),
            'correct_predictions': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'stories': {},
            'summary': {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
        }
        
        for i, (story_text, component) in enumerate(matched_stories):
            print(f"--- Story {i+1}/{len(matched_stories)} ---")
            print(f"Text: {story_text[:80]}{'...' if len(story_text) > 80 else ''}")
            
            # Get ground truth for this story
            gt_story_violations = gt_violations[story_text]
            has_minimal_gt = 'Minimal' in gt_story_violations
            
            try:
                # Convert raw dictionary to QUSComponent
                qus_component = create_qus_component(component)
                
                # Run MinimalAnalyzer on single component with LLM client and specific model
                detected_violations, _ = MinimalAnalyzer.run(clients, model_idx, qus_component)
                has_minimal_detected = len(detected_violations) > 0
                
                # Compare results
                if has_minimal_gt and has_minimal_detected:
                    results['correct_predictions'] += 1
                    result_type = "TRUE POSITIVE"
                elif not has_minimal_gt and not has_minimal_detected:
                    results['correct_predictions'] += 1
                    result_type = "TRUE NEGATIVE"
                elif has_minimal_detected and not has_minimal_gt:
                    results['false_positives'] += 1
                    result_type = "FALSE POSITIVE"
                else:  # has_minimal_gt and not has_minimal_detected
                    results['false_negatives'] += 1
                    result_type = "FALSE NEGATIVE"
                
                print(f"Ground Truth: {'Minimal' if has_minimal_gt else 'No Minimal'}")
                print(f"Detected: {len(detected_violations)} violations")
                if detected_violations:
                    for v in detected_violations:
                        print(f"  - {v.issue}")
                print(f"Result: {result_type}")
                
                # Store results
                results['stories'][story_text] = {
                    'ground_truth_has_minimal': has_minimal_gt,
                    'detected_violations_count': len(detected_violations),
                    'detected_violations': [v.issue for v in detected_violations],
                    'result_type': result_type
                }
                
            except Exception as e:
                print(f"Error analyzing story: {e}")
                results['stories'][story_text] = {'error': str(e)}
            
            print("-" * 60)
        
        # Calculate metrics for this model
        tp = sum(1 for story in results['stories'].values() 
                if story.get('result_type') == 'TRUE POSITIVE')
        fp = results['false_positives']
        fn = results['false_negatives']
        tn = sum(1 for story in results['stories'].values() 
                if story.get('result_type') == 'TRUE NEGATIVE')
        
        # Handle edge cases for precision and recall
        # When tp=0 and fp=0, precision is 1 (no false positives)
        # When tp=0 and fn=0, recall is 1 (no false negatives)
        precision = 1.0 if (tp == 0 and fp == 0) else tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = 1.0 if (tp == 0 and fn == 0) else tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Special case: when all are true negatives, all metrics should be 1
        if tp == 0 and fp == 0 and fn == 0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        
        results['summary']['precision'] = precision
        results['summary']['recall'] = recall
        results['summary']['f1'] = f1
        results['summary']['true_positives'] = tp
        results['summary']['false_positives'] = fp
        results['summary']['false_negatives'] = fn
        results['summary']['true_negatives'] = tn
        
        # Print summary for this model
        print(f"\nMINIMAL ANALYZER SUMMARY - {model_name}")
        print("=" * 60)
        print(f"Stories tested: {len(matched_stories)}")
        print(f"True Positives: {tp}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Negatives: {tn}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")
        
        # Save results for this model
        model_filename = f"minimal_{model_name.replace(' ', '_').replace('/', '_')}_results.json"
        output_file = output_dir / model_filename
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
    
    print(f"\n{'='*60}")
    print("ALL MODELS TESTED - Check results/minimal/ for individual model results")
    print(f"{'='*60}")

if __name__ == "__main__":
    try:
        test_minimal_analyzer()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run 'python extract_gt.py' first and that env/models are configured.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()