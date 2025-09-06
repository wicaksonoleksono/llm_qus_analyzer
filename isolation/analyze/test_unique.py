#!/usr/bin/env python3
"""Test Unique Analyzer with pairwise, fullset, and dependency modes for each model separately."""

import sys
import json
from pathlib import Path

# Add root directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_qus_analyzer.set.unique import UniqueAnalyzer
from llm_qus_analyzer.chunker.models import QUSComponent
from llm_qus_analyzer.chunker.parser import Template
from llm_qus_analyzer import Settings, LLMClient

_end = {
    "deepseek",
    "gpt", 
    "llama",
    "mistral"
}

def load_chunk_per_model():
    """Load chunked data per model separately."""
    models_data = {}
    
    for j in _end:
        chunk_file = Path(f"chunked_story/chunk_{j}.json")
        
        with open(chunk_file, 'r') as f:
            raw_chunks = json.load(f)
        
        # Extract components and convert to QUSComponent format
        model_components = []
        for item in raw_chunks:
            if 'component' in item:
                comp_data = item['component']
                # Add model source info
                comp_data['model_source'] = j
                model_components.append(comp_data)
        
        models_data[j] = model_components
        print(f"Loaded {len(model_components)} components from {j} model")
    
    return models_data

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
    
    # Use the component's ID if available, otherwise create one based on text
    component_id = raw_comp.get('id')
    if not component_id:
        # Create ID based on original text hash for consistency
        original_text = raw_comp.get('original_text', raw_comp.get('text', ''))
        component_id = f"story_{abs(hash(original_text)) % 10000}"
    
    return QUSComponent(
        text=raw_comp.get('text', ''),
        role=raw_comp.get('role', []),
        means=raw_comp.get('means'),
        ends=raw_comp.get('ends'),
        template=dummy_template,
        id=component_id,
        original_text=raw_comp.get('original_text', raw_comp.get('text', ''))
    )

def test_unique_analyzer():
    """Test UniqueAnalyzer with all 3 methods for each model separately."""
    
    print("UNIQUE ANALYZER - COMPREHENSIVE TEST PER MODEL")
    print("=" * 60)
    
    # Load data per model
    models_data = load_chunk_per_model()
    gt_violations = load_ground_truth_violations()
    
    # Create results directory
    output_dir = Path("results/unique")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize LLM client
    try:
        setting = Settings()
        setting.configure_paths_and_load(
            env_path=Path('../.env'),
            model_config_path=Path('../models.yaml'),
        )
        client = LLMClient(from_settings=setting)
        print("LLM Client initialized successfully")
        print(f"Available models: {client.names}")
    except Exception as e:
        print(f"Failed to initialize LLM Client: {e}")
        return
    
    # Test each model separately
    for model_name in _end:
        print(f"\n" + "="*80)
        print(f"TESTING MODEL: {model_name.upper()}")
        print(f"="*80)
        
        # Get components for this model
        components_raw = models_data[model_name]
        components = [create_qus_component(comp) for comp in components_raw]
        
        # Find stories that exist in both datasets
        matched_stories = []
        for component in components:
            original_text = component.original_text
            if original_text in gt_violations:
                matched_stories.append(component)
        
        print(f"Found {len(matched_stories)} stories in both datasets for {model_name}")
        print(f"Total components for {model_name}: {len(components)}")
        
        if len(matched_stories) < 2:
            print(f"Skipping {model_name} - need at least 2 components for testing")
            continue
        
        # Test 1: FULLSET MODE for this model
        print(f"\n{'='*60}")
        print(f"1. TESTING FULLSET MODE - {model_name.upper()}")
        print(f"{'='*60}")
        
        try:
            violations_fullset, usage_fullset = UniqueAnalyzer.run(
                client, 0, matched_stories, mode="fullset"
            )
            
            print(f"Fullset analysis completed!")
            print(f"Found {len(violations_fullset)} violations")
            
            # Save fullset results with model name
            fullset_results = {
                'analyzer_info': {
                    'analyzer_name': 'UniqueAnalyzer',
                    'mode': 'fullset',
                    'model': model_name,
                    'description': 'LLM-based duplicate detection'
                },
                'total_stories': len(matched_stories),
                'violations_count': len(violations_fullset),
                'violations': [
                    {
                        'story_ids': v.story_ids,
                        'parts_per_story': [list(parts) for parts in v.parts_per_story],
                        'issue': v.issue,
                        'suggestion': v.suggestion
                    } for v in violations_fullset
                ],
                'usage': usage_fullset
            }
            
            output_file = output_dir / f"unique_fullset_{model_name}_results.json"
            with open(output_file, 'w') as f:
                json.dump(fullset_results, f, indent=2)
            print(f"Results saved to {output_file}")
            
        except Exception as e:
            print(f"Error in fullset analysis for {model_name}: {e}")
        
        # Test 2: DEPENDENCY MODE for this model
        print(f"\n{'='*60}")
        print(f"2. TESTING DEPENDENCY MODE - {model_name.upper()}")
        print(f"{'='*60}")
        
        # Use a subset for dependency analysis (first 20 stories)
        dependency_components = matched_stories[:20]
        
        try:
            violations_dependency, usage_dependency = UniqueAnalyzer.run(
                client, 0, dependency_components, mode="dependency"
            )
            
            print(f"Dependency analysis completed!")
            print(f"Found {len(violations_dependency)} violations")
            
            # Convert LLMUsage to dict and sum up totals
            usage_dict = {}
            total_duration = 0
            total_tokens_in = 0
            total_tokens_out = 0
            
            for key, usage in usage_dependency.items():
                usage_dict[key] = {
                    'duration': usage.duration,
                    'num_token_in': usage.num_token_in,
                    'num_token_out': usage.num_token_out
                }
                total_duration += usage.duration
                total_tokens_in += usage.num_token_in
                total_tokens_out += usage.num_token_out
            
            usage_dict['total_summary'] = {
                'total_duration': total_duration,
                'total_tokens_in': total_tokens_in,
                'total_tokens_out': total_tokens_out,
                'clusters_processed': len(usage_dependency)
            }
            
            # Save dependency results with proper ID mapping
            dependency_results = {
                'analyzer_info': {
                    'analyzer_name': 'UniqueAnalyzer',
                    'mode': 'dependency',
                    'model': model_name,
                    'description': 'LLM-based verb-object conflict detection'
                },
                'total_stories': len(dependency_components),
                'stories': [
                    {
                        'index': i,
                        'component_id': comp.id,
                        'text': comp.text,
                        'means': comp.means,
                        'role': comp.role,
                        'ends': comp.ends
                    } for i, comp in enumerate(dependency_components)
                ],
                'violations_count': len(violations_dependency),
                'violations': [
                    {
                        'story_ids': v.story_ids,
                        'components': [
                            {
                                'index': story_idx,
                                'component_id': dependency_components[story_idx].id if story_idx < len(dependency_components) else f'unknown_{story_idx}',
                                'text': dependency_components[story_idx].text if story_idx < len(dependency_components) else 'Unknown story',
                                'means': dependency_components[story_idx].means if story_idx < len(dependency_components) else None,
                                'role': dependency_components[story_idx].role if story_idx < len(dependency_components) else None,
                                'ends': dependency_components[story_idx].ends if story_idx < len(dependency_components) else None
                            }
                            for story_idx in v.story_ids
                        ] if v.story_ids else [],
                        'parts_per_story': [list(parts) for parts in v.parts_per_story],
                        'issue': v.issue,
                        'suggestion': v.suggestion
                    } for v in violations_dependency
                ],
                'usage': usage_dict
            }
            
            output_file = output_dir / f"unique_dependency_{model_name}_results.json"
            with open(output_file, 'w') as f:
                json.dump(dependency_results, f, indent=2)
            print(f"Results saved to {output_file}")
            
        except Exception as e:
            print(f"Error in dependency analysis for {model_name}: {e}")
        
        # Test 3: PAIRWISE MODE for this model (subset test)
        print(f"\n{'='*60}")
        print(f"3. TESTING PAIRWISE MODE - {model_name.upper()}")
        print(f"{'='*60}")
        
        # Test pairwise on first few components
        pairwise_components = matched_stories[:5]
        pairwise_results = {
            'analyzer_info': {
                'analyzer_name': 'UniqueAnalyzer',
                'mode': 'pairwise',
                'model': model_name,
                'description': 'Individual component comparison with LLM semantic analysis'
            },
            'comparisons': [],
            'total_comparisons': 0,
            'violations_found': 0
        }
        
        try:
            for i in range(len(pairwise_components)):
                for j in range(i+1, len(pairwise_components)):
                    comp1 = pairwise_components[i]
                    comp2 = pairwise_components[j]
                    
                    violations_pairwise, usage_pairwise = UniqueAnalyzer.run(
                        client, 0, comp1, comp2, mode="pairwise"
                    )
                    
                    # Convert LLMUsage to dict for JSON serialization
                    usage_dict = {}
                    for key, usage in usage_pairwise.items():
                        usage_dict[key] = {
                            'duration': usage.duration,
                            'num_token_in': usage.num_token_in,
                            'num_token_out': usage.num_token_out
                        }
                    
                    comparison_result = {
                        'component1_id': comp1.id,
                        'component2_id': comp2.id,
                        'component1_text': comp1.text,
                        'component2_text': comp2.text,
                        'violations_count': len(violations_pairwise),
                        'violations': [
                            {
                                'first_id': v.first_id,
                                'second_id': v.second_id,
                                'first_parts': list(v.first_parts),
                                'second_parts': list(v.second_parts),
                                'issue': v.issue,
                                'suggestion': v.suggestion
                            } for v in violations_pairwise
                        ],
                        'usage': usage_dict
                    }
                    
                    pairwise_results['comparisons'].append(comparison_result)
                    pairwise_results['total_comparisons'] += 1
                    pairwise_results['violations_found'] += len(violations_pairwise)
                    
                    if violations_pairwise:
                        print(f"  Violation found: {comp1.id} vs {comp2.id}")
            
            print(f"Pairwise analysis completed!")
            print(f"Total comparisons: {pairwise_results['total_comparisons']}")
            print(f"Violations found: {pairwise_results['violations_found']}")
            
            # Save pairwise results
            output_file = output_dir / f"unique_pairwise_{model_name}_results.json"
            with open(output_file, 'w') as f:
                json.dump(pairwise_results, f, indent=2)
            print(f"Results saved to {output_file}")
            
        except Exception as e:
            print(f"Error in pairwise analysis for {model_name}: {e}")
    
    # Summary
    print(f"\n{'='*80}")
    print("UNIQUE ANALYZER TEST SUMMARY")
    print(f"{'='*80}")
    print("All 3 methods tested for each model separately:")
    print("  1. Fullset - LLM-based duplicate detection")
    print("  2. Dependency - LLM verb-object conflict detection") 
    print("  3. Pairwise - Individual component comparisons")
    print(f"\nResults saved in: {output_dir}/")
    for model in _end:
        print(f"  - unique_fullset_{model}_results.json")
        print(f"  - unique_dependency_{model}_results.json") 
        print(f"  - unique_pairwise_{model}_results.json")

if __name__ == "__main__":
    try:
        test_unique_analyzer()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run 'python extract_gt.py' first and that chunked data exists.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()