#!/usr/bin/env python3
"""Run all analyzer tests and generate comprehensive comparison report."""

import json
from pathlib import Path
from test_atomic import test_atomic_analyzer
from test_minimal import test_minimal_analyzer
from test_uniform import test_uniform_analyzer
from test_wellformed import test_wellformed_analyzer
from test_unique import test_unique_analyzer

def run_all_tests():
    """Run all individual analyzer tests and compile results."""
    
    print("COMPREHENSIVE ANALYZER TESTING")
    print("=" * 60)
    
    results = {}
    
    # Run all individual tests
    tests = [
        ('atomic', test_atomic_analyzer),
        ('minimal', test_minimal_analyzer),
        ('uniform', test_uniform_analyzer),
        ('wellformed', test_wellformed_analyzer),
        ('unique', test_unique_analyzer)
    ]
    
    for analyzer_name, test_func in tests:
        print(f"\n{'Running ' + analyzer_name.upper() + ' test'}")
        print("-" * 40)
        
        try:
            test_result = test_func()
            results[analyzer_name] = test_result
            print(f"{analyzer_name.upper()} test completed successfully")
        except Exception as e:
            print(f"Error in {analyzer_name} test: {e}")
            results[analyzer_name] = {'error': str(e)}
    
    # Generate comparative summary
    print(f"\n{'COMPARATIVE SUMMARY'}")
    print("=" * 60)
    
    summary_report = {
        'overall_stats': {},
        'analyzer_comparison': {},
        'component_similarity_ranking': {}
    }
    
    # Collect stats from all analyzers
    analyzer_stats = {}
    for analyzer_name, result in results.items():
        if 'error' not in result and 'summary' in result:
            analyzer_stats[analyzer_name] = result['summary']
    
    # Component similarity comparison
    components = ['avg_role_similarity', 'avg_means_similarity', 'avg_ends_similarity']
    
    for component in components:
        component_scores = {}
        for analyzer_name, stats in analyzer_stats.items():
            if component in stats:
                component_scores[analyzer_name] = stats[component]
        
        # Sort by similarity score
        sorted_scores = sorted(component_scores.items(), key=lambda x: x[1], reverse=True)
        summary_report['component_similarity_ranking'][component] = sorted_scores
        
        # Print ranking
        component_display = component.replace('avg_', '').replace('_similarity', '').upper()
        print(f"\n{component_display} Similarity Ranking:")
        for i, (analyzer, score) in enumerate(sorted_scores):
            print(f"  {i+1}. {analyzer.capitalize()}: {score:.3f}")
    
    # Overall similarity ranking
    overall_scores = {}
    for analyzer_name, stats in analyzer_stats.items():
        scores = [stats.get(comp, 0.0) for comp in components]
        if scores:
            overall_avg = sum(scores) / len(scores)
            overall_scores[analyzer_name] = overall_avg
    
    sorted_overall = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
    summary_report['overall_stats']['similarity_ranking'] = sorted_overall
    
    print(f"\nOVERALL SIMILARITY RANKING:")
    for i, (analyzer, score) in enumerate(sorted_overall):
        print(f"  {i+1}. {analyzer.capitalize()}: {score:.3f}")
    
    # Save comprehensive results
    output_path = Path("results")
    output_path.mkdir(exist_ok=True)
    
    # Save individual results
    with open(output_path / "all_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary report
    with open(output_path / "summary_report.json", 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    print(f"\nResults saved:")
    print(f"  - all_results.json: Complete test results")
    print(f"  - summary_report.json: Comparative summary")
    print(f"  - Individual analyzer results in results/ directory")
    
    return results, summary_report

if __name__ == "__main__":
    try:
        run_all_tests()
    except Exception as e:
        print(f"Error running complete test suite: {e}")
        import traceback
        traceback.print_exc()