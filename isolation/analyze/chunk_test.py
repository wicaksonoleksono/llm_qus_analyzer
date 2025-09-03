#!/usr/bin/env python3
"""Test chunk quality by comparing component similarity with ground truth."""

import sys
import json
import csv
import statistics
import time
from pathlib import Path
from difflib import SequenceMatcher

# Add root directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_qus_analyzer import Settings, LLMClient

def char_similarity(str1, str2):
    """Calculate character-level similarity between two strings."""
    if not str1 and not str2:
        return 1.0  # Both empty
    if not str1 or not str2:
        return 0.0  # One empty, one not
    
    return SequenceMatcher(None, str1.lower().strip(), str2.lower().strip()).ratio()

def load_chunk_best():
    """Load best quality chunked data with token usage tracking."""
    chunk_file = Path("chunked_story/chunk_best.json")
    
    if not chunk_file.exists():
        raise FileNotFoundError(f"Best chunks not found at {chunk_file}")
    
    with open(chunk_file, 'r') as f:
        raw_chunks = json.load(f)
    
    # Extract components and usage from the nested structure
    chunks = []
    usages = []
    for item in raw_chunks:
        if 'component' in item:
            chunks.append(item['component'])
            # Extract usage info if available
            if 'usage' in item:
                usages.append(item['usage'])
            else:
                usages.append(None)
    
    print(f"Loaded {len(chunks)} chunked components")
    return chunks, usages

def load_ground_truth_components():
    """Load ground truth component data."""
    gt_file = Path("ground_truth_components.json")
    
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth components not found at {gt_file}")
    
    with open(gt_file, 'r') as f:
        gt_components = json.load(f)
    
    print(f"Loaded ground truth for {len(gt_components)} stories")
    return gt_components

def compare_components(chunked_comp, gt_comp):
    """Compare chunked components against ground truth with similarity scores."""
    
    results = {
        'role': {'similarity': 0.0, 'chunked': [], 'ground_truth': []},
        'means': {'similarity': 0.0, 'chunked': None, 'ground_truth': None}, 
        'ends': {'similarity': 0.0, 'chunked': None, 'ground_truth': None}
    }
    
    # Role comparison (lists)
    chunked_role = chunked_comp.get('role', [])
    gt_role = gt_comp.get('role', [])
    
    results['role']['chunked'] = chunked_role
    results['role']['ground_truth'] = gt_role
    
    # Compare role lists (join as strings for similarity)
    chunked_role_str = ', '.join(chunked_role) if chunked_role else ''
    gt_role_str = ', '.join(gt_role) if gt_role else ''
    results['role']['similarity'] = char_similarity(chunked_role_str, gt_role_str)
    
    # Means comparison
    chunked_means = chunked_comp.get('means')
    gt_means = gt_comp.get('means')
    
    results['means']['chunked'] = chunked_means
    results['means']['ground_truth'] = gt_means
    results['means']['similarity'] = char_similarity(chunked_means or '', gt_means or '')
    
    # Ends comparison  
    chunked_ends = chunked_comp.get('ends')
    gt_ends = gt_comp.get('ends')
    
    results['ends']['chunked'] = chunked_ends
    results['ends']['ground_truth'] = gt_ends
    results['ends']['similarity'] = char_similarity(chunked_ends or '', gt_ends or '')
    
    return results

def calculate_statistics(values):
    """Calculate comprehensive statistics for a list of values."""
    if not values:
        return {
            'min': 0, 'max': 0, 'median': 0, 'std': 0, 
            'mean': 0, 'count': 0
        }
    
    return {
        'min': min(values),
        'max': max(values),
        'median': statistics.median(values),
        'std': statistics.stdev(values) if len(values) > 1 else 0,
        'mean': statistics.mean(values),
        'count': len(values)
    }

def extract_token_stats(usages):
    """Extract token usage statistics from usage data."""
    in_tokens = []
    out_tokens = []
    durations = []
    
    for usage in usages:
        if usage:
            # Handle different possible usage structures
            if isinstance(usage, dict):
                # Check for num_token_in/out (chunk_best.json format)
                if 'num_token_in' in usage:
                    in_tokens.append(usage['num_token_in'])
                elif 'input_tokens' in usage:
                    in_tokens.append(usage['input_tokens'])
                
                if 'num_token_out' in usage:
                    out_tokens.append(usage['num_token_out'])
                elif 'output_tokens' in usage:
                    out_tokens.append(usage['output_tokens'])
                
                if 'duration' in usage:
                    durations.append(usage['duration'])
                elif 'time_taken' in usage:
                    durations.append(usage['time_taken'])
            elif isinstance(usage, list) and usage:
                # If usage is a list, use the first item
                first_usage = usage[0]
                if 'num_token_in' in first_usage:
                    in_tokens.append(first_usage['num_token_in'])
                elif 'input_tokens' in first_usage:
                    in_tokens.append(first_usage['input_tokens'])
                
                if 'num_token_out' in first_usage:
                    out_tokens.append(first_usage['num_token_out'])
                elif 'output_tokens' in first_usage:
                    out_tokens.append(first_usage['output_tokens'])
                
                if 'duration' in first_usage:
                    durations.append(first_usage['duration'])
    
    return {
        'input_tokens': calculate_statistics(in_tokens),
        'output_tokens': calculate_statistics(out_tokens),
        'durations': calculate_statistics(durations)
    }

def save_csv_results(results, token_stats, output_dir):
    """Save results as CSV file."""
    csv_file = output_dir / "chunk_quality_results.csv"
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'Story_Index', 'Story_Text_Preview', 'Role_Similarity', 'Means_Similarity', 
            'Ends_Similarity', 'Average_Similarity', 'Role_Chunked', 'Role_GT', 
            'Means_Chunked', 'Means_GT', 'Ends_Chunked', 'Ends_GT'
        ])
        
        # Write story data
        for i, (story_text, story_data) in enumerate(results['stories'].items()):
            comp_results = story_data['component_comparison']
            avg_sim = story_data['average_similarity']
            
            writer.writerow([
                i + 1,
                story_text[:50] + '...' if len(story_text) > 50 else story_text,
                f"{comp_results['role']['similarity']:.3f}",
                f"{comp_results['means']['similarity']:.3f}",
                f"{comp_results['ends']['similarity']:.3f}",
                f"{avg_sim:.3f}",
                ', '.join(comp_results['role']['chunked']) if comp_results['role']['chunked'] else '',
                ', '.join(comp_results['role']['ground_truth']) if comp_results['role']['ground_truth'] else '',
                comp_results['means']['chunked'] or '',
                comp_results['means']['ground_truth'] or '',
                comp_results['ends']['chunked'] or '',
                comp_results['ends']['ground_truth'] or ''
            ])
    
    # Save token statistics as separate CSV
    token_csv_file = output_dir / "chunk_token_statistics.csv"
    
    with open(token_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write token statistics
        writer.writerow(['Metric', 'Type', 'Min', 'Max', 'Median', 'Mean', 'Std', 'Count'])
        
        for metric_name, stats in token_stats.items():
            if stats['count'] > 0:
                writer.writerow([
                    metric_name.replace('_', ' ').title(),
                    'Tokens' if 'token' in metric_name else 'Duration (seconds)',
                    f"{stats['min']:.3f}",
                    f"{stats['max']:.3f}",
                    f"{stats['median']:.3f}",
                    f"{stats['mean']:.3f}",
                    f"{stats['std']:.3f}",
                    stats['count']
                ])
    
    print(f"CSV results saved to {csv_file}")
    print(f"Token statistics saved to {token_csv_file}")

def test_chunk_quality():
    """Test chunk quality by comparing against ground truth components."""
    
    print("CHUNK QUALITY TEST - Component Similarity")
    print("=" * 60)
    
    # Load data
    chunks, usages = load_chunk_best()
    gt_components = load_ground_truth_components()
    
    # Extract token usage statistics
    token_stats = extract_token_stats(usages)
    
    # Create mapping from original_text to chunk
    chunk_mapping = {}
    usage_mapping = {}
    for i, chunk in enumerate(chunks):
        original_text = chunk.get('original_text', chunk.get('text', ''))
        chunk_mapping[original_text] = chunk
        if i < len(usages):
            usage_mapping[original_text] = usages[i]
    
    # Find stories that exist in both datasets
    matched_stories = []
    for story_text, gt_comp in gt_components.items():
        if story_text in chunk_mapping:
            matched_stories.append((story_text, chunk_mapping[story_text], gt_comp))
    
    print(f"Found {len(matched_stories)} stories in both datasets")
    print(f"Testing component similarity...\n")
    
    # Analyze each story
    results = {
        'total_stories': len(matched_stories),
        'stories': {},
        'summary': {
            'avg_role_similarity': 0.0,
            'avg_means_similarity': 0.0, 
            'avg_ends_similarity': 0.0,
            'perfect_role_matches': 0,
            'perfect_means_matches': 0,
            'perfect_ends_matches': 0,
            'similarity_stats': {
                'role': {},
                'means': {},
                'ends': {}
            }
        },
        'token_statistics': token_stats
    }
    
    role_similarities = []
    means_similarities = []
    ends_similarities = []
    
    for i, (story_text, chunked_comp, gt_comp) in enumerate(matched_stories):
        print(f"--- Story {i+1}/{len(matched_stories)} ---")
        print(f"Text: {story_text[:60]}{'...' if len(story_text) > 60 else ''}")
        
        # Compare components
        comp_results = compare_components(chunked_comp, gt_comp)
        
        # Print detailed comparison
        for comp_type, data in comp_results.items():
            similarity = data['similarity']
            chunked = data['chunked']
            gt = data['ground_truth']
            
            print(f"{comp_type.upper()}: Similarity = {similarity:.3f}")
            print(f"  Chunked:      {chunked}")
            print(f"  Ground Truth: {gt}")
            
            # Count perfect matches
            if similarity == 1.0:
                results['summary'][f'perfect_{comp_type}_matches'] += 1
        
        # Calculate average similarity for this story
        avg_similarity = sum(data['similarity'] for data in comp_results.values()) / len(comp_results)
        print(f"AVERAGE SIMILARITY: {avg_similarity:.3f}")
        print("-" * 80)
        
        # Store results
        results['stories'][story_text] = {
            'component_comparison': comp_results,
            'average_similarity': avg_similarity
        }
        
        # Collect similarities
        role_similarities.append(comp_results['role']['similarity'])
        means_similarities.append(comp_results['means']['similarity'])
        ends_similarities.append(comp_results['ends']['similarity'])
    
    # Calculate summary statistics for similarities
    if role_similarities:
        results['summary']['avg_role_similarity'] = sum(role_similarities) / len(role_similarities)
        results['summary']['similarity_stats']['role'] = calculate_statistics(role_similarities)
    if means_similarities:
        results['summary']['avg_means_similarity'] = sum(means_similarities) / len(means_similarities)
        results['summary']['similarity_stats']['means'] = calculate_statistics(means_similarities)
    if ends_similarities:
        results['summary']['avg_ends_similarity'] = sum(ends_similarities) / len(ends_similarities)
        results['summary']['similarity_stats']['ends'] = calculate_statistics(ends_similarities)
    
    # Print final summary
    print(f"\nCHUNK QUALITY SUMMARY")
    print("=" * 60)
    print(f"Total stories analyzed: {len(matched_stories)}")
    print(f"Average role similarity: {results['summary']['avg_role_similarity']:.3f}")
    print(f"Average means similarity: {results['summary']['avg_means_similarity']:.3f}")
    print(f"Average ends similarity: {results['summary']['avg_ends_similarity']:.3f}")
    print(f"Perfect role matches: {results['summary']['perfect_role_matches']}/{len(matched_stories)}")
    print(f"Perfect means matches: {results['summary']['perfect_means_matches']}/{len(matched_stories)}")
    print(f"Perfect ends matches: {results['summary']['perfect_ends_matches']}/{len(matched_stories)}")
    
    overall_avg = (results['summary']['avg_role_similarity'] + 
                   results['summary']['avg_means_similarity'] + 
                   results['summary']['avg_ends_similarity']) / 3
    print(f"OVERALL AVERAGE SIMILARITY: {overall_avg:.3f}")
    
    # Print token usage statistics
    print(f"\nTOKEN USAGE STATISTICS")
    print("=" * 60)
    for metric_name, stats in token_stats.items():
        if stats['count'] > 0:
            print(f"{metric_name.replace('_', ' ').title()}:")
            print(f"  Count: {stats['count']}")
            print(f"  Min: {stats['min']:.3f}")
            print(f"  Max: {stats['max']:.3f}")
            print(f"  Median: {stats['median']:.3f}")
            print(f"  Mean: {stats['mean']:.3f}")
            print(f"  Std: {stats['std']:.3f}")
        else:
            print(f"{metric_name.replace('_', ' ').title()}: No data available")
    
    # Print similarity statistics
    print(f"\nSIMILARITY STATISTICS")
    print("=" * 60)
    for comp_type, sim_stats in results['summary']['similarity_stats'].items():
        if sim_stats:
            print(f"{comp_type.title()} Similarity:")
            print(f"  Min: {sim_stats['min']:.3f}")
            print(f"  Max: {sim_stats['max']:.3f}")
            print(f"  Median: {sim_stats['median']:.3f}")
            print(f"  Mean: {sim_stats['mean']:.3f}")
            print(f"  Std: {sim_stats['std']:.3f}")
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON results
    with open(output_dir / "chunk_quality_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save CSV results
    save_csv_results(results, token_stats, output_dir)
    
    print(f"\nResults saved to results/chunk_quality_results.json")
    
    return results

if __name__ == "__main__":
    try:
        test_chunk_quality()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run 'python extract_gt.py' first to create ground truth files.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()