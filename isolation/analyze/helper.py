import sys
import json
from pathlib import Path
from difflib import SequenceMatcher

# Add root directory to path for imports (go up 2 levels from isolation/analyze/)
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_qus_analyzer.individual import AtomicAnalyzer, MinimalAnalyzer, WellFormAnalyzer
from llm_qus_analyzer.set import UniformAnalyzer, UniqueAnalyzer

def load_llama_chunks():
    """Load best quality chunked data."""
    chunk_file = Path("chunked_story/chunk_best.json")
    
    if not chunk_file.exists():
        raise FileNotFoundError(f"Best chunks not found at {chunk_file}")
    
    with open(chunk_file, 'r') as f:
        raw_chunks = json.load(f)
    
    # Extract components from the nested structure
    chunks = []
    for item in raw_chunks:
        if 'component' in item:
            chunks.append(item['component'])
    
    print(f"Loaded {len(chunks)} best quality chunks")
    return chunks

def load_ground_truth_components():
    """Load ground truth component data."""
    gt_file = Path("ground_truth_components.json")
    
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth components not found at {gt_file}")
    
    with open(gt_file, 'r') as f:
        gt_components = json.load(f)
    
    print(f"Loaded ground truth for {len(gt_components)} stories")
    return gt_components

def char_similarity(str1, str2):
    """Calculate character-level similarity between two strings."""
    if not str1 and not str2:
        return 1.0  # Both empty
    if not str1 or not str2:
        return 0.0  # One empty, one not
    
    return SequenceMatcher(None, str1.lower().strip(), str2.lower().strip()).ratio()

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

def get_story_components(chunks, story_text):
    """Find component data for a story text in chunks."""
    for component in chunks:
        if isinstance(component, dict):
            original_text = component.get('original_text', component.get('text', ''))
            if original_text == story_text:
                return component
    return None

def print_comparison_results(story_text, comp_results, violation_results=None):
    """Print formatted comparison results."""
    print(f"\nStory: {story_text[:60]}{'...' if len(story_text) > 60 else ''}")
    print("-" * 80)
    
    for comp_type, data in comp_results.items():
        similarity = data['similarity']
        chunked = data['chunked']
        gt = data['ground_truth']
        
        print(f"{comp_type.upper()}: Similarity = {similarity:.3f}")
        print(f"  Chunked:      {chunked}")
        print(f"  Ground Truth: {gt}")
    
    if violation_results:
        violations = [v.get('issue', 'Violation detected') for v in violation_results]
        print(f"VIOLATIONS: {violations}")
    
    avg_similarity = sum(data['similarity'] for data in comp_results.values()) / len(comp_results)
    print(f"AVERAGE SIMILARITY: {avg_similarity:.3f}")

def save_results(analyzer_name, results, output_dir="results"):
    """Save analysis results to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    output_file = output_path / f"{analyzer_name}_results.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")