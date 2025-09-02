import json
from pathlib import Path
from collections import defaultdict

def load_chunked_stories():
    """Load all chunked story data per model."""
    chunked_dir = Path("chunked_story")
    chunked_data = {}
    
    for json_file in chunked_dir.glob("*.json"):
        model_name = json_file.stem
        with open(json_file, 'r') as f:
            chunked_data[model_name] = json.load(f)
    
    return chunked_data

def load_analysis_results():
    """Load all analysis results per model and analyzer."""
    analysis_dir = Path("analysis_results")
    analysis_data = defaultdict(dict)
    
    for model_dir in analysis_dir.iterdir():
        if model_dir.is_dir():
            model_name = model_dir.name
            for analyzer_file in model_dir.glob("*.json"):
                analyzer_name = analyzer_file.stem
                with open(analyzer_file, 'r') as f:
                    analysis_data[model_name][analyzer_name] = json.load(f)
    
    return analysis_data

def get_violation_details(analysis_data, model_name, story_id, violation_type):
    """Extract detailed violation information for a story."""
    if model_name not in analysis_data or violation_type not in analysis_data[model_name]:
        return None
    
    analyzer_data = analysis_data[model_name][violation_type]
    
    # Handle different analyzer result formats
    if isinstance(analyzer_data, list):
        # Individual analyzers
        for component_data in analyzer_data:
            if component_data.get('component_id') == story_id:
                violations = component_data.get('violations', [])
                if violations:
                    return violations[0]  # Return first violation details
    elif isinstance(analyzer_data, dict):
        # Set analyzers
        violations = analyzer_data.get('violations', [])
        for violation in violations:
            story_ids = violation.get('story_ids', [])
            # Convert story indices to story_id format
            story_indices = [f"story_{idx + 1}" if isinstance(idx, int) else str(idx) for idx in story_ids]
            if story_id in story_indices:
                return violation
    
    return None

def load_ground_truth():
    """Load ground truth data."""
    with open('ground_truth_extracted.json', 'r') as f:
        return json.load(f)

def extract_violation_stories():
    """Extract stories with violations grouped by violation type for easier debugging."""
    
    # Load data sources
    with open('violations_collected.json', 'r') as f:
        violations_per_model = json.load(f)
    
    chunked_data = load_chunked_stories()
    analysis_data = load_analysis_results()
    ground_truth = load_ground_truth()
    
    # Create story text to component mapping
    story_text_to_component = {}
    for model_name, chunks in chunked_data.items():
        story_text_to_component[model_name] = {}
        
        # Handle both list and dict formats
        if isinstance(chunks, list):
            for i, component in enumerate(chunks):
                component_id = component.get('id', f'story_{i+1}')
                original_text = component.get('original_text', component.get('text', ''))
                story_text_to_component[model_name][original_text] = (component_id, component)
        else:
            for component_id, component in chunks.items():
                original_text = component.get('original_text', component.get('text', ''))
                story_text_to_component[model_name][original_text] = (component_id, component)
    
    violation_stories = {}
    
    for model_name, model_violations in violations_per_model.items():
        violation_stories[model_name] = {}
        
        print(f"Processing {model_name}...")
        
        for story_text, violation_types in model_violations.items():
            # Find component data
            if model_name not in story_text_to_component or story_text not in story_text_to_component[model_name]:
                print(f"  Warning: Could not find component for story: {story_text[:50]}...")
                continue
            
            component_id, component_data = story_text_to_component[model_name][story_text]
            
            # Get ground truth for this story
            ground_truth_labels = ground_truth.get(story_text, [])
            
            # Process each violation type
            for violation_type in violation_types:
                if violation_type not in violation_stories[model_name]:
                    violation_stories[model_name][violation_type] = {}
                
                # Get violation details
                details = get_violation_details(analysis_data, model_name, component_id, violation_type.lower())
                issue = details.get('issue', 'Violation detected') if details else 'Violation detected'
                
                violation_stories[model_name][violation_type][component_id] = {
                    'ground_truth': ground_truth_labels,
                    'violation': violation_type,
                    'original_user_story': story_text,
                    'components': {
                        'role': component_data.get('role', []),
                        'means': component_data.get('means'),
                        'ends': component_data.get('ends')
                    },
                    'issue': issue
                }
        
        total_violations = sum(len(violations) for violations in violation_stories[model_name].values())
        print(f"  Found {total_violations} violation instances")
    
    return violation_stories

def save_results(violation_stories):
    """Save results as JSON only."""
    
    # Save main JSON
    with open('violation_stories.json', 'w') as f:
        json.dump(violation_stories, f, indent=2)
    
    # Print summary statistics
    print(f"\nSUMMARY STATISTICS")
    print("=" * 50)
    
    for model_name, stories in violation_stories.items():
        print(f"\n{model_name}:")
        print(f"  Total stories with violations: {len(stories)}")
        
        # Count violations by type
        violation_counts = defaultdict(int)
        for story_data in stories.values():
            for violation in story_data['violations']:
                violation_counts[violation] += 1
        
        print(f"  Violation breakdown:")
        for violation_type, count in sorted(violation_counts.items()):
            print(f"    {violation_type}: {count}")

def main():
    print("Extracting violation stories with detailed component breakdown...")
    
    violation_stories = extract_violation_stories()
    save_results(violation_stories)
    
    print(f"\nResults saved: violation_stories.json")

if __name__ == "__main__":
    main()