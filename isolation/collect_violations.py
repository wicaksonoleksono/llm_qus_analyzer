import json
import pandas as pd
from pathlib import Path

def get_story_text_mapping():
    """Get mapping from component_id to story text."""
    df = pd.read_csv('ground_truth.csv')
    story_mapping = {}
    for i, row in df.iterrows():
        story_id = f"story_{i + 1}"  # story_1, story_2, etc.
        story_mapping[story_id] = row['Text']
        if i < 3:  # Debug first 3 mappings
            print(f"DEBUG: {story_id} -> {row['Text'][:50]}...")
    print(f"DEBUG: Total story mappings: {len(story_mapping)}")
    return story_mapping

def collect_violations():
    """Collect violations using folder structure - if analyzer file exists, violations were detected."""
    
    analysis_dir = Path("analysis_results")
    story_mapping = get_story_text_mapping()
    violations_per_model = {}
    
    # Define analyzer types
    analyzer_types = {
        'atomic': 'Atomic',
        'minimal': 'Minimal', 
        'uniform': 'Uniform',
        'well-formed': 'Well-formed'
    }
    
    # Get all model directories
    model_dirs = [d for d in analysis_dir.iterdir() if d.is_dir()]
    
    for model_dir in model_dirs:
        model_name = model_dir.name
        print(f"Processing {model_name}...")
        violations_by_text = {}
        
        # Check each analyzer type
        for analyzer_name, analyzer_title in analyzer_types.items():
            analyzer_file = model_dir / f"{analyzer_name}.json"
            
            if not analyzer_file.exists():
                continue
                
            # File exists, so this analyzer detected violations
            with open(analyzer_file, 'r') as f:
                data = json.load(f)
            
            # Get stories that have violations in this analyzer
            story_ids_with_violations = set()
            
            if isinstance(data, list):
                # Individual analyzers: check each component
                story_ids_with_violations = {
                    comp_data.get('component_id') 
                    for comp_data in data 
                    if comp_data.get('violations', [])
                }
                if analyzer_name == 'atomic' and model_name == list(model_dirs)[0].name:
                    print(f"DEBUG {analyzer_name}: Found story_ids with violations: {story_ids_with_violations}")
            elif isinstance(data, dict):
                # Set analyzers: get story_ids from violations
                violations = data.get('violations', [])
                for violation in violations:
                    story_ids = violation.get('story_ids', [])
                    story_ids_with_violations.update(
                        f"story_{idx + 1}" if isinstance(idx, int) else str(idx) 
                        for idx in story_ids
                    )
            
            # Map story_ids to story text and add analyzer type
            for story_id in story_ids_with_violations:
                story_text = story_mapping.get(story_id)
                if story_text:
                    if story_text not in violations_by_text:
                        violations_by_text[story_text] = []
                    
                    if analyzer_title not in violations_by_text[story_text]:
                        violations_by_text[story_text].append(analyzer_title)
        
        # Sort violations for consistency  
        for story_text in violations_by_text:
            violations_by_text[story_text] = sorted(violations_by_text[story_text])
        
        violations_per_model[model_name] = violations_by_text
    
    return violations_per_model

# Run collection
violations_per_model = collect_violations()

# Save results per model
with open('violations_collected.json', 'w') as f:
    json.dump(violations_per_model, f, indent=2)

print(f"Collected violations for {len(violations_per_model)} models")

# Show summary per model
for model_name, violations in violations_per_model.items():
    total_violations = sum(len(v) for v in violations.values())
    print(f"\n{model_name}:")
    print(f"  Stories with violations: {len(violations)}")
    print(f"  Total violation types: {total_violations}")