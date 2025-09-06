#!/usr/bin/env python3
"""Build templates for each model separately using the working approach from build_templates_from_chunked.py"""

import sys
import json
from pathlib import Path

# Add root directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_qus_analyzer.chunker.parser import TemplateParser, Template
from llm_qus_analyzer.chunker.models import QUSComponent

_end = {
    "deepseek",
    "gpt",
    "llama", 
    "mistral"
}

def build_template_from_components(role, means, ends, text):
    """Build a template from parsed components using TemplateParser."""
    try:
        # Prepare the parser
        TemplateParser.prepare()
        
        # Build template from the components
        template = TemplateParser.parse(text, role, means, ends)
        return template
    except Exception as e:
        print(f"Error building template: {e}")
        # Return a minimal template if parsing fails
        return Template(
            text=text,
            chunk={},
            tail=None,
            order=[]
        )

def build_templates_for_model(model_name):
    """Build templates for a specific model using the working approach."""
    print(f"\n{'='*60}")
    print(f"BUILDING TEMPLATES FOR {model_name.upper()}")
    print(f"{'='*60}")
    
    # Load chunked data for this model
    chunk_file = Path(f"chunked_story/chunk_{model_name}.json")
    
    if not chunk_file.exists():
        print(f"ERROR: File {chunk_file} not found!")
        return []
    
    with open(chunk_file, 'r') as f:
        raw_chunks = json.load(f)
    
    print(f"Loaded {len(raw_chunks)} components from {model_name}")
    
    # Build templates for each component (following build_templates_from_chunked.py logic)
    enhanced_components = []
    
    for i, item in enumerate(raw_chunks):
        if 'component' in item:
            comp_data = item['component']
            
            # Extract component data
            text = comp_data.get('text', '')
            role = comp_data.get('role', [])
            means = comp_data.get('means')
            ends = comp_data.get('ends')
            original_text = comp_data.get('original_text', text)
            component_id = comp_data.get('id', f"story_{i}")
            
            print(f"Building template for story {i+1}/{len(raw_chunks)}: {text[:50]}{'...' if len(text) > 50 else ''}")
            
            # Build template from the components
            template = build_template_from_components(role, means, ends, text)
            
            # Create enhanced component with template (same format as build_templates_from_chunked.py)
            enhanced_component = {
                'component': {
                    'text': text,
                    'role': role,
                    'means': means,
                    'ends': ends,
                    'template': {
                        'text': template.text,
                        'chunk': template.chunk,
                        'tail': template.tail,
                        'order': template.order
                    },
                    'id': component_id,
                    'original_text': original_text
                }
            }
            
            enhanced_components.append(enhanced_component)
    
    print(f"Successfully built templates for {len(enhanced_components)} components from {model_name}")
    return enhanced_components

def main():
    """Build templates for all models separately."""
    print("BUILDING TEMPLATES PER MODEL FOR UNIFORM TESTING")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path("chunked_story")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # Build templates for each model
    for model_name in _end:
        try:
            components_with_templates = build_templates_for_model(model_name)
            
            if components_with_templates:
                # Save to file with the expected naming convention
                output_file = output_dir / f"chunked_with_built_templates_{model_name}.json"
                
                with open(output_file, 'w') as f:
                    json.dump(components_with_templates, f, indent=2)
                
                print(f"Saved {len(components_with_templates)} components with templates to {output_file}")
                all_results[model_name] = len(components_with_templates)
            else:
                print(f"No templates built for {model_name}")
                all_results[model_name] = 0
                
        except Exception as e:
            print(f"ERROR processing {model_name}: {e}")
            all_results[model_name] = 0
    
    # Summary
    print(f"\n{'='*80}")
    print("TEMPLATE BUILDING SUMMARY")
    print(f"{'='*80}")
    
    total_components = 0
    for model_name, count in all_results.items():
        print(f"{model_name}: {count} components with templates")
        total_components += count
    
    print(f"\nTotal: {total_components} components with templates built")
    print(f"\nOutput files for test_uniform.py:")
    for model_name in _end:
        print(f"  - chunked_story/chunked_with_built_templates_{model_name}.json")
    
    print(f"\nNow you can run: python test_uniform.py")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()