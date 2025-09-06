#!/usr/bin/env python3
"""Build templates for each model separately from chunked data."""

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
        template = TemplateParser.build_template(
            role=role or [],
            means=means,
            ends=ends,
            original_text=text
        )
        
        return template
        
    except Exception as e:
        print(f"Error building template for text '{text[:50]}...': {e}")
        # Return a minimal template as fallback
        return Template(
            text=text,
            chunk={},
            tail=None,
            order=[]
        )

def build_templates_for_model(model_name):
    """Build templates for a specific model."""
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
    
    # Build templates for each component
    components_with_templates = []
    successful_builds = 0
    
    for i, item in enumerate(raw_chunks):
        if 'component' not in item:
            continue
            
        comp_data = item['component']
        
        # Extract component data
        text = comp_data.get('text', '')
        role = comp_data.get('role', [])
        means = comp_data.get('means')
        ends = comp_data.get('ends')
        original_text = comp_data.get('original_text', text)
        component_id = comp_data.get('id')
        
        print(f"Building template for component {i+1}/{len(raw_chunks)}: {text[:50]}...")
        
        # Build template
        template = build_template_from_components(role, means, ends, text)
        
        if template:
            successful_builds += 1
            
            # Create component with built template
            component_with_template = {
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
                'original_text': original_text,
                'model_source': model_name
            }
            
            components_with_templates.append({
                'component': component_with_template
            })
    
    print(f"Successfully built templates for {successful_builds}/{len(raw_chunks)} components")
    return components_with_templates

def main():
    """Build templates for all models."""
    print("BUILDING TEMPLATES PER MODEL")
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
                # Save to file
                output_file = output_dir / f"chunk_{model_name}_with_templates.json"
                
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
    print(f"\nOutput files:")
    for model_name in _end:
        print(f"  - chunked_story/chunk_{model_name}_with_templates.json")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()