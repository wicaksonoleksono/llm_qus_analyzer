#!/usr/bin/env python3
"""Build templates from existing chunked data without re-running the LLM."""

import sys
import json
from pathlib import Path

# Add root directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_qus_analyzer.chunker.parser import TemplateParser, Template
from llm_qus_analyzer.chunker.models import QUSComponent

def load_chunked_data():
    """Load the best quality chunked data."""
    chunk_file = Path("chunked_story/chunk_best.json")
    
    with open(chunk_file, 'r') as f:
        raw_chunks = json.load(f)
    
    print(f"Loaded {len(raw_chunks)} chunked components")
    return raw_chunks

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

def build_templates_from_chunked_data():
    """Build templates from existing chunked data."""
    print("BUILDING TEMPLATES FROM CHUNKED DATA")
    print("=" * 50)
    
    # Load chunked data
    raw_chunks = load_chunked_data()
    
    # Process each component to build templates
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
            
            # Create enhanced component with template
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
                },
                'usage': item.get('usage', {
                    'duration': 0,
                    'num_token_in': 0,
                    'num_token_out': 0
                })
            }
            
            enhanced_components.append(enhanced_component)
    
    print(f"Successfully built templates for {len(enhanced_components)} components")
    
    # Save enhanced components with templates
    output_file = Path("chunked_story/chunked_with_built_templates.json")
    with open(output_file, 'w') as f:
        json.dump(enhanced_components, f, indent=2)
    
    print(f"Enhanced components with templates saved to {output_file}")
    
    # Save template summary for easy inspection
    template_summary = []
    for item in enhanced_components:
        comp_data = item['component']
        template_summary.append({
            'id': comp_data['id'],
            'original_text': comp_data['original_text'],
            'template_text': comp_data['template']['text'],
            'components': comp_data['template']['chunk'],
            'order': comp_data['template']['order'],
            'tail': comp_data['template']['tail']
        })
    
    summary_file = Path("chunked_story/built_template_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(template_summary, f, indent=2)
    
    print(f"Template summary saved to {summary_file}")
    
    return enhanced_components

def create_qus_components_with_templates(enhanced_components):
    """Convert enhanced components to QUSComponent objects with templates."""
    qus_components = []
    
    for item in enhanced_components:
        comp_data = item['component']
        template_data = comp_data['template']
        
        # Create template object
        template = Template(
            text=template_data['text'],
            chunk=template_data['chunk'],
            tail=template_data['tail'],
            order=template_data['order']
        )
        
        # Create QUSComponent
        qus_component = QUSComponent(
            text=comp_data['text'],
            role=comp_data['role'],
            means=comp_data['means'],
            ends=comp_data['ends'],
            template=template,
            id=comp_data['id'],
            original_text=comp_data['original_text']
        )
        
        qus_components.append(qus_component)
    
    return qus_components

if __name__ == "__main__":
    try:
        enhanced_components = build_templates_from_chunked_data()
        print(f"\nTemplate building complete!")
        print(f"Created {len(enhanced_components)} components with templates")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()