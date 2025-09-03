#!/usr/bin/env python3
"""Build templates using the chunker and save them to a file."""

import sys
import json
from pathlib import Path

# Add root directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_qus_analyzer.chunker.models import QUSChunkerModel
from llm_qus_analyzer.chunker.parser import TemplateParser
from llm_qus_analyzer import Settings, LLMClient
from dataclasses import asdict

def load_user_stories():
    """Load user stories from the chunked data to get the original texts."""
    chunk_file = Path("chunked_story/chunk_best.json")
    
    with open(chunk_file, 'r') as f:
        raw_chunks = json.load(f)
    
    # Extract original user story texts
    user_stories = []
    story_ids = []
    for item in raw_chunks:
        if 'component' in item:
            comp_data = item['component']
            original_text = comp_data.get('original_text', comp_data.get('text', ''))
            user_stories.append(original_text)
            story_ids.append(comp_data.get('id', f"story_{len(story_ids)}"))
    
    print(f"Loaded {len(user_stories)} user stories")
    return user_stories, story_ids

def build_and_save_templates():
    """Use the chunker to build proper templates for all user stories and save them."""
    print("BUILDING AND SAVING TEMPLATES WITH CHUNKER")
    print("=" * 60)
    
    # Load user stories
    user_stories, story_ids = load_user_stories()
    
    # Setup LLM client and chunker
    setting = Settings()
    setting.configure_paths_and_load(
        env_path=Path('../.env'),
        model_config_path=Path('../models.yaml'),
    )
    clients = LLMClient(from_settings=setting)
    chunker = QUSChunkerModel()
    
    # Use the first available model
    model_idx = 0
    
    print(f"Processing {len(user_stories)} user stories with chunker...")
    
    # Process all user stories with the chunker to build templates
    try:
        results = chunker.analyze_list(clients, model_idx, user_stories, story_ids)
        
        # Convert components to serializable format
        serializable_components = []
        for component, usage in results:
            component_dict = {
                'component': {
                    'text': component.text,
                    'role': component.role,
                    'means': component.means,
                    'ends': component.ends,
                    'template': {
                        'text': component.template.text,
                        'chunk': component.template.chunk,
                        'tail': component.template.tail,
                        'order': component.template.order
                    },
                    'id': component.id,
                    'original_text': component.original_text
                },
                'usage': {
                    'duration': usage.duration,
                    'num_token_in': usage.num_token_in,
                    'num_token_out': usage.num_token_out
                }
            }
            serializable_components.append(component_dict)
        
        print(f"Successfully built templates for {len(serializable_components)} user stories")
        
        # Save to file
        output_file = Path("chunked_story/chunked_with_templates.json")
        with open(output_file, 'w') as f:
            json.dump(serializable_components, f, indent=2)
        
        print(f"Templates saved to {output_file}")
        
        # Also save a simplified version with just the templates for easy inspection
        template_summary = []
        for item in serializable_components:
            comp_data = item['component']
            template_summary.append({
                'id': comp_data['id'],
                'original_text': comp_data['original_text'],
                'template_text': comp_data['template']['text'],
                'components': comp_data['template']['chunk'],
                'order': comp_data['template']['order']
            })
        
        summary_file = Path("chunked_story/template_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(template_summary, f, indent=2)
        
        print(f"Template summary saved to {summary_file}")
        
        return serializable_components
        
    except Exception as e:
        print(f"Error building templates: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    try:
        build_and_save_templates()
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()