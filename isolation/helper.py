
from typing import List,Text
from dataclasses import asdict
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from llm_qus_analyzer.chunker.models import QUSComponent
from llm_qus_analyzer.chunker.parser import Template

def chunk_stories_to_json(chunker,clients,user_stories, model_idx=0, story_ids=None):
    """Chunk multiple user stories and return JSON-ready dicts."""
    results = chunker.analyze_list(clients, model_idx, user_stories, story_ids)
    print(model_idx)
    return [
        {
            'component': asdict(component),
            'usage': asdict(usage)
        }
        for component, usage in results
    ]

def analyze_individual_to_json(analyzer, client, model_idx, component):
    """Analyze single component with individual analyzer and return JSON-ready dict."""
    violations, usage_dict = analyzer.run(client, model_idx, component)
    return {
        'component_id': component.id or 'unknown',
        'violations': [asdict(violation) for violation in violations],
        'usage': {key: asdict(usage) for key, usage in usage_dict.items()}
    }

def analyze_set_to_json(analyzer, client, model_idx, components):
    """Analyze components with set analyzer and return JSON-ready dict."""
    result = analyzer.run(client, model_idx, components)
    
    # Handle different return formats from set analyzers
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], tuple):
        # UniformAnalyzer format: list of (violations, usage) tuples
        return [
            {
                'component_id': components[i].id or f'component_{i}',
                'violations': [asdict(violation) for violation in violations],
                'usage': {key: asdict(usage) for key, usage in usage_dict.items()} if isinstance(usage_dict, dict) else {}
            }
            for i, (violations, usage_dict) in enumerate(result)
        ]
    elif isinstance(result, tuple) and len(result) == 2:
        # UniqueAnalyzer format: (violations, usage_dict) 
        violations, usage_dict = result
        return {
            'violations': [asdict(violation) for violation in violations],
            'usage': {key: asdict(usage) for key, usage in usage_dict.items()}
        }
    else:
        # Fallback for unknown format
        return {'result': result}

def reconstruct_component_from_json(component_data):
    """Reconstruct QUSComponent from JSON data."""
    template_data = component_data['template']
    template = Template(
        text=template_data['text'],
        chunk=template_data['chunk'],
        tail=template_data['tail'],
        order=template_data['order']
    )
    
    return QUSComponent(
        text=component_data['text'],
        role=component_data['role'],
        means=component_data['means'],
        ends=component_data['ends'],
        template=template,
        id=component_data.get('id'),
        original_text=component_data.get('original_text')
    )

def csv_loader(pth:str)-> List[Text]:
    import pandas 
    story=pandas.read_csv(pth)
    return list(story.iloc[:, 1])  # Column 1 contains the actual user stories (Text column)