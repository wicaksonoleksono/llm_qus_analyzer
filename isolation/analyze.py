import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from llm_qus_analyzer.individual import AtomicAnalyzer,MinimalAnalyzer,WellFormAnalyzer
from llm_qus_analyzer.set import UniformAnalyzer,UniqueAnalyzer
from llm_qus_analyzer import Settings, LLMClient
from helper import analyze_individual_to_json, analyze_set_to_json, reconstruct_component_from_json, _convert_to_serializable
from dataclasses import asdict

# Initialize settings and client  
setting = Settings()
setting.configure_paths_and_load(
    env_path=Path('/home/wicaksonolxn/Documents/KJ/llm_qus_analyzer/isolation/.env'),
    model_config_path=Path('/home/wicaksonolxn/Documents/KJ/llm_qus_analyzer/isolation/models.yaml'),
)
client = LLMClient(from_settings=setting)
# Separate rule-based analyzers from LLM-based analyzers
rule_based_analyzers = {"minimal": MinimalAnalyzer, "well-formed": WellFormAnalyzer}
llm_based_analyzers = {"atomic": AtomicAnalyzer}
set_analyzers = {"uniform": UniformAnalyzer}  # "unique": UniqueAnalyzer
output_dir = Path("analysis_results")
output_dir.mkdir(exist_ok=True)
chunked_dir = Path("chunked_story") 
for chunked_file in chunked_dir.glob("*.json"):
    model_name = chunked_file.stem
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(exist_ok=True)
    
    # Check if all analysis files already exist
    llm_analyzer_files = [model_output_dir / f"{analyzer}.json" for analyzer in llm_based_analyzers.keys()]
    rule_based_dir = output_dir / "rule_based"
    rule_based_dir.mkdir(exist_ok=True)
    rule_based_files = [rule_based_dir / f"{analyzer}.json" for analyzer in rule_based_analyzers.keys()]
    existing_files = llm_analyzer_files + rule_based_files
    
    if all(f.exists() for f in existing_files):
        print(f"Skipping {model_name} - all analysis files already exist")
        continue
    
    print(f"Processing {model_name}...")
    
    with open(chunked_file, 'r') as f:
        chunked_data = json.load(f)
    components = [reconstruct_component_from_json(item['component']) for item in chunked_data]
    
    model_idx = 0
    for idx, client_model_name in enumerate(client.names):
        safe_name = client_model_name.replace("/", "_").replace(" ", "_")
        if safe_name == model_name:
            model_idx = idx
            break
    
        # LLM-based analyzers - process each component per model
    for analyzer_name, analyzer_class in llm_based_analyzers.items():
        output_file = model_output_dir / f"{analyzer_name}.json"
        if output_file.exists():
            print(f"  Skipping {analyzer_name} - already exists")
            continue
            
        print(f"  Running {analyzer_name}...")
        results = [analyze_individual_to_json(analyzer_class, client, model_idx, component) 
                  for component in components]
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Rule-based analyzers - process each component once (not per model)
    # Create a shared directory for rule-based analyzer results
    rule_based_dir = output_dir / "rule_based"
    rule_based_dir.mkdir(exist_ok=True)
    
    for analyzer_name, analyzer_class in rule_based_analyzers.items():
        output_file = rule_based_dir / f"{analyzer_name}.json"
        # Skip if already exists
        if output_file.exists():
            print(f"  Skipping {analyzer_name} - already exists")
            continue
            
        print(f"  Running {analyzer_name} (rule-based, model-independent)...")
        # Use model_idx=0 since rule-based analyzers don't use the model
        results = [analyze_individual_to_json(analyzer_class, client, 0, component) 
                  for component in components]
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Set analyzers - process all components together
    for analyzer_name, analyzer_class in set_analyzers.items():
        output_file = model_output_dir / f"{analyzer_name}.json"
        if output_file.exists():
            print(f"  Skipping {analyzer_name} - already exists")
            continue
            
        print(f"  Running {analyzer_name}...")
        result = analyze_set_to_json(analyzer_class, client, model_idx, components)
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
    
    print(f"Completed {model_name}")