import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from llm_qus_analyzer import Settings
from llm_qus_analyzer import LLMClient
from llm_qus_analyzer import QUSChunkerModel
from helper import csv_loader,chunk_stories_to_json
setting = Settings()
setting.configure_paths_and_load(
    env_path=Path('/home/wicaksonolxn/Documents/KJ/llm_qus_analyzer/isolation/.env'),
    model_config_path=Path('/home/wicaksonolxn/Documents/KJ/llm_qus_analyzer/isolation/models.yaml'),
)
clients=LLMClient(from_settings=setting)
chunker = QUSChunkerModel()
data = csv_loader('/home/wicaksonolxn/Documents/KJ/llm_qus_analyzer/isolation/ground_truth.csv')
output_dir = Path("chunked_story")
output_dir.mkdir(exist_ok=True)
for model_idx, model_name in enumerate(clients.names):
    output_file = output_dir / f"{model_name}.json"
    
    # Skip if already exists
    if output_file.exists():
        print(f"Skipping {model_name} - already exists at {output_file}")
        continue
    
    print(f"Processing {model_name}...")
    # Generate story IDs for proper component identification
    story_ids = [f"story_{i+1}" for i in range(len(data))]
    
    results = chunk_stories_to_json(chunker, clients, data, model_idx=model_idx, story_ids=story_ids)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Completed {model_name} -> {output_file}")