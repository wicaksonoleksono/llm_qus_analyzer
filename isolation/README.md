# LLM User Story Quality Analyzer - Evaluation Setup

This isolation environment provides a complete pipeline for evaluating LLM performance on user story quality analysis.

## Setup

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration Files

#### `.env` - API Keys
```
TOGETHER_API_KEY=your_together_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key  
OPENAI_API_KEY=your_openai_api_key
```

#### `models.yaml` - Model Configuration
```yaml
models:
  - id: "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
    name: "Llama 17B-Instruct FP8"
    source: "together"
  - id: "mistralai/Mistral-7B-Instruct-v0.3"
    source: "together"
    name: "Mistral 7B-Instruct v0.3"
  - id: "deepseek-chat"
    name: "deepseek-chat"
    source: "deepseek"
  - id: "gpt-4.1-mini-2025-04-14"
    name: "gpt-4.1-mini"
    source: "chatgpt"
```

### 3. Ground Truth Data
Ensure `ground_truth.csv` contains user stories with violation labels in format:
```csv
Text,GT
"As a user I want to login","Atomic,Minimal"
"As a manager I need access","Uniform"
```

## Pipeline Execution

### Step 1: Extract Ground Truth
```bash
python extract_gt.py
```
- Parses `ground_truth.csv`
- Creates `ground_truth_extracted.json`
- Converts comma-separated labels to lists

### Step 2: Chunk User Stories
```bash
python chunker.py
```
- Processes stories through each LLM model
- Extracts roles, means, ends components
- Creates templates from original text
- Outputs: `chunked_story/{model_name}.json`
- **Resume capability**: Skips existing files

### Step 3: Run Quality Analysis
```bash
python analyze.py
```
- Runs all analyzers (Atomic, Minimal, Uniform, Well-formed)
- Processes chunked stories through quality criteria
- Outputs: `analysis_results/{model_name}/{analyzer}.json`
- **Resume capability**: Skips existing analysis files

### Step 4: Collect Violations
```bash
python collect_violations.py
```
- Maps detected violations to story text
- Creates consistent format for evaluation
- Output: `violations_collected.json`

### Step 5: Calculate Metrics
```bash
python calculate_metrics.py
```
- Computes F1, Precision, Recall per model
- Provides per-criterion breakdown
- Outputs: `llm_stats.json`, `evaluation_results.json`

## Quality Analyzers

### Individual Analyzers
- **AtomicAnalyzer**: Detects compound user stories
- **MinimalAnalyzer**: Identifies stories with unnecessary details  
- **WellFormAnalyzer**: Validates story structure completeness

### Set Analyzers  
- **UniformAnalyzer**: Finds template inconsistencies across stories

## Output Files

### Analysis Results
```
analysis_results/
├── deepseek-chat/
│   ├── atomic.json
│   ├── minimal.json
│   ├── uniform.json
│   └── well-formed.json
├── gpt-4.1-mini/
└── ...
```

### Evaluation Metrics
- `llm_stats.json` - Clean per-model metrics
- `evaluation_results.json` - Detailed story-level results
- `violations_collected.json` - Collected violations per model

## Key Features

### Resume Functionality
All scripts check for existing output files and skip completed work:
- `chunker.py`: Skips if `chunked_story/{model}.json` exists
- `analyze.py`: Skips if `analysis_results/{model}/{analyzer}.json` exists

### Per-Criterion Metrics
The evaluation provides breakdown by quality criterion:
```
Per-Label Metrics:
Label        Precision  Recall   F1       TP   FP   FN  
--------------------------------------------------
Atomic       0.733      1.000    0.846    11   4    0   
Minimal      0.143      1.000    0.250    1    6    0   
Uniform      0.000      0.000    0.000    0    2    18  
```

### Template Preservation
The chunker uses original story text for template creation, not LLM-expanded text, ensuring authentic templates.

## Troubleshooting

### Common Issues
1. **Missing API Keys**: Check `.env` file configuration
2. **Model Source Errors**: Ensure `models.yaml` has `source` field for all models
3. **Template Issues**: Mistral may rewrite "I want" to "I am wanting" - this is expected model behavior

### Debug Output
- Chunker shows progress bars and model names
- UniformAnalyzer prints template debugging info
- All scripts show file skip messages for resume functionality