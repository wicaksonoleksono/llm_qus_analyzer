# LLM QUS Analyzer

A quality user story analyzer using LLM

## Installation

```
pip install https://github.com/SaveVic/llm_qus_analyzer/releases/download/v0.1.0/llm_qus_analyzer-0.1.0-py3-none-any.whl
```

## Usage

This guide provides all the necessary information to get you started configuring and using the package to analyze Quality-User-Story (QUS) components from text.

### Configuration

Before you can start analyzing user stories, you need to configure the settings, which involves setting up API keys and model configurations.

#### Environment Variables

The `LLMClient` requires API keys to communicate with the underlying language model provider. For now it only support Together AI provider. These keys should be stored in a `.env` file in the root directory of your project.

Create a file named `.env` and add your API key:

```
TOGETHER_API_KEY="your_api_key_here"
```

#### Model Configuration

The package uses a YAML file to define the configurations for the different LLMs you might want to use. This allows you to easily switch between models or update their settings without changing your code.

Create a file named `models.yaml` or anything you preffered with the following structure:

```yaml
models:
  - id: "deepseek-ai/DeepSeek-V3"
    name: "DeepSeek V3 Chat"

  - id: "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    name: "Llama 3.3 70B"

  - id: "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
    name: "Llama 4 Maverick Instruct"
```

#### Loading Settings in Code

The `Settings` class is used to load these configurations from your files.

```py
from llm_qus_analyzer import Settings

# Initialize the settings object
setting = Settings()

# Load configurations from the specified paths
setting.configure_paths_and_load(
    env_path=".env",
    model_config_path="models.yaml"
)
```

#### Create an LLM Client

The `LLMClient` handles all communication with the configured Large Language Model. It is initialized with the loaded `Settings` object and manages API requests, authentication, and responses.

```py
from llm_qus_analyzer import LLMClient

# The client is configured with your settings and is ready to make API calls.
client = LLMClient(from_settings=setting)
```

### Chunker

The first model used for this task is the `QUSChunkerModel`, which identifies the following parts of a user story:

- Role: Who is performing the action? (e.g., "As a manager...")

- Means: What action do they want to perform? (e.g., "...I want to be able to understand my colleagues' progress...")

- Ends: What is the ultimate goal or outcome? (e.g., "...so I can better report our successes and failures.")

First, you create an instance of the chunker model. This object holds the logic for how to analyze a user story.

```py
from llm_qus_analyzer import QUSChunkerModel

chunker = QUSChunkerModel()
```

You then call the `analyze_single` method. This is the main function that orchestrates the entire process from the single user story.

```py
# Select which model configuration to use from your models.yaml (0 for the first one).
model_idx = 0

# Define the user story to be analyzed
user_story = "As a manager, I want to be able to understand my colleagues progress, so I can better report our sucess and failures."

# The chunker model processes the text using the specified LLM.
component, usage = chunker.analyze_single(client, model_idx, user_story)
```

Now you can get each component, template, and LLM usage.

```bash
>>> component.role
['manager']
>>> component.means
'understand my colleagues progress'
>>> component.ends
'better report our sucess and failures'
>>> component.template.text
'As a {ROLE}, I am want to be able to {MEANS}, so I can {ENDS}.'
>>> usage.duration
2.8091282844543457
>>> usage.num_token_in
487
>>> usage.num_token_out
78
```

```bash
 Client Setup (Required First)
  from llm_qus_analyzer.client import LLMClient

  # Initialize the LLM client
  client = LLMClient()
  model_idx = 0  # Which LLM model to use

  2. Component Preparation
  from llm_qus_analyzer.chunker.models import QUSComponent

  # Your user stories as QUSComponent objects
  component1 = QUSComponent(role="user", means="login", ends="access account", text="As a
  user...")
  component2 = QUSComponent(role="admin", means="delete", ends="remove data", text="As an
  admin...")

  3. Analysis Execution

  For Individual Analysis:
  # Analyze single user story
  violations, usage = ConceptuallyAnalyzer.run(client, model_idx, component1)

  For Set Analysis (Pairwise):
  # Compare two user stories
  violations, usage = UniqueAnalyzer.analyze_pairwise(client, model_idx, component1, component2)

  For Set Analysis (Full Set):
  # Analyze entire set at once
  components = [component1, component2, component3]
  violations, usage = UniqueAnalyzer.analyze_full_set(client, model_idx, components)
```

```bash

How to use
 Set Pattern:

  # Pairwise
  violations, usage = Analyzer.run(client, model_idx, comp1, comp2, mode="pairwise")

  # Fullset
  violations, usage = Analyzer.run(client, model_idx, components, mode="fullset")

  Complete Usage Example:

  from llm_qus_analyzer.client import LLMClient
  from llm_qus_analyzer.individual import ConceptuallyAnalyzer
  from llm_qus_analyzer.set import ConflictFreeAnalyzer, UniqueAnalyzer

  # Setup
  client = LLMClient()
  stories = [component1, component2, component3]

  # Individual analysis
  for i, story in enumerate(stories):
      violations, usage = ConceptuallyAnalyzer.run(client, 0, story)
      print(f"Story {i}: {len(violations)} violations")

  # Set analysis - Pairwise
  violations, usage = ConflictFreeAnalyzer.run(client, 0, stories[0], stories[1],
  mode="pairwise")
  print(f"Conflict check: {len(violations)} conflicts")

  # Set analysis - Fullset
  violations, usage = UniqueAnalyzer.run(client, 0, stories, mode="fullset")
  print(f"Uniqueness check: {len(violations)} duplicates")
```
