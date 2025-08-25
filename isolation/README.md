# QUS Analyzer - Isolation Environment

## Chunker (cb.py)

The chunker preprocesses user stories by extracting Role, Means, and Ends components using LLM analysis.

### Usage

```bash
# Chunk specific criteria
python cb.py -c <criteria_name>

# Chunk all available criteria
python cb.py --all

# List available criteria
python cb.py --list
```

### Flags

| Flag | Description | Example |
|------|-------------|---------|
| `-c, --criteria` | Chunk specific criteria | `python cb.py -c atomic` |
| `--all` | Chunk all available criteria | `python cb.py --all` |
| `--list` | List available criteria | `python cb.py --list` |

### Available Criteria

**Individual Analyzers:**
- `atomic`, `conceptually-sound`, `estimatable`, `full-sentence`, `minimal`, `problem-oriented`, `unambiguous`, `well-formed`

**Set Analyzers:**
- `conflict-free`, `complete`, `independent`, `unique`

### How Chunking Works

1. **Input**: User stories from `testdata/by-criteria/{criteria}.json`
2. **LLM Analysis**: Extracts Role, Means, Ends components using structured prompts
3. **Template Generation**: Creates reusable templates (e.g., "As a {ROLE}, I want to {MEANS} so that {ENDS}")
4. **Output**: Structured chunks saved to `pre-chunked/{criteria}_chunks.json`

### Chunk Structure

```json
{
  "story_1": {
    "text": "Expanded user story",
    "role": ["User"],
    "means": "action to perform",
    "ends": "desired outcome",
    "id": "story_1",
    "original_story": "Original user story text",
    "template_data": {
      "text": "Template pattern",
      "chunk": {...},
      "order": ["[MEANS]", "[ENDS]"]
    }
  }
}
```

### Notes

- Chunks are cached - only new stories are processed on subsequent runs
- The chunker may infer roles when not explicitly stated (known behavior)
- Each criteria requires corresponding test data in `testdata/by-criteria/`
- Chunks are required before running analysis with `an.py`