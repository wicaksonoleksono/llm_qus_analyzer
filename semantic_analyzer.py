#!/usr/bin/env python3
"""
Semantic analyzer using pt categories for proper analysis routing.
"""

from llm_qus_analyzer.settings import Settings
from llm_qus_analyzer.chunker.models import QUSChunkerModel
from llm_qus_analyzer.client import LLMClient

# Set analyzers
from llm_qus_analyzer.set.conflict_free import ConflictFreeAnalyzer
from llm_qus_analyzer.set.complete import CompleteAnalyzer
from llm_qus_analyzer.set.independent import IndependentAnalyzer
from llm_qus_analyzer.set.unique import UniqueAnalyzer
from llm_qus_analyzer.set.uniform import UniformAnalyzer

# Individual analyzers
from llm_qus_analyzer.individual.well_form import WellFormAnalyzer
from llm_qus_analyzer.individual.minimal import MinimalAnalyzer
from llm_qus_analyzer.individual.atomic import AtomicAnalyzer
from llm_qus_analyzer.individual.conceptually import ConceptuallySoundAnalyzer
from llm_qus_analyzer.individual.problem_oriented import ProblemOrientedAnalyzer
from llm_qus_analyzer.individual.full_sentence import FullSentenceAnalyzer
from llm_qus_analyzer.individual.unambigous import UnambiguousAnalyzer
from llm_qus_analyzer.individual.estimatable import EstimatableAnalyzer
import json
import sys
import re
import time
from pathlib import Path
from typing import List, Dict, Any

sys.path.append(str(Path(__file__).parent))


# PT category mappings to analyzers
ANALYZER_MAP = {
    # Set analyzers (handle pairs/sets)
    "conflict-free": ConflictFreeAnalyzer,
    "complete": CompleteAnalyzer,
    "independent": IndependentAnalyzer,
    "uniform": UniformAnalyzer,
    "unique": UniqueAnalyzer,

    # Individual analyzers (handle single stories)
    "full-sentence": FullSentenceAnalyzer,
    "problem-oriented": ProblemOrientedAnalyzer,
    "unambiguous": UnambiguousAnalyzer,
    "conceptually-sound": ConceptuallySoundAnalyzer,
    "estimatable": EstimatableAnalyzer,
}

# Part mappings for each set analyzer (mirrors their internal _PART_MAP)
ANALYZER_PART_MAPS = {
    "conflict-free": {
        "[Role]": "role",
        "[Means]": "means",
        "[Ends]": "ends",
    },
    "complete": {
        "[Means]": "means",
    },
    "independent": {
        "[Role]": "role",
        "[Means]": "means",
        "[Ends]": "ends",
    },
    "unique": {
        "semantic_duplicate": "semantic",
    },
    "uniform": {},  # Template-based, no LLM part mapping
}

SET_CATEGORIES = ["complete", "conflict-free", "independent", "uniform", "unique"]
INDIVIDUAL_CATEGORIES = ["full-sentence", "problem-oriented", "unambiguous",
                         "conceptually-sound", "estimatable", "fs", "es", "una"]


def load_test_data(file_path: str) -> List[Dict]:
    """Load test data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_chunks(chunks: Dict, file_path: str):
    """Save chunked components to file."""
    with open(file_path, 'w') as f:
        json.dump(chunks, f, indent=2, default=str)
    print(f"Chunks saved to {file_path}")


def load_chunks(file_path: str) -> Dict:
    """Load chunked components from file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def chunk_all_stories(client: LLMClient, chunker: QUSChunkerModel,
                      test_data: List[Dict], chunks_file: str) -> Dict:
    """Chunk all stories and save components with enhanced tagging to avoid re-chunking."""

    # Try to load existing chunks
    existing_chunks = load_chunks(chunks_file)

    all_chunks = {}
    stories_to_chunk = []

    # Collect all unique stories with their context/tags
    story_metadata = {}  # Maps story_hash -> metadata

    for item in test_data:
        if "stories" in item and len(item["stories"]) == 2:
            # Pairwise stories (set analyzers)
            pt_category = item.get("pt", "unknown")
            expected_violation = item.get("violation", "")

            for j, story in enumerate(item["stories"]):
                story_hash = hash(story)

                # Store metadata for this story
                if str(story_hash) not in story_metadata:
                    story_metadata[str(story_hash)] = {
                        "pt_categories": set(),
                        "expected_violations": set(),
                        "analysis_types": set(),
                        "pair_contexts": []
                    }

                story_metadata[str(story_hash)]["pt_categories"].add(pt_category)
                story_metadata[str(story_hash)]["expected_violations"].add(expected_violation)
                story_metadata[str(story_hash)]["analysis_types"].add("pairwise")
                story_metadata[str(story_hash)]["pair_contexts"].append({
                    "pt": pt_category,
                    "violation": expected_violation,
                    "position_in_pair": j + 1,
                    "paired_with_hash": str(hash(item["stories"][1-j]))
                })

                if str(story_hash) not in existing_chunks:
                    stories_to_chunk.append((story, story_hash))
                else:
                    all_chunks[str(story_hash)] = existing_chunks[str(story_hash)]

        elif "story" in item:
            # Individual stories (individual analyzers)
            story = item["story"]
            story_hash = hash(story)
            pt_category = item.get("pt", "unknown")
            expected_violation = item.get("violation", "")

            # Store metadata for this story
            if str(story_hash) not in story_metadata:
                story_metadata[str(story_hash)] = {
                    "pt_categories": set(),
                    "expected_violations": set(),
                    "analysis_types": set(),
                    "individual_contexts": []
                }

            story_metadata[str(story_hash)]["pt_categories"].add(pt_category)
            story_metadata[str(story_hash)]["expected_violations"].add(expected_violation)
            story_metadata[str(story_hash)]["analysis_types"].add("individual")
            story_metadata[str(story_hash)]["individual_contexts"].append({
                "pt": pt_category,
                "violation": expected_violation
            })

            if str(story_hash) not in existing_chunks:
                stories_to_chunk.append((story, story_hash))
            else:
                all_chunks[str(story_hash)] = existing_chunks[str(story_hash)]

    if not stories_to_chunk:
        print("All stories already chunked!")
        return existing_chunks

    print(f"Chunking {len(stories_to_chunk)} new stories...")

    # Chunk new stories
    for i, (story, story_hash) in enumerate(stories_to_chunk):
        print(f"Chunking story {i+1}/{len(stories_to_chunk)}: {story[:50]}...")

        try:
            component, usage = chunker.analyze_single(client, 0, story, f"story_{story_hash}")

            # Get metadata for this story
            metadata = story_metadata.get(str(story_hash), {})

            # Convert sets to lists for JSON serialization
            serializable_metadata = {
                "pt_categories": list(metadata.get("pt_categories", set())),
                "expected_violations": list(metadata.get("expected_violations", set())),
                "analysis_types": list(metadata.get("analysis_types", set())),
                "pair_contexts": metadata.get("pair_contexts", []),
                "individual_contexts": metadata.get("individual_contexts", [])
            }

            # Store complete template data for proper reconstruction
            template_data = {
                "text": component.template.text,
                "chunk": component.template.chunk,
                "tail": component.template.tail,
                "order": component.template.order
            }

            # Convert component to serializable dict with enhanced tags
            chunk_data = {
                "text": component.text,
                "role": component.role,
                "means": component.means,
                "ends": component.ends,
                "id": component.id,
                "original_story": story,
                "story_hash": str(story_hash),

                # Store complete template for proper reconstruction
                "template_data": template_data,

                # Enhanced tagging for targeted analysis
                "tags": serializable_metadata,

                "usage": {
                    "duration": usage.duration,
                    "num_token_in": usage.num_token_in,
                    "num_token_out": usage.num_token_out
                },

                # Quick access fields
                "primary_pt": serializable_metadata["pt_categories"][0] if serializable_metadata["pt_categories"] else "unknown",
                "primary_violation": serializable_metadata["expected_violations"][0] if serializable_metadata["expected_violations"] else "",
                "is_pairwise": "pairwise" in serializable_metadata["analysis_types"],
                "is_individual": "individual" in serializable_metadata["analysis_types"]
            }

            all_chunks[str(story_hash)] = chunk_data

            # Save periodically
            if (i + 1) % 5 == 0:
                save_chunks(all_chunks, chunks_file)

            time.sleep(1)  # Rate limiting

        except Exception as e:
            print(f"Error chunking story {i+1}: {e}")
            continue

    # Add metadata to existing chunks that don't have it
    for story_hash_str, metadata in story_metadata.items():
        if story_hash_str in all_chunks and "tags" not in all_chunks[story_hash_str]:
            serializable_metadata = {
                "pt_categories": list(metadata.get("pt_categories", set())),
                "expected_violations": list(metadata.get("expected_violations", set())),
                "analysis_types": list(metadata.get("analysis_types", set())),
                "pair_contexts": metadata.get("pair_contexts", []),
                "individual_contexts": metadata.get("individual_contexts", [])
            }

            all_chunks[story_hash_str]["tags"] = serializable_metadata
            all_chunks[story_hash_str]["primary_pt"] = serializable_metadata["pt_categories"][0] if serializable_metadata["pt_categories"] else "unknown"
            all_chunks[story_hash_str]["primary_violation"] = serializable_metadata["expected_violations"][0] if serializable_metadata["expected_violations"] else ""
            all_chunks[story_hash_str]["is_pairwise"] = "pairwise" in serializable_metadata["analysis_types"]
            all_chunks[story_hash_str]["is_individual"] = "individual" in serializable_metadata["analysis_types"]

    # Final save
    save_chunks(all_chunks, chunks_file)
    return all_chunks


def filter_by_pt_category(test_data: List[Dict], category: str) -> List[Dict]:
    """Filter test data by pt category."""
    return [item for item in test_data if item.get("pt") == category]


def query_chunks_by_tags(chunks: Dict, **filters) -> Dict:
    """Query chunks by various tags and filters.

    Args:
        chunks: Dictionary of story chunks
        **filters: Filtering criteria like pt_category, violation_type, analysis_type, etc.

    Returns:
        Filtered chunks dictionary
    """
    filtered = {}

    for chunk_hash, chunk_data in chunks.items():
        tags = chunk_data.get("tags", {})

        # Check pt category filter
        if "pt_category" in filters:
            if filters["pt_category"] not in tags.get("pt_categories", []):
                continue

        # Check violation filter
        if "violation_contains" in filters:
            violations = tags.get("expected_violations", [])
            if not any(filters["violation_contains"].lower() in v.lower() for v in violations):
                continue

        # Check analysis type filter
        if "analysis_type" in filters:
            if filters["analysis_type"] not in tags.get("analysis_types", []):
                continue

        # Check component content filters
        if "has_role" in filters and filters["has_role"]:
            if not chunk_data.get("role"):
                continue

        if "has_means" in filters and filters["has_means"]:
            if not chunk_data.get("means"):
                continue

        if "has_ends" in filters and filters["has_ends"]:
            if not chunk_data.get("ends"):
                continue

        filtered[chunk_hash] = chunk_data

    return filtered


def print_chunk_summary(chunks: Dict):
    """Print a summary of chunks with their tags."""
    if not chunks:
        print("No chunks found!")
        return

    print(f"\nChunk Summary ({len(chunks)} stories):")
    print("="*50)

    # Count by PT categories
    pt_counts = {}
    violation_counts = {}

    for chunk_data in chunks.values():
        tags = chunk_data.get("tags", {})

        # Count PT categories
        for pt in tags.get("pt_categories", []):
            pt_counts[pt] = pt_counts.get(pt, 0) + 1

        # Count violation types
        for violation in tags.get("expected_violations", []):
            violation_counts[violation] = violation_counts.get(violation, 0) + 1

    print("PT Categories:")
    for pt, count in sorted(pt_counts.items()):
        print(f"  {pt}: {count} stories")

    print(f"\nTop violation types:")
    for violation, count in sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {violation}: {count} stories")


def clean_llm_json(raw_content: str) -> str:
    """Clean LLM response to extract valid JSON."""
    # Remove code blocks
    content = re.sub(r'```json\s*', '', raw_content)
    content = re.sub(r'```\s*', '', content)

    # Remove common prefixes
    content = re.sub(r'^[^{]*', '', content)

    # Find the first { and last }
    start = content.find('{')
    end = content.rfind('}')

    if start != -1 and end != -1 and end > start:
        return content[start:end+1]

    return content.strip()


def analyze_universal(client: LLMClient, chunker: QUSChunkerModel,
                      test_data: List[Dict], category: str) -> List[Dict]:
    """Universal analyzer that handles both set and individual analysis based on pt category."""

    if category not in ANALYZER_MAP:
        raise ValueError(f"Unknown category: {category}. Available: {list(ANALYZER_MAP.keys())}")

    analyzer = ANALYZER_MAP[category]
    results = []

    print(f"Using {analyzer.__name__} for category '{category}'")

    if category in SET_CATEGORIES:
        # Set analysis (pairwise or full set)
        return analyze_set_category(client, chunker, test_data, analyzer, category)
    else:
        # Individual analysis (single stories)
        return analyze_individual_category(client, chunker, test_data, analyzer)


def safe_analyzer_call(analyzer_func, *args, max_retries=3, **kwargs):
    """Safely call analyzer with error handling and retry logic for API failures."""
    for attempt in range(max_retries):
        try:
            return analyzer_func(*args, **kwargs)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                print("Max retries reached. This usually means the LLM returned malformed JSON.")
                return [], {}
            time.sleep(2)  # Wait before retry
        except Exception as e:
            error_msg = str(e).lower()
            if "503" in error_msg or "service unavailable" in error_msg or "503" in str(e):
                print(f"API service unavailable (attempt {attempt + 1}): Together API is down")
                if attempt == max_retries - 1:
                    print("Max retries reached. API service is unavailable.")
                    return [], {}
                print(f"Waiting 10 seconds before retry...")
                time.sleep(10)  # Longer wait for API issues
            elif "429" in error_msg or "rate limit" in error_msg:
                print(f"Rate limited (attempt {attempt + 1}): Too many requests")
                if attempt == max_retries - 1:
                    print("Max retries reached. Rate limit exceeded.")
                    return [], {}
                print(f"Waiting 15 seconds before retry...")
                time.sleep(15)  # Wait for rate limit
            else:
                print(f"Analyzer error: {e}")
                return [], {}

    return [], {}


def create_component_from_chunk(chunk_data: Dict, story_id: str = None):
    """Create QUSComponent from cached chunk data."""
    from llm_qus_analyzer.chunker.models import QUSComponent
    from llm_qus_analyzer.chunker.parser import Template

    # Create a simple template (we don't need the full template for analysis)
    template = Template(chunk_data["text"], [], [])

    return QUSComponent(
        text=chunk_data["text"],
        role=chunk_data["role"],
        means=chunk_data["means"],
        ends=chunk_data["ends"],
        template=template,
        id=story_id or chunk_data.get("id")
    )


def analyze_set_category_with_chunks(client: LLMClient, test_data: List[Dict],
                                     analyzer, category: str, chunks: Dict) -> List[Dict]:
    """Handle set analyzers using pre-chunked components."""
    results = []
    part_map = ANALYZER_PART_MAPS.get(category, {})

    for i, item in enumerate(test_data):
        if "stories" not in item or len(item["stories"]) != 2:
            continue

        stories = item["stories"]
        violation = item.get("violation", "")

        print(f"Analyzing pair {i+1}: {violation}")

        try:
            # Get components from cached chunks
            hash1 = str(hash(stories[0]))
            hash2 = str(hash(stories[1]))

            if hash1 not in chunks or hash2 not in chunks:
                print(f"Missing chunks for pair {i+1}, skipping...")
                continue

            comp1 = create_component_from_chunk(chunks[hash1], f"story_{i+1}_1")
            comp2 = create_component_from_chunk(chunks[hash2], f"story_{i+1}_2")

            # Run pairwise analysis with error handling
            violations, usage_dict = safe_analyzer_call(
                analyzer.analyze_pairwise, client, 0, comp1, comp2
            )

            # Small delay to avoid API rate limits
            time.sleep(1)

            # Process violations with proper part mapping
            processed_violations = []
            for v in violations:
                # Get parts with proper mapping based on analyzer type
                if hasattr(v, 'first_parts') and hasattr(v, 'second_parts'):
                    first_parts = list(v.first_parts)
                    second_parts = list(v.second_parts)
                else:
                    # Fallback for analyzers without explicit part separation
                    parts = list(v.parts) if hasattr(v, 'parts') else []
                    first_parts = parts
                    second_parts = parts

                processed_violations.append({
                    "issue": v.issue,
                    "suggestion": v.suggestion,
                    "first_parts": first_parts,
                    "second_parts": second_parts,
                    "analyzer_focus": list(part_map.values()) if part_map else ["all"]
                })

            results.append({
                "pair_id": i + 1,
                "expected_violation": violation,
                "stories": stories,
                "analyzer": analyzer.__name__,
                "analyzer_focus": list(part_map.values()) if part_map else ["template-based"],
                "components": {
                    "story_1": {"role": comp1.role, "means": comp1.means, "ends": comp1.ends},
                    "story_2": {"role": comp2.role, "means": comp2.means, "ends": comp2.ends}
                },
                "detected_violations": processed_violations,
                "has_conflict": len(violations) > 0,
                "usage_stats": usage_dict
            })

        except Exception as e:
            print(f"Error processing pair {i+1}: {e}")
            # Add error result
            results.append({
                "pair_id": i + 1,
                "expected_violation": violation,
                "stories": stories,
                "analyzer": analyzer.__name__,
                "error": str(e),
                "has_conflict": False,
                "detected_violations": []
            })

    return results


def analyze_individual_category_with_chunks(client: LLMClient, test_data: List[Dict],
                                            analyzer, chunks: Dict) -> List[Dict]:
    """Handle individual analyzers using pre-chunked components."""
    results = []

    for i, item in enumerate(test_data):
        if "story" in item:
            story = item["story"]
            violation = item.get("violation", "")

            print(f"Analyzing story {i+1}: {violation}")

            try:
                # Get component from cached chunks
                story_hash = str(hash(story))

                if story_hash not in chunks:
                    print(f"Missing chunk for story {i+1}, skipping...")
                    continue

                component = create_component_from_chunk(chunks[story_hash], f"story_{i+1}")

                # Run individual analysis with error handling
                violations, usage_dict = safe_analyzer_call(
                    analyzer.run, client, 0, component
                )

                # Small delay to avoid API rate limits
                time.sleep(1)

                # Process violations
                processed_violations = []
                for v in violations:
                    processed_violations.append({
                        "issue": v.issue,
                        "suggestion": v.suggestion,
                        "parts": list(v.parts) if hasattr(v, 'parts') else []
                    })

                results.append({
                    "story_id": i + 1,
                    "expected_violation": violation,
                    "story": story,
                    "analyzer": analyzer.__name__,
                    "component": {
                        "role": component.role,
                        "means": component.means,
                        "ends": component.ends
                    },
                    "detected_violations": processed_violations,
                    "has_violation": len(violations) > 0,
                    "usage_stats": usage_dict
                })

            except Exception as e:
                print(f"Error processing story {i+1}: {e}")
                # Add error result
                results.append({
                    "story_id": i + 1,
                    "expected_violation": violation,
                    "story": story,
                    "analyzer": analyzer.__name__,
                    "error": str(e),
                    "has_violation": False,
                    "detected_violations": []
                })

    return results


def analyze_individual_category(client: LLMClient, chunker: QUSChunkerModel,
                                test_data: List[Dict], analyzer) -> List[Dict]:
    """Handle individual analyzers (single story analysis)."""
    results = []

    for i, item in enumerate(test_data):
        if "story" in item:
            # Single story format
            story = item["story"]
            violation = item.get("violation", "")

            print(f"Analyzing story {i+1}: {violation}")

            try:
                # Parse story
                component, _ = chunker.analyze_single(client, 0, story, f"story_{i+1}")

                # Run individual analysis with error handling
                violations, usage_dict = safe_analyzer_call(
                    analyzer.run, client, 0, component
                )

                # Small delay to avoid API rate limits
                time.sleep(1)

                # Process violations
                processed_violations = []
                for v in violations:
                    processed_violations.append({
                        "issue": v.issue,
                        "suggestion": v.suggestion,
                        "parts": list(v.parts) if hasattr(v, 'parts') else []
                    })

                results.append({
                    "story_id": i + 1,
                    "expected_violation": violation,
                    "story": story,
                    "analyzer": analyzer.__name__,
                    "component": {
                        "role": component.role,
                        "means": component.means,
                        "ends": component.ends
                    },
                    "detected_violations": processed_violations,
                    "has_violation": len(violations) > 0,
                    "usage_stats": usage_dict
                })

            except Exception as e:
                print(f"Error processing story {i+1}: {e}")
                # Add error result
                results.append({
                    "story_id": i + 1,
                    "expected_violation": violation,
                    "story": story,
                    "analyzer": analyzer.__name__,
                    "error": str(e),
                    "has_violation": False,
                    "detected_violations": []
                })

    return results


def main():
    """Main function for universal analysis using pt categories."""

    # Configuration
    test_data_path = Path(__file__).parent / "testdata" / "semantic_pragmatic_test.json"
    models_path = Path(__file__).parent / "models.yaml"
    env_path = Path(__file__).parent / ".env"
    chunks_file = Path(__file__).parent / "story_chunks.json"

    # Choose category to analyze (can be changed)
    target_category = "conflict-free"  # Change this to test different analyzers

    # Test configuration
    test_multiple = True  # Set to True to test all set analyzers
    max_items = 2  # Limit items per analyzer for testing

    # TWO-PHASE APPROACH: Default to chunking only
    chunk_only = True  # Set to True to only do chunking phase
    analyze_only = False  # Set to True to only do analysis phase (requires existing chunks)

    try:
        # Setup (only if not analyze-only mode)
        if not analyze_only:
            settings = Settings()
            settings.configure_paths_and_load(env_path if env_path.exists() else None, models_path)

            client = LLMClient(settings)
            chunker = QUSChunkerModel()

            print(f"Initialized with model: {client.names[0]}")

        print(f"Available categories: {list(ANALYZER_MAP.keys())}")

        # Load test data
        test_data = load_test_data(test_data_path)

        # PHASE 1: CHUNKING
        if not analyze_only:
            print(f"\n{'='*50}")
            print("PHASE 1: CHUNKING ALL STORIES")
            print(f"{'='*50}")

            chunks = chunk_all_stories(client, chunker, test_data, chunks_file)
            print(f"Chunked {len(chunks)} unique stories")

            # Show chunk summary with enhanced tags
            print_chunk_summary(chunks)

            if chunk_only:
                print("\nChunking phase complete! Set chunk_only=False to run analysis.")
                print("Enhanced chunks saved with PT categories, violation types, and context tags!")
                return
        else:
            print("Loading existing chunks...")
            chunks = load_chunks(chunks_file)
            if not chunks:
                print("No chunks found! Run chunking phase first (set analyze_only=False)")
                return

        # PHASE 2: ANALYSIS
        print(f"\n{'='*50}")
        print("PHASE 2: RUNNING ANALYSIS WITH CACHED CHUNKS")
        print(f"{'='*50}")

        # Setup client for analysis phase
        if analyze_only:
            settings = Settings()
            settings.configure_paths_and_load(env_path if env_path.exists() else None, models_path)
            client = LLMClient(settings)
            print(f"Initialized with model: {client.names[0]}")

        if test_multiple:
            # Test all set analyzers
            categories_to_test = SET_CATEGORIES
            print(f"Testing {len(categories_to_test)} set analyzers...")
        else:
            # Test single category
            categories_to_test = [target_category]

        all_results = {}

        for category in categories_to_test:
            print(f"\n{'='*40}")
            print(f"Analyzing {category.upper()}")
            print(f"{'='*40}")

            # Filter data for this category
            filtered_data = filter_by_pt_category(test_data, category)

            if not filtered_data:
                print(f"No test data found for category '{category}'")
                continue

            print(f"Found {len(filtered_data)} items for '{category}' analysis")
            print(f"Analyzer focus: {ANALYZER_PART_MAPS.get(category, 'template-based')}")

            # Get analyzer
            analyzer = ANALYZER_MAP[category]

            # Run analysis with cached chunks
            if category in SET_CATEGORIES:
                results = analyze_set_category_with_chunks(
                    client, filtered_data[:max_items], analyzer, category, chunks
                )
            else:
                results = analyze_individual_category_with_chunks(
                    client, filtered_data[:max_items], analyzer, chunks
                )

            all_results[category] = results

            # Results summary
            print(f"\nResults for '{category}':")
            print(f"Items analyzed: {len(results)}")

            for result in results:
                if "pair_id" in result:
                    print(f"\nPair {result['pair_id']}: {result['expected_violation']}")
                    print(f"Analyzer: {result.get('analyzer', 'Unknown')}")
                    print(f"Focus: {result.get('analyzer_focus', 'Unknown')}")
                    print(f"Conflicts detected: {result['has_conflict']}")
                else:
                    print(f"\nStory {result['story_id']}: {result['expected_violation']}")
                    print(f"Analyzer: {result.get('analyzer', 'Unknown')}")
                    print(f"Violations detected: {result['has_violation']}")

                if result.get('detected_violations'):
                    for v in result['detected_violations']:
                        print(f"  Issue: {v['issue']}")
                        if 'first_parts' in v:
                            print(f"  Parts affected: {v['first_parts']} | {v['second_parts']}")
                        else:
                            print(f"  Parts affected: {v.get('parts', [])}")

        # Save results
        if test_multiple:
            output_file = Path(__file__).parent / "all_analyzers_results.json"
        else:
            output_file = Path(__file__).parent / f"{target_category}_results.json"

        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\n{'='*50}")
        print(f"Results saved to {output_file}")
        print(f"Two-phase analysis complete!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
