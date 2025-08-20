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
    "fs": FullSentenceAnalyzer,  # Alternative name
    "es": EstimatableAnalyzer,   # Alternative name
    "una": UnambiguousAnalyzer,  # Alternative name
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


def filter_by_pt_category(test_data: List[Dict], category: str) -> List[Dict]:
    """Filter test data by pt category."""
    return [item for item in test_data if item.get("pt") == category]


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


def analyze_set_category(client: LLMClient, chunker: QUSChunkerModel, 
                        test_data: List[Dict], analyzer, category: str) -> List[Dict]:
    """Handle set analyzers (pairwise analysis) with proper part mapping."""
    results = []
    part_map = ANALYZER_PART_MAPS.get(category, {})
    
    for i, item in enumerate(test_data):
        if "stories" not in item or len(item["stories"]) != 2:
            continue
            
        stories = item["stories"]
        violation = item.get("violation", "")
        
        print(f"Analyzing pair {i+1}: {violation}")
        
        try:
            # Parse stories
            comp1, _ = chunker.analyze_single(client, 0, stories[0], f"story_{i+1}_1")
            comp2, _ = chunker.analyze_single(client, 0, stories[1], f"story_{i+1}_2")
            
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
    
    # Choose category to analyze (can be changed)
    target_category = "conflict-free"  # Change this to test different analyzers
    
    # Test configuration
    test_multiple = True  # Set to True to test all set analyzers
    max_items = 2  # Limit items per analyzer for testing
    
    try:
        # Setup
        settings = Settings()
        settings.configure_paths_and_load(env_path if env_path.exists() else None, models_path)
        
        client = LLMClient(settings)
        chunker = QUSChunkerModel()
        
        print(f"Initialized with model: {client.names[0]}")
        print(f"Available categories: {list(ANALYZER_MAP.keys())}")
        
        # Load test data
        test_data = load_test_data(test_data_path)
        
        if test_multiple:
            # Test all set analyzers
            categories_to_test = SET_CATEGORIES
            print(f"\nTesting {len(categories_to_test)} set analyzers...")
        else:
            # Test single category
            categories_to_test = [target_category]
        
        all_results = {}
        
        for category in categories_to_test:
            print(f"\n{'='*50}")
            print(f"Testing {category.upper()} Analyzer")
            print(f"{'='*50}")
            
            # Filter data for this category
            filtered_data = filter_by_pt_category(test_data, category)
            
            if not filtered_data:
                print(f"No test data found for category '{category}'")
                continue
                
            print(f"Found {len(filtered_data)} items for '{category}' analysis")
            print(f"Analyzer focus: {ANALYZER_PART_MAPS.get(category, 'template-based')}")
            
            # Universal analysis
            results = analyze_universal(client, chunker, filtered_data[:max_items], category)
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
        print(f"Universal analyzer testing complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
