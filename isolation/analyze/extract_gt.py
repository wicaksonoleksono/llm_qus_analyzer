#!/usr/bin/env python3
"""Extract ground truth from CSV and convert to JSON format with components."""
# 
import json
import pandas as pd
from pathlib import Path

def extract_ground_truth():
    """Extract ground truth from CSV and convert to JSON format with components."""
    
    # CSV is in parent directory
    csv_path = Path("../ground_truth.csv")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Ground truth CSV not found at {csv_path}")
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    ground_truth_violations = {}
    ground_truth_components = {}
    
    for i, row in df.iterrows():
        story_text = row['Text']
        gt_value = row['GT']
        gt_role = row.get('role', '')
        gt_means = row.get('means', '')  
        gt_ends = row.get('ends', '')
        
        # Handle violation labels
        if pd.isna(gt_value) or gt_value == '' or str(gt_value).lower() == 'nan':
            gt_violations = []
        else:
            gt_violations = [item.strip() for item in str(gt_value).split(',')]
            gt_violations = [item for item in gt_violations if item]
        
        ground_truth_violations[story_text] = gt_violations
        
        # Handle component ground truth
        def clean_component(comp):
            if pd.isna(comp) or str(comp).lower() == 'nan' or comp == '':
                return None
            return str(comp).strip()
        
        # Parse role (can be multiple, comma-separated)
        role_clean = clean_component(gt_role)
        if role_clean:
            gt_role_list = [item.strip() for item in role_clean.split(',')]
            gt_role_list = [item for item in gt_role_list if item]
        else:
            gt_role_list = []
        
        ground_truth_components[story_text] = {
            'role': gt_role_list,
            'means': clean_component(gt_means),
            'ends': clean_component(gt_ends)
        }
        
        if i < 5:  # Debug first 5 entries
            print(f"Story {i+1}: {story_text[:50]}...")
            print(f"  GT Violations: {gt_violations}")
            print(f"  GT Components: {ground_truth_components[story_text]}")
    
    return ground_truth_violations, ground_truth_components

if __name__ == "__main__":
    try:
        gt_violations, gt_components = extract_ground_truth()
        with open('ground_truth_extracted.json', 'w') as f:
            json.dump(gt_violations, f, indent=2)
        with open('ground_truth_components.json', 'w') as f:
            json.dump(gt_components, f, indent=2)
        print(f"Extracted ground truth for {len(gt_violations)} stories")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()