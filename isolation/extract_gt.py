import pandas as pd
import json
from pathlib import Path

# Read the ground truth CSV
df = pd.read_csv('ground_truth.csv')
ground_truth = {}
for _, row in df.iterrows():
    text = row['Text']
    gt_value = row['GT']
    if pd.isna(gt_value) or gt_value == '':
        gt_violations = []
    else:
        gt_violations = [item.strip() for item in str(gt_value).split(',')]
    ground_truth[text] = gt_violations
with open('ground_truth_extracted.json', 'w') as f:
    json.dump(ground_truth, f, indent=2)

print(f"Extracted ground truth for {len(ground_truth)} stories")
print(f"Stories with violations: {sum(1 for gt in ground_truth.values() if gt)}")