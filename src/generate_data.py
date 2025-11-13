#!/usr/bin/env python3
"""
Generate sample syllogism data for SemEval 2026 Task 11
Creates synthetic training/validation/test sets for development
"""

import json
import random
import argparse
from pathlib import Path

# Set seed for reproducibility
random.seed(42)

# Sample syllogism templates
VALID_SYLLOGISMS = [
    {
        "template": "All {A} are {B}. All {B} are {C}. Therefore, all {A} are {C}.",
        "plausible_entities": [("dogs", "mammals", "animals"), ("cats", "felines", "carnivores")],
        "implausible_entities": [("rocks", "plants", "animals"), ("cars", "birds", "fish")]
    },
    {
        "template": "No {A} are {B}. All {C} are {B}. Therefore, no {C} are {A}.",
        "plausible_entities": [("fish", "mammals", "whales"), ("birds", "reptiles", "snakes")],
        "implausible_entities": [("trees", "animals", "dogs"), ("water", "solid", "ice")]
    },
]

INVALID_SYLLOGISMS = [
    {
        "template": "All {A} are {B}. Some {B} are {C}. Therefore, all {A} are {C}.",
        "plausible_entities": [("dogs", "animals", "pets"), ("cats", "mammals", "friendly")],
        "implausible_entities": [("rocks", "objects", "living"), ("cars", "vehicles", "flying")]
    },
    {
        "template": "Some {A} are {B}. All {B} are {C}. Therefore, all {A} are {C}.",
        "plausible_entities": [("birds", "animals", "flying"), ("fish", "creatures", "aquatic")],
        "implausible_entities": [("stones", "plants", "growing"), ("clouds", "solid", "heavy")]
    },
]

def generate_syllogism(template_data, is_valid, is_plausible):
    """Generate a single syllogism"""
    template = template_data["template"]
    entities = template_data["plausible_entities" if is_plausible else "implausible_entities"]
    
    # Pick random entity set
    entity_set = random.choice(entities)
    
    # Fill template
    syllogism = template.format(A=entity_set[0], B=entity_set[1], C=entity_set[2])
    
    return syllogism

def generate_dataset(num_samples, subtask=1):
    """Generate dataset for a subtask"""
    data = []
    
    for i in range(num_samples):
        # Balance valid/invalid and plausible/implausible
        is_valid = i % 2 == 0
        is_plausible = i % 4 < 2
        
        # Select template
        templates = VALID_SYLLOGISMS if is_valid else INVALID_SYLLOGISMS
        template_data = random.choice(templates)
        
        # Generate syllogism
        syllogism = generate_syllogism(template_data, is_valid, is_plausible)
        
        # Add irrelevant premise for subtask 2/4
        if subtask in [2, 4]:
            irrelevant = "The sky is blue."
            syllogism = irrelevant + " " + syllogism
        
        sample = {
            "id": str(i),
            "syllogism": syllogism,
            "validity": is_valid,
            "plausibility": is_plausible  # Only in training
        }
        
        data.append(sample)
    
    return data

def main():
    parser = argparse.ArgumentParser(description="Generate sample syllogism data")
    parser.add_argument("--subtask", type=int, default=1, choices=[1, 2, 3, 4],
                       help="Subtask number (1-4)")
    parser.add_argument("--train-size", type=int, default=1000,
                       help="Number of training samples")
    parser.add_argument("--val-size", type=int, default=200,
                       help="Number of validation samples")
    parser.add_argument("--test-size", type=int, default=100,
                       help="Number of test samples")
    
    args = parser.parse_args()
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    print(f"📦 Generating sample data for Subtask {args.subtask}...")
    
    # Generate datasets
    train_data = generate_dataset(args.train_size, args.subtask)
    val_data = generate_dataset(args.val_size, args.subtask)
    test_data = generate_dataset(args.test_size, args.subtask)
    
    # Remove plausibility from test (not provided in real test)
    for sample in test_data:
        del sample["plausibility"]
    
    # Save to JSON
    train_file = data_dir / f"train_subtask{args.subtask}.json"
    val_file = data_dir / f"val_subtask{args.subtask}.json"
    test_file = data_dir / f"test_subtask{args.subtask}.json"
    
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"  ✓ Created: {train_file} ({args.train_size} samples)")
    
    with open(val_file, 'w') as f:
        json.dump(val_data, f, indent=2)
    print(f"  ✓ Created: {val_file} ({args.val_size} samples)")
    
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"  ✓ Created: {test_file} ({args.test_size} samples)")
    
    print("\n✅ Data generation complete!")
    print(f"📁 Data saved to: {data_dir}")
    print("\n🚀 Next steps:")
    print(f"   python3 src/pipeline.py train --subtask {args.subtask}")

if __name__ == "__main__":
    main()
