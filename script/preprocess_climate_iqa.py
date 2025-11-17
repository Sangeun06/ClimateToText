import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

def prepare_climate_iqa_dataset():
    """
    Downloads the ClimateIQA dataset from Hugging Face, saves the images,
    and creates a metadata.jsonl file in the required format.
    """
    # 1. Configuration
    # Make sure to use the correct dataset name from Hugging Face
    dataset_name = "GPS-Lab/ClimateIQA"
    output_dir = Path("./data/ClimateIQA")
    images_dir = output_dir / "images"

    # 2. Create directories
    output_dir.mkdir(exist_ok=True, parents=True)
    images_dir.mkdir(exist_ok=True)
    
    print(f"Directories created at: {output_dir.resolve()}")

    # 3. Load dataset from Hugging Face
    print(f"Loading '{dataset_name}' from Hugging Face...")
    
    metadata_path = output_dir / "metadata.jsonl"
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        print(f"Processing dataset and saving images to '{images_dir.resolve()}'...")
        
        # The dataset has 'train', 'validation', and 'test' splits.
        for split in ["train", "validation", "test"]:
            print(f"Loading split: {split}")
            dataset = load_dataset(dataset_name, split=split)
            print("Dataset loaded successfully.")

            # 4. Create metadata.jsonl and save images
            for i, record in enumerate(tqdm(dataset, desc=f"Processing {split} split")):
                # The dataset has 'image', 'query', 'response', and 'explanation' fields.
                # We will use 'explanation' as the descriptive text for contrastive learning.
                image = record["image"]
                explanation = record["explanation"]

                # Skip if explanation is missing or empty
                if not explanation:
                    continue

                # Create a unique filename
                image_filename = f"{split}_{i}.jpg"
                image_path = images_dir / image_filename
                image.save(image_path)

                # Create metadata entry
                metadata_entry = {
                    "image_path": f"images/{image_filename}",
                    "event": explanation
                }

                # Write to JSONL file
                f.write(json.dumps(metadata_entry) + "\n")

    print("-" * 50)
    print("Dataset preparation complete!")
    print(f"Images saved in: {images_dir.resolve()}")
    print(f"Metadata file created at: {metadata_path.resolve()}")
    print(f"\nExample entry in {metadata_path.name}:")
    with open(metadata_path, "r", encoding="utf-8") as f:
        print(f.readline().strip())
    print("-" * 50)


if __name__ == "__main__":
    # You might need to install the 'datasets' and 'Pillow' libraries first:
    # pip install datasets Pillow tqdm
    prepare_climate_iqa_dataset()
