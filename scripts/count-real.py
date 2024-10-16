import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm
import random

# Function to calculate deepfake percentage
def calculate_deepfake_percentage(num_samples):
    # Load the dataset from disk
    data_path = "/data/deepfakes"
    loaded_data = load_from_disk(data_path)

    # Randomly select the specified number of samples with progress tracking
    sampled_data = []
    for _ in tqdm(range(num_samples), desc="Randomly selecting samples", unit="sample"):
        sampled_data.append(random.choice(loaded_data))

    # Convert the sampled dataset to a DataFrame
    df = pd.DataFrame(sampled_data)

    # Define a function to check if an image is a deepfake
    def is_deepfake(row):
        # Check if any image_gen or model_gen field is present
        return any(row.get(f'image_gen{i}') for i in range(4)) or any(row.get(f'model_gen{i}') for i in range(4))

    # Count deepfake images
    deepfake_count = df.apply(is_deepfake, axis=1).sum()

    # Calculate percentage
    total_images = len(df)
    deepfake_percentage = (deepfake_count / total_images) * 100 if total_images > 0 else 0

    # Report results
    print(f"Total Samples: {total_images}")
    print(f"Total Deepfakes: {deepfake_count}")
    print(f"Percentage of Deepfakes: {deepfake_percentage:.2f}%")

# Parameters
num_samples = 1000  # Set this to the desired number of samples to check

# Run the deepfake percentage calculation
calculate_deepfake_percentage(num_samples)
