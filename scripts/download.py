from datasets import load_dataset

# Load the dataset
elsa_data = load_dataset("elsaEU/ELSA_D3", split="validation")

# Save the dataset locally
elsa_data.save_to_disk("/data/deepfakes-val")

# Load the dataset from disk
loaded_data = load_dataset("/data/deepfakes-val")

# Print the first few examples
for i, example in enumerate(loaded_data):
    if i < 5:  # Change this number to see more or fewer examples
        print(example)
    else:
        break
