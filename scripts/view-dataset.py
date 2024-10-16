from datasets import load_from_disk
from PIL import Image
import matplotlib.pyplot as plt
import os
import requests
from io import BytesIO

# Load the dataset from disk
loaded_data = load_from_disk("/data/deepfakes")

# Create directories to save images if they don't exist
output_dir = "saved_images"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "real"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "fake"), exist_ok=True)

def is_valid_image(img_data):
    try:
        img = Image.open(BytesIO(img_data))
        img.verify()
        return True
    except Exception as e:
        print(f"Image verification error: {e}")
        return False

# Loop through the first 5 examples
for idx, example in enumerate(loaded_data):
    if idx >= 5:  # Stop after the first 5 examples
        break
    
    image_id = example['id']
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Example {idx}")
    
    # Process the real image
    if example.get('url'):
        real_image_url = example['url']
        try:
            response = requests.get(real_image_url)
            if is_valid_image(response.content):
                img = Image.open(BytesIO(response.content))
                img_format = img.format.lower()
                real_image_path = os.path.join(output_dir, "real", f"{image_id}.{img_format}")
                img.save(real_image_path)
                
                axs[0, 0].imshow(img)
                axs[0, 0].set_title("Real Image")
                axs[0, 0].axis('off')
            else:
                print(f"Invalid image format: {real_image_url}")
        except requests.RequestException:
            print(f"Failed to download image: {real_image_url}")
    
    # Process the fake images
    fake_images_count = 0
    for i in range(4):  # Adjust based on the number of image generations
        if example.get(f'image_gen{i}'):
            fake_image = example[f'image_gen{i}']
            fake_image_path = os.path.join(output_dir, "fake", f"{image_id}-{i}.png")
            
            # Directly save the fake images
            fake_image.save(fake_image_path)
            
            # Place fake images in appropriate subplot position
            row, col = (fake_images_count // 3), fake_images_count % 3
            axs[row, col].imshow(fake_image)
            axs[row, col].set_title(f"Fake Image {i}")
            axs[row, col].axis('off')
            fake_images_count += 1
    
    # Remove empty subplots
    for ax in axs.flat:
        if not ax.images:
            fig.delaxes(ax)
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"example_{idx}.png"))
    plt.close(fig)  # Properly close the figure

print("Images saved successfully.")
