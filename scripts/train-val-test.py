import os
import random
import pandas as pd
from datasets import load_from_disk
from PIL import Image, UnidentifiedImageError
import requests
from tqdm import tqdm
from io import BytesIO
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def requests_retry_session(
    retries=1,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=0)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def process_real_image(image_url, image_id, folder, session):
    try:
        response = session.get(image_url, timeout=10)
        response.raise_for_status()
        img_data = BytesIO(response.content)
        img = Image.open(img_data)
        img.verify()
        img = Image.open(img_data)
        img.load()
        img_format = img.format.lower() if img.format else 'png'
        image_path = os.path.join(folder, "real", f"{image_id}.{img_format}")
        img.save(image_path)
        return True
    except Exception as e:
        return False

def process_fake_image(image, image_id, folder):
    try:
        image_path = os.path.join(folder, "fake", f"{image_id}.png")
        image.save(image_path)
        return True
    except Exception as e:
        return False

def save_images(dataframe, folder, num_threads=10):
    real_images = 0
    skipped_real_images = 0
    deepfake_count = 0
    skipped_fake_images = 0

    session = requests_retry_session()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        real_futures = []
        fake_futures = []
        
        for idx, row in dataframe.iterrows():
            real_futures.append(executor.submit(process_real_image, row['url'], row['id'], folder, session))
            
            for i in range(4):
                if row.get(f'image_gen{i}') is not None:
                    fake_futures.append(executor.submit(process_fake_image, row[f'image_gen{i}'], f"{row['id']}-{i}", folder))

        for future in tqdm(as_completed(real_futures), total=len(real_futures), desc=f"Saving real images to {folder}"):
            if future.result():
                real_images += 1
            else:
                skipped_real_images += 1

        for future in tqdm(as_completed(fake_futures), total=len(fake_futures), desc=f"Saving fake images to {folder}"):
            if future.result():
                deepfake_count += 1
            else:
                skipped_fake_images += 1

    print(f"Saved {real_images} real images to {folder}.")
    print(f"Skipped {skipped_real_images} problematic real images.")
    print(f"Saved {deepfake_count} fake images to {folder}.")
    print(f"Skipped {skipped_fake_images} problematic fake images.")
    return deepfake_count

def create_datasets(num_train_images):
    print("Samples: ", num_train_images)
    data_path = "/data/deepfakes"
    loaded_data = load_from_disk(data_path)
    print(f"Loaded {len(loaded_data)} images from {data_path}")

    val_data_path = "/data/deepfakes-val"
    val_data = load_from_disk(val_data_path)
    print(f"Loaded {len(val_data)} validation images from {val_data_path}")

    output_dir = f"/data/elsa-{num_train_images}"
    for split in ['train', 'val', 'test']:
        for category in ['real', 'fake']:
            os.makedirs(os.path.join(output_dir, split, category), exist_ok=True)
    print(f"Created output directory at {output_dir}")

    train_samples = 0
    batch_size = 1000
    sampled_data = []
    deepfake_count = 0

    for idx, batch in enumerate(loaded_data.iter(batch_size=batch_size), start=1):
        print(f"Processing batch {idx}...")
        
        for i in range(len(batch['id'])):
            sample = {key: batch[key][i] for key in batch}
            sampled_data.append(sample)
            train_samples += 1
            if train_samples >= num_train_images:
                break
        
        if train_samples >= num_train_images:
            break

    if sampled_data:
        df = pd.DataFrame(sampled_data)
        deepfake_count = save_images(df, os.path.join(output_dir, "train"))

    # Process test images
    num_test_images = 10
    test_samples = 0
    test_data = []

    for batch in loaded_data.iter(batch_size=batch_size):
        for i in range(len(batch['id'])):
            sample = {key: batch[key][i] for key in batch}
            test_data.append(sample)
            test_samples += 1
            if test_samples >= num_test_images:
                break
        if test_samples >= num_test_images:
            break

    if test_data:
        df = pd.DataFrame(test_data)
        save_images(df, os.path.join(output_dir, "test"))

    num_val_images = max(1, num_train_images // 10)  # Ensure at least one image if num_train_images is low
    val_samples = 0
    val_data_selected = []

    for batch in val_data.iter(batch_size=batch_size):
        for i in range(len(batch['id'])):
            sample = {key: batch[key][i] for key in batch}
            val_data_selected.append(sample)
            val_samples += 1
            if val_samples >= num_val_images:
                break
        if val_samples >= num_val_images:
            break

    if val_data_selected:
        val_df = pd.DataFrame(val_data_selected)
        save_images(val_df, os.path.join(output_dir, "val"))

    # Calculate percentages
    total_train_images = len(sampled_data)
    deepfake_percentage = (deepfake_count / total_train_images) * 100 if total_train_images > 0 else 0

    # Report results
    print(f"Total Train Images: {total_train_images}")
    print(f"Total Test Images: {len(test_data)}")
    print(f"Total Validation Images: {len(val_df)}")
    print(f"Percentage of Deepfakes in Train Set: {deepfake_percentage:.2f}%")

if __name__ == "__main__":
    num_train_images = 100  # Set this to the desired number of training images
    create_datasets(num_train_images)