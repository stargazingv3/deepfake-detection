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
        response = session.get(image_url, timeout=2)
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

class ImageStats:
    def __init__(self):
        self.total_real_saved = 0
        self.total_real_skipped = 0
        self.total_fake_saved = 0
        self.total_fake_skipped = 0

def save_images(dataframe, folder, num_threads=1024, stats=None):
    if stats is None:
        stats = ImageStats()
    
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

    stats.total_real_saved += real_images
    stats.total_real_skipped += skipped_real_images
    stats.total_fake_saved += deepfake_count
    stats.total_fake_skipped += skipped_fake_images

    print(f"This batch: Saved {real_images} real images, skipped {skipped_real_images}")
    print(f"This batch: Saved {deepfake_count} fake images, skipped {skipped_fake_images}")
    return deepfake_count, stats

def create_datasets(num_train_images):
    print("Samples: ", num_train_images)
    data_path = "/data/deepfakes"
    loaded_data = load_from_disk(data_path)
    print(f"Loaded {len(loaded_data)} images from {data_path}")

    val_data_path = "/data/deepfakes-val"
    val_data = load_from_disk(val_data_path)
    print(f"Loaded {len(val_data)} validation images from {val_data_path}")

    output_dir = f"/data/elsa"
    for split in ['train', 'val', 'test']:
        for category in ['real', 'fake']:
            os.makedirs(os.path.join(output_dir, split, category), exist_ok=True)
    print(f"Created output directory at {output_dir}")

    # Initialize statistics trackers
    train_stats = ImageStats()
    val_stats = ImageStats()
    test_stats = ImageStats()

    # Set up sample sizes
    num_val_images = 4800
    num_test_images = 4800
    num_train_images -= num_val_images
    
    # Process validation images
    val_samples = 0
    val_data_selected = []
    batch_size = 1024

    for batch in val_data.iter(batch_size=batch_size):
        for i in range(len(batch['id'])):
            sample = {key: batch[key][i] for key in batch}
            val_data_selected.append(sample)
            val_samples += 1
            if val_samples >= num_val_images:
                break

        if val_data_selected:
            val_df = pd.DataFrame(val_data_selected)
            _, val_stats = save_images(val_df, os.path.join(output_dir, "test"), stats=val_stats)

        if val_samples >= num_val_images:
            break

    # Process training images
    train_samples = 0
    total_deepfake_count = 0
    batch_count = 0

    for idx, batch in enumerate(loaded_data.iter(batch_size=batch_size), start=1):
        sampled_data = []
        print(f"Processing batch {idx}...")
        
        for i in range(len(batch['id'])):
            sample = {key: batch[key][i] for key in batch}
            sampled_data.append(sample)
            train_samples += 1
            if train_samples >= num_train_images:
                break
        
        if sampled_data:
            df = pd.DataFrame(sampled_data)
            deepfake_count, train_stats = save_images(df, os.path.join(output_dir, "train"), stats=train_stats)
            total_deepfake_count += deepfake_count

        if train_samples >= num_train_images:
            break
        batch_count += 1

    # Process test images
    test_samples = 0
    test_data = []

    for batch in loaded_data.iter(batch_size=batch_size, start=batch_count):
        for i in range(len(batch['id'])):
            sample = {key: batch[key][i] for key in batch}
            test_data.append(sample)
            test_samples += 1
            if test_samples >= num_test_images:
                break

        if test_data:
            df = pd.DataFrame(test_data)
            _, test_stats = save_images(df, os.path.join(output_dir, "val"), stats=test_stats)

        if test_samples >= num_test_images:
            break

    # Report final results
    print("\nFinal Statistics:")
    print("Training Set:")
    print(f"Total real images saved: {train_stats.total_real_saved}")
    print(f"Total real images skipped: {train_stats.total_real_skipped}")
    print(f"Total fake images saved: {train_stats.total_fake_saved}")
    print(f"Total fake images skipped: {train_stats.total_fake_skipped}")
    
    print("\nValidation Set:")
    print(f"Total real images saved: {val_stats.total_real_saved}")
    print(f"Total real images skipped: {val_stats.total_real_skipped}")
    print(f"Total fake images saved: {val_stats.total_fake_saved}")
    print(f"Total fake images skipped: {val_stats.total_fake_skipped}")
    
    print("\nTest Set:")
    print(f"Total real images saved: {test_stats.total_real_saved}")
    print(f"Total real images skipped: {test_stats.total_real_skipped}")
    print(f"Total fake images saved: {test_stats.total_fake_saved}")
    print(f"Total fake images skipped: {test_stats.total_fake_skipped}")

    # Calculate overall percentages
    total_train_images = train_stats.total_real_saved + train_stats.total_real_skipped
    train_deepfake_percentage = (train_stats.total_fake_saved / total_train_images) * 100 if total_train_images > 0 else 0
    total_val_images = val_stats.total_real_saved + val_stats.total_real_skipped
    val_deepfake_percentage = (val_stats.total_fake_saved / total_val_images) * 100 if total_val_images > 0 else 0
    total_test_images = test_stats.total_real_saved + test_stats.total_real_skipped
    test_deepfake_percentage = (test_stats.total_fake_saved / total_test_images) * 100 if total_test_images > 0 else 0

    print(f"\nOverall Statistics:")
    print(f"Total Train Images: {total_train_images} (Deepfakes: {train_deepfake_percentage:.2f}%)")
    print(f"Total Validation Images: {total_val_images} (Deepfakes: {val_deepfake_percentage:.2f}%)")
    print(f"Total Test Images: {total_test_images} (Deepfakes: {test_deepfake_percentage:.2f}%)")

if __name__ == "__main__":
    num_train_images = 2306629
    create_datasets(num_train_images)