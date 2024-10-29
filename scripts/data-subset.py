import os
import shutil
import random

def create_subsets(base_path, num_train_real):
    # Define paths
    train_real_path = os.path.join(base_path, 'train', 'real')
    train_fake_path = os.path.join(base_path, 'train', 'fake')
    
    test_real_path = os.path.join(base_path, 'test', 'real')
    test_fake_path = os.path.join(base_path, 'test', 'fake')

    # Calculate the number of images for the test and validation sets
    num_test_images = 4268  # Number of images for test/val sets

    # Create a new directory for subsets
    subsets_path = f'/data/elsa-{num_train_real}'
    os.makedirs(subsets_path, exist_ok=True)

    # Create train subset
    train_subset_path = os.path.join(subsets_path, 'train')
    os.makedirs(train_subset_path, exist_ok=True)

    selected_train_real = random.sample(os.listdir(train_real_path), num_train_real)
    selected_train_fake = random.sample(os.listdir(train_fake_path), num_train_real)

    os.makedirs(os.path.join(train_subset_path, 'real'), exist_ok=True)
    os.makedirs(os.path.join(train_subset_path, 'fake'), exist_ok=True)

    for img in selected_train_real:
        shutil.copy(os.path.join(train_real_path, img), os.path.join(train_subset_path, 'real', img))

    for img in selected_train_fake:
        shutil.copy(os.path.join(train_fake_path, img), os.path.join(train_subset_path, 'fake', img))

    # Create validation subset
    val_subset_path = os.path.join(subsets_path, 'val')
    os.makedirs(val_subset_path, exist_ok=True)

    selected_val_real = random.sample(os.listdir(train_real_path), num_test_images)
    selected_val_fake = random.sample(os.listdir(train_fake_path), num_test_images)

    os.makedirs(os.path.join(val_subset_path, 'real'), exist_ok=True)
    os.makedirs(os.path.join(val_subset_path, 'fake'), exist_ok=True)

    for img in selected_val_real:
        shutil.copy(os.path.join(train_real_path, img), os.path.join(val_subset_path, 'real', img))

    for img in selected_val_fake:
        shutil.copy(os.path.join(train_fake_path, img), os.path.join(val_subset_path, 'fake', img))

    # Create test subset
    test_subset_path = os.path.join(subsets_path, 'test')
    os.makedirs(test_subset_path, exist_ok=True)

    selected_test_real = random.sample(os.listdir(test_real_path), num_test_images)
    selected_test_fake = random.sample(os.listdir(test_fake_path), num_test_images)

    os.makedirs(os.path.join(test_subset_path, 'real'), exist_ok=True)
    os.makedirs(os.path.join(test_subset_path, 'fake'), exist_ok=True)

    for img in selected_test_real:
        shutil.copy(os.path.join(test_real_path, img), os.path.join(test_subset_path, 'real', img))

    for img in selected_test_fake:
        shutil.copy(os.path.join(test_fake_path, img), os.path.join(test_subset_path, 'fake', img))

    print(f'Subsets created in {subsets_path} with {num_train_real} train real images.')

if __name__ == "__main__":
    base_path = '/data/elsa'
    num_train_real = 100000  # Specify the number of real train images to select
    create_subsets(base_path, num_train_real)
