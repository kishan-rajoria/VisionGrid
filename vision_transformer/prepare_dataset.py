import os
import json
import random
import requests
from tqdm import tqdm
from pycocotools.coco import COCO
import zipfile
import io

# Create dataset directories if they don't exist
os.makedirs('dataset/images/train', exist_ok=True)
os.makedirs('dataset/images/val', exist_ok=True)
os.makedirs('dataset/images/test', exist_ok=True)
os.makedirs('dataset/annotations', exist_ok=True)

# Define the 5 classes we want to use
CLASSES = ['person', 'car', 'dog', 'cat', 'bicycle']
CLASS_IDS = [1, 3, 17, 16, 2]  # COCO class IDs for our classes

# Download and extract COCO annotations
print("Downloading COCO annotations...")
annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
response = requests.get(annotations_url, stream=True)
total_size = int(response.headers.get('content-length', 0))

with open('annotations.zip', 'wb') as f, tqdm(
    desc="Downloading annotations",
    total=total_size,
    unit='iB',
    unit_scale=True,
    unit_divisor=1024,
) as pbar:
    for data in response.iter_content(chunk_size=1024):
        size = f.write(data)
        pbar.update(size)

print("Extracting annotations...")
with zipfile.ZipFile('annotations.zip', 'r') as zip_ref:
    zip_ref.extractall('.')

# Initialize COCO API with the extracted annotations
coco = COCO('annotations/instances_val2017.json')

# Function to download image
def download_image(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    return False

# Get all image IDs that contain our classes
image_ids = []
for class_id in CLASS_IDS:
    img_ids = coco.getImgIds(catIds=[class_id])
    image_ids.extend(img_ids)

# Remove duplicates and shuffle
image_ids = list(set(image_ids))
random.shuffle(image_ids)

# Select 150 images
selected_ids = image_ids[:150]

# Split into train/val/test (100/25/25)
train_ids = selected_ids[:100]
val_ids = selected_ids[100:125]
test_ids = selected_ids[125:]

# Create annotation files
def create_annotation_file(image_ids, split):
    annotations = []
    images = []
    
    for img_id in tqdm(image_ids, desc=f'Processing {split} set'):
        img_info = coco.loadImgs(img_id)[0]
        img_url = img_info['coco_url']
        
        # Download image
        save_path = f'dataset/images/{split}/{img_info["file_name"]}'
        if download_image(img_url, save_path):
            images.append({
                'id': img_id,
                'file_name': img_info['file_name'],
                'height': img_info['height'],
                'width': img_info['width']
            })
            
            # Get annotations for this image
            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=CLASS_IDS)
            anns = coco.loadAnns(ann_ids)
            
            for ann in anns:
                annotations.append({
                    'id': ann['id'],
                    'image_id': img_id,
                    'category_id': ann['category_id'],
                    'bbox': ann['bbox'],
                    'area': ann['area'],
                    'iscrowd': ann['iscrowd']
                })
    
    # Create COCO format annotation file
    coco_format = {
        'images': images,
        'annotations': annotations,
        'categories': [{'id': class_id, 'name': class_name} 
                      for class_id, class_name in zip(CLASS_IDS, CLASSES)]
    }
    
    with open(f'dataset/annotations/instances_{split}2017.json', 'w') as f:
        json.dump(coco_format, f)

# Create annotation files for each split
create_annotation_file(train_ids, 'train')
create_annotation_file(val_ids, 'val')
create_annotation_file(test_ids, 'test')

# Clean up downloaded files
os.remove('annotations.zip')
print("\nDataset preparation completed!")
print(f"Train set: {len(train_ids)} images")
print(f"Val set: {len(val_ids)} images")
print(f"Test set: {len(test_ids)} images") 