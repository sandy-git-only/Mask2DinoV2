from datasets import load_dataset
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig, Dinov2Config, Dinov2Model
from PIL import Image
from torch.utils.data import DataLoader
from transformers import Mask2FormerImageProcessor
import albumentations as A
import numpy as np
import torch
from utils.palette import color_palette
from utils.dataset import ImageSegmentationDataset
from utils.collate import collate_fn, preprocessor
import json
from huggingface_hub import hf_hub_download
import os
import matplotlib.pyplot as plt
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
'''Dataset'''
dataset = load_dataset("segments/sidewalk-semantic")

# dataset = dataset.shuffle(seed=1)
dataset = dataset["train"].train_test_split(test_size=0.2)
train_ds = dataset["train"]
test_ds = dataset["test"]

palette = color_palette()

#Normalization
ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

test_transform = A.Compose([
    A.Resize(width=512, height=512),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
])

test_dataset = ImageSegmentationDataset(test_ds, transform=test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
repo_id = f"segments/sidewalk-semantic"
filename = "id2label.json"
id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
id2label = {int(k):v for k,v in id2label.items()}


'''Model'''
# Replace the head of the pre-trained model
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-ade-semantic",
                                                          id2label=id2label,
                                                          ignore_mismatched_sizes=True)

processor = Mask2FormerImageProcessor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

model.eval()
batch = next(iter(test_dataloader))
# for k,v in batch.items():
#   if isinstance(v, torch.Tensor):
#     print(k,v.shape)
#   else:
#     print(k,len(v))

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
dir_class = 'mask2former'
date = '20240312'
checkpoint_dir = dir_class + '_results_'+ date
checkpoint_name = 'model_checkpoint_epoch_' + '95' + '.pth'
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
print(checkpoint_path)
if checkpoint_path:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from file: {checkpoint_path}")
else:
    print("No checkpoint found.")


with torch.no_grad():
  outputs = model(batch["pixel_values"].to(device))
original_images = batch["original_images"]
target_sizes = [(image.shape[0], image.shape[1]) for image in original_images]

# predict segmentation maps
predicted_segmentation_maps = preprocessor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
image = batch["original_images"][0]
segmentation_map = predicted_segmentation_maps[0].cpu().numpy()

color_segmentation_map = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3
for label, color in enumerate(palette):
    color_segmentation_map[segmentation_map == label, :] = color
# Convert to BGR
ground_truth_color_seg = color_segmentation_map[..., ::-1]

img = image * 0.5 + ground_truth_color_seg * 0.5
img = img.astype(np.uint8)

plt.figure(figsize=(15, 10))
plt.imshow(img)
plt.show()

segmentation_map = batch["original_segmentation_maps"][0]

color_segmentation_map = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3
for label, color in enumerate(palette):
    color_segmentation_map[segmentation_map == label, :] = color
# Convert to BGR
ground_truth_color_seg = color_segmentation_map[..., ::-1]

img = image * 0.5 + ground_truth_color_seg * 0.5
img = img.astype(np.uint8)

plt.figure(figsize=(15, 10))
plt.imshow(img)
plt.show()