from datasets import load_dataset
from utils import color_palette, ImageSegmentationDataset, collate_fn,preprocessor
import numpy as np
from huggingface_hub import hf_hub_download
import json
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import albumentations as A
from PIL import Image
from transformers import Mask2FormerImageProcessor
from torch.utils.data import DataLoader
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig, Dinov2Config, Dinov2Model, Dinov2Backbone
import os
import evaluate
import torch
from tqdm.auto import tqdm
import argparse
from datetime import datetime
current_date = datetime.now().strftime("%Y%m%d")
print("current date: ",current_date )
parser = argparse.ArgumentParser(description='Mask2DinoV2 Semantic Segmentation')
parser.add_argument('--mode', type=str, default="mask2dinov2", help="mask2dino/mask2former")
parser.add_argument('--checkpoint_path', type=str,  help='Checkpoint path')
parser.add_argument('--folder_name', type=str, default=current_date)
parser.add_argument('--lr', type=float, default='1.5e-6')
parser.add_argument('--epoch', type=int, default='150')

args = parser.parse_args()

saved_folder = f'results/{args.mode}/{args.folder_name}'

dataset = load_dataset("segments/sidewalk-semantic")

print("checkpoint_path: ",args.checkpoint_path)

def save_loss_plot(losses, save_path=f'{args.checkpoint_path}/loss.png'):
    plt.figure(figsize=(10, 5))
    plt.plot( losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.xticks( rotation='vertical')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


# shuffle + split dataset
dataset = dataset.shuffle(seed=1)
dataset = dataset["train"].train_test_split(test_size=0.2)
train_ds = dataset["train"]
test_ds = dataset["test"]

# let's look at one example (images are pretty high resolution)
example = train_ds[1]
image = example['pixel_values']
image


# load corresponding ground truth segmentation map, which includes a label per pixel
segmentation_map = np.array(example['label'])
segmentation_map

repo_id = f"segments/sidewalk-semantic"
filename = "id2label.json"
id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
id2label = {int(k):v for k,v in id2label.items()}

labels = [id2label[label] for label in np.unique(segmentation_map)]
print(labels)

palette = color_palette()

color_segmentation_map = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3
for label, color in enumerate(palette):
    color_segmentation_map[segmentation_map - 1 == label, :] = color
# Convert to BGR
ground_truth_color_seg = color_segmentation_map[..., ::-1]

img = np.array(image) * 0.5 + ground_truth_color_seg * 0.5
img = img.astype(np.uint8)

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

train_transform = A.Compose([
    A.LongestMaxSize(max_size=1333),
    A.RandomCrop(width=512, height=512),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
])

test_transform = A.Compose([
    A.Resize(width=512, height=512),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
])

train_dataset = ImageSegmentationDataset(train_ds, transform=train_transform)
test_dataset = ImageSegmentationDataset(test_ds, transform=test_transform)

image, segmentation_map, _, _ = train_dataset[0]
print(image.shape)
print(segmentation_map.shape)

unnormalized_image = (image * np.array(ADE_STD)[:, None, None]) + np.array(ADE_MEAN)[:, None, None]
unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
Image.fromarray(unnormalized_image)

labels = [id2label[label] for label in np.unique(segmentation_map)]
print(labels)

color_segmentation_map = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3
for label, color in enumerate(palette):
    color_segmentation_map[segmentation_map == label, :] = color
# Convert to BGR
ground_truth_color_seg = color_segmentation_map[..., ::-1]

img = np.moveaxis(image, 0, -1) * 0.5 + ground_truth_color_seg * 0.5
img = img.astype(np.uint8)

# Create a preprocessor
preprocessor = Mask2FormerImageProcessor(ignore_index=0, reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)


batch = next(iter(train_dataloader))
for k,v in batch.items():
  if isinstance(v, torch.Tensor):
    print(k,v.shape)
  else:
    print(k,v[0].shape)

pixel_values = batch["pixel_values"][0].numpy()
pixel_values.shape

unnormalized_image = (pixel_values * np.array(ADE_STD)[:, None, None]) + np.array(ADE_MEAN)[:, None, None]
unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
Image.fromarray(unnormalized_image)


# verify class labels
labels = [id2label[label] for label in batch["class_labels"][0].tolist()]
print(labels)

# verify mask labels
batch["mask_labels"][0].shape

def visualize_mask(labels, label_name):
  print("Label:", label_name)
  idx = labels.index(label_name)

  visual_mask = (batch["mask_labels"][0][idx].bool().numpy() * 255).astype(np.uint8)
  return Image.fromarray(visual_mask)

visualize_mask(labels, "flat-road")

"""
Define model
"""


# Store Dinov2 weights locally
dinov2_backbone_model = Dinov2Model.from_pretrained("facebook/dinov2-small", out_indices=[6, 8, 10, 12])
if not os.path.exists("dinov2-small.pth"):
    torch.save(dinov2_backbone_model.state_dict(), "dinov2-small.pth")
else:
    print("DinoV2 weights already exist")

model_config = Mask2FormerConfig.from_pretrained("facebook/mask2former-swin-tiny-ade-semantic", id2label=id2label, ignore_mismatched_sizes=True)
model_config.backbone_config = Dinov2Config.from_pretrained("facebook/dinov2-small", out_indices = (6, 8, 10 ,12))
# model_config.backbone_config = Dinov2Config(out_indices = ( 6, 8, 10 ,12))

# Instantiate Mask2Former model with Dinov2 backbone (random weights)
model = Mask2FormerForUniversalSegmentation(model_config)

# # Load Dinov2 weights into Mask2Former backbone
dinov2_backbone = model.model.pixel_level_module.encoder
dinov2_backbone.load_state_dict(torch.load("dinov2-small.pth"))

"""Train the model"""
metric = evaluate.load("mean_iou")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) #5e-5
# initialize training
start_epoch = 0
running_loss = 0.0
num_samples = 0

#check if checkpoint exists
checkpoint_prefix = 'model_checkpoint_epoch_'
checkpoint_files = [f for f in os.listdir('.') if os.path.isfile(f) and f.startswith(checkpoint_prefix)]
if checkpoint_files:
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print(latest_checkpoint)
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']+1
    running_loss = checkpoint['running_loss']
    num_samples = checkpoint['num_samples']
    print(f"Loaded checkpoint from epoch {start_epoch}")
else:
    print("No checkpoint found. Starting training from scratch.")

epoch = args.epoch
losses = [] 
miou_values = []
for epoch in range(start_epoch, epoch):
  print("Epoch:", epoch)
  model.train() 
  for idx, batch in enumerate(tqdm(train_dataloader)):
      # Reset the parameter gradients
      optimizer.zero_grad()

      # Forward pass
      outputs = model(
          pixel_values=batch["pixel_values"].to(device),
          mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
          class_labels=[labels.to(device) for labels in batch["class_labels"]],
      )

      # Backward propagation
      loss = outputs.loss
      loss.backward()

      batch_size = batch["pixel_values"].size(0)
      running_loss += loss.item()
      num_samples += batch_size

      if idx % 100 == 0:
        print("Loss:", running_loss/num_samples)

      # Optimization
      optimizer.step()
  loss = running_loss / num_samples
  losses.append(loss)
    # After each epoch, save the model
  checkpoint_path = f'results/{args.saved_folder}/{args.folder_name}/{epoch}.pth'
  checkpoint = {
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'running_loss': running_loss,
      'num_samples': num_samples,
  }

  result_path = f'{saved_folder}/loss_epoch.png'
  save_loss_plot(losses, save_path=result_path)
  
  torch.save(checkpoint, checkpoint_path)
  print(f"Checkpoint saved at {checkpoint_path}")

  model.eval()
  for idx, batch in enumerate(tqdm(test_dataloader)):
    if idx > 5:
      break

    pixel_values = batch["pixel_values"]

    # Forward pass
    with torch.no_grad():
      outputs = model(pixel_values=pixel_values.to(device))

    # get original images
    original_images = batch["original_images"]
    target_sizes = [(image.shape[0], image.shape[1]) for image in original_images]
    # predict segmentation maps
    predicted_segmentation_maps = preprocessor.post_process_semantic_segmentation(outputs,
                                                                                  target_sizes=target_sizes)

    # get ground truth segmentation maps
    ground_truth_segmentation_maps = batch["original_segmentation_maps"]

    metric.add_batch(references=ground_truth_segmentation_maps, predictions=predicted_segmentation_maps)
    miou = metric.compute(num_labels = len(id2label), ignore_index = 0)['mean_iou']
    miou_values.append(miou)
  with open(f'{saved_folder}/result.txt', 'a') as f:
    f.write(f'Epoch {epoch}: Loss={loss}, mIoU={miou}\n')
    
  print("Mean IoU:", miou)