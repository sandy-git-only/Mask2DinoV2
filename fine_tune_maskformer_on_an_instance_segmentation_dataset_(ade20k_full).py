

from datasets import load_dataset

dataset = load_dataset("scene_parse_150", "instance_segmentation")

"""Let's look at it in more detail. We have 3 splits:"""

dataset

"""## Create id2label

It's important to create a mapping between integer IDs and their corresponding label names.

In this case, as noted on the [dataset card](https://huggingface.co/datasets/scene_parse_150), there are 100 semantic categories for the instance segmentation split of the ADE20k dataset. It includes the sentence "See [here](https://github.com/CSAILVision/placeschallenge/blob/master/instancesegmentation/instanceInfo100_train.txt) for an overview".

So we can download this file to get all label names.
"""

# !wget https://raw.githubusercontent.com/CSAILVision/placeschallenge/master/instancesegmentation/instanceInfo100_train.txt

"""Let's read it in using Pandas."""

import pandas as pd

data = pd.read_csv('./instanceInfo100_train.txt',
                   sep='\t', header=0)
data.head(5)

id2label = {id: label.strip() for id, label in enumerate(data["Object Names"])}
print(id2label)

"""## Prepare a single example using MaskFormerImageProcessor

Next, let's showcase how a single training example will be prepared for the model.

Note: we won't use the image processor for resizing since it uses Pillow.
"""

example = dataset['train'][1]
image = example['image']
image

example['annotation']

"""From the [dataset card](https://huggingface.co/datasets/scene_parse_150):

> Note: in the instance annotation masks, the R(ed) channel encodes category ID, and the G(reen) channel encodes instance ID. Each object instance has a unique instance ID regardless of its category ID. In the dataset, all images have <256 object instances. Refer to this file (train split) and to this file (validation split) for the information about the labels of the 100 semantic categories. To find the mapping between the semantic categories for instance_segmentation and scene_parsing, refer to this file.
"""

import numpy as np

seg = np.array(example['annotation'])
# get green channel
instance_seg = seg[:, :, 1]
instance_seg

np.unique(instance_seg)

"""We can create a mapping between instance IDs and their corresponding semantic category IDs:"""

instance_seg = np.array(example["annotation"])[:,:,1] # green channel encodes instances
class_id_map = np.array(example["annotation"])[:,:,0] # red channel encodes semantic category
class_labels = np.unique(class_id_map)

# create mapping between instance IDs and semantic category IDs
inst2class = {}
for label in class_labels:
    instance_ids = np.unique(instance_seg[class_id_map == label])
    inst2class.update({i: label for i in instance_ids})
print(inst2class)

"""Let's visualize the binary mask of the first instance:"""

from PIL import Image

print("Visualizing instance:", id2label[inst2class[1] - 1])

# let's visualize the first instance (ignoring background)
mask = (instance_seg == 1)
visual_mask = (mask * 255).astype(np.uint8)
Image.fromarray(visual_mask)

"""Let's visualize the binary mask of the second instance:"""

print("Visualizing instance:", id2label[inst2class[2] - 1])

# let's visualize the second instance
mask = (instance_seg == 2)
visual_mask = (mask * 255).astype(np.uint8)
Image.fromarray(visual_mask)

"""We can visualize all masks in one go using the following formula:"""

R = seg[:, :, 0]
G = seg[:, :, 1]
masks = (R / 10).astype(np.int32) * 256 + (G.astype(np.int32))

visual_mask = (masks * 255).astype(np.uint8)
Image.fromarray(visual_mask)

"""Note that for this particular example dataset, different instances don't overlap. However it is technically possible for instance segmentation datasets to have several instances whose masks overlap. But this is not the case here.

Let's show how this image + corresponding set of binary masks gets prepared for the model.
"""

from transformers import Mask2FormerImageProcessor

processor = Mask2FormerImageProcessor(reduce_labels=True, ignore_index=255, do_resize=False, do_rescale=False, do_normalize=False)

"""We first apply a resize + normalize operation on the image + mask. Note that normalization only happens on the image, not the mask."""

import albumentations as A

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

transform = A.Compose([
    A.Resize(width=512, height=512),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
])

transformed = transform(image=np.array(image), mask=instance_seg)
pixel_values = np.moveaxis(transformed["image"], -1, 0)
instance_seg_transformed = transformed["mask"]
print(pixel_values.shape)
print(instance_seg_transformed.shape)

np.unique(instance_seg_transformed)

"""Next, we provide those to the image processor, which will turn the single instance segmentation map into a set of binary masks and corresponding labels. This is the format that MaskFormer expects (as it casts any image segmentation task to this format - also called "binary mask classification")."""

inputs = processor([pixel_values], [instance_seg_transformed], instance_id_to_semantic_id=inst2class, return_tensors="pt")

import torch

for k,v in inputs.items():
  if isinstance(v, torch.Tensor):
    print(k,v.shape)
  else:
    print(k,[x.shape for x in v])

"""Let's verify that the different binary masks it creates for a given example are different."""

assert not torch.allclose(inputs["mask_labels"][0][0], inputs["mask_labels"][0][1])

"""Let's check the corresponding class labels."""

inputs["class_labels"]

"""Let's visualize one of the binary masks + corresponding label:"""

from PIL import Image

# visualize first one
print("Label:", id2label[inputs["class_labels"][0][0].item()])

visual_mask = (inputs["mask_labels"][0][0].numpy() * 255).astype(np.uint8)
Image.fromarray(visual_mask)

# visualize second one
print("Label:", id2label[inputs["class_labels"][0][1].item()])

visual_mask = (inputs["mask_labels"][0][1].numpy() * 255).astype(np.uint8)
Image.fromarray(visual_mask)

# visualize third one
print("Label:", id2label[inputs["class_labels"][0][2].item()])

visual_mask = (inputs["mask_labels"][0][2].numpy() * 255).astype(np.uint8)
Image.fromarray(visual_mask)

# visualize fourth one
print("Label:", id2label[inputs["class_labels"][0][3].item()])

visual_mask = (inputs["mask_labels"][0][3].numpy() * 255).astype(np.uint8)
Image.fromarray(visual_mask)

"""As can be seen, these look very similar to the ones we visualized initially, except that all images + masks are now of size 512x512 due to the resize operation.

## Create PyTorch Dataset

Now that we've shown how a single example gets prepared, we can define a general PyTorch Dataset. This dataset will return any given training example, entirely prepared for the model.
"""

import numpy as np
from torch.utils.data import Dataset

class ImageSegmentationDataset(Dataset):
    """Image segmentation dataset."""

    def __init__(self, dataset, processor, transform=None):
        """
        Args:
            dataset
        """
        self.dataset = dataset
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = np.array(self.dataset[idx]["image"].convert("RGB"))

        instance_seg = np.array(self.dataset[idx]["annotation"])[:,:,1]
        class_id_map = np.array(self.dataset[idx]["annotation"])[:,:,0]
        class_labels = np.unique(class_id_map)

        inst2class = {}
        for label in class_labels:
            instance_ids = np.unique(instance_seg[class_id_map == label])
            inst2class.update({i: label for i in instance_ids})

        # apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, mask=instance_seg)
            image, instance_seg = transformed['image'], transformed['mask']
            # convert to C, H, W
            image = image.transpose(2,0,1)

        if class_labels.shape[0] == 1 and class_labels[0] == 0:
            # Some image does not have annotation (all ignored)
            inputs = self.processor([image], return_tensors="pt")
            inputs = {k:v.squeeze() for k,v in inputs.items()}
            inputs["class_labels"] = torch.tensor([0])
            inputs["mask_labels"] = torch.zeros((0, inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1]))
        else:
          inputs = self.processor([image], [instance_seg], instance_id_to_semantic_id=inst2class, return_tensors="pt")
          inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k,v in inputs.items()}

        return inputs

import albumentations as A

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

# note that you can include more fancy data augmentation methods here
train_transform = A.Compose([
    A.Resize(width=512, height=512),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
])

train_dataset = ImageSegmentationDataset(dataset["train"], processor=processor, transform=train_transform)

inputs = train_dataset[0]
for k,v in inputs.items():
  if isinstance(v, torch.Tensor):
    print(k,v.shape)

inputs["class_labels"]

inputs = train_dataset[1]
for k,v in inputs.items():
  print(k,v.shape)

inputs["class_labels"]

"""## Create PyTorch DataLoader

Next, one can define a corresponding PyTorch DataLoader, which allows to get batches from the dataset (as neural networks are typically trained in batches for stochastic gradient descent).

We define a custom collate function (which PyTorch allows) to define the logic to batch examples, given by the PyTorch dataset above, together.
"""

from torch.utils.data import DataLoader

def collate_fn(batch):
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    pixel_mask = torch.stack([example["pixel_mask"] for example in batch])
    class_labels = [example["class_labels"] for example in batch]
    mask_labels = [example["mask_labels"] for example in batch]
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "class_labels": class_labels, "mask_labels": mask_labels}

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

batch = next(iter(train_dataloader))
for k,v in batch.items():
  if isinstance(v, torch.Tensor):
    print(k,v.shape)
  else:
    print(k,len(v))

"""## Verify data (!!)

As always, it's very important to check whether the data which we'll feed to the model makes sense. Let's do some sanity checks.

One of them is denormalizing the pixel values to see whether we still get an image that makes sense.
"""

from PIL import Image

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

batch_index = 1

unnormalized_image = (batch["pixel_values"][batch_index].numpy() * np.array(ADE_STD)[:, None, None]) + np.array(ADE_MEAN)[:, None, None]
unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
Image.fromarray(unnormalized_image)

"""In this case, there are many class labels it seems."""

# batch["class_labels"][batch_index]

# id2label[1]

# """Let's check the corresponding binary masks. No less than 31 binary masks were created for this example!"""

# batch["mask_labels"][batch_index].shape

# """Let's visualize a couple of them, see if they make sense."""

# print("Visualizing mask for:", id2label[batch["class_labels"][batch_index][0].item()])

# visual_mask = (batch["mask_labels"][batch_index][0].bool().numpy() * 255).astype(np.uint8)
# Image.fromarray(visual_mask)

# print("Visualizing mask for:", id2label[batch["class_labels"][batch_index][1].item()])

# visual_mask = (batch["mask_labels"][batch_index][1].bool().numpy() * 255).astype(np.uint8)
# Image.fromarray(visual_mask)

"""## Define the model

Next, let's define the model. Here we will only replace the classification head with a new one, all other parameters will use pre-trained ones.

Note that we're loading a [checkpoint](https://huggingface.co/facebook/maskformer-swin-base-ade) of MaskFormer fine-tuned on a semantic segmentation dataset, but that's ok, it will work fine when fine-tuning on an instance segmentation dataset.

We'll load a checkpoint with a Swin backbone as these are pretty strong.
"""

from transformers import Mask2FormerForUniversalSegmentation

# Replace the head of the pre-trained model
# We specify ignore_mismatched_sizes=True to replace the already fine-tuned classification head by a new one
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-cityscapes-instance",
                                                          id2label=id2label,
                                                          ignore_mismatched_sizes=True)

"""The warning is telling us that we're throwing away the classification head and randomly initializing a new one.

## Calculate initial loss

Let's take the first batch of the training dataset and forward it through the model, see if we get a loss that makes sense.

This is another trick from [this amazing blog post](http://karpathy.github.io/2019/04/25/recipe/) if you wanna debug your neural networks.
"""

# batch = next(iter(train_dataloader))
# for k,v in batch.items():
#   if isinstance(v, torch.Tensor):
#     print(k,v.shape)
#   else:
#     print(k,len(v))

# print([label.shape for label in batch["class_labels"]])

# print([label.shape for label in batch["mask_labels"]])

# outputs = model(
#           pixel_values=batch["pixel_values"],
#           mask_labels=batch["mask_labels"],
#           class_labels=batch["class_labels"],
#       )
# outputs.loss

"""## Train the model

Let's train the model in familiar PyTorch fashion.
"""

import torch
from tqdm.auto import tqdm
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# initialize training
start_epoch = 0
running_loss = 0.0
num_samples = 0

#check if checkpoint exists
checkpoint_prefix = 'model_checkpoint_epoch_'
checkpoint_files = [f for f in os.listdir('.') if os.path.isfile(f) and f.startswith(checkpoint_prefix)]
if checkpoint_files:
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    running_loss = checkpoint['running_loss']
    num_samples = checkpoint['num_samples']
    print(f"Loaded checkpoint from epoch {start_epoch - 1}, file: {latest_checkpoint}")
else:
    print("No checkpoint found. Starting training from scratch.")



for epoch in range(100):
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

  # After each epoch, save the model
  checkpoint_path = f'{checkpoint_prefix}{epoch}.pth'
  checkpoint = {
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'running_loss': running_loss,
      'num_samples': num_samples,
  }
  torch.save(checkpoint, checkpoint_path)
  print(f"Checkpoint saved at {checkpoint_path}")

