from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, Trainer, TrainingArguments,  Mask2FormerImageProcessor
from PIL import Image
import requests
import torch
from datasets import load_dataset
import numpy as np
import transformers
# 載入模型
model = transformers.Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")

# 設定訓練參數
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = transformers.get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=1000,
    num_training_steps=10000,
)

# 載入訓練資料
dataset = load_dataset("nateraw/ade20k-tiny")
image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-ade-semantic")

from PIL import Image

def preprocess_function(examples):
    images = examples["image"]
    labels = examples["label"]
    target_size = (512, 512)  # Define a target size

    if not isinstance(images, list):
        images = [images]

    processed_images = []
    for image in images:
        # Ensure image is a PIL Image for consistency in processing
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Resize image to target size if necessary
        # Note: This resizing step is optional and depends on your specific needs
        # It's included here for demonstration purposes
        resized_image = image.resize(target_size, resample=Image.BILINEAR)
        # Convert resized image to numpy array if it's not already
        image_np = np.array(resized_image)

        # Add batch dimension and convert image to tensor
        image_tensor = torch.tensor(image_np).unsqueeze(0)
        processed_images.append(image_tensor)

    # Stack all image tensors to create a batch
    pixel_values = torch.cat(processed_images, dim=0)

    resized_labels = []
    for label in labels:
        # Ensure label is in numpy array format
        if isinstance(label, Image.Image):
            label = np.array(label)

        # Convert label to PIL Image for resizing
        label_pil = Image.fromarray(label)
        resized_label = label_pil.resize(target_size, resample=Image.NEAREST)
        resized_labels.append(np.array(resized_label, dtype=np.int64))

    # Stack resized labels into a tensor
    label_array = torch.tensor(np.stack(resized_labels))
    return {"pixel_values": pixel_values, "mask_labels": label_array}



batch_size = 2
# 將資料轉換成Dataset格式
train_dataset = dataset["train"].map(preprocess_function, batched=True, batch_size=batch_size)
# print("train dataset",train_dataset[0]["pixel_values"])
eval_dataset = dataset["validation"].map(preprocess_function, batched=True, batch_size=batch_size)
from torch.utils.data import DataLoader

def collate_fn(batch):
    pixel_values_list = [item['pixel_values'] for item in batch]
    mask_labels_list = [item['mask_labels'] for item in batch]

    # Convert lists to tensors if they are not already
    if isinstance(pixel_values_list[0], list):
        pixel_values = torch.stack([torch.tensor(pv) for pv in pixel_values_list])
    else:
        pixel_values = torch.stack(pixel_values_list)

    if isinstance(mask_labels_list[0], list):
        mask_labels = torch.stack([torch.tensor(ml, dtype=torch.int64) for ml in mask_labels_list])
    else:
        mask_labels = torch.stack(mask_labels_list)

    return {"pixel_values": pixel_values, "mask_labels": mask_labels}


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# 訓練模型
for epoch in range(10):
    for batch in train_loader:
        
        # 將資料送入模型
        images = batch["pixel_values"].to(device)
        labels = batch["mask_labels"].to(device)
        outputs = model(pixel_values=images, mask_labels=labels)

        # 計算損失
        loss = outputs.loss

        # 更新參數
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新學習率
        scheduler.step()

# 儲存模型
model.save_pretrained("output/mask2former")
