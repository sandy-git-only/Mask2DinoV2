from datasets import load_dataset
import os

dataset = load_dataset("segments/sidewalk-semantic")
dataset = dataset.shuffle(seed=1)
dataset = dataset["train"].train_test_split(test_size=0.2)
train_ds = dataset["train"]
test_ds = dataset['test']
save_dir = "dataset/sidewalk/test"

os.makedirs(save_dir, exist_ok=True)

for i, item in enumerate(test_ds):
    print("i", i, "item", item['pixel_values'])
    image = item['pixel_values']
    save_path = os.path.join(save_dir, f"image_{i}.png")
    image.save(save_path)
