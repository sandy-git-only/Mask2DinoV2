import argparse
import clip
import numpy as np
import os
import PIL
import torch
import time
from tqdm import tqdm
from datasets import load_dataset


def zero_shot(gpus, images, classTest):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model
    model, preprocess = clip.load('ViT-B/32', device)
    
    # Encode the images and text
    image_inputs = torch.stack([preprocess(image) for image in tqdm(images, desc="Processing images")]).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classTest]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_inputs)
        text_features = model.encode_text(text_inputs)

    # Zero-shot classification
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    return {'similarity': similarity, 'text_features': text_features}


def save_text_feature(result, image_paths):
    values = 0
    for i in tqdm(range(len(image_paths)), desc="Saving text features"):
        value, index = result['similarity'][i].topk(1)
        values += value.item()
        # save text feature for each image
        text_feature = result['text_features'][index].cpu().numpy()
        save_path = image_paths[i].replace('.jpg', '.npy').replace('.png', '.npy').replace('hazy', 'text_feature')
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        np.save(save_path, text_feature)

    return values / len(image_paths)



def main(args):
    # Prepare the inputs
    # image_folder_train = os.path.join(args.data_dir, args.dataset, 'train', 'hazy')
    # image_folder_test = os.path.join(args.data_dir, args.dataset, 'test', 'hazy')

    dataset = load_dataset("segments/sidewalk-semantic")
    dataset = dataset.shuffle(seed=1)
    dataset = dataset["train"].train_test_split(test_size=0.2)
    train_ds = dataset["train"]
    test_ds = dataset["test"]
    image_paths=[]
    # for image in os.listdir(image_folder_train):
    #     if image.endswith('.jpg') or image.endswith('.png'):
    #         image_paths.append(os.path.join(image_folder_train, image))
    # for image in os.listdir(image_folder_test):
    #     if image.endswith('.jpg') or image.endswith('.png'):
    #         image_paths.append(os.path.join(image_folder_test, image))
    
    # images = [PIL.Image.open(path) for path in image_paths]
    classTest = ['flat-road', 'flat-sidewalk', 'flat-parkingdriveway', 'flat-curb', 'human-rider', 'vehicle-car', 'vehicle-bicycle', 'construction-building', 'construction-wall', 'object-pole', 'nature-vegetation', 'sky', 'void-static']
    
    # Print number of images with hazy
    print('number of images with sidewalk: ', len(dataset))
    # Print classes
    print('classes: ', classTest)

    # Run the model for zero-shot classification
    print('--------start zero-shot classification--------')
    time_start = time.time()
    result = zero_shot(args.gpus, dataset, classTest)
    time_end = time.time()
    print('time cost: ', time_end - time_start)
    print('--------end zero-shot classification--------')


    # Save text feature
    average = save_text_feature(result, image_paths)
    print('average similarity: ', average)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./dataset/sidewalk/test', type=str, help='path to dataset')
    parser.add_argument('--dataset', default='sidewalk', type=str, help='dataset name')
    parser.add_argument('--gpus', default='1', type=str, help='GPUs used for training')

    args = parser.parse_args()
    main(args)