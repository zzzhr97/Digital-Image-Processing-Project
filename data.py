import os
import cv2
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import time
import matplotlib.pyplot as plt

from transform import transform_method

TASK_NAME = {
    1: 'Hypertensive Classification',
    2: 'Hypertensive Retinopathy Classification',
}

def load_data(args, transform_method_origin, seed):
    """
    load the data.

    train/validation data format: 
        [idx: {'image': image, 'label': label, 'name': name}]

    Returns: dataset, train data, validation data.
    """
    transform = transform_method(method=transform_method_origin)
    dataset = hyper_dataset(
        args.data_dir, 
        task=args.task, 
        n_valid=args.n_valid, 
        is_shuffle=args.is_shuffle, 
        seed=seed,
        transform=transform
    )
    train_data, valid_data = dataset.get_data()
    return dataset, train_data, valid_data

class hyper_dataset(Dataset):
    def __init__(self, data_dir, task=1, n_valid=64, is_shuffle=False, seed=123, transform=None):
        """
        Initializes the custom dataset.

        Parameters:
        - data_dir (str): Path to the directory containing the dataset.
        - task (int): Specifies the task associated with the dataset. 
            Should be either 1 or 2.
        - n_valid (int): Number of validation images.
        - is_shuffle (bool): Flag indicating whether to shuffle the dataset.
        - seed (int): Random seed.
        - transform (callable, optional): Optional transform to be applied on each image.

        Attributes:
        - data_dir (str): Path to the dataset directory.
        - task (int): Task identifier (1 or 2).
        - n_valid (int): Number of validation images.
        - is_shuffle (bool): Flag indicating whether to shuffle the dataset.
        - seed (int): Random seed.
        - transform (callable, optional): Transform applied to each image.
        - image_dir (str): Path to the images directory.
        - label_dir (str): Path to the labels directory.
        - label_file_name (str): Name of the csv file containing the labels.
        - image_paths (list): List of file paths for each image in the dataset.
        - labels (dict): Dictionary of labels for each image in the dataset.
        - images (list): List of images in the dataset.
        - permutation (list): List of indices used to shuffle the dataset.
            Only used if is_shuffle is True.

        """
        self.data_dir = data_dir
        self.task = task
        self.n_valid = n_valid
        self.is_shuffle = is_shuffle
        self.seed = seed
        self.transform = transform 
        self.image_dir = os.path.join(data_dir, f'{task}-' + TASK_NAME[task], '1-Images', '1-Training Set') 
        self.label_dir = os.path.join(data_dir, f'{task}-' + TASK_NAME[task], '2-Groundtruths')
        self.label_file_name = 'HRDC ' +  TASK_NAME[task] + ' Training Labels.csv'
        self.image_paths = [os.path.join(self.image_dir, img) \
                            for img in os.listdir(self.image_dir) \
                            if img.endswith(('.jpg', '.png'))]
        
        # set random seed
        self.set_seed()
        
        # load label of each image (a dictionary)
        self.load_labels(os.path.join(self.label_dir, self.label_file_name))

        # load all the data to memory
        self.load_data_to_memory()

    def set_seed(self):
        """Set random seed."""
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)    # numpy 
        random.seed(self.seed)
        
    def load_labels(self, label_file):
        """Returns a dictionary of labels from a csv file."""
        label_df = pd.read_csv(label_file)
        if self.task == 1:
            self.labels = {row['Image']: row['Hypertensive'] for _, row in label_df.iterrows()}
        if self.task == 2:
            self.labels = {row['Image']: row['Hypertensive Retinopathy'] for _, row in label_df.iterrows()}
    
    def get_data(self):
        """
        Returns training data and validation data.
    
        Format: [idx: {'image': image, 'label': label, 'name': name}]

        Returns:
        - train_data (list): List of training data dictionaries.
        - valid_data (list): List of validation data dictionaries.
        """
        data = [self.__getitem__(idx) for idx in range(len(self.image_paths))]
        if self.is_shuffle:
            self.permutation = np.random.permutation(len(data))
            data = [data[i] for i in self.permutation]

        # return all data for the first if n_valid is 0
        if self.n_valid == 0:
            return data, None
        
        train_data = data[:-self.n_valid]
        valid_data = data[-self.n_valid:]
        return train_data, valid_data
    
    def load_data_to_memory(self):
        """Load images to memory."""
        self.images = [cv2.imread(img_path, 1) for img_path in self.image_paths]

        # transform image if specified
        if self.transform:
            self.images = [self.transform(image) for image in self.images]

    def __len__(self):
        """Returns the total number of image files."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Returns an image and its label based on the index."""
        image = self.images[idx]  
        name = os.path.basename(self.image_paths[idx])
        label = self.labels[name]

        return {'image': image, 'label': label, 'name': name}


# test the dataset
if __name__ == '__main__':

    batch_size = 64
    test_equality = 1           # whether to test data consistency
    data_dir = 'dataset'
    transform = transform_method(method=1)
    dataset = hyper_dataset(data_dir, task=1, n_valid=64, is_shuffle=True, seed=123, transform=transform)

    # show one image and its label
    idx_show = 1
    image_name = dataset[idx_show]['name']
    image = dataset[idx_show]['image']
    image_label = dataset[idx_show]['label']   
    print('Number of images:', len(dataset))
    print('Image name:', image_name)
    print('Image shape:', image.shape)
    print('Image label:', image_label)
    print('Image part:', image[0, 150:160, 150:160])

    # use dataloader (slow)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # not use dataloader (fast)
    data, _ = dataset.get_data()
    print(len(data), len(_))
    n_batches = len(data) // batch_size
    if len(data) % batch_size != 0:
        n_batches += 1

    for epoch in tqdm(range(1)):
        permutation = np.random.permutation(len(data))
        data = [data[i] for i in permutation]
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            if end_idx > len(data):
                end_idx = len(data)

            batch_data = data[start_idx:end_idx]
            images = torch.stack([x['image'] for x in batch_data])
            labels = torch.tensor([x['label'] for x in batch_data])
            names = [x['name'] for x in batch_data]

            if epoch == 0 and batch_idx == 0:
                print("Batch images shape:", images.shape)
                print("Batch labels shape:", labels.shape)

            if test_equality and dataset[idx_show]['name'] in names:
                print("Found image!")
                idx = 0
                for i in range(len(names)):
                    if dataset[idx_show]['name'] == names[i]:
                        idx = i
                        break
                print("Image shape:", images[idx].shape)
                print("Label:", labels[idx])
                print("Name:", names[idx])
                print("Is equal:", torch.equal(images[idx], image))

    # 转换tensor为NumPy数组
    img_array = image.permute(1, 2, 0).numpy()
    img_array = img_array[:, :, [2, 1, 0]]  # BRG --> RGB
    # 在图像上添加文本注释
    text_info = f'Name: {image_name}\nShape: {image.shape}\nLabel: {image_label}'
    plt.annotate(text_info, xy=(0, 0), xytext=(10, -30), textcoords='offset points', ha='left', va='top', color='white', fontsize=10)

    # 显示第<idx_show>个图像  
    plt.imshow(img_array)
    plt.show()
