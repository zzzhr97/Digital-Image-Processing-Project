import argparse
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cv2

from data import load_data

def save_batch_image(batch_data, save_path, is_show, batch_height=2, batch_width=5):
    """Save a batch of images and labels to a figure."""
    image_height, image_width = batch_data[0]['image'].shape[:2]
    assert image_height == 800 and image_width == 800, f'Image size must be 800x800, but got {image_height}x{image_width}.'

    # create subplots
    fig, axs = plt.subplots(batch_height, batch_width, figsize=(batch_width * 2, batch_height * 2))

    # add images and labels to the subplots
    for i in range(batch_height):
        for j in range(batch_width):
            single_data = batch_data[i * batch_width + j]
            axs[i, j].imshow(single_data['image'])
            axs[i, j].axis('off')

            label_text = f"{single_data['name'][:-4]} [{single_data['label']}]"
            axs[i, j].text(0.5, 1.1, label_text, transform=axs[i, j].transAxes,
                verticalalignment='top', horizontalalignment='center', color='black', fontsize=9)
            
    plt.tight_layout()

    # save figure
    plt.savefig(save_path, dpi=500)

    # show figure
    if is_show == '1':
        plt.axis('off')
        plt.show()
    plt.close()

def main(args):
    print(f"Task: {args.task}")

    # load data
    print("Loading data...", end=' ')
    _, total_data, _ = load_data(args, 0, seed=123)
    print("Done.")
    print("Number of samples:", len(total_data))

    # processing data: convert BGR to RGB
    print("Processing data...", end=' ')
    with tqdm(total=len(total_data), desc="Processing", unit="sample") as pbar:
        for idx, sample in enumerate(total_data):
            # convert BGR to RGB
            total_data[idx]['image'] = cv2.cvtColor(total_data[idx]['image'], cv2.COLOR_BGR2RGB)

            pbar.update(1)
            pbar.set_postfix_str(f"Processing {idx:03d}-th sample")

        pbar.set_description("Processing done")
        pbar.set_postfix_str()

    os.makedirs(args.save_dir, exist_ok=True)

    # set number of batches, the last batch will be dropped if its size is not equal to batch_size
    batch_size = args.batch_height * args.batch_width
    n_batches = len(total_data) // batch_size

    # save and show images
    with tqdm(total=n_batches, desc="Generating", unit="batch") as pbar:
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            assert end_idx < len(total_data), f"end_idx {end_idx} >= len(total_data) {len(total_data)}"

            batch_data = total_data[start_idx:end_idx]
            save_path = os.path.join(args.save_dir, f'batch_{batch_idx:03d}.png')
            save_batch_image(batch_data, save_path, args.is_show)

            pbar.update(1)
            pbar.set_postfix_str(f"Generating {batch_idx:03d}-th figure")

        pbar.set_description("Generating done")
        pbar.set_postfix_str()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize the dataset.')

    parser.add_argument('--is_show', type=str, default=1, help='Whether to show the images. 1 for yes, 0 for no.')
    parser.add_argument('--data_dir', type=str, default='dataset', help='Path to the dataset directory.')
    parser.add_argument('--save_dir', type=str, default='visual', help='Path to the save directory.')
    parser.add_argument('--task', type=int, default=1, help='task number')
    parser.add_argument('--begin_idx', type=int, default=0, help='begin index')
    parser.add_argument('--batch_height', type=int, default=2, help='the height of the figure (unit: image)')
    parser.add_argument('--batch_width', type=int, default=5, help='the width of the figure (unit: image)')

    args = parser.parse_args()
    setattr(args, 'is_shuffle', False)
    setattr(args, 'n_valid', 0)

    main(args)