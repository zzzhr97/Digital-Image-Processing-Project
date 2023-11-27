import torch
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

def set_seed(seed):
    """set random seed."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)    # numpy 
    random.seed(seed)      

    # ensure reproducibility
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def checkpoint(net, file_name, save_path, device):
    """save the model weights to a file."""
    net.cpu()       # move to cpu
    ckpt_dict = net.state_dict()
    torch.save(ckpt_dict, os.path.join(save_path, file_name))
    net.to(device)  # move back to the device

    print(f'\tCheckpoint saved to {file_name}')

def load_checkpoint(net, file_name, load_path, device):
    """load the model weights from a file."""
    ckpt_dict = torch.load(os.path.join(load_path, file_name), map_location=device)
    net.load_state_dict(ckpt_dict)

    print(f'\tCheckpoint loaded to {file_name}')

def visualize_results(results, best_valid_result):
    """visualize the results."""
    results = results[:-1]  # remove the last element {'best_threshold': best threshold in validation set}
    counter = [i['epoch'] for i in results]
    train_losses = [i['train_score']['Loss'] for i in results]
    valid_losses = [i['valid_score']['Loss'] for i in results]
    train_avg_score = [i['train_score']['Average'] for i in results]
    valid_avg_score = [i['valid_score']['Average'] for i in results]

    plt.subplot(1, 2, 1)
    plt.plot(counter, train_losses, color='red')
    plt.plot(counter, valid_losses, color='green')
    plt.title("Loss")
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(counter, train_avg_score, color='red')
    plt.plot(counter, valid_avg_score, color='green')
    plt.title("Average Score")
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Average Score')

    plt.tight_layout()
    plt.show()

def show_image(image, label=None, name=None):
    """show an image."""
    # 转换tensor为NumPy数组
    img_array = image.permute(1, 2, 0).numpy()
    img_array = img_array[:, :, [2, 1, 0]]  # BRG --> RGB

    # 添加文本注释
    text_info = f'Name: {name}\nShape: {image.shape}\nLabel: {label}'
    plt.annotate(text_info, xy=(0, 0), xytext=(10, -30), textcoords='offset points', ha='left', va='top', color='white', fontsize=10)

    plt.imshow(img_array)
    plt.show()

def save_results(results, file_name, save_path):
    """save the results to a file."""
    #with open(os.path.join(save_path, file_name), 'w') as f:
        #json.dump(results, f, indent=4)

    df = pd.json_normalize(results) # convert results to a dataframe
    df.to_csv(os.path.join(save_path, file_name), index=False)

    print(f'\tResults saved to {file_name}')

def cal_scores(losses, TP, FP, TN, FN):
    """
    calculate scores.

    Returns: 
        - scores (dict), example: {'Loss': 10000.0, 'Kappa': 0.0, 'F1': 0.0, 'Specificity': 0.0, 'Average': 0.0}
    """
    scores = {'Loss': 10000.0, 'Kappa': 0.0, 'F1': 0.0, 'Specificity': 0.0, 'Average': 0.0}

    # loss
    scores['Loss'] = sum(losses) / len(losses)

    # kappa
    po = (TP + TN) * 1.0 / (TP + TN + FP + FN)
    pe = ((TP + FP) * (TP + FN) + (TN + FP) * (TN + FN)) * 1.0 / (TP + TN + FP + FN) ** 2
    scores['Kappa'] = (po - pe) / (1 - pe)

    # F1 score
    precision = TP * 1.0 / (TP + FP + 1e-10)
    recall = TP * 1.0 / (TP + FN + 1e-10)
    scores['F1'] = 2 * precision * recall / (precision + recall + 1e-10)

    # specificity
    scores['Specificity'] = TN * 1.0 / (TN + FP + 1e-10)

    # average score of the three
    scores['Average'] = (scores['Kappa'] + scores['F1'] + scores['Specificity']) / 3

    return scores
