import torch
import torch.nn.functional as F
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

import transform as tr

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
    plt.legend(['Train', 'Validation'], loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(counter, train_avg_score, color='red')
    plt.plot(counter, valid_avg_score, color='green')
    plt.title("Average Score")
    plt.legend(['Train', 'Validation'], loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Average Score')

    plt.tight_layout()
    plt.savefig('./results.png')
    print(f'\tImage results saved to results.png')

def show_image(image, label=None, name=None):
    """show an image."""
    # 转换tensor为NumPy数组
    img_array = image.permute(1, 2, 0).numpy()
    img_array = img_array[:, :, [2, 1, 0]]  # BRG --> RGB

    if img_array.max() > 1:
        img_array = img_array * 1.0 / 255

    # 添加文本注释
    text_info = f'Name: {name}\nShape: {image.shape}\nLabel: {label}'
    plt.annotate(text_info, xy=(0, 0), xytext=(10, -30), textcoords='offset points', ha='left', va='top', color='white', fontsize=10)

    plt.imshow(img_array)
    plt.show()

def save_results(args, results, file_name, save_path):
    """save the results to a file."""
    #with open(os.path.join(save_path, file_name), 'w') as f:
        #json.dump(results, f, indent=4)

    total_results = []

    # save args
    for key, value in vars(args).items():
        total_results.append({'param': key, 'value': value})

    # add results and best threshold to total_results
    if args.is_search:
        total_results = results[:-1] + total_results + [results[-1]]
    else:
        total_results = results + total_results

    df = pd.json_normalize(total_results) # convert results to a dataframe
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

    TP = TP + 1e-10
    FP = FP + 1e-10
    TN = TN + 1e-10
    FN = FN + 1e-10

    # kappa
    po = (TP + TN)  / (TP + TN + FP + FN)
    pe = ((TP + FP) * (TP + FN) + (TN + FP) * (TN + FN)) / (TP + TN + FP + FN) ** 2
    scores['Kappa'] = (po - pe) / (1 - pe)

    # F1 score
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    scores['F1'] = 2 * precision * recall / (precision + recall + 1e-10)

    # specificity
    scores['Specificity'] = TN / (TN + FP)

    # average score of the three
    scores['Average'] = (scores['Kappa'] + scores['F1'] + scores['Specificity']) / 3

    return scores

def transform_data(data, transform_method_epoch):
    """
    Transform the data.

    :param data_list (torch.tensor): Input batch data. Shape: (batch_size, 3, x, x)
    :param transform_method: number of transform method.
    :return: Transformed data.
    """
    if transform_method_epoch:
        transform = tr.transform_method(method=transform_method_epoch)
        for i in range(len(data)):
            data[i] = transform(data[i])

    return data

def cross_entropy_loss(output, label):
    """cross entropy loss with one-hot encoding."""
    loss = torch.nn.CrossEntropyLoss()(output, label.squeeze(1).long())
    return loss
