import torch
import torch.nn.functional as F
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

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

    print(f'\tCheckpoint loaded from {file_name}')

# def visualize_results(results, best_valid_result):
#     """visualize the results."""
#     if list(results[-1].keys())[0] == 'best_threshold':
#         results = results[:-1]  # remove the last element {'best_threshold': best threshold in validation set}
#     counter = [i['epoch'] for i in results]
#     train_losses = [i['train_score']['Loss'] for i in results]
#     valid_losses = [i['valid_score']['Loss'] for i in results]
#     train_avg_score = [i['train_score']['Average'] for i in results]
#     valid_avg_score = [i['valid_score']['Average'] for i in results]

#     plt.subplot(1, 2, 1)
#     plt.plot(counter, train_losses, color='red')
#     plt.plot(counter, valid_losses, color='green')
#     plt.title("Loss")
#     plt.legend(['Train', 'Validation'], loc='best')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')

#     plt.subplot(1, 2, 2)
#     plt.plot(counter, train_avg_score, color='red')
#     plt.plot(counter, valid_avg_score, color='green')
#     plt.title("Average Score")
#     plt.legend(['Train', 'Validation'], loc='best')
#     plt.xlabel('Epoch')
#     plt.ylabel('Average Score')

#     plt.tight_layout()
#     plt.savefig('./results.png')
#     print(f'\tImage results saved to results.png')
    
def visualize_results(results, best_valid_result):
    """Visualize the results."""
    if list(results[-1].keys())[0] == 'best_threshold':
        results = results[:-1]  # remove the last element {'best_threshold': best threshold in validation set}
    
    counter = [i['epoch'] for i in results]
    train_losses = [i['train_score']['Loss'] for i in results]
    valid_losses = [i['valid_score']['Loss'] for i in results]
    train_avg_score = [i['train_score']['Average'] for i in results]
    valid_avg_score = [i['valid_score']['Average'] for i in results]

    # Increase figure size for better readability
    plt.figure(figsize=(12, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(counter, train_losses, color='red', label='Train')
    plt.plot(counter, valid_losses, color='green', label='Validation')
    plt.title("Loss", fontsize=16, fontweight='bold')  # Increase title font size and make it bold
    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)  # Change grid color

    # Plot Average Score
    plt.subplot(1, 2, 2)
    plt.plot(counter, train_avg_score, color='red', label='Train')
    plt.plot(counter, valid_avg_score, color='green', label='Validation')
    plt.title("Average Score", fontsize=16, fontweight='bold')  # Increase title font size and make it bold
    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)  # Change grid color

    # Adjust layout and save the figure
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

def restore_results(result_path):
    """Generate a result list by a .csv file """
    df = pd.read_csv(result_path)
    n_epochs = int(df[df['param'] == 'n_epochs']['value'].values[0])

    selected_columns = df.columns[:-2]
    original_result = df[selected_columns].to_dict('records')[:n_epochs]

    new_result = []
    for original_dict in original_result:
        new_dict = {}
        for key, value in original_dict.items():
            keys = key.split('.')
            current_dict = new_dict

            for k in keys[:-1]:
                current_dict = current_dict.setdefault(k, {})
            current_dict[keys[-1]] = value

        new_result.append(new_dict)
    return new_result

def save_results_with_writer(result, save_path):
    """Save the results to a tensorboard file."""
    os.makedirs(save_path, exist_ok=True)
    writer = SummaryWriter(save_path)
    for i, epoch_result in enumerate(result):
        writer.add_scalar('1.Average/train', epoch_result['train_score']['Average'], i+1)
        writer.add_scalar('1.Average/valid', epoch_result['valid_score']['Average'], i+1)
        writer.add_scalar('2.Loss/train', epoch_result['train_score']['Loss'], i+1)
        writer.add_scalar('2.Loss/valid', epoch_result['valid_score']['Loss'], i+1)
        writer.add_scalar('3.Kappa/train', epoch_result['train_score']['Kappa'], i+1)
        writer.add_scalar('3.Kappa/valid', epoch_result['valid_score']['Kappa'], i+1)
        writer.add_scalar('4.F1/train', epoch_result['train_score']['F1'], i+1)
        writer.add_scalar('4.F1/valid', epoch_result['valid_score']['F1'], i+1)
        writer.add_scalar('5.Specificity/train', epoch_result['train_score']['Specificity'], i+1)
        writer.add_scalar('5.Specificity/valid', epoch_result['valid_score']['Specificity'], i+1)

    writer.close()
    print(f'\tImages visualization saved to {save_path}')

def eval(net, train_data, valid_data, threshold, loss_fn, out_dim, device):
    """
    Evaluate the model on the training set and validation set.
    This will use Kappa, F1 score and Specificity.

    Format: {
        'Loss': average loss, 
        'Kappa': kappa, 
        'F1': f1, 
        'Specificity': specificity,
        'Average': average score of the above three
    }

    Returns: train scores, validation scores
    """
    train_scores, train_TFPN = eval_scores(net, train_data, threshold, loss_fn, out_dim, device)
    valid_scores, valid_TFPN = eval_scores(net, valid_data, threshold, loss_fn, out_dim, device)
    return train_scores, train_TFPN, valid_scores, valid_TFPN

def eval_scores(net, data, threshold, loss_fn, out_dim, device):
    """
    Evaluate the model on the given data.
    This will use Kappa, F1 score and Specificity.
    """
    if data is None:
        return None
    
    net.eval()
    with torch.no_grad():
        TP, TN, FP, FN = 0, 0, 0, 0
        losses = []
        for sample in data:
            image = sample['image'].unsqueeze(0).to(device, torch.float)
            label = torch.tensor(sample['label'])
            output = net(image).cpu()

            #show_image(sample['image'], sample['label'], sample['name'])

            # calculate loss
            loss = loss_fn(output, label.view(-1, 1).float())
            losses.append(loss.item())

            if out_dim == 1:
                pr = F.sigmoid(output).item()
                label_pred = int(pr >= threshold)
            elif out_dim == 2:
                label_pred = int(output.argmax(1).item())

            TP += int(label_pred and label.item() == 1)
            FP += int(label_pred and label.item() == 0)
            TN += int(not label_pred and label.item() == 0)
            FN += int(not label_pred and label.item() == 1)

        scores = cal_scores(losses, TP, FP, TN, FN)
        TFPN = {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}

    net.train() 
    return scores, TFPN

def cal_scores(losses, TP, FP, TN, FN):
    """
    calculate scores.

    Returns: 
        - scores (dict), example: {'Loss': 10000.0, 'Kappa': 0.0, 'F1': 0.0, 'Specificity': 0.0, 'Average': 0.0}
    """
    scores = {}

    # loss
    if losses is not None:
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
    scores['F1'] = 2 * precision * recall / (precision + recall)

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
