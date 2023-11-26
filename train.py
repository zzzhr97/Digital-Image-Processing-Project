import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import argparse
import random
import numpy as np
import time
import json
import pandas as pd
import matplotlib.pyplot as plt

from transform import transform_method
from data import hyper_dataset
import network

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

def load_data(args, seed):
    """
    load the data.

    :return: dataset, train data, validation data.
    """
    transform = transform_method(method=args.transform_method)
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

def eval(net, train_data, valid_data, threshold, loss_fn, device):
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
    train_scores = eval_scores(net, train_data, threshold, loss_fn, device)
    valid_scores = eval_scores(net, valid_data, threshold, loss_fn, device)
    return train_scores, valid_scores

def eval_scores(net, data, threshold, loss_fn, device):
    """
    Evaluate the model on the given data.
    This will use Kappa, F1 score and Specificity.
    """
    if data is None:
        return None
    
    net.eval()
    with torch.no_grad():
        TP, TN, FP, FN = 0, 0, 0, 0
        scores = {'Kappa': 0, 'F1': 0, 'Specificity': 0}
        losses = []
        for sample in data:
            image = sample['image'].unsqueeze(0).to(device, torch.float)
            label = torch.tensor(sample['label'])
            output = net(image).view(-1, 1).cpu()
            pr = F.sigmoid(output).item()

            # calculate loss
            loss = loss_fn(output, label.view(-1, 1).float())
            losses.append(loss.item())

            label_pred = int(pr >= threshold)
            TP += int(label_pred and label.item() == 1)
            FP += int(label_pred and label.item() == 0)
            TN += int(not label_pred and label.item() == 0)
            FN += int(not label_pred and label.item() == 1)

        # loss
        scores['Loss'] = sum(losses) / len(losses)

        # kappa
        po = (TP + TN) * 1.0 / (TP + TN + FP + FN)
        pe = ((TP + FP) * (TP + FN) + (TN + FP) * (TN + FN)) * 1.0 / (TP + TN + FP + FN) ** 2
        scores['Kappa'] = (po - pe) / (1 - pe)

        # F1 score
        precision = TP * 1.0 / (TP + FP)
        recall = TP * 1.0 / (TP + FN)
        scores['F1'] = 2 * precision * recall / (precision + recall)

        # specificity
        scores['Specificity'] = TN * 1.0 / (TN + FP)

        # average score of the three
        scores['Average'] = (scores['Kappa'] + scores['F1'] + scores['Specificity']) / 3

    net.train() 
    return scores

def checkpoint(net, file_name, save_path, device):
    """save the model weights to a file."""
    print("\tSaving checkpoint...")
    net.cpu()       # move to cpu
    ckpt_dict = net.state_dict()
    torch.save(ckpt_dict, os.path.join(save_path, file_name))
    net.to(device)  # move back to the device

    print(f'\tCheckpoint saved.')
    print(f'\tFile name: {file_name}')

def load_checkpoint(net, file_name, load_path, device):
    """load the model weights from a file."""
    print("\tLoading checkpoint...")
    ckpt_dict = torch.load(os.path.join(load_path, file_name), map_location=device)
    net.load_state_dict(ckpt_dict)

    print(f'\tCheckpoint loaded.')
    print(f'\tFile name: {file_name}')

def save_results(results, file_name, save_path):
    """save the results to a file."""
    print("\tSaving results...")
    #with open(os.path.join(save_path, file_name), 'w') as f:
        #json.dump(results, f, indent=4)

    df = pd.json_normalize(results) # convert results to a dataframe
    df.to_csv(os.path.join(save_path, file_name), index=False)

    print(f'\tResults saved.')
    print(f'\tFile name: {file_name}')

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

def search_threshold(args, seed, loss_fn, load_path, data, device):
    """search the best threshold in validation set for classification."""
    print('Searching the best threshold in validation set for classification...')
    net = getattr(network, args.model)()
    net = net.to(device)
    load_checkpoint(net, f'lr{args.lr}_bs{args.batch_size}_epochs{args.n_epochs}_seed{seed}_best.pth', load_path, device)
    net.eval()

    with torch.no_grad():
        _, valid_scores = eval(net, None, data, args.threshold, loss_fn, device)

        best_threshold = 0.5
        best_scores = valid_scores
        th_range = torch.arange(0.4, 0.6, 0.01)
        for threshold in th_range:
            if threshold == 0.5:
                continue
            _, new_scores = eval(net, None, data, threshold, loss_fn, device)

            if new_scores["Average"] > best_scores["Average"]:
                best_scores = new_scores
                best_threshold = threshold

        print(f'\tBest valid score {best_scores}')
        print(f'\tBest threshold {best_threshold}')

    return best_threshold.item()

def train(args, seed=123):
    # set seed
    set_seed(seed)

    # load data
    dataset, train_data, valid_data = load_data(args, seed)
    print(f'# of training data: {len(train_data)} \t# of validation data: {len(valid_data)}')

    # path to save checkpoints
    ckpt_save_path = os.path.join(args.ckpt_dir, f'task-{args.task}', f'{args.model}')
    result_save_path = os.path.join(args.result_dir, f'task-{args.task}', f'{args.model}')
    os.makedirs(ckpt_save_path, exist_ok=True)
    os.makedirs(result_save_path, exist_ok=True)

    # set device
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    print(f'device: {device}')

    # init model
    # example: if args.model == 'ResNet34', then net = network.ResNet34()
    net = getattr(network, args.model)()
    net = net.to(device)

    # calculate the fraction of (<nagetive labels number> / <positive labels number>)
    # this is used to balance the loss function
    n_positive = sum([x['label'] == 1 for x in train_data])
    n_negative = len(train_data) - n_positive
    print(f'\t# of positive labels: {n_positive}\t# of negative labels: {n_negative}')
    pos_weight = torch.tensor(n_negative * 1.0 / n_positive)

    # loss function: sigmoid + BCELoss
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # optimizer
    params = net.parameters()
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(.9, .999), weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=.9, weight_decay=args.weight_decay)

    best_valid_result = {'Loss': 1e10, 'Kappa': 0.0, 'F1': 0.0, 'Specificity': 0.0, 'Average': 0.0}
    results = []
    losses = []
    start_time = time.time()

    # args.lr_decay_epochs为一个列表，指定在哪几个轮次降低学习率
    # args.lr_decay_values为一个列表，指定在这几个轮次中分别降低到哪个值
    assert len(args.lr_decay_epochs) == len(args.lr_decay_values), \
        f"lr_decay_epochs and lr_decay_values should have the same length, " \
        f"but got {len(args.lr_decay_epochs)} and {len(args.lr_decay_values)}"
    lr_decay = dict(zip(args.lr_decay_epochs, args.lr_decay_values))

    # calculate the number of batches
    n_batches = len(train_data) // args.batch_size
    if len(train_data) % args.batch_size != 0:
        n_batches += 1

    # training for each epoch
    for epoch in range(args.n_epochs):
        
        # learning rate decay
        if epoch in lr_decay.keys():
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay[epoch]
            print(f"At the {epoch}-th epoch, decay lr to {lr_decay[epoch]}.")

        # shuffle the training data for each epoch
        permutation = np.random.permutation(len(train_data))
        train_data = [train_data[i] for i in permutation]

        # training for each batch
        for batch_idx in range(n_batches):

            # calculate training data index in this batch
            start_idx = batch_idx * args.batch_size
            end_idx = start_idx + args.batch_size
            if end_idx > len(train_data):
                end_idx = len(train_data)

            # get the training data for this batch
            batch_data = train_data[start_idx:end_idx]
            images = torch.stack([x['image'] for x in batch_data])
            labels = torch.tensor([x['label'] for x in batch_data])

            # print the input shape of the first batch in the first epoch
            if epoch == 0 and batch_idx == 0:   
                print(f'\tBatch images shape: {images.shape}')
                print(f'\tBatch labels shape: {labels.shape}')

            # move data to device
            images = images.to(device, torch.float)
            labels = labels.to(device, torch.float)

            # forward pass
            output = net(images)

            # print the output shape of the first batch in the first epoch
            if epoch == 0 and batch_idx == 0:
                print(f'\tBatch output shape: {output.shape}')

            # calculate loss
            loss = loss_fn(output, labels.view(-1, 1).float())

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print the loss
            if (batch_idx + 1) % args.print_every == 0:
                print(f'\tEpoch {epoch + 1} || Batch {batch_idx + 1:3} || Loss: {loss.item():.4f}')

        # evaluate the model on the validation set
        if (epoch + 1) % args.eval_every == 0:
            eval_start_time = time.time()
            train_scores, valid_scores = eval(net, train_data, valid_data, args.threshold, loss_fn, device)
            print(f'Epoch {epoch + 1}',
                f'|| train score: {train_scores["Average"]:.4f}',
                f'|| valid score: {valid_scores["Average"]:.4f}',
                f'|| train loss: {train_scores["Loss"]:.4f}',
                f'|| valid loss: {valid_scores["Loss"]:.4f}',
                f'|| time: {time.time() - start_time:.2f}s',
                f'|| eval time: {time.time() - eval_start_time:.2f}s'
            )
            
            # save the best network weights
            if valid_scores["Loss"] < best_valid_result["Loss"]:
                best_valid_result = valid_scores
                checkpoint(net, f'lr{args.lr}_bs{args.batch_size}_epochs{args.n_epochs}_seed{seed}_best.pth', ckpt_save_path, device)

            # save the results
            results.append({
                'epoch': epoch + 1,
                'train_score': train_scores,
                'valid_score': valid_scores,
            })

        # save the network weights
        if (epoch + 1) % args.ckpt_every == 0:
            checkpoint(net, f'lr{args.lr}_bs{args.batch_size}_epochs{args.n_epochs}_seed{seed}_epoch{epoch + 1}.pth', ckpt_save_path, device)

    # training finished
    print(f'Best valid score: {best_valid_result}')
    print(f'Total time: {time.time() - start_time:.2f}s')

    # search the best threshold in validation set for classification
    best_threshold = search_threshold(args, seed, loss_fn, ckpt_save_path, valid_data, device)
    results.append({'best_threshold': best_threshold})

    # save the results
    # format: 
    #   the last element is {'best_threshold': best threshold in validation set}
    #   the other elements are {'epoch': epoch, 'train_score': train scores list, 'valid_score': valid scores list}
    save_results(results, f'lr{args.lr}_bs{args.batch_size}_epochs{args.n_epochs}_seed{seed}.csv', result_save_path)

    # visualize the results
    visualize_results(results, best_valid_result)

def main(args):
    for seed in args.seed:
        train(args, seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # base parameters
    parser.add_argument('--data_dir', type=str, default='dataset', help='path to dataset')
    parser.add_argument('--task', type=int, default=1, help='task number')
    parser.add_argument('--seed', nargs="+", type=int, default=[123], help='random seed list')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='device used')

    # optimization parameters
    parser.add_argument('--n_valid', type=int, default=64, help='number of validation images')
    parser.add_argument('--transform_method', type=int, default=1, help='transform method number')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_epochs', nargs="+", type=int, default=[],
                        help="decay learning rate at these epochs")
    parser.add_argument('--lr_decay_values', nargs="+", type=float, default=[], 
                        help='modify learning rate by this value')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--is_shuffle', type=bool, default=False, help='shuffle the training/validation data partition')
    parser.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam', help='optimizer')

    # model parameters
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold for classification')
    parser.add_argument('--model', choices=['ResNet34', 'ResNet50', 'ResNet152', 'TestNet'], default='ResNet34', help='model name')

    # logging parameters
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='path to saved checkpoints')
    parser.add_argument('--result_dir', type=str, default='results', help='path to saved results')
    parser.add_argument('--ckpt_every', type=int, default=25, help='epochs between checkpoint save')
    parser.add_argument('--eval_every', type=int, default=5, help='epochs between evaluation on validation set')
    parser.add_argument('--print_every', type=int, default=2, help='batches between print in each epoch')

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = parser.parse_args()
    print("-------------------BEGIN-------------------")
    main(args)
