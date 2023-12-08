import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
import time

from data import load_data
import transform
import network
import utils

def search_threshold(args, seed, loss_fn, load_path, data, device):
    """search the best threshold in validation set for classification."""
    print('Searching the best threshold in validation set for classification...')
    net = getattr(network, args.model)(num_classes=args.out_dim)
    net = net.to(device)
    utils.load_checkpoint(net, f'lr{args.lr}_bs{args.batch_size}_epochs{args.n_epochs}_seed{seed}_best.pth', load_path, device)
    net.eval()

    with torch.no_grad():
        _, _, valid_scores, _ = utils.eval(net, None, data, args.threshold, loss_fn, args.out_dim, device)

        best_threshold = 0.5
        best_scores = valid_scores
        th_range = torch.arange(0.4, 0.6, 0.01).tolist()
        for threshold in th_range:
            if threshold == 0.5:
                continue
            _, _, new_scores, _ = utils.eval(net, None, data, threshold, loss_fn, args.out_dim, device)

            if new_scores["Average"] > best_scores["Average"]:
                best_scores = new_scores
                best_threshold = threshold

        print(f'========(Please ignore this line)Best valid score {best_scores} in threshold {best_threshold}')

    assert type(best_threshold) == float, f'type of best threshold is {type(best_threshold)}, but should be float'
    return best_threshold

def train(args, seed=123):
    # set seed
    utils.set_seed(seed)

    # load data
    print("Loading data...", end=' ')
    dataset, train_data, valid_data = load_data(args, 
        transform_method_origin=args.transform_method_origin, 
        seed=seed)
    print("Done.")
    print(f'number of training data: {len(train_data)} \tnumber of validation data: {len(valid_data)}')

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
    print(f'Device: {device}')

    # init model
    # example: if args.model == 'ResNet34', then net = network.ResNet34()
    net = getattr(network, args.model)(num_classes=args.out_dim)
    net = net.to(device)

    if args.out_dim == 1: 
        # calculate the fraction of (<nagetive labels number> / <positive labels number>)
        # this is used to balance the loss function
        n_positive = sum([x['label'] == 1 for x in train_data])
        n_negative = len(train_data) - n_positive
        print(f'\tnumber of positive labels: {n_positive}\tnumber of negative labels: {n_negative}')
        pos_weight = torch.tensor(n_negative * 1.0 / n_positive)

        # loss function: sigmoid + BCELoss
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif args.out_dim == 2:
        print(f'\tOutput dimension: {args.out_dim} and will use CrossEntropyLoss.')
        loss_fn = utils.cross_entropy_loss

    # tensorboard visualization
    log_dir = os.path.join('log', f'task-{args.task}', f'{args.model}')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # optimizer
    params = net.parameters()
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(args.betas[0], args.betas[1]), weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    best_valid_result_loss = {'Loss': 1e10, 'Kappa': 0.0, 'F1': 0.0, 'Specificity': 0.0, 'Average': 0.0}
    best_valid_result_score = {'Loss': 1e10, 'Kappa': 0.0, 'F1': 0.0, 'Specificity': 0.0, 'Average': 0.0}
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
    print("Begin training.")
    net.train()
    for epoch in range(args.n_epochs):
        
        # learning rate decay
        if epoch in lr_decay.keys():
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay[epoch]
            print(f"At the {epoch}-th epoch, decay lr to {lr_decay[epoch]}.")

        # shuffle the training data for each epoch
        permutation = np.random.permutation(len(train_data))
        train_data_shuffled = [train_data[i] for i in permutation]

        # training for each batch
        for batch_idx in range(n_batches):

            # calculate training data index in this batch
            start_idx = batch_idx * args.batch_size
            end_idx = start_idx + args.batch_size
            if end_idx > len(train_data):
                end_idx = len(train_data)

            # get the training data for this batch
            batch_data = train_data_shuffled[start_idx:end_idx]
            images = torch.stack([x['image'] for x in batch_data])
            labels = torch.tensor([x['label'] for x in batch_data])

            # writer: show images
            if epoch == 0:
                current_step = batch_idx
                writer.add_images(f"Train/Image", images, global_step=current_step, walltime=None, dataformats='NCHW')

            # print the input shape of the first batch in the first epoch
            if epoch == 0 and batch_idx == 0:   
                print(f'\tBatch images shape: {images.shape}')
                print(f'\tBatch labels shape: {labels.shape}')

            # move data to device
            images = images.to(device, torch.float)
            labels = labels.to(device, torch.float)

            #if batch_idx == 0: 
                #utils.show_image(images[0].detach().cpu(), labels[0].detach().cpu(), batch_data[0]['name'])   

            # implement transform_method_epoch
            images = utils.transform_data(images, args.transform_method_epoch)

            #if batch_idx == 0: 
                #utils.show_image(images[0].detach().cpu(), labels[0].detach().cpu(), batch_data[0]['name'])   

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

                current_step = epoch * n_batches + batch_idx
                writer.add_scalar("Train_batch/Loss", loss.item(), current_step)

        # evaluate the model on the validation set
        if (epoch + 1) % args.eval_every == 0:
            eval_start_time = time.time()
            train_scores, train_TFPN, valid_scores, valid_TFPN = utils.eval(
                net, train_data, valid_data, args.threshold, loss_fn, args.out_dim, device)
            print(f'[ Epoch {epoch + 1:3} ]',
                f'train score: {train_scores["Average"]:.4f}',
                f'|| valid score: {valid_scores["Average"]:.4f}',
                f'|| train loss: {train_scores["Loss"]:.4f}',
                f'|| valid loss: {valid_scores["Loss"]:.4f}',
                f'|| time: {time.time() - start_time:8.2f}s',
                f'|| eval time: {time.time() - eval_start_time:4.2f}s\n',
                f'             train TP/TN/FP/FN: {train_TFPN["TP"]:3}/{train_TFPN["TN"]:3}/{train_TFPN["FP"]:3}/{train_TFPN["FN"]:3}',
                f'|| valid TP/TN/FP/FN: {valid_TFPN["TP"]:2}/{valid_TFPN["TN"]:2}/{valid_TFPN["FP"]:2}/{valid_TFPN["FN"]:2}',
            )
            
            # save the best network weights (best loss)
            if valid_scores["Loss"] < best_valid_result_loss["Loss"]:
                best_valid_result_loss = valid_scores
                utils.checkpoint(net, f'lr{args.lr}_bs{args.batch_size}_epochs{args.n_epochs}_seed{seed}_best_loss.pth', ckpt_save_path, device)

            # save the best network weights (best score)
            if valid_scores["Average"] > best_valid_result_score["Average"]:
                best_valid_result_score = valid_scores
                utils.checkpoint(net, f'lr{args.lr}_bs{args.batch_size}_epochs{args.n_epochs}_seed{seed}_best_score.pth', ckpt_save_path, device)

            # save the results
            results.append({
                'epoch': epoch + 1,
                'train_score': train_scores,
                'valid_score': valid_scores,
            })

            # visualize
            current_step = epoch
            writer.add_scalar("Valid/Average", valid_scores["Average"], current_step)
            writer.add_scalar("Train/Average", train_scores["Average"], current_step)
            writer.add_scalar("Train/Loss", train_scores["Loss"], current_step)
            writer.add_scalar("Valid/Loss", valid_scores["Loss"], current_step)
            writer.add_scalar("Train/Kappa", train_scores["Kappa"], current_step)
            writer.add_scalar("Valid/Kappa", valid_scores["Kappa"], current_step)
            writer.add_scalar("Train/F1", train_scores["F1"], current_step)
            writer.add_scalar("Valid/F1", valid_scores["F1"], current_step)
            writer.add_scalar("Train/Specificity", train_scores["Specificity"], current_step)
            writer.add_scalar("Valid/Specificity", valid_scores["Specificity"], current_step)


        # save the network weights
        if (epoch + 1) % args.ckpt_every == 0:
            utils.checkpoint(net, f'lr{args.lr}_bs{args.batch_size}_epochs{args.n_epochs}_seed{seed}_epoch{epoch + 1}.pth', ckpt_save_path, device)

    # close writer of tensorboard
    writer.close()

    # training finished
    best_valid_result = best_valid_result_score
    print(f'Best valid score (best score): {best_valid_result_score}')
    print(f'Best valid score (best loss): {best_valid_result_loss}')
    print(f'Total time: {time.time() - start_time:.2f}s')

    # search the best threshold in validation set for classification
    if args.is_search:
        print(args.is_search)
        best_threshold = search_threshold(args, seed, loss_fn, ckpt_save_path, valid_data, device)
        results.append({'best_threshold': best_threshold})

    # save the results
    # format: 
    #   the last element is {'best_threshold': best threshold in validation set}
    #   the other elements are {'epoch': epoch, 'train_score': train scores list, 'valid_score': valid scores list}
    utils.save_results(args, results, f'lr{args.lr}_bs{args.batch_size}_epochs{args.n_epochs}_seed{seed}.csv', result_save_path)

    # visualize the results
    utils.visualize_results(results, best_valid_result)

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
    parser.add_argument('--transform_method_origin', type=int, default=1, 
                        help='transform method number for each image while loading data')
    parser.add_argument('--transform_method_epoch', type=int, default=2,
                        help='transform method number for each image while in each epoch')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--is_shuffle', type=str, choices=['0', '1'], default='1', help='shuffle the training/validation data partition')
    
    # optimizer parameters
    parser.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_epochs', nargs="+", type=int, default=[],
                        help="decay learning rate at these epochs")
    parser.add_argument('--lr_decay_values', nargs="+", type=float, default=[], 
                        help='modify learning rate by this value')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--betas', nargs="+", type=float, default=[0.9, 0.999], help='beta for adam optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd optimizer')

    # model parameters
    parser.add_argument('--out_dim', type=int, default=1, choices=[1, 2], help='output dimension')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold for classification')
    parser.add_argument('--model', type=str, default='ResNet34', help='model name')
    parser.add_argument('--is_search', type=str, choices=['0', '1'], default='0', help='whether to search the best threshold')

    # logging parameters
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='path to saved checkpoints')
    parser.add_argument('--result_dir', type=str, default='results', help='path to saved results')
    parser.add_argument('--ckpt_every', type=int, default=25, help='epochs between checkpoint save')
    parser.add_argument('--eval_every', type=int, default=5, help='epochs between evaluation on validation set')
    parser.add_argument('--print_every', type=int, default=2, help='batches between print in each epoch')

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = parser.parse_args()
    args.is_search = True if args.is_search == '1' else False
    args.is_shuffle = True if args.is_shuffle == '1' else False
    print("-------------------BEGIN-------------------")
    main(args)
