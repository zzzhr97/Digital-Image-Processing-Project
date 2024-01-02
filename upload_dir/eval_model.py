import torch
from tqdm import tqdm

import data
import utils
import model

task = 1

class ARGS():
    def __init__(self):
        self.data_dir = '../dataset'
        self.task = task
        self.n_valid = 0
        self.is_shuffle = False
        self.k_fold = 0

def main(args):
    # print parameters
    print(f"Task: {args.task}")

    # load data
    print("Loading data...", end=' ')
    _, total_data, _ = data.load_data(args, 0, seed=123)
    print("Done")
    print("Number of samples:", len(total_data))

    # load model
    print("Loading model...", end=' ')
    eval_model = model.model(torch.device('cuda'))
    eval_model.load('.')
    print("Done")

    TP, TN, FP, FN = 0, 0, 0, 0

    # predict labels
    print("Begin predicting labels.")
    with tqdm(total=len(total_data), desc="Predicting", unit="sample") as pbar:
        for idx, sample in enumerate(total_data):
            image = sample['image']
            label = sample['label']
            name = sample['name']

            assert type(label) == int, f'Label type is {type(label)}'

            pred_class = eval_model.predict(image)

            TP += int(pred_class and label == 1)
            FP += int(pred_class and label == 0)
            TN += int(not pred_class and label == 0)
            FN += int(not pred_class and label == 1)

            pbar.update(1)
            pbar.set_postfix_str(f"Predicting {idx:03d}-th sample")

        pbar.set_description("Prediction done")
        pbar.set_postfix_str()
    
    # calculate scores
    scores = utils.cal_scores(None, TP, FP, TN, FN)

    # print results
    print("\n------------------ Results ------------------")
    print("\n\tTP/TN/FP/FN: {:03d}/{:03d}/{:03d}/{:03d}\n".format(TP, TN, FP, FN))
    print("\t[Kappa]      \t{:.8f}".format(scores['Kappa']))
    print("\t[F1]         \t{:.8f}".format(scores['F1']))
    print("\t[Specificity]\t{:.8f}".format(scores['Specificity']))
    print("\t[Average]    \t{:.8f}\n".format(scores['Average']))

if __name__ == '__main__':
    args = ARGS()
    main(args)