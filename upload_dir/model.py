import os
import torch
from torch import nn

import transform, transform1, transform2, transform3, transform4, transform5, transform6
import Net, Net1, Net2, Net3, Net4, Net5, Net6

# voting or not
is_vote = True

# not voting
net = Net.ResNet101
ckpt_path = "Net.pth"

nets = []
ckpt_paths = []
transforms = []

# voting
nets = [Net1.DenseNet121, Net2.DenseNet121, Net3.ResNet101, Net4.ResNet101, Net5.ResNet101, Net6.ResNet101]
ckpt_paths = ["Net1.pth", "Net2.pth", "Net3.pth", "Net4.pth", "Net5.pth", "Net6.pth"]
transforms = [transform1, transform2, transform3, transform4, transform5, transform6]

nets = [Net1.DenseNet121, Net2.DenseNet121, Net3.ResNet101, Net4.ResNet101]
ckpt_paths = ["Net1.pth", "Net2.pth", "Net3.pth", "Net4.pth"]
transforms = [transform1, transform2, transform3, transform4]

num_classes = 2
in_channel = 3

transform_method_origin = 1
threshold = 0.5

pretrained = False

class model:
    def __init__(self, device=torch.device("cpu")):
        self.checkpoint = ckpt_path
        self.checkpoints = ckpt_paths
        self.device = device
        self.models = []

    def get_vote_models(self, dir_path):
        assert len(nets) == len(self.checkpoints)
        for idx, vote_net in enumerate(nets):
            self.models.append(vote_net(num_classes=num_classes, in_channel=in_channel, pretrained=False))
            
            checkpoint_path = os.path.join(dir_path, self.checkpoints[idx])
            self.models[idx].load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            self.models[idx].to(self.device)
            self.models[idx].eval()

    def get_single_model(self, dir_path):
        self.model = net(num_classes=num_classes, in_channel=in_channel, pretrained=False)
        
        checkpoint_path = os.path.join(dir_path, self.checkpoint)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def load(self, dir_path):
        """
        load the model and weights.
        dir_path is a string for internal use only - do not remove it.
        all other paths should only contain the file name, these paths must be
        concatenated with dir_path, for example: os.path.join(dir_path, filename).
        make sure these files are in the same directory as the model.py file.
        :param dir_path: path to the submission directory (for internal use only).
        :return:
        """
        if is_vote:
            self.get_vote_models(dir_path)
        else:
            self.get_single_model(dir_path)

    def get_pred_class(self, score):
        """Get prediction given score"""
        if num_classes == 1:
            pr = torch.sigmoid(score).detach().cpu().item()
            pred_class = int(pr >= threshold)
        elif num_classes == 2:
            _, pred_class = torch.max(score, dim=1)
            pred_class = pred_class.detach().cpu()
            pred_class = int(pred_class)

        return pred_class
    
    def pre_process(self, input_image, seq=-1):
        """Pre process image"""
        # image transform
        if seq == -1:
            image = transform.transform_method(method=transform_method_origin)(input_image)
        else:
            tr = transforms[seq]
            image = tr.transform_method(method=transform_method_origin)(input_image)

        image = image.unsqueeze(0)   # (3, x, x) -> (1, 3, x, x)
        image = image.to(self.device, torch.float)
        return image

    def single_predict(self, image, seq=-1):
        """Get prediction from a model"""
        cur_net = self.model if seq == -1 else self.models[seq]
        with torch.no_grad():
            score = cur_net(image)
        return self.get_pred_class(score)

    def predict(self, input_image):
        """
        perform the prediction given an image.
        input_image is a ndarray read using cv2.imread(path_to_image, 1).
        note that the order of the three channels of the input_image read by cv2 is BGR.
        :param input_image: the input image to the model.
        :return: an int value indicating the class for the input image
        """
        if not is_vote:
            image = self.pre_process(input_image)
            pred_class = self.single_predict(image, seq=-1)

        else:
            pred_classes = []
            for idx in range(len(self.models)):
                image = self.pre_process(input_image, seq=idx)
                pred_classes.append(self.single_predict(image, seq=idx))
            pred_class = max(set(pred_classes), key=pred_classes.count)

        return pred_class
