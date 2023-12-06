import os
import torch
from torch import nn

import transform
from Net import TestNet

net = TestNet
num_classes = 1
ckpt_path = "upload_dir/Net.pth"
transform_method_origin = 1
threshold = 0.5

class model:
    def __init__(self, device=torch.device("cpu")):
        self.checkpoint = ckpt_path
        self.device = device

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
        self.model = net(num_classes=num_classes)
        # join paths
        checkpoint_path = os.path.join(dir_path, self.checkpoint)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, input_image):
        """
        perform the prediction given an image.
        input_image is a ndarray read using cv2.imread(path_to_image, 1).
        note that the order of the three channels of the input_image read by cv2 is BGR.
        :param input_image: the input image to the model.
        :return: an int value indicating the class for the input image
        """
        # image transform
        image = transform.transform_method(method=transform_method_origin)(input_image)

        # image dimension expansion (do not change)
        image = image.unsqueeze(0)   # (3, x, x) -> (1, 3, x, x)

        # image to device
        image = image.to(self.device, torch.float)

        with torch.no_grad():
            score = self.model(image)

        if num_classes == 1:
            pr = torch.sigmoid(score).detach().cpu().item()
            pred_class = int(pr >= threshold)
        elif num_classes == 2:
            _, pred_class = torch.max(score, dim=1)
            pred_class = pred_class.detach().cpu()
            pred_class = int(pred_class)

        return pred_class
