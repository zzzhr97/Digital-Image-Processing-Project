import torchvision.transforms as transforms
import cv2
import torch

class transform_method:
    def __init__(self, method=1):
        self.transform1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.transform2 = transforms.Compose([transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.method = method

    def __call__(self, image):
        """Return the transformation applied on the input image."""

        # Example: 
        # self.method = 1, then return self.method_1(image)
        return getattr(self, f'method_{self.method}')(image)

    def method_1(self, image):
        # resize image
        image = cv2.resize(image, (512, 512))

        # convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1)    # (3, 512, 512)

        # normalize
        #image = image / 255.0

        return image