import torchvision.transforms as transforms
import cv2
import torch
import random

class transform_method:
    def __init__(self, method=1):
        self.transform1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.transform2 = transforms.Compose([
            transforms.RandomResizedCrop(224),
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
    
    """
    transform_method_origin:在加载数据时,所有数据均会先经过这一method
    transform_method_epoch:加载完毕数据之后,训练时的每一个epoch都会经过这一method

    也就是说,train_data会经过transform_method_origin,然后经过transform_method_epoch
    而valid_data只会经过transform_method_origin

    所以,transform_method_epoch一般用于实现数据增强,比如随机裁剪,随机旋转等
    而transform_method_origin一般用于实现数据预处理,比如resize,normalize等

    如果不需要数据增强,则在命令行参数中令 transform_method_epoch = 0 即可
    """

    def method_0(self, image):
        """Return the original image."""
        return image

    def method_1(self, image):
        # resize image
        image = cv2.resize(image, (512, 512)) # (800, 800, 3) -> (512, 512, 3)

        # convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).to(torch.float)    # (3, 512, 512)

        # normalize
        #image = image * 1.0 / 255

        return image
    
    def method_2(self, image):
        # input shape torch.tensor([3, x, x])
        assert image.shape[0] == 3, f'Input image must have 3 channels, but got {image.shape[0]} channels.'

        # torch.flip(image, [2]) 执行垂直翻转
        # torch.flip(image, [1]) 执行水平翻转
        # 即0.25的几率进行水平翻转,0.25的几率进行垂直翻转,0.5的几率不进行翻转
        if random.random() < 0.25:
            image = torch.flip(image, [1])
        elif random.random() < 0.5:
            image = torch.flip(image, [2])

        return image