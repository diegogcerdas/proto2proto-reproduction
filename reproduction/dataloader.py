from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from .arguments import Arguments


class CUBDataLoader(object):
    def __init__(self, args: Arguments) -> None:

        self.test_dir = args.dataLoader.testDir
        self.test_batch_size = args.dataLoader.testBatchSize
        self.num_workers = args.dataLoader.numWorkers
        self.img_size = args.imgSize

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        transform = transforms.Compose(
            [
                transforms.Resize(size=(self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        transform_push = transforms.Compose(
            [
                transforms.Resize(size=(self.img_size, self.img_size)),
                transforms.ToTensor(),
            ]
        )

        test_set_norm = ImageFolder(self.test_dir, transform=(transform))
        self.test_loader_norm = DataLoader(
            test_set_norm,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        test_set = ImageFolder(self.test_dir, transform=(transform_push))
        self.test_loader = DataLoader(
            test_set,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        self.classes = test_set.classes

        for i in range(len(self.classes)):
            self.classes[i] = self.classes[i].split(".")[1]
