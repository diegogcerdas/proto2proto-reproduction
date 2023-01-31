from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from .arguments import Arguments


class CUBDataLoader(object):
    def __init__(self, args: Arguments) -> None:

        self.test_dir = args.dataLoader.testDir
        self.project_dir  = args.dataLoader.projectDir
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

        test_set = ImageFolder(self.test_dir, transform=transform)
        self.test_loader = DataLoader(
            test_set,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )

        project_set = ImageFolder(self.project_dir, transform=transform_push)
        self.project_loader = DataLoader(
            project_set,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )

        self.classes = test_set.classes

        for i in range(len(self.classes)):
            self.classes[i] = self.classes[i].split(".")[1]
