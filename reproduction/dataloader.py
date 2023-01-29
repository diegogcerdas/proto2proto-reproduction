from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


class CUBDataLoader(object):
    def __init__(
        self,
        train_dir: str = None,
        project_dir: str = None,
        test_dir: str = None,
        train_batch_size: int = None,
        test_batch_size: int = None,
        num_workers: int = 0,
        img_size=224,
        normalize_test=True,
    ):

        assert train_dir or project_dir or test_dir

        self.train_dir = train_dir
        self.project_dir = project_dir
        self.test_dir = test_dir
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.img_size = img_size

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        normalize = transforms.Normalize(mean=mean, std=std)
        transform = transforms.Compose(
            [
                transforms.Resize(size=(self.img_size, self.img_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )

        transform_push = transforms.Compose(
            [
                transforms.Resize(size=(self.img_size, self.img_size)),
                transforms.ToTensor(),
            ]
        )

        if self.train_dir is not None:
            assert self.train_batch_size is not None
            train_set = ImageFolder(self.train_dir, transform=transform)
            self.train_loader = DataLoader(
                train_set,
                batch_size=self.train_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
            self.classes = train_set.classes

        if self.project_dir is not None:
            assert self.train_batch_size is not None
            project_set = ImageFolder(self.project_dir, transform=transform_push)
            self.project_loader = DataLoader(
                project_set,
                batch_size=int(self.train_batch_size // 4),
                shuffle=False,
                num_workers=self.num_workers // 4,
            )
            self.classes = project_set.classes

        if self.test_dir is not None:
            assert self.test_batch_size is not None
            test_set = ImageFolder(self.test_dir, transform=(transform if normalize_test else transform_push))
            self.test_loader = DataLoader(
                test_set,
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            self.classes = test_set.classes

        for i in range(len(self.classes)):
            self.classes[i] = self.classes[i].split(".")[1]
