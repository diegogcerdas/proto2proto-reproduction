import torch
from .lib.protopnet.model import PPNet
from .lib.features import init_backbone
from .lib.utils.evaluate import acc_from_cm
import numpy as np
from tqdm import tqdm

class PPNetWrapper():

    def __init__(
        self,
        args,
        dataloader,
        device
    ):
        self.args = args
        self.dataloader = dataloader
        self.device = device
        self.model = self.load_model().to(device)

    def load_model(self):
        features, _ = init_backbone(self.args.backbone)
        model = PPNet(
            num_classes=len(self.dataloader.classes), 
            feature_net=features, 
            args=self.args
        )
        state_dict = torch.load(self.args.backbone.loadPath)
        model.load_state_dict(state_dict)
        return model

    def evaluate_accuracy(self):
        num_classes = len(self.dataloader.classes)
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        self.model.eval()
        data_iter = iter(self.dataloader.test_loader)

        for (xs, ys) in tqdm(data_iter):
            with torch.no_grad():
                ys = ys.to(self.device)
                xs = xs.to(self.device)
                ys_pred, _ = self.model.forward(xs)

            ys_pred = torch.argmax(ys_pred, dim=1)

            for y_pred, y_true in zip(ys_pred, ys):
                confusion_matrix[y_true][y_pred] += 1

        acc = acc_from_cm(confusion_matrix)

        return acc
