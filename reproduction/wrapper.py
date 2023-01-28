import torch
import pytorch_lightning as pl
from lib.protopnet.model import PPNet
from lib.features import init_backbone
from lib.utils import evaluate

class PPNetWrapper():

    def __init__(
        self,
        backbone,
        dataloader,
        checkpoint_path,
        args
    ):
        self.backbone = backbone
        self.dataloader = dataloader
        self.args = args
        self.model = self.load_model(checkpoint_path)

    def load_model(self, checkpoint_path):
        features, _ = init_backbone(self.backbone)
        model = PPNet(
            num_classes=len(self.dataloader.classes), 
            feature_net=features, 
            args=self.args
        )
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)
        return model

    def evaluate_accuracy(self, x):
        accuracy = evaluate.evaluate_model(
            self.model, 
            self.dataloader.test_loader,
            mgpus=False, 
            num_classes=len(self.dataloader.classes)
        )
        return accuracy
