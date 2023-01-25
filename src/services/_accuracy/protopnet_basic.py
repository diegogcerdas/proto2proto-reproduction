import torch
from src.mgr import manager
from lib import init_proto_model
from lib.utils import evaluate


class Service(object):

    def __init__(self, dataset_loader):

        self.manager = manager
        self.mgpus = self.manager.common.mgpus
        self.dataset_loader = dataset_loader

        self.teacher_model, _, _ = init_proto_model(
            manager, dataset_loader.classes, manager.settingsConfig.Teacherbackbone)

        self.student_kd_model, _, _ = init_proto_model(
            manager, dataset_loader.classes, manager.settingsConfig.StudentKDbackbone)

        self.student_baseline_model, _, _ = init_proto_model(
            manager, dataset_loader.classes, manager.settingsConfig.StudentBaselinebackbone)

    def evaluate(self):

        result_teacher = evaluate.evaluate_model(self.teacher_model, self.dataset_loader.test_loader,
                                         mgpus=self.mgpus, num_classes=200)
        result_baseline = evaluate.evaluate_model(self.student_baseline_model, self.dataset_loader.test_loader,
                                         mgpus=self.mgpus)
        result_kd = evaluate.evaluate_model(self.student_kd_model, self.dataset_loader.test_loader,
                                         mgpus=self.mgpus)

        return result_teacher, result_baseline, result_kd

    def __call__(self):

        self.teacher_model.eval()
        self.student_kd_model.eval()
        self.student_baseline_model.eval()

        if self.mgpus:
            # Optimize class distributions in leafs
            self.eye = torch.eye(self.student_kd_model.module._num_classes)
        else:
            self.eye = torch.eye(self.student_kd_model._num_classes)

        result_teacher, result_baseline, result_kd = self.evaluate()
        print('Teacher ', result_teacher)
        print('Baseline ', result_baseline)
        print('KD ', result_kd)




