import torch, os, yaml
import numpy as np
from reproduction.wrapper import PPNetWrapper
from reproduction.arguments import Arguments
from reproduction.lib.protopnet.helpers import makedir

def main():

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    # VGG19 -> VGG 11
    teacher = init_model('arguments/vgg19_teacher.yaml', device)
    baseline_student = init_model('arguments/vgg11_baseline.yaml', device)
    kd_student = init_model('arguments/vgg11_kd.yaml', device)
    run_experiment('vgg19_11', teacher, baseline_student, kd_student)

    # VGG19 -> VGG 16
    baseline_student = init_model('arguments/vgg16_baseline.yaml', device)
    kd_student = init_model('arguments/vgg16_kd.yaml', device)
    run_experiment('vgg19_16', teacher, baseline_student, kd_student)

def init_model(args_filename, device):
    print(f"Initializing from {args_filename}.")
    args = Arguments(args_filename)
    model = PPNetWrapper(args, device)
    model.compute_indices_scores()
    model.find_nearest_patches()
    return model

def run_experiment(experiment_name, teacher, baseline_student, kd_student):
    print(f'Running experiment: {experiment_name}')
    for dist_threshold in [0.01, 0.1, 0.2, 0.45, 1.0, 3.0, 5.0, None]:
        results = {
            'aap': {
                'teacher': teacher.compute_aap(dist_threshold),
                'baseline_student': baseline_student.compute_aap(dist_threshold),
                'kd_student': kd_student.compute_aap(dist_threshold)
            },
            'ajs': {
                'baseline_student': baseline_student.compute_ajs(dist_threshold, teacher.indices_scores),
                'kd_student': kd_student.compute_ajs(dist_threshold, teacher.indices_scores)
            },
            'accuracy': {
                'teacher': teacher.compute_accuracy(),
                'baseline_student': baseline_student.compute_accuracy(),
                'kd_student': kd_student.compute_accuracy()
            },
            'pms': {}
        }
        dir = os.path.join('results', experiment_name, str(dist_threshold))
        makedir(dir)
        pms, best_allocation = baseline_student.compute_pms(dist_threshold, teacher.indices_scores)
        results['pms']['baseline_student'] = pms
        np.save(os.path.join(dir, 'best_allocation_baseline.npy'), best_allocation)
        pms, best_allocation = kd_student.compute_pms(dist_threshold, teacher.indices_scores)
        results['pms']['kd_student'] = pms
        np.save(os.path.join(dir, 'best_allocation_kd.npy'), best_allocation)
        with open(os.path.join(dir, 'metrics.yaml'), 'w') as file:
            yaml.dump(results, file)
    print(f'Finished experiment: {experiment_name}')

if __name__ == '__main__':
    main()
