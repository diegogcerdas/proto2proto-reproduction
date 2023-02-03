import torch
from reproduction.wrapper import PPNetWrapper
from reproduction.arguments import Arguments

def main():

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    list_of_model_arguments = [
        'arguments/vgg19_teacher.yaml',
        'arguments/vgg11_baseline.yaml',
        'arguments/vgg11_kd.yaml',
        'arguments/vgg16_baseline.yaml',
        'arguments/vgg16_kd.yaml',
    ]

    # For each of our models...
    for model_arg in list_of_model_arguments:
        model = init_model(model_arg, device)  # Initialize through argument file
        model.find_nearest_patches()  # Find nearest training patches

def init_model(args_filename, device):
    print(f"Initializing from {args_filename}.")
    args = Arguments(args_filename)
    model = PPNetWrapper(args, device)
    return model

if __name__ == '__main__':
    main()