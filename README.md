# On the reproducibility of ”Proto2Proto: Can you recognize the car, the way I do?”

This repository concerns a reproducibility study of the paper: [_Proto2Proto: Can You Recognize the Car, the Way I Do?_ ](https://openaccess.thecvf.com/content/CVPR2022/html/Keswani_Proto2Proto_Can_You_Recognize_the_Car_the_Way_I_Do_CVPR_2022_paper.html) by Keswani et al. (2022). It builds upon the authors' [original implementation](https://github.com/archmaester/proto2proto) and provides wrapping code to run evaluation in an easy and efficient manner.

This project was developed by Diego Garcia Cerdas, Rens Kierkels, Thomas Jurriaans, and Sergei Agaronian as part of the FACT-AI course at the University of Amsterdam.

## Creating Conda Environment
    conda env create -f environment.yml python=3.6
    conda activate proto2proto

## Setting Up Dataset and Required Code
Running the following command ensures that the CUB-200-2011 dataset is downloaded into `./datasets/CUB_200_2011/`, and the required code (a folder called `lib`) is cloned into `./reproduction/` setup:

    bash setup_reproduction.sh

Please make sure to run the above command before running other scripts or notebooks.

## Model Weights and Results from our Reproducibility Study

We provide the model weights of the networks used in our study through [this storage](https://drive.google.com/drive/folders/1ZgEKQe9tX6loGBip4TQ1HIRK45SqWMBd?usp=sharing), in `checkpoints.zip`. You can also download `results.zip`, containing the metrics for our experiments and prototype matches between teacher and student, and `nearest.zip` containing the nearest training patches for each model's prototypes.

- Please unzip these files into the current directory as `./checkpoints/`, `./results/`, and `./nearest/` respectively.

### Example notebook

To reproduce the figures from our study, we provide a Jupyter notebook `example.ipynb`. This notebook further explains the structure of the folders setup above.

## Running Evaluation from Scratch

If you wish to perform evaluation from scratch, please download the provided checkpoints and run:

    # For interpretability metrics, accuracy, and prototype-matching
    python evaluation.py  

    # For finding nearest training patches for each model's prototypes
    python find_nearest.py  

- We provide all arguments needed for evaluation thorugh the YAML files in `./arguments/`.

- To perform evaluation on additional ProtoPNet models, simply create new argument files and modify the `main` method in the above scripts to point to your models' arguments.

## Aknowledgement

Our code is built on top of [Proto2Proto](https://github.com/archmaester/proto2proto), [ProtoPNet](https://github.com/cfchen-duke/ProtoPNet), and [ProtoTree](https://github.com/M-Nauta/ProtoTree).
