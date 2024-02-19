# [Label Noise Robustness-PyTorch] Implementation for several training algorithms.

This repository implements different approaches for label noise robustness on the WebVision Dataset, a real-world noisy dataset. Additionally we also support adding synthetic noise to standard datasets.

## Setup the directory.

```bash
# Clone the repository
git clone https://github.com/sangamesh-kodge/LabelNoiseRobustness

# Change to the project directory
cd <path-to-cloned-LabelNoiseRobustness-directory>
```

## Dependency Installation
To set up the environment and install dependencies, follow these steps:
### Installation using conda
Install the packages either manually or use the environment.yml file with conda. 
- Installation using yml file
    ```bash
    conda env create -f environment.yml
    ```
    OR
- Manual Installation with conda environment 
    ```bash    
    ### Create Envirornment (Optional, but recommended)
        conda create --name label_noise python=3.11.4
        conda activate label_noise

        ### Install Packages
        pip install wandb 
        pip install argparse 
        pip install scikit-learn
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

<!-- ### Installation using pip [Unverfied]
You can install the required packages using pip. It is recommended to create a virtual environment first.

```bash
### Create Environment (Optional, but recommended)
python -m venv label_noise_env
source label_noise_env/bin/activate  # For Linux/macOS
.\label_noise_env\Scripts\activate  # For Windows

### Install Packages
pip install -r requirements.txt
``` -->

## Methods Implemented
1. Vanilla SGD - Standard Stochastic Gradient Descent algorithm. 
2. [Mixup](https://arxiv.org/pdf/1806.05236.pdf)- enhances model robustness by linearly interpolating between pairs of training examples and their corresponding labels. Specifically, it generates augmented training samples by blending two input samples and their labels. This process introduces beneficial noise during training, which helps the model learn more effectively even when the training data contains noisy labels. To use mixup add the cli argument ```--mixup-alpha <value-of-hyperparameter-alpha>```. For example, ```--mixup-alpha 0.2``` means the alpha hyperparameter is set to 0.1.

2. [SAM (Sharpness-Aware Minimization)](https://arxiv.org/pdf/2010.01412.pdf)- Instead of solely minimizing the loss value, SAM aims to find a balance between low loss and smoothness. It encourages the model to explore regions with uniformly low loss, avoiding sharp spikes that might lead to overfitting. SAM exhibits remarkable resilience to noisy labels. To use SAM add the cli argument ```--sam-rho <value-of-hyperparameter-rho>```. For example, ```--sam-rho 0.1``` means the rho hyperparameter is set to 0.1.


3. [Generalized Label Smoothening](https://arxiv.org/pdf/2106.04149.pdf)(NLS) - is a variant of Label Smoothing (LS) that uses a negative or positive weight to combine the hard and soft labels. It is designed to improve the robustness of the model when learning with noisy labels, especially in high noise regimes. To use NLS, you can add the cli argument ```--gls-smoothing <value-of-hyperparameter-smoothing-rate>```. For example, ```--gls-smoothing -0.2``` means using NLS with a weight of -0.2 for the soft labels.

4. [Early stopping](https://arxiv.org/abs/1903.11680) - is a regularization technique that stops the training of a neural network when the performance on a validation set stops improving or starts to deteriorate. It prevents overfitting by avoiding training the model for too many epochs, which can cause the model to memorize the training data and lose generalization ability. To use Early Stopping, you can add the cli argument ```--estop-delta <value-of-hyperparameter-min-delta>```. For example, ```--estop-delta 0.05``` means using min_delta  of 0.05.
5. [MentorNet](https://arxiv.org/pdf/1712.05055.pdf) - learns a data-driven curriculum dynamically with StudentNet. To use MentorNet, you can add the cli argument ```--mnet-gamma-p <value-of-hyperparameter-gamma-p>```. For example, ```--mnet-gamma-p 0.85``` means using gamma-p  of 0.85.

6. [MentorMix](https://arxiv.org/pdf/1911.09781.pdf) develops on the idea of MentorNet and Mixup. To use MentorMix, you can add the cli argument ```--mnet-gamma-p <value-of-hyperparameter-gamma-p> --mmix-alpha <value-of-hyperparameter-alpha >```. For example, ```--mnet-gamma-p 0.85 --mmix-alpha  0.2``` means using gamma-p  of 0.85 and alpha 0.2.
### Stay tuned for future updates
The following methods are planned to be implemented. 
- [Curvature Penalty](https://openreview.net/pdf?id=2B2xIJ299rx) 
- [Sample Sieve](https://openreview.net/forum?id=2VXyy9mIyU3)


## Supported Datasets
### Real-world noisy dataset
The real-world noisy dataset used in this project is the WebVision dataset, designed to facilitate research on learning visual representation from noisy web data. 

1. [WebVision 1.0](https://data.vision.ee.ethz.ch/cvl/webvision/dataset2017.html)- The WebVision dataset is designed to facilitate the research on learning visual representation from noisy web data. The dataset is extracted using code from [WebVision1.0](https://github.com/sangamesh-kodge/WebVision1.0).

2. [Mini-WebVision](https://arxiv.org/abs/1911.09781)- is a subset of the first 50 classes of Goole partition of WebVision 1.0 dataset (contains about 61234 training images). We use [Mini-WebVision](https://github.com/sangamesh-kodge/Mini-WebVision) code to extract dataset.

### Synthetic Noise in standard dataset. 
In addition to the real-world noisy dataset, synthetic noise is introduced into standard datasets for further analysis and evaluation of label noise robustness. Set the ```--percentage-mislabeled``` command line argument to desired level of label noise percentage for adding synthetic uniform noise to standard dataset. The following standard datasets are used with synthetic noise:
- [MNIST](https://ieeexplore.ieee.org/document/6296535)
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ImageNet](https://www.image-net.org/)


## Results
The results for Mini-WebVision dataset averaged over 3 randomly chosen seeds.
| Method          | Accuracy|
|---------------|-------|
| Vanilla SGD  | 63.81 $\pm$ 0.38|
| MixUp       | 65.01 $\pm$ 0.40 |
| MentorMix      | 65.35 $\pm$ 0.65 |
| SAM       | 65.68 $\pm$ 0.57 |


### License

This project is licensed under the [Apache 2.0 License](LICENSE).

## Citation.
If you find this repository helpful and use  it in your research, please consider citing our work using the following:

APA
```
Kodge, S. (2024). Implementation for several label noise robust training approaches. [Computer software]. https://github.com/sangamesh-kodge/LabelNoiseRobustness
```

Bibtex
```
@software{Kodge_Implementation_for_several_2024,
author = {Kodge, Sangamesh},
month = feb,
title = {{Implementation for several label noise robust training approaches.}},
url = {https://github.com/sangamesh-kodge/LabelNoiseRobustness},
year = {2024}
}
```