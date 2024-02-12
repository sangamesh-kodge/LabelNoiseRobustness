# Implementation for several label noise robust training approaches.

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

1. [Mixup](https://arxiv.org/pdf/1806.05236.pdf)- enhances model robustness by linearly interpolating between pairs of training examples and their corresponding labels. Specifically, it generates augmented training samples by blending two input samples and their labels. This process introduces beneficial noise during training, which helps the model learn more effectively even when the training data contains noisy labels.

2. [SAM (Sharpness-Aware Minimization)](https://arxiv.org/pdf/2010.01412.pdf)- Instead of solely minimizing the loss value, SAM aims to find a balance between low loss and smoothness. It encourages the model to explore regions with uniformly low loss, avoiding sharp spikes that might lead to overfitting. SAM exhibits remarkable resilience to noisy labels.



### Stay tuned for future updates
The following methods are planned to be implemented:

- [Sample Sieve](https://openreview.net/forum?id=2VXyy9mIyU3)
- [Label Smoothening](https://arxiv.org/pdf/2106.04149.pdf)
- [Early stopping](https://arxiv.org/abs/1903.11680)
- [Curvature Penalty](https://openreview.net/pdf?id=2B2xIJ299rx) 
- [MentorNet](https://arxiv.org/pdf/1712.05055.pdf)
- [MentorMix](https://arxiv.org/pdf/1911.09781.pdf) 


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
This section will be update soon with the results.

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