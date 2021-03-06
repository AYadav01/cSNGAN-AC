# cSNGAN-AC
This repository is the official implementation of ***Towards Reproducible Radiomic Features by Mitigating CT Variability using a Conditional Generative Adversarial Network***.

## Requirements
* CUDA > 10.2, cuDNN > 7.6
* Python virualenv
  To install requirements:
  ```setup
  pip install -r requirements.txt
  ```
* Pull docker container 
  ```
  docker pull nvcr.io/nvidia/pytorch:19.11-py3
  ```
<!-- > 📋Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc... -->

## Training

To train the model(s) in the paper, run this command:

```train
python train.py -opt options/train/<config_name>.json
```

<!-- > 📋Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters. -->

## Evaluation
To evaluate the model in test set, run:
```eval
python test.py -opt options/test/<config_name>.json
```
Note that current training and testing data must be specified in h5 format.  

## Pre-trained Models
Pretrained models are stored in experiemnts\\<run_name\>\models\xxx.pth folder

<!-- ## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

> 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

> 📋Pick a licence and describe how to contribute to your code repository.  -->