![Visitors](https://badges.pufler.dev/visits/VinayHajare/EfficientCLIP-GAN) 
[![License: GNU GPL v2.0](https://img.shields.io/badge/License-GPL_v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg)
![Packagist](https://img.shields.io/badge/Pytorch-1.9.0-red.svg)
![hardware](https://img.shields.io/badge/GPU-CPU-1abc9c.svg)
![Last Commit](https://img.shields.io/github/last-commit/VinayHajare/EfficientCLIP-GAN)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-blue.svg)]((https://github.com/VinayHajare/EfficientCLIP-GAN/graphs/commit-activity))
![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1a009c.svg)
[![Updated](https://badges.pufler.dev/updated/VinayHajare/EfficientCLIP-GAN)](https://vinayhajare.engineer) 
# EfficientCLIP-GAN: High-Speed Image Generation with Compact CLIP-GAN Architecture

<p align="center">
    <img src="Logo.png" width="500px"/>
</p>

# A high-quality, fast, and efficient text-to-image synthesis model


<p align="center">
<b>Generated Images
</p>
<p align="center">
    <img src="Samples.png"/>
</p>


## Requirements
- python 3.9
- Pytorch 1.9
- At least 1xTesla v100 32GB GPU (for training)
- Only CPU (for inference) 


EfficientCLIP-GAN is a small, rapid and efficient generative model which can generate multiple pictures in one second even on the CPU as compared to Diffusion Models.
## Installation

Clone this repo.
```
git clone https://github.com/VinayHajare/EfficientCLIP-GAN
pip install -r requirements.txt
```
Install [CLIP](https://github.com/openai/CLIP)


## Preparation
### Datasets
1. Download the preprocessed metadata for [birds](https://drive.google.com/file/d/1HG7M80UNo37xOxJlhY3d_uO-pXj6GRo_/view?usp=sharing)  and extract them to `data/`
2. Download the [birds](https://www.vision.caltech.edu/datasets/cub_200_2011/) image data. Extract them to `data/birds/`

  ***OR***
1. Download the preprocessed metadata and CUB dataset in a single zip [download](https://drive.google.com/drive/folders/1DLIf_iMvq_qLRn8881WH6KXKHlS_KH5V?usp=sharing) it and extract to `data/`

## Training
  ```
  cd EfficientCLIP-GAN/code/
  ```
### Train the EfficientCLIP-GAN model
  - For bird dataset: `bash scripts/train.sh ./cfg/bird.yml`

### Resume training process
If your training process is interrupted unexpectedly, set **state_epoch**, **log_dir**, and **pretrained_model_path** in train.sh with appropriate values to resume training.

### TensorBoard
Our code supports automate FID evaluation during training, the results are stored in TensorBoard files under ./logs. You can change the test interval by changing **test_interval** in the YAML file.

  - For bird dataset: `tensorboard --logdir=./code/logs/bird/train --port 8166`


## Evaluation

### Download Pretrained Model
- [EfficientCLIP-GAN for Birds](https://huggingface.co/VinayHajare/EfficientCLIP-GAN). Download and save it to `./code/saved_models/pretrained/`

### Evaluate EfficientCLIP-GAN model

  ```
  cd EfficientCLIP-GAN/code/
  ```
set **pretrained_model** in test.sh to models path
- For bird dataset: `bash scripts/test.sh ./cfg/bird.yml`


### Performance
The released model achieves better performance than the Latent Diffusion.


| Model            | Birds-FID↓ | Birds-CS↑  |
| ---------------- | ---------- | ---------- | 
| EfficientCLIP-GAN| 11.806      | 31.70      |


## Try Now  
The gradio demo is available as a hosted HuggingFace Space [here](https://huggingface.co/spaces/VinayHajare/Text-To-Image-EfficientCLIP-GAN).  
You can run this app locally  
```
cd EfficientCLIP-GAN/gradio app
pip install -r requirements.txt
```
then 
```
python app.py
```


## Note :  
***Weights are available on [HuggingFace Hub](https://huggingface.co/VinayHajare/EfficientCLIP-GAN)***  



## Inference (Sampling)
  
### Synthesize images from your text descriptions/Prompts 
  - the inference.ipynb can be used to sample

---
### Support EfficientCLIP-GAN

If you find this useful in your research, please consider giving a star to repository

The code is released for academic research use only. For commercial use, please contact [Vinay Hajare](https://vinayhajare.engineer).  

### Contributors
[![Contributors Display](https://badges.pufler.dev/contributors/VinayHajare/EfficientCLIP-GAN?size=50&padding=5&perRow=10&bots=false)]()
