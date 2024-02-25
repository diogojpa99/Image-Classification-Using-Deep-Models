# Image Classification Using Deep Models

## Overview

This repository hosts a comprehensive training pipeline for various deep learning models applied to image classification tasks. It includes support for a range of state-of-the-art Transformer and CNN-based models and is designed to investigate the effects of positional encoding on different data types. The goal is to discern the types of data that rely on spatial information and those where such information is less critical.

## Supported Models

The pipeline supports a palette of models, including but not limited to:
- ResNet variants (e.g., ResNet18, ResNet50)
- Vision Transformer (ViT) variants (e.g., ViT-B_16, ViT-small_patch16_224)
- DenseNet
- EfficientNet

These models are adaptable for both RGB and greyscale image classification tasks.

## Features

- **Positional Encoding Analysis:** The pipeline is set up to test the influence of positional encodings, providing insights into their importance for various types of image data.
- **Color Image Support:** The repository is equipped to handle standard RGB images for classification tasks.
- **Greyscale Image Adaptation:** For greyscale images, the pipeline can replicate the single channel three times to simulate an RGB image, allowing the use of pretrained feature extractors that expect three-channel input.

## Usage

1. **Clone the Repository**


## Usage

1. **Clone the Repository**
    ```bash
    git clone https://github.com/diogojpa99/Image-Classification-Using-Deep-Models.git
    ```

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare Your Data**
    - Use `data_setup.py` to organize your datasets for training, fine-tuning, or testing.

4. **Train or Fine-Tune the Models**
    - Run the main.py script with the desired arguments to train or fine-tune on your data:
    
    ```bash
    bash train.sh
    ```

## Contributing

Contributions to improve or expand the pipeline are welcome. Please fork the repository, commit your changes, and create a pull request with a clear explanation of your modifications or additions.

## Acknowledgments

This project is built upon the foundational work of the deep learning community. We express our gratitude to all the researchers and developers who have contributed to the open-source models and libraries utilized in this pipeline.

For more detailed information, please refer to the code comments and documentation provided within each script.
