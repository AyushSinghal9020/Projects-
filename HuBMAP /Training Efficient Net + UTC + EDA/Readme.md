# HuBMAP : Training EfficientNet + UTC + EDA
This notebook provides an overview of the steps involved in training an EfficientNet model for the HuBMAP vascular segmentation challenge. The notebook also includes an exploratory data analysis (EDA) of the dataset.

# Prerequisites
The following are the prerequisites for running this notebook:

* Python $3.8+$
* PyTorch $1.10.0+$
* Torchvision $0.12.0+$
* Segmentation Models PyTorch $0.3.3+$
* timm $0.9.2+$

# Instructions
To run this notebook, clone the repository and install the dependencies:
```
git clone https://github.com/AyushSinghal9020/Projects-.git
cd Projects-
pip install -r requirements.txt
```
Then, open the notebook in Jupyter and follow the instructions.

# EDA
The EDA section of the notebook explores the following aspects of the dataset:

* The distribution of the images in the dataset
* The distribution of the labels in the dataset
* The relationship between the image size and the segmentation quality
# Training

The training section of the notebook trains an EfficientNet model on the HuBMAP dataset. The model is trained for $100$ epochs using the AdamW optimizer with a learning rate of $0.0001$.

# Results
The best model achieved a Dice score of $0.82$ on the validation set.

# Conclusion
This notebook provides an overview of the steps involved in training an EfficientNet model for the HuBMAP vascular segmentation challenge. The notebook also includes an EDA of the dataset.
