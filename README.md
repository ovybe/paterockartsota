# Rock Art Classification Through Privacy-Guaranteed Ensemble Machine Learning

# Summary
This work currently presents an implementation of the Private Aggregation of Teacher Ensembles (PATE) framework, using individually trained models for which the code used to train said models was also provided. Due to the lack of dataset availability for rock art, we aim to use PATE's privacy guarantee to allow training using sensitive or unpublished rock art data. As a result, we apply Laplace Distribution over the dataset, load in the aggregate teacher models and use them to predict the noisy data. The result is used to build the data we load to train the student model. 
To test performance, we provide the possibility of loading the aggregate teachers ensemble homogeneously.
# Requirements
We created a conda environment for the PATE implementation of this repository, since it requires old versions of widely used libraries (e.g. numpy, torch, torchvision, syft, etc.):

* Python >=3.4
* PySyft v0.1.23a1 (Provided through a fork made by the author of this github repository: https://github.com/aristizabal95/Making-PATE-Bidirectionally-Private)
* PyTorch, installed through the fork above as it is a dependency.
* Jupyter Notebook: 
```conda install jupyter notebook```
# Running the training scripts
To train the individual models, you should use the notebook [image_classification.ipynb](https://github.com/ovybe/paterockartsota/blob/main/implementations/Project_PATE_ML.py) <br />
To train the student model through PATE, you should use the script [Project_PATE_ML.py](https://github.com/ovybe/paterockartsota/blob/main/implementations/image_classification.ipynb)
# Output Results
Our current output results have been provided as graphs in the repo, under the names:
* DENSENET101_100e_homogenous_dataset.png
* RESNET18_100e_homogenous_dataset.png
* RESNEXT64_100e_homogenous_dataset.png
