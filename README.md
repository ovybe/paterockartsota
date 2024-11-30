# Rock Art Classification Through Privacy-Guaranteed Ensemble Machine Learning

# Summary
This work currently presents an implementation of the Private Aggregation of Teacher Ensembles (PATE) framework, using individually trained models for which the code used to train said models was also provided. Due to the lack of dataset availability for rock art, we aim to use PATE's privacy guarantee to allow training using sensitive or unpublished rock art data. As a result, we apply Laplace Distribution over the dataset, load in the aggregate teacher models and use them to predict the noisy data. The result is used to build the data we load to train the student model. 
To test performance, we provide the possibility of loading the aggregate teachers ensemble homogeneously.
# Requirements
We created a conda environment for the PATE implementation of this repository, since it requires old versions of widely used libraries (e.g. numpy, torch, torchvision, syft, etc.):

* Python >=3.4
* PySyft v0.1.23a1 (Provided through a fork made by the author of this github repository: https://github.com/aristizabal95/Making-PATE-Bidirectionally-Private)
* PyTorch, which will be installed through the fork above as it is a dependency.
* Numpy, which will be installed through the fork above as it is a dependency.
* PIL
* matplotlib
* Jupyter Notebook: 
```conda install jupyter notebook```


For the script used for training the individual models, we used the current latest versions of pytorch and python:
* Python >=3.9
* Pytorch 2.5.1
* Numpy
* PIL
* matplotlib
* Jupyter Notebook: 
```conda install jupyter notebook```
# Running the scripts
To train the individual models, you should use the notebook [image_classification.ipynb](https://github.com/ovybe/paterockartsota/blob/main/implementations/Project_PATE_ML.py)

To train the student model through PATE, you should use the script [Project_PATE_ML.py](https://github.com/ovybe/paterockartsota/blob/main/implementations/image_classification.ipynb)

For both the PATE implementation and the script used for training the individual models:
- The folder holding the dataset must be in the root directory. The structure should look like this:
 - Root
   - dataset
     - train
       - example_class1
         - img1.jpg
         - img2.jpg
         - ...
       - example_class2
         - imgX.jpg
         - ... 
     -  val
        - example_class1
           - img1.jpg
           - ...
   - Project_PATE_ML.py
   - image_classification.ipynb
- Only the folder names should match the folders the script is looking into (the name can be changed in the **dataset_name** variable).
- The train and val folders should have the same amount of classes named the same way.
- The images' names do not matter.
- You can change what model you want to use in the script, in the **model**/**student_model** variables
- The hyperparameters can easily be tuned by modifying the values stored in the variables in both scripts.

For the script used for training the individual models:
- Each model's run is logged in the **runs** folder located in the root through tensorboard.
- Tensorboard is used to visualize the runs. To do so, run the command ```tensorboard --logdir=runs``` through a terminal inside the root folder. Change the logdir variable to match the folder where the logs are stored.
- It can be accessed as an .ipynb and have each code block ran in order.

For the PATE implementation script:
- Simply run the py file through the terminal ```python Project_PATE_ML.py```.
- You will know the script ended when the script displays the training/validation graphs it generated after the run.

# Output Results
Our current output results have been provided as graphs in the repo, under the names:
* DENSENET101_100e_homogenous_dataset.png
* RESNET18_100e_homogenous_dataset.png
* RESNEXT64_100e_homogenous_dataset.png
