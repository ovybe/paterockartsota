# 🎨 Rock Art Classification Through Privacy-Guaranteed Ensemble Machine Learning

# 📁 Summary
This work currently presents an implementation of the Private Aggregation of Teacher Ensembles (PATE) framework and a similar alternative approach called Knowledge Distillation, using individually trained models for which the code used to train said models was also provided. Due to the lack of dataset availability for rock art, we aim to use PATE or KD's privacy guarantee to allow training using sensitive or unpublished rock art data. As a result, we apply Laplace Distribution over the dataset, load in the aggregate teacher models and use them to predict the noisy data for PATE. For KD, we apply noise to the logits extracted from the teacher models. The result is used to build the data we load to train the student model. 
To test performance, we provide the possibility of loading the aggregate teachers ensemble in various ways, whether homogeneously or featuring different models.
# 🛠️ Requirements
We created a conda environment for the PATE implementation of this repository, since it requires old versions of widely used libraries (e.g. numpy, torch, torchvision, syft, etc.):

* Python >=3.4
* PySyft v0.1.23a1 (Provided through a fork made by the author of this github repository: https://github.com/aristizabal95/Making-PATE-Bidirectionally-Private)
* PyTorch, which will be installed through the fork above as it is a dependency.
* Numpy, which will be installed through the fork above as it is a dependency.
* PIL
* tensorboard
* matplotlib
* Jupyter Notebook: 
```conda install jupyter notebook```


For the script used for training the individual models and for knowledge distillation, we used the current latest versions of pytorch and python:
* Python >=3.9
* Pytorch 2.5.1
* Numpy
* PIL
* matplotlib
* tensorboard
* opacus
* Jupyter Notebook: 
```conda install jupyter notebook```
# 📖 Running the scripts
To train the individual models, you should use the notebook [image_classification.ipynb](./implementations/image_classification.ipynb).

To train the student model through PATE, you should use the script [Project_PATE_ML.py](./implementations/Project_PATE_ML.py).

To train the student model through Knowledge Distillation, you should use the notebook [knowledge_distillation.ipynb](./implementations/knowledge_distillation.ipynb).

For the PATE implementation, knowledge distillation implementation and the script used for training the individual models:
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
   - knowledge_distillation.ipynb
- Only the folder names should match the folders the script is looking into (the name can be changed in the **dataset_name** variable).
- The train and val folders should have the same amount of classes named the same way.
- The images' names do not matter.
- You can change what model you want to use in the script, in the **model**/**student_model** variables
- The hyperparameters can easily be tuned by modifying the values stored in the variables in both scripts.

### 🟥 For the script used for training the individual models:
- Each model's run is logged in the **runs** folder located in the root through tensorboard.
- Tensorboard is used to visualize the runs. To do so, run the command ```tensorboard --logdir=runs``` through a terminal inside the root folder. Change the logdir variable to match the folder where the logs are stored.
- It can be accessed as an .ipynb and have each code block ran in order.

### 🟩 For the PATE implementation script:
- Each student model's run is logged in the **runs** folder located in the root through tensorboard.
- Tensorboard is used to visualize the runs. To do so, run the command ```tensorboard --logdir=runs``` through a terminal inside the root folder. Change the logdir variable to match the folder where the logs are stored.
- Set the dataset directory in the script using the variable `dataset_name`.
- Set the number of teachers using the variable `teachers_num`.
- Set the teacher directory using the variable `teacher_folder`.
- You set the teachers used in the teacher directory (the type used goes here, for example ['densenet201','densenet201','resnext'] for 3 teachers) using the variable `teacher_name`.
- You can also adjust any other script parameters, change the model type used by the student or teacher models.
- Simply run the py file through the terminal ```python Project_PATE_ML.py```.
- You will know the script ended when the script displays the training/validation graphs it generated after the run.

### 🟦 For the Knowledge Distillation implementation notebook:
- Each student model's run is logged in the **knowledge_distillation** folder located in the root through tensorboard.
- Tensorboard is used to visualize the runs. To do so, run the command ```tensorboard --logdir=knowledge_distillation``` through a terminal inside the root folder. Change the logdir variable to match the folder where the logs are stored.
- Set the dataset directory in the script using the variable `dataset_name`.
- Set the teacher directory using the variable `teacher_dir`.
- The script will try to load the teachers under ResNet18, DenseNet201 or ResNext64 if the `.pth` file contains these names.
- Everything else is similar to the PATE implementation script, you can change the model type used by the student/teacher models or adjust parameters if you so wish.
- Run the code blocks in order and wait for the student model to be trained for the amount of input epochs. You can also save the model afterwards.

# 📑 Output Results
The code is written so that it uses Tensorboard to log the results gained from training/evaluating each epoch. 
Thus, we have provided our logs in the [runs](./runs/) folder of this repository for the run logs of the individual models and the [student_runs](./student_runs/) folder for the run logs of the trained student models.
We also showcase our current output results that were used in the research paper as graphs in the [graphs](./graphs/) folder of the repo, as .png images.

# Semester 2 Week 1 Update
We appended new images procured from the generated data of a GAN to the first and the second teacher's datasets. We trained the teacher models ResNet18, ResNext64, DenseNet201 using the datasets with appended GAN images. We then trained and logged the PATE and Knowledge Distillation student models with a homogeneous ensemble consisting of the best-performing earlier trained teacher models to analyze further. We also trained a student model consisting of the new best-performing models for each dataset to compare with the earlier experiment's resulted best-performing model.
