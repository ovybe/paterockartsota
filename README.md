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
- One of the code blocks is used to generate the older teacher models to be able to be used by PATE. The code block needs to be modified and ruled for every teacher model to be used. This will save them with their name with the addition of '_old' in the same folder they're in.
- There is also a code block used to run the model and predict an image, which can be modified accordingly to test the trained individual models.

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
- The script will try to load the teachers under ResNet18, DenseNet201 or ResNeXt101_64X4D if the `.pth` file contains these names.
- Everything else is similar to the PATE implementation script, you can change the model type used by the student/teacher models or adjust parameters if you so wish.
- Run the code blocks in order and wait for the student model to be trained for the amount of input epochs. You can also save the model afterwards.

# 📑 Output Results
The code is written so that it uses Tensorboard to log the results gained from training/evaluating each epoch. 
Thus, we have provided our logs in the [runs](./runs/) folder of this repository for the run logs of the individual models and the [student_runs](./student_runs/) folder for the run logs of the trained student models.
We also showcase our current output results that were used in the research paper as graphs in the [graphs](./graphs/) folder of the repo, as .png images.

# Semester 2 Week 1 Update
We append new images procured from the generated data of a GAN using the first and the second teacher's datasets to said datasets in hopes of improving model performance. We train the teacher models ResNet18, ResNeXt101_64X4D, DenseNet201 using the newly formed datasets with appended GAN images. Pick template for paper depending on where I'll apply.
# Semester 2 Week 2 Update
We train and log the PATE and Knowledge Distillation student models with a homogeneous ensemble consisting of the best-performing earlier trained teacher models to analyze further. We also train a student model consisting of the new best-performing models for each dataset to compare with the earlier experiment's resulted best-performing model.
# Semester 2 Week 3 Update
Present work done so far to my advisor. Work on solving any encountered issues and focus on given suggestions.
# Semester 2 Week 4 Update
Analyze achieved results and continue working on SOTA, perform necessary adjustments and work on presentation.
# Semester 2 Week 5 Update
Check-up on SOTA with advisor, work on any suggestions and continue work on SOTA.
# Semester 2 Week 6 Update
Analyze overall recall results and mark possible labels for removal.
# Semester 2 Week 7 Update
Check-up on work with advisor, run marked labels by him for advice. Proceed with removing marked label.
# Semester 2 Week 8 Update
Start training and logging of each teacher model with each dataset without the market labels.
# Semester 2 Week 9 Update
Continue previous week's work and check-up with advisor for further suggestions to follow.
# Semester 2 Week 10 Update
Work on research paper, add the new results and finish up.
# Semester 2 Week 11 Update
Run paper by advisor, modify/implement based on suggestions.
# Semester 2 Weeks 12-14 Update
Weekly meetings with advisor, proofreading and working on SOTA and presentation.

