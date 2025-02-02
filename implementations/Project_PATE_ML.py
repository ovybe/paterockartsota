#Importing  Libraries
import torch
from torchvision import  transforms, datasets, models
from torch.utils.data import Subset , Dataset, ConcatDataset
from torch.distributions import Laplace
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import torch.optim as optim
from syft.frameworks.torch.differential_privacy import pate
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score
#from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from torchvision.models.resnet import ResNet, Bottleneck
from torch.utils.tensorboard import SummaryWriter

def resnext101_64x4d(pretrained=False, **kwargs):
    """Constructs a ResNeXt-101 64x4d model."""
    model = ResNet(Bottleneck, [3, 4, 23, 3], groups=64, width_per_group=4, **kwargs)
    if pretrained:
        # state_dict = torch.utils.model_zoo.load_url(
        #     'https://download.pytorch.org/models/resnext101_64x4d-173b62eb.pth', 
        #     model_dir="./"
        # )
        # Load the TorchScript model
        state_dict = torch.load('resnext101_64x4d-173b62eb_old.pth')
        # Extract the state_dict from the ScriptModule
        model.load_state_dict(state_dict)
    return model

# Define a custom Laplace noise transform
class AddLaplaceNoise:
    def __init__(self, mean=0.0, scale=1.0):
        self.laplace_dist = Laplace(mean, scale)
    
    def __call__(self, img):
        # Convert the image to a tensor if not already done
        img_tensor = transforms.ToTensor()(img)  # [C, H, W], range [0, 1]
        
        # Generate Laplace noise of the same shape
        noise = self.laplace_dist.sample(img_tensor.shape)
        
        # Add noise and clamp to valid range [0, 1]
        noisy_img = img_tensor + noise
        noisy_img = torch.clamp(noisy_img, 0.0, 1.0)
        
        return noisy_img

# Define data transformations for data augmentation and normalization
train_transforms = [
        transforms.Resize(size=(180,180)),
        transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5)
        # transforms.RandomRotation(degrees=60),
        #transforms.RandomGrayscale(),
        #transforms.RandomHorizontalFlip(),
    ]
grayscale_transforms = train_transforms.copy()
grayscale_transforms.append(transforms.Grayscale(num_output_channels=3))
train_transforms_end = [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
data_transforms = {
    'train': transforms.Compose(train_transforms+train_transforms_end),
    'val': transforms.Compose([
        transforms.Resize(size=(180,180)),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
data_transforms_flipped = {
    'train': transforms.Compose(train_transforms+[transforms.RandomHorizontalFlip(p=1)]+train_transforms_end),
    'val': transforms.Compose([
        transforms.Resize(size=(180,180)),
        transforms.RandomHorizontalFlip(p=1),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Define the data directory
ROOT_DIR = os.path.abspath(os.curdir)
print(ROOT_DIR)
dataset_name='dataset'
data_dir = os.path.join(ROOT_DIR, dataset_name)
print(data_dir)

concatdatasets = []
concatdatasets_val = []

# Create data loaders
#og_image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transforms.Compose(train_transforms_end)) for x in ['train']}
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
tr_flipped_dataset = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms_flipped[x]) for x in ['train','val']}
tr_grayscale_dataset = {x: datasets.ImageFolder(os.path.join(data_dir, x), transforms.Compose(grayscale_transforms+train_transforms_end)) for x in ['train','val']}
grayscale_transforms.append(transforms.RandomHorizontalFlip(p=1))
tr_grayscale_flipped_dataset = {x: datasets.ImageFolder(os.path.join(data_dir, x), transforms.Compose(grayscale_transforms+train_transforms_end)) for x in ['train','val']}

#concatdatasets.append(og_image_datasets['train'])
concatdatasets.append(image_datasets['train'])
concatdatasets.append(tr_flipped_dataset['train'])
concatdatasets.append(tr_grayscale_dataset['train'])
concatdatasets.append(tr_grayscale_flipped_dataset['train'])

concatdatasets_val.append(image_datasets['val'])
concatdatasets.append(tr_flipped_dataset['val'])
concatdatasets_val.append(tr_grayscale_dataset['val'])
concatdatasets_val.append(tr_grayscale_flipped_dataset['val'])

r_times = 5;
rotate_transf = train_transforms
tr_rotate = []
print(rotate_transf)
for i in range(5):
    rotate_transf = train_transforms.copy()
    for j in range(i):
         rotate_transf.append(transforms.RandomRotation(degrees=(60,60)))
    concatdatasets.append(datasets.ImageFolder(os.path.join(data_dir, 'train'), transforms.Compose(rotate_transf+train_transforms_end)))
    concatdatasets_val.append(datasets.ImageFolder(os.path.join(data_dir, 'val'), transforms.Compose(rotate_transf+train_transforms_end)))

image_datasets['train'] = ConcatDataset(concatdatasets)
image_datasets['val'] = ConcatDataset(concatdatasets_val)
print(len(image_datasets['train']))
print(len(image_datasets['val']))


# Create the dataset and apply transform
#dataset = Dataset(root_dir=ROOT_DIR, dataset_folder=dataset_name, class_names=['Circle','Cross','Goat','Person','Spiral','Stag','Zigzag'], transform=transform)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print(dataset_sizes)

class_names = image_datasets['train'].datasets[0].classes
print(class_names)

## Define output directory
#output_dir = os.path.join(ROOT_DIR, dataset_name+'_noisy')  # Replace with desired output path
#os.makedirs(output_dir, exist_ok=True)

# Save the noisy dataset
#for idx, (image_tensor, label) in enumerate(dataset):
#    # Get class name
#    class_name = dataset.classes[label]
#    class_dir = os.path.join(output_dir, class_name)
#    os.makedirs(class_dir, exist_ok=True)
#    
#    # Convert the tensor back to a PIL image
#    noisy_image = transforms.ToPILImage()(image_tensor)
#    
#    # Save the image
#    save_path = os.path.join(class_dir, f"noisy_image_{idx}.png")
#    noisy_image.save(save_path)
#
#    if idx % 100 == 0:  # Progress update for large datasets
#        print(f"Processed {idx} images...")

#print("Noisy dataset created and saved successfully!")


# Define the number of teachers, batch size, and create teacher data loaders
num_teachers = 3
#batch_size = 32  # Adjust batch size based on your system's memory capacity

teachers = []
teacher_name = ['densenet201','densenet201','resnext']
teacher_folder = 'combined_standard'
teacher_dir = os.path.join(ROOT_DIR, teacher_folder)

for i in range(num_teachers):
    
    # Load the pre-trained ResNet-18 model
    #model = models.resnet18(pretrained=True)
    if teacher_name[i]=='resnext':
        model = resnext101_64x4d(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(class_names))  # Set the final layer to have 8 output classes
    else:
        model = models.densenet201(pretrained=True)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, len(class_names))  # Set the final layer to have 8 output classes
    # Freeze all layers except the final classification layer
    # for name, param in model.named_parameters():
    #     if "fc" in name:  # Unfreeze the final classification layer
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False
    
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)  # Use all parameters, lr = 0.001*2.82 do lr = 004 if batches are 64 sqrt(64/4)


    # Move the model to the GPU if available
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    model = model.to(device)

    model.load_state_dict(torch.load(os.path.join(teacher_dir,f"teacher{i+1}_{teacher_name[i]}_st.pth")))#, strict=False)
    print(f"Loaded teacher {i+1}")
    teachers.append(model)

print("all good")

# Function to make predictions using a model, predect student labels by teachers model
def predict(model, dataloader):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    device = "cpu"
    outputs = torch.zeros(0, dtype=torch.long, device=device)
    model.eval()
    for images, labels in dataloader:
        images = images.to(device)
        output = model.forward(images)
        ps = torch.argmax(torch.exp(output), dim=1)
        outputs = torch.cat((outputs, ps))
    return outputs

# Aggregate teacher predictions using Laplace mechanism
epsilon = 3.0 # or try 0.2
def aggregated_teacher(models, dataloader, dataset_size, epsilon):
    preds = torch.zeros((len(models),dataset_size), dtype=torch.long)
    for i, model in enumerate(models):
        results = predict(model, dataloader)
        preds[i] = results

    labels = np.array([]).astype(int)
    for image_preds in np.transpose(preds):
        label_counts = np.bincount(image_preds, minlength=7)
        beta = 1 / epsilon

        for i in range(len(label_counts)):
            label_counts[i] += np.random.laplace(0, beta, 1)
        label_counts = np.clip(label_counts, 0, None)  # Ensure no negative counts

        new_label = np.argmax(label_counts)
        labels = np.append(labels, new_label)
    return preds.numpy(), labels

preds, student_labels = aggregated_teacher(teachers, dataloaders['train'], dataset_sizes['train'], epsilon)

print(preds)
print(student_labels)

unique_labels = np.unique(student_labels)
print("Unique labels in student_labels:", unique_labels)
assert unique_labels.min() >= 0 and unique_labels.max() < len(class_names), "Invalid labels in student_labels!"

# Define a data loader for the student with labels generated from aggregated teacher predictions
def student_loader(student_train_loader, labels):
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i, (data, _) in enumerate(iter(student_train_loader)):
        yield data, torch.from_numpy(labels[i * len(data): (i + 1) * len(data)])


# Initialize the TensorBoard writer
writer = SummaryWriter(log_dir='runs/experiment1')  # Change the directory name as needed
student_model = models.resnet18(pretrained=True)
#student_model = models.densenet201(weights='DEFAULT')
#student_model = models.resnext101_64x4d(weights='DEFAULT')

num_features = student_model.fc.in_features
student_model.fc = nn.Linear(num_features, len(class_names))  # Set the final layer to have 8 output classes
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(student_model.parameters(), lr=0.0001, momentum=0.9)  # Use all parameters, do lr = 004 if batches are 64 sqrt(64/4)
student_model = student_model.to('cpu')
for param in student_model.parameters():
    param.requires_grad = True
epochs = 100
steps = 0
running_loss = 0
running_corrects = 0
total_samples = 0

# Lists to store training and validation accuracies and losses
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for e in range(epochs):
    student_model.train()
    train_loader = student_loader(dataloaders['train'], student_labels)
    for images, labels in train_loader:
        steps += 1
        optimizer.zero_grad()
        output = student_model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # Calculate training accuracy
        _, predss = torch.max(output, 1)
        running_corrects += torch.sum(predss == labels)
        total_samples += labels.size(0)
        running_corrects.double() / total_samples

    if steps % len(dataloaders['train']) == 0:
        test_loss = 0
        accuracy = 0
        all_predictions = []
        all_labels = []
        student_model.eval()
        with torch.no_grad():
            for images, labels in dataloaders['val']:
                log_ps = student_model(images)
                test_loss += criterion(log_ps, labels).item()
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                all_predictions.extend(top_class.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate confusion matrix and classification report
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        class_report = classification_report(all_labels, all_predictions, zero_division=0, output_dict=True)
        student_model.train()

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/Train', running_loss / len(dataloaders['train']), e)
        writer.add_scalar('Loss/Validation', test_loss / len(dataloaders['val']), e)
        writer.add_scalar('Accuracy/Train', running_corrects.double() / total_samples, e)
        writer.add_scalar('Accuracy/Validation', accuracy / len(dataloaders['val']), e)
        writer.add_scalar('F1_Score/Validation', f1_score(all_labels, all_predictions, average='weighted'), e)
        writer.add_scalar('Recall_Avg/Validation', recall_score(all_labels, all_predictions, average='weighted'), e)

        # Log recall for each class
        recall_scores = recall_score(all_labels, all_predictions, average=None)
        for i, recall in enumerate(recall_scores):
            writer.add_scalar(f'Recall_Class_{i}/Validation', recall, e)

        print("Epoch: {}/{}.. ".format(e + 1, epochs),
              "Train Loss: {:.3f}.. ".format(running_loss / len(dataloaders['train'])),
              "Train Accuracy: {:.3f}.. ".format(running_corrects.double() / total_samples),
              "val Loss: {:.3f}.. ".format(test_loss / len(dataloaders['val'])),
              "val Accuracy: {:.3f}".format(accuracy / len(dataloaders['val'])))

        # Store training and validation accuracies and losses
        train_losses.append(running_loss / len(dataloaders['train']))
        train_accuracies.append(running_corrects.double() / total_samples)
        val_losses.append(test_loss / len(dataloaders['val']))
        val_accuracies.append(accuracy / len(dataloaders['val']))

        print("Confusion Matrix:")
        print(conf_matrix)

        print("Classification Report:")
        print(class_report)

        # Reset variables for the next epoch
        running_loss = 0
        accuracy = 0
        all_predictions = []
        all_labels = []

# Close the TensorBoard writer
writer.close()

# Plotting
plt.figure(figsize=(12, 5))

# Plotting Losses
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting Accuracies
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracies')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate privacy using PATE
#data_dep_eps, data_ind_eps = pate.perform_analysis(teacher_preds=preds, indices=student_labels, noise_eps=epsilon,
#                                                   delta=1e-5)
#print("Data Independent Epsilon:", data_ind_eps)
#print("Data Dependent Epsilon:", data_dep_eps)
