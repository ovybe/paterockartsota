#Importing  Libraries
import torch
from torchvision import  transforms, datasets, models
from torch.utils.data import Subset , Dataset
from torch.distributions import Laplace
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import torch.optim as optim
from syft.frameworks.torch.differential_privacy import pate
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report
#from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from torchvision.models.resnet import ResNet, Bottleneck

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

# Set up data transformations with custom augmentation
transform = transforms.Compose([
    transforms.Resize((180, 180)),
    #AddLaplaceNoise(mean=0.0, scale=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the data directory
ROOT_DIR = os.path.abspath(os.curdir)
print(ROOT_DIR)
dataset_name='dataset'
data_dir = os.path.join(ROOT_DIR, dataset_name)
print(data_dir)

class Dataset(Dataset):
    def __init__(self, root_dir, dataset_folder, class_names=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.dataset_folder = dataset_folder
        self.class_names = class_names or sorted(os.listdir(root_dir))
        
        # Collect image paths and labels for all classes
        self.images = []
        self.labels = []
        
        for label, class_name in enumerate(self.class_names):
            class_dir = os.path.join(root_dir, dataset_folder, class_name)
            class_images = [os.path.join(class_dir, img) for img in os.listdir(class_dir)]
            self.images.extend(class_images)
            self.labels.extend([label] * len(class_images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image, label


# Create the dataset and apply transform
#dataset = Dataset(root_dir=ROOT_DIR, dataset_folder=dataset_name, class_names=['Circle','Cross','Goat','Person','Spiral','Stag','Zigzag'], transform=transform)
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform) for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print(dataset_sizes)

class_names = image_datasets['train'].classes
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

for i in range(num_teachers):
    
    # Load the pre-trained ResNet-18 model
    #model = models.resnet18(pretrained=True)
    #model = models.densenet201(pretrained=True)
    model = resnext101_64x4d(pretrained=True)
    num_features = model.fc.in_features
    #num_features = model.classifier.in_features
    model.fc = nn.Linear(num_features, len(class_names))  # Set the final layer to have 8 output classes
    #odel.classifier = nn.Linear(num_features, len(class_names))  # Set the final layer to have 8 output classes
    # Freeze all layers except the final classification layer
    # for name, param in model.named_parameters():
    #     if "fc" in name:  # Unfreeze the final classification layer
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False
    
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001*2.82, momentum=0.9)  # Use all parameters, do lr = 004 if batches are 64 sqrt(64/4)


    # Move the model to the GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    model.load_state_dict(torch.load(f"teacher{i+1}_resnext64_300e_old.pth"))#, strict=False)
    print(f"Loaded teacher {i+1}")
    teachers.append(model)

print("all good")

# Function to make predictions using a model, predect student labels by teachers model
def predict(model, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    outputs = torch.zeros(0, dtype=torch.long, device=device)
    model.eval()
    for images, labels in dataloader:
        images = images.to(device)
        output = model.forward(images)
        ps = torch.argmax(torch.exp(output), dim=1)
        outputs = torch.cat((outputs, ps))
    return outputs

# Aggregate teacher predictions using Laplace mechanism
epsilon = 0.2
def aggregated_teacher(models, dataloader, dataset_size, epsilon):
    preds = torch.zeros((len(models),dataset_size), dtype=torch.long)
    for i, model in enumerate(models):
        results = predict(model, dataloader)
        preds[i] = results

    labels = np.array([]).astype(int)
    for image_preds in np.transpose(preds):
        label_counts = np.bincount(image_preds, minlength=2)
        beta = 1 / epsilon

        for i in range(len(label_counts)):
            label_counts[i] += np.random.laplace(0, beta, 1)

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


# Train the student model using the labeled data
#student_model = models.resnet18(pretrained=True)
#student_model = models.densenet201(pretrained=True)
student_model = resnext101_64x4d(pretrained=True)
num_features = student_model.fc.in_features
student_model.fc = nn.Linear(num_features, len(class_names))  # Set the final layer to have 8 output classes
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001*2.82, momentum=0.9)  # Use all parameters, do lr = 004 if batches are 64 sqrt(64/4)
student_model = student_model.to('cpu')
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
            class_report = classification_report(all_labels, all_predictions, zero_division=0)
            student_model.train()
            print("Epoch: {}/{}.. ".format(e + 1, epochs),
                  "Train Loss: {:.3f}.. ".format(running_loss / len(dataloaders['train'])),
                  "Train Accuracy: {:.3f}.. ".format(running_corrects.double() / total_samples),
                  "val Loss: {:.3f}.. ".format(test_loss / len(dataloaders['val'])),
                  "val Accuracy: {:.3f}".format(accuracy / len(dataloaders['val']))),
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
