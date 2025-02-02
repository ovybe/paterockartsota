#Bouragaa Seif eddine Final project Machine learning Class

#Importing  Libraries
import torch
from torchvision import  transforms
from torch.utils.data import Subset , Dataset
from torchvision.utils import make_grid, draw_bounding_boxes
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import torch.optim as optim
from syft.frameworks.torch.differential_privacy import pate
import numpy as np
import matplotlib.pyplot as plt
import os
from bs4 import BeautifulSoup
from sklearn.metrics import confusion_matrix, classification_report
#from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


new_size = (224, 224)
# Set up data transformations with custom augmentation
transform = transforms.Compose([
    # transforms.RandomAffine(degrees=(-5, 5), scale=(1 - 0.1, 1 + 0.1)),  # Adjust scale and shear accordingly
    transforms.Resize(new_size),
    # transforms.RandomChoice([  # Adjust rotation if needed
    #     transforms.RandomChoice([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), transforms.RandomGrayscale(p=0.1)]),
    # ]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def rescale_bounding_boxes(original_shape: tuple, new_shape: tuple,
                           bounding_boxes: list) -> list:
    """
    Rescales bounding box coords according to new image size
    :param original_shape: (W, H)
    :param new_shape: (W, H)
    :param bounding_boxes: [[x1, y1, x2, y2], ...]
    :return: scaled bbox coords
    """
    original_w, original_h = original_shape
    new_w, new_h = new_shape
    bounding_boxes = np.array(bounding_boxes, dtype=np.float64)
    scale_h, scale_w = new_h / original_h, new_w / original_w
    bounding_boxes[:, 0] *= scale_w
    bounding_boxes[:, 1] *= scale_h
    bounding_boxes[:, 2] *= scale_w
    bounding_boxes[:, 3] *= scale_h
    bounding_boxes = np.clip(bounding_boxes, a_min=0, a_max=None)
    bounding_boxes = bounding_boxes.astype(np.uint32).tolist()
    return bounding_boxes


class RockArtDataset(object):
    def generate_box(self, obj):

        xmin = int(obj.find('xmin').text)
        ymin = int(obj.find('ymin').text)
        xmax = int(obj.find('xmax').text)
        ymax = int(obj.find('ymax').text)

        return [xmin, ymin, xmax, ymax]

    def generate_label(self, obj):
        if obj.find('name').text == "Person":
            return 1
        elif obj.find('name').text == "Goat":
            return 2
        elif obj.find('name').text == "Stag":
            return 3
        elif obj.find('name').text == "Circle":
            return 4
        elif obj.find('name').text == "Spiral":
            return 5
        elif obj.find('name').text == "Zigzag":
            return 6
        elif obj.find('name').text == "Cross":
            return 7
        return 0
    def generate_target(self, image_id, file):
        with open(file) as f:
            data = f.read()
            soup = BeautifulSoup(data, 'xml')
            objects = soup.find_all('object')

            num_objs = len(objects)

            # Bounding boxes for objects
            # In coco format, bbox = [xmin, ymin, width, height]
            # In pytorch, the input should be [xmin, ymin, xmax, ymax]
            boxes = []
            labels = []
            for i in objects:
                boxes.append(self.generate_box(i))
                labels.append(self.generate_label(i))
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # Labels (In my case, I only one class: target class or background)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            # Tensorise img_id
            img_id = torch.tensor([image_id])
            # Annotation is in dictionary format
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = img_id

            return target
    
    def __init__(self, transform, root_dir):
        self.transforms = transform
        # load all image files, sorting them to
        # ensure that they are aligned
        self.root_dir = root_dir
        self.imgs = list(sorted(os.listdir(f'{root_dir}/content/files/')))
        self.labels = list(sorted(os.listdir(f'{root_dir}/content/labels/')))

    def __getitem__(self, idx):
        # load images ad masks
        print(idx)
        file_image = self.imgs[idx]
        file_label = self.labels[idx]
        img_path = os.path.join(f'{self.root_dir}/content/files/', file_image)
        label_path = os.path.join(f'{self.root_dir}/content/labels', file_label)
        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        #Generate Label
        target = self.generate_target(idx, label_path)

        if self.transforms is not None:
            img, target["boxes"] = self.transforms(img, target["boxes"])

        return img, target

    def __len__(self):
        return len(self.imgs)
                

#     def __getitem__(self, idx):
#         if isinstance(self.images[idx], str):  # Check if the image is a path
#             img_path = self.images[idx]
#             image = Image.open(img_path).convert('RGB')
#             print("what am I")
#         else:
#             # If it's a NumPy array (resampled image), convert it to PIL Image
#             image = Image.fromarray(self.images[idx].astype('uint8'), 'RGB')
#             print("test")
#         if self.transform:
#             image = self.transform(image)

#         label = self.labels[idx]
#         bbox = self.boxes[idx]

#         return image, label

# Create the dataset and apply transform
dataset = RockArtDataset(transform=transform, root_dir='/home/lemawul/PATE')

print(dataset.__getitem__(1))
die()
# Load image data into a NumPy array
image_data = np.array([np.array(Image.open(img_path).convert('RGB').resize(new_size)) for img_path in dataset.images])
# Flatten the image data
image_data_flatten = image_data.reshape(image_data.shape[0], -1)
# Apply SMOTE to the entire dataset
if len(set(dataset.labels)) == 2:  # Check if it's a binary classification problem
    minority_class = 1  # Tuberculosis is the minority class
    minority_class_indices = [i for i, label in enumerate(dataset.labels) if label == minority_class]

    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(image_data_flatten, dataset.labels)

    # Reshape the resampled data
    X_resampled = X_resampled.reshape(X_resampled.shape[0], 224, 224, 3)

    # Update the dataset with the resampled data
    dataset.images = X_resampled
    dataset.labels = y_resampled

#split the dataset
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size]) #traning data for teachers and Test for student training and validation


# Define the number of teachers, batch size, and create teacher data loaders
num_teachers = 10
batch_size = 32  # Adjust batch size based on your system's memory capacity

# Get the labels from the dataset
labels = dataset.labels

# Convert labels to class names for better visualization
class_names = ['Normal', 'Tuberculosis']
class_labels = [class_names[label] for label in labels]

# Plot the distribution of images
plt.figure(figsize=(8, 6))
unique_labels, counts = np.unique(class_labels, return_counts=True)
plt.bar(unique_labels, counts, color=['blue', 'orange'], edgecolor='black', alpha=0.7)
plt.title('Distribution of Images')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Function to plot a grid of images with normalization
def plot_images(images, titles, h, w, rows=1, cols=5):
    plt.figure(figsize=(15, 3 * rows))
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)

        # Normalize pixel values for display
        image = images[i].permute(1, 2, 0).numpy()
        vmin, vmax = image.min(), image.max()
        image = (image - vmin) / (vmax - vmin)

        plt.imshow(image)
        plt.title(titles[i])
        plt.axis('off')
    plt.show()


# Get a sample of normal and tuberculosis images
normal_indices = [i for i, label in enumerate(dataset.labels) if label == 0][:5]
tb_indices = [i for i, label in enumerate(dataset.labels) if label == 1][:5]

normal_images = [dataset[i][0] for i in normal_indices]
tb_images = [dataset[i][0] for i in tb_indices]

# Convert labels to class names for titles
normal_titles = ['Normal'] * len(normal_images)
tb_titles = ['Tuberculosis'] * len(tb_images)

# Plot the sample of images
plot_images(normal_images + tb_images, normal_titles + tb_titles, 224, 224, rows=2, cols=5)


#creates DataLoader objects for each teacher, dividing the training data equally among them
def get_data_loaders(train_data, num_teachers):
    teacher_loaders = []
    data_size = len(train_data) // num_teachers
    teacher_data_counts = []
    for i in range(num_teachers):
        indices = list(range(i * data_size, (i + 1) * data_size))
        subset_data = Subset(train_data, indices)
        loader = torch.utils.data.DataLoader(subset_data, batch_size=batch_size)
        teacher_loaders.append(loader)
        teacher_data_counts.append(len(subset_data))  # Store the number of data points

    # Plot the number of data points for each teacher
    plt.bar(range(1, num_teachers + 1), teacher_data_counts, color='blue', edgecolor='black', alpha=0.7)
    plt.title('Number of Data Points for Each Teacher')
    plt.xlabel('Teacher Number')
    plt.ylabel('Number of Data Points')
    plt.show()
    return teacher_loaders


teacher_loaders = get_data_loaders(train_data, num_teachers)

indices = list(range(len(test_data)))
# Split indices into training and validation sets
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
# Create Subset objects using the selected indices
student_train_data = Subset(test_data, train_indices)
student_test_data = Subset(test_data, val_indices)
#Creates DataLoader objects for the student's training and validation data
student_train_loader = torch.utils.data.DataLoader(student_train_data, batch_size=batch_size)
student_test_loader = torch.utils.data.DataLoader(student_test_data, batch_size=batch_size)


# Define the neural network model , a simple CNN model

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(20 * 29 * 29, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Function to train a model , train teachers
def train(model, trainloader, criterion, optimizer, epochs=10):
    losses = []
    accuracies = []
    for e in range(epochs):
        running_loss = 0
        running_corrects = 0
        steps = 0
        model.train()
        for images, labels in trainloader:
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            steps += 1

            # Calculate accuracy
            _, preds = torch.max(output, 1)
            running_corrects += torch.sum(preds == labels)
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = running_corrects.double() / len(trainloader.dataset)
        losses.append(epoch_loss)
        accuracies.append(epoch_acc.item())
        print(f"Epoch {e + 1}/{epochs} | Loss: {np.round(epoch_loss, 3)} | Accuracy: {np.round(epoch_acc.item(), 3)}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


# Function to make predictions using a model, predect student labels by teachers model
def predict(model, dataloader):
    outputs = torch.zeros(0, dtype=torch.long)
    model.eval()
    for images, labels in dataloader:
        output = model.forward(images)
        ps = torch.argmax(torch.exp(output), dim=1)
        outputs = torch.cat((outputs, ps))
    return outputs


# Train teacher models
def train_models(num_teachers):
    models = []
    for i in range(num_teachers):
        model = Classifier()
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        print(f"Training Teacher {i + 1}...")
        train(model, teacher_loaders[i], criterion, optimizer)
        models.append(model)
        print(f"Teacher {i + 1} training completed.")
    return models


models = train_models(num_teachers)
# Aggregate teacher predictions using Laplace mechanism
epsilon = 0.2
def aggregated_teacher(models, dataloader, epsilon):
    preds = torch.zeros((len(models),1680), dtype=torch.long)
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


teacher_models = models
preds, student_labels = aggregated_teacher(teacher_models, student_train_loader, epsilon)


# Define a data loader for the student with labels generated from aggregated teacher predictions
def student_loader(student_train_loader, labels):
    for i, (data, _) in enumerate(iter(student_train_loader)):
        yield data, torch.from_numpy(labels[i * len(data): (i + 1) * len(data)])


# Initialize the SummaryWriter
writer = SummaryWriter('runs/student_model_experiment')

# Train the student model using the labeled data
student_model = models.resnet18(pretrained=True)
num_features = student_model.fc.in_features
student_model.fc = nn.Linear(num_features, len(class_names))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(student_model.parameters(), lr=0.001)
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
    train_loader = student_loader(student_train_loader, student_labels)
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

    if steps % len(student_train_loader) == 0:
        test_loss = 0
        accuracy = 0
        all_predictions = []
        all_labels = []
        student_model.eval()
        with torch.no_grad():
            for images, labels in student_test_loader:
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
        class_report = classification_report(all_labels, all_predictions, output_dict=True)
        
        # Calculate F1 score and recall
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        individual_recall = recall_score(all_labels, all_predictions, average=None)
        average_recall = recall_score(all_labels, all_predictions, average='macro')

        student_model.train()
        print("Epoch: {}/{}.. ".format(e + 1, epochs),
              "Train Loss: {:.3f}.. ".format(running_loss / len(student_train_loader)),
              "Train Accuracy: {:.3f}.. ".format(running_corrects.double() / total_samples),
              "val Loss: {:.3f}.. ".format(test_loss / len(student_test_loader)),
              "val Accuracy: {:.3f}".format(accuracy / len(student_test_loader)))

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/Train', running_loss / len(student_train_loader), e)
        writer.add_scalar('Loss/Validation', test_loss / len(student_test_loader), e)
        writer.add_scalar('Accuracy/Train', running_corrects.double() / total_samples, e)
        writer.add_scalar('Accuracy/Validation', accuracy / len(student_test_loader), e)
        writer.add_scalar('F1 Score/Validation', f1, e)
        writer.add_scalar('Recall/Average', average_recall, e)
        
        # Log individual recall scores
        for i, rec in enumerate(individual_recall):
            writer.add_scalar(f'Recall/Class_{i}', rec, e)

        # Store training and validation accuracies and losses
        train_losses.append(running_loss / len(student_train_loader))
        train_accuracies.append(running_corrects.double() / total_samples)
        val_losses.append(test_loss / len(student_test_loader))
        val_accuracies.append(accuracy / len(student_test_loader))

        print("Confusion Matrix:")
        print(conf_matrix)

        print("Classification Report:")
        print(class_report)

        # Reset variables for the next epoch
        running_loss = 0
        accuracy = 0
        all_predictions = []
        all_labels = []

# Close the SummaryWriter
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
data_dep_eps, data_ind_eps = pate.perform_analysis(teacher_preds=preds, indices=student_labels, noise_eps=epsilon,
                                                   delta=1e-5)
print("Data Independent Epsilon:", data_ind_eps)
print("Data Dependent Epsilon:", data_dep_eps)
