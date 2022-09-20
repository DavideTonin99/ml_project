import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, dataloader, DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.utils.data as data
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score, ConfusionMatrixDisplay

from pathlib import Path
from torch import optim

from NetGradCam import NetGradCam
import cv2
from Dataset import Dataset

conf = Dataset.parse_conf()

IMAGES_PATH = 'PlantVillage'
dir_path = Path(IMAGE_PATH)
transformer = torchvision.transforms.Compose([
    transforms.Resize(size = (224, 224)),
    transforms.ToTensor(),
])

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])

# Load image to Dataset
datafolder = torchvision.datasets.ImageFolder(dir_path, transform = transformer)
classes = datafolder.classes

# Set up device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Random split
train_set_size = int(len(datafolder) * 0.8)
valid_set_size = len(datafolder) - train_set_size
train_set, valid_set = data.random_split(datafolder, [train_set_size, valid_set_size])

network_type = "efficientbet_b0" # efficientnet_b0 | efficientnet_b1 | resnet50
model = torchvision.models.efficientnet_b0(weights = 'DEFAULT')
# model = torchvision.models.efficientnet_b1(weights = 'DEFAULT')
# model = torchvision.models.resnet50(pretrained = True)

for p in model.parameters():
    p.requires_grad = False

model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(classes), bias=True)

BATCH_SIZE = 250
train_loader = DataLoader(dataset = train_set, batch_size = BATCH_SIZE, shuffle = True)
valid_loader = DataLoader(dataset = valid_set, batch_size = BATCH_SIZE)
gradcam_loader = DataLoader(dataset = valid_set, batch_size = 1)

model.to(device)

loss_fn = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
scheduler = StepLR(optimizer, step_size = 10, gamma = 0.9)

EPOCHS = 101
history = {
    'train': {'accuracy': [], 'loss': []},
    'valid': {'accuracy': [], 'loss': []}
}

for epoch in range(EPOCHS):
    train_predictions = []
    train_ground_truth = []
    valid_predictions = []
    valid_ground_truth = []

    train_loss = 0
    model.train()
    scheduler.step()

    for batch, (X, y) in enumerate(train_loader):
        X = normalize(X).to(device)
        y = y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        _, train_predicted = torch.max(y_pred.data, 1)
        train_predictions.extend(train_predicted.detach().cpu().numpy().flatten().tolist())
        train_ground_truth.extend(y.detach().cpu().numpy().flatten().tolist())

    if epoch % 10 == 0:
        # every 10 epochs, validate and plot accuracy and loss
        valid_loss = 0

        model.eval()
        for batch, (X, y) in enumerate(valid_loader):
            X = normalize(X).to(device)
            y = y.to(device)

            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            valid_loss += loss.item()

            _, valid_predicted = torch.max(y_pred.data, 1)
            valid_predictions.extend(valid_predicted.detach().cpu().numpy().flatten().tolist())
            valid_ground_truth.extend(y.detach().cpu().numpy().flatten().tolist())

        train_accuracy = accuracy_score(train_ground_truth, train_predictions)
        valid_accuracy = accuracy_score(valid_ground_truth, valid_predictions)

        print(f"Epoch: {epoch}. Scheduler: {scheduler.get_lr()[0]:.2f}")
        print(f"Train Loss: {(train_loss/len(train_loader)):.4f}.Train Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {(valid_loss/len(valid_loader)):.4f}.Validation Accuracy: {valid_accuracy:.4f}")

        history["train"]["accuracy"].append(train_accuracy)
        history["train"]["loss"].append(train_loss / len(train_loader))
        history["valid"]["accuracy"].append(valid_accuracy)
        history["valid"]["loss"].append(valid_loss / len(valid_loader))

        xx = np.arange(len(history["train"]["accuracy"])) * 10
        # clear figure
        plt.clf()
        title = f"Epoch {epoch}. Train Accuracy: {train_accuracy:.4f}. Validation Accuracy: {valid_accuracy:.4f}"
        plt.suptitle(title)
        # show accuracy
        plt.subplot(121)
        plt.grid(True)
        plt.xlabel("Accuracy")
        plt.plot(xx, history["train"]["accuracy"])
        plt.plot(xx, history["valid"]["accuracy"], color="orange")

        # show loss
        plt.subplot(122)
        plt.grid(True)
        plt.xlabel("Loss")
        plt.plot(xx, history["train"]["loss"])
        plt.plot(xx, history["valid"]["loss"], color="orange")
        # update and display figure
        plt.pause(0.2)

# GRADCAM
net_gradcam = NetGradCam(model, network_type)
for p in net_gradcam.parameters():
    p.requires_grad = True

net_gradcam.classifier = model.classifier
net_gradcam.to(device)
net_gradcam.train()

for index, (X, y) in enumerate(gradcam_loader):
    X_normalized = normalize(X).to(device)
    y = y.to(device)
    y_pred = net_gradcam(X_normalized)

    _, prediction = torch.max(y_pred.data, 1)
    print(f"Prediction {classes[prediction]}. Ground Truth: {classes[y]}")

    heatmap_tensor = net_gradcam.get_activation_map(X_normalized)
    heatmap = heatmap_tensor.detach().cpu().numpy()

    img = X.squeeze().permute(1, 2, 0).numpy()

    # interpolate the heatmap
    heatmap_cv2 = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_cv2 = np.uint8(255 * heatmap_cv2)
    heatmap_jet = cv2.applyColorMap(heatmap_cv2, cv2.COLORMAP_JET)
    heatmap_jet = cv2.cvtColor(heatmap_jet, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 8))
    plt.suptitle(f"GradCAM visualization")
    plt.subplot(221)
    plt.title("Original image")
    plt.imshow(img)

    plt.subplot(222)
    plt.title("Raw activation map")
    plt.imshow(heatmap)

    plt.subplot(223)
    plt.title("Activation map mapped on image")
    plt.imshow(heatmap_jet)

    plt.subplot(224)
    plt.title("Superimposed image")
    plt.imshow(img)
    plt.imshow(heatmap_jet, cmap='jet', alpha=0.4)

    plt.pause(1)

# test phase
valid_predictions = []
valid_ground_truth = []
valid_loss = 0

for batch, (X, y) in enumerate(valid_loader):
    X = normalize(X).to(device)
    y = y.to(device)

    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    valid_loss += loss.item()

    _, valid_predicted = torch.max(y_pred.data, 1)
    valid_predictions.extend(valid_predicted.detach().cpu().numpy().flatten().tolist())
    valid_ground_truth.extend(y.detach().cpu().numpy().flatten().tolist())

valid_accuracy = accuracy_score(valid_ground_truth, valid_predictions)
valid_precision = precision_score(valid_ground_truth, valid_predictions, average='macro')
valid_recall = recall_score(valid_ground_truth, valid_predictions, average='macro')
valid_f1_score = f1_score(valid_ground_truth, valid_predictions, average='macro')

confusion_mat = confusion_matrix(valid_ground_truth, valid_predictions)
display_confusion_mat = ConfusionMatrixDisplay(confusion_matrix=confusion_mat)
display_confusion_mat.plot()
display_confusion_mat.figure_.savefig(
    os.path.join(conf["folder"]["OUTPUT_ANALYSIS"],
                    f'confusion_matrix_{random_state}_it{iteration}.png'))

output_str = f"Network: {network_type}\n" \
                f"n_train_samples: {len(train_set)}, n_test_samples: {len(valid_set)}\n" \
                f"accuracy: {valid_accuracy} \n" \
                f"precision: {valid_precision} \n" \
                f"recall: {valid_recall} \n" \
                f"f1 score: {valid_f1_score} \n"
print(output_str)
f = open(os.path.join(conf["folder"]["OUTPUT_ANALYSIS"], "{network_type}_stats.txt"), "a")
f.write(output_str)
f.close()
