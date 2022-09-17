import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1)

        # Pooling (Max)
        self.maxpool = nn.MaxPool2d(kernel_size=3)

        xin_fake = torch.rand((1,) + (3, 250, 250)).type(torch.FloatTensor)
        y1 = self.conv1(xin_fake)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        n_features = self.maxpool(y3).numel()

        # print(n_features)

        # Fully connected layers
        self.fc1 = nn.Linear(n_features, 64)
        self.fc2 = nn.Linear(64, 2)

        # Dropout
        self.dropout2d = nn.Dropout2d(p=0.1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # Convolution block 1
        out = F.relu(self.conv1(x))

        # Convolution block 2 
        out = F.relu(self.conv2(out))

        # Convolution block 3
        out = F.relu(self.conv3(out))
        out = self.maxpool(out)
        out = self.dropout2d(out)

        # print(out.shape)

        # Create dense vector representation
        # (Bs, 32, 6, 6) - > (Bs, 32*6*6)
        # out = out.view(out.size(0), -1)
        out = out.reshape(out.shape[0], -1)

        # Linear (FC) layer [Here we would also need a softmax (multiclass classification), but we'll talk about that in a second]
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out


# Initialing compute device (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_train = np.load("rs0_it0_x_train.npy")
y_train = np.load("rs0_it0_labels_train.npy")
x_test = np.load("rs0_it0_x_test.npy")
y_test = np.load("rs0_it0_labels_test.npy")

x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

x_train = torch.permute(x_train, (0, 3, 1, 2))
x_test = torch.permute(x_test, (0, 3, 1, 2))

print("x_train.shape", x_train.shape, "x_test.shape", x_test.shape)

train = TensorDataset(x_train, y_train)
test = TensorDataset(x_test, y_test)

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

net = CNN().to(device)
loss_function = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=0.001)
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.5)

for epoch in range(50):  # 3 full passes over the data
    net.train()
    predictions = []
    ground_truth = []

    for x, y in trainset:
        x, y = x.float().to(device), y.long().to(device)

        net.zero_grad()  # sets gradients to 0 before loss calc
        output = net(x)
        loss = loss_function(output, y)
        loss.backward()  # apply this loss backwards thru the network's parameters
        optimizer.step()

        # Get predictions from the maximum value and append them to calculate accuracy later
        _, predicted = torch.max(output.data, 1)

        predictions.extend(predicted.detach().cpu().numpy().flatten().tolist())
        ground_truth.extend(y.detach().cpu().numpy().flatten().tolist())

    if epoch % 10 == 0:
        accuracy = accuracy_score(ground_truth, predictions)
        print('Epoch: {}. Loss: {}. Accuracy (on trainset/self): {}'.format(epoch, loss.item(), accuracy))
        print(ground_truth)
        print(predictions)
    else:
        print(f"epoch {epoch}")

with torch.no_grad():
    for x, y in testset:
        # x, y = data
        curr_x, curr_y = x.float().to(device), y.long().to(device)

        output = net(curr_x)

        curr_y = curr_y.detach().cpu().numpy()
        output = output.detach().cpu().numpy()
        prediction = np.argmax(output, axis=1)

        accuracy = accuracy_score(curr_y, prediction)
        precision = precision_score(curr_y, prediction, average='macro')
        recall = recall_score(curr_y, prediction, average='macro')

        print(accuracy, precision, recall)
