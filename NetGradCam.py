import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

class NetGradCam(torch.nn.Module):

    def __init__(self, model, model_type) -> None:
        super().__init__()
        self.model_type = model_type
        self.gradients = None

        # Isolate the feature and classifier blocks
        # if model_type == 'vgg16':
        #     self.features = model.features
        #     self.avgpool = model.avgpool
        #     self.classifier = model.classifier
        # elif model_type == 'vgg19':
        #     self.features = model.features
        #     self.avgpool = model.avgpool
        #     self.classifier = model.classifier
        # elif model_type == 'resnet18':
        #     self.features = torch.nn.Sequential(
        #         model.conv1,
        #         model.bn1,
        #         model.relu,
        #         model.maxpool,
        #         model.layer1,
        #         model.layer2,
        #         model.layer3,
        #         model.layer4
        #     )
        #     self.avgpool = model.avgpool
        #     self.classifier = model.fc
        if model_type == 'resnet50':
            self.features = torch.nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4
            )
            self.avgpool = model.avgpool
            self.classifier = model.fc
        elif model_type == 'efficientnet_b0':
            self.features = model.features
            self.avgpool = model.avgpool
            self.classifier = model.classifier
        elif model_type == 'efficientnet_b1':
            self.features = model.features
            self.avgpool = model.avgpool
            self.classifier = model.classifier
        else:
            raise RuntimeError(f"Net not supported: {model_type}")

    def activation_hook(self, grad):
        self.gradients = grad

    def get_gradients(self):
        return self.gradients

    def get_activation_map(self, x):
        # store previous state
        state = self.training
        self.eval()
        # compute the prediction
        net_output = self.forward(x)
        pred = net_output[:, net_output.argmax(dim=1)]
        # get the gradients of the output with respect to the parameters of the model
        pred.backward()
        # pull the gradients across the channel
        gradients = self.get_gradients()
        # pool the gradients accross the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        # get the activations of the last convolutional layer
        activations = self.features(x)
        # weight the channels by corresponding gradients
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
        # average the channels by corresponding gradients
        heatmap = torch.mean(activations, dim=1).squeeze().detach().cpu().numpy()
        # relu on top of the heatmap
        # https://arxiv.org/pdf/1610.02391.pdf
        heatmap = np.maximum(heatmap, 0)
        # normalize the heatmap
        heatmap = torch.from_numpy(heatmap)
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)
        self.training = state

        return heatmap


    def forward(self, x):
        # extract the features
        x = self.features(x)
        # register the hook
        h = x.register_hook(self.activation_hook)

        # complete the forward passage
        x = self.avgpool(x)
        # flatten the output
        x = x.view((1, -1))
        x = self.classifier(x)

        return x