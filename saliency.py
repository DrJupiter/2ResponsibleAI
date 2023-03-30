# %%
## Standard Library
import os
import json

## External Libraries
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as functional
from torch.utils.data import Dataset, DataLoader
from skimage import io
import matplotlib.pyplot as plt
import requests
from PIL import Image

# %%

from torchvision.models import vgg16

## Initialize pre-trained VGG
model = vgg16(pretrained=True)

## Freeze layers
for param in model.parameters():
    param.requires_grad = False

## Get GPU 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
## Send to GPU
model.to(device)
print(f"Model is on {device}")

## Open the image
img = Image.open('white_pelican.jpg')

## Transforms
def format(image, size=224):
    transform = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def view(image):
    return transforms.ToPILImage()(image)

## Prepare image
image = format(img)

# fix weird 4th channel with only 255 values...
if image.size(1) > 3:
  image = image[:, :3, :, :]
print(image.size())
#view(image.squeeze(0))

image = Variable(image, requires_grad=True)

## Forward pass
model.eval()
scores = model(image.to(device)) #TODO [forward pass the image through our model]

## Get scores
class_index = scores.argmax()#TODO [get index of the highest score]
class_score = max(scores[0]) #TODO [get highest score

## Backpropagate score
#TODO [backpropagate class score]
class_score.backward()

## Get saliency of image (hint: grad.data)
saliency, _ = torch.max(image.grad.data.abs(),dim=1)
#TODO [compute saliency]

## Visualize input image and saliency map

fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# Show input image
ax1.imshow(view(image[0]))
ax1.axis('off')

# Show saliency map
ax2.imshow(saliency[0], cmap=plt.cm.hot)
ax2.axis('off')

# Print predicted class
print('Predicted Class: ' + str(class_index))

## Gradient Attack Class

class GradientAttack():
        def __init__(self, loss, epsilon):
            self.loss = loss
            self.epsilon = epsilon

        def forward(self, x, y_true, model):
            loss = self.loss(model(x), y_true)
            loss.backward()
            x = torch.clamp(x + torch.sign(x.grad) * self.epsilon, 0, 1)
            return x

loss_metric = nn.CrossEntropyLoss()
adv_attack = GradientAttack(loss_metric, 0.01)

labels = model(image.cuda()).argmax(dim=1)
image_cln = Variable(image.cuda(), requires_grad=True)
image_adv = adv_attack.forward(image_cln, labels, model)
image_adv.retain_grad()


print("original img:\n", image.size())
print("adv img:\n", image_adv.size())

## Show adversarial comparison with original
fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# Show original image
ax1.imshow(view(image_cln[0]))
ax1.axis('off')
ax1.set_title('Original Class: ' + str(model(image_cln[[0]]).argmax(dim=1)[0]));

# Show adversarial image
ax2.imshow(view(image_adv[0]))
ax2.axis('off')
ax2.set_title('Predicted Class: ' + str(model(image_adv[[0]]).argmax(dim=1)[0]));

#TODO [make a saliency map of the adversarial image]

model.eval()
scores_adv = model(image_adv)##  TODO: [Forward pass]

## TODO: [Get scores]
class_index_adv = scores_adv.argmax()
class_score_adv = max(scores_adv[0])

## TODO: [Backpropagate score]
class_score_adv.backward()
saliency_adv = torch.max(image_adv.grad.data.abs(),dim=1)

## Visualize saliency map of original image and adversarial image

fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# Show saliency map of original image
ax1.imshow(saliency[0], cmap=plt.cm.hot)
ax1.axis('off')
ax1.set_title('Original Image');

# Show saliency map of adversarial image
ax2.imshow(saliency_adv[0].squeeze(0).cpu().numpy(), cmap=plt.cm.hot)
ax2.axis('off')
ax2.set_title('Adversarial Image');