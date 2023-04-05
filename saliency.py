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
from PIL import Image
import glob
import random

# %%
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

from torchvision.models import vgg16
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_model():

    ## Initialize pre-trained VGG
    model = vgg16(pretrained=True)

    ## Freeze layers
    for param in model.parameters():
        param.requires_grad = False

    ## Get GPU 
    
    ## Send to GPU
    model.to(DEVICE)
    print(f"Model is on {DEVICE}")
    return model

## Transforms
def format(image, size=224):
    transform = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def view(image):
    return transforms.ToPILImage()(image)

def prepare_image(img):
    ## Prepare image
    image = format(img)

    # fix weird 4th channel with only 255 values...
    if image.size(1) > 3:
      image = image[:, :3, :, :]
    #print(image.size())
    #view(image.squeeze(0))

    image = Variable(image, requires_grad=True)
    return image

## Open the image

def saliency_smooth(model, image, n=10):

    s = 0
    classes = []
    for i in range(n): 
        with torch.no_grad():
            noise = torch.normal(mean=torch.zeros_like(image), std=torch.ones_like(image)*1/1000)
        x = image+noise
        x = Variable(x, requires_grad=True)
        m, c = saliency(model, x)
        s += m
        classes.append(c)
    
    return s/n, classes

def saliency(model, image):
    model.eval()


    scores = model(image.to(DEVICE)) #TODO [forward pass the image through our model]

    ## Get scores
    class_index = scores.argmax()#TODO [get index of the highest score]
    class_score = max(scores[0]) #TODO [get highest score

    ## Backpropagate score
    #TODO [backpropagate class score]
    class_score.backward()

    ## Get saliency of image (hint: grad.data)
    saliency, _ = torch.max(image.grad.data.abs(),dim=1)
    return saliency, class_index

def plot_saliency(image, saliency, class_index=None):
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
    if class_index is not None:
    # Print predicted class
        print('Predicted Class: ' + str(class_index))
    
    plt.show()

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


def get_adverserial(epsilon=0.01):
    loss_metric = nn.CrossEntropyLoss()
    adv_attack = GradientAttack(loss_metric, epsilon)
    return adv_attack

def adverserial_attack(model, image, adv_attack):


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
    plt.show()

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
    plt.show()



def path_to_images(path):
    images = glob.glob(path)
    processed_images = [] 
    for image in images:
        processed_images.append(prepare_image(Image.open(image)))
    
    return processed_images

def images_to_saliency(images, model, saliency):

    saliencies = []
    classes = []
    for image in images:
        s, c = saliency(model, image)
        saliencies.append(s)
        classes.append(c)
    
    return saliencies, classes

def saliences_to_rgb(saliences):
    rgb = [None]*len(saliences)
    for i in range(len(saliences)):
        rgb[i] = torch.vstack([saliences[i]]*3)
    return rgb

from torchmetrics.image.fid import FrechetInceptionDistance

def main(path, n = 1):
    model = get_model()
    for s_map in [saliency, saliency_smooth]:
        images = path_to_images(path)
        saliencies, classes = images_to_saliency(images, model, s_map)
        #plot_saliency(images[0], saliencies[0], classes[0])
        #print(saliencies[0].shape)
        from fid import FID_score
        saliencies_3 = saliences_to_rgb(saliencies)
        permutations = [np.random.permutation(len(saliencies_3)) for i in range(n)]
        print(FID_score(saliencies_3, saliencies_3))
        fid = 0
        for indices in permutations:
            fid += FID_score([saliencies_3[i] for i in indices[:len(indices)//2]], [saliencies_3[i] for i in indices[len(indices)//2:]])
        fid = fid/n
        print(fid)
if __name__ == "__main__":
    path = '/media/extra/Respsonible/CUB_200_2011/images/001.Black_footed_Albatross/*'
    main(path) 
