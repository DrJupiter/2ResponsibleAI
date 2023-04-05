import os
import numpy as np
import torch
from PIL import Image
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
from ProtoPNet.preprocess import mean, std, preprocess_input_function, undo_preprocess_input_function
from ProtoPNet.helpers import find_high_activation_crop
import copy 
class cfg:
    def __init__(self):
        self.prototype_info_path = os.getcwd() + '\\ProtoPNet\\saved_models\\vgg19\\003\\img\\epoch-150\\bb150.npy'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = 'ProtoPNet/saved_models/vgg19/003/150_17push0.2724.pth'


img_path = 'ProtoPNet/local_analysis/img_test/Painted_Bunting_0004_16641.JPEG'


def undo_preprocess(preprocessed_imgs, index=0):
    img_copy = copy.deepcopy(preprocessed_imgs[index:index+1])
    undo_preprocessed_img = undo_preprocess_input_function(img_copy)
    print('image index {0} in batch'.format(index))
    undo_preprocessed_img = undo_preprocessed_img[0]
    undo_preprocessed_img = undo_preprocessed_img.detach().cpu().numpy()
    undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1,2,0])

    return undo_preprocessed_img


def main(img_path, test_image_label):
    CFG = cfg()
    prototype_info = np.load(os.getcwd() + '\\ProtoPNet\\saved_models\\vgg19\\003\\img\\epoch-150\\bb150.npy')
    prototype_img_identity = prototype_info[:, -1]
    #### load model
    ppnet = torch.load(CFG.load_model_path, map_location=CFG.device)
    ppnet = ppnet.to(CFG.device)
    ppnet_multi = torch.nn.DataParallel(ppnet)

    print('Prototypes are chosen from ' + str(len(set(prototype_img_identity))) + ' number of classes.')
    print('Their class identities are: ' + str(prototype_img_identity))
    print(f"There are a total of N: {len(prototype_img_identity)}")
    print(prototype_info.shape)

    prototype_shape = ppnet.prototype_shape
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    img_size = 224
    normalize = transforms.Normalize(mean=mean,
                                 std=std)
    preprocess = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor(),
    normalize
    ])

    img_pil = Image.open(img_path)
    img_tensor = preprocess(img_pil)
    img_variable = torch.autograd.Variable(img_tensor.unsqueeze(0))
    images_test = img_variable.to(CFG.device)

    logits, min_distances = ppnet_multi(images_test)
    conv_output, distances = ppnet.push_forward(images_test)
    prototype_activations = ppnet.distance_2_similarity(min_distances)
    prototype_activation_patterns = ppnet.distance_2_similarity(distances)
    
    if ppnet.prototype_activation_function == 'linear':
        prototype_activations = prototype_activations + max_dist
        prototype_activation_patterns = prototype_activation_patterns + max_dist

    tables = []
    labels_test = torch.tensor([test_image_label])

    for i in range(logits.size(0)):
        tables.append((torch.argmax(logits, dim=1)[i].item(), labels_test[i].item()))
    +
    idx = 0
    predicted_cls = tables[idx][0]
    correct_cls = tables[idx][1]
    print('Predicted: ' + str(predicted_cls))
    print('Actual: ' + str(correct_cls))

    original_img = undo_preprocess(images_test, idx)
    
    array_act, sorted_indices_act = torch.sort(prototype_activations[idx])
    for i in range(1,11):
        activation_pattern = prototype_activation_patterns[idx][sorted_indices_act[-i].item()].detach().cpu().numpy()
        upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size),
                                                interpolation=cv2.INTER_CUBIC)
        
        # show the most highly activated patch of the image by this prototype
        high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern) # (start height, end height, start width, end width)
        # cropped image patch
        high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                    high_act_patch_indices[2]:high_act_patch_indices[3], :]
        
        
        



if __name__ == '__main__':
    main()