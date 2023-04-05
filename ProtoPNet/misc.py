import os
import numpy as np
import torch
from PIL import Image, ImageDraw
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
from preprocess import mean, std, preprocess_input_function, undo_preprocess_input_function
from helpers import find_high_activation_crop
import copy

CWD = os.getcwd()
class cfg:
    def __init__(self):
        self.prototype_info_path = "./npy_files/bb150.npy"# os.getcwd() + '\\ProtoPNet\\saved_models\\vgg19\\003\\img\\epoch-150\\bb150.npy'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model_path = "./150_17push0.2724.pth"#'ProtoPNet/saved_models/vgg19/003/150_17push0.2724.pth'

def undo_preprocess(preprocessed_imgs, index=0):
    img_copy = copy.deepcopy(preprocessed_imgs[index:index+1])
    undo_preprocessed_img = undo_preprocess_input_function(img_copy)
    print('image index {0} in batch'.format(index))
    undo_preprocessed_img = undo_preprocessed_img[0]
    undo_preprocessed_img = undo_preprocessed_img.detach().cpu().numpy()
    undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1,2,0])

    return undo_preprocessed_img


def run_analysis(img_path, test_image_label, saliency_path=None):
    CFG = cfg()
    prototype_info = np.load("./npy_files/bb150.npy")
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
    
    idx = 0
    predicted_cls = tables[idx][0]
    correct_cls = tables[idx][1]
    print('Predicted: ' + str(predicted_cls))
    print('Actual: ' + str(correct_cls))

    original_img = undo_preprocess(images_test, idx)
    
    # if saliency_path is not None:
    #     saliency_img = np.array(Image.open(saliency_path))
    #     max_sali = np.max(saliency_img)
    #     activations = []

    array_act, sorted_indices_act = torch.sort(prototype_activations[idx])
    patches, boxes = [], []
    for i in range(1,11):
        activation_pattern = prototype_activation_patterns[idx][sorted_indices_act[-i].item()].detach().cpu().numpy()
        upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size),
                                                interpolation=cv2.INTER_CUBIC)
        
        # show the most highly activated patch of the image by this prototype
        high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern) # (start height, end height, start width, end width)
        # cropped image patch
        high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                    high_act_patch_indices[2]:high_act_patch_indices[3], :]
        
        patches.append(high_act_patch)
        boxes.append(high_act_patch_indices)


    return patches, boxes


def compare_with_saliency_maps(boxes, saliency_img):
    N = len(boxes)
    top_N = min(N,3)
    activations = []
    activation_means = []

    max_sali = np.max(saliency_img)
    
    for i in range(top_N):
        # total_act = np.sum(saliency_map)
        # saliency_patch = saliency_map[boxes[i][0] : boxes[i][1], 
        #                             boxes[i][2] : boxes[i][3], :] 
        
        # act = np.sum(saliency_map/total_act)
        # activations.append(act)
        full_saliency_activation = np.sum(saliency_img/max_sali)
        high_act_patch_saliency = saliency_img[boxes[i][0]:boxes[i][1],
                        boxes[i][2]:boxes[i][3], :]
        patch_saliency_activation = np.sum(high_act_patch_saliency/max_sali)
        patch_saliency_activation_mean = np.mean(high_act_patch_saliency/max_sali)
        # Image.fromarray(np.array((1*high_act_patch_saliency/max_sali+1)*max_sali,dtype=np.uint8)).show()
        activation = patch_saliency_activation/full_saliency_activation*100
        activation_mean = patch_saliency_activation_mean/np.mean(saliency_img/max_sali)

        activations.append(activation)
        activation_means.append(activation_mean)

        sail_draw = Image.fromarray(saliency_img)
        img_draw = ImageDraw.Draw(sail_draw)
        shape = [(boxes[i][2],boxes[i][0]),(boxes[i][3],boxes[i][1])]
        img_draw.rectangle(shape, outline ="red")
        sail_draw.save(f"./birds_with_squares/bird_pelican_{i}_act_{activation}.jpg")

    print("% Activations:\n",activations)
    print("\n% Activation means within patch compared to mean in entire saliency map:\n", activation_means)


    return activations

if __name__ == '__main__':
    # img_path = './bird_imgs/Dark_Eyed_Junco_CUB.jpg'
    # saliency_path = "./saliency_imgs/saliency_dark_eyed.png"
    # img_path = './bird_imgs/Indigo_Bunting_CUB.jpg'
    # saliency_path = "./saliency_imgs/saliency_indigo_bunting.png"
    img_path = './bird_imgs/White_Pelican_CUB.jpg'
    saliency_path = "./saliency_imgs/saliency_pelican.png"
    img_class = 15
    saliency_map = np.array(Image.open(saliency_path))
    patches, boxes = run_analysis(img_path, img_class)
    acts = compare_with_saliency_maps(boxes, saliency_map)

