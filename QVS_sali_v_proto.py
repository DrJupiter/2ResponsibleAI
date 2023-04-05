#%%
import numpy as np
from PIL import Image

# Load file with receptive field / patch in it
prototype_info = np.load("./npy_files/bb150.npy")
# what index is the chosen file at
img_index = 1

# get values for patch location
bbox_height_start=prototype_info[img_index][1]
bbox_height_end=prototype_info[img_index][2]
bbox_width_start=prototype_info[img_index][3]
bbox_width_end=prototype_info[img_index][4]

# Load Saliency map
img = np.array(Image.open("./saliency_imgs/saliency_tst_img.png"))

# Convert to [0,1] and flip black to be 0 and white 1, and also remove last channel if there are 4.
img = -(img[:,:,:3]/255)+1

# get total activation
total_activation = np.sum(img)

# apply patch to saliency
patch_img = img[bbox_height_start:bbox_height_end,bbox_width_start:bbox_width_end]

# get activation after patch
patch_activation = np.sum(patch_img)

# find relative activation
activation = patch_activation/total_activation

# print stuff
print(activation)
Image.fromarray(np.round(-(patch_img*255)+1,0).astype(np.uint8)) # convert back and display
