import os, fileinput, sys
import random
import glob

import PIL
import numpy as np
import cv2
from PIL import Image
from PIL import ImageFilter
from skimage.util import random_noise

my_path = "Train"
# Location to move images to


# Get List of all images
files = glob.glob(my_path + '/**/*.jpg', recursive=True)

# For each image
for file in files:
    # Get File name and extension
    filename = os.path.basename(file)
    # check file ending might need to implement same method as noise i.e 'd' in word

    if filename.endswith('.jpg'):

        img = Image.open(file)
        number = random.randrange(1, 5)
        cvImage = cv2.imread(file)
        print("TEST")
        img_bw = cv2.cvtColor(cvImage, cv2.COLOR_BGR2GRAY)
        imgFinal_bw = PIL.Image.fromarray(img_bw)

        (name, extension) = os.path.splitext(file)

        # Save with "_blur" added to the filename
        imgFinal_bw.save(name + extension)