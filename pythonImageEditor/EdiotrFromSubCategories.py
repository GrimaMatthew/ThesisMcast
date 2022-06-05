import os, fileinput, sys
import random
import glob
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
        number = random.randrange(12, 16)
        cvImage = cv2.imread(file)
        print(number)
        imgBlur = img.filter(ImageFilter.BoxBlur(8))

        (name, extension) = os.path.splitext(file)

        # Save with "_blur" added to the filename
        imgBlur.save(name  + extension)



