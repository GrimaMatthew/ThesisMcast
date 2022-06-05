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
    # Copy the file with os.rename
    (name, extension) = os.path.splitext(file)

    if filename.endswith('.jpg'):             # and  "Noise" not in filename
        img = Image.open(file)
        number = random.uniform(0.2, 0.6)
        cvImage = cv2.imread(file)

        print("contains no noise")
        print(filename)
        # imgBlur = img.filter(ImageFilter(number))
        imgNoise = random_noise(cvImage, mode='s&p', amount=0.04)  # amount =

        imgNoise = np.array(255 * imgNoise, dtype='uint8')

        # cv2.imshow("potato", imgNoise)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()



        # Save with "_blur" added to the filename
        #imgNoise.save(name + '_GaussianBlur(1-4)' + extension)

        fName = name + extension

        cv2.imwrite(fName,imgNoise)
    else:

        print("Contains Noise")
        print(filename)



