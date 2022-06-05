from PIL import Image
from PIL import ImageFilter
import glob
import cv2
import os, fileinput, sys
import numpy as np



my_path = "Dataset/"

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
        # Specify the kernel size.
        # The greater the size, the more the motion.
        kernel_size = 20

        # Create the vertical kernel.
        kernel_v = np.zeros((kernel_size, kernel_size))


        # Fill the middle row with ones.
        kernel_v[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)

        # Normalize.
        kernel_v /= kernel_size
        print(file)

        # Apply the vertical kernel.
        vertical_mb = cv2.filter2D(np.float32(img), -1, kernel_v)
        (name, extension) = os.path.splitext(file)
        fName = name + extension

        cv2.imwrite(fName, vertical_mb)





