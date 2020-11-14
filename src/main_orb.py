

import cv2
import matplotlib.pyplot as plt


# Set the default figure size
plt.rcParams['figure.figsize'] = [10,10]

# Load the training image
image = cv2.imread('/home/garima/Desktop/Beer_images/database/hailstorm_brewing.png')

# Convert the training image to RGB
training_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the training image to gray Scale
training_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the images
plt.subplot(121)
plt.title('Original Training Image')
plt.imshow(training_image)
plt.subplot(122)
plt.title('Gray Scale Training Image')
plt.imshow(training_gray, cmap = 'gray')
plt.show()
