"""
In this file we implement the SIFT algorithm.
This implementation closely follows OpenCV's implementation
"""
import pdb
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_base_image(image, sigma, initial_assumed_blur, verbose=True):
    """
    This method doubles the input image in size and applies Gaussian blur.
    We need to double the size of the input image by `diff` to achieve the blur of `sigma`.
    Source: https://www.wikiwand.com/en/Gaussian_blur
    """
    if verbose:
        print('[INFO] Generating the base image')
    updated_image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    diff = np.sqrt(max((sigma ** 2) - ((2 * initial_assumed_blur) ** 2), 0.01))
    base_image = cv2.GaussianBlur(updated_image, (0, 0), sigmaX=diff, sigmaY=diff)
    return base_image

def get_nos_octaves(shape):
    """
    This method is used for computing number of octaves in the image pyramid
    This is used to know the number of times we can half an image till it becomes too small.
    The `-1` in the equation ensures that we have the a side length of at least 3.
    """
    min_side = min(shape)
    octaves = round(np.log(min_side) / np.log(2) - 1)
    return int(octaves)

def get_Gaussian_kernels(sigma, num_intervals, verbose=True):
    """
    This method generates a list of kernels at which we wish to blur the input image.
    We use the value of sigma, intervals, and octaves from Section 3 of Lowe's paper.
    Interested users can refer to Fig. 3, 4, and 5 in the paper.
    """
    if verbose:
        print('[INFO] Getting scales')
    count_per_octave = num_intervals + 3    # Covering the images one step above the first image in the layer and one step below the last image in the layer
    const = 2 ** (1. / num_intervals)
    kernels = np.zeros(count_per_octave)
    kernels[0] = sigma

    for i in range(1, count_per_octave):
        prev_sigma = (const ** (i - 1)) * sigma
        total_sigma = const * prev_sigma
        kernels[i] = np.sqrt(total_sigma ** 2 - prev_sigma ** 2)
    return kernels

def get_Gaussian_images(image, octaves, kernels, verbose=True):
    """
    This method is used for generating a scale-space pyramid of Gaussian images
    """
    if verbose:
        print('[INFO] Preparing Gaussian images')
    images = list()
    for octave_count in range(octaves):
        images_in_octave = [image]
        for kernel in kernels[1:]:
            image = cv2.GaussianBlur(image, (0, 0), sigmaX=kernel, sigmaY=kernel)
            images_in_octave.append(image)
        images.append(images_in_octave)
        base = images_in_octave[-3]
        image = cv2.resize(base, (int(base.shape[1]/2), int(base.shape[0]/2)), interpolation=cv2.INTER_NEAREST)
    return np.array(images)

def visualize_pyramid(gaus_images, layer):
    """
    This method is used to visualise a layer from the image pyramid
    """
    images = gaus_images[layer]
    fig = plt.figure()
    if len(images) != 6:
        print('[ERROR] Number of images != 6; they are equal to == {}'.format(len(images)))
        return None
    for i in range(len(images)):
        fig.add_subplot(2, 3, i + 1)
        plt.title('{} image'.format(i + 1, layer))
        plt.imshow(images[i], 'gray')
        plt.xticks([]);plt.yticks([])
    plt.show()
    return None

def main(image, sigma=1.6, num_intervals=3, initial_assumed_blur=0.5, image_border_width=5, verbose=True):
    """
    This method is used for generating SIFT keypoints and detectors.
    All the values (sigma, num_intervals, initial_assumed_blur) are used from the paper by Lowe.
    """    
    image = image.astype('float32')
    base_image = get_base_image(image, sigma, initial_assumed_blur, verbose=verbose)
    octaves = get_nos_octaves(base_image.shape)
    kernels = get_Gaussian_kernels(sigma, num_intervals, verbose=verbose)
    gaus_images = get_Gaussian_images(image, octaves, kernels, verbose=verbose)
    return kernels, gaus_images

if __name__ == "__main__":
    image = cv2.imread('/Users/siddhantbansal/Desktop/IIIT-H/Courses/DIP/Project/project-dipsum/images/database/bira_blonde.jpg', 0)
    kernels, gaus_images = main(image, verbose=True)
    for i in range(len(gaus_images)):
        visualize_pyramid(gaus_images, i)
