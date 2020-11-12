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

def get_DoG_images(gaus_images, verbose=True):
    """
    This method is used for generating the Difference of Gaussian (DoG) Pyramid
    """
    if verbose:
        print('[INFO] Preparing Difference of Gaussian images')
    images = list()
    for octave_images in gaus_images:
        images_octave = list()
        for orig_img, follow_img in zip(octave_images, octave_images[1:]):
            images_octave.append(np.subtract(follow_img, orig_img))
        images.append(images_octave)
    return np.array(images)

def get_scale_space_keypoints(gaus_images, dog_images, num_intervals, sigma, image_border_width, contrast_threshold=0.04, verbose=True):
    """
    This method is used to generate the keypoints using the Gaussian blurred and Laplacian images
    Here we are checking the neighbouring pixels for classifying the current pixel as extrema or not
    """
    if verbose:
        print('[INFO] Getting extrema in scale space')
    thresh = np.floor(0.5 * contrast_threshold / num_intervals * 255)  # Using OpenCV's approach
    keypoints = list()
    for oct_i, dog_imgs_oct in enumerate(dog_images):
        for img_i, (first, second, third) in enumerate(zip(dog_imgs_oct, dog_imgs_oct[1:], dog_imgs_oct[2:])):
            for i in range(image_border_width, first.shape[0] - image_border_width):
                for j in range(image_border_width, first.shape[1] - image_border_width):
                    if is_extremum(first[i-1:i+2, j-1:j+2], second[i-1:i+2, j-1:j+2], third[i-1:i+2, j-1:j+2], thresh):
                        localization = localize_quadratic_fit(i, j, img_i + 1, oct_i, num_intervals, dog_imgs_oct, sigma, contrast_threshold, image_border_width)
                        if localization is not None:
                            keypoint, localized_i = localization
                            keypoints_oriented = get_keypoint_orientation(keypoint, oct_i, gaus_images[oct_i][localized_i])
                            for keypoint_oriented in keypoints_oriented:
                                keypoints.append(keypoint_oriented)
    return keypoints

def is_extremum(first_section, second_section, third_section, thresh):
    """
    This method determines if the point under consideration is an extrema or not for the given neighborhood
    """
    main_value = second_section[1, 1]
    if abs(main_value) > thresh:
        if main_value > 0:
            result = main_value >= second_section[1, 2] and np.all(main_value >= third_section) and np.all(main_value >= second_section[0, :]) and np.all(main_value >= second_section[2, :]) and main_value >= second_section[1, 0] and np.all(main_value >= first_section)
            return result
        elif main_value < 0:
            result = main_value <= second_section[1, 2] and np.all(main_value <= third_section) and np.all(main_value <= second_section[0, :]) and np.all(main_value <= second_section[2, :]) and main_value <= second_section[1, 0] and np.all(main_value <= first_section)
            return result
    return False

def localize_quadratic_fit(i, j, img_i, oct_i, num_intervals, dog_imgs_oct, sigma, contrast_threshold, image_border_width, eigen_ratio=10, num_attempts=5, verbose=True):
    """
    This method is used for fitting a quadratic curve to the extrema obtained using the state space.
    This is an iterative method and uses the extrema's neighbours for fitting the curve.
    This helps in refining and improving the position of the extrema.
    """
    if verbose:
        print('[INFO] Fitting the localization curve on state-space extrema')
    extrema_outside = False
    for attempt in range(num_attempts):
        first, second, third = dog_imgs_oct[img_i-1:img_i+2]
        cube = np.stack([first[i-1:i+2, j-1:j+2], second[i-1:i+2, j-1:j+2], third[i-1:i+2, j-1:j+2]]).astype('float32')/255 # rescaling pixel values to [0, 1] in the neighbourhood under consideration
        grad = get_grad(cube)
        hess = get_hess(cube)
        new_extremum = -np.linalg.lstsq(hess, grad, rcond=None)[0]
        if abs(new_extremum[0]) < 0.5 and abs(new_extremum[1]) < 0.5 and abs(new_extremum[2]) < 0.5:
            break
        i += int(round(new_extremum[1]))
        j += int(round(new_extremum[0]))
        img_i += int(round(new_extremum[2]))
        if i < image_border_width or i >= dog_imgs_oct[0].shape[0] - image_border_width or j < image_border_width or j >= dog_imgs_oct[0].shape[1] - image_border_width or img_i < 1 or img_i > num_intervals:
            extrema_outside = True
            break
    if extrema_outside:
        return None
    if attempt >= num_attempts - 1:
        return None
    new_extremum_value = cube[1, 1, 1] + 0.5 * np.dot(grad, new_extremum)
    if abs(new_extremum_value) * num_intervals >= contrast_threshold:
        xy_hess = hess[:2, :2]
        xy_hess_trace = np.trace(xy_hess)
        xy_hess_det = np.linalg.det(xy_hess)
        if xy_hess_det > 0 and eigen_ratio * (xy_hess_trace ** 2) < ((eigen_ratio + 1)**2) * xy_hess_det:
            keypoint = cv2.KeyPoint()
            keypoint.pt = ((j + new_extremum[0]) * (2 ** oct_i), (i + new_extremum[1]) * (2 ** oct_i))
            keypoint.octave = oct_i + img_i * (2 ** 8) + int(round((new_extremum[2] + 0.5) * 255)) * (2 ** 16)
            keypoint.size = sigma * (2 ** ((img_i + new_extremum[2]) / np.float32(num_intervals))) * (2 ** (oct_i + 1))
            keypoint.response = abs(new_extremum_value)
            return keypoint, img_i
    return None

def get_grad(cube):
    ds = 0.5 * (cube[1, 2, 1] - cube[1, 0, 1])
    dx = 0.5 * (cube[1, 1, 2] - cube[1, 1, 0])
    dy = 0.5 * (cube[2, 1, 1] - cube[0, 1, 1])
    return np.array([dx, dy, ds])

def get_hess(cube):
    main_val = cube[1, 1, 1]
    dss = cube[2, 1, 1] - 2 * main_val + cube[0, 1, 1]
    dxx = cube[1, 1, 2] - 2 * main_val + cube[1, 1, 0]
    dyy = cube[1, 2, 1] - 2 * main_val + cube[1, 0, 1]
    dxs = 0.25 * (cube[2, 1, 2] - cube[2, 1, 0] - cube[0, 1, 2] + cube[0, 1, 0])
    dys = 0.25 * (cube[2, 2, 1] - cube[2, 0, 1] - cube[0, 2, 1] + cube[0, 0, 1])
    dxy = 0.25 * (cube[1, 2, 2] - cube[1, 2, 0] - cube[1, 0, 2] + cube[1, 0, 0])
    hess = np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
    return hess

def visualize_pyramid(gaus_images, layer):
    """
    This method is used to visualise a layer from the image pyramid
    """
    images = gaus_images[layer]
    fig = plt.figure()
    if len(images) != 6 and len(images) != 5:
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
    dog_images = get_DoG_images(gaus_images, verbose=verbose)
    keypoints = get_scale_space_keypoints(gaus_images, dog_images, num_intervals, sigma, image_border_width)
    return keypoints

if __name__ == "__main__":
    image = cv2.imread('/Users/siddhantbansal/Desktop/IIIT-H/Courses/DIP/Project/project-dipsum/images/database/bira_blonde.jpg')
    image_gray = cv2.imread('/Users/siddhantbansal/Desktop/IIIT-H/Courses/DIP/Project/project-dipsum/images/database/bira_blonde.jpg', 0)
    keypoints = main(image_gray, verbose=True)
    img=cv2.drawKeypoints(image_gray,keypoints,image)
    plt.imshow(img, 'gray')
    plt.show()
    # pdb.set_trace()
    # for i in range(len(dog_images)):
    #     visualize_pyramid(dog_images, i)
