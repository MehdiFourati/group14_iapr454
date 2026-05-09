import cv2
import skimage
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.morphology import closing, opening, disk, remove_small_holes, remove_small_objects


def extract_rgb_channels(img):
    """
    Extract RGB channels from the input image.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.
    
    Return
    ------
    data_red: np.ndarray (M, N)
        Red channel of input image
    data_green: np.ndarray (M, N)
        Green channel of input image
    data_blue: np.ndarray (M, N)
        Blue channel of input image
    """

    # Get the shape of the input image
    M, N, _ = np.shape(img)

    # Define default values for RGB channels
    data_red = np.zeros((M, N))
    data_green = np.zeros((M, N))
    data_blue = np.zeros((M, N))

    # ------------------
    data_red = img[:,:,0]
    data_green = img[:,:,1]
    data_blue = img[:,:,2]
    # ------------------
    
    return data_red, data_green, data_blue


def apply_rgb_threshold(img, r_range=(0,130), g_range=(0,100), b_range=(0,150)):
    """
    Apply threshold to input image.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.
    
    Return
    ------
    img_th: np.ndarray (M, N)
        Thresholded image.
    """

    # Define the default value for the input image
    M, N, C = np.shape(img)
    img_th = np.zeros((M, N))

    # Use the previous function to extract RGB channels
    data_red, data_green, data_blue = extract_rgb_channels(img=img)
    
    # ------------------
    img_th = (
        (r_range[0] <= data_red) & (data_red <= r_range[1]) &
        (g_range[0] <= data_green) & (data_green <= g_range[1]) &
        (b_range[0] <= data_blue) & (data_blue <= b_range[1])
    )
    # ------------------
    
    return img_th


def apply_closing(img_th, disk_size):
    """
    Apply closing to input mask image using disk shape.

    Args
    ----
    img_th: np.ndarray (M, N)
        Image mask of size MxN.
    disk_size: int
        Size of the disk to use for opening

    Return
    ------
    img_closing: np.ndarray (M, N)
        Image after closing operation
    """

    # Define default value for output image
    img_closing = np.zeros_like(img_th)
    footprint = disk(disk_size)
    img_closing = closing(img_th, footprint)

    return img_closing


def apply_opening(img_th, disk_size):
    """
    Apply opening to input mask image using disk shape.

    Args
    ----
    img_th: np.ndarray (M, N)
        Image mask of size MxN.
    disk_size: int
        Size of the disk to use for opening

    Return
    ------
    img_opening: np.ndarray (M, N)
        Image after opening operation
    """

    # Define default value for output image
    img_opening = np.zeros_like(img_th)
    footprint = disk(disk_size)
    img_opening = opening(img_th, footprint)
    return img_opening


def find_active_player(img):
    mean_value = img.mean()
    
    n = []

    if mean_value > 200:
        img_filter = apply_rgb_threshold(img, (0,150), (0,150), (0,150))

    else:
        img_filter = apply_rgb_threshold(img, (200,255), (200,255), (0,80))

    img_1 = apply_opening(img_filter[1900:,500:3500], 20)
    img_2 = apply_opening(img_filter[:,3100:], 20)
    img_3 = apply_opening(img_filter[:900,500:3500], 20)
    img_4 = apply_opening(img_filter[:,:750], 20)

    n.append(sum(img_1[img_1 == True]))
    n.append(sum(img_2[img_2 == True]))
    n.append(sum(img_3[img_3 == True]))
    n.append(sum(img_4[img_4 == True]))

    player = np.argmax(n) + 1

    print(player)
    return player