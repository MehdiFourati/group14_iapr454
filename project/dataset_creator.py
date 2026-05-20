import cv2
import os
from utils import *


def rotate_image(image, angle):
    """
    Rotate image without cropping.

    Parameters
    ----------
    image : np.ndarray
    angle : float
        Rotation angle in degrees.
        Positive = counter-clockwise

    Returns
    -------
    rotated : np.ndarray
    """

    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    # Rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute new bounding dimensions
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Rotate
    rotated = cv2.warpAffine(image, M, (new_w, new_h))

    return rotated

def manual_filter(descriptors, contours, card_type):
    descriptors_out, contours_out = [], []
    if card_type == 'one':
        for idx, descriptor in enumerate(descriptors):
            if descriptor[0] < 1700 and descriptor[0] > 1200:
                descriptors_out.append(descriptor)
                contours_out.append(contours[idx])
                
    if card_type == 'zero':
        for idx, descriptor in enumerate(descriptors):
            if descriptor[0] < 2300 and descriptor[0] > 2000:
                descriptors_out.append(descriptor)
                contours_out.append(contours[idx])
    
    if card_type == 'two':
        for idx, descriptor in enumerate(descriptors):
            if descriptor[0] > 1300 and descriptor[3]<200:
                descriptors_out.append(descriptor)
                contours_out.append(contours[idx])
                
    if card_type == 'three':
        for idx, descriptor in enumerate(descriptors):
            if descriptor[0] > 1350 and descriptor[0] < 1550:
                descriptors_out.append(descriptor)
                contours_out.append(contours[idx])
                
    if card_type == 'four':
        for idx, descriptor in enumerate(descriptors):
            if descriptor[0] > 1600 and descriptor[0] < 1900:
                descriptors_out.append(descriptor)
                contours_out.append(contours[idx])
                
    if card_type == 'five':
        for idx, descriptor in enumerate(descriptors):
            if descriptor[0] > 1300 and descriptor[0] < 1550:
                descriptors_out.append(descriptor)
                contours_out.append(contours[idx])
                
    if card_type == 'six-nine':
        for idx, descriptor in enumerate(descriptors):
            if descriptor[1] > 750 and descriptor[1] < 1050:
                descriptors_out.append(descriptor)
                contours_out.append(contours[idx])
    
    if card_type == 'seven':
        for idx, descriptor in enumerate(descriptors):
            if descriptor[0] > 1400 and descriptor[0] < 1750 and descriptor[1]>200 and descriptor[1]<600:
                descriptors_out.append(descriptor)
                contours_out.append(contours[idx])
    
    if card_type == 'eight':
        for idx, descriptor in enumerate(descriptors):
            if descriptor[0] > 1900 and descriptor[1] < 2200 and descriptor[0] < 2500:
                descriptors_out.append(descriptor)
                contours_out.append(contours[idx])
                
    if card_type == 'nine':
        for idx, descriptor in enumerate(descriptors):
            if descriptor[1] > 750 and descriptor[1] < 1050:
                descriptors_out.append(descriptor)
                contours_out.append(contours[idx])
                
    if card_type == 'reverse':
        for idx, descriptor in enumerate(descriptors):
            if descriptor[0] > 1050 and descriptor[0] < 1300:
                descriptors_out.append(descriptor)
                contours_out.append(contours[idx])
                
    if card_type == 'skip':
        for idx, descriptor in enumerate(descriptors):
            if descriptor[0] > 2400 and descriptor[0] < 2600:
                descriptors_out.append(descriptor)
                contours_out.append(contours[idx])
                
    if card_type == 'plus':
        for idx, descriptor in enumerate(descriptors):
            if descriptor[2] < 150 and descriptor[0] > 1000 and descriptor[0] < 1500 and descriptor[1] < 200:
                descriptors_out.append(descriptor)
                contours_out.append(contours[idx])
                
    if card_type == 'wild':
        for idx, descriptor in enumerate(descriptors):
            if descriptor[0] > 400 and descriptor[0] < 650:
                descriptors_out.append(descriptor)
                contours_out.append(contours[idx])
    
    
    return descriptors_out, contours_out

def save_dataset_images(dataset, folder="saved_cards"):

    os.makedirs(folder, exist_ok=True)

    for _, images in dataset.items():

        for name, img in images:

            path = os.path.join(folder, f"{name}.png")

            if len(img.shape) == 3:
                img_to_save = cv2.cvtColor(
                    img,
                    cv2.COLOR_RGB2BGR
                )
            else:
                img_to_save = img

            cv2.imwrite(path, img_to_save)

def build_rotated_descriptor_dataset(images, label, n_rotations=11, angle_step=18):

    rotated_images = [[] for _ in images]
    contours_all = [[] for _ in images]
    descriptors_all = [[] for _ in images]

    if label == 'plus4' or label == 'plus2':
        label = 'plus'
    print(label)
    for img_idx, img in enumerate(images):

        for rot_idx in range(n_rotations):

            angle = rot_idx * angle_step
            rotated = rotate_image(img, angle=angle)
            gray = card_only_filter_blank(rotated)
            valid_contours = detect_number_contours(gray)
            descriptors, _, _ = fourier_descriptors(valid_contours)
            rotated_images[img_idx].append(rotated)
            segmented_descriptors, segmented_contours = manual_filter(descriptors, valid_contours, label)
            contours_all[img_idx].append(segmented_contours)
            descriptors_all[img_idx].extend(segmented_descriptors[:2])

    return (
        rotated_images,
        contours_all,
        np.array(descriptors_all)
    )