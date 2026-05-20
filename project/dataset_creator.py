import cv2
import os
from glob import glob
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

def make_reference_group(reference_path):

    references = []
    for path in glob(os.path.join(reference_path, "*")):
        img = cv2.imread(path, cv2.IMREAD_COLOR_RGB)

        if img is not None:
            references.append(img)
            
    r0 = references[0]

    one_two_three = r0[600:2490, 1250:2960]

    one   = one_two_three[1240:]
    two   = one_two_three[640:1270]
    three = one_two_three[10:650]

    one_yellow, one_red, one_blue, one_green = one[20:,60:400], one[40:,490:840], one[20:-20,940:1290], one[20:-20,1350:-10]
    two_yellow, two_red, two_blue, two_green = two[10:,50:390], two[10:,470:820], two[:-20,900:1250], two[:-20,1290:-10]
    three_yellow, three_red, three_blue, three_green = three[20:,20:380], three[20:,430:850], three[20:-10,890:1240], three[:-20,1310:-30]

    r1 = references[1]

    four_five_six = r1[550:2470,1470:3400]

    four = four_five_six[1280:]
    five = four_five_six[740:1320]
    six  = four_five_six[20:700]

    four_yellow, four_red, four_blue, four_green = four[40:-20,160:560], four[30:-10,560:950], four[30:-10,1050:1450], four[60:-10,1530:-10]
    five_yellow, five_red, five_blue, five_green = five[20:-20,100:460], five[10:-20,550:910], five[10:,1010:1400], five[10:-20,1500:-40]
    six_yellow, six_red, six_blue, six_green = six[90:-20,50:420], six[70:-20,490:890], six[60:-60,1000:1360], six[:-70,1450:-70]

    r2 = references[2]

    seven_eight_nine = r2[600:2550,1700:3680]

    seven = seven_eight_nine[1300:]
    eight = seven_eight_nine[720:1300]
    nine  = seven_eight_nine[30:680]

    wild     = r2[1760:2350,1200:1600]
    plus4    = r2[880:1520, 1130:1530]

    seven_yellow, seven_red, seven_blue, seven_green = seven[90:-20,170:550], seven[70:-30,630:1000], seven[30:-20,1040:1450], seven[30:-70,1550:-60]
    eight_yellow, eight_red, eight_blue, eight_green = eight[40:,110:470], eight[20:-20,560:920], eight[30:-10,1020:1380], eight[10:-20,1530:-90]
    nine_yellow, nine_red, nine_blue, nine_green = nine[60:-30,40:400], nine[:-50,520:950], nine[40:,1040:1470], nine[:-60,1550:-10]

    r3 = references[3]

    plus2s   = r3[1700:2300, 100:1800]
    reverses = r3[170:920, 1680:3500]
    skips    = r3[880:1630, 1730:3670]
    zeros    = r3[1570:2270, 1870:3680]

    plus2_yellow, plus2_blue, plus2_red, plus2_green = plus2s[:,:400], plus2s[:,450:850], plus2s[:,850:1250], plus2s[:,1250:]
    reverse_yellow, reverse_red, reverse_blue, reverse_green = reverses[150:,:400], reverses[100:-30,470:870], reverses[:650,930:1400], reverses[40:650,1420:]
    skip_yellow, skip_red, skip_blue, skip_green = skips[140:,50:480], skips[120:-30,470:880], skips[50:700,890:1350], skips[:650,1430:1890]
    zero_yellow, zero_red, zero_blue, zero_green = zeros[50:-10,:430], zeros[40:-50,450:870], zeros[10:630,910:1330], zeros[:620,1350:1790]

    image_groups = {
        "zero": [zero_blue, zero_green, zero_red, zero_yellow],
        "one": [one_blue, one_green, one_red, one_yellow],
        "two": [two_blue, two_green, two_red, two_yellow],
        "three": [three_blue, three_green, three_red, three_yellow],
        "four": [four_blue, four_green, four_red, four_yellow],
        "five": [five_blue, five_green, five_red, five_yellow],
        "six-nine": [six_blue, six_green, six_red, six_yellow, nine_green, nine_red, nine_yellow],
        "seven": [seven_blue, seven_green, seven_red, seven_yellow],
        "eight": [eight_blue, eight_green, eight_red, eight_yellow],
        "reverse": [reverse_blue, reverse_green, reverse_red, reverse_yellow],
        "skip": [skip_blue, skip_green, skip_red, skip_yellow],
        "plus": [plus2_blue, plus2_green, plus2_red, plus2_yellow, plus4],
        "wild": [wild],
    }
    
    return image_groups

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