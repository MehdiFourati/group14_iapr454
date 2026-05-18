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


def card_only_filter_blank(img):
    return ((img[:,:,0] < 180) |
            (img[:,:,1] < 180) |
            (img[:,:,2] < 180))
    
    
def card_only_filter_leaf(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_filtered = apply_rgb_threshold(rgb, r_range=(190,255), g_range=(190,255), b_range=(190,255))
    hsv_filtered = (hsv[:,:,0]<30) | ((hsv[:,:,0]<170)&(hsv[:,:,0]>160))
    return rgb_filtered & hsv_filtered


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

    return "p"+str(player)


def detect_card_contours(img, min_area=100000, max_area=250000):
    """
    Args:
        img: Input RGB image
        min_area: Minimum contour area (filter out noise)
        max_area: Maximum contour area (filter out large regions)
        
    Returns:
        List of valid card contours and the edge map
    """
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Apply dilation to connect nearby edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area and shape
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by area
        if area < min_area or area > max_area:
            continue
        
        # Get the convex hull
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        # Calculate solidity
        if hull_area > 0:
            solidity = float(area) / hull_area
            if solidity < 0.6: 
                continue
        
        # Get bounding rect and check aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        if w == 0 or h == 0:
            continue
        
        aspect_ratio = float(w) / h
        # Uno cards have aspect ratio around 0.6-1.6 
        if aspect_ratio < 0.4 or aspect_ratio > 2.0:
            continue
        
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Cards should have at least 4 corners
        if len(approx) >= 4:
            valid_contours.append(contour)
    
    return valid_contours, edges, dilated


def detect_number_contours(img, dilation_kernel = 2, min_area=1000, max_area=2500, thr_solidity=0.8, corners=8):
    """
    Args:
        img: Input RGB image
        dilation_kernel: Kernel size for dilation
        min_area: Minimum contour area (filter out noise)
        max_area: Maximum contour area (filter out large regions)
        
    Returns:
        List of valid card contours, edges, dilated image
    """
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Apply dilation to connect nearby edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel, dilation_kernel))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area and shape
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by area
        if area < min_area or area > max_area:
            continue
        
        # # Get the convex hull
        # hull = cv2.convexHull(contour)
        # hull_area = cv2.contourArea(hull)
        
        # # Calculate solidity
        # if hull_area > 0:
        #     solidity = float(area) / hull_area
        #     if solidity < thr_solidity: 
        #         continue
        
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) >= corners:
            valid_contours.append(contour)
    
    return valid_contours, edges, dilated


def filter_contours_by_distance(contours, min_distance=50, target_distance=480, distance_tolerance=50):
    """
    Args:
        contours : List of contours to filter
        min_distance : Minimum allowed distance between contours. Contours closer than this are merged.
        target_distance : Target distance. Contours at approximately this distance are removed (one of the pair).
        distance_tolerance : Tolerance around target_distance (e.g., 480 ± 50 means 430-530).
    
    Returns:
        filtered_contours : List of remaining contours after filtering/merging
    """
    if len(contours) <= 1:
        return contours
    
    # Calculate centroids
    centroids = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))
        else:
            centroids.append((0, 0))
    
    # Track which contours to merge and which to remove
    to_merge = {}  # Maps index to list of indices to merge with
    to_remove = set()
    
    # Check all pairs of contours
    for i in range(len(centroids)):
        if i in to_remove or i in to_merge:
            continue
            
        for j in range(i + 1, len(centroids)):
            if j in to_remove or j in to_merge:
                continue
            
            # Calculate distance between centroids
            dist = np.sqrt((centroids[i][0] - centroids[j][0])**2 + 
                          (centroids[i][1] - centroids[j][1])**2)
            
            # Merge if too close
            if dist < min_distance:
                if i not in to_merge:
                    to_merge[i] = [i]
                to_merge[i].append(j)
                to_remove.add(j)
            
            # Remove if at target distance (within tolerance)
            elif abs(dist - target_distance) <= distance_tolerance:
                to_remove.add(j)  # Remove the second one
    
    # Merge contours that are too close
    filtered_contours = []
    for i in range(len(contours)):
        if i in to_remove and i not in to_merge:
            continue
        
        if i in to_merge:
            # Merge this contour with others
            merged_points = []
            for idx in to_merge[i]:
                merged_points.extend(contours[idx].reshape(-1, 2))
            
            # Find convex hull of merged points
            merged_array = np.array(merged_points, dtype=np.int32)
            merged_contour = cv2.convexHull(merged_array)
            filtered_contours.append(merged_contour)
        else:
            filtered_contours.append(contours[i])
    
    return filtered_contours


def extract_and_normalize_card(img_rgb, contour, card_width=350, card_height=540):
    """
    Args:
        img_rgb: Input RGB image
        card_width: Target card width in pixels
        card_height: Target card height in pixels
        
    Returns:
        Normalized card image
    """
    # Get the four corners of the contour
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    
    # Calculate all pairwise distances between corners
    distances = {}
    for i in range(4):
        for j in range(i+1, 4):
            dist = np.linalg.norm(box[i] - box[j])
            distances[(i, j)] = dist
    
    # The 4 shortest distances are the sides, the 2 longest are diagonals
    sorted_dists = sorted(distances.items(), key=lambda x: x[1])
    
    # Extract which corners are connected by edges
    edge_pairs = set()
    for (i, j), dist in sorted_dists[:4]: 
        edge_pairs.add(((min(i, j), max(i, j)), dist))
    
    # Build adjacency list
    adj = {0: [], 1: [], 2: [], 3: []}
    for (i, j), dist in edge_pairs:
        adj[i].append((j, dist))
        adj[j].append((i, dist))
    
    # Traverse the rectangle edges to get correct ordering
    ordered = [0]
    prev = -1
    current = 0
    
    while len(ordered) < 4:
        next_corner = None
        min_dist = 10000000
        for neighbor, dist in adj[current]:
            if dist < min_dist and neighbor != prev:
                min_dist = dist
                next_corner = neighbor
        
        if next_corner is None:
            break
        
        ordered.append(next_corner)
        prev = current
        current = next_corner
    
    # Check orientation using the shoelace formula (signed area)
    # Positive = counter-clockwise, Negative = clockwise
    ordered_points = np.array([box[i] for i in ordered])
    signed_area = 0
    for i in range(len(ordered_points)):
        p1 = ordered_points[i]
        p2 = ordered_points[(i + 1) % len(ordered_points)]
        signed_area += (p2[0] - p1[0]) * (p2[1] + p1[1])
    
    if signed_area > 0:
        ordered = ordered[::-1]
    
    # Define source points in the correct order
    src_points = np.float32([box[i] for i in ordered]) # pyright: ignore[reportArgumentType]
    
    # Define destination points
    dst_points = np.float32([[0, 0],[card_width, 0],[card_width, card_height],[0, card_height]]) # pyright: ignore[reportArgumentType]
    
    # Get perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)  # type: ignore
    
    # Apply perspective transformation
    normalized_card = cv2.warpPerspective(img_rgb, matrix, (card_width, card_height))
    
    return normalized_card


def classify_card_color(img_rgb):
    """    
    Args:
        img_rgb: Input RGB image
        
    Returns:
        Detected color ('r', 'y', 'g', 'b', 'black')
    """    
    # Sample the mean color from the card's interior
    mean_rgb = cv2.mean(img_rgb)[:3]
    
    if np.argmax(mean_rgb) == 0 and mean_rgb[0] - mean_rgb[1] > 25 and np.mean(mean_rgb) > 100:
        return 'r'
    
    elif np.argmax(mean_rgb) == 0 and mean_rgb[0] - mean_rgb[1] < 25 and np.mean(mean_rgb) > 100:
        return 'y'
    
    elif np.argmax(mean_rgb) == 1 and np.mean(mean_rgb) > 100:
        return 'g'
    
    elif np.argmax(mean_rgb) == 2 and np.mean(mean_rgb) > 100:
        return 'b'
    
    else:
        return None
    

def classify_card_number(img):
    """
    Placeholder
    """
    return "PLACEHOLDER"


def find_central_card(img):
    """    
    Args:
        img: Input RGB image
        
    Returns:
        Detected card (final format)
    """
    img_cropped = img[700:1900, 900:3100, :]

    mean_value = img_cropped.mean()

    if mean_value > 200:
        hsv = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2HSV)
    
        # Extract saturation channel (colored regions have high saturation)
        saturation = hsv[:, :, 1]

        # Threshold to find colored pixels
        _, saturation_mask = cv2.threshold(saturation, 50, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        saturation_mask = cv2.morphologyEx(saturation_mask, cv2.MORPH_CLOSE, kernel)
        saturation_mask = cv2.morphologyEx(saturation_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100))
        saturation_mask = cv2.morphologyEx(saturation_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Convert back to RGB for compatibility
        img_for_contours = cv2.cvtColor(saturation_mask, cv2.COLOR_GRAY2RGB)
    else:
        img_for_contours = card_only_filter_leaf(cv2.cvtColor(img_cropped, cv2.COLOR_RGB2BGR))
        img_for_contours = remove_small_objects(img_for_contours, max_size=300)
        img_for_contours = remove_small_holes(img_for_contours, max_size=20)
        img_for_contours = cv2.cvtColor(img_for_contours.astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)

    valid_contours, _, _ = detect_card_contours(img_for_contours, 100000, 200000)
    if len(valid_contours) > 0:
        normalized = extract_and_normalize_card(img_cropped, valid_contours[0])
        color = classify_card_color(normalized)
        number = classify_card_number(normalized)

        if color:
            return color+"_"+number
        else:
            return number
        
    else:
        return None


def find_cards_per_player(img):

    org_img = img

    mean_value = img.mean()

    if mean_value > 200:

        img = card_only_filter_blank(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        img = remove_small_holes(img, max_size=20)
        img = remove_small_objects(img, max_size=10000)
        img = cv2.cvtColor(img.astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)

    else:

        img = card_only_filter_leaf(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        img = remove_small_objects(img, max_size=300)
        img = remove_small_holes(img, max_size=20)
        img = cv2.cvtColor(img.astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)

    players = {"p1": 1, "p2": 2, "p3": 3, "p4": 4}

    cards = {1:[],2:[],3:[],4:[]}
        
    valid_contours, _, _ = detect_number_contours(img, 3, 600, 1800, 0.7, 4)
    
    for contour in valid_contours:
        normalized = extract_and_normalize_card(org_img, contour)

    classified_contours = classify_contours(valid_contours)
    filtered_contours = filter_contours_by_distance(valid_contours, min_distance=80, target_distance=480, distance_tolerance=20)

    for contour, classification in filtered_contours:
        cards[detect_player(contour)].append(classification)
    
    return cards