import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from collections import Counter
from skimage.morphology import closing, opening, disk, remove_small_holes, remove_small_objects


class FourierDiscriminator:

    def __init__(
        self,
        n_neighbors=3,
        rejection_threshold=2.0
    ):

        self.scaler = StandardScaler()

        self.knn = KNeighborsClassifier(
            n_neighbors=n_neighbors
        )

        self.rejection_threshold = rejection_threshold

    def fit(
        self,
        datasets,
        test_size=0.2,
        random_state=42,
        verbose = False
    ):

        X, y = build_classifier_dataset(datasets)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.knn.fit(X_train, y_train)

        y_pred = self.knn.predict(X_test)
        if verbose:
            print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
            print(f"Macro F1 : {f1_score(y_test, y_pred, average='macro'):.4f}\n")
            print(classification_report(y_test, y_pred))

    def predict(self, descriptor):

        descriptor = np.asarray(descriptor).reshape(1, -1)
        descriptor = self.scaler.transform(descriptor)

        distances, _ = self.knn.kneighbors(descriptor)
        mean_distance = distances.mean()

        if mean_distance > self.rejection_threshold:
            return "None", mean_distance

        prediction = self.knn.predict(descriptor)[0]

        return prediction, mean_distance


def build_classifier_dataset(datasets):
    """
    Convert dictionary of descriptor datasets into X / y arrays.
    """

    X = []
    y = []

    for label, data in datasets.items():

        data = np.asarray(data)

        # flatten if needed
        data = data.reshape(data.shape[0], -1)

        X.append(data)
        y.extend([label] * len(data))

    X = np.vstack(X)
    y = np.array(y)

    return X, y


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
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    rgb = img
    rgb_filtered = apply_rgb_threshold(rgb, r_range=(200,255), g_range=(200,255), b_range=(200,255))
    hsv_filtered = (hsv[:,:,0]<30)
    return rgb_filtered & hsv_filtered


def find_active_player(img):
    return 'p1'
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


def detect_number_contours(gray, min_area=300, max_area=2200):
    """
    Args:
        img: Input RGB image
        min_area: Minimum contour area (filter out noise)
        max_area: Maximum contour area (filter out large regions)
        
    Returns:
        List of valid card contours, edges, dilated image
    """
    gray = remove_small_objects(gray.astype(np.uint8), min_size=max_area)
    gray = ~remove_small_objects(~gray.astype(np.uint8), min_size=min_area).astype(np.uint8)

    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # Filter by area
        if area < min_area or area > max_area:
            continue
        valid_contours.append(contour)
    
    return valid_contours


def filter_contours_by_distance(contours, min_distance=50, target_distance=480, distance_tolerance=50):
    """
    Args:
        contours : Tuple(List of contours to filter, List of classified values)
        min_distance : Minimum allowed distance between contours. Contours closer than this are merged.
        target_distance : Target distance. Contours at approximately this distance are removed (one of the pair).
        distance_tolerance : Tolerance around target_distance (e.g., 480 ± 50 means 430-530).
    
    Returns:
        filtered_contours : List of remaining contours after filtering/merging
    """
    if len(contours[0]) <= 1:
        return contours
    
    # Calculate centroids
    centroids = []
    for contour in contours[0]:
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
            elif abs(dist - target_distance) <= distance_tolerance and contours[1][i] == contours[1][j]:
                to_remove.add(j)  # Remove the second one
    
    # Merge contours that are too close
    filtered_contours = []
    filtered_classes = []
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
            filtered_classes.append(contours[1][i])
        else:
            filtered_contours.append(contours[0][i])
            filtered_classes.append(contours[1][i])
    
    return filtered_contours, filtered_classes


def extract_and_normalize_card(img_rgb, contour, card_width=350, card_height=540, extension=0):
    """
    Args:
        img_rgb: Input RGB image
        contour: Card contour
        card_width: Target card width in pixels
        card_height: Target card height in pixels
        extension: Extra pixels to extend the extracted area outward from card center
        
    Returns:
        Normalized card image
    """
    # Get the four corners of the contour
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)

    # Calculate center of the bounding box
    center = box.mean(axis=0)
    
    # Apply extension by moving each corner away from center
    if extension > 0:
        for i in range(4):
            direction = box[i] - center
            norm = np.linalg.norm(direction)
            if norm > 0:
                box[i] = box[i] + (direction / norm) * extension
    
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
        img_rgb: RGB normalized card image

    Returns:
        'r', 'y', 'g', 'b', 'black', or None
    """

    img = img_rgb.astype(np.uint8)

    valid_mask = card_only_filter_blank(img)
    valid_pixels = img[valid_mask]

    if len(valid_pixels) < 50:
        return None

    mean_rgb = valid_pixels.mean(axis=0)

    if mean_rgb.mean() < 60:
        return "black"

    dominant = np.argmax(mean_rgb)

    if dominant == 0:
        if abs(mean_rgb[0] - mean_rgb[1]) < 40:
            return "y"
        return "r"
    elif dominant == 1:
        return "g"
    elif dominant == 2:
        return "b"

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


def assign_card2player(center):

    top_l = lambda x: ((1800-2662)/(1050-500))*(x-500) + 2662
    top_r = lambda x: ((1800-2662)/(2800-3500))*(x-3500) + 2662
    bot_l = lambda x: ((900-0)/(1050-500))*(x-500)
    bot_r = lambda x: ((0-900)/(3500-2800))*(x-2800) + 900
    
    if center[1] < 900 and center[1] < bot_l(center[0]) and center[1] < bot_r(center[0]):
        return 'p3'
    elif center[0] < 1050 and center[1] > bot_l(center[0]) and center[1] < top_l(center[0]):
        return 'p4'
    elif center[1] > 1800 and center[1] > top_l(center[0]) and center[1] > top_r(center[0]):
        return 'p1'
    elif center[0] > 2800 and center[1] > bot_r(center[0]) and center[1] < top_r(center[0]):
        return 'p2'
    else:
        return 'center'


def fourier_descriptors(contours, n_samples=100, interpolation=True):
    """
    Compute translation and rotation invariant Fourier descriptors.
    
    Parameters
    ----------
    contours : list of (K,2) arrays
    n_samples : int
    interpolation : bool

    Returns
    -------
    descriptors : (N, n_samples-1)
        Translation + rotation invariant descriptors.

    centers : (N, 2)
        Contour centroids.

    rotations : (N,)
        Dominant contour orientation in radians.
    """

    descriptors = []
    centers = []
    rotations = []

    for contour in contours:

        contour = np.asarray(contour.squeeze(), dtype=np.float64)
        
        d = np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1))
        t = np.concatenate([[0], np.cumsum(d)])

        t_new = np.linspace(0, t[-1], n_samples)
        x = np.interp(t_new, t, contour[:, 0])
        y = np.interp(t_new, t, contour[:, 1])

        contour = np.stack([x, y], axis=1)
        
        center = contour.mean(axis=0)
        centers.append(center)

        z = contour[:, 0] + 1j * contour[:, 1]
        fd = np.fft.fft(z)
        fd[0] = 0

        rotation = np.angle(fd[1])
        rotations.append(rotation)
        fd *= np.exp(-1j * rotation)
        descriptors.append(np.abs(fd[1:]))

    return (
        np.array(descriptors),
        np.array(centers),
        np.array(rotations)
    )


def deduplicate_predictions(
    predictions,
    pair_distance_target=530
):
    """
    Keep only one prediction when:
    - same class
    - same color
    - same angle
    - AND centers are either:
        * extremely close (duplicate)
        * OR separated by ~480 +/- margin
    """

    kept = []

    for pred in predictions:

        keep = True

        center_a = np.array(pred["center"])
        angle_a = pred["angle"]
        class_a = pred["class"]
        color_a = pred["color"]

        for existing in kept:

            center_b = np.array(existing["center"])
            angle_b = existing["angle"]
            class_b = existing["class"]
            color_b = existing["color"]

            same_class = class_a == class_b
            same_color = color_a == color_b

            angle_diff = abs(angle_a - angle_b)
            angle_diff = min(angle_diff, 360 - angle_diff)

            #deprecated (does not work well)
            #same_angle = angle_diff < angle_thresh
            same_angle = True

            dist = np.linalg.norm(center_a - center_b)

            close_duplicate = (
                dist < pair_distance_target
            )
            
            if (
                same_class
                and same_color
                and same_angle
                and close_duplicate
            ):
                keep = False
                break

        if keep:
            kept.append(pred)

    return kept


def format_card(card_class, color):
    """
    Convert prediction into CSV card format.
    """
    mapping = {
        "plus2": "draw_2",
        "plus4": "draw_4",
        "reverse": "reverse",
        "skip": "skip",
        "wild": "wild",
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
    }

    if card_class in mapping:

        value = mapping[card_class]

        if card_class in ["wild", "plus4"]:
            return value

        return f"{color}_{value}"
    return f"{color}_{card_class}"


def classify_image(
    img,
    classifier,
    pair_distance_target=530,
    distance_threshold_plus=50
):
    
    mean_value = img.mean()

    if mean_value > 200:
        gray = card_only_filter_blank(img)
    else:
        gray = card_only_filter_leaf(img)
    contours = detect_number_contours(gray)
    descriptors, centers, angles = fourier_descriptors(contours)

    raw_predictions = []

    for idx, descriptor in enumerate(descriptors):

        descriptor = np.asarray(descriptor).flatten()
        card_type, dist = classifier.predict(descriptor)

        if card_type == "None":
            continue

        normalized_card = extract_and_normalize_card(
            img,
            contours[idx],
            extension=10
        )

        if card_type == "six-nine":
            # TODO: proper discrimination
            card_type = "six"

        color = classify_card_color(normalized_card)
        player = assign_card2player(centers[idx])

        if player is None:
            continue

        if color == "black" and card_type not in ["wild", "plus"]:
            continue

        raw_predictions.append({
            "class": card_type,
            "color": color,
            "mean_color": cv2.mean(normalized_card)[:3],
            "player": player,
            "center": centers[idx],
            "angle": angles[idx],
            "distance": dist
        })

    plus_cards = [
        p for p in raw_predictions
        if p["class"] == "plus"
    ]

    non_plus = [
        p for p in raw_predictions
        if p["class"] != "plus"
    ]

    final_predictions = []

    used_non_plus = set()

    for plus in plus_cards:

        plus_center = np.array(plus["center"])

        nearest = None
        nearest_dist = float("inf")

        nearest_type = None

        for i, p in enumerate(non_plus):

            if i in used_non_plus:
                continue
            if p["class"] not in ["two", "four"]:
                continue

            d = np.linalg.norm(
                np.array(p["center"]) - plus_center
            )
            print(d)
            if d < nearest_dist and d < distance_threshold_plus:
                nearest_dist = d
                nearest = i
                nearest_type = p["class"]

        plus_fixed = dict(plus)
        if nearest_type == "four":
            plus_fixed["class"] = "plus4"
        elif nearest_type == "two":
            plus_fixed["class"] = "plus2"

        #default
        else:
            plus_fixed["class"] = "plus2"

        final_predictions.append(plus_fixed)
        if nearest is not None:
            used_non_plus.add(nearest)

    for i, p in enumerate(non_plus):
        if i in used_non_plus:
            continue
        final_predictions.append(p)

    filtered_predictions = []
    for i, p in enumerate(final_predictions):

        if p["class"] != "zero":
            continue

        zero_center = np.array(p["center"])

        remove_zero = False

        for q in final_predictions:

            if q["class"] not in ["six", "eight", "nine"]:
                continue

            if q["color"] != p["color"]:
                continue

            d = np.linalg.norm(
                zero_center - np.array(q["center"])
            )

            if d < (2/3)*pair_distance_target:
                remove_zero = True
                break

        if not remove_zero:
            filtered_predictions.append(p)

    # keep everything that is NOT zero (including 6/8/9 always)
    for p in final_predictions:
        if p["class"] != "zero":
            filtered_predictions.append(p)

    final_predictions = filtered_predictions
    
    final_predictions = deduplicate_predictions(
        final_predictions,
        pair_distance_target=pair_distance_target
    )

    return final_predictions


def find_cards_per_player(img, classifier):

    cards = {"p1": [], "p2": [], "p3": [], "p4": [], "center": []}
        
    predictions = classify_image(img, classifier)

    for pred in predictions:
        player = pred["player"]
        color = pred["color"]
        number = pred["class"]

        string = format_card(number, color)

        cards[player].append(string)

    for key in cards:
        if len(cards[key]) == 0:
            cards[key] = ["EMPTY"]

    return predictions

def plot_contours(image, contours, color=(0, 255, 0), thickness=2):
    """
    Draw contours returned by cv2.findContours.

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or BGR).
    contours : list
        Contours returned by cv2.findContours.
    color : tuple
        Contour color in BGR.
    thickness : int
        Line thickness.
    """

    if len(image.shape) == 2:
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        output = image.copy()
        
    cv2.drawContours(output, contours, -1, color, thickness)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
    
def classify_folder(img_folder, classifier):

    result = pd.DataFrame(columns=["image_id","center_card","active_player","player_1_cards","player_2_cards","player_3_cards","player_4_cards"])
    img_folder = Path(img_folder)
    
    for path in img_folder.glob("*.jpg"):
        print(f'image: {path.stem}')
        original_img = cv2.imread(str(path))
        original_img = cv2.cvtColor(
            original_img,
            cv2.COLOR_BGR2RGB
        )
        row = []
        player_cards = find_cards_per_player(original_img, classifier)
        image_id = path.stem
        center_card = player_cards["center"][0]
        active_player = find_active_player(original_img)
        player_1_cards = ";".join(player_cards["p1"])
        player_2_cards = ";".join(player_cards["p2"])
        player_3_cards = ";".join(player_cards["p3"])
        player_4_cards = ";".join(player_cards["p4"])

        row.append(image_id)
        row.append(center_card)
        row.append(active_player)
        row.append(player_1_cards)
        row.append(player_2_cards)
        row.append(player_3_cards)
        row.append(player_4_cards)

        result.loc[len(result)] = row
    return result

def save_result(result):
    result.to_csv("submission.csv", index=False)