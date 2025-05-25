import functools
import math
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from scipy.ndimage import interpolation as inter

# Configure logger


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def edge_detection_using_canny_algo(img, t_lower=50, t_upper=50):
    # Setting parameter
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_blur = cv2.bilateralFilter(img_blur, 9, 75, 75)

    edge = cv2.Canny(img_blur, t_lower, t_upper, 3)
    edge = cv2.dilate(edge, None, iterations=2)
    return edge


def find_rectangle_corners(points):
    """
    Given a list of (x, y) points that lie along the boundary of a rectangle,
    use OpenCV to find the minimum-area bounding rectangle and return its corners.
    """
    # Convert the list to a NumPy array of type float32.
    pts = np.array(points, dtype=np.float32)

    # Compute the minimum-area bounding rectangle.
    # minAreaRect returns a tuple: ((center_x, center_y), (width, height), angle)
    rect = cv2.minAreaRect(pts)

    # Retrieve the four corner points of the rectangle.
    # boxPoints returns the four vertices of the rectangle.
    box = cv2.boxPoints(rect)

    # Convert the corner coordinates to integers (optional).
    box = np.int64(box)

    # Convert to a list of tuples for easier use.
    corners = [tuple(pt) for pt in box]
    return box, corners


def get_edge_list(edge):
    edge_list = []
    for x in range(len(edge)):
        for y in range(len(edge[0])):
            if edge[x][y] != 0:
                edge_list.append((x, y))
    return edge_list


def create_edge_lines(edge, corners):
    for corner in corners:
        print(f"Corner: {corner}")
        edge[corner[0]] = [255 for i in range(len(edge[0]))]
        y_value = corner[1]
        for x1 in range(len(edge)):
            for y1 in range(len(edge[0])):
                if y1 == y_value:
                    edge[x1][y1] = 255
    return edge


# There are potential of two orientations of the rectangle
# 1. Where the strip is pointing towards north-east
# 2. Where the strip is pointing towards north-west
def detect_strip_direction(edge_list):
    # Edge List should be sorted based on the x coordinate
    # 0 = north east
    # 1 = north west
    print(f"First two edges: {edge_list[0]}, {edge_list[1]}")
    if edge_list[0][1] >= edge_list[1][1]:
        return 0
    return 1


def detect_corners(edge_list, direction=None):  # Format tl, tr, br, bl
    if direction == 0:
        tl = edge_list[0]
        bl = edge_list[1]
        tr = edge_list[2]
        br = edge_list[3]
    elif direction == 1:
        bl = edge_list[0]
        tl = edge_list[1]
        br = edge_list[2]
        tr = edge_list[3]
    else:
        if edge_list[0][1] > edge_list[1][1]:
            tl = edge_list[0]
            bl = edge_list[1]
        else:
            tl = edge_list[1]
            bl = edge_list[0]
        if edge_list[2][1] > edge_list[3][1]:
            tr = edge_list[2]
            br = edge_list[3]
        else:
            tr = edge_list[3]
            br = edge_list[2]

    return tl, tr, br, bl


def get_angle_of_rotation(tl, tr, br, bl):
    theta = math.degrees(math.atan((tr[1] - tl[1]) / (tr[0] - tl[0])))
    print(f"Angle of rotation: {theta}")
    return theta


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def crop_rect(img, rect):
    # get the parameter of the small rectangle
    center = rect[0]
    size = rect[1]
    angle = rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    rows, cols = img.shape[0], img.shape[1]

    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (cols, rows))
    out = cv2.getRectSubPix(img_rot, size, center)
    return out, img_rot


def crop_image(img, tl, tr, br, bl):
    width = tr[0] - tl[0]
    height = tl[1] - bl[1]
    print(f"Height: {height}, Width: {width}")
    x = bl[0]
    y = bl[1]
    cropped_img = img[x : tr[0], y : tr[1]]
    return cropped_img


def determine_score(arr, angle):
    """Rotate arr by `angle` and return the sum-of-squared-differences in row sums."""
    data = inter.rotate(arr, angle, reshape=False, order=0)
    histogram = np.sum(data, axis=1, dtype=float)
    score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
    return (histogram, score)


def correct_skew_parallel(image, delta=1, limit=60):
    # Convert to gray and threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Build list of candidate angles
    angles = np.arange(-limit, limit + delta, delta)

    # Run determine_score in parallel over all angles
    with ProcessPoolExecutor() as executor:
        # We only need the 'score' for deciding best angle, so store them
        results = list(executor.map(functools.partial(determine_score, thresh), angles))

    # Extract the scores, find angle with max score
    scores = [score for (_, score) in results]
    best_angle = angles[np.argmax(scores)]

    # Correct the image using best_angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    return best_angle, corrected


def normalise_color(img, b_color_norm, g_color_norm, r_color_norm):
    img_copy = img.copy()
    img_copy[:, :, 0] = np.clip(img_copy[:, :, 0] * b_color_norm, 0, 255)
    img_copy[:, :, 1] = np.clip(img_copy[:, :, 1] * g_color_norm, 0, 255)
    img_copy[:, :, 2] = np.clip(img_copy[:, :, 2] * r_color_norm, 0, 255)
    return img_copy


def detect_background_color(actual_color, image, tl, tr, bl, br):
    img = image.copy()
    img[bl[0] : tr[0], bl[1] : tr[1]] = [255, 255, 255]
    print("Applied white strip to image")

    b_color = [
        img[h][w][0]
        for h in range(img.shape[0])
        for w in range(img.shape[1])
        if list(img[h][w]) != [255, 255, 255]
    ]
    g_color = [
        img[h][w][1]
        for h in range(img.shape[0])
        for w in range(img.shape[1])
        if list(img[h][w]) != [255, 255, 255]
    ]
    r_color = [
        img[h][w][2]
        for h in range(img.shape[0])
        for w in range(img.shape[1])
        if list(img[h][w]) != [255, 255, 255]
    ]

    actual_color = [1 if x == 0 else x for x in actual_color]
    b_color_norm = np.average(b_color) / actual_color[0]
    g_color_norm = np.average(g_color) / actual_color[1]
    r_color_norm = np.average(r_color) / actual_color[2]

    print(f"Normalisation Ratio: {b_color_norm}, {g_color_norm}, {r_color_norm}")
    return (b_color_norm, g_color_norm, r_color_norm)


def detect_blister(edge_image, actual_image):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(
        edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    bbox = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if bbox is not None:
        x, y, w, h = bbox
        cropped_image = actual_image[y : y + h, x : x + w]
        # cv2.rectangle(actual_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        return cropped_image

    return None


def correct_image_skew(image, max_angle=45):
    """
    Corrects the skew of an image by detecting the largest contour and rotating it.

    Args:
        image: Input image (numpy array).
        max_angle: Maximum allowed angle for skew correction (default: 45 degrees).
    Returns:
        Corrected image.
    """
    try:
        # Convert to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply thresholding to create a binary image
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours in the binary image
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return image  # Return the original image if no contours are found

        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the minimum area rectangle bounding the largest contour
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[-1]  # Get the angle of rotation

        # Adjust the angle to ensure it's within the allowed range
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90

        # Only correct the skew if the angle is significant
        if abs(angle) > 1:  # Skip correction for small angles
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)

            # Compute the rotation matrix
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Apply the rotation to the image
            corrected_image = cv2.warpAffine(
                image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
            return corrected_image
        else:
            return image  # Return the original image if the angle is too small

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return image  # Return the original image in case of an error


def cropped_image(img_data):
    print("Cropping Image")
    image = img_data.copy()

    # Detect edges
    edges = edge_detection_using_canny_algo(image, t_lower=100, t_upper=300)
    edge_list = get_edge_list(edges)

    # Find rectangle corners
    box, corners = find_rectangle_corners(edge_list)
    edge_list_sorted = sorted(corners, key=lambda x: x[0])
    print(f"Edge List: {edge_list_sorted}")

    # Detect strip direction
    strip_direction = detect_strip_direction(edge_list_sorted)
    print(f"Strip Direction: {strip_direction}")
    tl, tr, br, bl = detect_corners(edge_list_sorted, strip_direction)

    # Calculate rotation angle
    angle = get_angle_of_rotation(tl, tr, br, bl)
    print("Angle for the master image to be rotated is: " + str(angle))
    angle = -angle  # Adjust angle if needed

    # Rotate the image
    new_img = rotate_image(image, angle)

    # Detect blister and correct skew
    new_edge = edge_detection_using_canny_algo(new_img, t_lower=50, t_upper=150)
    cropped_img = detect_blister(new_edge, new_img)
    corrected_img = correct_image_skew(cropped_img)

    return corrected_img
