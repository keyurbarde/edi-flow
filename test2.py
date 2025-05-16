import cv2
import numpy as np
from scipy.spatial import KDTree

# def detect_lines_and_circles(image):
#     # blur = cv2.GaussianBlur(image, (7, 7), 1)
#     # canny = cv2.Canny(image, threshold1=50, threshold2=150)

#     # kernel = np.ones((3, 3), np.uint8)
#     # dilated = cv2.dilate(canny, kernel, iterations=1)

#     contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#     result = image.copy()
#     cv2.drawContours(result, contours, -1, (255, 255, 255), 5)
#     print(contours)
#     return result


def detect_lines_and_circles(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(image.shape) == 2:
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        result = image.copy()

    print(f"Contours found: {len(contours)}")

    if contours:
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue

            rect = cv2.minAreaRect(cnt)
            (rx, ry), (rw, rh), angle = rect
            aspect_ratio = max(rw, rh) / min(rw, rh)

            if aspect_ratio > 8:
                # Line detected
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                cv2.drawContours(result, [box], -1, (0, 0, 255), 2)
                cv2.putText(result, "Line", (int(rx), int(ry)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                continue

            hull = cv2.convexHull(cnt)
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

            if len(approx) > 6:
                # Circle detected
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                cv2.circle(result, (int(cx), int(cy)), int(radius), (0, 255, 255), 2)
                cv2.putText(result, "Circle", (int(cx), int(cy) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


    return result


# Load and convert image to grayscale
image = cv2.imread("img/test4.jpg")  
blur = cv2.GaussianBlur(image, (7, 7), 1)

gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
# _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

inverted = cv2.bitwise_not(gray)

kernel = np.ones((3, 3), np.uint8)  # You can increase kernel size for more erosion
eroded = cv2.dilate(inverted, kernel, iterations=1)

diluted_black = cv2.bitwise_not(eroded)

dst = cv2.cornerHarris(diluted_black, blockSize=5, ksize=3, k=0.1)

img_with_corners = image.copy()
# img_with_corners[dst > 0.01 * dst.max()] = [0, 0, 255]

threshold = 0.01 * dst.max()

# Get corner pixel coordinates
corner_y, corner_x = np.where(dst > threshold)  # Note: y first, x second
corner_coords = list(zip(corner_x, corner_y)) 


def merge_close_points(points, threshold=10):
    tree = KDTree(points)
    merged = []
    visited = set()
    
    for i, pt in enumerate(points):
        if i in visited:
            continue
        # Find all points within the threshold distance
        nearby_idxs = tree.query_ball_point(pt, threshold)
        # Get the mean (average) position of those points
        cluster = np.mean([points[j] for j in nearby_idxs], axis=0)
        merged.append(tuple(map(int, cluster)))
        visited.update(nearby_idxs)
    
    return merged

merged_corners = merge_close_points(corner_coords, threshold=30)  # Customizable threshold
print(f"Corners found:  {len(merged_corners)}")

output = image.copy()

for (x, y) in merged_corners:
    cv2.circle(output, (x, y), 5, (0, 0, 255), -1)  # Red color for merged corner

_, binary_image = cv2.threshold(diluted_black, 127, 255, cv2.THRESH_BINARY)
inverted = cv2.bitwise_not(binary_image)

for (x, y) in merged_corners:
    cv2.circle(inverted, (x, y), radius=20, color=0, thickness=-1)  # white dot

lines_img = detect_lines_and_circles(inverted)

resized_image = cv2.resize(output, None, fx=0.5, fy=0.5)
resized_inverted = cv2.resize(inverted, None, fx=0.5, fy=0.5)
# resized_lines_img = cv2.resize(lines_img, None, fx=0.5, fy=0.5)

cv2.imshow("lines", lines_img)
cv2.imshow("processed", resized_image)
cv2.imshow("inverted", resized_inverted)
cv2.waitKey(0)
cv2.destroyAllWindows()