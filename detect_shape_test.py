import cv2
import numpy as np
from scipy.spatial import KDTree

image = cv2.imread("img/test4.jpg")  
blur = cv2.GaussianBlur(image, (7, 7), 1)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
inverted = cv2.bitwise_not(gray)
kernel = np.ones((3, 3), np.uint8)
eroded = cv2.dilate(inverted, kernel, iterations=1)
diluted_black = cv2.bitwise_not(eroded)
dst = cv2.cornerHarris(diluted_black, blockSize=5, ksize=3, k=0.1)
threshold = 0.01 * dst.max()
corner_y, corner_x = np.where(dst > threshold)
corner_coords = list(zip(corner_x, corner_y))

def merge_close_points(points, threshold=10):
    tree = KDTree(points)
    merged = []
    visited = set()
    
    for i, pt in enumerate(points):
        if i in visited:
            continue
        nearby_idxs = tree.query_ball_point(pt, threshold)
        cluster = np.mean([points[j] for j in nearby_idxs], axis=0)
        merged.append(tuple(map(int, cluster)))
        visited.update(nearby_idxs)
    
    return merged

merged_corners = merge_close_points(corner_coords, threshold=30)
print(f"Corners found:  {len(merged_corners)}")

output = image.copy()

for (x, y) in merged_corners:
    cv2.circle(output, (x, y), 5, (0, 0, 255), -1)

_, binary_image = cv2.threshold(diluted_black, 127, 255, cv2.THRESH_BINARY)
inverted = cv2.bitwise_not(binary_image)

for (x, y) in merged_corners:
    cv2.circle(inverted, (x, y), radius=20, color=0, thickness=-1)


def detect_lines_and_circles(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(image.shape) == 2:
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        result = image.copy()

    print(f"Contours found: {len(contours)}")

    line_endpoints = []

    if contours:
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue

            rect = cv2.minAreaRect(cnt)
            (rx, ry), (rw, rh), angle = rect
            aspect_ratio = max(rw, rh) / min(rw, rh)

            # if aspect_ratio > 8:
            #     [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
                
            #     length = 1000  # Extend line in both directions
            #     x1 = int(x0 - vx * length)
            #     y1 = int(y0 - vy * length)
            #     x2 = int(x0 + vx * length)
            #     y2 = int(y0 + vy * length)
                
            #     start_point = (x1, y1)
            #     end_point = (x2, y2)
                
            #     line_endpoints.append((start_point, end_point))
            #     cv2.line(result, start_point, end_point, (255, 0, 0), 2)
            #     cv2.putText(result, "Line", start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if aspect_ratio > 8:
                [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
                direction = np.array([vx, vy]).reshape(2)
                point = np.array([x0, y0]).reshape(2)

                projections = []
                for p in cnt[:, 0, :]:
                    vec = p - point
                    t = np.dot(vec, direction)
                    proj = point + t * direction
                    projections.append(tuple(proj.astype(int)))

                start_point = min(projections, key=lambda pt: np.dot(np.array(pt) - point, direction))
                end_point = max(projections, key=lambda pt: np.dot(np.array(pt) - point, direction))

                line_endpoints.append((start_point, end_point))
                cv2.line(result, start_point, end_point, (255, 0, 0), 2)
                cv2.putText(result, "Line", start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            hull = cv2.convexHull(cnt)
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

            if len(approx) > 6:
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                cv2.circle(result, (int(cx), int(cy)), int(radius), (0, 255, 255), 2)
                cv2.putText(result, "Circle", (int(cx), int(cy) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return result, line_endpoints


lines_img, endpoints = detect_lines_and_circles(inverted)

cleaned_endpoints = [
    ((int(x1), int(y1)), (int(x2), int(y2)))
    for ((x1, y1), (x2, y2)) in endpoints
]
# Step 1: Find intersection of two lines
def line_intersection(p1, p2, p3, p4):
    """Find intersection of lines (p1, p2) and (p3, p4), return point or None"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if denom == 0:
        return None  # Parallel lines, no intersection

    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    return (int(px), int(py))

# Step 2: Check if two points are within a threshold distance
import math

def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Step 3: Find the closest points from a given point
def find_closest_points(point, currentLine, threshold=100):
    closest_points = []
    for line in cleaned_endpoints:
        if line == currentLine:
            continue
        for p in line:
            dist = distance(point, p)
            if dist < threshold:
                closest_points.append([p, line])
    return closest_points

def dfs(point, line, visited, polygonList):
    stack = []
    visitedStack = []

    # Use a tuple instead of list for consistency and hashability
    stack.append((point, line))

    while len(stack) != 0:
        flag = 0
        currPoint, currLine = stack.pop()
        visitedStack.append(currLine)
        visited.add(currLine)

        closest_points_and_lines = find_closest_points(currPoint, currLine, 50)

        for elem in closest_points_and_lines:
            elem_tuple = tuple(elem)
            if elem_tuple not in visited:
                flag = 1
                for pt in elem[1]:
                    if pt != elem[0]:
                        stack.append((pt, elem[1]))
                        visited.add(elem[1])

                if elem[1] in visitedStack :
                    currentTrace = [] 
                    while visitedStack:
                        top = visitedStack.pop()
                        currentTrace.append(top)
                        if top == elem[1]:
                            break

                    polygonList.append(currentTrace.copy())
                    print(currentTrace)
                    return
                
        if flag != 1:
            visitedStack.pop()

def draw_polygons_from_lines(image):
    visited = set()
    polygonList = []
    # dfs(cleaned_endpoints[4][0], cleaned_endpoints[0], visited, polygonList)
    # cv2.circle(output, (cleaned_endpoints[1][0][0], cleaned_endpoints[1][0][1]), 10, (255, 0, 0), -1)

    for line in cleaned_endpoints:
        if line not in visited:
            dfs(line[0], line, visited, polygonList)
    print(polygonList)
    for polygon in polygonList:
        # Collect all points from the lines in the polygon
        points = []
        for line in polygon:
            for pt in line:
                if pt not in points:
                    points.append(pt)
                    cv2.circle(output, (pt[0], pt[1]), 10, (255, 0, 0), -1)

draw_polygons_from_lines(output)

resized_image = cv2.resize(output, None, fx=0.5, fy=0.5)
resized_inverted = cv2.resize(inverted, None, fx=0.5, fy=0.5)

# cv2.imshow("lines", lines_img)
cv2.imshow("processed", resized_image)
# cv2.imshow("inverted", resized_inverted)
cv2.waitKey(0)
cv2.destroyAllWindows()
