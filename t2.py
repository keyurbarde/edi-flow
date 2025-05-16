import cv2
import numpy as np
from server import extract_texts_from_polygons
from scipy.spatial import KDTree

image = cv2.imread("C:/Users/athar/OneDrive/Desktop/Project/edi-flow/img/img7.jpg")  
blur = cv2.GaussianBlur(image, (7, 7), 1)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
inverted = cv2.bitwise_not(gray)
kernel = np.ones((3, 3), np.uint8)
eroded = cv2.dilate(inverted, kernel, iterations=1)
diluted_black = cv2.bitwise_not(eroded)
dst = cv2.cornerHarris(diluted_black, blockSize=5, ksize=3, k=0.1)
threshold = 0.02 * dst.max()
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

# Plot corner points
for (x, y) in merged_corners:
    cv2.circle(output, (x, y), 5, (0, 0, 255), -1)

_, binary_image = cv2.threshold(diluted_black, 128, 255, cv2.THRESH_BINARY)
inverted = cv2.bitwise_not(binary_image)

# Mask corner points
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

            if aspect_ratio > 2:
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

def line_intersection(p1, p2, p3, p4):
    """Find intersection of lines (p1, p2) and (p3, p4), return point or None"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if denom == 0:
        return None

    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    return (int(px), int(py))

import math

def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

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
    visitedLocal = set()
    stack.append((point, line))
    visitedLocal.add(line)
    # visitedStack.append((point, line))

    while len(stack) != 0:
        flag = 0
        currPoint, currLine = stack.pop()
        visitedStack.append((currPoint, currLine))
        # visited.add(currLine)

        # Find closest points within threshold
        closest_points_and_lines = find_closest_points(currPoint, currLine, 50)

        # elem = [point, line]
        for elem in closest_points_and_lines:
            elem_tuple = tuple(elem)
            if elem[1] not in visited:
                flag = 1
                opp_pt = ()
                for pt in elem[1]:
                    if pt != elem[0]:
                        stack.append((pt, elem[1]))
                        visitedLocal.add(elem[1])
                        opp_pt = pt

                if (opp_pt, elem[1]) in visitedStack :
                    print(opp_pt)
                    print(elem[1])
                    print(currLine)
                    # print(visitedStack)
                    # for elem in visitedStack:
                    #     print(elem)
                    currentTrace = [] 
                    while visitedStack:
                        _, top = visitedStack.pop()
                        currentTrace.append(top)
                        if top == elem[1]:
                            break
                    # print(currentTrace)
                    # print("")
                    polygonList.append(currentTrace.copy())
                    return
                
        if flag != 1:
            visitedStack.pop()

white_image = output.copy()
white_image[:] = 255

def is_black_pixel(img, x, y):
    h, w = img.shape[:2]
    if 0 <= x < w and 0 <= y < h:
        return all(img[y, x] == [0, 0, 0])
    return False

def extend_line_to_polygon(pt1, pt2, img, max_extension=200):
    def dda_extend(p1, p2, direction):
        x, y = p1
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        steps = int(max(abs(dx), abs(dy)))
        if steps == 0: return p1
        x_inc = dx / steps
        y_inc = dy / steps

        x_new = x
        y_new = y

        for _ in range(max_extension):
            x_new += direction * x_inc
            y_new += direction * y_inc
            if is_black_pixel(img, int(round(x_new)), int(round(y_new))):
                return (int(round(x_new)), int(round(y_new)))
        return (int(round(x_new)), int(round(y_new)))

    # Extend both directions
    ext_start = dda_extend(pt1, pt2, -1)
    ext_end = dda_extend(pt2, pt1, -1)
    return ext_start, ext_end



def draw_polygons_from_lines(image):

    visited = set()
    polygonList = []
    
    for line in cleaned_endpoints:
        if line not in visited:
            dfs(line[1], line, visited, polygonList)
            for polygon in polygonList:
                for line in polygon:
                    visited.add(line)
           

    
 
    polygonsVertexList = []

    for polygon in polygonList:
        print(polygon)
        singlePolygonVertexList = []
        # print(len(polygon))
        for i in range(0, len(polygon)):
            line1 = polygon[i]
            line2 = polygon[(i + 1) % len(polygon)]

            denom1 = line1[1][0] - line1[0][0]
            denom2 = line2[1][0] - line2[0][0]
            max_val = 999999
            if(denom1) == 0:
                denom1 = max_val
            if(denom2 == 0):
                denom2 = max_val

            slope1 = (line1[1][1] - line1[0][1]) / (denom1)
            slope2 = (line2[1][1] - line2[0][1]) / (denom2)

            # if (abs(slope1 - slope2) < 0.2):
            #     continue

            px, py = line_intersection(line1[0], line1[1], line2[0], line2[1])
            cv2.circle(output, (px, py), 10, (255, 0, 0), -1)
            singlePolygonVertexList.append((px, py))

        polygonsVertexList.append(singlePolygonVertexList.copy())
    print("from here -------------------- \n\n\n")
    

    texts = extract_texts_from_polygons("C:/Users/athar/OneDrive/Desktop/Project/edi-flow/img/img7.jpg", polygonsVertexList)
    print(texts)
    for i, polygon in enumerate(polygonsVertexList):
        if not polygon:
            continue
    
        for j in range(len(polygon)):
         pt1 = polygon[j]
         pt2 = polygon[(j + 1) % len(polygon)]
         cv2.line(white_image, pt1, pt2, (0, 0, 0), 2)


        if i < len(texts):
            text = texts[i]
            cx = int(np.mean([pt[0] for pt in polygon]))
            cy = int(np.mean([pt[1] for pt in polygon]))
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = cx - text_size[0] // 2
            text_y = cy + text_size[1] // 2
            cv2.putText(white_image, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    polygonLineList = []
    for polygon in polygonList:
        for line in polygon:
         polygonLineList.append(line)


    connectionLines = []
    for line in cleaned_endpoints:
        if line not in polygonLineList:
         connectionLines.append(line)
    print("connectionLines")
    print(connectionLines)


    for line in connectionLines:
        pt1, pt2 = line
        extended_pt1, extended_pt2 = extend_line_to_polygon(pt1, pt2, white_image)
        cv2.line(white_image, extended_pt1, extended_pt2, (0, 0, 0), 2)

    return white_image      
    


draw_polygons_from_lines(output)

resized_image = cv2.resize(output, None, fx=0.5, fy=0.5)
resized_inverted = cv2.resize(inverted, None, fx=0.5, fy=0.5)
resized_lines = cv2.resize(lines_img, None, fx=0.5, fy=0.5)
resized_white = cv2.resize(white_image, None, fx=0.5, fy=0.5)

# final_img = draw_polygons_from_lines(output)
# cv2.imshow("Final Output", final_img)
print("\n\n\n\n\n")
cv2.imshow("white", resized_white)
cv2.imshow("lines", resized_lines)
cv2.imshow("processed", resized_image)
cv2.imshow("inverted", resized_inverted)
cv2.waitKey(0)
cv2.destroyAllWindows()

