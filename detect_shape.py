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

import numpy as np
from collections import defaultdict

# Step 1: Clean data
cleaned_endpoints = [
    ((int(x1), int(y1)), (int(x2), int(y2)))
    for ((x1, y1), (x2, y2)) in endpoints
]

# Step 2: Build proximity-based graph
distance_threshold = 50  # Tune this based on your data

graph = defaultdict(set)

# Add original line segment connections
for u, v in cleaned_endpoints:
    graph[u].add(v)
    graph[v].add(u)

# Add edges between points that are close enough
all_points = list(graph.keys())
for i in range(len(all_points)):
    for j in range(i + 1, len(all_points)):
        p1, p2 = all_points[i], all_points[j]
        if np.linalg.norm(np.array(p1) - np.array(p2)) <= distance_threshold:
            graph[p1].add(p2)
            graph[p2].add(p1)

# Step 3: Normalize cycles
def normalize_cycle(cycle):
    cycle = list(cycle)
    min_idx = cycle.index(min(cycle))
    cycle = cycle[min_idx:] + cycle[:min_idx]
    rev = list(reversed(cycle))
    return tuple(min(cycle, rev))

# Step 4: DFS to find polygons
def find_polygons(max_length=10):
    found_cycles = set()

    def dfs(path, start, current, visited):
        if len(path) > max_length:
            return
        for neighbor in graph[current]:
            if neighbor == start and len(path) >= 3:
                found_cycles.add(normalize_cycle(path))
            elif neighbor not in visited:
                dfs(path + [neighbor], start, neighbor, visited | {neighbor})

    for node in graph:
        dfs([node], node, node, {node})

    return found_cycles

# Step 5: Run
polygons = find_polygons()
print(f"Detected {len(polygons)} polygon(s):")
for poly in polygons:
    print(poly)


# Draw the polygons
for poly in polygons:
    pts = np.array(poly, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))  # reshape for cv2.polylines
    cv2.polylines(output, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

# Optional: Draw vertices
for poly in polygons:
    for (x, y) in poly:
        cv2.circle(output, (x, y), radius=3, color=(255, 255, 255), thickness=-1)

resized_image = cv2.resize(output, None, fx=0.5, fy=0.5)
resized_inverted = cv2.resize(inverted, None, fx=0.5, fy=0.5)

# cv2.imshow("lines", lines_img)
cv2.imshow("processed", resized_image)
# cv2.imshow("inverted", resized_inverted)
cv2.waitKey(0)
cv2.destroyAllWindows()
