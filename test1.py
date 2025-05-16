import cv2
import numpy as np
from scipy.spatial import KDTree
import math

# Parameters
DIST_THRESHOLD = 20  # max distance to consider two endpoints connected
ANGLE_THRESHOLD = 100  # max angle allowed between connected lines

# Load and preprocess image
image = cv2.imread("test4.jpg")
blur = cv2.GaussianBlur(image, (7, 7), 1)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 50, 150)

# Dilate to strengthen edges
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(canny, kernel, iterations=1)

# Hough Line Detection
lines_p = cv2.HoughLinesP(dilated, 1, np.pi / 180, threshold=80, minLineLength=30, maxLineGap=10)

if lines_p is None:
    print("No lines detected")
    exit()

# Structure to hold endpoints
line_segments = []
points = []

for line in lines_p:
    x1, y1, x2, y2 = line[0]
    pt1, pt2 = (x1, y1), (x2, y2)
    line_segments.append({'start': pt1, 'end': pt2})
    points.append(pt1)
    points.append(pt2)

# Build KDTree for fast nearest-neighbor search
tree = KDTree(points)

# Build a graph (dictionary of point -> list of connected points with line index)
graph = {pt: [] for pt in points}

for idx, seg in enumerate(line_segments):
    for endpoint in [seg['start'], seg['end']]:
        nearby = tree.query_ball_point(endpoint, DIST_THRESHOLD)
        for i in nearby:
            neighbor = points[i]
            if neighbor != endpoint:
                graph[endpoint].append((neighbor, idx))

# Helper to compute angle between two vectors
def angle_between(v1, v2):
    unit1 = v1 / np.linalg.norm(v1)
    unit2 = v2 / np.linalg.norm(v2)
    dot_product = np.clip(np.dot(unit1, unit2), -1.0, 1.0)
    angle = np.arccos(dot_product)
    return np.degrees(angle)

# DFS to detect loops
visited_paths = set()
loops = []

def dfs(current, path, visited_edges):
    if len(path) > 2 and current == path[0]:
        loops.append(path.copy())
        return

    for neighbor, idx in graph.get(current, []):
        edge_key = tuple(sorted((current, neighbor)))
        if edge_key in visited_edges:
            continue

        # Angle check (only if path has at least two segments)
        if len(path) >= 2:
            prev_vec = np.array(path[-1]) - np.array(path[-2])
            curr_vec = np.array(neighbor) - np.array(current)
            ang = angle_between(prev_vec, curr_vec)
            if ang > ANGLE_THRESHOLD:
                continue

        visited_edges.add(edge_key)
        path.append(neighbor)
        dfs(neighbor, path, visited_edges)
        path.pop()
        visited_edges.remove(edge_key)

# Run DFS from all points
for pt in graph:
    dfs(pt, [pt], set())

# Filter unique loops (by set of points)
unique_loops = []
seen = []

def points_match(a, b):
    return set(a) == set(b)

for loop in loops:
    found = any(points_match(loop, s) for s in seen)
    if not found:
        seen.append(loop)
        unique_loops.append(loop)

# Draw the cleaned polygons
output = image.copy()

for loop in unique_loops:
    pts = np.array(loop, np.int32).reshape((-1, 1, 2))
    cv2.polylines(output, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.putText(output, f"{len(loop)}-sided", tuple(pts[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

# Show results
cv2.imshow("Cleaned Polygons", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
