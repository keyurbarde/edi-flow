import cv2
import numpy as np

image = cv2.imread("test4.jpg")  
blur = cv2.GaussianBlur(image, (7, 7), 1)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, threshold1=50, threshold2=150)

kernel = np.ones((3, 3), np.uint8)
dil = cv2.dilate(canny, kernel, iterations=1)

contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
result = image.copy()

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 400:
        continue

    x, y, w, h = cv2.boundingRect(cnt)
    rect = cv2.minAreaRect(cnt)
    (rx, ry), (rw, rh), angle = rect
    aspect_ratio = max(rw, rh) / min(rw, rh)

    if aspect_ratio > 10:
        cv2.putText(result, "Line", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(result, [box], -1, (0, 0, 255), 2)
        continue

    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

    shape = "Unknown"
    sides = len(approx)

    if sides == 3:
        shape = "Triangle"
        cv2.drawContours(result, [approx], -1, (255, 0, 255), 2)
    elif sides == 4:
        ar = w / float(h)
        shape = "Square" if 0.8 <= ar <= 1.2 else "Rectangle"
        cv2.drawContours(result, [approx], -1, (255, 0, 255), 2)
    elif sides > 6:
        shape = "Circle"
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        cv2.circle(result, (int(cx), int(cy)), int(radius), (0, 255, 255), 2)
    else:
        cv2.drawContours(result, [approx], -1, (255, 0, 255), 2)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(result, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imshow('Detected Shapes and Lines', result)
cv2.waitKey(0)
cv2.destroyAllWindows()