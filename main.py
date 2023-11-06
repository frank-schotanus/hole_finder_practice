import cv2
import numpy as np

filename = 'square.jpg'

#read in images
img = cv2.imread(filename, cv2.IMREAD_COLOR)
bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template = cv2.imread('holetemplate.jpg', cv2.IMREAD_COLOR)
bw_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

#resizing template image if needed
if(filename=='square.jpg'):
    dsize = (40, 40)
    bw_template = cv2.resize(bw_template,dsize)
w, h = bw_template.shape[::-1]


res = cv2.matchTemplate(bw_img, bw_template, cv2.TM_CCOEFF_NORMED)
threshold = .75
loc = np.column_stack(np.where(res >= threshold))
scores = [res[i[0], i[1]] for i in loc]


selected_points = []
circle = []
circle_scores = []
radius = h//2
for (x, y), score in zip(loc, scores):
    point = (x, y)
    circle.append(point)
    circle_scores.append(score)
    for pt in circle:
        if np.linalg.norm(np.array(point) - np.array(pt)) > radius:
            circle.pop()
            circle_scores.pop()
            max_index = circle_scores.index(max(circle_scores))
            selected_points.append(circle[max_index])
            cv2.circle(img, circle[max_index], 1, (0,0,255),1)
            circle = []
            circle_scores = []
            break

"""
#drawing circles and '+'
for pt in selected_points:
    center = (pt[0], pt[1])
    cv2.circle(img, center, h//4, (0, 0, 255), 1)
    cv2.line(img, (center[0] - h//5, center[1]), (center[0] + h//5, center[1]), (0, 0, 0), 1)
    cv2.line(img, (center[0], center[1] - h//5), (center[0], center[1] + h//5), (0, 0, 0), 1)
"""

#showing image
cv2.imshow("Detected Circles", img)
cv2.waitKey(0)
cv2.destroyAllWindows()