import cv2
import numpy as np

filename = 'subsquare.jpg'

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

#finding matches, loc = tuple of coordinates(row, column)
res = cv2.matchTemplate(bw_img, bw_template, cv2.TM_CCOEFF_NORMED)
threshold = .65 #adjust as needed
loc = np.where(res >= threshold)

def non_max_suppression(points, radius):
#function to get rid of all center points within one radius unit of each other
#meant to eliminate multiple circles being drawn for one circle
    selected_points = []
    for pt in points:
        add_point = True
        for sp in selected_points:
            if np.linalg.norm(np.array(pt) - np.array(sp)) < radius:
                add_point = False
                break
        if add_point:
            selected_points.append(pt)
    return selected_points

unique_circle_positions = non_max_suppression(zip(*loc[::-1]), h//3)  # Adjust the radius as needed

#drawing circles and '+'
for pt in unique_circle_positions:
    center = (pt[0] + w // 2, pt[1] + h // 2)
    cv2.circle(img, center, h//3, (0, 0, 255), 1)
    cv2.line(img, (center[0] - h//5, center[1]), (center[0] + h//5, center[1]), (0, 0, 0), 1)
    cv2.line(img, (center[0], center[1] - h//5), (center[0], center[1] + h//5), (0, 0, 0), 1)

#showing image
cv2.imshow("Detected Circles", img)
cv2.waitKey(0)
cv2.destroyAllWindows()