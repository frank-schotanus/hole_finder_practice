import cv2
import numpy as np

filename = 'square.jpg'

#read in images
og_img = cv2.imread(filename, cv2.IMREAD_COLOR)

#convert to black and white
bw_img = cv2.cvtColor(og_img, cv2.COLOR_BGR2GRAY)

"""
#changing contrast & brightness
#currently doesn't work for subsquare.jpg(??)
alpha = 2.5  # Increase contrast(>1)
beta = 0    #brightness(<0 dim, >0 bright)
cntrst_img = cv2.convertScaleAbs(bw_img, alpha=alpha, beta=beta)
"""

#apply blur
img = cv2.GaussianBlur(bw_img, (13, 13), 2)

#establish image dimensions so measurements can be made in angstroms instead of pixels
image_width_pixels = img.shape[1]

if (filename == 'subsquare.jpg'):
    image_width_units = 107315.2
elif(filename == 'square.jpg'):
    image_width_units = 745881.6
units_per_pixel = image_width_units / image_width_pixels
angstrom_max_radius = 15000
angstrom_min_dist = 10926

circles = cv2.HoughCircles(
    img,
    cv2.HOUGH_GRADIENT,
    dp=1,  # Resolution ratio
    minDist= int(angstrom_min_dist / units_per_pixel),  # Minimum distance between centers
    param1=70,  # Upper threshold for Canny edge detector
    param2=20,  # Threshold for circle detection
    minRadius=0,  # Minimum radius
    maxRadius = int(angstrom_max_radius / units_per_pixel )   # Maximum radius
)

#coordinates of circels
circles = np.round(circles[0, :]).astype("int")

#drawing circles on img
for (x, y, r) in circles:
    cv2.circle(img, (x, y), r, (0, 255, 0), 2)

cv2.imshow("Detected Circles", img)
cv2.waitKey(0)
cv2.destroyAllWindows()