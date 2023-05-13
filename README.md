# Vanishing-point-in-Image
#Finding a vanishing point in image first lines using hough transform then finds the intersections of point and then by taking a mean you #can find vanishing point.

from google.colab.patches import cv2_imshow
import numpy as np
import cv2

# reading the image 
img=cv2.imread("/content/image001.jpg")
# Display input image
cv2_imshow(img)

# Making an Numpy array
arr = np.array(img)
#Updating array with the pixels storing in gray_img to get gray image.
gray_img = np.dot(arr[...,:3], [0.2989, 0.5870, 0.1140])

# Display Gray image
cv2_imshow(gray_img)

# applying the sobel filter
# first set weights for horizontal and vertical edges
kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Now convolving the kernels with the pixels of gray_img to get edges in horizontal and vertical directions.
filtered_x = np.convolve(gray_img.flatten(), kernel_x.flatten(), mode='same').reshape(gray_img.shape)
filtered_y = np.convolve(gray_img.flatten(), kernel_y.flatten(), mode='same').reshape(gray_img.shape)

# Finding the magnitude of edges
edges_magnitude = np.sqrt(filtered_x**2 + filtered_y**2)
# finding the direction of the edges
edges_direction = np.arctan2(filtered_y, filtered_x)

# Threshold the gradient magnitude to obtain the edge map
sobel_threshold = 500
# making an epmpty numpy array with same size of edges magnitude
edge_map = np.zeros_like(edges_magnitude)
# Now declaring if edges is graeater than threshold so gives its value of Maximum pixel value 255.
edge_map[edges_magnitude >= sobel_threshold] = 255

# Display the result
cv2_imshow( edge_map)

#save the image 
np.save('sobel_filtered_image.npy', edge_map)

#Loading above save model
sobel_filtered_image = np.load('sobel_filtered_image.npy')

# Define Hough transform parameters
threshold = 200

# Define theta range (0-180 degrees)
theta_range = np.deg2rad(np.arange(0,180, 1))

# Define rho range (-diagonal length to diagonal length)
width, height = sobel_filtered_image.shape
# By using pythahoras formula we can find max distance from origin
max_distance = int(np.ceil(np.sqrt(width**2 + height**2)))
rho_range = np.arange(-max_distance, +max_distance, 1)

# Initialize accumulator array and makes it zero
accumulator = np.zeros((len(rho_range), len(theta_range)))

# Perform Hough transform
# In hough transform we will convert the components to polar form and update the accumulator
for x in range(width):
    for y in range(height):
        if sobel_filtered_image[x, y] > 0:
            for t_idx, theta in enumerate(theta_range):
                rho = int(x * np.sin(theta) + y * np.cos(theta))
                r_idx = np.argmin(np.abs(rho_range - rho))
                accumulator[r_idx, t_idx] += 1                

# Finding lines greater than hough transform threshold from the accumulator
lines = []
for r_idx, t_idx in np.argwhere(accumulator > threshold):
    rho = rho_range[r_idx]
    theta = theta_range[t_idx]
    lines.append((rho, theta))

# Converting Line fron polar form to Rectabgular form to display on Image.
# And finding first and last point of the line.
for line in lines:
    rho, theta = line
    a = np.cos(theta)
    b = np.sin(theta)
    x = a * rho
    y = b * rho
    x1 = int(x + 1000 * (-b))
    y1 = int(y + 1000 * (a))
    x2 = int(x - 1000 * (-b))
    y2 = int(y - 1000 * (a))
    cv2.line(img, (x2, y2), (x1, y1), (0, 0, 255), 2)
# Displaying lines on Image
cv2_imshow(img)

angle_threshold = 30
intersections = []
for i in range(len(lines)):
    for j in range(i + 1, len(lines)):
        rho1, theta1 = lines[i]
        rho2, theta2 = lines[j]
        angle_diff = np.abs(np.rad2deg(theta1 - theta2))
        if angle_diff > angle_threshold:
            a1, b1 = np.cos(theta1), np.sin(theta1)
            a2, b2 = np.cos(theta2), np.sin(theta2)
            x, y = np.linalg.solve([[a1, b1], [a2, b2]], [rho1, rho2])
            intersections.append((int(x), int(y)))

vanishing_point = np.mean(intersections, axis=0, dtype=np.int32)

# Draw the vanishing point on the image
cv2.circle(img, vanishing_point, 10, (255, 0, 0), -1)

# Display the result
cv2_imshow(img)
