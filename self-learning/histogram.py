import matplotlib.pyplot as plt
import numpy as np
import cv2

# Load image (in RGB format)
img = cv2.imread('/Users/akhi/Desktop/akhi/DIP/FLOWER.jpeg')# Loads in BGR
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

red_img = img_rgb[:, :, 0] #Red channel
green_img = img_rgb[:, :, 1] #Green channel
blue_img = img_rgb[:, :, 2] #Blue channel

#printing pixel matrix
print(img_rgb) #3D array with pixel values
print(red_img) #Red channel
print(green_img) #Green channel
print(blue_img) #Blue channel

print(img_rgb[:5, :5])  # 5 rows and 5 columns

gray_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
print(gray_img)  # 2D matrix with intensity values (0–255)

#--- Display images
img_set = [img_rgb, red_img, green_img, blue_img]
title_set = ['RGB', 'Red', 'Green', 'Blue']
color_set = ['', 'Reds', 'Greens', 'Blues']
display_imgset(img_set, color_set, title_set, row = 2, col = 2)

#--- Prepare histogram for each color channel separately
prepare_histogram(red_img, 'Red')
prepare_histogram(green_img, 'Green')
prepare_histogram(blue_img, 'Blue')

def prepare_histogram(img, color_channel):
    #--- Prepare an array to hold the number of pixels
    pixel_count = np.zeros((256,), dtype = np.uint64)
    
    #--- Count number of pixels
	h, w, c = img.shape
	for i in range(h):
		for j in range(w):
			pixel_value = img[i,j]
			pixel_count[pixel_value] += 1
	print(pixel_count)

	#--- Plot histogram in two ways
	x = np.arange(256)
	plt.figure(figsize = (20,20))
	plt.subplot(1, 2, 1)
	plt.plot(x, pixel_count, 'ro')
	plt.title('Histogram of ' + color_channel + ' Channel')
	plt.xlabel('Pixel Values')
	plt.ylabel('Number of Pixels')

	plt.subplot(1, 2, 2)
	plt.bar(x, pixel_count)
	plt.title('Histogram of ' + color_channel + ' Channel')
	plt.xlabel('Pixel Values')
	plt.ylabel('Number of Pixels')
	plt.show()
	plt.close()

def display_imgset(img_set, color_set, title_set = '', row = 1, col = 1):
	plt.figure(figsize = (20, 20))
	k = 1
	for i in range(1, row + 1):
		for j in range(1, col + 1):
			plt.subplot(row, col, k)
			img = img_set[k-1]
			if(len(img.shape) == 3):
				plt.imshow(img)
			else:
				plt.imshow(img, cmap = color_set[k-1])
			if(title_set[k-1] != ''):
				plt.title(title_set[k-1])
			k += 1

plt.show()
plt.close()
