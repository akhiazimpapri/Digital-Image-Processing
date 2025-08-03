import numpy as np
import matplotlib.pyplot as plt

def main():
    h, w = 100, 100
    steps = 5

    # Define 7 base colors
    color_dict = {
        "White": [255, 255, 255],
        "Red": [255, 0, 0],
        "Green": [0, 255, 0],
        "Blue": [0, 0, 255],
        "Yellow": [255, 255, 0],
        "Cyan": [0, 255, 255],
        "Magenta": [255, 0, 255]
    }

    img_set = []
    title_set = []
    color_set = []

    for color_name, rgb in color_dict.items():
        for i in range(steps):
            factor = i / (steps - 1)
            shade = (np.array(rgb) * factor).astype(np.uint8)
            shade_img = np.ones((h, w, 3), dtype=np.uint8) * shade
            img_set.append(shade_img)
            title_set.append(f"{color_name} Shade {i}")
            color_set.append(None)

    display_imgset(img_set, color_set, title_set, row=7, col=5)

def display_imgset(img_set, color_set, title_set='', row=1, col=1):
    plt.figure(figsize=(15, 10))
    for k in range(len(img_set)):
        plt.subplot(row, col, k + 1)
        img = img_set[k]
        if color_set[k]:
            plt.imshow(img, cmap=color_set[k], vmin=0, vmax=255)
        else:
            plt.imshow(img)
        plt.title(title_set[k], fontsize=9)
        plt.axis()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
