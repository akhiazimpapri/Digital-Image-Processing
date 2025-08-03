import numpy as np
import matplotlib.pyplot as plt

def main():
    h, w = 200, 200

    # --- White shades ---
    white_0 = np.zeros((h, w, 3), dtype=np.uint8)                   # Black
    white_1 = np.full((h, w, 3), 127, dtype=np.uint8)               # 50% White
    white_2 = np.full((h, w, 3), 255, dtype=np.uint8)               # White

    # --- Red shades ---
    red_0 = np.zeros((h, w, 3), dtype=np.uint8)
    red_1 = np.zeros((h, w, 3), dtype=np.uint8)
    red_2 = np.zeros((h, w, 3), dtype=np.uint8)
    red_1[:, :, 0] = 127
    red_2[:, :, 0] = 255

    # --- Green shades ---
    green_0 = np.zeros((h, w, 3), dtype=np.uint8)
    green_1 = np.zeros((h, w, 3), dtype=np.uint8)
    green_2 = np.zeros((h, w, 3), dtype=np.uint8)
    green_1[:, :, 1] = 127
    green_2[:, :, 1] = 255

    # --- Blue shades ---
    blue_0 = np.zeros((h, w, 3), dtype=np.uint8)
    blue_1 = np.zeros((h, w, 3), dtype=np.uint8)
    blue_2 = np.zeros((h, w, 3), dtype=np.uint8)
    blue_1[:, :, 2] = 127
    blue_2[:, :, 2] = 255

    # --- Cyan shades (G+B) ---
    cyan_0 = np.zeros((h, w, 3), dtype=np.uint8)
    cyan_1 = np.zeros((h, w, 3), dtype=np.uint8)
    cyan_2 = np.zeros((h, w, 3), dtype=np.uint8)
    cyan_1[:, :, 1:] = 127
    cyan_2[:, :, 1:] = 255

    # --- Magenta shades (R+B) ---
    magenta_0 = np.zeros((h, w, 3), dtype=np.uint8)
    magenta_1 = np.zeros((h, w, 3), dtype=np.uint8)
    magenta_2 = np.zeros((h, w, 3), dtype=np.uint8)
    magenta_1[:, :, [0, 2]] = 127
    magenta_2[:, :, [0, 2]] = 255

    # --- Yellow shades (R+G) ---
    yellow_0 = np.zeros((h, w, 3), dtype=np.uint8)
    yellow_1 = np.zeros((h, w, 3), dtype=np.uint8)
    yellow_2 = np.zeros((h, w, 3), dtype=np.uint8)
    yellow_1[:, :, [0, 1]] = 127
    yellow_2[:, :, [0, 1]] = 255

    # Combine all images
    img_set = [
        white_0, white_1, white_2,
        red_0, red_1, red_2,
        green_0, green_1, green_2,
        blue_0, blue_1, blue_2,
        cyan_0, cyan_1, cyan_2,
        magenta_0, magenta_1, magenta_2,
        yellow_0, yellow_1, yellow_2
    ]

    title_set = [
        'White 0%', 'White 50%', 'White 100%',
        'Red 0%', 'Red 50%', 'Red 100%',
        'Green 0%', 'Green 50%', 'Green 100%',
        'Blue 0%', 'Blue 50%', 'Blue 100%',
        'Cyan 0%', 'Cyan 50%', 'Cyan 100%',
        'Magenta 0%', 'Magenta 50%', 'Magenta 100%',
        'Yellow 0%', 'Yellow 50%', 'Yellow 100%'
    ]

    color_set = [None] * len(img_set)

    display_imgset(img_set, color_set, title_set, row=7, col=3)

def display_imgset(img_set, color_set, title_set='', row=1, col=1):
    plt.figure(figsize=(10, 12))
    for k in range(len(img_set)):
        plt.subplot(row, col, k + 1)
        img = img_set[k]
        if color_set[k]:
            plt.imshow(img, cmap=color_set[k], vmin=0, vmax=255)
        else:
            plt.imshow(img)
        plt.title(title_set[k], fontsize=9)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
