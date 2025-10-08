from PIL import Image
import matplotlib.pyplot as plt
import os

# --- Load image ---
img = Image.open("/Users/akhi/Desktop/DIP/images/flower.png")   # replace with your image name
w, h = img.size

# --- Define grid size (2 rows × 4 columns = 8 parts) ---
rows, cols = 10, 10
w_part, h_part = w // cols, h // rows

# --- Create output directory ---
os.makedirs("parts", exist_ok=True)

# --- Split, fix color mode, and save parts ---
parts = []
count = 0
for i in range(rows):
    for j in range(cols):
        left = j * w_part
        top = i * h_part
        right = left + w_part
        bottom = top + h_part
        cropped = img.crop((left, top, right, bottom))

        # Convert RGBA to RGB (JPEG doesn’t support transparency)
        if cropped.mode == "RGBA":
            cropped = cropped.convert("RGB")

        parts.append(cropped)
        cropped.save(f"parts/part_{count+1}.jpg")  # or use .png if preferred
        count += 1

print(f"✅ Image split into {count} parts and saved in 'parts/' folder.")

# --- Display all parts in a grid ---
fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(parts[i])
    ax.set_title(f"Part {i+1}")
    ax.axis("off")

plt.tight_layout()
plt.show()
