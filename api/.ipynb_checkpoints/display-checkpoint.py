import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import urllib


def display_images(image_files, rows, cols, fig_width = 15, fig_height = 5):
    fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    # If there's only one row but multiple columns
    if rows == 1 and cols > 1:
        axs = axs.ravel()  # Flattens the array
    # If there are multiple rows and columns
    elif rows > 1 and cols > 1:
        axs = axs.flatten()  # Flattens the array
    # If there's only one image to display
    elif rows == 1 and cols == 1:
        axs = np.array([axs])

    for i, file in enumerate(image_files):
        img = Image.open(file)
        axs[i].imshow(img)
        axs[i].set_title(image_files[i].split("_")[1].split(".")[0])
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()
    
    
def show_images_from_list(data, image_size=(512, 512)):
    num_images = len(data)
    rows = int(num_images / 4) + (num_images % 4 > 0)
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows))
    axes = axes.ravel()

    for i, item in enumerate(data):
        if "previewUrl" in item:
            image_url = item["previewUrl"][0]
            try:
                response = urllib.request.urlopen(image_url)
                image = Image.open(response)
                image = image.resize(image_size)  # Resize the image
                axes[i].imshow(image)
                axes[i].axis("off")
            except:
                print(f"Error loading image for item {i + 1}")
        else:
            print(f"No previewUrl field found for item {i + 1}")
        if i >= num_images - 1:
            break

    plt.tight_layout()
    plt.show()