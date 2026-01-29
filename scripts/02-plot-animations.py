import imageio.v2 as imageio
import os
from PIL import Image
import numpy as np

"""
Plots animations that show:
- How the baseline DLA model works
- How the friendliness parameter modifies the attachment probability in the DLA
- How the growth mode parameter modifies the attachment probability in the DLA
"""

img_folders = [
    "dla-animation",
    "friendliness-animation",
    "growth-mode-animation",
]

for img_folder in img_folders:
    image_path = f"../images/{img_folder}"
    filenames = sorted([f for f in os.listdir(image_path) if f.endswith('.jpg')])
    print(f"Filenames: {filenames}")
    
    # Get size of first image
    first_img = Image.open(os.path.join(image_path, filenames[0]))
    target_size = first_img.size
    
    # Make sure all images are the same size
    images = []
    for f in filenames:
        img = Image.open(os.path.join(image_path, f))
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
        images.append(np.array(img_resized))
    
    # Save as GIF
    imageio.mimsave(f'../images/{img_folder}.gif', images, fps=2, loop=0)
    print(f"Saved: ../images/{img_folder}.gif ({len(images)} frames)")