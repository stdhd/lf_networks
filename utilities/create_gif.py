from PIL import Image, ImageDraw
import os

images = []

width = 200
center = width // 2
color_1 = (0, 0, 0)
color_2 = (255, 255, 255)
max_radius = int(center * 1.5)
step = 8

path = "/home/sd/Documents/thesis/fvd/evaluations/predictions/40_1"
for file in os.listdir(path):
    if file.endswith('png'):
        with Image.open(file) as im:
            images.append(im)


images[0].save(os.path.join(path, 'imagedraw.gif'),
               save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)