import cv2
import glob
import re
from moviepy.editor import *

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)

filename_array = []
for filename in glob.glob('input_frames/*.jpg'):
    filename_array.append(filename)
filename_array = natural_sort(filename_array)

img_array = []
for filename in filename_array:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)
    print(filename)
out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 20, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

clip = (VideoFileClip('output_video.avi'))
clip.write_gif("output_gif.gif")