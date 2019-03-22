import cv2
import numpy as np
import scipy.misc
import glob


file_list = glob.glob('river_resize/*.png')

for i,file_path in enumerate(file_list):
    img = cv2.imread(file_path,0)
    filename = file_path.split('\\')
    edges = cv2.Canny(img,100,200)
    scipy.misc.imsave("river_resize_edge/"+filename[-1], edges)

