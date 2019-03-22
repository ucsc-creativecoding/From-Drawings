import cv2
import numpy as np
import scipy.misc
import glob

flowers = scipy.misc.imread("flowers.png", mode='RGB')/ 255.
flowers_mask = scipy.misc.imread("flowers_mask.png", mode='RGB')/ 255.
fruit = scipy.misc.imread("fruit.png", mode='RGB')/ 255.
fruit_mask = scipy.misc.imread("fruit_mask.png", mode='RGB')/ 255.
ocean = scipy.misc.imread("ocean.png", mode='RGB')/ 255.
ocean_mask = scipy.misc.imread("ocean_mask.png", mode='RGB')/ 255.
tree = scipy.misc.imread("tree.png", mode='RGB')/ 255.
tree_mask = scipy.misc.imread("tree_mask.png", mode='RGB')/ 255.

flower_result = flowers * flowers_mask
fruit_result = fruit * fruit_mask
ocean_result = ocean * (1- tree_mask)
tree_result = tree * tree_mask

scipy.misc.imsave("flower_result.jpg", flower_result)
scipy.misc.imsave("fruit_result.jpg", fruit_result)
scipy.misc.imsave("ocean_result.jpg", ocean_result)
scipy.misc.imsave("tree_result.jpg", tree_result)

result = ocean_result + tree_result

scipy.misc.imsave("result.jpg", result)