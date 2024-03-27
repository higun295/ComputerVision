import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./data/Lena.png', 0).astype(np.float32) / 255

kernel = cv2.getGaborKernel((21, 21), 5, 1, 10, 1, 0, cv2.CV_32F)
