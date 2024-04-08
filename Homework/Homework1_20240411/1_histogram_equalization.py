import cv2
import numpy as np
import matplotlib.pyplot as plt

grey = cv2.imread('./data/Lena.png', 0)
cv2.imshow('original grey', grey)
cv2.waitKey()