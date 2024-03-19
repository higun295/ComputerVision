import argparse
import cv2, numpy as np, random

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./data/Lena.png', help='Image path.')
params = parser.parse_args()
image = cv2.imread(params.path)
image_to_show = np.copy(image)
w, h = image.shape[1], image.shape[0]

def rand_pt():
    return (random.randrange(w),
            random.randrange(h))

finish = False
while not finish:
    cv2.imshow("result", image_to_show)
    key = cv2.waitKey(0)
    if key == ord('p'):
        for pt in [rand_pt() for _ in range(10)]:
            cv2.circle(image_to_show, pt, 3, (255, 0, 0), -1)
    elif key == ord('1'):
        cv2.line(image_to_show, rand_pt(), rand_pt(), (0, 255, 0), 3)