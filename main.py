import cv2
import numpy
import evoCnn
import math
from evoCnn import *


if __name__ == "__main__":
    config = Config("config.cfg")
    population = initalizePopulation(config)
    img = cv2.imread("lena.png")
    kernel = np.zeros((2,2))
    kernel[0:] = -2
    kernel[-1:] = 2
    kernel = np.array([[1,1],[1,1]])
    kernel = np.array([[1], [2],[3]])
    pic1 = convolve(img, (1,3), kernel, (1,1), 3)
    #pic1 = convolve(img, (3,3), np.array([[-1,-2,-1],[0,0,0],[1,2,1]]), (1,1), 3)
    #runGeneomeOnImage(population[0], img)
    while True:
        cv2.imshow("Original", img)
        b,g,r = cv2.split(img)
        cv2.imshow("Split", r)
        cv2.imshow("Output1", pic1)

        k = cv2.waitKey(1)
        if k == ord('q'):
            break


