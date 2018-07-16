import cv2
import numpy
import evoCnn
import math
from evoCnn import *
import sys


if __name__ == "__main__":
    config = Config("config.cfg")
    population = initalizePopulation(config)
    img = cv2.imread("lena.png")

    pic1 = runGeneomeOnImage(population[0], img)
    print pic1
    print "Result shape:", pic1.shape
    '''pic1 = pic1[:,:,0:3]
    pic1= (pic1*255).astype("uint8")
    while True:
        cv2.imshow("Original", img)
        cv2.imshow("Output1", pic1)

        k = cv2.waitKey(1)
        if k == ord('q'):
            break
            '''

    '''
    kernel = np.zeros((2,2))
    kernel[0:] = -2
    kernel[-1:] = 2
    kernel = np.array([[1,1],[1,1]])
    kernel = np.array([[1], [2],[3]])
    kernel = np.array([[1]])
    pic1 = convolve(img, (1,1), kernel, (2,2), "mean")
    pic1= (pic1*255).astype("uint8")
    while True:
        cv2.imshow("Original", img)
        b,g,r = cv2.split(img)
        cv2.imshow("Split", r)
        cv2.imshow("Output1", pic1)

        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    ''' 
