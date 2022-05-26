# Austin Whittaker
#
# CSC 340-001
#
# 11/21/21
#
# In the program we take 2 images of the same location and
# fuse the 2 images together to make 1 image, given they are
# in the same location with parts of the picture in the same frame.

# Give this program Time to load, it takes a little bit.

# This program was made at the University of North Carolina at Wilmington
# and is not to be copied or reproduced.

import np as np
import numpy as np
import cv2 as cv
import math
import random
from matplotlib import pyplot as plt


def main():
    print()
    print("Welcome to image stiching")
    print("'''''''''''''''''''''''''\n")
    print("Please pick an image you would like to work with\n"
          "\tPress 1 for Waterfall\n"
          "\tPress 2 for Mountain\n"
          "\tPress 3 for Florence\n"
          "\tPress 4 for Queenstown\n"    # in this section choose your own images
          )

    picChoice = input("Enter your choice here: ")

    if picChoice == '1':
        img1 = cv.imread('waterfall1.jpg')  # queryImage
        img2 = cv.imread('waterfall2.jpg')  # trainImage

    elif picChoice == '2':
        img1 = cv.imread('mountainLeft.jpg')  # queryImage
        img2 = cv.imread('mountainRight.jpg')  # trainImage

    elif picChoice == '3':
        img1 = cv.imread('florenceLeft.JPG')  # queryImage
        img2 = cv.imread('florenceRight.JPG')  # trainImage

    elif picChoice == '4':
        img1 = cv.imread('queenstownLeft.JPG')  # queryImage
        img2 = cv.imread('queenstownRight.JPG')  # trainImage

    elif picChoice == '5':
        img1 = cv.imread('neighborhoodleft.jpg')
        img2 = cv.imread('neighborhoodRight.jpg')

    elif picChoice == '6':
        img1 = cv.imread('neighborhoodleft2.jpg')
        img2 = cv.imread('neighborhoodRight2.jpg')

    elif picChoice == '7':
        img1 = cv.imread('campusLeft.jpg')
        img2 = cv.imread('campusRight.jpg')

    elif picChoice == '8':
        img1 = cv.imread('fountainLeft.jpg')
        img2 = cv.imread('fountainRight.jpg')

    elif picChoice == '9':
        img1 = cv.imread('libraryleft.jpg')
        img2 = cv.imread('libraryright.jpg')

    elif picChoice == '10':
        img1 = cv.imread('librarySmallLeft.jpg')
        img2 = cv.imread('librarySmallRight.jpg')

    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None,
                             flags=2)  # NOTE: 'None' parameter has to be added (not in documentation)

    plt.imshow(img3), plt.show()

    pts1 = np.zeros((len(good), 2), np.float32)
    pts2 = np.zeros((len(good), 2), np.float32)
    for m in range(len(good)):
        pts1[m] = kp1[good[m][0].queryIdx].pt
        pts2[m] = kp2[good[m][0].trainIdx].pt
    opencvH, mask = cv.findHomography(pts1, pts2, cv.RANSAC, 5.0)

    print("H matrix estimated by OpenCV (for Comparison):\n", opencvH)
    print()
    # Extra Syntax
    # Get coordinates for the ith match
    count = 0

    bestDistancePosition = [0, 0, 0, 0]
    bestDistanceAmount = []
    lowestDistance = 1000000

    for l in range(5000):
        i = random.randint(1, len(good) - 1)
        k = random.randint(1, len(good) - 1)
        m = random.randint(1, len(good) - 1)
        n = random.randint(1, len(good) - 1)

        while i == k:
            k = random.randint(1, len(good) - 1)
        while m == i or m == k:
            m = random.randint(1, len(good) - 1)
        while n == m or n == k or n == i:
            n = random.randint(1, len(good) - 1)

        qIdx = good[i][0].queryIdx
        tIdx = good[i][0].trainIdx
        x1 = kp1[qIdx].pt[0]
        y1 = kp1[qIdx].pt[1]
        x2 = kp2[tIdx].pt[0]
        y2 = kp2[tIdx].pt[1]

        qIdx = good[k][0].queryIdx
        tIdx = good[k][0].trainIdx
        x3 = kp1[qIdx].pt[0]
        y3 = kp1[qIdx].pt[1]
        x4 = kp2[tIdx].pt[0]
        y4 = kp2[tIdx].pt[1]

        qIdx = good[m][0].queryIdx
        tIdx = good[m][0].trainIdx
        x5 = kp1[qIdx].pt[0]
        y5 = kp1[qIdx].pt[1]
        x6 = kp2[tIdx].pt[0]
        y6 = kp2[tIdx].pt[1]

        qIdx = good[n][0].queryIdx
        tIdx = good[n][0].trainIdx
        x7 = kp1[qIdx].pt[0]
        y7 = kp1[qIdx].pt[1]
        x8 = kp2[tIdx].pt[0]
        y8 = kp2[tIdx].pt[1]

        a = [[0, 0, 0, (-x1 * 1), (-y1 * 1), (-1 * 1), (x1 * y2), (y1 * y2), y2],
             [(x1 * 1), (y1 * 1), 1, 0, 0, 0, (-x1 * x2), (-y1 * x2), (-1 * x2)],
             [0, 0, 0, (-x3 * 1), (-y3 * 1), (-1 * 1), (x3 * y4), (y3 * y4), y4],
             [(x3 * 1), (y3 * 1), 1, 0, 0, 0, (-x3 * x4), (-y3 * x4), (-1 * x4)],
             [0, 0, 0, (-x5 * 1), (-y5 * 1), (-1 * 1), (x5 * y6), (y5 * y6), y6],
             [(x5 * 1), (y5 * 1), 1, 0, 0, 0, (-x5 * x6), (-y5 * x6), (-1 * x6)],
             [0, 0, 0, (-x7 * 1), (-y7 * 1), (-1 * 1), (x7 * y8), (y7 * y8), y8],
             [(x7 * 1), (y7 * 1), 1, 0, 0, 0, (-x7 * x8), (-y7 * x8), (-1 * x8)]]

        U, s, V = np.linalg.svd(a, full_matrices=True)
        hCol = np.zeros((9, 1), np.float64)
        hCol = V[8, :]
        eval = [[hCol[0], hCol[1], hCol[2]],
                [hCol[3], hCol[4], hCol[5]],
                [hCol[6], hCol[7], hCol[8]]]

        totalDistance = 0

        for q in range(len(good)):
            pts1[q] = kp1[good[q][0].queryIdx].pt
            pts2[q] = kp2[good[q][0].trainIdx].pt
            hmm = [pts1[q][0], pts1[q][1], 1]
            new = [(eval[0][0] * hmm[0]) + (eval[0][1] * hmm[1]) + (eval[0][2] * hmm[2]),
                   (eval[1][0] * hmm[0]) + (eval[1][1] * hmm[1]) + (eval[1][2] * hmm[2]),
                   (eval[2][0] * hmm[0]) + (eval[2][1] * hmm[1]) + (eval[2][2] * hmm[2])]
            new2 = [new[0] / new[2], new[1] / new[2], new[2] / new[2]]
            distance = math.sqrt((new2[0] - pts2[q][0]) ** 2 + (new2[1] - pts2[q][1]) ** 2)
            totalDistance += distance

        if totalDistance < lowestDistance:
            lowestDistance = totalDistance
            BestH = eval
            bestDistancePosition[0] = i
            bestDistancePosition[1] = k
            bestDistancePosition[2] = m
            bestDistancePosition[3] = n
            bestDistanceAmount.append(totalDistance)


    print("Best H: ")
    something = BestH[2][2]
    BestH[0][0] = BestH[0][0] / something
    BestH[0][1] = BestH[0][1] / something
    BestH[0][2] = BestH[0][2] / something
    BestH[1][0] = BestH[1][0] / something
    BestH[1][1] = BestH[1][1] / something
    BestH[1][2] = BestH[1][2] / something
    BestH[2][0] = BestH[2][0] / something
    BestH[2][1] = BestH[2][1] / something
    BestH[2][2] = BestH[2][2] / something


    img1Row = img1.shape[0]
    img1Col = img1.shape[1]
    img2Row = img2.shape[0]
    img2Col = img2.shape[1]

    print()
    print("Image one size: ", img1Row, img1Col, "\n")
    print("Image two size: ", img2Row, img2Col, "\n")

    mappedCorner = [[0, 0, 1], [img1Col, 0, 1], [0, img1Row, 1], [img1Col, img1Row, 1]]
    image2Corner = [[0, 0, 1], [img2Col, 0, 1], [0, img2Row, 1], [img2Col, img2Row, 1]]

    ####### Fun part ########

    #     Multiply by H for first corner
    mappedCorner[0][0] = (opencvH[0][0] * mappedCorner[0][0]) + (opencvH[0][1] * mappedCorner[0][1]) + (
                opencvH[0][2] * mappedCorner[0][2])
    mappedCorner[0][1] = (opencvH[1][0] * mappedCorner[0][0]) + (opencvH[1][1] * mappedCorner[0][1]) + (
                opencvH[1][2] * mappedCorner[0][2])
    mappedCorner[0][2] = (opencvH[2][0] * mappedCorner[0][0]) + (opencvH[2][1] * mappedCorner[0][1]) + (
                opencvH[2][2] * mappedCorner[0][2])
    #     Divide by homogenous
    mappedCorner[0][0] = mappedCorner[0][0] / mappedCorner[0][2]
    mappedCorner[0][1] = mappedCorner[0][1] / mappedCorner[0][2]
    mappedCorner[0][2] = mappedCorner[0][2] / mappedCorner[0][2]

    #     Multiply by H for second corner
    mappedCorner[1][0] = (opencvH[0][0] * mappedCorner[1][0]) + (opencvH[0][1] * mappedCorner[1][1]) + (
                opencvH[0][2] * mappedCorner[1][2])
    mappedCorner[1][1] = (opencvH[1][0] * mappedCorner[1][0]) + (opencvH[1][1] * mappedCorner[1][1]) + (
                opencvH[1][2] * mappedCorner[1][2])
    mappedCorner[1][2] = (opencvH[2][0] * mappedCorner[1][0]) + (opencvH[2][1] * mappedCorner[1][1]) + (
                opencvH[2][2] * mappedCorner[1][2])
    #     Divide by homogenous
    mappedCorner[1][0] = mappedCorner[1][0] / mappedCorner[1][2]
    mappedCorner[1][1] = mappedCorner[1][1] / mappedCorner[1][2]
    mappedCorner[1][2] = mappedCorner[1][2] / mappedCorner[1][2]

    #     Multiply by H for third corner
    mappedCorner[2][0] = (opencvH[0][0] * mappedCorner[2][0]) + (opencvH[0][1] * mappedCorner[2][1]) + (
                opencvH[0][2] * mappedCorner[2][2])
    mappedCorner[2][1] = (opencvH[1][0] * mappedCorner[2][0]) + (opencvH[1][1] * mappedCorner[2][1]) + (
                opencvH[1][2] * mappedCorner[2][2])
    mappedCorner[2][2] = (opencvH[2][0] * mappedCorner[2][0]) + (opencvH[2][1] * mappedCorner[2][1]) + (
                opencvH[2][2] * mappedCorner[2][2])
    #     Divide by homogenous
    mappedCorner[2][0] = mappedCorner[2][0] / mappedCorner[2][2]
    mappedCorner[2][1] = mappedCorner[2][1] / mappedCorner[2][2]
    mappedCorner[2][2] = mappedCorner[2][2] / mappedCorner[2][2]

    #     Multiply by H for fourth corner
    mappedCorner[3][0] = (opencvH[0][0] * mappedCorner[3][0]) + (opencvH[0][1] * mappedCorner[3][1]) + (
                opencvH[0][2] * mappedCorner[3][2])
    mappedCorner[3][1] = (opencvH[1][0] * mappedCorner[3][0]) + (opencvH[1][1] * mappedCorner[3][1]) + (
                opencvH[1][2] * mappedCorner[3][2])
    mappedCorner[3][2] = (opencvH[2][0] * mappedCorner[3][0]) + (opencvH[2][1] * mappedCorner[3][1]) + (
                opencvH[2][2] * mappedCorner[3][2])
    #     Divide by homogenous
    mappedCorner[3][0] = mappedCorner[3][0] / mappedCorner[3][2]
    mappedCorner[3][1] = mappedCorner[3][1] / mappedCorner[3][2]
    mappedCorner[3][2] = mappedCorner[3][2] / mappedCorner[3][2]


    minX = min(mappedCorner[0][0], mappedCorner[1][0], mappedCorner[2][0], mappedCorner[3][0],
               0)
    maxX = max(mappedCorner[0][0], mappedCorner[1][0], mappedCorner[2][0], mappedCorner[3][0],
               image2Corner[0][0], image2Corner[1][0], image2Corner[2][0], image2Corner[3][0])
    minY = min(mappedCorner[0][1], mappedCorner[1][1], mappedCorner[2][1], mappedCorner[3][1],
               0)
    maxY = max(mappedCorner[0][1], mappedCorner[1][1], mappedCorner[2][1], mappedCorner[3][1],
               image2Corner[0][1], image2Corner[1][1], image2Corner[2][1], image2Corner[3][1])

    x = int(maxX - minX)
    y = int(maxY - minY)

    panorama = np.zeros((y, x, 3), np.float32)

    panRow = panorama.shape[0]
    panCol = panorama.shape[1]

    #print("Panorama shape: ", panRow, panCol)

    for i in range(img2Row):
        for j in range(img2Col):
            panorama[i + (int(abs(minY)))][j + int(abs(minX))] = img2[i][j]

    totalRedDifference = 0.0
    totalGreenDifference = 0.0
    totalBlueDifference = 0.0
    totalPixel = 0
    list = []

    for i in range(panRow):
        for j in range(panCol):
            hmm = [j - int(abs(minX)), i - int(abs(minY)), 1]
            Hinv = np.linalg.inv(opencvH)
            new3 = [((Hinv[0][0] * hmm[0]) + (Hinv[0][1] * hmm[1]) + (Hinv[0][2] * hmm[2])),
                    ((Hinv[1][0] * hmm[0]) + (Hinv[1][1] * hmm[1]) + (Hinv[1][2] * hmm[2])),
                    ((Hinv[2][0] * hmm[0]) + (Hinv[2][1] * hmm[1]) + (Hinv[2][2] * hmm[2]))]

            new4 = [(new3[0] / new3[2]), (new3[1] / new3[2]), new3[2] / new3[2]]

            x = int(new4[0])
            y = int(new4[1])

            if x > 0 and y > 0 and y < img1Row and x < img1Col:

                if panorama[i][j][0] != 0 and panorama[i][j][1] != 0 and panorama[i][j][2] != 0:
                    totalRedDifference += img1[y][x][0] - panorama[i][j][0]
                    totalGreenDifference += img1[y][x][1] - panorama[i][j][1]
                    totalBlueDifference += img1[y][x][2] - panorama[i][j][2]
                    print(totalRedDifference, totalGreenDifference, totalBlueDifference)
                    totalPixel += 1
                    list.append(i)
                    list.append(j)
                panorama[i, j] = img1[y][x]

    totalRedDifference /= totalPixel
    totalGreenDifference /= totalPixel
    totalBlueDifference /= totalPixel

    for i in range(0, len(list), 2):
        panorama[list[i]][list[i + 1]][0] -= totalRedDifference
        panorama[list[i]][list[i + 1]][1] -= totalGreenDifference
        panorama[list[i]][list[i + 1]][2] -= totalBlueDifference

    cv.imshow("new image", panorama / 255.0)   # show the images together
    cv.imwrite('Panorama9.jpg', panorama)      # save the new image.
    cv.waitKey(0)
    cv.destroyAllWindows()


main()
