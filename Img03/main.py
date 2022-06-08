import cv2
import numpy as np
import sys
import math


def makesGray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, threshold_image = cv2.threshold(img, 220, 255, 0)
    return threshold_image


def Gauss(img):
    return cv2.GaussianBlur(img, (11, 11), 0)


def Canny_alg(img):
    low = 20
    high = 100
    img = cv2.Canny(img, low, high)
    return img


def region(image):
    w = 300
    tr = np.array([
        [(w - 50, 700), (280 + w, 480), (450 + w, 480), (750 + w, 700)]
    ])

    mask = np.zeros_like(image)

    mask = cv2.fillPoly(mask, tr, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask


def Hafa(img):
    dst = cv2.Canny(img, 50, 200, None, 3)
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 + 1000 * (a)))
            cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

            linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
            if linesP is not None:
                for i in range(0, len(linesP)):
                    l = linesP[i][0]
                    cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
    return cdstP


def combine(img, mainimg):
    return cv2.addWeighted(mainimg, 0.8, img, 1, 0)


def sayMyName(img):
    cv2.putText(img=img, text='Vronskiy V.', org=(1150, 700), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5,
                color=(0, 255, 0), thickness=1)
    return img


def for_1frame(img):
    img = Gauss(img)
    cv2.imshow('gray', img)
    cv2.waitKey()
    img = Canny_alg(img)
    cv2.imshow('gray', img)
    cv2.waitKey()
    img = region(img)
    cv2.imshow('gray', img)
    cv2.waitKey()
    img = Hafa(img)
    cv2.imshow('gray', img)
    cv2.waitKey()
    return img


img1 = cv2.imread('1_l_gray.jpg')
for_1frame(img1)

cap = cv2.VideoCapture('SantaMonica.mp4')
i = 0

while True:
    _, img = cap.read()
    imgC = np.copy(img)
    if img is None:
        break
    img = makesGray(img)
    img = Gauss(img)
    img = Canny_alg(img)
    img = region(img)
    img = Hafa(img)
    img = combine(img, imgC)
    img = sayMyName(img)
    cv2.imshow('gray', img)
    i += 1

    k = cv2.waitKey(30) & 0xff
    if k == 0:
        break
print(i)
cv2.waitKey()
cv2.destroyAllWindows()
