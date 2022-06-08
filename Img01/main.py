import cv2

img = cv2.imread('img1.png')

cv2.imshow('Image01', img)
cv2.waitKey(0)

cut = img[100:300, 200:500]
cv2.imshow('cut', cut)
cv2.waitKey(0)

resize = cv2.resize(img, (200, 200))
cv2.imshow('cut', resize)
cv2.waitKey(0)

h, w = img.shape[0:2]
center = (w // 2, h // 2)
Matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotate = cv2.warpAffine(img, Matrix, (w, h))
cv2.imshow('rotated', rotate)
cv2.waitKey(0)

blured = cv2.GaussianBlur(img, (11, 11), 0)
cv2.imshow('blur', blured)
cv2.waitKey(0)

imgRectangle = img
cv2.rectangle(imgRectangle, (150, 150), (500, 500), (0, 0, 255))
cv2.imshow('rectangle', imgRectangle)
cv2.waitKey(0)

imgText = img
font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
cv2.putText(imgText, "Lab01", (10, 100), font, 4,(255, 0, 0), 4, cv2.LINE_4)
cv2.imshow('Text', imgText)
cv2.waitKey(0)

cv2.destroyAllWindows()
