import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

page = 'handwriting'

def main():
    img = cv2.imread(page + '.jpg')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    msk = cv2.inRange(hsv, np.array([95,40,40]), np.array([140,255,255]))
    res = cv2.bitwise_and(img.copy(), img.copy(), mask=msk)

    imgGray = res[...,2]
    ret, imgGray = cv2.threshold(imgGray, thresh=1, maxval=255, type=cv2.THRESH_BINARY)

    # morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    closing = imgGray
    closing = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)

    # dilate = cv2.dilate(imgGray, kernel, iterations=3)
    imgDilation = cv2.dilate(closing, np.ones((30,180), np.uint8), iterations=1)
    # find contours
    _, contours, hierarchy = cv2.findContours(
        imgDilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # imgContours = cv2.drawContours(img, contours, -1, (0,0,255), 3)

    for i, cnt in enumerate(sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])):
        area = cv2.contourArea(cnt) #, print(area)
        if area < 12000:
            continue
        processCroppedImage(i, cnt, img, closing)

    # equ = cv2.equalizeHist(imgGray)
    # imgGrayRes = equimgGray

    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB));
    plt.subplot(122)
    plt.imshow(closing, cmap='gray')
    plt.show()
    cv2.imwrite('demo.jpg', img)

def processCroppedImage(i, cnt, img, closing):
    x, y, w, h = cv2.boundingRect(cnt)
    cropRange = (slice(y-10,y+h+10), slice(x,x+w))
    croppedImage = closing[y-10:y+h+10, x:x+w].copy()
    print(croppedImage.shape)
    
    if not os.path.exists(page):
        os.mkdir(page)
    
    # make background transparent
    cv2.imwrite(
        page + '\\' + page + '_' + 'result' + str(i) + '.png',
        cv2.merge([croppedImage, croppedImage, croppedImage, croppedImage],4)
    )
    img = cv2.rectangle(img, (x, y-6), (x+w, y+h+6), (0,0,255), thickness=5)
    # plt.imshow(croppedImage)
    # plt.show()

if __name__ == '__main__':
    main()
