from glob import glob
import cv2


i = 1
for dock in glob("ImagesTest/*"):
    files = glob(dock + '/*.jpg')
    for myFile in files:
        image = cv2.imread(myFile)
        blur = cv2.medianBlur(image, 5)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, element)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(image, contours, -1, (0, 0, 255), 3)
        cv2.imwrite('Contoured/' + str(i) + '.jpg', image)
        i += 1