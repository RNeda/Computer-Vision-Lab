import matplotlib.pyplot as plt
import cv2 as cv

if __name__ == '__main__':

    imgIn = cv.imread("../coins.jpeg")
    plt.imshow(imgIn)
    plt.title("original image:")
    #plt.show()

    #prebacujemo u grayscale
    imgGrayscale = cv.cvtColor(imgIn, cv.COLOR_BGR2GRAY)
    plt.imshow(imgGrayscale, cmap='gray')
    plt.title("grayscale image:")
    #plt.show()

    _, imgGrayThreshold = cv.threshold(imgGrayscale, 200, 255, cv.THRESH_BINARY_INV)
    #_, imgGrayThreshold = cv.threshold(imgGrayscale, 0, 255, cv.THRESH_BINARY_INV +cv.THRESH_OTSU) da imamo THRESH_OTSU automatski bi se odredjivao prag za threshold
    plt.imshow(imgGrayThreshold)
    plt.title("grayscale threshold image:")
    #plt.show()

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=(3, 3))

    imgOpen = cv.morphologyEx(imgGrayThreshold, op=cv.MORPH_OPEN, kernel=kernel)

    imgOpenClose = cv.morphologyEx(imgOpen, op=cv.MORPH_CLOSE, kernel=kernel)
    #cv.imshow("OpenClose:", imgOpenClose2)

    #dodatna diletacija kernelom 5x5 da bi se popunile rupe
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=(5, 5))
    imgDilate = cv.dilate(imgOpenClose, kernel=kernel)
    cv.imshow("Dilate", imgDilate)

    #mask2 = 255 - imgOpenClose2
    mask2 = 255 - imgDilate
    img_res2 = cv.bitwise_and(imgIn, imgIn, mask=mask2)
    cv.imshow("mask all", img_res2)

    #prebacujemo u hsv
    imgHsv = cv.cvtColor(imgIn, cv.COLOR_BGR2HSV)
    plt.imshow(imgHsv)
    plt.title("hsv image:")
    #plt.show()

    saturation_channel = imgHsv[:, :, 1] #uzimamo samo saturation info iz hsv slike
    plt.imshow(saturation_channel, cmap="gray")
    plt.title("saturation channel image:")
    #plt.show()
    _, img = cv.threshold(saturation_channel, 50, 255, cv.THRESH_BINARY_INV)
    #cv.imshow("hsv threshold image:", img)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=(5, 5))

    imgOpen2 = cv.morphologyEx(img, op=cv.MORPH_OPEN, kernel=kernel)

    imgOpenClose2 = cv.morphologyEx(imgOpen2,op=cv.MORPH_CLOSE, kernel=kernel)
    #cv.imshow("OpenClose",imgOpenClose2)

    # dodatna diletacija kernelom 2x2 da bi se popunile rupe, mada i ne mora, okej je i bez toga
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=(2, 2))
    imgDilate2 = cv.dilate(imgOpenClose2, kernel=kernel)
    cv.imshow("Dilate", imgDilate2)

    mask = 255 - imgDilate2  #imgOpenClose2

    img_res=cv.bitwise_and(imgIn,imgIn,mask=mask)
    cv.imshow("result image",img_res)

    cv.waitKey(0)
    cv.destroyAllWindows()