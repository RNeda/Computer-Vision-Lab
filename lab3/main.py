import cv2
import numpy as np
import matplotlib.pyplot as plt


def SpojDve(img1,img2):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    #koristimo brut force matcher da nadjemo najbolja poklapanja deskriptora dve slike
    bf = cv2.BFMatcher()
    matches12 = bf.knnMatch(descriptors1, descriptors2, k=2)
    #filtriramo poklapanja
    good_matches = []
    for m, n in matches12:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    #nalazimo matricu homography
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    #transformisemo coskove druge slike na osnovu homography matrice
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_corners2 = cv2.perspectiveTransform(corners2, homography)

    #racunamo velicinu panorame
    corners = np.concatenate((corners1, warped_corners2), axis=0)
    [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(corners.max(axis=0).ravel() - 0.5)

    #pravimo panoramsku sliku tako sto drugu sliku warpujemo i spajamo sa prvom
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
    warped_img2 = cv2.warpPerspective(img2, Ht @ homography, (xmax - xmin, ymax - ymin))
    warped_img2[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img1

    return warped_img2


if __name__ == '__main__':
    image1 = cv2.imread('1.JPG')
    image2 = cv2.imread('2.JPG')
    image3 = cv2.imread('3.JPG')
    warp_img12=SpojDve(image1, image2)
    #warp_img23=SpojDve(image2, image3)
    warp123 = SpojDve(warp_img12, image3)

    cv2.imshow('Panoramska slika', warp123)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
