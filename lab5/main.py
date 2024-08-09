import cv2
from cv2 import aruco
from pathlib import Path
from tqdm import tqdm
import numpy as np
#import yaml

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
marker_length = 3.42
marker_separation = 0.67
board = aruco.GridBoard([5, 7], marker_length, marker_separation, aruco_dict)
aruco_params = aruco.DetectorParameters()

calib_images_path = Path('C:/Users/Korisnik/Desktop/faks prez/rv/lab5/aruco')

img_list = []
calib_filenames = calib_images_path.glob('*.jpg')
for _, fn in enumerate(calib_filenames):
    img = cv2.imread(str(calib_images_path.joinpath(fn)))
    img_list.append(img)
print(str(len(img_list)) + ' images loaded')

counter = []
corners_list = []
ids_list = []
first_done = False
img_shape = img_list[0].shape
for im in tqdm(img_list):
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco.detectMarkers(img_gray, aruco_dict, parameters=aruco_params)
    if first_done:
        corners_list = np.vstack((corners_list, corners))
        ids_list = np.vstack((ids_list, ids))
    else:
        corners_list = corners
        ids_list = ids
        first_done = True
    counter.append(len(ids))
print('Number of detected markers per image:')
print(counter)
counter = np.array(counter)
camera_matrix_init = np.array([[img_shape[1], 0, img_shape[1] / 2], [0, img_shape[1], img_shape[0] / 2], [0, 0, 1]])
dist_coeffs_init = np.zeros((5, 1))
ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(corners_list, ids_list, counter, board, img_shape[:2],
                                                          camera_matrix_init, dist_coeffs_init)


video_path = Path('C:/Users/Korisnik/Desktop/faks prez/rv/lab5/aruco/Aruco_board.mp4')
cap = cv2.VideoCapture(str(video_path))
new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, img_shape[:2], 1, img_shape[:2])
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_undistorted = cv2.undistort(frame, mtx, dist, None, new_mtx)
    corners, ids, rejected = aruco.detectMarkers(img_undistorted, aruco_dict, parameters=aruco_params)
    if ids is not None:
        aruco.drawDetectedMarkers(img_undistorted, corners, ids)
        rvec = np.zeros((3, 1))
        tvec = np.zeros((3, 1))
        retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, new_mtx, dist, rvec, tvec)
        if retval != 0:
            cv2.drawFrameAxes(img_undistorted, new_mtx, dist, rvec, tvec, 10)
    cv2.imshow('frame', img_undistorted)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
