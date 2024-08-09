import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,	help="path to input image")
ap.add_argument("-p", "--prototxt", required=True, 	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,	help="path to Caffe pre-trained model")
ap.add_argument("-l", "--labels", required=True, help="path to ImageNet labels (i.e., syn-sets)")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

x, y, w, h = 165, 180, 1440, 720

image = image[y:y+h, x:x+w]
cv2.imshow("Image", image)
cv2.waitKey(0)
 
rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

for i in [1,2,4]:
	win_size = (180*i, 180*i)  
	for y in range(0, image.shape[0] - win_size[1] + 1, win_size[1]):
		for x in range(0, image.shape[1] - win_size[0] + 1, win_size[0]):
			roi = image[y:y+win_size[1], x:x+win_size[0]]
			blob = cv2.dnn.blobFromImage(roi, 1, (224, 224), (104, 117, 123))
   
			net.setInput(blob)
			preds = net.forward()
			idx = np.argmax(preds)

			confidence = preds[0][idx]
   
			if confidence > 0.75:
				label = "{}".format(classes[idx])
				if "cat" in label:
					cv2.rectangle(image, (x, y), (x+win_size[0], y+win_size[1]), (0, 0, 255), 2)
					cv2.putText(image, label, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
				elif "dog" in label:
					cv2.rectangle(image, (x, y), (x+win_size[0], y+win_size[1]), (0, 255, 0), 2)
					cv2.putText(image, label, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)

cv2.imwrite("output.jpg", image)



