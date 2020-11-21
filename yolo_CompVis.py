import numpy as np
import argparse
import time
import cv2
import os
import glob

#argument processing
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

#loading the yolo model and coco labels
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

#defining yolo paths and weights
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

#load YOLO object detector trained on COCO dataset
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#looping through image folders and images in them
numWrongs = 0
for folders in glob.glob(args["image"]):
	for file in glob.glob(folders+"/*"):
		#load our input image and its spatial dimensions
		image = cv2.imread(file)
		(H, W) = image.shape[:2]

		#determine layer names
		ln = net.getLayerNames()
		ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

		#construct a blob from the input image and then perform a forward pass of the YOLO object detector
		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
			swapRB=True, crop=False)
		net.setInput(blob)
		start = time.time()
		layerOutputs = net.forward(ln)
		end = time.time()

		#initialize our lists of confidences, and class IDs
		confidences = []
		classIDs = []

		#loop over each of the layer outputs and detections
		for output in layerOutputs:
			for detection in output:
				#get the class ID and confidence
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]

				#filter out weak predictions by ensuring the detected probability is greater than the minimum probability
				if confidence > args["confidence"]:
					confidences.append(float(confidence))
					classIDs.append(classID)

		#superman 0, aeroplane 4, bird 14
		if (folders == r"./testSeparatedEven\bird" and (len(classIDs) == 0 or 14 not in classIDs)):
			numWrongs += 1
			print('WRONG')
		elif(folders == r"./testSeparatedEven\plane" and (len(classIDs) == 0 or 4 not in classIDs)):
			numWrongs += 1
			print('WRONG')
		elif(folders == r"./testSeparatedEven\superman" and (len(classIDs) == 0 or 0 not in classIDs)):
			numWrongs += 1
			print('WRONG')

print("NUM WRONG: ", numWrongs)
print("ACCURACY %: ", 100-((numWrongs/384)*100))
