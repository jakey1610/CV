import cv2
import os
import math
import numpy as np
from postprocess import *
from distance import *
import copy

def yolo(img,disparity,imgPath,dS,heur,regions,hdT,wdT,sparse=False,disableWLS=False):
    #heur is a 10x10 matrix for the regions of the image. Starts off as even probability of choice
    #develops over time. Each image we check the regions and the number of objects out of the total,
    #and the region with the most gets +0.05, and the least gets -0.05. If none detected then do nothing.
    regionCount = 50
    keep_processing = True

    confThreshold = 0.6  # Confidence threshold
    nmsThreshold = 0.4  # Non-maximum suppression threshold
    inpWidth = 416# / 10      # Width of network's input image
    inpHeight = 416# / 10      # Height of network's input image

    # Load names of classes from file

    classesFile = "coco.names"
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    # load configuration and weight files for the model and load the network using them

    net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    #Switch to OPENCL for better performance if your computer supports it.
    #net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    output_layer_names = getOutputsNames(net)
    distances_boxes = []

    #make choice on regions to use based on probMat
    indChoice = list(np.random.choice(100,regionCount,replace=False, p=heur.flatten()))
    indChoice.sort()
    indChoice = [(x//10,x%10) for x in indChoice]

    #by the regions chosen stack the frames together into frame
    imgO = copy.deepcopy(img)
    frame = copy.deepcopy(img)
    regionsCover = []
    regionsCoord = []
    for i in indChoice:
        region = regions[i[0],i[1]]
        regionsCoord.append(region)
        regionsCover.append(imgO[region[0]:region[2], region[1]:region[3]])
    #Run the dnn on the regions individually


    while (keep_processing):

        # start a timer (to see how long processing and display takes)
        start_t = cv2.getTickCount()

        #all images stacked together
        regionsGrid = []
        for r in range(0,regionCount,10):
            regionsGrid.append(np.concatenate(np.array(regionsCover[r:r+10]),axis=1))
        regionsGrid = np.concatenate(np.array(regionsGrid),axis=0)
        # cv2.imwrite("regions.png", regionsGrid)
        # height = hdT*(regionCount//5)
        # width = wdT*5
        # region = region.reshape((height,width,3))

        # create a 4D tensor (OpenCV 'blob') from image frame (pixels scaled 0->1, image resized)
        tensor = cv2.dnn.blobFromImage(regionsGrid, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        # set the input to the CNN network
        net.setInput(tensor)

        # runs forward inference to get output of the final output layers
        results = net.forward(output_layer_names)

        # remove the bounding boxes with low confidence
        classIDs, confidences, boxes = postprocess(regionsGrid, results, confThreshold, nmsThreshold)

        #distance = focal length * (Baseline/disparity difference)
        distance = (0.48)*(210/disparity)
        if disableWLS:
            distance *= 10
        # draw resulting detections on image
        for detected_object in range(0, len(boxes)):
            box = boxes[detected_object]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            reg = regionsCoord[left//wdT+(top//hdT)*10]
            left = reg[1]
            top = reg[0]
            # width = reg[2]-reg[0]
            # height = reg[3]-reg[1]
            if top < 450:
                #calculate the distance to the detected object
                colour = (255, 178, 50)
                if classes[classIDs[detected_object]] == "car":
                    colour = (255,0,0)
                elif classes[classIDs[detected_object]] == "person":
                    colour = (0,0,255)
                elif classes[classIDs[detected_object]] == "van":
                    colour = (255,0,0)
                elif classes[classIDs[detected_object]] == "truck":
                    colour = (255,0,0)
                elif classes[classIDs[detected_object]] == "bus":
                    colour = (255,0,0)


                #Look into taking the average of many points in the bounding box
                # print(distance[top:top+height+1][left:left+width+1])
                if sparse:
                    dist = detDistSparse(dS,top,left,width,height)
                else:
                    dist = detDist(distance, top, left, width, height)
                if dist != float('inf') and not np.isnan(dist):
                    cv2.imshow("frame",frame[reg[0]:reg[2], reg[1]:reg[3]])
                    drawPred(frame, classes[classIDs[detected_object]], confidences[detected_object], left, top, left + width, top + height, colour, dist)
                    distances_boxes.append(dist)
                    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
                t, _ = net.getPerfProfile()
                label = 'Inference time: %.2f ms' % (t * 1000.0 / (cv2.getTickFrequency()))
                cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        # for region in range(len(regionsCover)):
        #     reg = regionsCoord[region]
        #     frame[reg[0]:reg[2], reg[1]:reg[3]] = regionsCover[region]
        # display image
        path = imgPath.replace("./assets/TTBB-durham-02-10-17-sub10/left-images/", "").replace("./assets/TTBB-durham-02-10-17-sub10/right-images/","")
        path_folder = "images/"
        cv2.imwrite(path_folder+path,frame)
        # cv2.imshow('imgO'+path[::-1][4],imgO)
        cv2.imshow('frame'+path[::-1][4],frame)


        # stop the timer and convert to ms. (to see how long processing and display takes)
        stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000

        #getting and returning the min distance
        dist = np.array(distances_boxes)
        if len(distances_boxes) == 0:
            return "None"
        return dist.min()


#####################################################################
# Draw the predicted bounding box on the specified image
# image: image detection performed on
# class_name: string name of detected object_detectiontest
# left, top, right, bottom: rectangle parameters for detection
# colour: to draw detection rectangle in

def drawPred(image, class_name, confidence, left, top, right, bottom, colour, distance):
    # Draw a bounding box.
    cv2.rectangle(image, (left, top), (right, bottom), colour, 3)

    # construct label
    label = '%s:%.2fm' % (class_name, distance)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(image, (left, top - round(1.5*labelSize[1])),
        (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

################################################################################
# Get the names of the output layers of the CNN network
# net : an OpenCV DNN module network object

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
