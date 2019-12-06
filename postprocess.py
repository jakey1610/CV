import cv2
import os
import math
import numpy as np

#####################################################################
# Remove the bounding boxes with low confidence using non-maxima suppression
# image: image detection performed on
# results: output from YOLO CNN network
# threshold_confidence: threshold on keeping detection
# threshold_nms: threshold used in non maximum suppression

def postprocess(image, results, threshold_confidence, threshold_nms):
    frameHeight = image.shape[0]
    frameWidth = image.shape[1]

    classIds = []
    confidences = []
    boxes = []

    # Scan through all the bounding boxes output from the network and..
    # 1. keep only the ones with high confidence scores.
    # 2. assign the box class label as the class with the highest score.
    # 3. construct a list of bounding boxes, class labels and confidence scores

    classIds = []
    confidences = []
    boxes = []
    for result in results:
        for detection in result:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > threshold_confidence:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences
    classIds_nms = []
    confidences_nms = []
    boxes_nms = []

    indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold_confidence, threshold_nms)
    for i in indices:
        i = i[0]
        classIds_nms.append(classIds[i])
        confidences_nms.append(confidences[i])
        boxes_nms.append(boxes[i])

    # return post processed lists of classIds, confidences and bounding boxes
    return (classIds_nms, confidences_nms, boxes_nms)

# postprocess the disparity map before depth measurements
def postprocessD(imgL,dispL,dispR,max_disparity,sGBM,disableWLS=False):
    if not disableWLS:
        # create the WLS filter
        lambdaParam = 8000
        sigma = 1.2
        visual_multiplier = 1.0

        wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=sGBM)
        wls.setLambda(lambdaParam)
        wls.setSigmaColor(sigma)

        # use WLS filter

        filterImg = wls.filter(dispL,imgL,None,dispR)

        filterImg = cv2.normalize(src=filterImg, dst=filterImg, beta=255, alpha=0, norm_type=cv2.NORM_MINMAX);
        filterImg = np.int16(filterImg)
    else:
        filterImg = dispL
    _, filterImg = cv2.threshold(filterImg,1, 128, cv2.THRESH_TOZERO);
    filterImg8 = filterImg.astype('uint8')
    cv2.imshow('DM', filterImg8)
    # cv2.imwrite("disparityWLS.png", filterImg8)


    return filterImg


# # Old use of JBF
# #perform joint bilateral filter on right image and disparity to reduce noise in the disparity map
# img = imgL.astype('float32')
# disp = dispL.astype('float32')
# d = cv2.ximgproc.jointBilateralFilter(img,disp,5,30,50)
# disparity_scaled = (d).astype('uint8');
# cv2.imwrite("disparityJBF.png", disparity_scaled)
