import cv2
import os
import math
import numpy as np

#distance is the distance matrix on the disparity map
def detDist(distance,top,left,width,height):
    left = max(0,left)
    top = max(0,top)
    dist = np.median(distance[top:top+height+1, left:min(distance.shape[1], left+width+1)])
    return dist

def detDistSparse(distance,top,left,width,height):
    left = max(0,left)
    top = max(0,top)
    dist = np.array(distance[top:top+height+1, left:min(distance.shape[1], left+width+1)])
    dist = np.median(dist[~np.isinf(dist)])
    return dist
