import cv2
import os
import math
import numpy as np
from sympy import pprint, Matrix
from matplotlib import pyplot as plt
from preprocess import *
from distance import *
from postprocess import *
from sparseStereo import *


#Set to True to use sparse stereo instead of dense
sparse = False

#Set heuristic to True to make use of heuristic in processing
def gauss_2d():
    x, y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
    d = np.sqrt(x*x+y*y)
    sigma, mu = 10, 0.0
    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    # g = (lambda x: 1-x)(g)
    sumG = np.sum(g)
    g = g/sumG
    return g
heuristic = False
if heuristic:
    probMat = gauss_2d()
    from recognitionHeuristic import *
else:
    from recognition import *


#disable preprocessing of disparity map
disableWLS = False

master_path_to_dataset = "./assets/TTBB-durham-02-10-17-sub10";
#TESTING
directory_to_cycle_left = "left-images";     # folder containing the left images
directory_to_cycle_right = "right-images";   # folder containing the right images
#ACTUAL
# directory_to_cycle_left = "left-images";     # folder containing the left images
# directory_to_cycle_right = "right-images";   # folder containing the right images

skip_forward_file_pattern = ""; # set to timestamp to skip forward to

crop_disparity = False; # display full or cropped disparity image
#####################################################################

# resolve full directory location of data set for left / right images

full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left);
full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right);

# get a list of the left image files and sort them (by timestamp in filename)

left_file_list = sorted(os.listdir(full_path_directory_left));


max_disparity = 128; # maximum number of different disparity values
window_size =15
# stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21); # used to create the disparity map
left_match = cv2.StereoSGBM_create(
   minDisparity=0,
   numDisparities=max_disparity,
   blockSize=5,
   P1=8*3*window_size**2,
   P2=32*3*window_size**2,
   disp12MaxDiff=1,
   uniquenessRatio=15,
   speckleWindowSize=0,
   speckleRange=2,
   preFilterCap=63,
   mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
   )
right_match = cv2.ximgproc.createRightMatcher(left_match)

for filename_left in left_file_list:

    # skip forward to start a file we specify by timestamp (if this is set)

    if ((len(skip_forward_file_pattern) > 0) and not(skip_forward_file_pattern in filename_left)):
        continue;
    elif ((len(skip_forward_file_pattern) > 0) and (skip_forward_file_pattern in filename_left)):
        skip_forward_file_pattern = "";

    # from the left image filename get the correspondoning right image

    filename_right = filename_left.replace("_L", "_R");
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left);
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right);

    # check the file is a PNG file (left) and check a correspondoning right image
    # actually exists

    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)) :

        # read left and right images and display in windows

        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)

        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)

        # preprocess the left image to give better disparity measurements

        imgL = preprocessIR(imgL)
        imgR = preprocessIR(imgR)

        # remember to convert to grayscale (as the disparity matching works on grayscale)

        grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY);
        grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY);

        # perform preprocessing - raise to the power, as this subjectively appears
        # to improve subsequent disparity calculation

        grayL = np.power(grayL, 0.75).astype('uint8');
        grayR = np.power(grayR, 0.75).astype('uint8');

        # compute disparity image from undistorted, preprocessed and rectified stereo images that we have loaded

        disparityL = left_match.compute(imgL,imgR)   #stereoProcessor.compute(grayL,grayR);
        disparityR = right_match.compute(imgR,imgL)   #stereoProcessor.compute(grayR,grayL);

        # sparse stereo map
        distSparse = sparseStereo(grayL,grayR)

        # filter out noise and speckles (adjust parameters as needed)

        dispNoiseFilter = 5; # increase for more agressive filtering
        cv2.filterSpeckles(disparityL, 0, 4000, max_disparity - dispNoiseFilter);

        disparityL = (disparityL).astype('int16');
        disparityR = (disparityR).astype('int16');

        disparity = postprocessD(imgL,disparityL,disparityR,max_disparity,left_match,disableWLS=disableWLS)
        if heuristic:
            #regions defined below
            h,w = imgL.shape[0],imgL.shape[1]
            hdT,wdT = h//10,w//10
            regions = np.full((10,10,4),np.array([0,0,0,0]))
            for j in range(10):
                for i in range(10):
                    regions[i,j] = np.array([i*hdT,j*wdT,(i+1)*hdT,(j+1)*wdT])
            nDL = yolo(imgL,disparity,full_path_filename_left,distSparse,probMat,regions,hdT,wdT,sparse=sparse,disableWLS=disableWLS)
            nDR = yolo(imgR,disparity,full_path_filename_right,distSparse,probMat,regions,hdT,wdT,sparse=sparse,disableWLS=disableWLS)
        else:
            nDL = yolo(imgL,disparity,full_path_filename_left,distSparse,sparse=sparse,disableWLS=disableWLS)
            nDR = yolo(imgR,disparity,full_path_filename_right,distSparse,sparse=sparse,disableWLS=disableWLS)
        if nDL == "None":
            nD = nDR
        elif nDR == "None":
            nD = nDL
        else:
            nD = min(nDL,nDR)

        # for sanity print out these filenames
        path_L = full_path_filename_left.replace("./assets/TTBB-durham-02-10-17-sub10/left-images/", "")
        ext = "m"
        if nD == "None":
            ext = ""

        path_R = full_path_filename_right.replace("./assets/TTBB-durham-02-10-17-sub10/right-images/","") + ": nearest detected scene object (" + str(nD) + ext + ")"

        print(path_L);
        print(path_R);
        print();

        # keyboard input for exit (as standard)
        # exit - x

        key = cv2.waitKey(40 * (True)) & 0xFF; # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        if (key == ord('x')):       # exit
            break; # exit
    else:
            print();

#close all windows
cv2.destroyAllWindows()

#####################################################################
