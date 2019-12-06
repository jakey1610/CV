import numpy as np
import cv2
import math
def sparseStereo(imgL,imgR):
    img1 = imgL
    img2 = imgR
    # cv2.imshow("1a",img1)
    # cv2.imshow("1b",img2)
    # Initiate STAR detector
    orb = cv2.ORB_create()

    # find the keypoints with ORB
    kp1,des1 = orb.detectAndCompute(img1,None)
    kp2,des2 = orb.detectAndCompute(img2,None)

    des1 = des1.astype("float32")
    des2 = des2.astype("float32")

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    disp = np.zeros((img1.shape[0],img1.shape[1]))
    # disparity = np.array([0 for i in range(len(matches))])
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
            mPt = kp1[min(n.queryIdx,len(kp1)-1)].pt
            nPt = kp2[min(m.queryIdx,len(kp2)-1)].pt
            disp[int(mPt[1]),int(mPt[0])]=math.sqrt((mPt[0]-nPt[0])**2 + (mPt[1]-nPt[1])**2)
    dist = (4.8)*(210/disp)

    return dist
    # draw_params = dict(matchColor = (0,255,0),
    #                    singlePointColor = (255,0,0),
    #                    matchesMask = matchesMask,
    #                    flags = 0)
    #
    # img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    # cv2.imshow("2",img5)
