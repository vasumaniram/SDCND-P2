import numpy as np
import cv2
import glob
import os

def calibtrate_camera():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((9*6,3),np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray,(9,6),None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)        
    return mtx, dist
def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    x = 1 if orient == 'x' else 0
    y = 1 if orient == 'y' else 0
    sobel = cv2.Sobel(gray,cv2.CV_64F,x,y,ksize=sobel_kernel)
    sobel_abs = np.absolute(sobel)
    sobel_scaled = np.uint8(255*sobel_abs/np.max(sobel_abs))
    grad_binary = np.zeros_like(sobel_scaled)
    grad_binary[(sobel_scaled >= thresh[0]) & (sobel_scaled <= thresh[1])] = 1
    return grad_binary

def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    sobel_x = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
    sobel_mag = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    sobel_mag_abs = np.absolute(sobel_mag)
    sobel_mag_scaled = np.uint8(255*sobel_mag_abs/np.max(sobel_mag_abs))
    mag_binary = np.zeros_like(sobel_mag_scaled)
    mag_binary[(sobel_mag_scaled >= mag_thresh[0]) & (sobel_mag_scaled <= mag_thresh[1])] = 1
    return mag_binary

def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    sobel_x = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
    sobel_abs_dir = np.arctan2(np.absolute(sobel_y),np.absolute(sobel_x))
    dir_binary = np.zeros_like(sobel_abs_dir)
    dir_binary[(sobel_abs_dir >= thresh[0]) & (sobel_abs_dir <= thresh[1])] = 1
    return dir_binary
def get_hls_channels(img):
    hls_img = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    H = hls_img[:,:,0]
    L = hls_img[:,:,1]
    S = hls_img[:,:,2]
    return H,L,S
def color_threshold(img,thresh):
    _,_,S = get_hls_channels(img)
    binary = np.zeros_like(S)
    binary[(S >= thresh[0]) & (S <= thresh[1])]=1
    return binary
def gradient_color_threshold(img):
    ksize = 15
    S_binary = color_threshold(img,(80,240))
    _,_,S_channel = get_hls_channels(img)
    gradx = abs_sobel_thresh(S_channel, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    combined = np.zeros_like(gradx)
    combined[((S_binary == 1 ) | (gradx == 1))] = 1
    return combined

def warp(image):
    img_size = image.shape[0:2:][::-1]
    src = np.float32(
        [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
        [((img_size[0] / 6) - 10), img_size[1]],
        [(img_size[0] * 5 / 6) + 60, img_size[1]],
        [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
    dst = np.float32(
        [[(img_size[0] / 4), 0],
        [(img_size[0] / 4), img_size[1]],
        [(img_size[0] * 3 / 4), img_size[1]],
        [(img_size[0] * 3 / 4), 0]])
    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
    return warped
def hist(img):
    bottom_half = img[img.shape[0]//2:,:]
    histogram = np.sum(bottom_half,axis=0)
    return histogram
mtx, dist = calibtrate_camera()
def advanced_lane_finding_pipeline(img):
    undist_img = cv2.undistort(img,mtx,dist,None,mtx)
    warp_img = warp(undist_img)
    binary_warped_img = gradient_color_threshold(warp_img)
    return binary_warped_img
