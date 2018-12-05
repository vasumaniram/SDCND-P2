import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
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
    ksize = 11
    S_binary = color_threshold(img,(110,250))
    _,_,S_channel = get_hls_channels(img)
    gradx = abs_sobel_thresh(S_channel, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    combined = np.zeros_like(gradx)
    combined[((S_binary == 1 ) | (gradx == 1))] = 1
    return combined

def get_warp_matrices(img_size):
    src = np.float32([[  595,  450 ],
            [  685,  450 ],
            [ 1000,  660 ],
            [  280,  660 ]])

    dst = np.float32([[  300,    0 ],
            [  980,    0 ],
            [  980,  720 ],
            [  300,  720 ]])
    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    return M,Minv
def hist(img):
    bottom_half = img[img.shape[0]//2:,:]
    histogram = np.sum(bottom_half,axis=0)
    return histogram
def get_offset(img,xm_per_pix):
    histogram = hist(img)
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    mid_x_base =  (rightx_base + leftx_base) / 2
    image_center = img.shape[1]/2
    offset = np.absolute(image_center - mid_x_base) * xm_per_pix
    return offset
def find_lane_pixels(binary_warped):
    histogram = hist(binary_warped)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
  
    # Choosing the number of sliding windows
    nwindows = 9
    # Setting the width of the windows +/- margin
    margin = 80
    # Setting minimum number of pixels found to recenter window
    minpix = 50

    # Setting height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        # Four below boundaries of the window #
        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this
        
        #Draw the windows on the visualization image
#         cv2.rectangle(out_img,(win_xleft_low,win_y_low),
#         (win_xleft_high,win_y_high),(0,255,0), 2) 
#         cv2.rectangle(out_img,(win_xright_low,win_y_low),
#         (win_xright_high,win_y_high),(0,255,0), 2)
         # To Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If found > minpix pixels, recenter next window #
        #(`right` or `leftx_current`) on their mean position #
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    ## Visualization ##
    # Colors in the left and right lane regions
#     out_img[lefty, leftx] = [255, 0, 0]
#     out_img[righty, rightx] = [0, 0, 255]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped,leftx, lefty, rightx, righty,ym_per_pix=1,xm_per_pix=1):
    # Find our lane pixels first
    #leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
    # Fitting a second order polynomial to each using `np.polyfit` #
    left_fit = np.polyfit(lefty*ym_per_pix,leftx*xm_per_pix,2)
    right_fit = np.polyfit(righty*ym_per_pix,rightx*xm_per_pix,2)
    
    # Generating x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = np.poly1d(left_fit)(ploty)
        right_fitx = np.poly1d(right_fit)(ploty)
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    # out_img[lefty, leftx] = [255, 0, 0]
    # out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')

    return ploty,left_fit,right_fit,left_fitx,right_fitx

def measure_curvature_pixels(binary_warped,ploty,left_fit,right_fit,leftx, lefty, rightx, righty):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    ploty, left_fit, right_fit, _,_ = fit_polynomial(binary_warped,leftx, lefty, rightx, righty,ym_per_pix,xm_per_pix)
    
    # Defining y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Calculation of R_curve (radius of curvature)#
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0]) 
    
    return left_curverad, right_curverad

def draw_lanes(undist,ingimg_size,Minv,warped,ploty,left_fitx,right_fitx,left_curverad, right_curverad,offset=0):
    # Creat an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recasting the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Drawing the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warping the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, img_size) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    # Displaying the curvature values
    curvature_text = 'Left Curve Rad. : ' + str(left_curverad) + ' mts   Right Curve Rad. : ' + str(right_curverad) + ' mts'
    offset_text = 'Offset from center : ' + str(offset) + ' mts'
    cv2.putText(result,curvature_text,(100,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(result,offset_text,(200,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)
    plt.imshow(result)
    return result

def get_image_size(img):
    return img.shape[0:2:][::-1]

mtx, dist = calibtrate_camera()
def pipeline(img):
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    undist = cv2.undistort(img,mtx,dist,None,mtx) # Undistoring the image frame
    img_size = get_image_size(undist)
    M,Minv = get_warp_matrices(img_size) # Warp matrices M and Minv for perspective transform and back
    warped = cv2.warpPerspective(undist, M, img_size,flags=cv2.INTER_LINEAR) # Warp perspective to change the perspective into Bird's eye view 
    binary_warped = gradient_color_threshold(warped) # Using the color and grandient thresholds to find the lane lines under different conditions
    leftx, lefty, rightx, righty, _ = find_lane_pixels(binary_warped)
    offset = get_offset(binary_warped,xm_per_pix)
    ploty, left_fit, right_fit,left_fitx,right_fitx = fit_polynomial(binary_warped,leftx, lefty, rightx, righty) # Find the lanes from the binay image and fit the polynomial curve
    left_curverad, right_curverad = measure_curvature_pixels(binary_warped,ploty,left_fit,right_fit,leftx, lefty, rightx, righty) # Calculate the curvature radius of both left and right lane curves
    out_img = draw_lanes(undist,img_size,Minv,binary_warped,ploty,left_fitx,right_fitx,round(left_curverad), round(right_curverad),str(round(offset,2))) # Draw the lanes and the curvature radius on the image frame
    #print(str(left_curverad) + ' ' + str(right_curverad))
    return out_img
def pipeline_test(img,img_name):
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    undist = cv2.undistort(img,mtx,dist,None,mtx) # Undistoring the image frame
    img_size = get_image_size(undist)
    M,Minv = get_warp_matrices(img_size) # Warp matrices M and Minv for perspective transform and back
    warped = cv2.warpPerspective(undist, M, img_size,flags=cv2.INTER_LINEAR) # Warp perspective to change the perspective into Bird's eye view 
    cv2.imwrite(os.path.join('output_images/warp_images',img_name),warped)
    binary_warped = gradient_color_threshold(warped) # Using the color and grandient thresholds to find the lane lines under different conditions
    cv2.imwrite(os.path.join('output_images/binary_images',img_name),binary_warped.astype('uint8') * 255)
    leftx, lefty, rightx, righty, sli_img = find_lane_pixels(binary_warped)
    plt.imsave(os.path.join('output_images/sliding_window',img_name),sli_img)
    ploty, left_fit, right_fit,left_fitx,right_fitx = fit_polynomial(binary_warped,leftx, lefty, rightx, righty) # Find the lanes from the binay image and fit the polynomial curve
    # Plots the left and right polynomials on the lane lines
#     plt.plot(left_fitx, ploty, color='yellow')
#     plt.plot(right_fitx, ploty, color='yellow')
    left_curverad, right_curverad = measure_curvature_pixels(binary_warped,ploty,left_fit,right_fit,leftx, lefty, rightx, righty) # Calculate the curvature radius of both left and right lane curves
    out_img = draw_lanes(undist,img_size,Minv,binary_warped,ploty,left_fitx,right_fitx,round(left_curverad), round(right_curverad)) # Draw the lanes and the curvature radius on the image frame
    #print(str(left_curverad) + ' ' + str(right_curverad))
    return out_img
###########################################################################################################################################
###########################################################################################################################################

def fit_poly(img_shape, leftx, lefty, rightx, righty):
     # Fitting a second order polynomial to each with np.polyfit()
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    # Calculating both polynomials using ploty, left_fit and right_fit
#     left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
#     right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    try:
        left_fitx = np.poly1d(left_fit)(ploty)
        right_fitx = np.poly1d(right_fit)(ploty)
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    return left_fitx, right_fitx, ploty,left_fit,right_fit    

def search_around_poly(binary_warped,left_fit,right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100
    print('search_around_poly')
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit new polynomials
    if leftx.size == 0 or lefty.size == 0 or rightx.size == 0 or righty.size == 0:
        print('x r y size 0')
        leftx, lefty, rightx, righty, _ = find_lane_pixels(binary_warped)
        ploty, left_fit, right_fit,left_fitx,right_fitx = fit_polynomial(binary_warped,leftx, lefty, rightx, righty) 
    else:    
        left_fitx, right_fitx, ploty,left_fit,right_fit = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##
    
    return left_fitx, right_fitx, ploty,left_fit,right_fit,leftx, lefty, rightx, righty,result

from Line import Line
left_line = Line()
right_line = Line()
def pipeline_smart(img):
    import random
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    undist = cv2.undistort(img,mtx,dist,None,mtx) # Undistoring the image frame
    img_size = get_image_size(undist)
    M,Minv = get_warp_matrices(img_size) # Warp matrices M and Minv for perspective transform and back
    warped = cv2.warpPerspective(undist, M, img_size,flags=cv2.INTER_LINEAR) # Warp perspective to change the perspective into Bird's eye view 
    binary_warped = gradient_color_threshold(warped) # Using the color and grandient thresholds to find the lane lines under different conditions
    img_name = str(random.randint(1,1001)) + '.jpg' 
    cv2.imwrite(os.path.join('output_images/warp_images',img_name),warped)
    cv2.imwrite(os.path.join('output_images/binary_images',img_name),binary_warped.astype('uint8') * 255)
#     print(left_line.current_fit)
#     print(right_line.current_fit)
    window_img = None
    if left_line.current_fit is None or right_line.current_fit is None:
        print('inside 1')
        leftx, lefty, rightx, righty, _ = find_lane_pixels(binary_warped)
        ploty, left_fit, right_fit,left_fitx,right_fitx = fit_polynomial(binary_warped,leftx, lefty, rightx, righty) # Find the lanes from the binay image and fit the polynomial curve
        left_curverad, right_curverad = measure_curvature_pixels(binary_warped,ploty,left_fit,right_fit,leftx, lefty, rightx, righty) # Calculate the curvature radius of both left and right lane curves
    else:
        print('inside 2')
        left_fit = left_line.current_fit
        right_fit = right_line.current_fit
        left_fitx, right_fitx, ploty,left_fit,right_fit,leftx, lefty, rightx, righty,window_img = search_around_poly(binary_warped,left_fit,right_fit)
        left_curverad, right_curverad = measure_curvature_pixels(binary_warped,ploty,left_fit,right_fit,leftx, lefty, rightx, righty) # Calculate the curvature radius of both left and right lane curves
    left_line.current_fit = left_fit
    right_line.current_fit = right_fit
    if window_img is None:    
        window_img = draw_lanes(undist,img_size,Minv,binary_warped,ploty,left_fitx,right_fitx,round(left_curverad), round(right_curverad)) # Draw the lanes and the curvature radius on the image frame
    #print(str(left_curverad) + ' ' + str(right_curverad))
    return window_img

def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    #print(len(titles))
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        a.set_axis_off()
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)