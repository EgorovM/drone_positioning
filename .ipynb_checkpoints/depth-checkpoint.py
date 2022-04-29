import cv2
import numpy as np
import pandas as pd


K1 = np.array([[1.80836116e+03, 0.00000000e+00, 9.01640157e+02],
       [0.00000000e+00, 1.81468291e+03, 6.06344069e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
D1 = np.array([[-0.02986817,  0.81384246,  0.01216111, -0.00578491, -1.74122658]])
K2 = np.array([[1.69258208e+03, 0.00000000e+00, 8.84363790e+02],
       [0.00000000e+00, 1.70496465e+03, 5.58035532e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
D2 = np.array([[ 0.09084974,  0.37762944,  0.00312737, -0.01385034, -1.98001889]])
R = np.array([[ 0.99989374,  0.00247096,  0.01436672],
       [-0.00332355,  0.99821515,  0.05962776],
       [-0.01419374, -0.05966917,  0.99811729]])
T = np.array([[-9.34900829],
       [-0.80682317],
       [-2.72978355]])
E = np.array([[ 2.37925445e-03,  2.77305376e+00, -6.42533279e-01],
       [-2.86219083e+00, -5.64592757e-01,  9.29218881e+00],
       [ 8.37809361e-01, -9.33032808e+00, -5.45869020e-01]])
F = np.array([[-1.30483384e-09, -1.51550381e-06,  1.55732163e-03],
       [ 1.55828635e-06,  3.06315100e-07, -1.07392877e-02],
       [-1.64612014e-03,  9.80000959e-03,  1.00000000e+00]])

width, height = (1920, 1080)

R1, R2, P1, P2, Q, B1, B2 = cv2.stereoRectify(K1, D1, K2, D2, (width, height), R, T)

def depth_map(imgL, imgR):
    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
    # SGBM Parameters -----------------
    window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=5*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.3
    visual_multiplier = 6

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)

    return filteredImg


if __name__ == '__main__':
    # is camera stream or video
    if False:
        cap_left = cv2.VideoCapture(1, cv2.CAP_V4L2)
        cap_right = cv2.VideoCapture(3, cv2.CAP_V4L2)
    else:
        cap_left = cv2.VideoCapture(1)
        cap_right = cv2.VideoCapture(3)

    if not cap_left.isOpened() and not cap_right.isOpened():  # If we can't get images from both sources, error
        print("Can't opened the streams!")
        sys.exit(-9)

    # Change the resolution in need
    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # float
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # float

    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # float
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # float

    while True:  # Loop until 'q' pressed or stream ends
        # Grab&retreive for sync images
        if not (cap_left.grab() and cap_right.grab()):
            print("No more frames")
            break

        _, leftFrame = cap_left.retrieve()
        _, rightFrame = cap_right.retrieve()
        height, width, channel = leftFrame.shape  # We will use the shape for remap

        # Undistortion and Rectification part!
        leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (width, height), cv2.CV_32FC1)
        left_rectified = cv2.remap(leftFrame, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (width, height), cv2.CV_32FC1)
        right_rectified = cv2.remap(rightFrame, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

        # We need grayscale for disparity map.
        gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

        disparity_image = depth_map(gray_left, gray_right)  # Get the disparity map

        # Show the images
        cv2.imshow('left(R)', leftFrame)
        cv2.imshow('right(R)', rightFrame)
        cv2.imshow('Disparity', disparity_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Get key to stop stream. Press q for exit
            break

    # Release the sources.
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()