import numpy as np
import cv2

CENTER_OF_MASS = 0
HOUGH_CIRCLES = 1

x = np.linspace(0, 1, 256)
y = np.linspace(0, 1, 256)
_X, _Y = np.meshgrid(x, y)

def detectGreenCircle(image, method=CENTER_OF_MASS):
    '''Detects a red, a green and a blue circle in image'''
    image_flipped = cv2.flip(image, 0)

    threshold = 250
    
    # Split color channels
    image_green = np.logical_and(image_flipped[:,:,1] > threshold, np.logical_and(image_flipped[:,:,0] < threshold, image_flipped[:,:,2] < threshold))
    
    # Treatment for opencv
    image_green = 255 * image_green.astype(np.uint8)
    
    f = np.zeros(2)

    if method == CENTER_OF_MASS:
        f[0] = 255 * np.sum(_X*image_green)/np.sum(image_green)
        f[1] = 255 * np.sum(_Y*image_green)/np.sum(image_green)
    elif method == HOUGH_CIRCLES:
        # Detecting circles with Hough Circles
        # REMEMBER: If maxRadius < 0, HOUGH_GRADIENT returns centers without finding the radius, we don't need radius
        k = 0
        k_max = 5

        circles_green = None
        while circles_green is None and k < k_max:
            # A smoother border aids HoughCircles
            image_green = cv2.GaussianBlur(image_green, (5,5), 0)
            circles_green = cv2.HoughCircles(image_green, cv2.HOUGH_GRADIENT, 2, 100, param1=50,param2=30,minRadius=0,maxRadius=-1)
            k += 1
        
        if (circles_green is None):
            raise Exception('problem in hough circles')

        # Defining features vector
        f[0] = circles_green[0][0][0]
        f[1] = circles_green[0][0][1]
    
    else:
        raise Exception('Unknown method')


    return f

def detectRGBCircles(image, method=CENTER_OF_MASS):
    '''Detects a red, a green and a blue circle in image'''
    image_flipped = cv2.flip(image, 0)

    threshold = 250
    
    # Split color channels
    image_red = np.logical_and(image_flipped[:,:,0] > threshold, np.logical_and(image_flipped[:,:,1] < threshold, image_flipped[:,:,2] < threshold))
    image_green = np.logical_and(image_flipped[:,:,1] > threshold, np.logical_and(image_flipped[:,:,0] < threshold, image_flipped[:,:,2] < threshold))
    image_blue = np.logical_and(image_flipped[:,:,2] > threshold, np.logical_and(image_flipped[:,:,0] < threshold, image_flipped[:,:,1] < threshold))
    
    # Treatment for opencv
    image_red = 255 * image_red.astype(np.uint8)
    image_green = 255 * image_green.astype(np.uint8)
    image_blue = 255 * image_blue.astype(np.uint8)
    
    f = np.zeros(6)

    if method == CENTER_OF_MASS:
        f[0] = 255 * np.sum(_X*image_red)/np.sum(image_red)
        f[1] = 255 * np.sum(_Y*image_red)/np.sum(image_red)
        f[2] = 255 * np.sum(_X*image_green)/np.sum(image_green)
        f[3] = 255 * np.sum(_Y*image_green)/np.sum(image_green)
        f[4] = 255 * np.sum(_X*image_blue)/np.sum(image_blue)
        f[5] = 255 * np.sum(_Y*image_blue)/np.sum(image_blue)
    elif method == HOUGH_CIRCLES:
        # Detecting circles with Hough Circles
        # REMEMBER: If maxRadius < 0, HOUGH_GRADIENT returns centers without finding the radius, we don't need radius
        k = 0
        k_max = 5
        circles_red = None
        while circles_red is None and k < k_max:
            # A smoother border aids HoughCircles
            image_red = cv2.GaussianBlur(image_red, (5,5), 0)
            circles_red = cv2.HoughCircles(image_red, cv2.HOUGH_GRADIENT, 2, 100, param1=50,param2=30,minRadius=0,maxRadius=-1)
            k += 1

        k = 0
        circles_green = None
        while circles_green is None and k < k_max:
            # A smoother border aids HoughCircles
            image_green = cv2.GaussianBlur(image_green, (5,5), 0)
            circles_green = cv2.HoughCircles(image_green, cv2.HOUGH_GRADIENT, 2, 100, param1=50,param2=30,minRadius=0,maxRadius=-1)
            k += 1

        k = 0
        circles_blue = None
        while circles_blue is None and k < k_max:
            # A smoother border aids HoughCircles
            image_blue = cv2.GaussianBlur(image_blue, (5,5), 0)
            circles_blue = cv2.HoughCircles(image_blue, cv2.HOUGH_GRADIENT, 2, 100, param1=50,param2=30,minRadius=0,maxRadius=-1)
            k += 1
        
        if (circles_red is None) or (circles_green is None) or (circles_blue is None):
            raise Exception('problem in hough circles')

        # Defining features vector
        f[0] = circles_red[0][0][0]
        f[1] = circles_red[0][0][1]
        f[2] = circles_green[0][0][0]
        f[3] = circles_green[0][0][1]
        f[4] = circles_blue[0][0][0]
        f[5] = circles_blue[0][0][1]
    
    else:
        raise Exception('Unknown method')


    return f

def detect4Circles(image, method=CENTER_OF_MASS):
    '''Detects a red, a green, a blue and a pink circle in image'''
    f = np.zeros(8)
    try:
        f[0:6] = detectRGBCircles(image, method)
    except Exception as e:
        raise e
    
    image_flipped = cv2.flip(image, 0)

    threshold = 250

    # Split color channels
    image_pink = np.logical_and(image_flipped[:,:,0] > threshold, np.logical_and(image_flipped[:,:,1] < threshold, image_flipped[:,:,2] > threshold))
    
    # Treatment for opencv
    image_pink = 255 * image_pink.astype(np.uint8)

    if method == CENTER_OF_MASS:
        f[6] = 255 * np.sum(_X*image_pink)/np.sum(image_pink)
        f[7] = 255 * np.sum(_Y*image_pink)/np.sum(image_pink)
        
    elif method == HOUGH_CIRCLES:
        # Detecting circles with Hough Circles
        # REMEMBER: If maxRadius < 0, HOUGH_GRADIENT returns centers without finding the radius, we don't need radius
        k = 0
        k_max = 5
        circles_pink = None
        while circles_pink is None and k < k_max:
            # A smoother border aids HoughCircles
            image_pink = cv2.GaussianBlur(image_pink, (5,5), 0)
            circles_pink = cv2.HoughCircles(image_pink, cv2.HOUGH_GRADIENT, 2, 100, param1=50,param2=30,minRadius=0,maxRadius=-1)
            k += 1

        if (circles_pink is None):
            raise Exception('problem in hough circles')
    
        # Defining features vector
        f[6] = circles_pink[0][0][0]
        f[7] = circles_pink[0][0][1]
    else:
        raise Exception('Unknown method')
    return f

def saveSampleImage(image, name):
    cv2.imwrite(name, image)

def gaussianKernel(e, bw):
    return np.exp(-0.5 * e**2 / bw**2)

def quat2euler(h):
    roll = np.arctan2(2*(h[0]*h[1] + h[2]*h[3]), 1 - 2*(h[1]**2 + h[2]**2))
    pitch = np.arcsin(2*(h[0]*h[2] - h[3]*h[1]))
    yaw = np.arctan2(2*(h[0]*h[3] + h[1]*h[2]), 1 - 2*(h[2]**2 + h[3]**2))

    return (roll, pitch, yaw)