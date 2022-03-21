import cv2
import numpy as np

def video_to_frames(path):
     frames_list = []
     videoCapture = cv2.VideoCapture()
     videoCapture.open(path)
     fps = videoCapture.get(cv2.CAP_PROP_FPS) 
     frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
     for i in range(int(frames)):
          ret, frame = videoCapture.read()
          frames_list.append(frame)
     return frames_list

### example code

# binary mask for 'colors'

def get_lane_lines_mask(hsv_image, colors):
    masks = []
    for color in colors:
        if 'low_th' in color and 'high_th' in color:
            mask = cv2.inRange(hsv_image, color['low_th'], color['high_th'])
            masks.append(mask)
    return cv2.add(*masks)
    


# apply mask
def draw_binary_mask(binary_mask, img):
    masked_image = np.zeros_like(img)
    for i in range(3): 
        masked_image[:,:,i] = binary_mask.copy()
    return masked_image


sensitivity = 90
lower_white = np.array([0,0,255-sensitivity])
upper_white = np.array([255,sensitivity,255])
lower_yellow = np.array([41,170,100])
upper_yellow = np.array([255,255,255])


WHITE_LINES = { 'low_th': lower_white,
                'high_th': upper_white}

YELLOW_LINES = { 'low_th': lower_yellow,
                 'high_th': upper_yellow}




def draw_lanes_on_img(i, history):
    hsv_image = cv2.cvtColor(i, cv2.COLOR_RGB2HSV) 

    binary_mask = get_lane_lines_mask(hsv_image, [WHITE_LINES, YELLOW_LINES])
    masked_image = draw_binary_mask(binary_mask, hsv_image)

    edges_mask = cv2.Canny(masked_image, 50, 150)
    blank_image = np.zeros_like(i)
    #edges_img = draw_binary_mask(edges_mask, blank_image)

    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges_mask)   
    ignore_mask_color = 255
    # Define a four sided polygon to mask
    imshape = i.shape
    vertices = np.array([
        [
            (0,imshape[0]), # bottom left
            (0, imshape[0] * .5),  # top left 
            (imshape[1], imshape[0] * .5), # top right
            (imshape[1],imshape[0]) # bottom right
        ]
    ], dtype=np.int32)
    
    # Do the Masking
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges_mask, mask)


    # Define the Hough transform parameters
    rho = 2 # distance resolution in pixels
    theta = np.pi/180 # angular resolution in radians
    threshold = 30 # minimum number of votes 
    min_line_length = 20 # minimum number of pixels making up a line
    max_line_gap = 1 # maximum gap in pixels between lines

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of lines
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    mxb = np.array([[0,0]])
    if(lines is not None):
        # Iterate over the output "lines" to calculate m's and b's 
        for line in lines:
            for x1,y1,x2,y2 in line:
                m = (y2-y1) / (x2-x1)
                b = y1 + -1 * m * x1   
                mxb = np.vstack((mxb, [m, b]))


    median_right_m = np.median(mxb[mxb[:,0] > 0,0])
    median_left_m = np.median(mxb[mxb[:,0] < 0,0])
    median_right_b = np.median(mxb[mxb[:,0] > 0,1])
    median_left_b = np.median(mxb[mxb[:,0] < 0,1])

    # Calculate the Intersect point of our two lines
    x_intersect = (median_left_b - median_right_b) / (median_right_m - median_left_m)
    y_intersect = median_right_m * (median_left_b - median_right_b) / (median_right_m - median_left_m) + median_right_b
    # Calculate the X-Intercept Points
    # x = (y - b) / m
    left_bottom = (imshape[0] - median_left_b) / median_left_m
    right_bottom = (imshape[0] - median_right_b) / median_right_m

    history = np.array([[0, 0, 0, 0]]) # Initialize History
    images_list = []

    # Create a History array for smoothing
    num_frames_to_median = 19
    new_history = [left_bottom, right_bottom, x_intersect, y_intersect]
    if (history.shape[0] == 1): # First time, create larger array
        history = new_history
        for j in range(num_frames_to_median):
            history = np.vstack((history, new_history))
    elif (not(np.isnan(new_history).any())): 
        history[:-1,:] = history[1:]
        history[-1, :] = new_history
    # Calculate the smoothed line points
    left_bottom_median = np.median(history[:,0])
    right_bottom_median = np.median(history[:,1])
    x_intersect_median = np.median(history[:,2])
    y_intersect_median = np.median(history[:,3])

    # Create our Lines
    cv2.line(
        i,
        (np.int_(left_bottom_median), imshape[0]),  
        (np.int_(x_intersect_median), np.int_(y_intersect_median)),   
        (255,0,0),10
    )
    cv2.line(
        i,
        (np.int_(right_bottom_median), imshape[0]),
        (np.int_(x_intersect_median), np.int_(y_intersect_median)),
        (0,0,255),10
    )
    # Draw the lines on the image
    return [i, history]


# Process each video frame, note how "history" is passed around
path = "../20160310230308_morty.video.mp4"
frames_list = []
frames_list = video_to_frames(path)
history = np.array([[0, 0, 0, 0]]) # Initialize History
images_list = []

for frame in frames_list:
        [image, history] = draw_lanes_on_img(frame, history)
        images_list.append(image)
# Create a new ImageSequenceClip, in other words, our video!
height, width, layers = images_list[0].shape
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video = cv2.VideoWriter('../video.mp4',fourcc,60.0,(width,height))
_ = [video.write(i) for i in images_list]
video.release()
cv2.destroyAllWindows()

