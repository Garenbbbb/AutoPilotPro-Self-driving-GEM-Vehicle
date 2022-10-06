import cv2
import time
import math
import copy
import rospy
import numpy as np

from skimage import morphology
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from gazebo_msgs.srv import GetModelState, GetModelStateResponse

from Line import Line
from line_fit import line_fit, tune_fit, bird_fit, final_viz

class VehiclePerception:
    def __init__(self, model_name='gem', resolution=0.1, side_range=(-20., 20.),
            fwd_range=(-20., 20.), height_range=(-1.6, 0.5)):
        self.lane_detector = lanenet_detector()
        self.lidar = LidarProcessing(resolution=resolution, side_range=side_range, fwd_range=fwd_range, height_range=height_range)

        self.bridge = CvBridge()
        self.model_name = model_name

    def cameraReading(self):
        # Get processed reading from the camera on the vehicle
        # Input: None
        # Output:
        # 1. Lateral tracking error from the center line of the lane
        # 2. The lane heading with respect to the vehicle
        return self.lane_detector.lateral_error, self.lane_detector.lane_theta

    def lidarReading(self):
        # Get processed reading from the Lidar on the vehicle
        # Input: None
        # Output: Distance between the vehicle and object in the front
        res = self.lidar.processLidar()
        return res

    def gpsReading(self):
        # Get the current state of the vehicle
        # Input: None
        # Output: ModelState, the state of the vehicle, contain the
        #   position, orientation, linear velocity, angular velocity
        #   of the vehicle
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            serviceResponse = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            modelState = serviceResponse(model_name=self.model_name)
        except rospy.ServiceException as exc:
            rospy.loginfo("Service did not process request: "+str(exc))
            modelState = GetModelStateResponse()
            modelState.success = False
        return modelState

class LidarProcessing:
    def __init__(self, resolution=0.1, side_range=(-20., 20.), fwd_range=(-20., 20.),
                         height_range=(-1.6, 0.5)):
        self.resolution = resolution
        self.side_range = side_range
        self.fwd_range = fwd_range
        self.height_range = height_range

        self.cvBridge = CvBridge()

        # empty initial image
        self.birdsEyeViewPub = rospy.Publisher("/mp4/BirdsEye", Image, queue_size=1)
        self.pointCloudSub = rospy.Subscriber("/velodyne_points", PointCloud2, self.__pointCloudHandler, queue_size=10)
        x_img = np.floor(-0 / self.resolution).astype(np.int32)
        self.vehicle_x = x_img - int(np.floor(self.side_range[0] / self.resolution))

        y_img = np.floor(-0 / self.resolution).astype(np.int32)
        self.vehicle_y = y_img + int(np.ceil(self.fwd_range[1] / self.resolution))


        self.x_front = float('nan')
        self.y_front = float('nan')

    def __pointCloudHandler(self, data):
        """
            Callback function for whenever the lidar point clouds are detected

            Input: data - lidar point cloud

            Output: None

            Side Effects: updates the birds eye view image
        """
        gen = point_cloud2.readgen = point_cloud2.read_points(cloud=data, field_names=('x', 'y', 'z', 'ring'))

        lidarPtBV = []
        for p in gen:
            lidarPtBV.append((p[0],p[1],p[2]))

        self.construct_birds_eye_view(lidarPtBV)

    def construct_birds_eye_view(self, data):
        """
            Call back function that get the distance between vehicle and nearest wall in given direction
            The calculated values are stored in the class member variables

            Input: data - lidar point cloud
        """
        # create image from_array
        x_max = 1 + int((self.side_range[1] - self.side_range[0]) / self.resolution)
        y_max = 1 + int((self.fwd_range[1] - self.fwd_range[0]) / self.resolution)
        im = np.zeros([y_max, x_max], dtype=np.uint8)

        if len(data) == 0:
            return im

        # Reference: http://ronny.rest/tutorials/module/pointclouds_01/point_cloud_birdseye/
        data = np.array(data)

        x_points = data[:, 0]
        y_points = data[:, 1]
        z_points = data[:, 2]

        # Only keep points in the range specified above
        x_filter = np.logical_and((x_points >= self.fwd_range[0]), (x_points <= self.fwd_range[1]))
        y_filter = np.logical_and((y_points >= self.side_range[0]), (y_points <= self.side_range[1]))
        z_filter = np.logical_and((z_points >= self.height_range[0]), (z_points <= self.height_range[1]))

        filter = np.logical_and(x_filter, y_filter)
        filter = np.logical_and(filter, z_filter)
        indices = np.argwhere(filter).flatten()

        x_points = x_points[indices]
        y_points = y_points[indices]
        z_points = z_points[indices]

        def scale_to_255(a, min_val, max_val, dtype=np.uint8):
            a = (((a-min_val) / float(max_val - min_val) ) * 255).astype(dtype)
            tmp = copy.deepcopy(a)
            a[:] = 0
            a[tmp>0] = 255
            return a

        # clip based on height for pixel Values
        pixel_vals = np.clip(a=z_points, a_min=self.height_range[0], a_max=self.height_range[1])

        pixel_vals = scale_to_255(pixel_vals, min_val=self.height_range[0], max_val=self.height_range[1])

        # Getting sensor reading for front
        filter_front = np.logical_and((y_points>-2), (y_points<2))
        filter_front = np.logical_and(filter_front, x_points > 0)
        filter_front = np.logical_and(filter_front, pixel_vals > 128)
        indices = np.argwhere(filter_front).flatten()

        self.x_front = np.mean(x_points[indices])
        self.y_front = np.mean(y_points[indices])

        # convert points to image coords with resolution
        x_img = np.floor(-y_points / self.resolution).astype(np.int32)
        y_img = np.floor(-x_points / self.resolution).astype(np.int32)

        # shift coords to new original
        x_img -= int(np.floor(self.side_range[0] / self.resolution))
        y_img += int(np.ceil(self.fwd_range[1] / self.resolution))

        # Generate a visualization for the perception result
        im[y_img, x_img] = pixel_vals

        img = im.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        center = (self.vehicle_x, self.vehicle_y)
        cv2.circle(img, center, 5, (0,0,255), -1, 8, 0)

        center = self.convert_to_image(self.x_front, self.y_front)
        cv2.circle(img, center, 5, (0,255,0), -1, 8, 0)
        if not np.isnan(self.x_front) and not np.isnan(self.y_front):
            cv2.arrowedLine(img, (self.vehicle_x,self.vehicle_y), center, (255,0,0))

        x1, y1 = self.convert_to_image(20,2)
        x2, y2 = self.convert_to_image(0,-2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0))

        birds_eye_im = self.cvBridge.cv2_to_imgmsg(img, 'bgr8')

        self.birdsEyeViewPub.publish(birds_eye_im)


    def convert_to_image(self, x, y):
        """
            Convert point in vehicle frame to position in image frame
            Inputs:
                x: float, the x position of point in vehicle frame
                y: float, the y position of point in vehicle frame
            Outputs: Float, the x y position of point in image frame
        """

        x_img = np.floor(-y / self.resolution).astype(np.int32)
        y_img = np.floor(-x / self.resolution).astype(np.int32)

        x_img -= int(np.floor(self.side_range[0] / self.resolution))
        y_img += int(np.ceil(self.fwd_range[1] / self.resolution))
        return (x_img, y_img)

    def processLidar(self):
        """
            Compute the distance between vehicle and object in the front
            Inputs: None
            Outputs: Float, distance between vehicle and object in the front
        """
        front = np.sqrt(self.x_front ** 2 + self.y_front ** 2)

        return front

    def get_lidar_reading(self):
        return self.x_front, self.y_front

class lanenet_detector():
    def __init__(self):

        self.bridge = CvBridge()
        self.sub_image = rospy.Subscriber('/gem/front_single_camera/front_single_camera/image_raw', Image, self.img_callback, queue_size=1)
        self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)
        self.left_line = Line(n=5)
        self.right_line = Line(n=5)
        self.detected = False

        # initialization for the lateral tracking error and lane heading
        self.lateral_error = 0.0
        self.lane_theta = 0.0

        # determine the meter-to-pixel ratio
        lane_width_meters = 4.4
        lane_width_pixels = 265.634
        self.meter_per_pixel = lane_width_meters / lane_width_pixels


    def img_callback(self, data):

        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()
        mask_image, bird_image, lateral_error, lane_theta = self.detection(raw_img)

        if mask_image is not None and bird_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')

            # Publish image message in ROS
            self.pub_image.publish(out_img_msg)
            self.pub_bird.publish(out_bird_msg)

        # publish the lateral tracking error and lane heading
        if lateral_error is not None:
            self.lateral_error = lateral_error

        if lane_theta is not None:
            self.lane_theta = lane_theta


    def gradient_thresh(self, img, thresh_min=25, thresh_max=100):
        """
        Apply sobel edge detection on input image in x, y direction
        """
        # TODO: Use your MP1 implementation for this function

        # thresh_min higher OH

        ddepth = cv2.CV_16S
        # ddepth = cv2.CV_64F # OH
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gauss = cv2.GaussianBlur(gray, (15, 15), sigmaX=2, sigmaY=2)
        # gauss = c/v2.GaussianBlur(gray, (3, 3), sigmaX=2, sigmaY=2)

        grad_x = cv2.Sobel(gauss, ddepth, 1, 0)
        grad_y = cv2.Sobel(gauss, ddepth, 0, 1)

        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        # check 0 - 255
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        mask = np.logical_and(grad >= thresh_min, grad <= thresh_max).astype(np.uint8)

        return mask



    def color_thresh(self, img, thresh=(20, 70)):
        """
        Convert RGB to HSL and threshold to binary image using S channel
        """
        # TODO: Use your MP1 implementation for this function
        threshold_lower, threshold_upper = thresh
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # mask for yellow
        mask_yellow = cv2.inRange(img_hsv, (15,0,0), (36, 255, 255))
        
        # mask for white
        lower_white = np.array([0,0,0], dtype=np.uint8)
        upper_white = np.array([0,0,255], dtype=np.uint8)
        mask_white = cv2.inRange(img_hsv, lower_white, upper_white)

        # normalize between 0 - 1
        mask = cv2.bitwise_or(mask_yellow, mask_white)
        mask[mask>= 1] = 1

        return mask



    def combinedBinaryImage(self, img):
        """
        Get combined binary image from color filter and sobel filter
        """
        # TODO: Use your MP1 implementation for this function
        SobelOutput = self.gradient_thresh(img)
        ColorOutput = self.color_thresh(img)
        # r_binary, s_binary = self.color_thresh(img)

        ####

        binaryImage = np.zeros_like(SobelOutput)
        binaryImage[(ColorOutput == 1) & (SobelOutput == 1)] = 1
        # binaryImage[(ColorOutput == 1) | (SobelOutput == 1)] = 1 # OH
       
        # binaryImage[((s_binary == 1) | (r_binary == 1)) & (SobelOutput == 1)] = 1

        # Remove noise from binary image
        binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'), min_size=50, connectivity=2)

        return binaryImage.astype(np.uint8)



    def perspective_transform(self, img, verbose=False):
        """
        Get bird's eye view from input image
        """
        # TODO: Use your MP1 implementation for this function
        height, width = img.shape

        padding = int(0.20 * width)

        pts_src = np.array([
            [290, 259],
            [348, 259],
            [650, 451],
            [10, 451]
        ], np.float32)

        pts_dst = np.array([
            [padding, 0],
            [width - 1 - padding, 0],
            [width - 1 - padding, height - 1],
            [padding, height - 1]
        ], np.float32)

        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        Minv = cv2.getPerspectiveTransform(pts_dst, pts_src)

        warped_img = cv2.warpPerspective(img, M, dsize=img.shape[::-1], flags=cv2.INTER_LINEAR)

        return warped_img, M, Minv



    def detection(self, img):

        binary_img = self.combinedBinaryImage(img)
        img_birdeye, M, Minv = self.perspective_transform(binary_img)
        h, w = img_birdeye.shape

        # Fit lane with the newest image
        ret = line_fit(img_birdeye)

        if ret is not None:
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            left_lane_inds = ret['left_lane_inds']
            right_lane_inds = ret['right_lane_inds']

            left_fit = self.left_line.add_fit(left_fit)
            right_fit = self.right_line.add_fit(right_fit)

        # return lane detection results
        bird_fit_img = None
        combine_fit_img = None
        lateral_error = None
        lane_theta = None
        if ret is not None:
            bird_fit_img = bird_fit(img_birdeye, ret, save_file=None)
            combine_fit_img = final_viz(img, left_fit, right_fit, Minv)

            # instead of estimating the lateral tracking error and the lane heading separately at
            # different locations in the bird's eye view image as shown below, one can set a point
            # along the center line in front of the vehicle as a reference point and use a similar
            # controller as in MP2.

            # TODO :calculate the lateral tracking error from the center line
            # Hint: positive error should occur when the vehicle is to the right of the center line
            A_left, B_left, C_left = left_fit
            A_right, B_right, C_right = right_fit

            lookahead_y = h / 2

            if (abs((A_left * h ** 2 + B_left * h + C_left) - (A_right * h ** 2 + B_right * h + C_right)) <= w / 2): lateral_error = None
            else:
                lane_middle_x = ((A_left * h ** 2 + B_left * h + C_left) + (A_right * h ** 2 + B_right * h + C_right)) / 2
                lateral_error = (- lane_middle_x + w / 2) * self.meter_per_pixel

            # TODO: calculate the lane heading error
            # Hint: use the lane heading a few meters before the vehicle to avoid oscillation
            # 2 * A * y + B
            left_theta = np.arctan(2 * A_left * lookahead_y + B_left)
            right_theta = np.arctan(2 * A_right * lookahead_y + B_right)
            lane_theta = (left_theta + right_theta) / 2
            
        else:
            print("Unable to detect lanes")

        return combine_fit_img, bird_fit_img, lateral_error, lane_theta

if __name__ == "__main__":
    VehiclePerception('gem1')

    rospy.spin()
