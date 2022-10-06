import time
import math
import numpy as np
import cv2
import rospy

from line_fit import line_fit, tune_fit, bird_fit, final_viz
from Line import Line
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from skimage import morphology
import matplotlib.pyplot as plt
from argparse import ArgumentParser


class lanenet_detector():
    def __init__(self, args):
        self.args = args

        self.bridge = CvBridge()
        # NOTE
        # Uncomment this line for lane detection of GEM car in Gazebo
        # self.sub_image = rospy.Subscriber('/gem/front_single_camera/front_single_camera/image_raw', Image,
        #                                   self.img_callback, queue_size=1)
        # Uncomment this line for lane detection of videos in rosbag
        self.sub_image = rospy.Subscriber('camera/image_raw', Image, self.img_callback, queue_size=1)
        self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)
        self.left_line = Line(n=5)
        self.right_line = Line(n=5)
        self.detected = False
        self.hist = True

    def img_callback(self, data):

        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()
        mask_image, bird_image = self.detection(raw_img)

        if mask_image is not None and bird_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')

            # Publish image message in ROS
            self.pub_image.publish(out_img_msg)
            self.pub_bird.publish(out_bird_msg)

    def gradient_thresh(self, img):
        """
        Apply sobel edge detection on input image in x, y direction
        """
        # 1. Convert the image to gray scale
        # 2. Gaussian blur the image
        # 3. Use cv2.Sobel() to find derievatives for both X and Y Axis
        # 4. Use cv2.addWeighted() to combine the results
        # 5. Convert each pixel to unint8, then apply threshold to get binary image

        ## TODO

        ####
        thresh_min, thresh_max = self.args.gradient_low, self.args.gradient_high
        fig = plt.figure()
        plt.imshow(img)
        plt.savefig("/home/jiahuiw4/Desktop/img/img.png")
        cv2.imwrite("/home/jiahuiw4/Desktop/img/orig_img.png", img)
        plt.close(fig)

        ddepth = cv2.CV_16S
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gauss = cv2.GaussianBlur(gray, (15, 15), sigmaX=2, sigmaY=2)

        grad_x = cv2.Sobel(gauss, ddepth, 1, 0)
        grad_y = cv2.Sobel(gauss, ddepth, 0, 1)

        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        mask = np.logical_and(grad >= thresh_min, grad <= thresh_max).astype(np.uint8)

        return mask


    def color_thresh(self, img):
        """
        Convert RGB to HSL and threshold to binary image using S channel
        """
        # 1. Convert the image from RGB to HSL
        # 2. Apply threshold on S channel to get binary image
        # Hint: threshold on H to remove green grass
        ## TODO
        threshold_lower, threshold_upper = self.args.color_low, self.args.color_high
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        binary_output = np.logical_and(img[:, :, -1] >= threshold_lower, img[:, :, -1] <= threshold_upper).astype(np.uint8)
        ####

        return binary_output


    def combinedBinaryImage(self, img):
        """
        Get combined binary image from color filter and sobel filter
        """
        # 1. Apply sobel filter and color filter on input image
        # 2. Combine the outputs
        ## Here you can use as many methods as you want.

        ## TODO
        SobelOutput = self.gradient_thresh(img)
        ColorOutput = self.color_thresh(img)

        ####

        binaryImage = np.zeros_like(SobelOutput)
        binaryImage[(ColorOutput == 1) | (SobelOutput == 1)] = 1
        # Remove noise from binary image
        binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'), min_size=50, connectivity=2)

        return binaryImage.astype(np.uint8)


    def perspective_transform(self, img, verbose=False):
        """
        Get bird's eye view from input image
        """
        # 1. Visually determine 4 source points and 4 destination points
        # 2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
        # 3. Generate warped image in bird view using cv2.warpPerspective()

        ## TODO
        # plt.imshow(img, cmap="gray")
        # plt.show()
        ####

        # 480 * 640
        height, width = img.shape
        print("orig img shape", img.shape)
        fig = plt.figure()
        plt.imshow(img, cmap="gray")
        plt.savefig("/home/jiahuiw4/Desktop/img/before_transform.png")
        # cv2.imwrite("/home/jiahuiw4/Desktop/img/before_transform.png", img.astype('float32'))
        plt.close(fig)
        print(img)

        padding = int(0.25 * width)

        # pts_src = np.array([
        #     [303, 251],
        #     [337, 251],
        #     [638, 451],
        #     [10, 451]
        # ], np.float32)

        pts_src = np.array([
            [290, 259],
            [348, 259],
            [638, 451],
            [10, 451]
        ], np.float32)

        pts_dst = np.array([
            [0, 0],
            [width - 1 - padding, 0],
            [width - 1 - padding, height - 1],
            [0, height - 1]
        ], np.float32)

        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        Minv = cv2.getPerspectiveTransform(pts_dst, pts_src)

        warped_img = cv2.warpPerspective(img, M, dsize=img.shape[::-1], flags=cv2.INTER_LINEAR)

        fig = plt.figure()
        plt.imshow(warped_img.astype(np.uint8), cmap="gray")
        plt.savefig("/home/jiahuiw4/Desktop/img/bird.png")
        plt.close(fig)

        return warped_img, M, Minv

    def detection(self, img):
        binary_img = self.combinedBinaryImage(img)
        img_birdeye, M, Minv = self.perspective_transform(binary_img)

        if not self.hist:
            # Fit lane without previous result
            ret = line_fit(img_birdeye)
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            left_lane_inds = ret['left_lane_inds']
            right_lane_inds = ret['right_lane_inds']

        else:
            # Fit lane with previous result
            if not self.detected:
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

                    self.detected = True

            else:
                left_fit = self.left_line.get_fit()
                right_fit = self.right_line.get_fit()
                ret = tune_fit(img_birdeye, left_fit, right_fit)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                else:
                    self.detected = False

            # Annotate original image
            bird_fit_img = None
            combine_fit_img = None
            if ret is not None:

                bird_fit_img = bird_fit(img_birdeye, ret, save_file=None)
                combine_fit_img = final_viz(img, left_fit, right_fit, Minv)

                fig = plt.figure()
                plt.imshow(bird_fit_img.astype(np.uint8), cmap="gray")
                plt.savefig("/home/jiahuiw4/Desktop/img/birdz-fit.png")
                plt.close(fig)

                # fig, ax = plt.subplots(ncols=3)
                # ax[0].imshow(img_birdeye)
                # ax[1].imshow(bird_fit_img)
                # ax[2].imshow(combine_fit_img)
                # plt.show()
            else:
                print("Unable to detect lanes")

            return combine_fit_img, bird_fit_img


if __name__ == '__main__':
    # init args
    parser = ArgumentParser()
    parser.add_argument("gradient_low", type=int, default=100)
    parser.add_argument("gradient_high", type=int, default=255)
    parser.add_argument("color_low", type=int, default=100)
    parser.add_argument("color_high", type=int, default=255)
    args = parser.parse_args()
    print(args)
    
    rospy.init_node('lanenet_node', anonymous=True)
    lanenet_detector(args)
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)
