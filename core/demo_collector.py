import os
import pickle

from cv2 import ROTATE_180
import rospy

import rospy
import cv2
import numpy as np

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState, Image
from visualization_msgs.msg import Marker

from ik_teleop.ik_core.allegro_ik import AllegroInvKDL

from cv_bridge import CvBridge, CvBridgeError

# Human data
ABSOLUTE_POSE_COORD_TOPIC = '/absolute_mediapipe_joint_pixels'
MEDIAPIPE_RGB_IMG_TOPIC = '/mediapipe_rgb_image'
MEDIAPIPE_DEPTH_IMG_TOPIC = '/mediapipe_depth_image'

# Robot data
JOINT_STATE_TOPIC = "/allegroHand/joint_states"
MARKER_TOPIC = "/visualization_marker"
ROBOT_IMAGE_STATE_TOPIC = "/cam_1/color/image_raw"

# Intrinsics for the top view camera
ROTATION_CAMERA_INTRINSICS_MATRIX = [916.500732421875, 0.0, 630.3333129882812, 0.0, 916.3279418945312, 358.1403503417969, 0.0, 0.0, 1.0]
FLIPPING_CAMERA_INTRINSICS_MATRIX = [908.85107421875, 0.0, 637.8613891601562, 0.0, 909.2920532226562, 360.6807556152344, 0.0, 0.0, 1.0]
SPINNING_CAMERA_INTRINSICS_MATRIX = [916.9735717773438, 0.0, 626.5108642578125, 0.0, 916.377197265625, 357.61029052734375, 0.0, 0.0, 1.0]

# Utility functions
def check_dir(dir):
    if not os.path.exists(dir):
        print('Making directory: {}'.format(dir))
        os.makedirs(dir)
    else:
        print('{} - Directory already exists'.format(dir)) 

def store_pickle_data(pickle_path, data):
    file = open(pickle_path, 'ab')
    pickle.dump(data, file)
    file.close()

# Main class
class DemoCollector(object):
    def __init__(self, task, storage_path, frequency = 3):
        try:
            rospy.init_node("demo_collector")
        except:
            pass

        self.allegro_ik = AllegroInvKDL(urdf_path = "/home/vm/rpl/DIME-IK-TeleOp/ik_teleop/urdf_template/allegro_right.urdf")

        self.rate = rospy.Rate(frequency)
        self.task = task

        self.allegro_joint_state = None
        self.robot_image = None
        self.pose_rgb_image = None
        self.mp_joint_coords = None
        self.object_data = None

        if self.task == "rotate":
            self.hand_base_data = None

        self.storage_path = storage_path

        self.bridge = CvBridge()

        if self.task == "rotate":
            self.intrinsics = np.array(ROTATION_CAMERA_INTRINSICS_MATRIX).reshape((3, 3))
        elif self.task == "flip":
            self.intrinsics = np.array(FLIPPING_CAMERA_INTRINSICS_MATRIX).reshape((3, 3))
        elif self.task == "spin":
            self.intrinsics = np.array(SPINNING_CAMERA_INTRINSICS_MATRIX).reshape((3, 3))

        # Pose data subscribers
        rospy.Subscriber(MEDIAPIPE_RGB_IMG_TOPIC, Image, self._callback_pose_rgb_image, queue_size = 1)
        rospy.Subscriber(ABSOLUTE_POSE_COORD_TOPIC, Float64MultiArray, self._callback_mp_coords, queue_size = 1)

        # Robot data subscribers
        rospy.Subscriber(JOINT_STATE_TOPIC, JointState, self._callback_joint_state, queue_size = 1)
        rospy.Subscriber(ROBOT_IMAGE_STATE_TOPIC, Image, self._callback_robot_image, queue_size = 1)
        rospy.Subscriber(MARKER_TOPIC, Marker, self._callback_ar_marker_data, queue_size = 1)

    def _callback_ar_marker_data(self, data):
        if data.id == 0 or data.id == 5:
            self.object_data = data
        elif data.id == 8:
            self.hand_base_data = data

    def _callback_joint_state(self, data):
        self.allegro_joint_state = data

    def _callback_mp_coords(self, data):
        self.mp_joint_coords = np.array(data.data).reshape(11, 2) * np.array([1280, 720])

    def _callback_robot_image(self, data):
        try:
            self.robot_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def _callback_pose_rgb_image(self, data):
        try:
            self.pose_rgb_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def get_ar_marker_data(self):
        if self.task == "rotate":
            object_position = np.array([self.object_data.pose.position.x, self.object_data.pose.position.y, self.object_data.pose.position.z]) - np.array([self.hand_base_data.pose.position.x, self.hand_base_data.pose.position.y, self.hand_base_data.pose.position.z])
            object_orientation = np.array([self.object_data.pose.orientation.x, self.object_data.pose.orientation.y, self.object_data.pose.orientation.z, self.object_data.pose.orientation.w])
            return object_position, object_orientation
        elif self.task == "flip":
            object_position = np.array([self.object_data.pose.position.x, self.object_data.pose.position.y, self.object_data.pose.position.z])
            object_orientation = np.array([self.object_data.pose.orientation.x, self.object_data.pose.orientation.y, self.object_data.pose.orientation.z, self.object_data.pose.orientation.w])
            return object_position, object_orientation

    def calculate_pixels(self, coordinates):
        pixel_values = np.matmul(self.intrinsics, np.array(coordinates))
        pixel_values = pixel_values / pixel_values[-1]
        return int(pixel_values[0]), int(pixel_values[1])

    def get_tip_coords(self, joint_angles):
        index_coord = self.allegro_ik.finger_forward_kinematics('index', joint_angles[:4])[0]
        middle_coord = self.allegro_ik.finger_forward_kinematics('middle', joint_angles[4:8])[0]
        ring_coord = self.allegro_ik.finger_forward_kinematics('ring', joint_angles[8:12])[0]
        thumb_coord = self.allegro_ik.finger_forward_kinematics('thumb', joint_angles[12:])[0]
        return list(index_coord) + list(middle_coord) + list(ring_coord) + list(thumb_coord)

    def collect_data(self, demo_number = 0):
        state_number = 1
        demo_storage_path = os.path.join(self.storage_path, 'demonstration_{}'.format(demo_number))
        check_dir(demo_storage_path)

        rospy.sleep(2)

        while True:
            if self.pose_rgb_image is None:
                # print("No hand pose RGB image")
                continue

            if self.mp_joint_coords is None:
                # print("No Mediapipe Pixels predictions")
                continue

            if self.task == "rotate" or self.task == "flip":
                if self.object_data is None:
                    print("Object not found")
                    continue

            if self.task == "rotate":
                if self.hand_base_data is None:
                    print("Hand base tracker not found!")
            
            if self.allegro_joint_state is None:
                print("No Joint State value")
                continue

            if self.robot_image is None:
                print("No input image")
                continue

            state = {}

            state['state_number'] = state_number

            # Hand pose data
            state['hand_pose_pixels'] = self.mp_joint_coords
            state['hand_pose_rgb_image'] = self.pose_rgb_image

            # Object data
            if self.task == "rotate" or self.task == "flip":
                state['object_position_coordinates'], state['object_orientation'] = self.get_ar_marker_data()
                ar_tracker_x_pixel, ar_tracker_y_pixel = self.calculate_pixels(np.array([
                    self.object_data.pose.position.x,
                    self.object_data.pose.position.y,
                    self.object_data.pose.position.z
                ]))

                if self.task == "rotate":
                    hand_base_x_pixel, hand_base_y_pixel = self.calculate_pixels(np.array([
                        self.hand_base_data.pose.position.x,
                        self.hand_base_data.pose.position.y,
                        self.hand_base_data.pose.position.z
                    ]))

                state['object_position_pixels'] = [ar_tracker_x_pixel, ar_tracker_y_pixel]

            # Robot data
            state['joint_angles'], state['joint_velocities'], state['joint_torques'] = np.array(self.allegro_joint_state.position), np.array(self.allegro_joint_state.velocity), np.array(self.allegro_joint_state.effort)
            state['finger_tip_coords'] = self.get_tip_coords(np.array(self.allegro_joint_state.position))
            state['robot_image'] = self.robot_image

            # Storing data in a pickle file
            state_pickle_path = os.path.join(demo_storage_path, '{}'.format(state_number))
            store_pickle_data(state_pickle_path, state)

            state_number += 1

            # Embedding the AR Tracker pixels
            if self.task == "rotate" or self.task == "flip":
                self.robot_image = cv2.line(self.robot_image, (ar_tracker_x_pixel, 0), (ar_tracker_x_pixel, 720), (255,255,0), 2)
                self.robot_image = cv2.line(self.robot_image, (0, ar_tracker_y_pixel), (1280, ar_tracker_y_pixel), (255,255,0), 2)

            if self.task == "rotate":
                self.robot_image = cv2.line(self.robot_image, (hand_base_x_pixel, 0), (hand_base_x_pixel, 720), (255,0,0), 2)
                self.robot_image = cv2.line(self.robot_image, (0, hand_base_y_pixel), (1280, hand_base_y_pixel), (255,0,0), 2)

            display_image = cv2.rotate(self.robot_image, cv2.ROTATE_180)

            cv2.imshow("Image", display_image)
            cv2.waitKey(1)

            self.rate.sleep()
