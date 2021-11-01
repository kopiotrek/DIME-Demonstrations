import os
import pickle

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import JointState, Image
from visualization_msgs.msg import Marker

from cv_bridge import CvBridge, CvBridgeError

from IPython import embed

JOINT_STATE_TOPIC = "/allegroHand_0/joint_states"
MARKER_TOPIC = "/visualization_marker"

IMAGE_STATE_TOPIC = "/cam_1/color/image_raw"

CAMERA_INTRINSICS_MATRIX = [916.500732421875, 0.0, 630.3333129882812, 0.0, 916.3279418945312, 358.1403503417969, 0.0, 0.0, 1.0]
KINOVA_ARM_POSITION = [4.5575866699219, 211.01708984375, 47.678035736083984, 198.78610229492188, 96.60987091064453, 230.55865478515625, 0]

class DemoCollector(object):
    def __init__(self, frequency = 5):
        try:
            rospy.init_node("demo_collector")
        except:
            pass

        self.rate = rospy.Rate(frequency)

        self.ar_marker_data = None
        self.allegro_joint_state = None
        self.image = None

        self.storage_path = os.getcwd()

        self.bridge = CvBridge()

        self.intrinsics = np.array(CAMERA_INTRINSICS_MATRIX).reshape((3, 3))

        rospy.Subscriber(MARKER_TOPIC, Marker, self._callback_ar_marker_data, queue_size = 1)
        rospy.Subscriber(JOINT_STATE_TOPIC, JointState, self._callback_joint_state, queue_size = 1)
        rospy.Subscriber(IMAGE_STATE_TOPIC, Image, self._callback_image, queue_size=1)

    def _callback_ar_marker_data(self, data):
        self.ar_marker_data = data

    def _callback_joint_state(self, data):
        self.allegro_joint_state = data

    def _callback_image(self, data):
        try:
            self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def create_directory(self, demo_number):
        demo_directory = os.path.join(self.storage_path, 'demonstation_{}'.format(demo_number))
        if not os.path.exists(demo_directory):
            print('Making directory: {}'.format(demo_directory))
            os.makedirs(demo_directory)
        else:
            print('Directory already exists')

        return demo_directory

    def store_pickle_data(self, pickle_path, data):
        file = open(pickle_path, 'ab')

        pickle.dump(data, file)
        file.close()

    def get_ar_marker_data(self):
        cube_position = np.array([self.ar_marker_data.pose.position.x, self.ar_marker_data.pose.position.y, self.ar_marker_data.pose.position.z])
        cube_orientation = np.array([self.ar_marker_data.pose.orientation.x, self.ar_marker_data.pose.orientation.y, self.ar_marker_data.pose.orientation.z, self.ar_marker_data.pose.orientation.w])

        return cube_position, cube_orientation

    def calculate_pixels(self, coordinates):
        pixel_values = np.matmul(self.intrinsics, np.array(coordinates))
        pixel_values = pixel_values / pixel_values[-1]

        return int(pixel_values[0]), int(pixel_values[1])

    def collect_data(self, demo_number = 0):
        state_number = 1

        demo_storage_path = self.create_directory(demo_number)

        while True:
            try:
                if self.ar_marker_data is None:
                    print("No AR Tracker value")
                    continue
                
                if self.allegro_joint_state is None:
                    print("No Joint State Value")
                    continue

                if self.image is None:
                    print("No input image")
                    continue

                state = {}

                state['state_number'] = state_number

                state['cube_position_coordinates'], state['cube_orientation'] = self.get_ar_marker_data()
                state['joint_angles'], state['joint_velocities'] = np.array(self.allegro_joint_state.position), np.array(self.allegro_joint_state.velocity) 

                ar_tracker_x_pixel, ar_tracker_y_pixel = self.calculate_pixels(state['cube_position_coordinates'])

                state['cube_position_pixels'] = [ar_tracker_x_pixel, ar_tracker_y_pixel]

                state['image'] = self.image

                # Storing data in a pickle file
                state_pickle_path = os.path.join(demo_storage_path, '{}'.format(state_number))
                self.store_pickle_data(state_pickle_path, state)

                state_number += 1

                # Embedding the AR Tracker pixels
                self.image = cv2.line(self.image, (ar_tracker_x_pixel, 0), (ar_tracker_x_pixel, 720), (255,255,0), 2)
                self.image = cv2.line(self.image, (0, ar_tracker_y_pixel), (1280, ar_tracker_y_pixel), (255,255,0), 2)

                cv2.imshow("Image", self.image)
                cv2.waitKey(1)

                self.rate.sleep()
            
            except KeyboardInterrupt:
                print('Demostration recording stopped! Start the function over for next demo!')
                break
            

if __name__ == "__main__":
    allegro = DemoCollector()
    embed()