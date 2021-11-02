import os
import pickle
import cv2

import torch
import numpy as np

from allegro_ik import AllegroInvKDL

class FeatureExtractor():
    def __init__ (self, storage_dir, with_images = False):
        self.allegro_ik = AllegroInvKDL(cfg = None, urdf_path = "/home/sridhar/dexterous_arm/ik_stuff/urdf_template/allegro_right.urdf")

        self.storage_dir = storage_dir

        self.extract_images = with_images
        
        if not os.path.exists(self.storage_dir):
            print('Making directory: {}'.format(self.storage_dir))
            os.makedirs(self.storage_dir)
        else:
            print('Directory already exists')

    def create_directory(self, dir):
        if not os.path.exists(dir):
            print('Making directory: {}'.format(dir))
            os.makedirs(dir)
        else:
            print('Directory already exists: {}'.format(dir))

    def getting_data_dir_list(self, dir):
        files = os.listdir(dir)
        files.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

        return files

    def get_finger_tip_coords(self, joint_state):
        thumb_coordinates = self.allegro_ik.finger_forward_kinematics('thumb', joint_state[12:16])[0]
        pinky_coordinates = self.allegro_ik.finger_forward_kinematics('ring', joint_state[8:12])[0]
        hand_coordinates = list(thumb_coordinates) + list(pinky_coordinates)

        return hand_coordinates

    def get_state_data(self, demo_dir, demo_idx):
        # Getting the list of data        
        state_list = self.getting_data_dir_list(demo_dir)

        extracted_demo_state_data = []
        extracted_demo_action_data = []
        
        if self.extract_images is True:
            demo_images_folder_path = os.path.join(self.storage_dir, "demo_images_{}".format(demo_idx))
            self.create_directory(demo_images_folder_path)

        for idx, state in enumerate(state_list):
            state_path = os.path.join(demo_dir, state)
            state_file = open(state_path, 'rb')
            state_data = pickle.load(state_file)

            # Hand data
            hand_coordinates = self.get_finger_tip_coords(state_data['joint_angles'])

            if idx == 0:
                previous_hand_coordinates = hand_coordinates

            # Object data
            cube_pos = list(state_data['cube_position_coordinates'])
            # cube_rot = state_data['cube_orientation']

            extracted_state_features = hand_coordinates + cube_pos

            # Also adding the image to the list if it is requested
            if self.extract_images is True:
                extracted_demo_image_path = os.path.join(demo_images_folder_path, "state_{}.jpg".format(idx))
                cv2.imwrite(extracted_demo_image_path, state_data['image'])

            if idx < len(state_list) - 1:
                extracted_demo_state_data.append(extracted_state_features)

            if idx > 0:
                action_taken = np.array(hand_coordinates) - np.array(previous_hand_coordinates)
                extracted_demo_action_data.append(action_taken)
                previous_hand_coordinates = hand_coordinates


        return torch.tensor(extracted_demo_state_data), torch.tensor(extracted_demo_action_data)        

    def get_all_demo_data(self, dir):

        demos_list = os.listdir(dir)

        for idx, demo in enumerate(demos_list):
            demo_path = os.path.join(dir, demo)

            demo_state_data, demo_action_data = self.get_state_data(demo_path, idx + 1)

            print("Extracted demo state data:{}\n First data: {}\n".format(len(demo_state_data), demo_state_data[0]))
            print("Extracted demo action data:{}\n First data: {}\n".format(len(demo_action_data), demo_action_data[0]))
            
            demo_states_file_path = os.path.join(self.storage_dir, "demo_states_{}.pt".format(idx + 1))
            demo_actions_file_path = os.path.join(self.storage_dir, "demo_actions_{}.pt".format(idx + 1))
            demo_images_folder_path = os.path.join(self.storage_dir, "demo_images_{}".format(idx + 1))

            print("Number of images extracted from the demo: {}".format(len(os.listdir(demo_images_folder_path))))
            
            torch.save(demo_state_data, demo_states_file_path)
            torch.save(demo_action_data, demo_actions_file_path)