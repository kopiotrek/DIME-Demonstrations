import os
import pickle
import cv2

import torch
import numpy as np

def check_dir(dir):
    if not os.path.exists(dir):
        print('Making directory: {}'.format(dir))
        os.makedirs(dir)
    else:
        print('{} - Directory already exists'.format(dir))

class FeatureExtractor():
    def __init__(self, task, data_path, storage_path, abs = True, with_images = True):
        self.task = task
        self.data_path = data_path
        self.storage_path = storage_path
        self.extract_images = with_images
        self.absolute = abs

        check_dir(self.storage_path)

    def getting_data_dir_list(self, dir):
        files = os.listdir(dir)
        files.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

        return files

    def get_state_data(self, demo_dir, demo_idx):
        # Getting the list of data        
        state_list = self.getting_data_dir_list(demo_dir)

        extracted_demo_state_data = []
        extracted_demo_action_data = []
        
        if self.extract_images is True:
            demo_images_folder_path = os.path.join(self.storage_path, "demo_images_{}".format(demo_idx))
            check_dir(demo_images_folder_path)

        for idx, state in enumerate(state_list):
            state_path = os.path.join(demo_dir, state)
            state_file = open(state_path, 'rb')
            state_data = pickle.load(state_file)

            # Hand data
            hand_coordinates = np.array(state_data['finger_tip_coords']).reshape(1, 12)

            if idx == 0:
                previous_hand_coordinates = hand_coordinates

            # Object data
            if self.task == "rotate":
                object_pos = np.array(state_data['cube_position_coordinates'])
            elif self.task == "flip":
                object_pos = np.array(state_data['flipping_object_position_coordinates'])
            elif self.task == "spin":
                object_pos = np.array([0, 0, 0])

            extracted_state_features = np.append(hand_coordinates, object_pos)

            # Also adding the image to the list if it is requested
            if self.extract_images is True:
                extracted_demo_image_path = os.path.join(demo_images_folder_path, "state_{}.jpg".format(idx))
                cv2.imwrite(extracted_demo_image_path, state_data['robot_image'])

            if idx < len(state_list) - 1:
                extracted_demo_state_data.append(extracted_state_features)

            if idx > 0:
                if self.absolute is True:
                    action_taken = np.array(hand_coordinates)
                else:
                    action_taken = np.array(hand_coordinates) - np.array(previous_hand_coordinates)
                
                extracted_demo_action_data.append(action_taken)
                previous_hand_coordinates = hand_coordinates


        return torch.tensor(extracted_demo_state_data).squeeze(), torch.tensor(extracted_demo_action_data).squeeze()        

    def get_all_demo_data(self):

        demos_list = os.listdir(self.data_path)

        for idx, demo in enumerate(demos_list):
            demo_path = os.path.join(self.data_path, demo)

            demo_state_data, demo_action_data = self.get_state_data(demo_path, idx + 1)

            print("Extracted demo state data:{}\n First data: {}\n".format(len(demo_state_data), demo_state_data[0]))
            print("Extracted demo action data:{}\n First data: {}\n".format(len(demo_action_data), demo_action_data[0]))
            
            demo_states_file_path = os.path.join(self.storage_path, "demo_states_{}.pt".format(idx + 1))
            demo_actions_file_path = os.path.join(self.storage_path, "demo_actions_{}.pt".format(idx + 1))
            demo_images_folder_path = os.path.join(self.storage_path, "demo_images_{}".format(idx + 1))

            print("Number of images extracted from the demo: {}".format(len(os.listdir(demo_images_folder_path))))
            
            torch.save(demo_state_data, demo_states_file_path)
            torch.save(demo_action_data, demo_actions_file_path)