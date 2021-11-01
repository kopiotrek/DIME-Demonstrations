import os
import pickle
import numpy as np

from allegro_ik import AllegroInvKDL

MINIMUM_DISTANCE_THRESHOLD = 0.01

class StateExtractor():
    def __init__(self, data_path, new_data_path):
        self.data_path = data_path
        self.new_data_path = new_data_path

        self.demos_list = os.listdir(self.data_path)
        self.demos_list.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

        self.allegro_ik = AllegroInvKDL(cfg = None, urdf_path = "/home/sridhar/dexterous_arm/ik_stuff/urdf_template/allegro_right.urdf")

    def check_dir(self, dir):
        if not os.path.exists(dir):
            print('Making directory: {}'.format(dir))
            os.makedirs(dir)
        else:
            print('Directory already exists')

    def store_pickle_data(self, pickle_path, data):
        file = open(pickle_path, 'ab')

        pickle.dump(data, file)
        file.close()

    def calculate_tip_coords(self, joint_angles):
        thumb_coord = self.allegro_ik.finger_forward_kinematics('thumb', joint_angles[12:16])[0]
        ring_coord = self.allegro_ik.finger_forward_kinematics('ring', joint_angles[8:12])[0]

        return thumb_coord, ring_coord

    def calculate_delta_finger_tip(self, initial_pos, final_pos):
        return np.linalg.norm(np.array(initial_pos) - np.array(final_pos))

    def extract_states(self, demo_path):
        states_list = os.listdir(demo_path)

        states_list.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

        demo_state_data = []

        for state_name in states_list:
            state_path = os.path.join(demo_path, state_name)

            state_file = open(state_path, "rb")
            state_data = pickle.load(state_file)

            demo_state_data.append(state_data)

        return demo_state_data

    def resample_states_from_demo(self, demo_path):
        demo_state_data = self.extract_states(demo_path)

        new_demo_states = []

        for idx, state_data in enumerate(demo_state_data):
            if idx == 0:
                new_demo_states.append(state_data)
                # Setting an initial value for resampling the data with the first state.
                previous_extracted_state = state_data
                prev_state_thumb_coord, prev_state_ring_coord = self.calculate_tip_coords(previous_extracted_state['joint_angles'])
                continue

            state_thumb_coord, state_ring_coord = self.calculate_tip_coords(state_data['joint_angles'])

            delta_dis_thumb = self.calculate_delta_finger_tip(prev_state_thumb_coord, state_thumb_coord)
            delta_dis_ring = self.calculate_delta_finger_tip(prev_state_ring_coord, state_ring_coord)

            total_dis = delta_dis_thumb + delta_dis_ring

            if total_dis > MINIMUM_DISTANCE_THRESHOLD:
                # Adding new state to list
                new_demo_states.append(state_data)
                previous_extracted_state = state_data
                prev_state_thumb_coord, prev_state_ring_coord = self.calculate_tip_coords(previous_extracted_state['joint_angles'])

        return new_demo_states

    def store_new_demo_states(self, new_demo_path, new_demo_states):
        self.check_dir(new_demo_path)

        for idx, state_data in enumerate(new_demo_states):
            new_state_path = os.path.join(new_demo_path, "state_{}".format(idx + 1))
            self.store_pickle_data(new_state_path, state_data)

    def resample_demos(self):
        for idx, demo in enumerate(self.demos_list):
            demo_path = os.path.join(self.data_path, demo)

            new_demo_states = self.resample_states_from_demo(demo_path)
            new_demo_path = os.path.join(self.new_data_path, "demo_{}".format(idx + 1))
            self.store_new_demo_states(new_demo_path, new_demo_states)
