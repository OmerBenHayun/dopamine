import os
import pickle

base_path = os.path.join('expiriments','rubust_dqn_asterix_expiriment','code')
input_dir = os.path.join(base_path,'evaluation_output_res_data')
num_of_episodes = 100
output_res_name = 'evaluation_compression_num_of_episodes_{}.dict'.format(num_of_episodes)

def load_results() -> dict:
    print('load')
    with open(os.path.join(input_dir,output_res_name) , 'rb') as handle:
        res = pickle.load(handle)
    return res



def get_num_of_episodes() -> int: #dont really need this
    return num_of_episodes
