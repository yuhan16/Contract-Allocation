"""run1.py implements the allocation algorithm for each sceanrio using provided configurations."""
import sys, os
sys.path.insert(1, os.path.realpath('.'))

import numpy as np
import json
from utils import ServiceProvider, ServiceRobot, Utilities


def run1(config_id):
    """Run normal test using the config file."""
    param = json.load(open(f'configs/config{config_id}.json'))
    run_time = 50
    rand_seed = np.random.default_rng(param['seed']).integers(low=0, high=2**16, size=run_time) # no need, for one time ???
    util = Utilities()

    for ii, sd in enumerate(rand_seed):
        print(f'Run simulation {ii+1}...')
        param['seed'] = sd
        sp = ServiceProvider(param)
        rob_list = sp.init_rob(param)
        loc_E_traj = []

        # compute payment and identify user type
        rho_opt = sp.compute_optimal_payment()

        # distributed allocation
        iter = 0
        prob_option = 'rand_samp'   # ["robust", "contract", "rand_max", "rand_samp"]
        belief = sp.choose_belief(prob_option)
        loc_E = sp.compute_expected_L(rob_list, prob=belief)
        loc_E_traj.append(loc_E)
        while iter < 500:
            print(f'iter: {iter}, loc energy: {loc_E:.3f}.')
            # process each type of robots
            for k in range(sp.K):
                rob_idx = sp.get_rob_id_by_type(rob_list, k)                        # find type k robots
                user_nb_dict = sp.partition_user_by_type(rob_list, k, prob=belief)  # find neighboring users for type k robots
                for i in rob_idx:
                    user_nb = user_nb_dict[i]['pos']
                    rob_nb = rob_list[i].find_neighbor_rob(rob_list)
                    rob_list[i].update(user_nb, rob_nb)
            
            loc_E_new = sp.compute_expected_L(rob_list, prob=belief)
            loc_E_traj.append(loc_E_new)
            if np.abs(loc_E - loc_E_new) < 1e-1:
                break

            iter += 1
            loc_E = loc_E_new
            sp.collision_detection(rob_list)
#'''
        # save results
        fname = f'config{config_id}_test{ii+1}_{prob_option}.txt'
        util.save_result(fname, sp, rob_list)

        # save loc_E_traj
        with open('exp_data/'+fname, 'a') as f:
            f.write(f'\nUse {prob_option} method for assignment.\n')
            f.write(f'loc_traj: {loc_E_traj}\n')
#'''


if __name__ == '__main__':
    # run 1
    for i in [1,2,3,4,5,6,7,8]:
        run1(i)