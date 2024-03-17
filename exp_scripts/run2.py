"""run2.py implements the allocation algorithm for each sceanrio using batch initialization. The purpose is for better comparison."""
import sys, os
sys.path.insert(1, os.path.realpath('.'))

import numpy as np
import json
from utils import ServiceProvider, ServiceRobot, Utilities


def run2(config_id, prob_option, N_test=50):
    """run batch simulation using batch initialization"""
    param = json.load(open(f'configs/config{config_id}.json'))
    rand_seed = np.random.default_rng(param['seed']).integers(low=0, high=2**16, size=N_test)
    util = Utilities()

    # get batch initialization for eight scenarios
    # set same random seed for same user/robot positions, for different algorithm performance comparison
    # or use different random seeds
    u_seed = [1, 1, 2, 3, 4, 4, 5, 6]                   # scenario 1-2, 5-6 same
    r_seed = [11, 12, 12, 13, 13, 14, 14, 15]           # scenario 2-3, 4-5, 6-7 same
    u_init = util.user_batch_init(u_seed[config_id-1], param['user_type_num'], N_test)
    r_init = util.rob_batch_init(r_seed[config_id-1], param['robot_type_num'], N_test)

    for ii, sd in enumerate(rand_seed):
        print(f'Run simulation {ii+1}...')
        param['seed'] = sd      # use new random seed for each test case
        sp = ServiceProvider(param)
        rob_list = sp.init_rob(param)
        loc_E_traj = []

        # re-initialize sp and robot using given initialization to keep stable initialization.
        ui = u_init[f't{ii+1}']
        ri = r_init[f't{ii+1}']
        sp.user_pos = ui['user_pos']
        sp.user_prob = ui['user_prob']
        sp.user_theta = ui['user_theta']
        for jj, rob in enumerate(rob_list):
            rob.theta = ri['rob_theta'][jj]
            rob.p0 = ri['rob_pos'][jj]
            rob.p = rob.p0
            rob.p_traj[0] = rob.p0

        # compute payment and identify user type
        rho_opt = sp.compute_optimal_payment()

        # distributed allocation
        belief = sp.choose_belief(prob_option)      
        loc_E = sp.compute_expected_L(rob_list, prob=belief)
        loc_E_traj.append(loc_E)
        iter = 0
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
            util.collision_detection(rob_list)
#'''
        # save results
        fname = f'config{config_id}_test{ii+1}_{prob_option}.txt'
        util.save_result(fname, sp, rob_list)

        # save loc_E_traj
        with open('exp_data/'+fname, 'a') as f:
            f.write(f'\nUse user {prob_option} for allocation.\n')
            f.write(f'loc_traj: {loc_E_traj}\n')
        
        # save wroing matching for rand_max and rand_samp
        if prob_option == 'rand_max' or prob_option == 'rand_samp':
            b_det = sp.choose_belief('contract')
            # mismatch: assigned robot type (belief) is lower than true user type, similar for justmatch and overmatch
            mismatch_num = 0
            overmatch_num = 0
            justmatch_num = 0
            for i in range(sp.Mtotal):
                if belief[i].argmax() < b_det[i].argmax():
                    mismatch_num += 1
                elif belief[i].argmax() == b_det[i].argmax():
                    justmatch_num += 1
                else:
                    overmatch_num += 1
            
            with open('exp_data/'+fname, 'a') as f:    
                f.write(f'\nUse {prob_option} for assignment.\n')
                f.write(f'mismatch: {mismatch_num}\n')
                f.write(f'justmatch: {justmatch_num}\n')
                f.write(f'overmatch: {overmatch_num}\n')
#'''    



if __name__ == '__main__':
    for i in [1,2,3,4,5,6,7,8]:
        for prob_option in ["robust", "contract", "rand_max", "rand_samp"]:
            run2(i, prob_option, N_test=50)