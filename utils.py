"""Script for class implementations."""
import numpy as np


class ServiceProvider:
    def __init__(self, p):
        self.seed = p['seed']
        self.rng = np.random.default_rng(self.seed)
        self.K = p['total_type']
        self.gam = p['gam']
        self.ws_len = p['ws_len']

        # user initialization
        self.Mtotal = p['total_user']
        self.M = p['user_type_num']
        if self.Mtotal != sum(self.M):
            raise Exception('Total user number does not match with type number.')
        self.gain = p['user_gain']
        self.gainf = lambda x: np.log(x+1) / (2*np.log(self.K+1)) + 1    # gain function

        # initialize user necessary parameter (user position, type, type prob)
        if p['user_pos']:
            self.user_pos = np.array(p['user_pos'])
        else:
            self.user_pos = self.rng.random((self.Mtotal, 2)) * self.ws_len         # random position
        if p['user_belief']:
            self.user_prob = np.array(p['user_belief'])
        else:
            self.user_prob = self.rng.random((self.Mtotal, self.K))         # user's type prob
            self.user_prob = self.user_prob / self.user_prob.sum(axis=1)[:, np.newaxis]      # normalize type prob
        if p['user_type'] and len(p['user_type'] == self.Mtotal):
            self.user_theta = np.array(p['user_theta'], dtype=np.int64)
        else:
            self.user_theta = np.zeros(self.Mtotal, dtype=np.int64)
            # create user type to meet user type number M
            tmp = np.arange(self.Mtotal)
            for theta, m in enumerate(self.M):
                pb = self.user_prob[tmp, theta]
                pb = pb / pb.sum()
                idx = self.rng.choice(np.arange(tmp.shape[0]), m, p=pb, replace=False)
                self.user_theta[tmp[idx]] = theta
                tmp = np.delete(tmp, idx)

        # robot initialization, SP knows two parameters
        self.Ntotal = p['total_robot']
        self.N = p['robot_type_num']
        if self.Ntotal != sum(self.N):
            raise Exception('Total user number does not match with type number.')
        

    def init_rob(self, p):
        """Initialize robot parameter. Use list to collect all robots."""
        rob_list = [ServiceRobot() for i in range(self.Ntotal)]

        # get robot type
        if p['robot_type'] and len(p['robot_type'] == self.Ntotal):
            self.rob_theta = np.array(p['rob_type'], dtype=np.int64)
        else:
            tmp_theta = []  # temporary robot type list
            for i, n in enumerate(self.N):
                tmp_theta = tmp_theta + [i] * n
            self.rob_theta = np.array(tmp_theta, dtype=np.int64)
        
        for i, rob in enumerate(rob_list):
            rob.id = i
            rob.theta = self.rob_theta[i]
            if p['robot_pos']:
                self.p0 = np.array(p['robot_pos'][i])
            else:
                rob.p0 = self.rng.random(2) * self.ws_len
            rob.p = rob.p0
            rob.p_traj.append(rob.p0)
            rob.safe_r = p['safe_r']
            rob.step = p['step']
        return rob_list


    def get_rob_id_by_type(self, rob_list, theta):
        ll = []
        for rob in rob_list:
            if rob.theta == theta:
                ll.append(rob.id)
        return ll
    

    def get_rob_pos_by_type(self, rob_list, theta):
        idx = self.get_rob_id_by_type(rob_list, theta)
        rob_theta = [rob_list[i] for i in idx ]
        x_theta = np.zeros((len(rob_theta), 2))
        for i, rob in enumerate(rob_theta):
            x_theta[i] = rob.p
        return np.array(x_theta)
        
    
    def qos(self, x):
        """quality of service, f function in the locational energy."""
        f = x**2    # or define other qos functions
        return f


    def partition_user_by_type(self, rob_list, theta, prob=None):
        """
        This function partitions the user for type theta robots given the user type probability.
        If the probability is deterministic, only partition fixed user.
        - prob: Mtotal x K matrix.
        """
        if prob is None: 
            prob = self.user_prob

        # obtain robot position with type theta
        rob_idx = self.get_rob_id_by_type(rob_list, theta)
        x_theta = self.get_rob_pos_by_type(rob_list, theta)
        if len(x_theta) == 0:
            print(f'No type {theta} robots.')
        
        # obtain user position with type theta based on prob, select all users with positive prob with type theta
        user_idx = prob[:, theta].nonzero()[0]
        q_theta = self.user_pos[user_idx]
        if len(q_theta) == 0:
            print(f'No type {theta} users.')

        # construct dist matrix and f matrix, row is user, col is robot with type theta
        dmat = np.zeros((q_theta.shape[0], x_theta.shape[0]))
        fmat = np.zeros((q_theta.shape[0], x_theta.shape[0]))
        for i in range(q_theta.shape[0]):
            dmat[i, :] = np.linalg.norm(x_theta-q_theta[i], axis=1)
            fmat[i, :] = self.qos(dmat[i, :])
        
        # check fmat to determine allocation for every user, use dict to store neighbor info
        nb_info = {}
        for i in rob_idx:
            nb_info[i] = {'prob': [], 'pos': []}
        tmp = fmat.argmin(axis=1)
        for i, ii in enumerate(tmp):
            user_id = user_idx[i]
            rob_id = rob_idx[ii]
            nb_info[rob_id]['prob'].append(prob[user_id, theta])
            nb_info[rob_id]['pos'].append(self.user_pos[user_id])
        
        return nb_info
        

    def compute_expected_L(self, rob_list, prob=None):
        """compute SP's expected locational energy."""
        if prob is None:
            prob = self.user_prob
        f = 0
        # get robot types
        for theta in range(self.K):
            rob_idx = self.get_rob_id_by_type(rob_list, theta)
            user_nb = self.partition_user_by_type(rob_list, theta, prob=prob)
            for i in rob_idx:
                rob = rob_list[i]
                info = user_nb[i]
                for pp, q in zip(info['prob'], info['pos']):
                    f += pp * self.qos(np.linalg.norm(q-rob.p))
        return f
    

    def compute_optimal_payment(self):
        """Compute optimal payment."""
        if self.gainf(1) >= self.K / (self.K-1):
            print('Gain assumption not satisfied.')
            return
        rho_opt = [(self.K-k+1)*self.gain-(self.K-k)*self.gainf(1)*self.gain for k in range(1,self.K+1)]
        return rho_opt


    def test_optimal_payment(self):
        """Solve LP for optimal payment test. b: M x K allocation plan."""
        from scipy.optimize import LinearConstraint, minimize
        constr = []
        constr.append( LinearConstraint(np.eye(self.K), lb=np.zeros(self.K), ub=np.ones(self.K)*self.gain) )

        for k in range(1, self.K):
            for l in range(k+1, self.K+1):
                a = np.zeros(self.K)
                a[l-1] = 1
                a[k-1] = -1
                constr.append( LinearConstraint(a, lb=(self.gainf(l-k) * self.gain - self.gain), ub=np.inf) )

        pp = self.user_prob[13]         # select one user to verify
        myobj = lambda x: - pp @ x      # -min is max
        x0 = np.random.rand(self.K) * self.ws_len
        res = minimize(myobj, x0, constraints=constr)
        #print(res)
        return res.x
    

    def choose_belief(self, option):
        """
        Generate different beliefs (methods) for allocation.
        option belongs to: ['robust', 'contract', 'rand_max', 'rand_samp']
        """
        if option == 'robust':                       # initial user type prob
            b = self.user_prob
        elif option == 'contract':                  # deterministic user type prob after payment
            b = np.zeros_like(self.user_prob)           
            for i in range(self.Mtotal):
                b[i, self.user_theta[i]] = 1
        elif option == 'rand_max':                  # choose max type prob as deterministic
            b = np.zeros_like(self.user_prob)       
            for i in range(self.Mtotal):
                b[i, self.user_prob[i,:].argmax()] = 1
        elif option == 'rand_samp':                 # choose one according to type prob
            b = np.zeros_like(self.user_prob)
            for i in range(self.Mtotal):
                j = np.random.choice(np.arange(self.K), p=self.user_prob[i,:])
                b[i, j] = 1
        else:
            raise Exception('Wrong argument, use of the following: ["robust", "contract", "rand_max", "rand_samp"].')
        return b
    

    def collision_detection(self, rob_list):
        p = np.zeros((len(rob_list), 2))
        for i, rob in enumerate(rob_list):
            p[i] = rob.p
        
        # construct distance mat
        dmat = np.zeros((p.shape[0], p.shape[0]))
        for i in range(p.shape[0]):
            dmat[i, :] = np.linalg.norm(p-p[i], axis=1)
        iu = np.triu_indices(p.shape[0], k=1)
        iu = np.ravel_multi_index(iu, (p.shape[0], p.shape[0]))
        dist = dmat.ravel()[iu]
        if np.any(dist < 0.01):
            raise Exception('Collision Detected.')


class ServiceRobot:
    def __init__(self) -> None:
        self.id = -1
        self.theta = None
        self.p0 = None      # initial position
        self.p = None       # current position
        self.safe_r = 1     # safety region
        self.step = 0.1     # time step for each iteration

        self.p_traj = []    # store trajectory
        self.u_traj = []


    def find_neighbor_rob(self, rob_list):
        """Find all robots within the safety radius."""
        nb_pos = []
        for rob in rob_list:
            if rob.id == self.id:
                continue
            if np.linalg.norm(rob.p - self.p) < self.safe_r:
                nb_pos.append(rob.p)
        return nb_pos


    def dloc(self, user_nb):
        """Compute gradient of locational energy.
            f_energy = sum_i f(|x-q_i|), f(x) = x^2
        """
        df = np.zeros(2)
        for q in user_nb:
            df += 2 * (self.p - q)
        return df
        

    def dsafe(self, rob_nb):
        """Compute gradient of safety barrier.
            f_safe = sum_j -beta*log(|x-x_j|)
        """
        beta = -10
        df = np.zeros(2)
        for x in rob_nb:
            df += beta* (self.p-x) / (np.linalg.norm(self.p-x)**2)
        return df
    

    def update(self, user_nb, rob_nb):
        """Update next step."""
        df_loc = self.dloc(user_nb)
        df_safe = self.dsafe(rob_nb)

        # normalize gradient and update u
        u = -df_loc - df_safe
        if np.linalg.norm(u) > 1:
            u = u / np.linalg.norm(u)
        #u = -df_loc / np.linalg.norm(df_loc) - df_safe / np.linalg.norm(df_safe)
        self.p = self.p + u*self.step

        self.p_traj.append(self.p)
        self.u_traj.append(u)
        # or return u and store trajectory outside


class Utilities:
    def __init__(self) -> None:
        pass


    def user_batch_init(self, seed, M, N_test, ws_len=10):
        """Batch initialization for user. Initialize N_test cases using given seed."""
        rng = np.random.default_rng(seed)
        Mtotal = sum(M)
        K = len(M)
        aa = {}
        for i in range(N_test):
            aa[f't{i+1}'] = {}
            user_pos = rng.random((Mtotal, 2)) * ws_len
            user_prob = rng.random((Mtotal, K))
            user_prob = user_prob / user_prob.sum(axis=1)[:, np.newaxis]
            user_theta = np.zeros(Mtotal, dtype=np.int64)

            # sample user type to meet user type number M
            tmp = np.arange(Mtotal)    
            for theta, m in enumerate(M):
                p = user_prob[tmp, theta]
                p = p / p.sum()
                idx = rng.choice(np.arange(tmp.shape[0]), size=m, p=p, replace=False)
                #idx = rng.choice(np.arange(tmp.shape[0]), size=m, replace=False)   # origin, no p
                user_theta[tmp[idx]] = theta
                tmp = np.delete(tmp, idx)
            
            aa[f't{i+1}']['user_pos'] = user_pos
            aa[f't{i+1}']['user_prob'] = user_prob
            aa[f't{i+1}']['user_theta'] = user_theta
        return aa


    def rob_batch_init(self, seed, N, N_test, ws_len=10):
        """Batch initialization for robot. Initialize N_test cases using given seed."""
        rng = np.random.default_rng(seed)
        Ntotal = sum(N)
        K = len(N)
        aa = {}
        for i in range(N_test):
            aa[f't{i+1}'] = {}
            rob_pos = rng.random((Ntotal, 2)) * ws_len
            rob_theta = []
            for j, n in enumerate(N):
                rob_theta = rob_theta + [j] * n
            
            aa[f't{i+1}']['rob_pos'] = rob_pos
            aa[f't{i+1}']['rob_theta'] = np.array(rob_theta)
        return aa


    def save_result(self, fname, sp, rob_list):
        """Save results to txt files."""
        import os
        if not os.path.exists('exp_data/'):
            os.mkdir('exp_data')
        
        with open('exp_data/'+fname, 'w') as f:
            f.write(f'Contract with {sp.Mtotal} users, {sp.Ntotal} robots, {sp.K} types.\n')
            f.write(f'seed: {sp.seed}\n')
            f.write(f'M: {sp.M}\n')
            f.write(f'N: {sp.N}\n')
            f.write(f'user_pos: {sp.user_pos.tolist()}\n')
            f.write(f'user_prob: {sp.user_prob.tolist()}\n')
            f.write(f'user_theta: {sp.user_theta.tolist()}\n')

            f.write(f'optimal_rho: {sp.compute_optimal_payment()}\n')

            #f.write(f'rob_theta: {sp.rob_theta.tolist()}\n')
            for rob in rob_list:
                ptraj = np.vstack(rob.p_traj)   # first element is p0
                utraj = np.vstack(rob.u_traj)
                f.write(f'rob_{rob.id}: theta: {rob.theta}\n')
                f.write(f'rob_{rob.id}: p_traj: {ptraj.tolist()}\n')
                f.write(f'rob_{rob.id}: u_traj: {utraj.tolist()}\n')


    def collision_detection(self, rob_list):
        p = np.zeros((len(rob_list), 2))
        for i, rob in enumerate(rob_list):
            p[i] = rob.p
        
        # construct distance mat
        dmat = np.zeros((p.shape[0], p.shape[0]))
        for i in range(p.shape[0]):
            dmat[i, :] = np.linalg.norm(p-p[i], axis=1)
        iu = np.triu_indices(p.shape[0], k=1)
        iu = np.ravel_multi_index(iu, (p.shape[0], p.shape[0]))
        dist = dmat.ravel()[iu]
        if np.any(dist < 0.01):
            raise Exception('Collision Detected.')
    
