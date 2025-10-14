import numpy as np
import gymnasium as gym
from tqdm import tqdm
from time import sleep
from gymnasium.wrappers import RecordVideo


class MountainCarDiscreteMDP(object):

    def __init__(self, x_bin, vel_bin, trials_per_state_action=100, gamma=0.99, complete_reward=100):

        # THE BIN
        self.x_space = np.linspace(-1.2, 0.6, x_bin)
        self.vel_space = np.linspace(-0.07, 0.07, vel_bin)

        # MDP states

        self.A = [0, 1, 2] # MDP actions: 0 (push left), 1 (no push), 2 (push right)

        self.S = [] # MDP states

        prev_x, prev_v = self.x_space[0], self.vel_space[0]

        for x in self.x_space[1:]:

            for v in self.vel_space[1:]:

                self.S.append({

                    'x_range': (prev_x, x),

                    'vel_range': (prev_v, v),

                    'p': {a:{} for a in self.A},

                    'v': 0

                })

                prev_v = v

            prev_x = x


        self.trials_per_state_action = trials_per_state_action

        self.gamma = gamma

        self.complete_reward = complete_reward

        self._build_mdp(trials_per_state_action=trials_per_state_action)

    

    def _build_mdp(self, trials_per_state_action=100):

        '''

        Build an MDP model for the Mountain Car environment using Monte-Carlo simultions.

        '''

        env = gym.make('MountainCar-v0')

        # for each state action, do N trials to compute transition probability

        for s in tqdm(self.S, desc='Building MDP model...'):

            for a in self.A:

                for _ in range(trials_per_state_action):

                    env.reset()

                    x_init = np.random.uniform(low=s['x_range'][0], high=s['x_range'][1])

                    vel_init = np.random.uniform(low=s['vel_range'][0], high=s['vel_range'][1])

                    env.unwrapped.state = (x_init, vel_init)

                    (dest_x, dest_v), r, isDone, _, _ = env.step(a)

                    if isDone:

                        r = self.complete_reward

                    _, dest_idx = self.get_state(dest_x, dest_v)

                    if s['p'][a].get(dest_idx) is None:

                        s['p'][a][dest_idx] = (1, r)

                    else:

                        s['p'][a][dest_idx] = (s['p'][a][dest_idx][0] + 1, s['p'][a][dest_idx][1] + r)

        env.close()

                    

    def get_state(self, x, v):

        '''

        Given the current horizontal position x and velocity of the car v, find the corresponding MDP state.

        '''

        a = max(0, np.searchsorted(self.x_space, min(x, self.x_space[-1]), side='left') - 1)

        b = np.searchsorted(self.vel_space, min(v, self.vel_space[-1]), side='left') - 1

        idx = a * (self.vel_space.shape[0] - 1) + b

        if idx >= len(self.S) or idx < 0:

            raise IndexError(f'Index {idx} out of range for x, v of {(x, v)} and a, b of {(a, b)}')

        return self.S[idx], idx

    

    def policy_iteration(self, max_iter=10, max_iter_2=10):

        '''

        Find optimal policy pi* with policy iteration.

        '''
        # initialize a random policy

        policy = np.random.choice(self.A, size=len(self.S))

        # initialize Q(s,a) table
        self.qa_table = np.zeros((len(self.S), len(self.A)))

        idx = 0

        while idx < max_iter:

            idx_1 = 0

            print(f'Running iter {idx}...', end='\n')
                
            while idx_1 < max_iter_2:

                for s_id, s in enumerate(self.S):

                    sums = 0

                    for dest_idx, (freq, rewards) in s['p'][policy[s_id]].items():

                        p = freq / self.trials_per_state_action

                        r = rewards / self.trials_per_state_action

                        v = self.gamma * self.S[dest_idx]['v']

                        sums += p * (r + v)


                    self.qa_table[s_id][policy[s_id]] = sums 

                    v_prime = np.max(self.qa_table[s_id ])
                
                    s['v'] = v_prime

                idx_1 += 1
            
            idx_2 = 0

            while idx_2 < max_iter_2:

                for s_id, s in enumerate(self.S):

                    a_res = []

                    for a in self.A:

                        sums = 0

                        for dest_idx, (freq, reward) in s['p'][a].items():

                            p = freq / self.trials_per_state_action

                            r = reward / self.trials_per_state_action

                            v = self.gamma * self.S[dest_idx]['v']

                            sums += p * (r + v)

                        a_res.append(sums)

                    
                    best_a = np.argmax(a_res)

                    policy[s_id] = best_a

            
                idx_2 += 1

            idx += 1
        self.p = policy
        print("\nDone")
            
    

    def get_optimal_action(self, s_id):

        '''

        pi(s), return optimal action to take given state s.

        '''

        if not hasattr(self, 'qa_table'):

            raise AttributeError('Missing QA table, did you run "policy_iteration()"?')

        # get new v value and update

        #a = np.argmax(self.qa_table[s_id])

        a = self.p[s_id]

        return a

    

    def solve(self, max_steps=1000, record=False, episodes=1):

        '''

        Let the AI agent play the Mountain Car game using the computed optimal policy. Set `record` to True to record and save as video (will not render a window). Set `episodes` to specify number of sessions for the AI agent to play.

        '''

        if not hasattr(self, 'qa_table'):

            raise AttributeError('Missing QA table, did you run "policy_iteration()"?')

        env = gym.make('MountainCar-v0', render_mode='rgb_array' if record else 'human')

        if record:

            # wrap env with video recorder

            env = RecordVideo(env, './vid', episode_trigger=lambda x: True, name_prefix='mc-discrete')

        for _ in range(episodes):

            (x, v), _ = env.reset()

            isDone = False

            idx = 0

            rewards = 0

            while not isDone and idx < max_steps:

                # observe state

                _, s_id = self.get_state(x, v)

                # get optimal action

                a = self.get_optimal_action(s_id)

                # take optimal action

                (x, v), r, isDone, _, _ = env.step(a)

                # give reward if done

                if isDone:

                    r = self.complete_reward

                rewards += r

                idx += 1

                sleep(0.01)

            print(f'total steps: {idx},  total rewards: {rewards:.3f}')

        env.close()



if __name__ == '__main__':

    mdp = MountainCarDiscreteMDP(x_bin=20, vel_bin=20, trials_per_state_action=100, gamma=0.99, complete_reward=100)

    mdp.policy_iteration(max_iter=25, max_iter_2=50)

    mdp.solve(max_steps=200, record=False, episodes=3)