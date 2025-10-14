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

    

    def value_iteration(self, theta=1e-3, max_iter=5000):

        '''

        Find optimal value function V* with value iteration.

        '''

        # initialize Q(s,a) table

        self.qa_table = np.zeros((len(self.S), len(self.A)))

        delta = -1

        idx = 0

        while ((delta == -1 or delta > theta) and idx < max_iter):

            print(f'Running iter {idx}...', end='')

            delta = 0

            for s_id, s in enumerate(self.S):

                for a in self.A:

                    # get transition probability given action

                    sums = 0

                    for dest_idx, (freq, rewards) in s['p'][a].items():

                        p = freq / self.trials_per_state_action

                        r = rewards / self.trials_per_state_action

                        v = self.gamma * self.S[dest_idx]['v']

                        sums += p * (r + v)

                    # update Q-value                    

                    self.qa_table[s_id, a] = sums

                # get new v value and update

                v_prime = np.max(self.qa_table[s_id])

                delta += np.abs(v_prime - s['v'])

                s['v'] = v_prime

            idx += 1

            print(f', delta={delta:.5f}')

        print('\nDone')

    

    def get_optimal_action(self, s_id):

        '''

        pi(s), return optimal action to take given state s.

        '''

        if not hasattr(self, 'qa_table'):

            raise AttributeError('Missing QA table, did you run "value_iteration()"?')

        # get new v value and update

        a = np.argmax(self.qa_table[s_id])

        return a

    

    def solve(self, max_steps=1000, record=False, episodes=1):

        '''

        Let the AI agent play the Mountain Car game using the computed optimal policy. Set `record` to True to record and save as video (will not render a window). Set `episodes` to specify number of sessions for the AI agent to play.

        '''

        if not hasattr(self, 'qa_table'):

            raise AttributeError('Missing QA table, did you run "value_iteration()"?')

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

    mdp = MountainCarDiscreteMDP(x_bin=20, vel_bin=20, trials_per_state_action=100, gamma=0.99, complete_reward=200)

    mdp.value_iteration(theta=1e-3, max_iter=5000)

    mdp.solve(max_steps=200, record=False, episodes=3)