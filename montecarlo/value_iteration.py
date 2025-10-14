import numpy as np
import gymnasium as gym
from tqdm import tqdm
from time import sleep
import random
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

        self.monte_carlo()

    

    def monte_carlo(self, episodes=1000, epsilon=0.01):

        '''

        Build an MDP model for the Mountain Car environment using Monte-Carlo simultions.

        '''

        env = gym.make('MountainCar-v0')
        
        qa_table = np.zeros((len(self.S), len(self.A)))
        N = np.zeros((len(self.S), len(self.A)))

        for i in tqdm(range(episodes), desc='Rolling dice...'):
            
            state, _ = env.reset()
            _, s = self.get_state(state[0], state[1])
            history = []
            done = False
            iter = 0

            while not done and iter < 1000:
                if random.random() < epsilon:
                    a = random.choice(self.A)
                else:
                    a = np.argmax(qa_table[s])
                
                next_state, reward, done, _, _ = env.step(a)
                _, next_s = self.get_state(next_state[0], next_state[1])

                history.append((s, a, reward))
                s = next_s
                iter += 1
            
            G = 0
            visited = set()

            for s, a, r in reversed(history):
                G = r + self.gamma * G
                if (s, a) in visited:
                    continue
                N[s, a] += 1
                qa_table[s, a] += (G - qa_table[s, a]) / N[s, a]
                visited.add((s, a))

        self.qa_table = qa_table

        env.close()
        return qa_table

                    

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

    mdp = MountainCarDiscreteMDP(x_bin=20, vel_bin=20, trials_per_state_action=100, gamma=0.99, complete_reward=100)

    mdp.solve(max_steps=200, record=False, episodes=3)