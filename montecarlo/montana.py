import numpy as np
import gymnasium as gym
from tqdm import tqdm
from time import sleep
import random
from gymnasium.wrappers import RecordVideo
import pygame


x_bin = 20
vel_bin = 20
gamma = 0.99
complete_reward = 500
x_space = np.linspace(-1.2, 0.6, x_bin)
vel_space = np.linspace(-0.07, 0.07, vel_bin)

A = [0, 1, 2] 

S = [] # MDP states

prev_x, prev_v = x_space[0], vel_space[0]

for x in x_space[1:]:

    for v in vel_space[1:]:

        S.append({

            'x_range': (prev_x, x),

            'vel_range': (prev_v, v),

            'a': {a:{'reward':0, 'quon': 0 } for a in A}

        })

        prev_v = v

    prev_x = x


gamma = gamma

complete_reward = complete_reward

episodes = int(100000 / 500 * 5)
epsilon = 0.001


def get_state(x, v):

    a = max(0, np.searchsorted(x_space, min(x, x_space[-1]), side='left') - 1)

    b = np.searchsorted(vel_space, min(v, vel_space[-1]), side='left') - 1

    idx = a * (vel_space.shape[0] - 1) + b

    if idx >= len(S) or idx < 0:

        raise IndexError(f'Index {idx} out of range for x, v of {(x, v)} and a, b of {(a, b)}')

    return S[idx], idx

 


env= gym.make('MountainCar-v0', render_mode='human')

for i in range(1):
    state, _ = env.reset()
    history = []
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break

        keys = pygame.key.get_pressed()
        action = 1 
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            action = 0  
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            action = 2 

        _, s_idx = get_state(state[0], state[1])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        history.append((s_idx, action, reward))
        state = next_state
        env.render()
        sleep(0.02)


    G = 0
    visited = set()
    for s, a, r in reversed(history):
        G = r + gamma* G
        if (s, a) not in visited:
            S[s]['a'][a]['quon'] += 1
            S[s]['a'][a]['reward'] += (G - S[s]['a'][a]['reward']) / S[s]['a'][a]['quon']
            visited.add((s, a))

env.close()



env = gym.make('MountainCar-v0')

for i in tqdm(range(episodes), desc='Rolling dice...'):
    
    state, _ = env.reset()
    ac, s = get_state(state[0], state[1])
    history = []
    done = False
    iter = 0

    while not done and iter < 1000:
        if random.random() < epsilon:
            a = random.choice(A)
        else:
            a = np.argmax([ac['a'][action]['reward'] for action in A])
        
        next_state, reward, done, _, _ = env.step(a)
        _, next_s = get_state(next_state[0], next_state[1])

        history.append((s, a, reward))
        s = next_s
        iter += 1
    
    if done:
        print("Перемога!!!")
    
    G = 0
    visited = set()

    for s, a, r in reversed(history):
        G = r + gamma * G
        if (s, a) in visited:
            continue
        S[s]['a'][a]['quon'] = S[s]['a'][a]['quon'] + 1
        S[s]['a'][a]['reward'] += (G - S[s]['a'][a]['reward']) / S[s]['a'][a]['quon']
        visited.add((s, a))

env.close()


# real test 


def solve(max_steps=300, episodes=1):

    env = gym.make('MountainCar-v0', render_mode='human')

    for _ in range(episodes):

        (x, v), _ = env.reset()

        isDone = False

        idx = 0

        rewards = 0

        while not isDone and idx < max_steps:

            # observe state

            _, s_id = get_state(x, v)

            # get optimal action

            a = np.argmax([S[s_id]['a'][action]['reward'] for action in A])

            # take optimal action

            (x, v), r, isDone, _, _ = env.step(a)

            # give reward if done

            if isDone:

                r = complete_reward

            rewards += r

            idx += 1

            sleep(0.01)

        print(f'total steps: {idx},  total rewards: {rewards:.3f}')

    env.close()

solve()

