import gymnasium as gym

from gymnasium.utils.play import play



play(gym.make('MountainCar-v0', render_mode='rgb_array'), 

    keys_to_action={

        'a': 0,

        'd': 2,

    }, 

    noop=1)