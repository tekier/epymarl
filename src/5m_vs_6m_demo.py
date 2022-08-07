import os
import time
import pathlib

from griddly import GymWrapperFactory, gd, GymWrapper
import gym

env_name = '5m-vs-6m-lite'
# TODO: This is a monkey-patch for Gym API compatibility
GymWrapper.metadata = {"render_modes": ["human", "rgb_array"]}
root_dir = pathlib.Path(__file__).parent.resolve().parent.resolve()
render_wait_time = 0.5


if __name__ == "__main__":
    image_dir = os.path.join(root_dir, 'smac_lite/resources')
    config_dir = os.path.join(root_dir, 'smac_lite/gdy')

    wrapper = GymWrapperFactory()
    wrapper.build_gym_from_yaml(env_name, os.path.join(root_dir, 'smac_lite/gdy/5m_vs_6m_lite.yml'))
    env = gym.make(f'GDY-{env_name}-v0',
                   gdy_path=config_dir,
                   image_path=image_dir,
                   player_observer_type=gd.ObserverType.ISOMETRIC,
                   global_observer_type=gd.ObserverType.ISOMETRIC)
    env.reset()

    for i in range(10000):
        actions = env.action_space.sample()
        obs, reward, done, info = env.step(actions)
        env.render(observer="global")
        for p in range(env.player_count):
            # Renders the environment from the perspective of a single player
            env.render(observer=p)
        time.sleep(render_wait_time)

        if done:
            break
