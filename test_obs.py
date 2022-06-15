from pushbox.src.pushbox_with_obstacles import PushBox
import numpy as np

env = PushBox()

# push forward
for i in range(5):
    action = np.array([0.1,0,0])
    env.step(action)

# go around the box
action = np.array([-0.1,0,0])
env.step(action)
action = np.array([0,0,0.3])
env.step(action)
action = np.array([1.0,0,0])
env.step(action)
action = np.array([0,0.3,0])
env.step(action)
action = np.array([0,0,-0.6])
env.step(action)
action = np.array([0,-0.4,0])
env.step(action)
action = np.array([0,0,0.4])
env.step(action)
for i in range(10):
    action = np.array([0,0,0])
    env.step(action)