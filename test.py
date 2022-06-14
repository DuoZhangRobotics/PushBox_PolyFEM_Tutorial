from pushbox.src.pushbox import PushBox
import numpy as np

env = PushBox()
for i in range(10):
    action = np.array([0.1,0,0])
    env.step(action)