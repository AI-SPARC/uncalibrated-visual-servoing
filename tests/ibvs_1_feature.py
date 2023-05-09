##
#   This code aims to replicate the simulation results from article
#   "Uncalibrated Image-Based Visual Servoing Control with Maximum Correntropy
#   Kalman Filter" by Ren Xiaolin and Li Hongwen.
#   @author glauberrleite
##
import numpy as np

from zmqRemoteApi import RemoteAPIClient
import pandas as pd
from matplotlib import pyplot as plt

import sys
sys.path.append("..")

from ur10_simulation import UR10Simulation
from utils import detectGreenCircle

TS = 0.05
GAIN = 0.75
T_MAX = 15

#q = np.array([0.0, 0.0, np.pi/2, 0.0, -np.pi/2, 0.0]) # Desired starting configuration
q = np.array([0.0, -np.pi/8, np.pi/2 + np.pi/8, 0.0, -np.pi/2, 0.0]) # Desired starting configuration
robot = UR10Simulation()
robot.start(q)

# Waiting robot to arrive at starting location
while (t := robot.sim.getSimulationTime()) < 3:
    robot.step()

input()

desired_f = np.array([256.0/2, 256.0/2]) # Desired position for feature

error_log = np.zeros((int(T_MAX/TS), 2))
f_log = np.zeros((int(T_MAX/TS), 2))
q_log = np.zeros((int(T_MAX/TS), 6))
camera_log = np.zeros((int(T_MAX/TS), 6))
t_log = np.zeros(int(T_MAX/TS))
desired_f_log = np.zeros((int(T_MAX/TS), 2))
k = 0

while (t := robot.sim.getSimulationTime()) < T_MAX:
    # Getting camera image and features
    image, resolution = robot.getCameraImage()
    try:
        f = detectGreenCircle(image)
    except Exception as e:
        print(e) # only print problem in hough circles, but continue with older f
        break

    #robot.fkine(recalculate=True)
    
    # Calculate image jacobian
    u = f[0]
    v = f[1]
    #Z = 0.128 + 0.6127 - 0.0922 # Considering fixed Z
    Z = robot.getCameraHeight(recalculate_fkine=True) # Considering fixed Z
    focal = 1/(0.5/resolution[0])
    J_image = np.zeros((len(f), 6)) # need to find
    J_image[0, 0] = -focal/Z
    J_image[1, 1] = -focal/Z
    J_image[0, 2] = u/Z
    J_image[1, 2] = v/Z
    J_image[0, 3] = u*v/focal
    J_image[1, 3] = (focal**2 + v**2)/focal
    J_image[0, 4] = -(focal**2 + u**2)/focal
    J_image[1, 4] = -u*v/focal
    J_image[0, 5] = v
    J_image[1, 5] = -u

    error = f - desired_f
    # IBVS Control Law
    dp = - GAIN * np.linalg.pinv(J_image) @ error.reshape((len(f), 1))
    dx = np.kron(np.eye(2), robot.getCameraRotation().T) @ dp
    #dp = np.array([0.0, 0.0, 0.0, 0.01, 0.0, 0.0]).reshape((6,1))

    # Invkine
    dq = np.linalg.pinv(robot.jacobian()) @ dx
    new_q = robot.getJointsPos() + dq.ravel() * TS

    #logging
    q_log[k] = robot.getJointsPos()
    camera_log[k] = robot.computePose()
    f_log[k] = f
    desired_f_log[k] = desired_f
    error_log[k] = error
    t_log[k] = k*TS

    k += 1
    print(t)

    # Send theta command to robot
    robot.setJointsPos(new_q)

    # next_step
    robot.step()

t_log = np.delete(t_log, [i for i in range(k, len(t_log))], axis=0)
error_log = np.delete(error_log, [i for i in range(k, len(error_log))], axis=0)
q_log = np.delete(q_log, [i for i in range(k, len(q_log))], axis=0)
f_log = np.delete(f_log, [i for i in range(k, len(f_log))], axis=0)
desired_f_log = np.delete(desired_f_log, [i for i in range(k, len(desired_f_log))], axis=0)
camera_log = np.delete(camera_log, [i for i in range(k, len(camera_log))], axis=0)

plt.plot(t_log, error_log[:, 0], color='blue')
plt.plot(t_log, error_log[:, 1], color='red')
plt.show()

dataframe = pd.DataFrame(data={
    't': t_log,
    'q_1': q_log[:, 0],
    'q_2': q_log[:, 1],
    'q_3': q_log[:, 2],
    'q_4': q_log[:, 3],
    'q_5': q_log[:, 4],
    'q_6': q_log[:, 5],
    'camera_x': camera_log[:, 0],
    'camera_y': camera_log[:, 1],
    'camera_z': camera_log[:, 2],
    'camera_roll': camera_log[:, 3],
    'camera_pitch': camera_log[:, 4],
    'camera_yaw': camera_log[:, 5],
    'f_1': f_log[:, 0],
    'f_2': f_log[:, 1],
    'desired_f_1': desired_f_log[:, 0],
    'desired_f_2': desired_f_log[:, 1],
})

dataframe.to_csv('results/data/1_feat_ibvs.csv')