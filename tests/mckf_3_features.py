##
#   This code aims to replicate the simulation results from article
#   "Uncalibrated Image-Based Visual Servoing Control with Maximum Correntropy
#   Kalman Filter" by Ren Xiaolin and Li Hongwen.
#   @author glauberrleite
##
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import sys
sys.path.append("..")

from ur10_simulation import UR10Simulation
from utils import detectRGBCircles, gaussianKernel

TS = 0.05
GAIN = 0.001
T_MAX = 100
ERROR_THRESHOLD = 0.01

KERNEL_BANDWIDTH = 200
THRESHOLD = 0.01
EPOCH_MAX = 100

#q = np.array([0.0, 0.0, np.pi/2, 0.0, -np.pi/2, 0.0]) # Desired starting configuration
q = np.array([0.0, -np.pi/8, np.pi/2 + np.pi/8, 0.0, -np.pi/2, 0.0]) # Desired starting configuration
robot = UR10Simulation()
robot.start(q)

# Waiting robot to arrive at starting location
while (t := robot.sim.getSimulationTime()) < 3:
    robot.step()

input()

desired_f = np.array([153.5, 144.5, 125.5, 116.5, 99.5, 144.5]) # Desired position for feature

f = np.zeros(6)
f_old = None

# initial parameters for kalman filter
m = 6
n = 6
X = np.random.rand(m*n, 1)
#X = np.zeros((m*n, 1))
Z = np.zeros((m, 1))
H = np.zeros((m, m*n))
P = np.eye(m*n)
K = np.zeros((m*n, m))
Q = np.eye(m*n)
R = np.eye(m)

error_log = np.zeros((int(T_MAX/TS), len(f)))
f_log = np.zeros((int(T_MAX/TS), len(f)))
q_log = np.zeros((int(T_MAX/TS), 6))
camera_log = np.zeros((int(T_MAX/TS), 6))
t_log = np.zeros(int(T_MAX/TS))
desired_f_log = np.zeros((int(T_MAX/TS), len(f)))

X_log = np.zeros((int(T_MAX/TS), m*n))
k = 0

dp = np.zeros((6, 1))
error = np.ones((len(f), 1))
'''
# Giving initial guess
# Getting camera image and features

image, resolution = robot.getCameraImage()
try:
    f = detect4Circles(image)
except Exception as e:
    print(e) # only print problem in hough circles, but continue with older f

# Calculate image jacobian
J_image = np.zeros((m, n)) # need to find
#Z = robot.getCameraHeight(recalculate_fkine=True) # Considering fixed Z
Z_camera = robot.computeZ(3, recalculate_fkine=True)
focal = 2/(0.5/resolution[0])
for i in range(0, int(len(f)/2)):
    u = f[2*i]
    v = f[2*i+1]
    J_image[2*i, 0] = -focal/Z_camera[i]
    J_image[2*i+1, 1] = -focal/Z_camera[i]
    J_image[2*i, 2] = u/Z_camera[i]
    J_image[2*i+1, 2] = v/Z_camera[i]
    J_image[2*i, 3] = u*v/focal
    J_image[2*i+1, 3] = (focal**2 + v**2)/focal
    J_image[2*i, 4] = -(focal**2 + u**2)/focal
    J_image[2*i+1, 4] = -u*v/focal
    J_image[2*i, 5] = v
    J_image[2*i+1, 5] = -u

X = J_image.reshape((m*n, 1))
'''
first_run = True
dp_real = np.zeros(6)
old_pose = robot.computePose(recalculate_fkine=True)

while ((t := robot.sim.getSimulationTime()) < T_MAX) and np.linalg.norm(error) > ERROR_THRESHOLD:
    # Getting camera image and features
    image, resolution = robot.getCameraImage()
    f_old = f.copy()
    try:
        f = detectRGBCircles(image)
        if (f_old is None):
            f_old = f.copy()
    except Exception as e:
        print(e) # only print problem in hough circles, but continue with older f
        break

    #robot.fkine(recalculate=True) # update values for fkine-based functions
    
    # Calculate image jacobian using kalman filter
    # Prediction
    X = X
    P = P + Q

    # Measurement
    Z[0,0] = f[0] - f_old[0]
    Z[1,0] = f[1] - f_old[1]
    Z[2,0] = f[2] - f_old[2]
    Z[3,0] = f[3] - f_old[3]
    Z[4,0] = f[4] - f_old[4]
    Z[5,0] = f[5] - f_old[5]

    new_pose = robot.computePose(recalculate_fkine=True)
    dp_real = new_pose - old_pose
    if first_run:
        first_run = False
    else:
        H = np.kron(np.eye(m), dp.ravel())

    #print(dp_real)
    #print(dp.ravel())
    
    # Finding Bp using Cholesky
    Bp = np.linalg.cholesky(P)
    Br = np.linalg.cholesky(R) # Maybe it can be outside the loop
    B = np.vstack([np.hstack([Bp, np.zeros((Bp.shape[0], Br.shape[1]))]), np.hstack([np.zeros((Br.shape[0], Bp.shape[1])), Br])])
    B_inv = np.linalg.pinv(B)

    # Correction
    difference = np.inf
    epoch = 0
    X_corrected = X.copy()
    while difference > THRESHOLD and epoch < EPOCH_MAX:        
        ## Correntropy kernels
        D = B_inv @ np.vstack([X, Z]) # Our y is Z
        W = B_inv @ np.vstack([np.eye(m*n), H])
        e = D - W @ X_corrected
        
        Cx = np.diag([gaussianKernel(e[i, 0], KERNEL_BANDWIDTH) for i in range(0, m*n)])
        Cy = np.diag([gaussianKernel(e[i, 0], KERNEL_BANDWIDTH) for i in range(m*n, m*n + m)])
        #print(np.linalg.det(Cx))
        #print(np.linalg.det(Cy))

        ## Compute optimal gain
        P_hat = Bp @ np.linalg.inv(Cx) @ Bp.T
        R_hat = Br @ np.linalg.inv(Cy) @ Br.T

        K = P_hat @ H.T @ np.linalg.inv(H @ P_hat @ H.T + R_hat)

        ## Correct X
        X_corrected_old = X_corrected.copy()
        X_corrected = X + K @ (Z - H @ X)

        difference = np.linalg.norm(X_corrected - X_corrected_old)/np.linalg.norm(X_corrected_old)
        epoch += 1
        if epoch == EPOCH_MAX:
            print("Reached max epoch")
    X = X_corrected

    ## Correct P
    P = (np.eye(m*n) - K @ H) @ P @ (np.eye(m*n) - K @ H).T + K @ R @ K.T

    J_image = X.reshape((m, n))

    error = f - desired_f
    # IBVS Control Law
    dp = - GAIN * np.linalg.inv(J_image) @ error.reshape((m, 1))
    dx = np.kron(np.eye(2), robot.getCameraRotation().T) @ dp
    #dp = np.array([0.0, 0.0, 0.0, 0.01, 0.0, 0.0]).reshape((6,1))

    # Invkine
    dq = np.linalg.pinv(robot.jacobian()) @ dx
    new_q = robot.getJointsPos() + dq.ravel() * TS

    #logging
    X_log[k, :] = X.ravel()
    q_log[k] = robot.getJointsPos()
    camera_log[k] = robot.computePose()
    f_log[k] = f
    desired_f_log[k] = desired_f
    error_log[k] = error
    t_log[k] = k*TS
    k += 1
    print('time: ' + str(t) + '; error: ' + str(np.linalg.norm(error)))

    # Send theta command to robot
    robot.setJointsPos(new_q)

    # Save new_pose as old_pose for next iteration
    old_pose = new_pose

    # next_step
    robot.step()

X_log = np.delete(X_log, [i for i in range(k, len(X_log))], axis=0)
t_log = np.delete(t_log, [i for i in range(k, len(t_log))], axis=0)
error_log = np.delete(error_log, [i for i in range(k, len(error_log))], axis=0)
q_log = np.delete(q_log, [i for i in range(k, len(q_log))], axis=0)
f_log = np.delete(f_log, [i for i in range(k, len(f_log))], axis=0)
desired_f_log = np.delete(desired_f_log, [i for i in range(k, len(desired_f_log))], axis=0)
camera_log = np.delete(camera_log, [i for i in range(k, len(camera_log))], axis=0)

_, _, vh = np.linalg.svd(J_image)
w, v = np.linalg.eig(vh)
print(w)
print(v)

plt.subplot(2, 1, 1)
plt.plot(t_log, error_log[:, 0], color='blue')
plt.plot(t_log, error_log[:, 1], color='red')
plt.plot(t_log, error_log[:, 2], color='green')
plt.plot(t_log, error_log[:, 3], color='yellow')
plt.plot(t_log, error_log[:, 4], color='purple')
plt.plot(t_log, error_log[:, 5], color='black')
plt.subplot(2, 1, 2)
for i in range(m*n):
    plt.plot(t_log, X_log[:, i])
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
    'f_3': f_log[:, 2],
    'f_4': f_log[:, 3],
    'f_5': f_log[:, 4],
    'f_6': f_log[:, 5],
    'desired_f_1': desired_f_log[:, 0],
    'desired_f_2': desired_f_log[:, 1],
    'desired_f_3': desired_f_log[:, 2],
    'desired_f_4': desired_f_log[:, 3],
    'desired_f_5': desired_f_log[:, 4],
    'desired_f_6': desired_f_log[:, 5],
})

dataframe.to_csv('results/data/3_feat_mckf.csv')