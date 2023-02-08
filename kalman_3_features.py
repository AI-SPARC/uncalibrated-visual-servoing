##
#   This code aims to replicate the simulation results from article
#   "Uncalibrated Image-Based Visual Servoing Control with Maximum Correntropy
#   Kalman Filter" by Ren Xiaolin and Li Hongwen.
#   @author glauberrleite
##
import numpy as np
from matplotlib import pyplot as plt

from ur10_simulation import UR10Simulation
from utils import detectRGBCircles

TS = 0.05
GAIN = 0.25
T_MAX = 400
ERROR_THRESHOLD = 2

q = np.array([0.0, 0.0, np.pi/2, 0.0, -np.pi/2, 0.0]) # Desired starting configuration
robot = UR10Simulation(q)

# Waiting robot to arrive at starting location
while (t := robot.sim.getSimulationTime()) < 3:
    robot.step()

#desired_f = np.array([148.0, 150.0, 128.0, 128.0, 108.0, 150.0]) # Desired position for feature
#desired_f = np.array([149.5, 167.5, 126.5, 143.5, 100.5, 167.5]) # Desired position for feature
desired_f = np.array([153.5, 144.5, 125.5, 116.5, 99.5, 144.5]) # Desired position for feature
f = np.zeros(6)
f_old = None

error_log = np.zeros((int(T_MAX/TS), len(f)))
q_log = np.zeros((int(T_MAX/TS), 6))
f_log = np.zeros((int(T_MAX/TS), len(f)))
camera_log = np.zeros((int(T_MAX/TS), 6))
t_log = np.zeros(int(T_MAX/TS))
k = 0

# initial parameters for kalman filter
m = 6
n = 6
X = np.ones((m*n, 1))
Z = np.zeros((m, 1))
H = np.zeros((m, m*n))
P = np.eye(m*n)
K = np.zeros((m*n, m))
Q = np.eye(m*n)
R = np.eye(m)

dp = np.zeros((6, 1))
error = np.ones((len(f), 1))

# Giving initial guess
# Getting camera image and features
'''
image, resolution = robot.getCameraImage()
try:
    f = detectRGBCircles(image)
except Exception as e:
    print(e) # only print problem in hough circles, but continue with older f
    
# Calculate image jacobian
J_image = np.zeros((len(f), 6)) # need to find
#Z = robot.getCameraHeight(recalculate_fkine=True) # Considering fixed Z
Z_camera = robot.computeZ(recalculate_fkine=True)
focal = 1/(0.5/resolution[0])
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
while ((t := robot.sim.getSimulationTime()) < T_MAX) and np.linalg.norm(error) > ERROR_THRESHOLD:
    # Getting camera image and features
    image, resolution = robot.getCameraImage()
    f_old = f
    try:
        f = detectRGBCircles(image)
        if (f_old is None):
            f_old = f
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

    H = np.kron(np.eye(m), dp.ravel())

    # Correction
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    X = X + K @ (Z - H @ X)
    P = (np.eye(m*n) - K @ H) @ P

    J_image = X.reshape((m, n))

    error = f - desired_f
    # IBVS Control Law
    dp = - GAIN * np.linalg.pinv(J_image) @ error.reshape((m, 1))
    dx = np.kron(np.eye(2), robot.getCameraRotation(recalculate_fkine=True).T) @ dp
    #dp = np.array([0.0, 0.0, 0.0, 0.01, 0.0, 0.0]).reshape((6,1))

    # Invkine
    dq = np.linalg.pinv(robot.jacobian()) @ dx
    new_q = robot.getJointsPos() + dq.ravel() * TS

    #logging
    error_log[k] = error
    t_log[k] = k*TS
    k += 1
    print('time: ' + str(t) + '; error: ' + str(np.linalg.norm(error)))

    # Send theta command to robot
    robot.setJointsPos(new_q)

    # next_step
    robot.step()

t_log = np.delete(t_log, [i for i in range(k, len(t_log))], axis=0)
error_log = np.delete(error_log, [i for i in range(k, len(error_log))], axis=0)

plt.plot(t_log, error_log[:, 0], color='blue')
plt.plot(t_log, error_log[:, 1], color='red')
plt.plot(t_log, error_log[:, 2], color='green')
plt.plot(t_log, error_log[:, 3], color='yellow')
plt.plot(t_log, error_log[:, 4], color='purple')
plt.plot(t_log, error_log[:, 5], color='black')
plt.show()