##
#   This code aims to replicate the simulation results from article
#   "Uncalibrated Image-Based Visual Servoing Control with Maximum Correntropy
#   Kalman Filter" by Ren Xiaolin and Li Hongwen.
#   @author glauberrleite
##
import numpy as np
from matplotlib import pyplot as plt

from ur10_simulation import UR10Simulation
from utils import detect4Circles, saveSampleImage

TS = 0.05
GAIN = 0.5
T_MAX = 50

print("Instantiating robot")
#q = np.array([0.0, 0.0, np.pi/2, 0.0, -np.pi/2, 0.0]) # Desired starting configuration
q = np.array([0.0, -np.pi/8, np.pi/2 + np.pi/8, 0.0, -np.pi/2, 0.0]) # Desired starting configuration
robot = UR10Simulation(q)

# Waiting robot to arrive at starting location
print("Moving robot to starting position")
while (t := robot.sim.getSimulationTime()) < 3:
    robot.step()

#desired_f = np.array([148.0, 150.0, 128.0, 128.0, 108.0, 150.0]) # Desired position for feature
#desired_f = np.array([149., 145., 125., 121., 101., 145., 125., 169.]) # Center
desired_f = np.array([125., 121., 101., 145., 125., 169., 149., 145.]) # Rotation
f = np.zeros(8)

error_log = np.zeros((int(T_MAX/TS), len(f)))
q_log = np.zeros((int(T_MAX/TS), 6))
f_log = np.zeros((int(T_MAX/TS), len(f)))
camera_log = np.zeros((int(T_MAX/TS), 6))
t_log = np.zeros(int(T_MAX/TS))
k = 0

while (t := robot.sim.getSimulationTime()) < T_MAX:
    # Getting camera image and features
    image, resolution = robot.getCameraImage()
    try:
        f = detect4Circles(image, 0)
    except Exception as e:
        print(e) # only print problem in hough circles, but continue with older f
        #saveSampleImage(image, 'sample.jpg')
        break
    
    #robot.fkine(recalculate=True)
    
    # Calculate image jacobian
    J_image = np.zeros((len(f), 6)) # need to find
    #Z = robot.getCameraHeight(recalculate_fkine=True) # Considering fixed Z
    Z = robot.computeZ(4, recalculate_fkine=True)
    focal = 2/(0.5/resolution[0])
    for i in range(0, int(len(f)/2)):
        u = f[2*i]
        v = f[2*i+1]
        J_image[2*i, 0] = -focal/Z[i]
        J_image[2*i+1, 1] = -focal/Z[i]
        J_image[2*i, 2] = u/Z[i]
        J_image[2*i+1, 2] = v/Z[i]
        J_image[2*i, 3] = u*v/focal
        J_image[2*i+1, 3] = (focal**2 + v**2)/focal
        J_image[2*i, 4] = -(focal**2 + u**2)/focal
        J_image[2*i+1, 4] = -u*v/focal
        J_image[2*i, 5] = v
        J_image[2*i+1, 5] = -u
    

    error = f - desired_f
    # IBVS Control Law
    dp = - GAIN * np.linalg.pinv(J_image) @ error.reshape((len(f), 1))
    dx = np.kron(np.eye(2), robot.getCameraRotation().T) @ dp
    #dp = np.array([0.0, 0.0, 0.0, 0.01, 0.0, 0.0]).reshape((6,1))

    # Invkine
    dq = np.linalg.pinv(robot.jacobian()) @ dx
    new_q = robot.getJointsPos() + dq.ravel() * TS

    #logging
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

plt.plot(t_log, error_log[:, 0], color='blue')
plt.plot(t_log, error_log[:, 1], color='red')
plt.plot(t_log, error_log[:, 2], color='green')
plt.plot(t_log, error_log[:, 3], color='yellow')
plt.plot(t_log, error_log[:, 4], color='purple')
plt.plot(t_log, error_log[:, 5], color='black')
plt.plot(t_log, error_log[:, 6], color='gray')
plt.plot(t_log, error_log[:, 7], color='pink')
plt.show()