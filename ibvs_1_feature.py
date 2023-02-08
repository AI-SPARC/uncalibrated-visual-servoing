##
#   This code aims to replicate the simulation results from article
#   "Uncalibrated Image-Based Visual Servoing Control with Maximum Correntropy
#   Kalman Filter" by Ren Xiaolin and Li Hongwen.
#   @author glauberrleite
##
import numpy as np

import cv2
from zmqRemoteApi import RemoteAPIClient

from matplotlib import pyplot as plt

TS = 0.05
GAIN = 0.75
T_MAX = 25

class UR10Simulation():
    def __init__(self, q):
        # New instance of API client
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')
        
        self.client.setStepping(True)
        self.sim.startSimulation()

        # Getting joint handles and setting home position
        self.joints = [self.sim.getObject('./joint', {'index': i}) for i in range(6)]
        self.cameraHandle = self.sim.getObject('./sensor')
        self.q = q
        self.dq = np.zeros(6)
        self.setJointsPos(q)
        self.T_0_6, self.T_0_5, self.T_0_4, self.T_0_3, self.T_0_2, self.T_0_1 = self.fkine(recalculate=True)
    
    def setJointsPos(self, q):
        self.sim.setJointTargetPosition(self.joints[0], q[0])
        self.sim.setJointTargetPosition(self.joints[1], q[1])
        self.sim.setJointTargetPosition(self.joints[2], q[2])
        self.sim.setJointTargetPosition(self.joints[3], q[3])
        self.sim.setJointTargetPosition(self.joints[4], q[4])
        self.sim.setJointTargetPosition(self.joints[5], q[5])
    
    def getJointsPos(self):
        self.q[0] = self.sim.getJointPosition(self.joints[0])
        self.q[1] = self.sim.getJointPosition(self.joints[1])
        self.q[2] = self.sim.getJointPosition(self.joints[2])
        self.q[3] = self.sim.getJointPosition(self.joints[3])
        self.q[4] = self.sim.getJointPosition(self.joints[4])
        self.q[5] = self.sim.getJointPosition(self.joints[5])

        return self.q

    def getJointsVel(self):
        self.dq[0] = self.sim.getJointVelocity(self.joints[0])
        self.dq[1] = self.sim.getJointVelocity(self.joints[1])
        self.dq[2] = self.sim.getJointVelocity(self.joints[2])
        self.dq[3] = self.sim.getJointVelocity(self.joints[3])
        self.dq[4] = self.sim.getJointVelocity(self.joints[4])
        self.dq[5] = self.sim.getJointVelocity(self.joints[5])

        return self.dq

    def fkine(self, recalculate=False):
        if recalculate:
            self.getJointsPos()
            self.T_0_1 = self.dh(self.q[0], 0.128, 0, -np.pi/2)
            self.T_0_2 = self.T_0_1 @ self.dh(self.q[1] - np.pi/2, 0, 0.6127, 0)
            self.T_0_3 = self.T_0_2 @ self.dh(self.q[2], 0, 0.5716, 0)
            self.T_0_4 = self.T_0_3 @ self.dh(self.q[3] - np.pi/2, 0.1639, 0, -np.pi/2)
            self.T_0_5 = self.T_0_4 @ self.dh(self.q[4], 0.1157, 0, np.pi/2)
            self.T_0_6 = self.T_0_5 @ self.dh(self.q[5] + np.pi, 0.0922, 0, 0)

        return self.T_0_6, self.T_0_5, self.T_0_4, self.T_0_3, self.T_0_2, self.T_0_1

    def jacobian(self, recalculate_fkine=False):

        T_0_6, T_0_5, T_0_4, T_0_3, T_0_2, T_0_1 = self.fkine(recalculate=recalculate_fkine)
        
        # Jacobian computation as shown in section 3.1.3 on Siciliano's Robotics book
        z_0 = np.array([0, 0, 1])
        z_1 = T_0_1[0:3, 2]
        z_2 = T_0_2[0:3, 2]
        z_3 = T_0_3[0:3, 2]
        z_4 = T_0_4[0:3, 2]
        z_5 = T_0_5[0:3, 2]

        p_0 = np.array([0, 0, 0])
        p_1 = T_0_1[0:3, -1]
        p_2 = T_0_2[0:3, -1]
        p_3 = T_0_3[0:3, -1]
        p_4 = T_0_4[0:3, -1]
        p_5 = T_0_5[0:3, -1]
        p_e = T_0_6[0:3, -1]

        J1 = np.vstack([np.cross(z_0, (p_e-p_0)).reshape(3,1), z_0.reshape(3, 1)])
        J2 = np.vstack([np.cross(z_1, (p_e-p_1)).reshape(3,1), z_1.reshape(3, 1)])
        J3 = np.vstack([np.cross(z_2, (p_e-p_2)).reshape(3,1), z_2.reshape(3, 1)])
        J4 = np.vstack([np.cross(z_3, (p_e-p_3)).reshape(3,1), z_3.reshape(3, 1)])
        J5 = np.vstack([np.cross(z_4, (p_e-p_4)).reshape(3,1), z_4.reshape(3, 1)])
        J6 = np.vstack([np.cross(z_5, (p_e-p_5)).reshape(3,1), z_5.reshape(3, 1)])

        return np.hstack([J1, J2, J3, J4, J5, J6])

    def getCameraRotation(self, recalculate_fkine=False):
        T_0_6, _, _, _, _, _ = self.fkine(recalculate=recalculate_fkine)

        return T_0_6[0:3, 0:3]

    def getCameraHeight(self, recalculate_fkine=False):
        T_0_6, _, _, _, _, _ = self.fkine(recalculate=recalculate_fkine)

        return T_0_6[2, -1]
    
    def getCameraImage(self):
        image_data, resolution = self.sim.getVisionSensorImg(self.cameraHandle)
        return np.frombuffer(image_data, np.uint8).reshape((resolution[0], resolution[1], 3)), resolution

    def step(self):
        self.client.step()

    def __trotx(self, alpha, a):
        return np.array([[1.0, 0.0, 0.0, a], [0.0, np.cos(alpha), -np.sin(alpha), 0.0], [0.0, np.sin(alpha), np.cos(alpha), 0.0], [0.0, 0.0, 0.0, 1.0]])
    
    def __trotz(self, theta, d):
        return np.array([[np.cos(theta), -np.sin(theta), 0.0, 0.0], [np.sin(theta), np.cos(theta), 0.0, 0.0], [0.0, 0.0, 1.0, d], [0.0, 0.0, 0.0, 1.0]])

    def dh(self, theta, d, a, alpha):
        return self.__trotz(theta, d) @ self.__trotx(alpha, a)

q = np.array([0.0, 0.0, np.pi/2, 0.0, -np.pi/2, 0.0]) # Desired starting configuration
robot = UR10Simulation(q)

# Waiting robot to arrive at starting location
while (t := robot.sim.getSimulationTime()) < 3:
    robot.step()

desired_f = np.array([256.0/2, 256.0/2]) # Desired position for feature

error_log = np.zeros((int(T_MAX/TS), 2))
q_log = np.zeros((int(T_MAX/TS), 6))
camera_log = np.zeros((int(T_MAX/TS), 6))
t_log = np.zeros(int(T_MAX/TS))
k = 0

while (t := robot.sim.getSimulationTime()) < T_MAX:
    # Getting camera image and features
    image, resolution = robot.getCameraImage()
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.GaussianBlur(image_gray, (5,5), 0)
    image_gray = cv2.flip(image_gray, 0)
    circles = cv2.HoughCircles(image_gray, cv2.HOUGH_GRADIENT, 1, 100, param1=50,param2=30,minRadius=0,maxRadius=0)
    if (circles is not None):
        f = np.array(circles[0][0][0:2])
    else:
        print('problem in hough circles')

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
plt.show()