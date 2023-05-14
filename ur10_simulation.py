import numpy as np
import utils
from zmqRemoteApi import RemoteAPIClient
from time import sleep
import logging

class UR10Simulation():
    def __init__(self, logger: object = None, visualization: bool = True) -> None:
        # New instance of API client
        self.client = RemoteAPIClient()

        self.sim = self.client.getObject('sim')

        self.client.setStepping(True)

        # Getting joint handles and setting home position
        self.joints = [self.sim.getObject('./joint', {'index': i}) for i in range(6)]
        self.cameraHandle = self.sim.getObject('./sensor')
        
        self.q = np.zeros(6)
        self.dq = np.zeros(6)

        self.perspective_angle = 65
        self.visualization = visualization
        
        self.logger = logging.getLogger(__name__)
        if logger is not None:
            self.logger.setLevel(logger.level)
            for handler in logger.handlers:
                self.logger.addHandler(handler)
    
    def __del__(self) -> None:
        del self.client

    def start(self, q: list = None):
        if q is not None:
            self.q = q.copy()
            
            # Setting initial values
            self.setJointsPos(self.q)
            self.sim.setJointPosition(self.joints[0], self.q[0])
            self.sim.setJointPosition(self.joints[1], self.q[1])
            self.sim.setJointPosition(self.joints[2], self.q[2])
            self.sim.setJointPosition(self.joints[3], self.q[3])
            self.sim.setJointPosition(self.joints[4], self.q[4])
            self.sim.setJointPosition(self.joints[5], self.q[5])
        
        else:
            self.q = self.getJointsPos()

        self.sim.startSimulation()
        self.logger.debug("Starting simulation")
        while (self.sim.getSimulationState() != self.sim.simulation_advancing_firstafterstop) and (self.sim.getSimulationState() != self.sim.simulation_advancing_running):
            sleep(0.1)
        self.sim.setBoolParameter(self.sim.boolparam_display_enabled, self.visualization)
        self.logger.debug("Simulation started")
        self.step()

        self.T_0_6, self.T_0_5, self.T_0_4, self.T_0_3, self.T_0_2, self.T_0_1 = self.fkine(recalculate=True, all_transforms=True)

    def stop(self):
        self.sim.stopSimulation()
        self.logger.debug("Stopping simulation")
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            sleep(0.1)
        self.logger.debug("Simulation stopped")


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

    def fkine(self, recalculate=False, all_transforms=False):
        if recalculate:
            self.getJointsPos()
            self.T_0_1 = self.dh(self.q[0], 0.128, 0, -np.pi/2)
            self.T_0_2 = self.T_0_1 @ self.dh(self.q[1] - np.pi/2, 0, 0.6127, 0)
            self.T_0_3 = self.T_0_2 @ self.dh(self.q[2], 0, 0.5716, 0)
            self.T_0_4 = self.T_0_3 @ self.dh(self.q[3] - np.pi/2, 0.1639, 0, -np.pi/2)
            self.T_0_5 = self.T_0_4 @ self.dh(self.q[4], 0.1157, 0, np.pi/2)
            self.T_0_6 = self.T_0_5 @ self.dh(self.q[5] + np.pi, 0.0922, 0, 0)

        if all_transforms:
            return self.T_0_6, self.T_0_5, self.T_0_4, self.T_0_3, self.T_0_2, self.T_0_1
        else:
            return self.T_0_6

    def jacobian(self, recalculate_fkine=False):

        T_0_6, T_0_5, T_0_4, T_0_3, T_0_2, T_0_1 = self.fkine(recalculate=recalculate_fkine, all_transforms=True)
        
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
        T_0_6 = self.fkine(recalculate=recalculate_fkine)

        return T_0_6[0:3, 0:3]

    def getCameraHeight(self, recalculate_fkine=False):
        T_0_6 = self.fkine(recalculate=recalculate_fkine)

        return T_0_6[2, -1]
    
    def getCameraPosition(self, recalculate_fkine=False):
        T_0_6 = self.fkine(recalculate=recalculate_fkine)

        return T_0_6[0:3, -1]
    
    def computePose(self, recalculate_fkine=False):
        #M = self.fkine(recalculate=recalculate_fkine)
        cameraPose = np.array(self.sim.getObjectPose(self.cameraHandle + self.sim.handleflag_wxyzquat, -1))

        pose = np.zeros(6)
        #pose[0:3] = M[0:3, -1].ravel()
        pose[0:3] = cameraPose[0:3]
        # pose[3] = np.arctan2(M[2,1], M[2,2])
        # pose[4] = np.arctan2(-M[2,0], np.sqrt(M[2,1]**2 + M[2,2]**2))
        # pose[5] = np.arctan2(M[1,0], M[0,0])
        pose[3], pose[4], pose[5] = utils.quat2euler(cameraPose[3:7])

        return pose
    
    def computeZ(self, n=1, recalculate_fkine=False):
        disc_r = self.sim.getObject('./red_disc')
        if (n > 1):
            disc_g = self.sim.getObject('./green_disc')
            disc_b = self.sim.getObject('./blue_disc')
            if (n > 3):
                disc_p = self.sim.getObject('./pink_disc')

        r = self.sim.getObjectPosition(disc_r, -1)
        if (n > 1):
            g = self.sim.getObjectPosition(disc_g, -1)
            b = self.sim.getObjectPosition(disc_b, -1)
            if (n > 3):
                p = self.sim.getObjectPosition(disc_p, -1)

        Z = np.zeros(n)
        camera_pos = self.getCameraPosition(recalculate_fkine)

        Z[0] = np.linalg.norm(camera_pos - r)
        if (n > 1):
            Z[1] = np.linalg.norm(camera_pos - g)
            Z[2] = np.linalg.norm(camera_pos - b)
            if (n > 3):
                Z[3] = np.linalg.norm(camera_pos - p)

        return Z
    
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