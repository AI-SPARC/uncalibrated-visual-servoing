from enum import Enum
from utils import detect4Circles, gaussianKernel
import numpy as np

class Method(Enum):
    IBVS = 1
    KF = 2
    MCKF = 3

class ExperimentStatus(Enum):
    OK = 0
    ERROR = 1

class Experiment:
    def __init__(self, q_start: list, desired_f: list, noise_prof: object = None, t_s: float = 0.05, t_max: float = 25, ibvs_gain: float = 0.5, robot: object = None, method: enumerate = Method.IBVS, **method_params) -> None:
       
        self.q_start = q_start
        self.desired_f = desired_f
        self.robot = robot
        self.noise_prof = noise_prof
        self.t_s = t_s
        self.t_max = t_max
        self.ibvs_gain = ibvs_gain
        self.method = method
        
        if self.method == Method.KF or self.method == Method.MCKF:
            self.initial_guess = method_params['initial_guess'] if "initial_guess" in method_params else True

            if self.method == Method.MCKF:
                self.kernel_bw = method_params['kernel_bw'] if "kernel_bw" in method_params else 20
                self.fpi_threshold = method_params['fpi_threshold'] if "fpi_threshold" in method_params else 0.01
                self.fpi_epoch_max = method_params['fpi_epoch_max'] if "fpi_epoch_max" in method_params else 100
        

    def run(self) -> list:
        self.robot.start(self.q_start)

        m = len(self.desired_f)
        n = 6

        f = np.zeros(m)
        f_old = None
        noise = np.zeros(m)

        error_log = np.zeros((int(self.t_max/self.t_s), len(f)))
        f_log = np.zeros((int(self.t_max/self.t_s), len(f)))
        q_log = np.zeros((int(self.t_max/self.t_s), 6))
        camera_log = np.zeros((int(self.t_max/self.t_s), n))
        t_log = np.zeros(int(self.t_max/self.t_s))
        desired_f_log = np.zeros((int(self.t_max/self.t_s), len(f)))
        noise_log = np.zeros((int(self.t_max/self.t_s), len(f)))

        if self.method == Method.KF or self.method == Method.MCKF:
            X = np.zeros((m*n, 1))
            Z = np.zeros((m, 1))
            H = np.zeros((m, m*n))
            P = np.eye(m*n)
            K = np.zeros((m*n, m))
            Q = np.eye(m*n)
            R = np.eye(m)

            dp = np.zeros((n, 1))

            first_run = True
            dp_real = np.zeros(6)
            old_pose = self.robot.computePose(recalculate_fkine=True)

            if self.initial_guess:
                # Getting camera image and features
                image, resolution = self.robot.getCameraImage()
                try:
                    f = detect4Circles(image)
                except Exception as e:
                    print(e) # only print problem in hough circles, but continue with older f

                # Calculate image jacobian
                J_image = np.zeros((m, n)) # need to find
                Z_camera = self.robot.computeZ(4)
                focal = resolution[0]/(2*np.tan(0.5*np.deg2rad(self.robot.perspective_angle)))
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
            else:
                X = np.random.default_rng().random((m*n, 1))

        k = 0

        # Main loop
        while (t := self.robot.sim.getSimulationTime()) < self.t_max:
            # Getting f
            image, resolution = self.robot.getCameraImage()
            f_old = f.copy()
            try:
                f = detect4Circles(image)

                if self.noise_prof is not None:
                    # Adding noise
                    noise = self.noise_prof.getNoise()
                    f += noise

                if (f_old is None):
                    f_old = f.copy()
            except Exception as e:
                print(e) # only print problem in hough circles, but continue with older f
                break

            # Calculating / Estimating jacobian
            J_image = np.zeros((len(f), 6))
            
            if self.method == Method.IBVS:
                Z_camera = self.robot.computeZ(4, recalculate_fkine=True)
                focal = resolution[0]/(2*np.tan(0.5*np.deg2rad(self.robot.perspective_angle)))
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

            elif self.method == Method.KF or self.method == Method.MCKF:
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
                Z[6,0] = f[6] - f_old[6]
                Z[7,0] = f[7] - f_old[7]

                new_pose = self.robot.computePose(recalculate_fkine=True)
                dp_real = new_pose - old_pose
                if first_run:
                    first_run = False
                else:
                    H = np.kron(np.eye(m), dp.ravel())

                if self.method == Method.KF:
                    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
                    X = X + K @ (Z - H @ X)
                elif self.method == Method.MCKF:
                    # Finding Bp using Cholesky
                    Bp = np.linalg.cholesky(P)
                    Br = np.linalg.cholesky(R) # Maybe it can be outside the loop
                    B = np.vstack([np.hstack([Bp, np.zeros((Bp.shape[0], Br.shape[1]))]), np.hstack([np.zeros((Br.shape[0], Bp.shape[1])), Br])])
                    B_inv = np.linalg.pinv(B)

                    # Correction
                    difference = np.inf
                    epoch = 0
                    X_corrected = X.copy()
                    while difference > self.fpi_threshold and epoch < self.fpi_epoch_max:
                        ## Correntropy kernels
                        D = B_inv @ np.vstack([X, Z]) # Our y is Z
                        W = B_inv @ np.vstack([np.eye(m*n), H])
                        e = D - W @ X_corrected
                        
                        Cx = np.diag([gaussianKernel(e[i, 0], self.kernel_bw) for i in range(0, m*n)])
                        Cy = np.diag([gaussianKernel(e[i, 0], self.kernel_bw) for i in range(m*n, m*n + m)])
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
                        if epoch == self.fpi_epoch_max:
                            print("Reached max epoch")
                    X = X_corrected
                
                P = (np.eye(m*n) - K @ H) @ P @ (np.eye(m*n) - K @ H).T + K @ R @ K.T
                J_image = X.reshape((m, n))

            error = f - self.desired_f
            
            # IBVS Control Law
            dp = - self.ibvs_gain * np.linalg.pinv(J_image) @ error.reshape((m, 1))
            dx = np.kron(np.eye(2), self.robot.getCameraRotation().T) @ dp

            # Invkine
            dq = np.linalg.pinv(self.robot.jacobian()) @ dx
            new_q = self.robot.getJointsPos() + dq.ravel() * self.t_s

            # Logging
            q_log[k] = self.robot.getJointsPos()
            camera_log[k] = self.robot.computePose()
            f_log[k] = f
            desired_f_log[k] = self.desired_f
            error_log[k] = error
            noise_log[k] = noise
            t_log[k] = k*self.t_s
            k += 1
            print('time: ' + str(t) + '; error: ' + str(np.linalg.norm(error)))

            # Send theta command to robot
            self.robot.setJointsPos(new_q)

            if self.method == Method.KF or self.method == Method.MCKF:
                # Save new_pose as old_pose for next iteration
                old_pose = new_pose

            # next_step
            self.robot.step()
    
        t_log = np.delete(t_log, [i for i in range(k, len(t_log))], axis=0)
        error_log = np.delete(error_log, [i for i in range(k, len(error_log))], axis=0)
        q_log = np.delete(q_log, [i for i in range(k, len(q_log))], axis=0)
        f_log = np.delete(f_log, [i for i in range(k, len(f_log))], axis=0)
        desired_f_log = np.delete(desired_f_log, [i for i in range(k, len(desired_f_log))], axis=0)
        camera_log = np.delete(camera_log, [i for i in range(k, len(camera_log))], axis=0)
        noise_log = np.delete(noise_log, [i for i in range(k, len(noise_log))], axis=0)

        self.robot.stop()

        return ExperimentStatus.OK, t_log, error_log, q_log, f_log, desired_f_log, camera_log, noise_log