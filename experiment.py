from enum import Enum
from utils import detect4Circles, gaussianKernel
import numpy as np
import logging

class Method(Enum):
    ANALYTICAL = 1
    KF = 2
    MCKF = 3
    IMCCKF = 4
    GMCKF = 5

class ExperimentStatus(Enum):
    SUCCESS = 0
    FAIL = 1

class Experiment:
    def __init__(self, q_start: list, desired_f: list, noise_prof: object, t_s: float, t_max: float, ibvs_gain: float, robot: object, method: enumerate, logger: object = None, **method_params) -> None:
       
        self.q_start = q_start
        self.desired_f = desired_f
        self.robot = robot
        self.noise_prof = noise_prof
        self.t_s = t_s
        self.t_max = t_max
        self.ibvs_gain = ibvs_gain
        self.method = method
        
        if "method_params" in method_params:
            method_params = method_params["method_params"]

        if self.method == Method.KF or self.method == Method.MCKF or self.method == Method.IMCCKF or self.method == Method.GMCKF:
            
            self.initial_guess = method_params['initial_guess']

            if self.method == Method.MCKF or self.method == Method.IMCCKF or self.method == Method.GMCKF:
                self.kernel_bw = method_params['kernel_bw']
                self.fpi_threshold = method_params['fpi_threshold']
                self.fpi_epoch_max = method_params['fpi_epoch_max']
                self.annealing = method_params['annealing']
        
        self.logger = logging.getLogger(__name__)
        if logger is not None:
            self.logger.setLevel(logger.level)
            for handler in logger.handlers:
                self.logger.addHandler(handler)

    def run(self) -> list:
        experiment_status = ExperimentStatus.SUCCESS
    
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
        kernel_bw_log = np.zeros((int(self.t_max/self.t_s)))

        if self.method == Method.KF or self.method == Method.MCKF or self.method == Method.IMCCKF or self.method == Method.GMCKF:
            X = np.zeros((m*n, 1))
            Z = np.zeros((m, 1))
            H = np.zeros((m, m*n))
            P = np.eye(m*n)
            K = np.zeros((m*n, m))
            Q = np.eye(m*n)
            R = np.eye(m)

            dp = np.zeros((n, 1))
            dq = np.zeros((n, 1))

            first_run = True
            dp_real = np.zeros(6)
            old_pose = self.robot.computePose(recalculate_fkine=True)
            old_q = self.robot.getJointsPos().copy()

            if self.initial_guess:
                # Getting camera image and features
                image, resolution = self.robot.getCameraImage()
                try:
                    f = detect4Circles(image)
                except Exception as e:
                    self.logger.error(e) # only print problem in hough circles, but continue with older f

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

                J_feature = J_image @ np.kron(np.eye(2), self.robot.getCameraRotation().T) @ self.robot.jacobian()

                X = J_feature.reshape((m*n, 1))
                #X = J_image.reshape((m*n, 1))
            else:
                X = np.random.default_rng().random((m*n, 1))

        k = 0
        k_max = int(self.t_max/self.t_s)
        Br = np.linalg.cholesky(R)
        Br_inv = np.linalg.inv(Br)

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
                self.logger.error(e) # only print problem in hough circles, but continue with older f

            # Calculating / Estimating jacobian
            J_image = np.zeros((len(f), 6))
            
            if self.method == Method.ANALYTICAL:
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
                
                J_feature = J_image @ np.kron(np.eye(2), self.robot.getCameraRotation().T) @ self.robot.jacobian()

            elif self.method == Method.KF or self.method == Method.MCKF or self.method == Method.IMCCKF or self.method == Method.GMCKF:
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
                dq_real = self.robot.getJointsPos() - old_q
                #print(dq_real)
                if first_run:
                    first_run = False
                else:
                    #H = np.kron(np.eye(m), dp.ravel())
                    #H = np.kron(np.eye(m), dp_real)
                    H = np.kron(np.eye(m), dq.ravel())

                skip_correction = False
                if self.method == Method.KF:
                    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
                    X = X + K @ (Z - H @ X)
                elif self.method == Method.MCKF:
                    kernel_bw = self.kernel_bw
                    if self.annealing:
                        # Annealing changes the kernel bandwidth throughout the simulation
                        temp = 1 - k/k_max
                        kernel_bw = self.kernel_bw + 100*temp


                    # Finding Bp using Cholesky
                    Bp = np.linalg.cholesky(P)
                    Br = np.linalg.cholesky(R) # Maybe it can be outside the loop
                    B = np.vstack([np.hstack([Bp, np.zeros((Bp.shape[0], Br.shape[1]))]), np.hstack([np.zeros((Br.shape[0], Bp.shape[1])), Br])])
                    B_inv = np.linalg.pinv(B)

                    # Correction
                    difference = np.inf
                    epoch = 0
                    X_corrected = X.copy()

                    D = B_inv @ np.vstack([X, Z]) # Our y is Z
                    W = B_inv @ np.vstack([np.eye(m*n), H])
                    while difference > self.fpi_threshold and epoch < self.fpi_epoch_max:
                        ## Correntropy kernels
                        e = D - W @ X_corrected
                        
                        Cx = np.diag([gaussianKernel(e[i, 0], kernel_bw) for i in range(0, m*n)])
                        Cy = np.diag([gaussianKernel(e[i, 0], kernel_bw) for i in range(m*n, m*n + m)])
                        #print(np.linalg.det(Cx))

                        ## Compute optimal gain
                        P_hat = Bp @ np.linalg.inv(Cx) @ Bp.T
                        try:
                            # If extremely large noises, Cy may be nearly singular, resulting in numerical problems
                            # So we check the following conditions to choose if only the prediction step will happen
                            #if (np.linalg.det(Cy) == 0.0):
                            #    raise np.linalg.LinAlgError('Singular matrix')
                            #Cy_inv = Cy.T @ np.linalg.inv(Cy @ Cy.T + 0.001**2 * np.eye(m))
                            Cy_inv = np.linalg.inv(Cy)
                            R_hat = Br @ Cy_inv @ Br.T
                        except np.linalg.LinAlgError:
                            self.logger.warning("Cy is singular, skipping correction step")
                            skip_correction = True
                            break

                        K = P_hat @ H.T @ np.linalg.inv(H @ P_hat @ H.T + R_hat)

                        ## Correct X
                        X_corrected_old = X_corrected.copy()
                        X_corrected = X + K @ (Z - H @ X)

                        difference = np.linalg.norm(X_corrected - X_corrected_old)/np.linalg.norm(X_corrected_old)
                        epoch += 1
                        if epoch == self.fpi_epoch_max:
                            self.logger.warning("Reached max epoch")
                            skip_correction = True
                    if not skip_correction:
                        X = X_corrected
                elif self.method == Method.IMCCKF:
                    kernel_bw = self.kernel_bw
                    if self.annealing:
                        # Annealing changes the kernel bandwidth throughout the simulation
                        temp = 1 - k/k_max
                        kernel_bw = self.kernel_bw + 100*temp

                    innov = Z - H @ X
                    norm_innov = np.sqrt(innov.T @ np.linalg.inv(R) @ innov) # computing innovation error in the norm of R_inv
                    
                    Cy = gaussianKernel(norm_innov, kernel_bw)
                    R_e = Cy * H @ P @ H.T + R
                    #K = Cy @ np.linalg.inv(np.linalg.inv(P) + Cy @ H.T @ np.linalg.inv(R) @ H) @ H.T @ np.linalg.inv(R)
                    K = Cy * P @ H.T @ np.linalg.inv(R_e)
                    X = X + K @ innov
                elif self.method == Method.GMCKF:
                    kernel_bw = self.kernel_bw
                    if self.annealing:
                        # Annealing changes the kernel bandwidth throughout the simulation
                        temp = 1 - k/k_max
                        kernel_bw = self.kernel_bw + 100*temp
                        
                    ## Correntropy kernels
                    e = Br_inv @ Z - Br_inv @ H @ X
                    
                    Cy = np.diag([gaussianKernel(e[i, 0], kernel_bw) for i in range(0, m)])
                    
                    try:
                        #Cy_inv = Cy.T @ np.linalg.inv(Cy @ Cy.T + 0.001**2 * np.eye(m))
                        Cy_inv = np.linalg.inv(Cy + 0.001**2 * np.eye(m))
                        R_hat = Br @ Cy_inv @ Br.T                    

                        S = H @ P @ H.T + R_hat
                        #S_inv = S.T @ np.linalg.inv(S @ S.T + 0.001**2 * np.eye(m))
                        S_inv = np.linalg.inv(S)
                        K = P @ H.T @ S_inv

                        #self.logger.debug(K)
                        #Cy_inv = np.linalg.inv(Cy)

                        X = X + K @ (Z - H @ X)
                    except np.linalg.LinAlgError:
                        self.logger.warning("Cy is singular, skipping control step")
                        skip_correction = True
               
                if not skip_correction:
                    P = (np.eye(m*n) - K @ H) @ P @ (np.eye(m*n) - K @ H).T + K @ R @ K.T
                    #P = (np.eye(m*n) - K @ H) @ P
                #J_image = X.reshape((m, n))
                J_feature = X.reshape((m, n))

            error = f - self.desired_f
            
            try:
                # IBVS Control Law
                kappa = np.ones(error.shape)
                if self.method == Method.GMCKF:
                    kappa = np.array([gaussianKernel(e[i, 0], kernel_bw) for i in range(0, m)])
                
                #dp = - self.ibvs_gain * np.linalg.pinv(J_image) @ error.reshape((m, 1))
                #dx = np.kron(np.eye(2), self.robot.getCameraRotation().T) @ dp
                dq = - self.ibvs_gain * np.linalg.pinv(J_feature) @ (kappa.reshape((m, 1)) * error.reshape((m, 1)))
            except:
                experiment_status = ExperimentStatus.FAIL
                self.logger.error('Experiment failed')
                break

            # Invkine
            #dq = np.linalg.pinv(self.robot.jacobian()) @ dx
            new_q = self.robot.getJointsPos() + dq.ravel() * self.t_s

            # Logging
            q_log[k] = self.robot.getJointsPos()
            camera_log[k] = self.robot.computePose()
            f_log[k] = f
            desired_f_log[k] = self.desired_f
            error_log[k] = error
            noise_log[k] = noise
            t_log[k] = t
            kernel_bw_log[k] = kernel_bw if self.method == Method.MCKF else -1
            k += 1
            self.logger.debug('time: ' + str(t) + '; error: ' + str(np.linalg.norm(error)))

            # Send theta command to robot
            self.robot.setJointsPos(new_q)

            if self.method == Method.KF or self.method == Method.MCKF or self.method == Method.IMCCKF or self.method == Method.GMCKF:
                # Save new_pose as old_pose for next iteration
                old_pose = new_pose.copy()
                old_q = self.robot.getJointsPos().copy()

            # next_step
            self.robot.step()
    
        t_log = np.delete(t_log, [i for i in range(k, len(t_log))], axis=0)
        error_log = np.delete(error_log, [i for i in range(k, len(error_log))], axis=0)
        q_log = np.delete(q_log, [i for i in range(k, len(q_log))], axis=0)
        f_log = np.delete(f_log, [i for i in range(k, len(f_log))], axis=0)
        desired_f_log = np.delete(desired_f_log, [i for i in range(k, len(desired_f_log))], axis=0)
        camera_log = np.delete(camera_log, [i for i in range(k, len(camera_log))], axis=0)
        noise_log = np.delete(noise_log, [i for i in range(k, len(noise_log))], axis=0)
        kernel_bw_log = np.delete(kernel_bw_log, [i for i in range(k, len(kernel_bw_log))], axis=0)

        self.robot.stop()

        if experiment_status == ExperimentStatus.SUCCESS:
            self.logger.info("Experiment success")

        return experiment_status, t_log, error_log, q_log, f_log, desired_f_log, camera_log, noise_log, kernel_bw_log