from experiment import Experiment, Method
from noise import NoiseProfiler, NoiseType
from ur10_simulation import UR10Simulation
import numpy as np
import pandas as pd
import time
from math import floor

EPOCH = 2
SEED = 12345

filename = time.strftime("%d_%M_%Y_%H_%M_%S", time.localtime())

rho_list = np.linspace(0, 0.2, 12)
experiments = []
q = np.array([0.0, -np.pi/8, np.pi/2 + np.pi/8, 0.0, -np.pi/2, 0.0])

# 4 features
desired_f = np.array([149., 145., 125., 121., 101., 145., 125., 169.]) # Center
#desired_f = np.array([125., 121., 101., 145., 125., 169., 149., 145.]) # Rotation

robot = UR10Simulation()

# Preparing experiment queue
for rho in rho_list:
    for i in range(EPOCH):
        # Noise generation
        noise_prof = NoiseProfiler(num_features=len(desired_f), noise_type=NoiseType.GAUSSIAN_MIXTURE, mean=5, std=0.25, rho=rho, seed=SEED)
    
        experiments.append(Experiment(robot=robot, noise_prof=noise_prof, method=Method.MCKF, q_start=q, desired_f=desired_f, initial_guess=True))

for i, experiment in enumerate(experiments):
    print("Experiment " + str(i+1) + " of " + str(len(experiments)))
    # Running experiment
    status, t_log, error_log, q_log, f_log, desired_f_log, camera_log, noise_log = experiment.run()

    # Saving data
    dataframe = pd.DataFrame(data={
        'experiment_id': i,
        'rho': rho_list[int(floor(i/2))], # Temporary
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
        'f_7': f_log[:, 6],
        'f_8': f_log[:, 7],
        'desired_f_1': desired_f_log[:, 0],
        'desired_f_2': desired_f_log[:, 1],
        'desired_f_3': desired_f_log[:, 2],
        'desired_f_4': desired_f_log[:, 3],
        'desired_f_5': desired_f_log[:, 4],
        'desired_f_6': desired_f_log[:, 5],
        'desired_f_7': desired_f_log[:, 6],
        'desired_f_8': desired_f_log[:, 7],
        'noise_1': noise_log[:, 0],
        'noise_2': noise_log[:, 1],
        'noise_3': noise_log[:, 2],
        'noise_4': noise_log[:, 3],
        'noise_5': noise_log[:, 4],
        'noise_6': noise_log[:, 5],
        'noise_7': noise_log[:, 6],
        'noise_8': noise_log[:, 7]
    })

    
    dataframe.to_csv('results/' + filename + '.csv', mode='a', index=False)


