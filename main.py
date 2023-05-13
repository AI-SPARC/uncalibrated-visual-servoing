from experiment import Experiment, Method
from noise import NoiseProfiler, NoiseType
from ur10_simulation import UR10Simulation
import numpy as np
import pandas as pd
import time
from math import floor
import logging
from json import load
import sys

out_filename = time.strftime("%d_%m_%Y_%H_%M_%S", time.localtime())

# Reading config
with open("config.json", "r") as config_file:
    config = load(config_file)
    
    try:
        # Logging
        logger = logging.getLogger(__name__)
        ch = logging.StreamHandler(sys.stdout)

        if config["log_level"] == "DEBUG":
            logger.setLevel(logging.DEBUG)
        elif config["log_level"] == "INFO":
            logger.setLevel(logging.INFO)
        elif config["log_level"] == "WARNING":
            logger.setLevel(logging.WARNING)
        elif config["log_level"] == "ERROR":
            logger.setLevel(logging.ERROR)
        elif config["log_level"] == "CRITICAL":
            logger.setLevel(logging.CRITICAL)

        fh = logging.FileHandler(out_filename+'.log')
        fh.setLevel(logging.DEBUG)

        log_formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s - %(message)s')
        fh.setFormatter(log_formatter)
        ch.setFormatter(log_formatter)

        logger.addHandler(ch)
        logger.addHandler(fh)

        # Experiments config
        logger.debug("Loading experiments config")
        experiments_config = config["experiments"]

        for key, value in experiments_config.items():
            logger.debug(str(key) + " : " + str(value))

        dt = experiments_config["dt"]
        t_max = experiments_config["t_max"]
        epoch = experiments_config["epoch"]
        ibvs_gain = experiments_config["ibvs_gain"]
        q = np.array(experiments_config["q_start"])
        desired_f = np.array(experiments_config["desired_f"])

        # Estimator config
        logger.debug("Loading estimator config")
        estimator_config = config["estimator"]
        
        for key, value in estimator_config.items():
            logger.debug(str(key) + " : " + str(value))

        method_name = estimator_config["method"]
        if method_name in [e.name for e in Method]:
            method = Method[method_name].value
        else: 
            logger.critical("Estimation method " + method_name + " unknown.")
            sys.exit()
        method_params = estimator_config["estimator_params"]

        # Noise config
        logger.debug("Loading noise config")
        noise_config = config["noise"]

        for key, value in noise_config.items():
            logger.debug(str(key) + " : " + str(value))

        noise_type_name = noise_config["type"]
        if noise_type_name in [e.name for e in NoiseType]:
            noise_type = NoiseType[noise_type_name].value
        else: 
            logger.critical("Noise type " + noise_type_name + " unknown.")
            sys.exit()
        noise_params = noise_config["noise_params"]
        seed = noise_config["seed"]
    
    except KeyError as e:
        logger.critical("Could not load config key: " + str(e))
        sys.exit()

rho_list = np.linspace(0, 0.2, 12)
experiments = []

robot = UR10Simulation(logger=logger)

# Preparing experiment queue
for rho in rho_list:
    
    noise_params["rho"] = rho
    for i in range(epoch):
        # Noise generation
        noise_prof = NoiseProfiler(num_features=len(desired_f), noise_type=noise_type, seed=seed, noise_params=noise_params)
    
        experiments.append(Experiment(q_start=q, desired_f=desired_f, noise_prof=noise_prof, t_s=dt, t_max=t_max, ibvs_gain=ibvs_gain, robot=robot, logger=logger, method=method, method_params=method_params))

first_experiment = True
file_header = True

for i, experiment in enumerate(experiments):
    logger.info("Experiment " + str(i+1) + " of " + str(len(experiments)))
    # Running experiment
    status, t_log, error_log, q_log, f_log, desired_f_log, camera_log, noise_log = experiment.run()

    # Saving data
    logger.debug("Saving experiment data in csv")  
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

    if first_experiment:
        file_header = False
        first_experiment = False
  
    dataframe.to_csv('results/' + out_filename + '.csv', mode='a', index=False, header=file_header)