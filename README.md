# Uncalibrated Visual Servoing with RMCKF
This repository sources the code for the work: Regularized Maximum-Correntropy Criterion Kalman Filter for uncalibrated visual servoing in the presence of non-gaussian feature tracking noise

## Abstract

Some advantages of using cameras as sensor devices on feedback systems are the flexibility of the data it represents, the possibility to extract real-time information, and it does not require contact to operate. However, in unstructured scenarios, Image-Based Visual Servoing (IBVS) robot tasks are challenging. Camera calibration and robot kinematics can approximate a jacobian that maps the image features space to the robot actuation space, but they can become error-prone or require online changes. Uncalibrated visual servoing (UVS) aims at guessing the jacobian using environment information on an estimator, such as the Kalman filter. The Kalman filter is optimal with Gaussian noise, but unstructured environments may present target occlusion, reflection, and other characteristics that confuse feature extraction algorithms, generating outliers. This work proposes RMCKF, a correntropy-induced estimator based on the Kalman Filter and the Maximum Correntropy Criterion that can handle non-gaussian feature extraction noise. Unlike other approaches, we designed RMCKF for particularities in UVS, to deal with independent features, the IBVS control action, and simulated annealing. We designed Monte Carlo experiments to test RMCKF with non-gaussian Kalman Filter based techniques. The results showed that the proposed technique could outperform its relatives, especially in impulsive noise scenarios and various starting configurations.

## Methods
The UVS scenarios were simulated using the CoppeliaSim robotics simulator, which implements the Bullet physics engine. That tool allows us to model a robot, control its joints, place cameras attached to or outside the robot structure, and set different parameters on physics, image acquisition, and environment. CoppeliaSim provides an Application Programming Interface (API) in various programming languages to implement custom control algorithms. This study used the Python programming language since its numerical and scientific libraries offer fast prototyping for the discussed techniques.

A Universal Robots UR10 was placed in the simulation using one of the predefined models in CoppeliaSim. That robot was chosen by a correlated study, which described a UVS system using MCKF . Using the standard DH formulation and link sizes obtained from the official tech specifications. The robot joints are configured in Kinematic mode on the simulator to receive joint position commands. The regular API that CoppeliaSim provides functions to read the joint positions directly, simulating an absolute encoder response.

## Running

Considering that you already have a python and pip install, run the following scripts:

* Create and source an isolated python virtual environemnt 

```
$ python -m venv ./virtualenv
$ source ./virtualenv/bin/activate
```

* Install required packages

```
$ pip install -r requirements.txt
```

* Open a CoppeliaSim scenario from the *scenarios* folder 

* Setup simulation configuration file (*config.json*)

* Run experiments batch

```
$ python main.py
```