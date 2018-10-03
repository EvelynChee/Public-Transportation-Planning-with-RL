PUBLIC TRANSPORTATION PLANNING - BUS TRAVELING IN A LOOP WITH PEAK HOURS
-------------------------------------------------------------------------

The aim of the experiment is to examine if reinforcement learning could be implemented in 
public transportation planning. Using reinforcement learning, we hope to provide better 
insights of the current system and hence improve the planning process. A simulated environment 
will be set up to closely replicate the transportation system. 

In this experiment, we will be using buses traveling in a loop as an example. In other words,
there is only one terminal and buses will always be back in the terminal at the end of each
trip. In addition, there are peak hours throughout the day. During the peak hours, the arrival
rate of the passengers at the stations may or may not be higher. There are only two actions 
available for the agent: to deploy or not to deploy a bus. As for the state, it contains 
information on the current time of the day, the number of passengers at each bus station and 
the distance of the nearest bus from the terminal.


File Description
------------------
1. env.py -- Contains classes (Station, Bus, BusLine) to set up elements of the environment.
2. helper.py -- Contains helper functions (e.g. hand-crafted policies, reinforcement learning
		algorithm) which are used in the training process.
3. network.py -- Contains classes for the non-linear estimators (ActionValueEstimator, 
		 StateValueEstimator, PolicyEstimator) which will be trained and used for
		 prediction. 
4. Theory & Algorithms for Reinforcement Learning.pdf -- Introduction to theories and algorithms
		for reinforcement learning with results on its application in public transportation
		planning.

Example
---------
The 'sample' directory contains an example on how to execute the training process. It contains:
1. config.py -- Sets up the configurations for the environment and training parameters.
2. train.py -- Run this file to execute the training process  

