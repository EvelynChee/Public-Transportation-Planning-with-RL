Public Transportation Planning using RL
----------------------------------------


Introduction -  Bus Traveling in a Loop with Peak Hours
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

API Documentation
~~~~~~~~~~~~~~~~~~~

.. toctree::
  :maxdepth: 1


  env
  network
  helper
