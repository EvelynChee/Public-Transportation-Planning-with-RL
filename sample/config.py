# Bus line setups.
# In this case, the bus line is of length 50 and there are 8 bus stations
# in total. Each station are described using a 3-tuple (a, b, c), where a is
# the position of the bus station, b is the mean time interval in between
# the arrival between two passengers at that stations during peak hours and c
# is the same as b but for non-peak hours (e.g. the first station is at
# position 5 and has mean time interval 5 and 35 for peak and non-peak hours
# respectively).  
D = 60
stations = [(1,5,20), (9,1,10), (13,1,15), (17,6,30), (21,5,20), (27,8,12), (33,1,10), (39,6,15), (43,10,30), (47,4,15), (51,15,30)]

# Parameter setups for the reward system and elements of the environment.
max_passenger_per_station = 20
cost_per_deploy = 10
penalty_per_overtake = 10
no_bus_deploy_penalty = 10
passenger_boarded_per_time = 2
waiting_time_cost = 0.05
reward_per_passenger = 0

# Parameter setups for the network and training process.
learning_rate = 0.01
n_iters = 4000
n_epochs = 5000
display_step = 100
batch_size = 4096
gamma = 0.9

# The size of the state vector.
nS = len(stations) + 2
