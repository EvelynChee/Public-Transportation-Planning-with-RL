import sys
import os
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, '../..')

print("Start: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

from config import *
from network import *
from helper import *

tf.reset_default_graph()

# Set up the state-value estimator and policy estimator
estimator_policy = PolicyEstimator(learning_rate, nS)
estimator_value = StateValueEstimator(learning_rate, nS)

# Initialize bus lines
line1 = create_line(D, stations, max_passenger_per_station, waiting_time_cost, \
                   cost_per_deploy, passenger_boarded_per_time, penalty_per_overtake, \
                    no_bus_deploy_penalty, reward_per_passenger)

line2 = create_line(D, stations, max_passenger_per_station, waiting_time_cost, \
                   cost_per_deploy, passenger_boarded_per_time, penalty_per_overtake, \
                    no_bus_deploy_penalty, reward_per_passenger)

# Add ops to save and restore all the variables
saver = tf.train.Saver()

with tf.Session() as sess:
    print("Starting policy gradient...")
    # Run the initializer
    sess.run(tf.global_variables_initializer())
    
    # Start training
    print("Training in progress...")
    avg_rewards = reinforce(line1, estimator_policy, estimator_value, n_epochs, \
                            n_iters, batch_size, display_step)

    # Save the variables to disk
    print("Saving model...")
    save_path = saver.save(sess, os.getcwd() + "/policy_grad.ckpt")

# Plot the graphs of average rewards during testing phases of the policy
# gradient method and the hand-crafted policies
print("Plotting resulting graph...")
plt.figure()
plt.plot(avg_rewards, label="reinforce")
print("Plotting baseline 1...")
plt.plot(fixed(line2, 30, int(n_epochs/display_step)), label="after 30")
plt.legend(loc='best',prop={'size': 10})
plt.ylabel("Average reward")
plt.xlabel("Epochs/100")
plt.savefig('train.png')

print("End: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
 
