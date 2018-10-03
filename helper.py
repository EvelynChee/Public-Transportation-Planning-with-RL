from env import *

def create_line(D, stations, max_passengers, waiting_cost, deploy_cost, \
                boarding_rate, overtake_penalty, no_bus_penalty, pickup_reward):
    """Creates an object of the BusLine class from the env module.

    Args:
       D (int): Distance of the bus route.
       stations (list): List of 3-tuples with the 1st being the location of
                        the station in the bus route, the 2nd being the mean
                        time interval between the arrival of two passengers
                        at the station during peak hour and the 3rd is the same
                        but during non-peak hour.
       max_passengers (int): Max number of passengers each station could contain.
       waiting_cost (float): Cost for each waiting passenger per unit time.
       deploy_cost (float): Cost of deploying a bus.
       boarding_rate (int): Number of passengers boarding a bus per unit time.
       overtake_penalty (float): Cost for a bus overtaking another.
       no_bus_penalty (float): Penalty for deploying a bus when there is none.
       pickup_reward (float): Reward for picking up a passenger.
    
    Returns:
       BusLine created.

    """
    # Creates a new bus route with buses traveling in a loop.
    line = BusLine(D, waiting_cost, overtake_penalty, no_bus_penalty, pickup_reward)
    
    # Adding stations to the bus route.
    for loc, peak_rate, non_peak_rate in stations:
        line.add_station(peak_rate, non_peak_rate, loc, max_passengers)
    
    # Adding 10 buses to the terminal of the bus route with all buses having
    # the same deployment cost and boarding rate.
    for _ in range(10):
        line.add_bus(deploy_cost, boarding_rate)
    
    return line

def batch_sampling3(x, y, z, batch_size):
    """Randomly pick elements from the three arrays. Elements of the same indexes
    are picked from the three arrays and the pairing will be retained.

    Args:
       x (array): Array of elements.
       y (array): Array of elements with the same length as x.
       z (array): Array of elements with the same length as x.
       batch_size (int): Number of elements to be sampled.

    Returns:
       Three arrays with elements sampled from array x, y and z respectively.

    """
    samples = np.random.randint(len(x), size=batch_size)
    return x[samples], y[samples], z[samples]    

###################################################################################
#                             Handcrafted Policies                                #
###################################################################################

def no_bus(line, passenger, loc, n_epochs, n_test=1000):
    """To get the average rewards of a handcrafted policy. The policy is to deploy
    a bus when the total number of passengers in the bus route exceeds a certain
    amount and the nearest bus from the terminal is further than a certain distance. 

    Args:
       line (BusLine): Bus route that the policy will be implemented on.
       passenger (float): The threshold that the total number of passengers out of
                          out of the total maximum that the bus route could contain
                          should exceed to deploy a bus.
       loc (int): The distance from the terminal that the nearest bus should reach
                  before another bus could be deployed.
       n_epochs (int): Number of epochs to run.
       n_test (int, optional): Number of iterations in each epochs. Default is 1000.
       
    Returns:
       An array with elements being the average reward per unit time in each epoch.
    
    """
    avg = np.empty(n_epochs)    # Stores the average rewards of each epochs.
    rewards = np.empty(n_test)  # Stores the rewards at each time step in one epoch.
    for ep in range(n_epochs):
        for i in range(n_test):
            # Get the current representation of the environment.
            state = line.get_feature()[0]

            # Take the action of deploying a bus if both conditions are satisfied.
            if np.sum(state[1:-1]) > passenger and state[-1] >= loc/(line.D+1):
                rewards[i] = line.take_action(1)
            else:
                rewards[i] = line.take_action(0)

        avg[ep] = np.mean(rewards)
                
    return avg

def no_bus_timing(line, passenger1, passenger2, loc, n_epochs, n_test=1000):
    """To get the average rewards of a handcrafted policy. The policy is to deploy
    a bus when the total number of passengers in the bus route exceeds a certain
    amount and the nearest bus from the terminal is further than a certain distance.
    The threshold on total number of passengers could be different during peak and
    non-peak hour.

    Args:
       line (BusLine): Bus route that the policy will be implemented on.
       passenger1 (float): The threshold that the total number of passengers out of
                           out of the total maximum that the bus route could contain
                           should exceed to deploy a bus during peak hour.
       passenger2 (float): The threshold that the total number of passengers out of
                           out of the total maximum that the bus route could contain
                           should exceed to deploy a bus during non-peak hour.
       loc (int): The distance from the terminal that the nearest bus should reach
                  before another bus could be deployed.
       n_epochs (int): Number of epochs to run.
       n_test (int, optional): Number of iterations in each epochs. Default is 1000.
       
    Returns:
       An array with elements being the average reward per unit time in each epoch.
    
    """
    avg = np.empty(n_epochs)    # Stores the average rewards of each epochs.
    rewards = np.empty(n_test)  # Stores the rewards at each time step in one epoch.
    for ep in range(n_epochs):
        for i in range(n_test):
            # Get the current representation of the environment.
            state = line.get_feature()[0]

            # Check if it is currently a peak hour and set the threshold accordingly.
            # Take the action of deploying a bus if both conditions are satisfied.
            if 300 <= line.time <= 400 or 700 <= line.time <= 800:
                if np.sum(state[1:-1]) > passenger1 and state[-1] >= loc/(line.D+1):
                    rewards[i] = line.take_action(1)
                else:
                    rewards[i] = line.take_action(0)
            else:
                if np.sum(state[1:-1]) > passenger2 and state[-1] >= loc/(line.D+1):
                    rewards[i] = line.take_action(1)
                else:
                    rewards[i] = line.take_action(0)

        avg[ep] = np.mean(rewards)
                
    return avg

def fixed(line, loc, n_epochs, n_test=1000):
    """To get the average rewards of a handcrafted policy. The policy is to deploy
    a bus when the nearest bus from the terminal is further than a certain distance.
    The number of passengers at each station does not effect the decision of
    whether to deploy a bus.

    Args:
       line (BusLine): Bus route that the policy will be implemented on.
       loc (int): The distance from the terminal that the nearest bus should reach
                  before another bus could be deployed.
       n_epochs (int): Number of epochs to run.
       n_test (int, optional): Number of iterations in each epochs. Default is 1000.
       
    Returns:
       An array with elements being the average reward per unit time in each epoch.
    
    """
    avg = np.empty(n_epochs)    # Stores the average rewards of each epochs.
    rewards = np.empty(n_test)  # Stores the rewards at each time step in one epoch.
    for ep in range(n_epochs):
        for i in range(n_test):
            # Get the current representation of the environment.
            state = line.get_feature()[0]
                
            # Take the action of deploying a bus reaches the location specified.  
            if state[-1] >= loc/(line.D+1):
                rewards[i] = line.take_action(1)
            else:
                rewards[i] = line.take_action(0)        
        
        avg[ep] = np.mean(rewards)

    return avg

###################################################################################
#                       Reinforcement Learning Methods                            #
###################################################################################

def reinforce(line, estimator_policy, estimator_value, n_epochs, n_iters, \
              batch_size, display_step, n_test=1000):
    """REINFORCE with baseline is implemented to find an optimal policy. 
    At certain epochs, the performance of the current policy will be tested. 

    Args:
       line (BusLine): Bus route that the algorithm will be implemented on.
       estimator_policy (PolicyEstimator): Policy function approximator to be
                                           optimized.
       estimator_value (StateValueEstimator): State-value function approximator
                                              which will be used as a baseline.
       n_epochs (int): Number of epochs to run.
       n_iters (int): Number of iterations in each epochs during training.
       batch_size (int): Number of samples to used during each training phase.
       display_step (int): Number of epochs to run in between each testing phase.
       n_test (int, optional): Number of iterations during testing. Default is 1000.
       
    Returns:
       Array containing the average reward per unit time in each testing phase.
    
    """
    avg = []                # Stores the average rewards of each testing phase.
    test = np.empty(n_test) # Stores the rewards at each time step in testing.
    
    # Initialize variables to store information on transition during training.
    states = np.empty((n_iters, line.N+2))
    actions = np.empty(n_iters)
    rewards = np.empty(n_iters)
    
    for epoch in range(n_epochs):
        total = 0
        
        for i in range(n_iters):
            # Choose action based on the policy function and take the action.
            cur_state = line.get_feature()
            action_probs = estimator_policy.predict(cur_state)[0]
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            R = line.take_action(action)

            # Keep track of the transition.
            states[i] = cur_state[0]
            rewards[i] = R
            actions[i] = action

            # Add reward to total after half of the total iterations (steady state) 
            if i >= np.floor(n_iters/2):
                total += R

        # Average reward of current policy.
        total /= np.ceil(n_iters/2)  
        
        # Returns is the total differences between rewards and average reward.
        returns = rewards - total
        returns = np.expand_dims(np.cumsum(returns[::-1])[::-1] , axis=1)
        
        # Sample the transitions.
        bstates, breturns, bactions = batch_sampling3(states, returns, actions, batch_size)
        
        # Run optimization on value estimator
        estimator_value.update(bstates, breturns)
        # Calculate the baseline of these states and get the difference with the returns
        baseline = estimator_value.predict(bstates)
        delta = breturns - baseline
        # Run optimization on policy estimator.
        estimator_policy.update(bstates, delta, bactions)
            
        # Test the current policy and get the average reward per time step. 
        if (epoch+1) % display_step == 0:
            for j in range(n_test):
                # Get the current state and choose action based on policy function.
                state = line.get_feature()
                action_probs = estimator_policy.predict(state)[0]
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                test[j] = line.take_action(action)
                
            avg.append(np.mean(test))    
            print("Epoch " + str(epoch+1) + ", Average reward = " + "{:.3f}".format(avg[-1]))

    return avg

