import tensorflow as tf

class ActionValueEstimator():
    """Action-value Function approximator. Given a representation of the environment,
    returns estimated values over the two actions of the state.

    Args:
       learning_rate (float): Learning rate used by optimizer.
       num_input (int): Size of the state array, i.e. the input.
       scope (str): Variable scope name of the operations or variable name
                    in the network. Default is value_estimator.

    """
    def __init__(self, learning_rate, num_input, scope="value_estimator"):
        with tf.variable_scope(scope):
            # Inputs are the state of the environment.
            self.state = tf.placeholder(shape=[None, num_input], dtype=tf.float32)
            # Action that was selected.
            self.action = tf.placeholder(shape=[None], dtype=tf.int32)
            # Targeted value of the selected action of the state.
            self.target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            
            # Store layers weight & bias.
            W1 = tf.Variable(tf.random_uniform([num_input,2*num_input],0,0.01))
            b1 = tf.Variable(tf.zeros([1,2*num_input]))
            W2 = tf.Variable(tf.random_uniform([2*num_input,2],0,0.01))
            b2 = tf.Variable(tf.zeros([1,2]))
            
            # Build a 2-layer network.
            layer1 = tf.nn.sigmoid(tf.matmul(self.state, W1) + b1)
            self.estimate = tf.matmul(layer1, W2) + b2

            # Get the predictions for the chosen actions only.
            batch_size = tf.shape(self.state)[0]
            indices = tf.stack([tf.range(0, batch_size, 1, dtype=tf.int32), self.action], axis=1)
            self.picked_action = tf.expand_dims(tf.gather_nd(self.estimate, indices),1)
            
            # Define loss and optimizer
            self.loss = tf.reduce_mean(tf.square(self.picked_action-self.target))
            self.trainer = tf.train.AdamOptimizer(learning_rate)
            self.update_model = self.trainer.minimize(self.loss)
                                    
    def predict(self, state, sess=None):
        """Get the estimated values of the two actions of the given state. 

        Args:
           state (array): Representations of the environement.
           sess (Session): Environment in which the operations are executed
                           and objects are evaluated.

        Returns:
           Array consisting estimated values of the two actions.

        """
        sess = sess or tf.get_default_session()
        return sess.run(self.estimate, feed_dict={self.state:state})

    def update(self, state, target, action, sess=None):
        """Run the optimizer and update the network given the targeted values. 

        Args:
           state (array): Representations of the environement.
           target (array): Targeted values the state-action pair.
           action (array): Actions that was selected in each state.
           sess (Session): Environment in which the operations are executed
                           and objects are evaluated.

        """
        sess = sess or tf.get_default_session()
        sess.run(self.update_model, feed_dict={self.state:state, self.target:target, self.action:action})                                  

class StateValueEstimator():
    """State-value Function approximator. Given a representation of the environment,
    returns a estimated value of the state.

    Args:
       learning_rate (float): Learning rate used by optimizer.
       num_input (int): Size of the state array, i.e. the input.
       scope (str): Variable scope name of the operations or variable name
                    in the network. Default is value_estimator.

    """
    def __init__(self, learning_rate, num_input, scope="value_estimator"):
        with tf.variable_scope(scope):
            # Inputs are the state of the environment.
            self.state = tf.placeholder(shape=[None, num_input], dtype=tf.float32)
            # Targeted value of the state.
            self.target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

            # Store layers weight & bias.
            W1 = tf.Variable(tf.random_uniform([num_input,2*num_input],0,0.01))
            b1 = tf.Variable(tf.zeros([1,2*num_input]))
            W2 = tf.Variable(tf.random_uniform([2*num_input,1],0,0.01))
            b2 = tf.Variable(tf.zeros([1,1]))

            # Build a 2-layer network.
            layer1 = tf.nn.sigmoid(tf.matmul(self.state, W1) + b1)
            self.estimate = tf.matmul(layer1, W2) + b2

            # Define loss and optimizer.
            self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.estimate-self.target),1))
            self.trainer = tf.train.AdamOptimizer(learning_rate)
            self.update_model = self.trainer.minimize(self.loss)
                                    
    def predict(self, state, sess=None):
        """Get the estimated values of the given state. 

        Args:
           state (array): Representations of the environement.
           sess (Session): Environment in which the operations are executed
                           and objects are evaluated.

        Returns:
           Estimated value of the given state.

        """
        sess = sess or tf.get_default_session()
        return sess.run(self.estimate, feed_dict={self.state:state})
    
    def update(self, state, target, sess=None):
        """Run the optimizer and update the network given the targeted values. 

        Args:
           state (array): Representations of the environement.
           target (array): Targeted values the states.
           sess (Session): Environment in which the operations are executed
                           and objects are evaluated.

        """
        sess = sess or tf.get_default_session()
        sess.run(self.update_model, feed_dict={self.state:state, self.target:target})

class PolicyEstimator():
    """Policy Function approximator. Given a representation of the environment,
    returns probabilities of taking the two actions of the state.

    Args:
       learning_rate (float): Learning rate used by optimizer.
       num_input (int): Size of the state array, i.e. the input.
       scope (str): Variable scope name of the operations or variable name
                    in the network. Default is policy_estimator.

    """
    def __init__(self, learning_rate, num_input, scope="policy_estimator"):
        with tf.variable_scope(scope):
            # Inputs are the state of the environment.
            self.state = tf.placeholder(shape=[None, num_input], dtype=tf.float32)
            # Action that was selected.
            self.action = tf.placeholder(shape=[None], dtype=tf.int32)
            # Targeted value of the selected action of the state.
            self.target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

            # Store layers weight & bias
            W1 = tf.Variable(tf.random_uniform([num_input,2*num_input],0,0.01))
            b1 = tf.Variable(tf.zeros([1,2*num_input]))
            W2 = tf.Variable(tf.random_uniform([2*num_input,2],0,0.01))
            b2 = tf.Variable(tf.zeros([1,2]))

            # Build a 2-layer network.
            layer1 = tf.nn.sigmoid(tf.matmul(self.state, W1) + b1)
            output = tf.matmul(layer1, W2) + b2
            self.action_probs = tf.nn.softmax(output)
            
            # Get the probabilites for the chosen actions only.
            batch_size = tf.shape(self.state)[0]
            indices = tf.stack([tf.range(0, batch_size, 1, dtype=tf.int32), self.action], axis=1)
            self.picked_action_prob = tf.expand_dims(tf.gather_nd(self.action_probs, indices),1)
            
            # Define loss and optimizer.
            self.loss = -tf.reduce_sum(tf.log(self.picked_action_prob) * self.target)
            self.trainer = tf.train.AdamOptimizer(learning_rate)
            self.update_model = self.trainer.minimize(self.loss)
                                    
    def predict(self, state, sess=None):
        """Get the probabilities of taking the two actions of the given state. 

        Args:
           state (array): Representations of the environement.
           sess (Session): Environment in which the operations are executed
                           and objects are evaluated.

        Returns:
           Array consisting probabilities of the two actions.

        """
        sess = sess or tf.get_default_session()
        return sess.run(self.action_probs, feed_dict={self.state:state})

    def update(self, state, target, action, sess=None):
        """Run the optimizer and update the network given the targeted values. 

        Args:
           state (array): Representations of the environement.
           target (array): Targeted values the selected actions.
           action (array): Actions that was selected in each state.
           sess (Session): Environment in which the operations are executed
                           and objects are evaluated.

        """
        sess = sess or tf.get_default_session()
        sess.run(self.update_model, feed_dict={self.state:state, self.target:target, self.action:action})                                  
        
