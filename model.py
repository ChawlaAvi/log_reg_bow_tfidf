"""
model.py

This file implements the Logistic Regression model for classification.
"""

import numpy as np


def sigmoid(x):
    """
    Sigmoid function implemented using numpy.

    @x: A numpy array of size n.

    returns: a numpy array with each element has been replaced by its sigmoid.
    """

   

    ''' WRITE YOUR CODE HERE (use just one line of vectorized code) '''
    
    output = 1/(1+np.exp(-x))

    ''' END YOUR CODE HERE '''

    return output


class LogisticRegression(object):
    """
    A binary Logistic Regression model.
    """


    def __init__(self, add_bias = False):
        """
        Initialise the model. Here, we only initialise the weights as 'None',
        as the input size of the model will become apparent only when some
        training samples will be given to us.

        @add_bias: Whether to add bias.
        """

        # Initialise model parameters placeholders. Don't need another placeholder
        # for bias, as it can be incorporated in the weights matrix by adding
        # another feature in the input vector whose value is always 1.
        self.weights = None

        # Store the add_bias property too.
        self.add_bias = add_bias


    def loss(self, X_batch, Y_batch):
        """
        Loss function for Logistic Regression.

        Calculates the loss and gradients of weights with respect to loss
        for a batch of the logistic regression model.

        @X_batch: The input for batch, a numpy array of size NxD, where
                  N is the number of samples and D is the feature vector.
                  Assume that if 'self.add_bias' was set to True, D will
                  include a column for bias, with values always 1.
        @Y_batch: The correct labels for the batch, a numpy array of size N.

        returns: A tuple (loss, gradient).
        """
        
        

        loss, grad = None, None

        ''' START YOUR CODE HERE '''

        # Make prediction - use the sigmoid function and self.weights (use hint.pdf in case you forget the equations involved)
        y_pred = (sigmoid(np.matmul(X_batch,(self.weights)))).reshape((-1,1))
        
        Y_batch = Y_batch.reshape((-1,1))
        loss = -1*np.sum( (Y_batch*np.log(y_pred))/X_batch.shape[0] )
        loss += -1*np.sum( ( (1-Y_batch)*np.log(1-y_pred) )/X_batch.shape[0] )
        
        grad = (np.matmul( np.transpose(X_batch),(y_pred-Y_batch) )/(X_batch.shape[0])).reshape((-1,1))
        '''print("grad->",grad.shape)
        print("y_pred->",y_pred.shape)
        print("y_batch->",Y_batch.shape)
        print("X_batch->",X_batch.shape)
        print("askd",(y_pred-Y_batch))
        print("dskfjkfw",X_batch)'''
        
        
        
        # calculate loss (stored in variable name loss) and gradients (stored in variable name grad)

        ''' END YOUR CODE HERE '''

        return loss, grad


    def predict(self, X_batch):
        """
        Predict the correct labels for the examples in X_batch.

        Remember: We don't need sigmoid for this. If x > 0 then sigmoid(x) > 0.5, and
        if x <= 0 then sigmoid(x) <= 0.5.

        @X_batch: The input to predict for. A numpy array of size NxD,
                  where N is the number of examples, and D is the size of input vector.
                  Since this will probably be called from outside this class,
                  we need to chech the @self.add_bias variable and adequately
                  pass the X_batch.

        returns: A vector of size N which contains 0 or 1 indicating the class label.
        """

        predict_func = np.vectorize(lambda x: 0 if x < 0 else 1)

        if self.add_bias:
            score = np.dot(np.hstack((X_batch, np.ones((X_batch.shape[0], 1)))), self.weights)
        else:
            score = np.dot(X_batch, self.weights)

        predictions = predict_func(score)

        return predictions


    def score(self, X_test, Y_test):
        """
        Score the performance of the model with the given test labels.

        @X_test: Test input numpy array of size NxD.
        @Y_test: Test correct labels numpy array of size N.

        returns: The accuracy of the model. A single float value between 0 and 1.
        """
        Y_test=Y_test.reshape(-1,1)
        Y_pred = self.predict(X_test)
        accuracy = 1 - np.float(np.count_nonzero(Y_pred - Y_test)) / X_test.shape[0]

        return accuracy


    def train(self,
              X_train,
              Y_train,
              lr = 1e-3,
              reg_const = 1.0,
              num_iters = 10000,
              num_print = 100):
        """            self.weights = self.weights - lr*grad

        Learn the parameters from the given supervised samples with the given hyper parameters, using
        stochastic gradient descent.

        @X_train: Examples to learn from. It is a vector of size NxD,
                  with N being the number of samples, and D being the
                  number of features. Here, we need to check the @self.add_bias
                  flag and adequately pad the input vector with ones.
        @Y_train: Correct labels for X_train. Vector of size N.
        @lr: Learning rate for gradient descent.
        @reg_const: The regularization constant which is used to control the regularization.
        @num_iterations: Number of iterations over the training set.
        @num_print: Number of print statements during iterations.
        """

        ''' START YOUR CODE HERE '''
        

        # Initialise weights with normal distribution in the variable self.weights,
        # while taking in consideration the bias (so two cases according to the boolean value of add_bias).

        size_row_x = X_train.shape[1]
        size_col_x = X_train.shape[0]
        if self.add_bias:
            self.weights = np.random.randn(size_row_x+1,1) # Fill a correct value here
            p = np.copy(self.weights[1:])
            p = (np.insert(p,0,0)).reshape((-1,1))
            
        else:
            self.weights = np.random.randn(size_row_x,1) # Fill a correct value here
            p = np.copy(self.weights)
        Y_train=Y_train.reshape(-1,1)
        
        # print("p",p)
        # print("initial->",self.weights)        
        self.weights = (self.weights).reshape((-1,1))
        X_train_adj = None

        # Create a new input array by adding bias column with all values one in X_train_adj array
        # while taking in consideration the bias (so two cases according to the boolean value of add_bias).
        if self.add_bias:
            a=np.ones((size_col_x,1))
            X_train_adj = np.c_[a,X_train]  # Fill a correct value here
        else:
            X_train_adj = X_train
            
        
        

        # Perform iterations, use output of loss function above to get unregularized loss and gradient
        for i in range(num_iters):
            loss, grad = self.loss(X_train_adj, Y_train)

            
            # Add Regularisation to the loss and grad (remember to use the reg_const being passed to the function)
            # See hint.pdf for the equations
            
            loss += (reg_const/p.shape[0])*((np.sum(p*p))/2)
            grad += (reg_const/(p.shape[0]))*p
            print("grad->",grad)
            
            
            self.weights = self.weights - lr*grad
            p = np.copy(self.weights)
                

            # Update weights (remember to use the learning rate being passed to the function)
            # See hint.pdf for equations

            if i % (num_iters // num_print) == 0:
                print("Iteration %d of %d. Loss: %f" % (i, num_iters, loss))

        ''' END YOUR CODE HERE '''

        
