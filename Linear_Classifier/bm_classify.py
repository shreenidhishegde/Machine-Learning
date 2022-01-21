import numpy as np

#######################################################
# DO NOT MODIFY ANY CODE OTHER THAN THOSE TODO BLOCKS #
#######################################################

def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data (either 0 or 1)
    - loss: loss type, either perceptron or logistic
	- w0: initial weight vector (a numpy array)
	- b0: initial bias term (a scalar)
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the final trained weight vector
    - b: scalar, the final trained bias term

    Find the optimal parameters w and b for inputs X and y.
    Use the *average* of the gradients for all training examples
    multiplied by the step_size to update parameters.	
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2

    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0
        
    new_X = np.insert(X, 0, 1, axis=1)
    new_weight = np.insert(w, 0, b, axis=0)
    y = np.where(y == 0, -1, 1)

    if loss == "perceptron":
        ################################################
        # TODO 1 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize perceptron loss (use -1 as the   #
		# derivative of the perceptron loss at 0)      # 
        ################################################
        for _ in range(max_iterations+1):
            y_pred = binary_predict(new_X, new_weight, 1111) 
            #changing the label to -1 if it is 0
            y_pred = np.where(y_pred == 0, -1, 1) 
            #update only if the model misclassifies
            indicator_y = np.where((y * y_pred) <= 0, 1, 0) 
            indicator_y = indicator_y * y
            indicator_y_x = np.dot(indicator_y, new_X)
            eeta_by_N = step_size / N
            sub_gradient = eeta_by_N * (indicator_y_x)
            new_weight = np.add(new_weight, sub_gradient)

        
        

    elif loss == "logistic":
        ################################################
        # TODO 2 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize logistic loss                    # 
        ################################################
        for _ in range(max_iterations+1):
            indicator_y_x = y * np.dot(new_X, new_weight)
            sigma_of_indicator_y_x = sigmoid(-indicator_y_x)
            sigma_of_indicator_y_x_wt_y = sigma_of_indicator_y_x * y
            sigma_of_indicator_y_x_wt_y_wt_x = np.dot(sigma_of_indicator_y_x_wt_y, new_X)
            eeta_by_N = step_size / N
            new_weight = new_weight + (eeta_by_N) * sigma_of_indicator_y_x_wt_y_wt_x

        

    else:
        raise "Undefined loss function."
        
    b = new_weight[0]
    w = np.delete(new_weight, 0) #removing bias term

    assert w.shape == (D,)
    return w, b


def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after applying the sigmoid function 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : fill in the sigmoid function    #
    ############################################
    value = 1 / (1 + np.exp(-z))
    return value


def binary_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    
    Returns:
    - preds: N-dimensional vector of binary predictions (either 0 or 1)
    """
    N, D = X.shape
        
    #############################################################
    # TODO 4 : predict DETERMINISTICALLY (i.e. do not randomize)#
    #############################################################
    if b!=1111:
        X = np.insert(X, 0, 1, axis=1)
        w = np.insert(w, 0, b, axis=0)
    z = np.dot(X, w)
    preds = np.sign(z)
    preds = np.where(preds == -1, 0, 1)

    assert preds.shape == (N,) 
    return preds



def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data (0, 1, ..., C-1)
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform (stochastic) gradient descent

    Returns:
    - w: C-by-D weight matrix, where C is the number of classes and D 
    is the dimensionality of features.
    - b: a bias vector of length C, where C is the number of classes
	
    Implement multinomial logistic regression for multiclass 
    classification. Again for GD use the *average* of the gradients for all training 
    examples multiplied by the step_size to update parameters.
	
    You may find it useful to use a special (one-hot) representation of the labels, 
    where each label y_i is represented as a row of zeros with a single 1 in
    the column that corresponds to the class y_i. Also recall the tip on the 
    implementation of the softmax function to avoid numerical issues.
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0
        
    bias_one = np.ones((N,1))
    new_X = np.append(X, bias_one, axis=1)
    new_weight = np.append(w, np.array([b]).T, axis=1)

    np.random.seed(42) #DO NOT CHANGE THE RANDOM SEED IN YOUR FINAL SUBMISSION
    if gd_type == "sgd":

        for it in range(max_iterations):
            n = np.random.choice(N)
            weight_x = np.matmul(new_weight, new_X[n].T)
            numerator = np.subtract(weight_x, np.amax(weight_x, axis=0))
            numerator = [np.exp(i) for i in numerator]
            denominator = np.sum(numerator)
            prob = [i / denominator for i in numerator]
            prob[y[n]] -= 1
            sgd = np.dot(np.array([prob]).T, np.array([new_X[n]]))
            new_weight = new_weight - (step_size) * sgd
            ####################################################
            # TODO 5 : perform "max_iterations" steps of       #
            # stochastic gradient descent with step size       #
            # "step_size" to minimize logistic loss. We already#
            # pick the index of the random sample for you (n)  #
            ####################################################			
        
        

    elif gd_type == "gd":
        
        ####################################################
        # TODO 6 : perform "max_iterations" steps of       #
        # gradient descent with step size "step_size"      #
        # to minimize logistic loss.                       #
        ####################################################
        y = np.eye(C)[y]  # taking the yth row of identity matrix
        for i in range(max_iterations):
            x_weight = new_X.dot(new_weight.T)
            numerator = np.exp(x_weight - np.amax(x_weight))
            denominator = np.sum(numerator, axis=1)
            z = (numerator.T / denominator).T
            z = z - y
            gd = np.dot(z.T, new_X)
            new_weight = new_weight - (step_size / N) * gd
        
    else:
        #raise "Undefined algorithm."
        pass
    
    
    b = new_weight[:, -1]
    w = np.delete(new_weight, -1, 1)
    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained model, C-by-D 
    - b: bias terms of the trained model, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Predictions should be from {0, 1, ..., C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    #############################################################
    # TODO 7 : predict DETERMINISTICALLY (i.e. do not randomize)#
    #############################################################

    updated_w = np.insert(w, 0, b, axis=1)
    updated_X = np.insert(X, 0, 1, axis=1)
    w_transpose_X = np.dot(updated_X, updated_w.transpose())
    preds = np.argmax(w_transpose_X, axis=1)
    assert preds.shape == (N,)
    return preds




        