
import numpy as np
import pandas as pd

############################################################################
# DO NOT MODIFY CODES ABOVE 
# DO NOT CHANGE THE INPUT AND OUTPUT FORMAT
############################################################################

###### Part 1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean square error of a model parameter w on a test set X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test features
    - y: A numpy array of shape (num_samples, ) containing test labels
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # TODO 1: Fill in your code here   
  
    y_pred = X.dot(w)
    difference = y_pred - y
    mean_sq = np.mean(np.square(difference))
    #####################################################
    err = mean_sq
    return err

###### Part 1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing features
  - y: A numpy array of shape (num_samples, ) containing labels
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here                    #

  x_transpose = X.transpose()
  inner_product = x_transpose.dot(X)
  innersum_inverse = np.linalg.inv(inner_product)
  outer_product = x_transpose.dot(y)
  weight = innersum_inverse.dot(outer_product)
  # print(weight.shape)

  #####################################################		
  #w = None
  w = weight
  return w


###### Part 1.3 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing features
    - y: A numpy array of shape (num_samples, ) containing labels
    - lambd: a float number specifying the regularization parameter
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here                    #

    x_transpose = X.transpose()
    inner_product = x_transpose.dot(X)
    lambda_I = lambd * np.identity(np.size(inner_product, 1))
    outer_product = x_transpose.dot(y)
    inner_inverse = np.linalg.inv(np.add(inner_product, lambda_I))
    regularized_weight = inner_inverse.dot(outer_product)
    print(regularized_weight)

  #####################################################		
    w = regularized_weight
    return w

###### Part 1.4 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training features
    - ytrain: A numpy array of shape (num_training_samples, ) containing training labels
    - Xval: A numpy array of shape (num_val_samples, D) containing validation features
    - yval: A numpy array of shape (num_val_samples, ) containing validation labels
    Returns:
    - bestlambda: the best lambda you find among 2^{-14}, 2^{-13}, ..., 2^{-1}, 1.
    """
    ####################################################
    # TODO 5: Fill in your code here   
    #x = -14
    best_weight = float('inf')
    bestlambda_obtained = None
    for i in range(-14,1):
      lambda_value = 1/(np.power(2,abs(i)))
      pred_w = regularized_linear_regression(Xtrain,ytrain,lambda_value)
      val_w = mean_square_error(pred_w,Xval,yval)
      if val_w < best_weight:
        best_weight = val_w
        bestlambda_obtained = lambda_value
        
    #####################################################		
    bestlambda = bestlambda_obtained
    return bestlambda
    
   
###### Part 1.6 ######
def mapping_data(X, p):
    """
    Augment the data to [X, X^2, ..., X^p]
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training features
    - p: An integer that indicates the degree of the polynomial regression
    Returns:
    - X: The augmented dataset. You might find np.insert useful.
    """
    #####################################################
    # TODO 6: Fill in your code here                    #
    
    X_start = X
    for i in range(2,p+1):
      X = np.concatenate((X, np.power(X_start,i)), axis=1)

    #####################################################		
    
    return X


"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""

