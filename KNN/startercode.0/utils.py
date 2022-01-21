
import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)

    #true_positive = sum([ 1 if (predicted_labels == 1 and real_labels == 1) else 0 for i in range(len(real_labels))])
    #false_positive = sum([ 1 if (predicted_labels == 1 and real_labels == 0) else 0 for i in range(len(real_labels))])
    #false_negative = sum([ 1 if (predicted_labels == 0 and real_labels == 1) else 0 for i in range(len(real_labels))])
    
    real_labels = np.array(real_labels).reshape((len(real_labels),1))
    predicted_labels = np.array(predicted_labels).reshape((len(predicted_labels),1))
    
    #precision is  the number of true positives results divided by the number of all positive results( true positive + false positive) returned    by the classifier
    precision_score = np.sum((real_labels ==1) & (predicted_labels==1)) / np.sum(predicted_labels)
    recall_score = np.sum((real_labels ==1) & (predicted_labels==1)) / np.sum(real_labels)
    
    f1_score = 2 * (precision_score * recall_score)/(precision_score + recall_score)
    return f1_score
    raise NotImplementedError


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """

        #m_distance = sum([np.absolute(a - b) ** 3 for a, b in zip(point1, point2)])
        #return float(np.cbrt(m_distance))
        #raise NotImplementedError
        m_distance = 0
        for a, b in zip(point1, point2):
            m_distance = m_distance + np.absolute(a - b) ** 3
        return float(np.cbrt(m_distance))

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        e_distance = np.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))
        return float(e_distance)

        raise NotImplementedError

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        norm_point1 = 0
        norm_point2 = 0
        product_sum = 0
        for x, y in zip(point1, point2):
            norm_point1 = norm_point1 + x ** 2
            norm_point2 = norm_point2 + y ** 2
            product_sum = product_sum + x * y
        if norm_point1 == 0 or norm_point2 == 0:
            return 1
        return float(1 - float(product_sum) / float(np.sqrt(norm_point1) * np.sqrt(norm_point2)))

        raise NotImplementedError

class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values of k (among 1, 3, 5, ... , 29), and find the best model with the highest f1-score on the given validation set.
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] training labels to train your KNN model
        :param x_val:  List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), and model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check the distance function:  euclidean > Minkowski > cosine_dist 
		(this will also be the insertion order in "distance_funcs", to make things easier).
        For the same distance function, further break tie by prioritizing a smaller k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None
        best_f1_score = 0

        #print(f1_score([1,0,1,0],[1,1,1,1]))

        for distance_function in distance_funcs:
            #print(len(x_train),"herere")
            for k in range(1, min(31, len(x_train)+1), 2):
                # print(k)
                model = KNN(k, distance_funcs[distance_function])
                model.train(x_train, y_train)
                y_val_pred = model.predict(x_val)
                print(y_val_pred)
                final_f1_score = f1_score(y_val, y_val_pred)
                print(final_f1_score)

                if final_f1_score > best_f1_score:
                    self.best_k = k
                    self.best_distance_function = distance_function
                    self.best_model = model
                    best_f1_score = final_f1_score



    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is the same as "tuning_without_scaling", except that you also need to try two different scalers implemented in Part 1.3. More specifically, before passing the training and validation data to KNN model, apply the scalers in scaling_classes to both of them. 
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param scaling_classes: dictionary of scalers (key is the scaler name, value is the scaler class) you need to try to normalize your data
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), scaler (its name), and model (an instance of KNN), and assign them to self.best_k, self.best_distance_function, best_scaler, and self.best_model respectively.
        
        NOTE: When there is a tie, choose the model based on the following priorities:
        First check scaler, prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of scaling_classes). Then follow the same rule as in "tuning_without_scaling".
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None
        best_f1_score = 0

        for scalar in scaling_classes:
            scalar_obj = scaling_classes[scalar]()
            scaled_training_data = scalar_obj.__call__(x_train)
            scaled_validation_data = scalar_obj.__call__(x_val)
            for distance_function in distance_funcs:
            #print(len(x_train),"herere")
                for k in range(1, min(31, len(scaled_validation_data)+1), 2):
                    # print(k)
                    model = KNN(k, distance_funcs[distance_function])
                    model.train(scaled_training_data, y_train)
                    y_val_pred = model.predict(scaled_validation_data)
                    print(y_val_pred)
                    final_f1_score = f1_score(y_val, y_val_pred)
                    print(final_f1_score)

                    if final_f1_score > best_f1_score:
                        self.best_k = k
                        self.best_distance_function = distance_function
                        self.best_model = model
                        best_f1_score = final_f1_score
                        self.best_scaler = scalar

class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """

        input_length = len(features)
        features_length = len(features[0])
        normalised_std_features = [[0] * features_length for x in range(input_length)]
        for i in range(input_length):
            norm = 0
            for j in range(features_length):
                norm += (features[i][j] ** 2)
            norm = np.sqrt(norm)
            if norm == 0:
                normalised_std_features[i] = features[i]
                continue
            for j in range(features_length):
                normalised_std_features[i][j] = 0 if features[i][j] == 0 else features[i][j] / norm
        return normalised_std_features
        raise NotImplementedError


class MinMaxScaler:
    def __init__(self):
        pass

    # TODO: min-max normalize data
    def __call__(self, features):
        """
		For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
        For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
		This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
		The minimum value of this feature is thus min=-1, while the maximum value is max=2.
		So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
		leading to 1, 0, and 0.333333.
		If max happens to be same as min, set all new values to be zero for this feature.
		(For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """

        input_length = len(features)
        features_length = len(features[0])

        #initialise a list to store normalised features
        normalised_minmax_features = [[0]*features_length for i in range(input_length)]

        min_list = [float("inf")] * features_length
        max_list = [float("-inf")] * features_length

        for i in range(input_length):
            for j in range(features_length):
                val = features[i][j]
                min_list[j] = min(val,min_list[j] )
                max_list[j] = max(val,max_list[j] )


        for i in range(input_length):
            for j in range(features_length):
                max_min_diff = max_list[j] - min_list[j]
                normalised_minmax_features[i][j] = 0 if max_min_diff == 0 else ((features[i][j] - min_list[j]) / max_min_diff)
        return normalised_minmax_features
        raise NotImplementedError
