import math, random, statistics as stats, numpy as np, matplotlib.pyplot as plt

"""This method calculates the covariance value between any 2 parameter values using the Matern 5/2
Kernel Covariance Function, necessary for constructing covariance matrices."""
def calculate_kernel_value(l, x1, x2):
    r = abs((x1 - x2) / l)
    kernel_value = (1 + (pow(5, 0.5) * r) + ((5 / 3) * pow(r, 2))) * pow(math.e, (-1 * pow(5, 0.5)) * r)
    return kernel_value

"""This method generates the covariance function for the multivariate Gaussian Distribution
between a given training data set and a given parameter value, necessary for calculating the
predictive mean and variance for a conditional posterior predictive distribution of data for
such given hyperparameter value."""
def calculate_joint_gaussian_distribution_covariance_matrix(training_data_points, x):
    training_data_covariance_matrix = []
    for i in range(len(training_data_points)):
        training_data_covariance_matrix.append([])
        for j in range(len(training_data_points)):
            xi = training_data_points[i][0]
            xj = training_data_points[j][0]
            training_data_covariance_matrix[i].append(calculate_kernel_value(1, xi, xj))
    cross_covariance_matrix = []
    for i in range(len(training_data_points)):
        xi = training_data_points[i][0]
        cross_covariance_matrix.append([calculate_kernel_value(1, xi, x)])
    transposed_cross_covariance_matrix = [[]]
    for i in range(len(training_data_points)):
        xi = training_data_points[i][0]
        transposed_cross_covariance_matrix[0].append(calculate_kernel_value(1, x, xi))
    predictive_data_covariance_matrix = [[calculate_kernel_value(1, x, x)]]
    covariance_matrix = [[training_data_covariance_matrix, cross_covariance_matrix], [transposed_cross_covariance_matrix, predictive_data_covariance_matrix]]
    return covariance_matrix

"""This method calculates the predictive mean for a conditional posterior predictive distribution
of data modeling the true objective cost value for a given hyperparameter value based on training data,
using the formula for mean of a conditional distribution of a Multivariate Gaussian Distribution.
Utilizes Cholesky Decomposition."""
def calculate_predictive_mean(covariance_matrix, training_data_points):
    training_data_covariance_matrix = covariance_matrix[0][0]
    training_data_value_vector = []
    for i in range(len(training_data_points)):
        training_data_value_vector.append(training_data_points[i][1])
    transposed_cross_covariance_matrix = covariance_matrix[1][0]
    L = np.linalg.cholesky(np.array(training_data_covariance_matrix))
    m = np.linalg.solve(L, np.array(training_data_value_vector))
    alpha = np.linalg.solve(L.T, m)
    mean = np.array(transposed_cross_covariance_matrix) @ alpha
    return mean[0]

"""This method calculates the predictive variance for a conditional posterior predictive distribution
of data modeling the true objective cost value for a given hyperparameter value based on training data,
using the formula for mean of a conditional distribution of a Multivariate Gaussian Distribution.
Utilizes Cholesky Decomposition."""
def calculate_predictive_variance(covariance_matrix, noise):
    training_data_covariance_matrix = covariance_matrix[0][0]
    cross_covariance_matrix = covariance_matrix[0][1]
    transposed_cross_covariance_matrix = covariance_matrix[1][0]
    predictive_data_covariance_matrix = covariance_matrix[1][1]
    L = np.linalg.cholesky(np.array(training_data_covariance_matrix))
    n = np.linalg.solve(L, np.array(cross_covariance_matrix))
    beta = np.linalg.solve(L.T, n)
    predictive_variance = np.add(-1.0 * (np.array(transposed_cross_covariance_matrix) @ beta), predictive_data_covariance_matrix)
    return predictive_variance[0][0] + noise

"""This function formulates the conditional posterior predictive distribution of data modeling the
true objective cost value for a given hyperparameter value, providing the predictive mean and variance."""
def calculate_gaussian_process_parameter_posterior_predictive_distribution(training_data_points, noise, x):
    covariance_matrix = calculate_joint_gaussian_distribution_covariance_matrix(training_data_points, x)
    predictive_mean = calculate_predictive_mean(covariance_matrix, training_data_points)
    predictive_variance = calculate_predictive_variance(covariance_matrix, noise)
    return predictive_mean, predictive_variance

"""This function formulates a Gaussian Process, representing a conditional posterior distribution of
functions modeling the true objective cost function with respect to a given hyperparameter, based on the
predictive means and variances of a range of hyperparameter values along an axis. The function formulates
functions for the predictive mean and bounds of the distribution within a 95% confidence interval."""
def calculate_gaussian_process_interval(training_data_points, noise, x_values, show):
    predictive_mean_values = []
    predictive_std_dev_values = []
    predictive_lower_bound_values = []
    predictive_upper_bound_values = []
    for i in range(len(x_values)):
        predictive_mean, predictive_variance = calculate_gaussian_process_parameter_posterior_predictive_distribution(training_data_points, noise, x_values[i])
        predictive_mean_values.append(predictive_mean)
        predictive_std_dev = math.sqrt(predictive_variance)
        predictive_std_dev_values.append(predictive_std_dev)
        predictive_lower_bound_values.append(predictive_mean + (stats.NormalDist().inv_cdf(0.025) * predictive_std_dev))
        predictive_upper_bound_values.append(predictive_mean + (stats.NormalDist().inv_cdf(0.975) * predictive_std_dev))
    if show:
        plt.plot(x_values, predictive_lower_bound_values, label = "Lower 95% Confidence Bound")
        plt.plot(x_values, predictive_mean_values, label = "Mean")
        plt.plot(x_values, predictive_upper_bound_values, label = "Upper 95% Confidence Bound")
        plt.xlabel("Hyperparameter Test Value")
        plt.ylabel("Cost Value")
        plt.title("Gaussian Process Distribution 95% Confidence Interval")
        plt.legend()
        plt.show()
    return predictive_mean_values, predictive_std_dev_values

def get_minimum_point(points):
    minimum_point = points[0]
    for i in range(len(points)):
        if points[i][1] < minimum_point[1]:
            minimum_point = points[i]
    return minimum_point

"""This function runs an Expected Improvement Acquisition function, used to determine the next
optimal hyperparameter value to test in a Gaussian Process modeling the true objective cost function
for a given hyperparameter."""
def calculate_optimal_acquisition_value(training_data_points, noise, slack, x_values, show):
    expected_improvement_values = []
    minimum_value = get_minimum_point(training_data_points)[1]
    if show:
        predictive_mean_values, predictive_std_dev_values = calculate_gaussian_process_interval(training_data_points, noise, x_values, True)
    else:
        predictive_mean_values, predictive_std_dev_values = calculate_gaussian_process_interval(training_data_points, noise, x_values, False)
    for i in range(len(x_values)):
        predictive_mean = predictive_mean_values[i]
        predictive_std_dev = predictive_std_dev_values[i]
        z = (slack + minimum_value - predictive_mean) / predictive_std_dev
        expected_improvement = ((slack + minimum_value - predictive_mean) * stats.NormalDist().cdf(z)) + (predictive_std_dev * stats.NormalDist().pdf(z))
        expected_improvement_values.append(expected_improvement)
    optimal_value = 0
    optimal_parameter_value = 0
    for i in range(len(expected_improvement_values)):
        if expected_improvement_values[i] > optimal_value:
            optimal_value = expected_improvement_values[i]
            optimal_parameter_value = x_values[i]
    if show:
        plt.plot(x_values, expected_improvement_values, label = "Expected Improvement")
        plt.xlabel("Hyperparameter Test Value")
        plt.ylabel("Cost Value")
        plt.title("Expected Improvement Acquisition Function")
        plt.legend()
        plt.show()
    return optimal_parameter_value

"""This function calculates the error between 2 values, necessary to determine if optimal acquisition
hyperparameter values are close enough to be considered similar."""
def calculate_error(x1, x2):
    if x1 != 0:
        return abs(x1 - x2) / x1
    else:
        return x2

"""Runs the acquisition function for each numerical hyperparameter in the neural network model to
determine the optimal value for each hyperparameter, based on the model's training data. The function
keeps evaluating optimal acquisition hyperparameter values for a given parameter until the the last 3 optimal
acquisition hyperparameter values are close enough - with an error of less than 1% - to be considered similar
(or if the acquisition function returns the same values)."""
def optimize_neural_network_hyperparameters(neural_network, feed_data_reference, hyperparameters, cost_attribute_reference, slack, show):
    optimal_hyperparameter_values = {}
    for hyperparameter in hyperparameters:
        success = False
        while not success:
            try:
                training_data_points = []
                if hyperparameters[hyperparameter][2] != None and hyperparameters[hyperparameter][3] != None:
                    test_space = np.linspace(hyperparameters[hyperparameter][2], hyperparameters[hyperparameter][3], hyperparameters[hyperparameter][4]).tolist()
                elif hyperparameters[hyperparameter][2] != None:
                    test_space = np.linspace(hyperparameters[hyperparameter][2], hyperparameters[hyperparameter][0] * hyperparameters[hyperparameter][4], hyperparameters[hyperparameter][4]).tolist()
                elif hyperparameters[hyperparameter][3] != None:
                    test_space = np.linspace(1 / hyperparameters[hyperparameter][4], hyperparameters[hyperparameter][3], hyperparameters[hyperparameter][4]).tolist()
                else:
                    test_space = np.linspace(1 / hyperparameters[hyperparameter][4], hyperparameters[hyperparameter][0] * hyperparameters[hyperparameter][4], hyperparameters[hyperparameter][4]).tolist()
                if hyperparameters[hyperparameter][1]:
                    test_space = list(dict.fromkeys([round(test_value) for test_value in test_space]))
                hyperparameter_reference = {
                    hyperparameter: hyperparameters[hyperparameter][0]
                }
                model = neural_network(**feed_data_reference, **hyperparameter_reference)
                objective_cost = eval("model." + cost_attribute_reference["cost"])
                training_data_points.append((hyperparameters[hyperparameter][0], objective_cost))
                while not(len(training_data_points) >= 3 and calculate_error(training_data_points[-3][0], training_data_points[-2][0]) <= 0.01 and calculate_error(training_data_points[-2][0], training_data_points[-1][0]) <= 0.01 and calculate_error(training_data_points[-3][0], training_data_points[-1][0]) <= 0.01):
                    noise = stats.variance(model.get_average_validation_cost_values_over_epochs())
                    if show:
                        optimal_acquisition_test_value = calculate_optimal_acquisition_value(training_data_points, noise, slack, test_space, True)
                    else:
                        optimal_acquisition_test_value = calculate_optimal_acquisition_value(training_data_points, noise, slack, test_space, False)
                    if any(point[0] == optimal_acquisition_test_value for point in training_data_points):
                        training_data_points.append((optimal_acquisition_test_value, None))
                        break
                    hyperparameter_reference = {
                        hyperparameter: optimal_acquisition_test_value
                    }
                    model = neural_network(**feed_data_reference, **hyperparameter_reference)
                    objective_cost = eval("model." + cost_attribute_reference["cost"])
                    training_data_points.append((optimal_acquisition_test_value, objective_cost))
                    iterations = range(1, len(training_data_points) + 1)
                    objective_cost_values = [point[1] for point in training_data_points]
                    if show:
                        plt.plot(iterations, objective_cost_values, label = "Objective Cost")
                        plt.xlabel("Iteration")
                        plt.ylabel("Cost Value")
                        plt.title("Convergence")
                        plt.legend()
                        plt.show()
                optimal_hyperparameter_values[hyperparameter] = training_data_points[-1][0]
                success = True
            except Exception:
                continue
    return optimal_hyperparameter_values
