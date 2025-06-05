import math, random, statistics as stats, numpy as np, matplotlib.pyplot as plt
from neural_network_class import NeuralNetworkModel

def main():
    hyperparameters = {
    "model_size": (5, None, None, None),
    "neuron_size_base": (2, None, None, None),
    "training_epochs": (75, None, None, None),
    "training_data_proportion": (0.75, 6, None, None),
    "delta": (1.0, 6, None, None),
    "learning_rate": (0.001, 6, None, None),
    "learning_rate_decay_rate": (0.0001, 6, None, None),
    "momentum_factor": (0.9, 6, None, None),
    "max_norm_benchmark": (90, 6, None, 100),
    "ld": (0.01, 6, None, None)
    }
    hyperparameters_2 = {
    "max_norm_benchmark": (90, 6, None, 100)
    }
    optimal_hyperparameter_values = optimize_neural_network_hyperparameters("processed-us-medical-insurance-costs-data.csv", hyperparameters_2)
    print(optimal_hyperparameter_values)
"""model = NeuralNetworkModel(training_data_file_name = "processed-us-medical-insurance-costs-data.csv")
print(model.average_validation_cost_values_over_epochs[model.minimum_cost_index])
model = NeuralNetworkModel(training_data_file_name = "processed-us-medical-insurance-costs-data.csv", training_epochs = 130)
print(model.average_validation_cost_values_over_epochs[model.minimum_cost_index])
model = NeuralNetworkModel(training_data_file_name = "processed-us-medical-insurance-costs-data.csv", training_epochs = 205)
print(model.average_validation_cost_values_over_epochs[model.minimum_cost_index])
model = NeuralNetworkModel(training_data_file_name = "processed-us-medical-insurance-costs-data.csv", training_epochs = 280)
print(model.average_validation_cost_values_over_epochs[model.minimum_cost_index])
model = NeuralNetworkModel(training_data_file_name = "processed-us-medical-insurance-costs-data.csv", training_epochs = 243)
print(model.average_validation_cost_values_over_epochs[model.minimum_cost_index])
model = NeuralNetworkModel(training_data_file_name = "processed-us-medical-insurance-costs-data.csv", training_epochs = 93)
print(model.average_validation_cost_values_over_epochs[model.minimum_cost_index])
model = NeuralNetworkModel(training_data_file_name = "processed-us-medical-insurance-costs-data.csv", training_epochs = 168)
print(model.average_validation_cost_values_over_epochs[model.minimum_cost_index])"""

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
    predictive_lower_bound_values = []
    predictive_mean_values = []
    predictive_upper_bound_values = []
    for i in range(len(x_values)):
        predictive_mean, predictive_variance = calculate_gaussian_process_parameter_posterior_predictive_distribution(training_data_points, noise, x_values[i])
        predictive_std_dev = math.sqrt(predictive_variance)
        predictive_lower_bound_values.append(predictive_mean + (stats.NormalDist(mu = 0, sigma = 1).inv_cdf(0.025) * predictive_std_dev))
        predictive_mean_values.append(predictive_mean)
        predictive_upper_bound_values.append(predictive_mean + (stats.NormalDist(mu = 0, sigma = 1).inv_cdf(0.975) * predictive_std_dev))
    if show:
        plt.plot(x_values, predictive_lower_bound_values)
        plt.plot(x_values, predictive_mean_values)
        plt.plot(x_values, predictive_upper_bound_values)
        plt.show()
    return predictive_lower_bound_values, predictive_mean_values, predictive_upper_bound_values

"""This function runs an Expected Improvement Acquisition function, used to determine the next
optimal hyperparameter value to test in a Gaussian Process modeling the true objective cost function
for a given hyperparameter."""
def calculate_optimal_acquisition_value(training_data_points, noise, x_values, show):
    expected_improvement_values = []
    minimum_value = training_data_points[0][1]
    for i in range(len(training_data_points)):
        if training_data_points[i][1] < minimum_value:
            minimum_value = training_data_points[i][1]
    if show:
        _, predictive_mean_values, _ = calculate_gaussian_process_interval(training_data_points, noise, x_values, True)
    else:
        _, predictive_mean_values, _ = calculate_gaussian_process_interval(training_data_points, noise, x_values, False)
    for i in range(len(x_values)):
        expected_improvement_values.append(max(minimum_value - predictive_mean_values[i], 0))
    optimal_value = 0
    optimal_parameter_value = 0
    for i in range(len(expected_improvement_values)):
        if expected_improvement_values[i] > optimal_value:
            optimal_value = expected_improvement_values[i]
            optimal_parameter_value = x_values[i]
    if show:
        plt.plot(x_values, expected_improvement_values)
        plt.show()
    return optimal_parameter_value

"""This function calculates the error between 2 values, necessary to determine if optimal acquisition
hyperparameter values are close enough to be considered similar."""
def calculate_error(x1, x2):
    return abs(x1 - x2) / x1

"""This function runs the acquisition function for each numerical hyperparameter in the neural network
model to determine the optimal value for each hyperparameter, based on the model's training data. The function
keeps evaluating optimal acquisition hyperparameter values for a given parameter until the the last 3 optimal acquisition
hyperparameter values are close enough - with an error of less than 1% - to be considered similar and the actual resulting cost
value from the last optimal hyperparameter value is the minimum of all costs from previous optimal hyperparameter values
yielded by the acquisiton function (or if the acquisition functions returns the same values)."""
def optimize_neural_network_hyperparameters(model_training_data_file_name, hyperparameters, model_reference = None):
    optimal_hyperparameter_values = {}
    for hyperparameter in hyperparameters:
        optimal_acquisition_hyperparameter_values = []
        optimal_cost_values = []
        if hyperparameters[hyperparameter][2] == None and hyperparameters[hyperparameter][3] == None:
            initial_hyperparameter_training_values = np.linspace(hyperparameters[hyperparameter][0] / 10, hyperparameters[hyperparameter][0] * 10, 10)
        elif hyperparameters[hyperparameter][2] != None and hyperparameters[hyperparameter][3] != None:
            initial_hyperparameter_training_values = np.linspace(hyperparameters[hyperparameter][2], hyperparameters[hyperparameter][3], 10)
        elif hyperparameters[hyperparameter][2] != None:
            initial_hyperparameter_training_values = np.linspace(hyperparameters[hyperparameter][2], hyperparameters[hyperparameter][0] * 10, 10)
        else:
            initial_hyperparameter_training_values = np.linspace(hyperparameters[hyperparameter][0] / 10, hyperparameters[hyperparameter][3], 10)
        if hyperparameters[hyperparameter][1] == None:
            for i in range(len(initial_hyperparameter_training_values)):
                initial_hyperparameter_training_values[i] = int(initial_hyperparameter_training_values[i])
        else:
            for i in range(len(initial_hyperparameter_training_values)):
                initial_hyperparameter_training_values[i] = round(initial_hyperparameter_training_values[i], hyperparameters[hyperparameter][1])
        training_data_points = []
        print("Step A")
        print(initial_hyperparameter_training_values)
        for i in range(len(initial_hyperparameter_training_values)):
            hyperparameter_reference = {
                hyperparameter: initial_hyperparameter_training_values[i]
            }
            if model_reference == None:
                model = NeuralNetworkModel(training_data_file_name = model_training_data_file_name, **hyperparameter_reference)
            else:
                model = NeuralNetworkModel(training_data_file_name = model_training_data_file_name, text_reference = model_reference, **hyperparameter_reference)
            training_data_points.append((initial_hyperparameter_training_values[i], model.average_validation_cost_values_over_epochs[model.minimum_cost_index]))
        print("Step B")
        print(training_data_points)
        x_values = np.linspace(1 / 10, initial_hyperparameter_training_values[-1], int(initial_hyperparameter_training_values[-1] * 10))
        optimal_acquisition_hyperparameter_value = calculate_optimal_acquisition_value(training_data_points, abs(model.std_dev_noise), x_values, False)
        if hyperparameters[hyperparameter][1] == None:
            optimal_acquisition_hyperparameter_value = int(optimal_acquisition_hyperparameter_value)
        else:
            optimal_acquisition_hyperparameter_value = round(optimal_acquisition_hyperparameter_value, hyperparameters[hyperparameter][1])
        optimal_acquisition_hyperparameter_values.append(optimal_acquisition_hyperparameter_value)
        hyperparameter_reference = {
            hyperparameter: optimal_acquisition_hyperparameter_value
        }
        if model_reference == None:
            model = NeuralNetworkModel(training_data_file_name = model_training_data_file_name, **hyperparameter_reference)
        else:
            model = NeuralNetworkModel(training_data_file_name = model_training_data_file_name, text_reference = model_reference, **hyperparameter_reference)
        optimal_cost_value = model.average_validation_cost_values_over_epochs[model.minimum_cost_index]
        training_data_points.append((optimal_acquisition_hyperparameter_value, optimal_cost_value))
        optimal_cost_values.append(optimal_cost_value)
        print("Step C")
        print(optimal_acquisition_hyperparameter_value)
        print(optimal_cost_value)
        minimum_optimal_cost_value = 0
        iteration = 1
        while not(len(training_data_points) > 1 and training_data_points[-1][0] == training_data_points[-2][0]) and (optimal_cost_value != minimum_optimal_cost_value or not(len(optimal_acquisition_hyperparameter_values) > 2 and calculate_error(optimal_acquisition_hyperparameter_values[-3], optimal_acquisition_hyperparameter_values[-1]) <= 0.01 and calculate_error(optimal_acquisition_hyperparameter_values[-2], optimal_acquisition_hyperparameter_values[-1]) <= 0.01)):
            minimum_optimal_cost_value = optimal_cost_values[0]
            optimal_acquisition_hyperparameter_value = calculate_optimal_acquisition_value(training_data_points, abs(model.std_dev_noise), x_values, False)
            if hyperparameters[hyperparameter][1] == None:
                optimal_acquisition_hyperparameter_value = int(optimal_acquisition_hyperparameter_value)
            else:
                optimal_acquisition_hyperparameter_value = round(optimal_acquisition_hyperparameter_value, hyperparameters[hyperparameter][1])
            optimal_acquisition_hyperparameter_values.append(optimal_acquisition_hyperparameter_value)
            hyperparameter_reference = {
                hyperparameter: optimal_acquisition_hyperparameter_value
            }
            if model_reference == None:
                model = NeuralNetworkModel(training_data_file_name = model_training_data_file_name, **hyperparameter_reference)
            else:
                model = NeuralNetworkModel(training_data_file_name = model_training_data_file_name, text_reference = model_reference, **hyperparameter_reference)
            optimal_cost_value = model.average_validation_cost_values_over_epochs[model.minimum_cost_index]
            training_data_points.append((optimal_acquisition_hyperparameter_value, optimal_cost_value))
            optimal_cost_values.append(optimal_cost_value)
            if optimal_cost_value < minimum_optimal_cost_value:
                minimum_optimal_cost_value = optimal_cost_value
            print("Step D - Iteration " + str(iteration))
            print(optimal_acquisition_hyperparameter_value)
            print(optimal_cost_value)
            print(minimum_optimal_cost_value)
            iteration += 1
        optimal_hyperparameter_values[hyperparameter] = optimal_acquisition_hyperparameter_values[-1]
        print("Step E")
    return optimal_hyperparameter_values

main()
