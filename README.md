# bayesian-optimization
An algorithm for efficient optimization of neural network hyperparameters.

REFERENCE: General-Purpose Regression Neural Network - https://docs.google.com/document/d/1WYskp_TW9B5trKevxpwKKoMcgW6UpGYwqKvJP5Pl37U/edit?usp=sharing

This program is designed to perform Bayesian Optimization for neural network hyperparameters, specifically designed for a previously developed general-purpose regression neural network. The purpose of this program is to explore the computational benefits of using stochastic gradient descent after fine-tuning hyperparameters through Bayesian Optimization. The general-purpose regression neural network utilized stochastic gradient descent (SGD) in order to adjust and optimize model parameters. SGD provides a way for more dynamic and efficient learning for neural network models, but can often lead to noisy convergence and make models more sensitive to hyperparameters, such as learning rate and momentum. 

Through this program, we aim to evaluate how employing SGD along with hyperparameter-tuning through Bayesian Optimization can allow neural networks to have robust learning and predictive performance while providing computational efficiency and dynamic responsiveness to new data.

