# Greenhouse-cross_modality-regression (CMR)

Description: 
This project develops a deep learning framework for energy and bandwidth management in IoT and wireless senor networks. The proposed paradigm, particularly targeted for agricultural IoT applications, uses a multivariate regression approach for predicting different sensing parameters from energy harvested voltage readings. This approach reduces the bandwidth usage and energy consumption at the resource-constrained sensor nodes, by reducing the amount of data that needs to be transmitted. In addition, the number of sensing modalities is significantly reduced, thus reducing the hardware cost of the IoT system. This is done while ensuring a successful reconstruction of all the signals that the sensor nodes intend to send to the receiver.

The files regression_PAR.py, regression_light.py, regression_temperature.py, regression_humidity.py contain the code for execution of the CMR models for prediction of PAR, light intensity, temperature and humidity respectively.

To run the code:

1. Import all the required machine learning and input-output libraries
2. Load the dataset from the local directory as a csv file
3. Define the number of learning epochs
4. Preprocess the dataset to make it suitable for NN model
5. Shuffle the dataset and split it into training set, test set and validation set
6. Define the NN model with all its parameters, including loss function, optimizer, evaluation metric
7. Provide an early stopping criterion to avoid overtraining
8. Compile and run the model
9. Find the regression output for a given set of input samples
10. Plot the true value and the corresponding CMR output
