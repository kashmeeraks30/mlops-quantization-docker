import numpy as np
import joblib
import os
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from utils import load_model,extract_parameters


def parameter_quantization(parameters, nbits):
    level = 2 ** nbits - 1
    value = np.round(parameters * level).astype(np.uint8)
    return value


def parameter_dequantization(quant_parameters, min_value, max_value, nbits):
    level = 2 ** nbits - 1
    # Convert back to float and normalize to [0, 1]
    normalized_params = quant_parameters.astype(np.float32) / level
    # Scale back to original range
    dequantized_params = normalized_params * (max_value - min_value) + min_value
    return dequantized_params


class LinearRegressionTorch(nn.Module):
    def __init__(self, num_features, num_outputs):
        super().__init__()
        self.linear = nn.Linear(num_features, num_outputs)
    
    def forward(self, x):
        return self.linear(x)


def main():
    # Load California housing dataset
    housingdata = fetch_california_housing()
    X, y = housingdata.data, housingdata.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_normalize_train = scaler.fit_transform(X_train)
    X_normalize_test = scaler.transform(X_test)  # Use same scaler, just transform

    # Train the model on scaled data
    model = LinearRegression()
    model.fit(X_normalize_train, y_train)
    
    # Get original performance for comparison
    y_pred = model.predict(X_normalize_test)
    r2_original = r2_score(y_test, y_pred)

    #Loading the saved scikit-learn model from .joblib file
    model=load_model()

    #Extracting the parameters 
    coef,intercept=extract_parameters(model)

    print(coef)
    print(intercept)

    
    os.makedirs('Models', exist_ok=True)
    
    # Unquantized parameters stored in a dictionary and saved as unquant_params.joblib
    unquantized_param = {'coef': coef, 'intercept': intercept}
    joblib.dump(unquantized_param, 'Models/unquant_params.joblib')
    print(f"Size of original sklearn model: {(os.path.getsize('Models/unquant_params.joblib')/1024):.2f} KB")

    # Normalizing the parameter values between 0 and 1
    parameters = np.asarray(coef)
    parameters = np.append(parameters, intercept)

    min_value = parameters.min()
    max_value = parameters.max()
    
    # Normalize to [0, 1] range
    parameters_normalized = (parameters - min_value) / (max_value - min_value)
    
    # Performing manual quantization of parameters to unsigned 8 bit integer
    quantized_parameters = parameter_quantization(parameters_normalized, 8)
    
    # Quantized parameters stored in a dictionary and saved as manual_quant_params.joblib
    quantized_coef = quantized_parameters[:-1]
    quantized_intercept = quantized_parameters[-1]

    quantized_param = {'coef': quantized_coef, 'intercept': quantized_intercept}
    joblib.dump(quantized_param, 'Models/manual_quant_params.joblib')
    print(f"Size of manually quantized model: {(os.path.getsize('Models/manual_quant_params.joblib')/1024):.2f} KB")

    # Creating a single layer PyTorch 1-layer neural network model
    weights = np.asarray(coef, dtype=np.float32)
    bias = np.asarray(intercept, dtype=np.float32)

    num_outputs = 1
    num_features = weights.shape[0]

    pt_model = LinearRegressionTorch(num_features, num_outputs)

    # Ensure weights are 2D for Linear layer
    weight_tensor = torch.from_numpy(weights.reshape(1, -1))
    # Ensure bias is 1D
    bias_tensor = torch.from_numpy(np.array([bias]))
    
    pt_model.linear.weight.data = weight_tensor
    pt_model.linear.bias.data = bias_tensor

    # Applying dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        pt_model, {nn.Linear}, dtype=torch.qint8)

    dump(quantized_model.state_dict(), "Models/quant_params.joblib")
    print(f'Size of PyTorch quantized model: {(os.path.getsize("Models/quant_params.joblib"))/1024:.2f} KB')

    # Performing inference with de-quantized weights
    quantized_params = np.append(quantized_coef, quantized_intercept)
    
    # De-quantize parameters
    dequantized_params = parameter_dequantization(quantized_params, min_value, max_value, 8)
    
    # Split back into coefficients and intercept
    dequant_coef = dequantized_params[:-1]
    dequant_intercept = dequantized_params[-1]
    
    # Use scaled test data for inference
    y_pred_dequant = np.dot(X_normalize_test, dequant_coef) + dequant_intercept

    r2_quantized_model = r2_score(y_test, y_pred_dequant)
    mse_quantized = mean_squared_error(y_test, y_pred_dequant)
    
    print(f"\nInference results with de-quantized weights")
    print(f"Original model R2 score: {r2_original:.6f}")
    print(f"Quantized model R2 score: {r2_quantized_model:.6f}")
    print(f"Original MSE: {mean_squared_error(y_test, y_pred):.6f}")
    print(f"Quantized MSE: {mse_quantized:.6f}")
    

if __name__ == "__main__":
    main()