import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator

from qiskit.quantum_info import SparsePauliOp

class HybridQNN(nn.Module):
    def __init__(self, n_inputs, n_qubits=2, n_outputs=2):
        super(HybridQNN, self).__init__()
        self.n_inputs = n_inputs
        self.n_qubits = n_qubits
        self.n_outputs = n_outputs
        
        # 1. Classical Input Layer
        self.cl_layer1 = nn.Linear(n_inputs, n_qubits)
        
        # 2. Quantum Layer
        feature_map = ZZFeatureMap(n_qubits)
        ansatz = RealAmplitudes(n_qubits, reps=1)
        qc = feature_map.compose(ansatz)
        
        # Define observables: Measure Z on each qubit
        observables = [SparsePauliOp.from_list([("I" * i + "Z" + "I" * (n_qubits - 1 - i), 1)]) for i in range(n_qubits)]
        
        qnn = EstimatorQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            observables=observables,
            estimator=Estimator()
        )
        self.q_layer = TorchConnector(qnn)
        
        # 3. Classical Output Layer
        # Maps quantum output (expectation values) to class probabilities
        # EstimatorQNN output size is usually num_qubits (if measuring all Z) or 1.
        # Default EstimatorQNN with no observables measures all qubits Z.
        self.cl_layer2 = nn.Linear(n_qubits, n_outputs)
        
    def forward(self, x):
        x = torch.tanh(self.cl_layer1(x)) # Tanh to keep in range for rotation angles
        x = self.q_layer(x)
        x = torch.sigmoid(self.cl_layer2(x)) # Sigmoid/Softmax for classification
        return x

class HybridModel:
    def __init__(self, model_type="Hybrid_QNN", n_qubits=2, epochs=10):
        self.model_type = model_type
        self.n_qubits = n_qubits
        self.epochs = epochs
        self.model = None
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None

    def train(self, X_train, y_train):
        # Convert to Tensor
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.LongTensor(y_train)
        
        n_features = X_train.shape[1]
        n_classes = len(np.unique(y_train))
        
        # Initialize Model
        self.model = HybridQNN(n_features, self.n_qubits, n_classes)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        
        start_time = time.time()
        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = self.criterion(output, y_tensor)
            loss.backward()
            self.optimizer.step()
        end_time = time.time()
        
        return end_time - start_time

    def evaluate(self, X_test, y_test):
        X_tensor = torch.FloatTensor(X_test)
        y_tensor = torch.LongTensor(y_test)
        
        start_time = time.time()
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_tensor).sum().item()
            score = correct / len(y_test)
            
        end_time = time.time()
        return score, end_time - start_time
