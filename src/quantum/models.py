from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import QSVC, VQC, VQR
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.primitives import Sampler, Estimator
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.state_fidelities import ComputeUncompute

# Runtime imports
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler as RuntimeSampler, Estimator as RuntimeEstimator, Session
    RUNTIME_AVAILABLE = True
except ImportError:
    RUNTIME_AVAILABLE = False

import time
import numpy as np

class QuantumModel:
    def __init__(self, model_type, n_qubits=None, api_token=None, custom_url=None):
        """
        n_qubits: Number of features/qubits.
        api_token: IBM Quantum API Token.
        custom_url: Optional custom URL for IBM Quantum service (to bypass DNS blocks).
        """
        self.model_type = model_type
        self.n_qubits = n_qubits
        self.api_token = api_token
        self.custom_url = custom_url
        self.model = None
        self.optimizer = COBYLA(maxiter=50) 

    def _build_model(self, n_features):
        self.n_qubits = n_features
        
        # Setup Primitives
        sampler = Sampler()
        estimator = Estimator()
        
        if self.api_token:
            if not RUNTIME_AVAILABLE:
                raise ImportError("qiskit-ibm-runtime is not installed.")
            
            # Initialize Service
            try:
                # Prepare args
                kwargs = {"channel": "ibm_quantum", "token": self.api_token}
                if self.custom_url:
                    kwargs["url"] = self.custom_url
                
                service = QiskitRuntimeService(**kwargs)
            except Exception as e:
                if "getaddrinfo failed" in str(e) or "Max retries exceeded" in str(e):
                    raise ConnectionError("Network Error: Unable to connect to IBM Quantum. Please check your internet connection or VPN.") from e
                else:
                    raise e
                
            backend = service.least_busy(operational=True, simulator=False)
            print(f"Using backend: {backend.name}")
            
            # Create Runtime Primitives with a Session
            session = Session(service=service, backend=backend)
            sampler = RuntimeSampler(session=session)
            estimator = RuntimeEstimator(session=session)

        if self.model_type == "Q_QSVM":
            feature_map = ZZFeatureMap(feature_dimension=self.n_qubits, reps=2, entanglement='linear')
            
            # Use ComputeUncompute to leverage the Runtime Sampler
            fidelity = ComputeUncompute(sampler=sampler)
            kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
            
            self.model = QSVC(quantum_kernel=kernel)
            
        elif self.model_type == "Q_VQC":
            feature_map = ZZFeatureMap(feature_dimension=self.n_qubits)
            ansatz = RealAmplitudes(num_qubits=self.n_qubits, reps=2)
            self.model = VQC(feature_map=feature_map,
                             ansatz=ansatz,
                             optimizer=self.optimizer,
                             sampler=sampler)
                             
        elif self.model_type == "Q_VQR":
            feature_map = ZZFeatureMap(feature_dimension=self.n_qubits)
            ansatz = RealAmplitudes(num_qubits=self.n_qubits, reps=2)
            self.model = VQR(feature_map=feature_map,
                             ansatz=ansatz,
                             optimizer=self.optimizer,
                             estimator=estimator)
        else:
             raise ValueError(f"Unknown quantum model: {self.model_type}")

    def train(self, X_train, y_train):
        # Infer n_qubits from data if not set
        n_features = X_train.shape[1]
        if self.model is None or self.n_qubits != n_features:
            self._build_model(n_features)
            
        start_time = time.time()
        self.model.fit(X_train, y_train)
        end_time = time.time()
        return end_time - start_time

    def evaluate(self, X_test, y_test):
        start_time = time.time()
        score = self.model.score(X_test, y_test)
        end_time = time.time()
        return score, end_time - start_time
