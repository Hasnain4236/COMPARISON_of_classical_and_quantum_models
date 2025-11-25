# Quantum vs Classical Machine Learning Comparison Tool

A comprehensive benchmarking tool to compare **Classical Machine Learning (CML)**, **Quantum Machine Learning (QML)**, and **Hybrid Quantum-Classical** models. This project is designed to empirically analyze the performance, training time, and efficiency of quantum algorithms on standard datasets.

## 🚀 Features
- **Classical Models**: SVM (RBF Kernel), Random Forest, Linear Regression, SVR.
- **Quantum Models**: Quantum SVM (Fidelity Kernel), Variational Quantum Classifier (VQC), Quantum Regression (VQR).
- **Hybrid Models**: PyTorch + Qiskit Hybrid Quantum Neural Networks (QNN).
- **Interactive UI**: Built with Streamlit for easy dataset selection and model configuration.
- **Hardware Support**: Seamless switching between Local Simulators (Qiskit Aer) and Real IBM Quantum Hardware.

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Hasnain4236/COMPARISON_of_classical_and_quantum_models.git
   cd COMPARISON_of_classical_and_quantum_models
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Ensure you have `qiskit`, `qiskit-machine-learning`, `torch`, `streamlit`, `scikit-learn`, `pandas`, `matplotlib` installed)*

## 🖥️ Usage

1. **Run the Application**
   ```bash
   streamlit run src/app.py
   ```
2. **Select Dataset**: Choose from Iris, Wine, Breast Cancer, or upload your own CSV.
3. **Choose Hardware**: Toggle between Simulator (default) and Real Hardware (requires API Token).
4. **Select Models**: Check the boxes for the models you want to compare.
5. **Run Comparison**: Click "Run Comparison" to see accuracy and time metrics.

## 🤝 Contributing
We welcome contributions! Whether it's adding new quantum algorithms, optimizing existing ones, or improving the UI.

1. **Fork the Project**
2. **Create your Feature Branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your Changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the Branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Ideas for Contribution
- Implement **Quantum PCA** for dimensionality reduction.
- Add **Pegasos QSVM** for faster training.
- Integrate **Pennylane** specific models.
- Improve the **Hybrid QNN** architecture for better convergence.

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.
