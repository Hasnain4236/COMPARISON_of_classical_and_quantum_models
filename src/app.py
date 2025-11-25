import streamlit as st
import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.data_loader import load_benchmark, preprocess_data

st.set_page_config(page_title="QML vs CML Comparison", layout="wide")

st.title("Quantum vs Classical Machine Learning Comparison")
st.markdown("""
This tool compares **Quantum Machine Learning (QML)** models against **Classical Machine Learning (CML)** models.
Configure your experiment in the sidebar and click 'Run Comparison'.
""")

# Sidebar for controls
st.sidebar.header("Configuration")

# 1. Dataset Selection
st.sidebar.subheader("1. Data")
dataset_source = st.sidebar.radio("Dataset Source", ["Standard Benchmark", "Upload CSV"])
dataset_name = None
df = None
target_col = None

if dataset_source == "Standard Benchmark":
    dataset_name = st.sidebar.selectbox("Select Dataset", ["Iris", "Breast Cancer", "Wine", "California Housing"])
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Loaded {len(df)} rows.")
        target_col = st.sidebar.selectbox("Select Target Column", df.columns)

# 2. Hardware Selection
st.sidebar.subheader("2. Hardware")
hardware = st.sidebar.toggle("Use Real Quantum Hardware (IBM)", value=False)

# Try to load API key from file
default_token = ""
try:
    import json
    key_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'apikey.json')
    if os.path.exists(key_path):
        with open(key_path, 'r') as f:
            data = json.load(f)
            default_token = data.get("apikey", "")
            if default_token:
                st.sidebar.success("Loaded API Key from file.")
except Exception as e:
    pass

api_token = ""
if hardware:
    api_token = st.sidebar.text_input("IBM Quantum API Token", value=default_token, type="password")
    if not api_token:
        st.sidebar.warning("Please enter your API Token to proceed with real hardware.")
    else:
        st.sidebar.info("Using IBM Quantum Runtime.")
        
    # Advanced: Custom URL
    with st.sidebar.expander("Advanced: Network Settings"):
        custom_url = st.text_input("Custom Auth URL (Optional)", value="https://auth.quantum-computing.ibm.com/api")
        st.caption("Try 'https://quantum.cloud.ibm.com' if default fails.")
else:
    st.sidebar.info("Using Local Simulators (Qiskit Aer / PennyLane).")
    custom_url = None

# 3. Model Selection
st.sidebar.subheader("3. Models")
task_type = st.sidebar.selectbox("Task Type", ["Classification", "Regression", "Reinforcement Learning"])

models_to_run = []
if task_type == "Classification":
    if st.sidebar.checkbox("Classical: SVM (RBF)", value=True): models_to_run.append("C_SVM")
    if st.sidebar.checkbox("Classical: Random Forest", value=True): models_to_run.append("C_RF")
    if st.sidebar.checkbox("Quantum: QSVM (Kernel)", value=True): models_to_run.append("Q_QSVM")
    if st.sidebar.checkbox("Quantum: VQC", value=False): models_to_run.append("Q_VQC")
    if st.sidebar.checkbox("Hybrid: QNN (PyTorch)", value=True): models_to_run.append("H_QNN")

elif task_type == "Regression":
    if st.sidebar.checkbox("Classical: Linear Regression", value=True): models_to_run.append("C_LinReg")
    if st.sidebar.checkbox("Classical: SVR", value=True): models_to_run.append("C_SVR")
    if st.sidebar.checkbox("Quantum: VQR", value=True): models_to_run.append("Q_VQR")

elif task_type == "Reinforcement Learning":
    st.sidebar.info("RL Environment: CartPole-v1")
    if st.sidebar.checkbox("Classical: DQN", value=True): models_to_run.append("C_DQN")
    if st.sidebar.checkbox("Quantum: Q-Learning (VQC)", value=True): models_to_run.append("Q_DQN")

# 4. Execution
if st.button("Run Comparison", type="primary"):
    if not models_to_run:
        st.error("Please select at least one model.")
    else:
        st.write(f"### Running Experiments for {task_type}...")
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Load Data
        X_train, X_test, y_train, y_test = None, None, None, None
        
        if task_type != "Reinforcement Learning":
            status_text.text("Loading and preprocessing data...")
            if dataset_source == "Standard Benchmark":
                X, y, _ = load_benchmark(dataset_name)
            else:
                if df is not None and target_col is not None:
                    X = df.drop(columns=[target_col]).values
                    y = df[target_col].values
                else:
                    st.error("Please upload data and select a target column.")
                    st.stop()
            
            # Preprocess
            X_train, X_test, y_train, y_test = preprocess_data(X, y, task_type)
            
            # Reduce dimensions for Quantum (Simulators are slow with > 4-8 qubits)
            if "Quantum" in str(models_to_run) and X_train.shape[1] > 4:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=4)
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)
                st.info("Reduced data to 4 dimensions using PCA for Quantum compatibility.")

        # Import Models
        from classical.models import ClassicalModel
        from quantum.models import QuantumModel
        from hybrid.models import HybridModel
        
        total_models = len(models_to_run)
        
        for i, model_name in enumerate(models_to_run):
            status_text.text(f"Training {model_name}...")
            
            try:
                # Classification / Regression
                if task_type in ["Classification", "Regression"]:
                    score = 0
                    train_time = 0
                    inference_time = 0
                    
                    if model_name.startswith("C_"):
                        model = ClassicalModel(model_name)
                        train_time = model.train(X_train, y_train)
                        score, inference_time = model.evaluate(X_test, y_test)
                    elif model_name.startswith("Q_"):
                        # Pass api_token if hardware is selected
                        token_to_use = api_token if hardware else None
                        url_to_use = custom_url if hardware and custom_url else None
                        model = QuantumModel(model_name, api_token=token_to_use, custom_url=url_to_use)
                        train_time = model.train(X_train, y_train)
                        score, inference_time = model.evaluate(X_test, y_test)
                    elif model_name.startswith("H_"):
                        model = HybridModel(model_name)
                        train_time = model.train(X_train, y_train)
                        score, inference_time = model.evaluate(X_test, y_test)
                        
                    results.append({
                        "Model": model_name,
                        "Score (Acc/R2)": round(score, 4),
                        "Train Time (s)": round(train_time, 4),
                        "Inference Time (s)": round(inference_time, 4)
                    })

                # Reinforcement Learning
                elif task_type == "Reinforcement Learning":
                    import gym
                    env = gym.make("CartPole-v1")
                    
                    if model_name == "C_DQN":
                        from classical.rl_models import ClassicalRL
                        agent = ClassicalRL(env)
                        rewards = agent.train(episodes=20) # Short run for demo
                        avg_reward = sum(rewards[-5:]) / 5
                        results.append({
                            "Model": model_name,
                            "Avg Reward (Last 5)": avg_reward,
                            "Total Episodes": 20
                        })
                    elif model_name == "Q_DQN":
                        # Placeholder for Q-RL
                        results.append({
                            "Model": model_name,
                            "Avg Reward (Last 5)": "N/A (Not Implemented)",
                            "Total Episodes": 0
                        })
                        
            except Exception as e:
                st.error(f"Error running {model_name}: {e}")
            
            progress_bar.progress((i + 1) / total_models)
            
        st.success("Comparison Complete!")
        
        # Results
        res_df = pd.DataFrame(results)
        st.table(res_df)
        
        # Plotting
        if task_type != "Reinforcement Learning":
            st.subheader("Performance Comparison")
            st.bar_chart(res_df.set_index("Model")["Score (Acc/R2)"])
            
            st.subheader("Time Comparison")
            st.bar_chart(res_df.set_index("Model")["Train Time (s)"])
