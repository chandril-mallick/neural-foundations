import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

from nn.network import NeuralNetwork
from nn.layers import Dense
from nn.activations import Sigmoid, ReLU
from nn.loss import MSE, BinaryCrossEntropy
from utils import generate_xor_data, generate_blobs

# Page Config
st.set_page_config(page_title="Neural Network Playground", page_icon="🧠", layout="wide")

# Styling
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("Neural Network Playground")
st.markdown("Implemented in pure **Python + NumPy**. No frameworks.")

# Sidebar
st.sidebar.header("Hyperparameters")
dataset_name = st.sidebar.selectbox("Dataset", ["XOR", "Blobs (Linear)"])
lr = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.1)
epochs = st.sidebar.slider("Epochs", 100, 10000, 2000)
refresh_rate = st.sidebar.slider("Refresh Rate (Epochs)", 10, 500, 100)

st.sidebar.markdown("---")
st.sidebar.subheader("Architecture")
hidden_neurons = st.sidebar.slider("Hidden Neurons", 1, 10, 3)

# Load Data
if dataset_name == "XOR":
    X, y = generate_xor_data()
    st.sidebar.info("XOR is a non-linear problem. Linear models fail here.")
else:
    X, y = generate_blobs(samples=200)
    st.sidebar.info("Two separated clusters. Easy for simple networks.")

# Initialize Network
if 'nn' not in st.session_state or st.sidebar.button("Reset Model"):
    model = NeuralNetwork(loss_function=MSE())
    model.add(Dense(2, hidden_neurons))
    model.add(Sigmoid()) # Using Sigmoid for stability in demo
    model.add(Dense(hidden_neurons, 1))
    model.add(Sigmoid())
    st.session_state['nn'] = model
    st.session_state['loss_history'] = []
    st.session_state['epoch_count'] = 0
    st.session_state['weight_history'] = {
        'w_hidden_0': [], 'w_hidden_1': [], 'w_output_0': []
    }

if 'weight_history' not in st.session_state:
    st.session_state['weight_history'] = {
        'w_hidden_0': [], 'w_hidden_1': [], # Track first couple of weights
        'w_output_0': []
    }

nn = st.session_state['nn']

# Tabs for Visualizations
tab1, tab2, tab3 = st.tabs(["🎯 Decision Boundary", "📉 Weight Dynamics", "🧠 Hidden Activations"])

with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Controls")
        start_btn = st.button("Train Network")
        st.metric("Current Loss", f"{st.session_state['loss_history'][-1]:.6f}" if st.session_state['loss_history'] else "0.0")
        st.metric("Epochs", f"{st.session_state['epoch_count']}")
        
        st.markdown("### Loss Curve")
        loss_chart_placeholder = st.empty()
        
    with col2:
        st.subheader("Decision Boundary")
        boundary_placeholder = st.empty()

with tab2:
    st.subheader("Weight Evolution")
    st.markdown("Tracking weights from Input → Hidden (Layer 1) and Hidden → Output (Layer 2).")
    weights_chart_placeholder = st.empty()

with tab3:
    st.subheader("Hidden Layer Representations")
    st.markdown("Visualizing what the hidden neurons are 'seeing' across the input space.")
    activations_placeholder = st.empty()


# Plotting Helper Functions (defined to keep loop clean)
def plot_loss_curve(placeholder):
    if st.session_state['loss_history']:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(st.session_state['loss_history'], color='#ff4b4b')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        placeholder.pyplot(fig)
        plt.close(fig)

def plot_decision_boundary_vis(placeholder, X, y, model, epoch):
    fig, ax = plt.subplots(figsize=(6, 5))
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Forward pass for grid
    out = grid_points
    for layer in model.layers:
        out = layer.forward(out)
    Z = out.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y.flatten(), s=80, cmap=plt.cm.Spectral, edgecolors='k')
    ax.set_title(f"Decision Boundary (Epoch {epoch})")
    placeholder.pyplot(fig)
    plt.close(fig)

def plot_weight_dynamics_vis(placeholder):
    if st.session_state['weight_history']['w_hidden_0']:
        fig, ax = plt.subplots(figsize=(10, 4))
        for k, v in st.session_state['weight_history'].items():
            ax.plot(v, label=k)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weight Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        placeholder.pyplot(fig)
        plt.close(fig)

def plot_hidden_activations_vis(placeholder, X, model):
    # Visualize activations of first 3 hidden neurons
    # Get grid
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    h = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Forward to hidden layer
    # Assuming standard structure: Dense -> Act -> Dense -> Act
    # We want output of first Activation layer
    hidden_out = grid
    # Layer 0: Dense, Layer 1: Activation
    if len(model.layers) >= 2:
        hidden_out = model.layers[0].forward(hidden_out)
        hidden_out = model.layers[1].forward(hidden_out)
    
    neurons_to_plot = min(hidden_out.shape[1], 3)
    fig, axes = plt.subplots(1, neurons_to_plot, figsize=(12, 4))
    if neurons_to_plot == 1: axes = [axes]
    
    for i in range(neurons_to_plot):
        Z = hidden_out[:, i].reshape(xx.shape)
        ax = axes[i]
        c = ax.contourf(xx, yy, Z, cmap='viridis')
        ax.set_title(f"Neuron {i+1}")
        ax.axis('off')
        
    placeholder.pyplot(fig)
    plt.close(fig)

# Initial Plots
plot_loss_curve(loss_chart_placeholder)
plot_decision_boundary_vis(boundary_placeholder, X, y, nn, st.session_state['epoch_count'])
plot_weight_dynamics_vis(weights_chart_placeholder)
plot_hidden_activations_vis(activations_placeholder, X, nn)

# Training Loop
if start_btn:
    progress_bar = st.progress(0)
    
    for i in range(epochs):
        # Step
        err = nn.train_one_epoch(X, y, lr)
        st.session_state['loss_history'].append(err)
        st.session_state['epoch_count'] += 1
        
        # Track Weights (Sample a few)
        # Layer 0 is Input->Hidden Dense
        if len(nn.layers) > 0 and hasattr(nn.layers[0], 'weights'):
            w = nn.layers[0].weights
            st.session_state['weight_history']['w_hidden_0'].append(w[0, 0])
            st.session_state['weight_history']['w_hidden_1'].append(w[0, 1] if w.shape[1] > 1 else 0)
        # Layer 2 is Hidden->Output Dense (index 2 because 0=Dense, 1=Act, 2=Dense)
        if len(nn.layers) > 2 and hasattr(nn.layers[2], 'weights'):
            w_out = nn.layers[2].weights
            st.session_state['weight_history']['w_output_0'].append(w_out[0, 0])
        
        # Update UI
        if (i + 1) % refresh_rate == 0 or (i + 1) == epochs:
            plot_loss_curve(loss_chart_placeholder)
            plot_decision_boundary_vis(boundary_placeholder, X, y, nn, st.session_state['epoch_count'])
            plot_weight_dynamics_vis(weights_chart_placeholder)
            plot_hidden_activations_vis(activations_placeholder, X, nn)
            
            progress_bar.progress((i + 1) / epochs)
            
    st.success("Training Complete!")
