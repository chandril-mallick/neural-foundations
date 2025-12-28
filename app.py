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

nn = st.session_state['nn']

# Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Training Control")
    start_btn = st.button("Train Network")
    st.metric("Current Loss", f"{st.session_state['loss_history'][-1]:.6f}" if st.session_state['loss_history'] else "0.0")
    st.metric("Epochs", f"{st.session_state['epoch_count']}")

    # Plot Loss
    st.markdown("### Loss Curve")
    chart_placeholder = st.empty()
    if st.session_state['loss_history']:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(st.session_state['loss_history'])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        chart_placeholder.pyplot(fig)
        plt.close(fig)

with col2:
    st.subheader("Decision Boundary")
    boundary_placeholder = st.empty()

# Training Loop
if start_btn:
    progress_bar = st.progress(0)
    
    for i in range(epochs):
        # Step
        err = nn.train_one_epoch(X, y, lr)
        st.session_state['loss_history'].append(err)
        st.session_state['epoch_count'] += 1
        
        # Update UI
        if (i + 1) % refresh_rate == 0 or (i + 1) == epochs:
            # Loss Chart
            fig_loss, ax_loss = plt.subplots(figsize=(4, 2.5))
            ax_loss.plot(st.session_state['loss_history'], color='#ff4b4b')
            ax_loss.set_title("Training Loss")
            ax_loss.grid(True, alpha=0.3)
            chart_placeholder.pyplot(fig_loss)
            plt.close(fig_loss)
            
            # Decision Boundary
            fig_b, ax_b = plt.subplots(figsize=(6, 5))
            
            # Meshgrid
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            h = 0.05
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            
            # Manual Predict for grid
            # Ideally nn.predict would handle batch, but we used a loop implementation
            # Let's just loop quickly or rely on array broadcasting if layers supported it (Dense does)
            # Our Dense implementation uses np.dot(input, weights) so it supports (N, D).
            # But Network.predict forced loop for tutorial clarity on shapes.
            # We can bypass Network.predict and call layers directly for speed here.
            
            # Optimized forward pass for vis
            out = grid_points
            for layer in nn.layers:
                out = layer.forward(out)
                
            Z = out.reshape(xx.shape)
            
            ax_b.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
            ax_b.scatter(X[:, 0], X[:, 1], c=y.flatten(), s=80, cmap=plt.cm.Spectral, edgecolors='k')
            ax_b.set_title(f"Decision Boundary (Epoch {st.session_state['epoch_count']})")
            boundary_placeholder.pyplot(fig_b)
            plt.close(fig_b)
            
            progress_bar.progress((i + 1) / epochs)
            
    st.success("Training Complete!")
