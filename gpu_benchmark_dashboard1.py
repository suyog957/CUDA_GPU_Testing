# gpu_benchmark_dashboard.py

# Import necessary libraries
import torch  # PyTorch for tensor operations and GPU acceleration
import time  # Time measurement for benchmarking
import streamlit as st  # Streamlit is used to build an interactive web dashboard where the benchmark results will be displayed
from PIL import Image  # Placeholder for future image operations
import numpy as np  # For numerical benchmarks
import pandas as pd  # For handling benchmarking data
import matplotlib.pyplot as plt  # For visualizing performance trends

# Setup the dashboard title and description
st.title("🚀 GPU Benchmarking Suite")
st.markdown("Compare matrix ops, elementwise ops, and convolutional ops on CPU vs GPU")

# Detect and display the current active device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"**Active Device:** `{device}`")

# Data containers for plotting
if 'matrix_results' not in st.session_state:
    st.session_state.matrix_results = []
if 'relu_results' not in st.session_state:
    st.session_state.relu_results = []
if 'conv_results' not in st.session_state:
    st.session_state.conv_results = []

# Section 1: Matrix Multiplication Benchmark
st.header("1️⃣ Matrix Multiplication")
size = st.slider("Matrix size (NxN)", min_value=512, max_value=8192, step=512, value=2048)

if st.button("Run Matrix Benchmark"):
    a = torch.randn(size, size)
    b = torch.randn(size, size)

    start = time.time()
    _ = torch.matmul(a, b)
    cpu_time = time.time() - start

    if torch.cuda.is_available():
        a_gpu = a.to(device)
        b_gpu = b.to(device)
        _ = torch.matmul(a_gpu, b_gpu)  # warm-up
        torch.cuda.synchronize()
        start = time.time()
        _ = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
    else:
        gpu_time = None

    st.write(f"🧠 CPU Time: `{cpu_time:.4f}s`")
    if gpu_time:
        st.write(f"💻 GPU Time: `{gpu_time:.4f}s`")
        speedup = cpu_time / gpu_time
        st.write(f"⚡ Speedup (CPU / GPU): `{speedup:.2f}x`")
        st.session_state.matrix_results.append({'size': size, 'cpu': cpu_time, 'gpu': gpu_time})

    if len(st.session_state.matrix_results) > 1:
        df = pd.DataFrame(st.session_state.matrix_results)
        fig, ax = plt.subplots()
        ax.plot(df['size'], df['cpu'], label='CPU')
        ax.plot(df['size'], df['gpu'], label='GPU')
        ax.set_title("Matrix Multiplication Time vs Size")
        ax.set_xlabel("Matrix Size (NxN)")
        ax.set_ylabel("Time (s)")
        ax.legend()
        st.pyplot(fig)

# Section 2: Elementwise Operation Benchmark
st.header("2️⃣ Elementwise ReLU Operation")
if st.button("Run ReLU Benchmark"):
    a = torch.randn(10000, 10000)

    start = time.time()
    _ = torch.relu(a)
    cpu_time = time.time() - start

    if torch.cuda.is_available():
        a_gpu = a.to(device)
        _ = torch.relu(a_gpu)
        torch.cuda.synchronize()
        start = time.time()
        _ = torch.relu(a_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
    else:
        gpu_time = None

    st.write(f"🧠 CPU Time: `{cpu_time:.4f}s`")
    if gpu_time:
        st.write(f"💻 GPU Time: `{gpu_time:.4f}s`")
        speedup = cpu_time / gpu_time
        st.write(f"⚡ Speedup (CPU / GPU): `{speedup:.2f}x`")
        st.session_state.relu_results.append({'run': len(st.session_state.relu_results)+1, 'cpu': cpu_time, 'gpu': gpu_time})

    if len(st.session_state.relu_results) > 1:
        df = pd.DataFrame(st.session_state.relu_results)
        fig, ax = plt.subplots()
        ax.plot(df['run'], df['cpu'], label='CPU')
        ax.plot(df['run'], df['gpu'], label='GPU')
        ax.set_title("ReLU Operation Time over Multiple Runs")
        ax.set_xlabel("Run #")
        ax.set_ylabel("Time (s)")
        ax.legend()
        st.pyplot(fig)

# Section 3: Convolution Simulation Benchmark
st.header("3️⃣ Convolutional Layer Simulation")
if st.button("Run Convolution Benchmark"):
    conv = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
    input_tensor = torch.randn(32, 3, 224, 224)

    start = time.time()
    _ = conv(input_tensor)
    cpu_time = time.time() - start

    if torch.cuda.is_available():
        conv = conv.to(device)
        input_tensor_gpu = input_tensor.to(device)
        _ = conv(input_tensor_gpu)
        torch.cuda.synchronize()
        start = time.time()
        _ = conv(input_tensor_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
    else:
        gpu_time = None

    st.write(f"🧠 CPU Time: `{cpu_time:.4f}s`")
    if gpu_time:
        st.write(f"💻 GPU Time: `{gpu_time:.4f}s`")
        speedup = cpu_time / gpu_time
        st.write(f"⚡ Speedup (CPU / GPU): `{speedup:.2f}x`")
        st.session_state.conv_results.append({'run': len(st.session_state.conv_results)+1, 'cpu': cpu_time, 'gpu': gpu_time})

    if len(st.session_state.conv_results) > 1:
        df = pd.DataFrame(st.session_state.conv_results)
        fig, ax = plt.subplots()
        ax.plot(df['run'], df['cpu'], label='CPU')
        ax.plot(df['run'], df['gpu'], label='GPU')
        ax.set_title("Convolution Time over Multiple Runs")
        ax.set_xlabel("Run #")
        ax.set_ylabel("Time (s)")
        ax.legend()
        st.pyplot(fig)

# Section footer
st.markdown("---")
st.caption("Built with PyTorch and Streamlit for GPU benchmarking")
