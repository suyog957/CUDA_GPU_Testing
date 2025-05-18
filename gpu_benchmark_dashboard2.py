# gpu_benchmark_dashboard.py

# Import necessary libraries
import torch  # PyTorch for tensor operations and GPU acceleration
import time  # Time measurement for benchmarking
import streamlit as st  # Streamlit is used to build an interactive web dashboard where the benchmark results will be displayed
import pandas as pd  # For handling benchmarking data
import matplotlib.pyplot as plt  # For visualizing performance trends

# Setup the dashboard title and description
st.set_page_config(layout="wide")
st.title("🚀 GPU Benchmarking Suite")
st.markdown("### Compare matrix multiplication performance on CPU vs GPU across different matrix sizes")

# Detect and display the current active device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.markdown(f"<span style='font-size:40px;'>🖥️ Device: `{device}`</span>", unsafe_allow_html=True)

# Data container for plotting
if 'matrix_results' not in st.session_state:
    st.session_state.matrix_results = []

# Matrix Multiplication Benchmark
st.subheader("🔢 Matrix Multiplication Benchmark")
col1, col2 = st.columns([1, 2])

# Always show the plot first if data exists
with col2:
    if st.session_state.matrix_results:
        df = pd.DataFrame(st.session_state.matrix_results)
        df = df.sort_values(by='size')
        fig, ax = plt.subplots(figsize=(6, 3))  # smaller plot size
        ax.plot(df['size'], df['cpu'], label='CPU', marker='o')
        ax.plot(df['size'], df['gpu'], label='GPU', marker='o')
        ax.set_title("Matrix Multiplication Time vs Matrix Size", fontsize=10)
        ax.set_xlabel("Matrix Size (NxN)", fontsize=8)
        ax.set_ylabel("Time (seconds)", fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(fontsize=8)
        st.pyplot(fig, clear_figure=True)

# Run benchmark controls
with col1:
    size = st.slider("Matrix size (NxN)", min_value=512, max_value=8192, step=512, value=2048)
    if st.button("Run Benchmark"):
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

        st.session_state.matrix_results.append({'size': size, 'cpu': cpu_time, 'gpu': gpu_time})

        st.markdown(f"<span style='font-size:40px;'>🧠 CPU Time: `{cpu_time:.4f}s`</span>", unsafe_allow_html=True)
        if gpu_time:
            st.markdown(f"<span style='font-size:35px;'>💻 GPU Time: `{gpu_time:.4f}s`</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size:35px;'>⚡ Speedup: `{cpu_time / gpu_time:.2f}x`</span>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Built with PyTorch and Streamlit")
