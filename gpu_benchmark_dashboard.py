# gpu_benchmark_dashboard.py

# Import necessary libraries
import torch  # PyTorch for tensor operations and GPU acceleration
import time  # Time measurement for benchmarking
import streamlit as st  # Streamlit is used to build an interactive web dashboard where the benchmark results will be displayed
from transformers import AutoTokenizer, AutoModel  # Hugging Face for transformer inference benchmarking
from PIL import Image  # Placeholder for image generation modules
import numpy as np  # For potential numerical benchmarks and placeholder features

# Setup the dashboard title and description
st.title("🚀 GPU Benchmarking Suite")
st.markdown("Compare matrix ops, transformer inference, and image generation on CPU vs GPU")

# Detect and display the current active device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"**Active Device:** `{device}`")

# Section 1: Matrix Multiplication Benchmark
st.header("1️⃣ Matrix Multiplication")
# Allow the user to select the size of the matrix
size = st.slider("Matrix size (NxN)", min_value=512, max_value=8192, step=512, value=2048)

# When the user clicks the button, the benchmark runs
if st.button("Run Matrix Benchmark"):
    # Generate random NxN matrices
    a = torch.randn(size, size)
    b = torch.randn(size, size)

    # CPU benchmark
    start = time.time()
    _ = torch.matmul(a, b)
    cpu_time = time.time() - start

    # GPU benchmark (if available)
    if torch.cuda.is_available():
        a_gpu = a.to(device)
        b_gpu = b.to(device)
        _ = torch.matmul(a_gpu, b_gpu)  # warm-up to stabilize performance
        torch.cuda.synchronize()
        start = time.time()
        _ = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
    else:
        gpu_time = None

    # Display timing results
    st.write(f"🧠 CPU Time: `{cpu_time:.4f}s`")
    if gpu_time:
        st.write(f"💻 GPU Time: `{gpu_time:.4f}s`")
        speedup = cpu_time / gpu_time
        st.write(f"⚡ Speedup (CPU / GPU): `{speedup:.2f}x`")

# Section 2: Transformer Inference Benchmark
st.header("2️⃣ Transformer Inference")
# Input text for the transformer model
text = st.text_input("Sample Text", value="The quick brown fox jumps over the lazy dog")

# When the user clicks the button, the benchmark runs
if st.button("Run Transformer Benchmark"):
    # Load pre-trained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)

    # Tokenize the text and move to the active device
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Run the model and time the inference
    with torch.no_grad():
        start = time.time()
        _ = model(**inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_time = time.time() - start

    # Display inference time
    st.write(f"⏱ Transformer Inference Time: `{inference_time:.4f}s`")

# Section 3: Placeholder for future image generation benchmark
st.header("3️⃣ Image Generation (Coming Soon)")
st.info("Stable Diffusion or small GAN will be added in the next version.")

# Section footer
st.markdown("---")
st.caption("Built with PyTorch, HuggingFace, and Streamlit")
