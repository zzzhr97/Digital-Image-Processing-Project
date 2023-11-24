import torch

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    print("CUDA is available! You can use GPU.")
else:
    print("CUDA is not available. Switching to CPU.")

# Print PyTorch version
print("PyTorch version:", torch.__version__)
