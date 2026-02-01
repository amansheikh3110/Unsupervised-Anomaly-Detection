import torch
print("=" * 50)
print("ML Environment Setup Verification")
print("=" * 50)

# PyTorch & CUDA
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Test tensor on GPU
if torch.cuda.is_available():
    x = torch.randn(3, 3).cuda()
    print(f"\nGPU Tensor Test: SUCCESS")
    print(f"Tensor device: {x.device}")

print("\n" + "=" * 50)
print("All systems ready for I-JEPA project!")
print("=" * 50)
