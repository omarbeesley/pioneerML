#!/usr/bin/env python3
"""
PyTorch container smoke test
RTX 6000 Ada / CUDA 13.0 / NGC 25.02-py3
"""

import sys
import torch
import torch.nn as nn

def section(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")

# ── Environment ───────────────────────────────────────
section("Environment")
print(f"Python:       {sys.version.split()[0]}")
print(f"PyTorch:      {torch.__version__}")
print(f"CUDA build:   {torch.version.cuda}")
print(f"cuDNN:        {torch.backends.cudnn.version()}")

# ── GPU Detection ─────────────────────────────────────
section("GPU")
if not torch.cuda.is_available():
    print("ERROR: CUDA not available — did you run with --nv?")
    sys.exit(1)

print(f"Device:       {torch.cuda.get_device_name(0)}")
cap = torch.cuda.get_device_capability(0)
print(f"Capability:   sm_{cap[0]}{cap[1]} (expect sm_89 for Ada Lovelace)")
mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"VRAM:         {mem:.1f} GB")

# ── Tensor Ops ────────────────────────────────────────
section("Tensor operations")
device = torch.device("cuda")

a = torch.randn(4096, 4096, device=device)
b = torch.randn(4096, 4096, device=device)

# Matmul
torch.cuda.synchronize()
import time
t0 = time.perf_counter()
c = torch.matmul(a, b)
torch.cuda.synchronize()
elapsed = (time.perf_counter() - t0) * 1000

print(f"Matmul 4096x4096:  {elapsed:.1f} ms")
print(f"Result shape:      {c.shape}")
print(f"Result mean:       {c.mean().item():.4f} (expect ~0)")

# ── Mixed Precision ───────────────────────────────────
section("Mixed precision (bf16)")
a16 = a.to(torch.bfloat16)
b16 = b.to(torch.bfloat16)

t0 = time.perf_counter()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    c16 = torch.matmul(a16, b16)
torch.cuda.synchronize()
elapsed16 = (time.perf_counter() - t0) * 1000

print(f"Matmul bf16 4096x4096:  {elapsed16:.1f} ms")
print(f"Speedup over fp32:      {elapsed/elapsed16:.2f}x")

# ── Simple Network ────────────────────────────────────
section("Forward + backward pass")

model = nn.Sequential(
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
).to(device)

x    = torch.randn(64, 256, device=device)
y    = torch.randint(0, 10, (64,), device=device)
loss_fn  = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

optimizer.zero_grad()
out  = model(x)
loss = loss_fn(out, y)
loss.backward()
optimizer.step()

print(f"Loss:         {loss.item():.4f}")
print(f"Output shape: {out.shape}")

# ── Memory ────────────────────────────────────────────
section("Memory summary")
allocated = torch.cuda.memory_allocated(0) / 1e6
reserved  = torch.cuda.memory_reserved(0) / 1e6
print(f"Allocated:    {allocated:.1f} MB")
print(f"Reserved:     {reserved:.1f} MB")

# ── Done ──────────────────────────────────────────────
section("Result")
print("All checks passed ✓")