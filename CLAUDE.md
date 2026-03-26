# CLAUDE.md - Sionna 2.10 (PyTorch) Development Guide

## Tech Stack
- Engine: Sionna 2.10.0+ (PyTorch-based)
- Computing: PyTorch 2.0+ (torch.Tensor, torch.nn.Module)
- Acceleration: CUDA required. Hardware-accelerated Ray Tracing via RT Cores.
- Core Modules: sionna.rt, sionna.channel, sionna.fec, sionna.mapping

## Model Adaptation (Qwen3-Coder-Next)
- Paradigm: Enforce nn.Module class-based structure for all simulation links.
- VRAM Efficiency: Use torch.no_grad() for inference/testing to save local GPU memory.
- Shape Documentation: Annotate Tensor shape transitions before permute, reshape, or view.

## Common Commands
- Check Version: python -c "import sionna; print(sionna.__version__)"
- Run Sim: python main.py
- Test: pytest tests/
- Monitor GPU: nvidia-smi or torch.cuda.memory_summary()

## Coding Standards (Sionna 2.0 PyTorch)
- Dtypes: complex64 for baseband signals; float32 for real gains/metrics.
- Device: Always use device = torch.device("cuda" if torch.cuda.is_available() else "cpu").
- Modularity: Components (Encoders, Mappers, Channels) must be sub-classes of nn.Module.
- Pathing: Use os.path.abspath for .xml or .ply scene files to ensure server compatibility.

## Workflow
1. Scene: scene = sionna.rt.load_scene(...) to initialize 3D environments.
2. Config: Define Antenna, Transmitter, and Receiver with proper polarization.
3. Propagation: Execute paths = scene.compute_paths(...) for multipath data.
4. Analysis: Calculate BER/FER or Spectral Efficiency; log via matplotlib or tensorboard.