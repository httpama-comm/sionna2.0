# Sionna - Technical Context

## Technology Stack

### Core Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| Python | 3.11+ | Programming language |
| PyTorch | 2.9+ | Deep learning framework, automatic differentiation |
| NumPy | 2.2.6+ | Numerical computing |
| SciPy | 1.15.3+ | Scientific computations |
| Matplotlib | 3.10.8+ | Visualization and plotting |
| h5py | 3.15.1+ | HDF5 file handling |
| sionna-rt | Separate package | Ray tracing engine |

### Optional Dependencies

**Documentation:**
- ipython, nbsphinx==0.9.8, pydata-sphinx-theme==0.16.1
- sphinx==9.1.0, sphinx-copybutton==0.5.2, sphinxcontrib-bibtex

**Testing:**
- pytest, pytest-xdist

## Architecture Overview

### Package Structure

```
sionna/
├── __init__.py              # Main package initialization
│
├── phy/                     # Physical Layer Module
│   ├── __init__.py
│   ├── block.py            # Base Block class for signal processing chains
│   ├── config.py           # Configuration and precision settings
│   ├── constants.py        # Physical constants and parameters
│   ├── mapping.py          # Modulation/demodulation mappings
│   ├── object.py           # Base Object class with shared functionality
│   │
│   ├── channel/            # Channel models
│   │   ├── awgn.py         # Additive White Gaussian Noise
│   │   ├── ofdm_channel.py # OFDM-specific channels
│   │   ├── time_channel.py # Time-varying channels
│   │   ├── rayleigh_block_fading.py
│   │   ├── cir_dataset.py  # Channel impulse response datasets
│   │   └── optical/        # Optical channel models
│   │       └── tr38901/    # 3GPP TR 38901 optical channels
│   │
│   ├── fec/                # Forward Error Correction Codes
│   │   ├── ldpc.py         # LDPC codes (5G NR compatible)
│   │   ├── polar.py        # Polar codes
│   │   ├── turbo.py        # Turbo codes
│   │   ├── conv.py         # Convolutional codes
│   │   └── crc.py          # CRC checksums
│   │
│   ├── mimo/               # MIMO systems and detection
│   │   ├── detection.py    # Detection algorithms (MMSE, ZF, etc.)
│   │   └── precoding.py    # Precoding techniques
│   │
│   ├── ofdm/               # OFDM modulation/demodulation
│   │   ├── ofdm.py         # OFDM modulator/demodulator
│   │   └── cp.py           # Cyclic prefix handling
│   │
│   ├── signal/             # Signal processing utilities
│   │   ├── pulse_shaping.py
│   │   └── interpolation.py
│   │
│   ├── nr/                 # 5G New Radio specific implementations
│   │   └── pusch.py        # Physical Uplink Shared Channel
│   │
│   └── utils/              # Utility functions
│       ├── linalg.py       # Linear algebra utilities
│       ├── metrics.py      # Performance metrics (BER, BLER)
│       ├── plotting.py     # Visualization helpers
│       └── tensors.py      # Tensor operations
│
├── sys/                     # System Level Module
│   ├── __init__.py
│   ├── effective_sinr.py   # EESM - Effective SINR calculation
│   ├── link_adaptation.py  # Link adaptation algorithms
│   ├── phy_abstraction.py  # PHY abstraction for system level
│   ├── power_control.py    # Power control mechanisms
│   ├── scheduling.py       # User scheduling algorithms
│   ├── topology.py         # Network topology (hexagonal grids)
│   └── utils.py            # System-level utilities
│
├── rt/                      # Ray Tracer Module (separate package sionna-rt)
│   └── (implemented in C++ with Python bindings via Dr.Jit)
│
└── sys/                     # Helper modules
    ├── bler_tables/        # Pre-computed BLER lookup tables
    └── esm_params/         # ESM parameter configurations
```

## Design Patterns

### 1. Block-Based Signal Processing Chain

The `Block` class in `sionna.phy.block` provides a composable framework for building signal processing chains:

```python
# Example pattern
chain = MyReceiver()
output = chain(input)
```

Each block implements:
- `__call__()`: Main processing method
- `reset()`: Reset internal state
- Shape validation and broadcasting

### 2. Differentiable Layers

All components leverage PyTorch's autograd for end-to-end differentiability:
- Custom nn.Module subclasses where applicable
- Native tensor operations maintaining gradient flow
- Support for both training and inference modes

### 3. Precision Management

The `config` module provides precision control:
```python
from sionna.phy import config, Precision

# Set global precision
config.precision = Precision.FLOAT32  # or FLOAT16, COMPLEX64, etc.
```

## Key Technical Features

### Automatic Differentiation

- All operations are PyTorch tensor-based
- Full gradient support through entire transmission chain
- Enables:
  - Neural receiver training
  - End-to-end system optimization
  - Gradient-based parameter tuning

### GPU Acceleration

- Native CUDA support via PyTorch tensors
- Dr.Jit backend for ray tracing (Sionna RT)
- Device-agnostic code (`torch.device` management)

### Channel Modeling

**Wireless Channels:**
- AWGN, Rayleigh, Rician fading
- OFDM-specific frequency-selective channels
- 3GPP TR 38901 channel models (DL/UL)
- Customizable impulse responses

**Optical Channels:**
- Lumped amplification models
- Fiber propagation effects
- 3GPP optical specifications

### FEC Code Implementation

| Code Type | Features |
|-----------|----------|
| LDPC | 5G NR compatible, configurable matrices |
| Polar | Successive cancellation, list decoding |
| Turbo | Convolutional-based turbo codes |
| Convolutional | Viterbi decoding support |
| CRC | Multiple polynomial options (6, 11, 16, 24A/B/C) |

## Development Environment

### Requirements for GPU Support

- NVIDIA CUDA drivers installed
- PyTorch with CUDA support (`torch.cuda.is_available()`)
- For Sionna RT: LLVM backend or CUDA backend via Dr.Jit

### Building Documentation

```bash
cd doc
pip install '.[doc]'
make html
make serve  # Serve at http://localhost:8000/sionna/
```

### Running Tests

```bash
cd test/unit
pytest  # Or from root: pytest test/unit
```

## Version Information

- **Current Version**: 2.0.0
- **License**: Apache-2.0
- **Repository**: https://github.com/NVlabs/sionna
- **Documentation**: https://nvlabs.github.io/sionna/

## Installation Options

1. **Full package** (with RT): `pip install sionna`
2. **RT only**: `pip install sionna-rt`
3. **Without RT**: `pip install sionna-no-rt`
4. **From source**: Clone with submodules, run pip install

## Constraints and Considerations

### System Requirements

- **OS**: Recommended Ubuntu 24.04 (earlier versions may work)
- **Python**: 3.11+
- **PyTorch**: 2.9+ (minimum for compatibility)
- **RAM**: Varies by simulation size; GPU recommended for large simulations

### Known Limitations

1. Sionna RT requires additional setup (Mitsuba dependencies)
2. Some advanced channel models may need custom configuration
3. Python virtual environment or Docker recommended for isolation

### Future Considerations

- Enhanced 6G channel model support
- Expanded neural network architectures
- Additional FEC code families
- Improved multi-GPU support