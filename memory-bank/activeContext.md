# Sionna - Active Context

## Current Session Focus

**Task**: Initialize memory bank based on project structure analysis

**Date**: March 26, 2026

## Project Overview (Current Understanding)

### What is Sionna?

Sionna v2.0.0 is an NVIDIA-developed open-source Python library for communication systems research. It provides differentiable simulation capabilities through PyTorch integration, enabling gradient-based optimization of communication system designs.

### Key Insights Discovered

1. **Three Main Components**:
   - **PHY**: Physical layer link-level simulator (wireless & optical)
   - **SYS**: System-level simulator using PHY abstraction
   - **RT**: Ray tracer for radio propagation (separate sionna-rt package)

2. **Core Differentiator**: Automatic differentiation through entire simulation chain, enabling end-to-end training of neural networks and gradient-based optimization.

3. **Target Domain**: 5G/6G wireless communication research with strong emphasis on:
   - MIMO systems (SU-MIMO, MU-MIMO)
   - OFDM modulation
   - Channel coding (LDPC, Polar, Turbo codes per 3GPP specs)
   - Neural receiver architectures

## Project Structure Summary

```
sionna/
├── src/sionna/          # Main package
│   ├── phy/            # Physical layer module
│   │   ├── channel/    # Channel models (AWGN, Rayleigh, OFDM, 3GPP)
│   │   ├── fec/        # Forward error correction codes
│   │   ├── mimo/       # MIMO systems and detection
│   │   ├── ofdm/       # OFDM modulation/demodulation
│   │   ├── nr/         # 5G New Radio implementations
│   │   └── signal/     # Signal processing utilities
│   │
│   ├── sys/            # System level module
│   │   ├── topology.py      # Hexagonal grid networks
│   │   ├── link_adaptation.py
│   │   ├── phy_abstraction.py
│   │   └── scheduling.py
│   │
│   └── rt/             # Ray tracer (C++ with Dr.Jit bindings)
│
├── tutorials/          # Jupyter notebook examples
│   ├── phy/            # Physical layer tutorials
│   └── sys/            # System level tutorials
│
├── test/unit/          # Unit tests
└── doc/source/         # Sphinx documentation
```

## Recent Observations

### Test Files Found
- `test_ber_conv_qam_qwen.py` - BER simulation for convolutional codes with QAM
- `ber_simulation_comprehensive.py` - Comprehensive BER simulation tests

These suggest active work on:
- Link-level performance validation
- Neural network-based receiver research (Qwen reference)

### Documentation Structure
- Extensive RST documentation in `doc/source/`
- Separate sections for PHY, SYS, and RT modules
- API reference generation configured
- Tutorial notebooks integrated with Sphinx

## Current Work Status

### Completed
- [x] Analyzed project structure
- [x] Read README.md and pyproject.toml
- [x] Created memory-bank directory
- [x] Documented product context (purpose, components, target users)
- [x] Documented technical context (architecture, dependencies, patterns)

### In Progress
- Creating activeContext.md - documenting current session and findings
- Creating progress.md - tracking project state and TODOs

## Key Decisions & Considerations

1. **Memory Bank Structure**: Following .clinerules specification with 4 core files:
   - productContext.md (why the project exists)
   - techContext.md (how it's built)
   - activeContext.md (current focus)
   - progress.md (what's done/remaining)

2. **Documentation Depth**: Balancing comprehensive coverage with actionable insights for future work sessions.

## Next Steps

1. Complete activeContext.md documentation
2. Create progress.md with TODO list
3. Mark task as complete

## Important Patterns Identified

### Block-Based Processing Chain
```python
from sionna.phy import Block, Config

class MyReceiver(Block):
    def __call__(self, x):
        # Process and return tensor
        return output
```

### Precision Configuration
```python
from sionna.phy import config, Precision
config.precision = Precision.COMPLEX64  # For complex baseband signals
```

### Usage Pattern
```python
import sionna as snn

# Access submodules
phy = snn.phy
sys = snn.sys
rt = snn.rt
```

## Notes for Future Sessions

- **GPU Setup**: Verify CUDA availability before large simulations
- **RT Package**: Separate installation (sionna-rt) with Mitsuba dependencies
- **Testing Framework**: Uses pytest in test/unit directory
- **Version**: Currently at 2.0.0 - significant updates from previous versions

## Open Questions

1. What specific research area is the user focusing on? (PHY, SYS, RT, or all?)
2. Are there any custom modifications to the base Sionna codebase?
3. What are the current development priorities or active feature requests?