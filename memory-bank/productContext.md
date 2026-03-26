# Sionna - Product Context

## What is Sionna?

Sionna is an **open-source Python-based library for research on communication systems**, developed and maintained by NVIDIA. It provides a comprehensive framework for simulating, analyzing, and optimizing wireless and optical communication systems at both the physical layer (PHY) and system level (SYS).

### Core Purpose

Sionna exists to enable researchers and engineers to:
- **Rapidly prototype** new communication system designs
- **Simulate realistic channel conditions** including MIMO, OFDM, and complex propagation effects
- **Integrate deep learning** into traditional communication system design
- **Optimize systems using gradient-based methods** through automatic differentiation
- **Bridge the gap** between physical-layer simulations and system-level analysis

## Key Components

### 1. Sionna PHY - Physical Layer Simulator
A link-level simulator for wireless and optical communication systems featuring:
- **Modulation schemes**: QAM, PSK, and custom modulations
- **Channel models**: AWGN, Rayleigh fading, OFDM channels, 3GPP TR 38901 standards
- **MIMO systems**: Spatial multiplexing, beamforming, detection algorithms
- **OFDM**: Complete orthogonal frequency-division multiplexing support
- **Forward Error Correction (FEC)**: LDPC, Polar codes, Turbo codes, Convolutional codes, CRC
- **5G NR support**: 3GPP New Radio physical layer specifications

### 2. Sionna SYS - System Level Simulator
A system-level simulator based on physical-layer abstraction for network-level analysis:
- **Topology models**: Hexagonal grid deployments for cellular networks
- **Link adaptation**: Inner and outer loop algorithms with EESM (Exponential Effective SINR Mapping)
- **PHY Abstraction**: Efficient mapping from link-level to system-level simulations
- **Power control**: Open-loop uplink and downlink fair power control
- **Scheduling**: Proportional Fair Scheduler (PFS) for multi-user MIMO

### 3. Sionna RT - Ray Tracer
A lightning-fast, differentiable ray tracer for radio propagation modeling:
- Built on Mitsuba 3 renderer with Dr.Jit backend
- Hardware-accelerated (GPU support via CUDA)
- Differentiable simulation for gradient-based optimization
- Realistic multipath channel generation

## Why Sionna?

### Problems It Solves

1. **Differentiability**: Unlike traditional simulators, Sionna supports automatic differentiation through PyTorch, enabling end-to-end gradient-based optimization of communication systems.

2. **Hardware Acceleration**: Built on NVIDIA's ecosystem, providing GPU acceleration for computationally intensive simulations.

3. **Standards Compliance**: Implements 3GPP TR 38901 channel models and NR specifications for realistic 5G simulations.

4. **Unified Framework**: Combines PHY-level detail with SYS-level efficiency through abstraction techniques.

5. **Research-Focused**: Designed specifically for academic and industrial research, not production deployment.

## Target Users

- **Academic Researchers**: Studying new modulation schemes, channel coding, MIMO architectures
- **Telecommunications Engineers**: Designing 5G/6G systems, optimizing network performance
- **ML/AI Practitioners**: Developing neural receivers, end-to-end learning for communications
- **Industry R&D Teams**: Prototyping next-generation wireless technologies

## Key Features Summary

| Feature | Description |
|---------|-------------|
| Differentiable Simulation | Automatic differentiation through entire communication chain |
| GPU Acceleration | Hardware-accelerated simulations via CUDA/Dr.Jit |
| 5G NR Support | Complete 3GPP New Radio PHY implementation |
| Multiple FEC Codes | LDPC, Polar, Turbo, Convolutional, CRC |
| Channel Models | AWGN, Rayleigh, CDL (3GPP), Optical channels |
| MIMO Systems | SU-MIMO and MU-MIMO with various detection algorithms |
| Neural Networks Integration | Custom layers for deep learning-based receivers |

## Relationship to Other Tools

- **MATLAB Communications Toolbox**: Sionna offers similar functionality but with automatic differentiation and GPU acceleration
- **NS-3**: Sionna focuses on physical layer while NS-3 is network protocol focused
- **GNU Radio**: Sionna is research-oriented with differentiability; GNU Radio emphasizes real-time signal processing

## Project Status

This is version 2.0.0 of Sionna, representing a significant evolution from previous versions with enhanced differentiability, improved performance, and expanded feature set.