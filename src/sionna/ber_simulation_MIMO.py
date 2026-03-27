import torch
from sionna.channel import (
    IndependentChannel,
    RMA,
    UMa,
    CDL
)
from sionna.mimo import (
    MIMOReceiver,
    MIMOTransmitter
)
from sionna.utils import (
    snr2ber,
    ber2snr
)
import matplotlib.pyplot as plt

# Parameters
num_bits = 1000
EbN0_dB = torch.arange(0, 25, 0.5)
snr_dB = ber2snr(snr2ber(EbN0_dB), num_bits)

# SISO channel model
siso_channel = IndependentChannel(num_tx=1, num_rx=1)

# MIMO channel models
mimo_2x2_channel = IndependentChannel(num_tx=2, num_rx=2)
mimo_3x3_channel = IndependentChannel(num_tx=3, num_rx=3)
mimo_4x4_channel = IndependentChannel(num_tx=4, num_rx=4)

def simulate_mimo_system(channel_model):
    tx = MIMOTransmitter()
    rx = MIMOReceiver()

    # Simulate transmission and reception
    y, _ = channel_model(tx())
    ber = snr2ber(snr_dB)

    return ber

# Calculate BER for each MIMO scenario
siso_ber = simulate_mimo_system(siso_channel)
mimo_2x2_ber = simulate_mimo_system(mimo_2x2_channel)
mimo_3x3_ber = simulate_mimo_system(mimo_3x3_channel)
mimo_4x4_ber = simulate_mimo_system(mimo_4x4_channel)

plt.figure()
plt.semilogy(EbN0_dB, siso_ber, label='SISO')
plt.semilogy(EbN0_dB, mimo_2x2_ber, label='2x2 MIMO')
plt.semilogy(EbN0_dB, mimo_3x3_ber, label='3x3 MIMO')
plt.semilogy(EbN0_dB, mimo_4x4_ber, label='4x4 MIMO')

plt.xlabel('Eb/N0 (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.legend()
plt.title('MIMO BER vs. Eb/N0 Performance')
plt.grid(True)
plt.show()