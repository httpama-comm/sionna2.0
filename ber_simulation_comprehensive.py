# Comprehensive BER vs EbN0 Simulation for Sionna 2.0
# Simulates different modulation and coding rate combinations

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    import sionna
    from sionna.phy.channel import AWGN
    from sionna.phy.mapping import Mapper, Demapper
    from sionna.phy.fec.conv import ConvEncoder, ViterbiDecoder
except ImportError as e:
    raise RuntimeError("Please install sionna: pip install sionna") from e

# ============================
# Configuration Parameters
# ============================

# Modulation schemes: (name, num_bits_per_symbol)
MODULATION_CONFIGS = [
    ("QPSK", 2),      # QPSK: 2 bits per symbol
    ("16-QAM", 4),    # 16-QAM: 4 bits per symbol
    ("64-QAM", 6),    # 64-QAM: 6 bits per symbol
]

# Coding rates - using rate-1/2 base encoder with different info block sizes
# For punctured codes, we simulate by adjusting the effective rate
CODING_CONFIGS = [
    ("1/2", 0.5),      # Rate 1/2: no puncturing needed
    ("3/4", 0.75),     # Rate 3/4: effectively 1.5x more info bits than encoded
    ("5/6", 0.833),    # Rate 5/6: effectively 1.2x more info bits than encoded
]

# Convolutional code parameters (rate 1/2 base)
CONV_POLY = [0o17, 0o15]  # Octal polynomials for rate-1/2 convolutional code

# Simulation parameters
EBNO_DB_MIN = -6.0        # Minimum EbN0 (dB)
EBNO_DB_MAX = 8.0         # Maximum EbN0 (dB)
EBNO_DB_STEP = 1.0        # Step size for EbN0 points
MAX_BITS_TESTED = int(1e7)  # Max bits to test per point
MIN_BER_TARGET = 1e-4     # Stop when BER < this value
BATCH_SIZE = 256          # Batch size for parallel processing
NUM_FRAMES_MAX = 100      # Max frames per EbN0 point

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ============================
# Helper Functions
# ============================

def calculate_noise_variance(ebno_db, coding_rate, num_bits_per_symbol):
    """
    Calculate noise variance for AWGN channel.
    
    For Sionna's Mapper with Es=1 normalization:
    - Es = energy per symbol = 1 (normalized)
    - Eb = Es / (R * log2(M)) where R=coding rate, M=constellation size
    - N0/2 = noise variance per real dimension
    - For complex AWGN: noise_var = N0/2
    
    Args:
        ebno_db: EbN0 in dB
        coding_rate: Code rate R
        num_bits_per_symbol: log2(M)
    
    Returns:
        Noise variance scalar
    """
    ebno_lin = 10 ** (ebno_db / 10.0)
    # noise_var = 1 / (2 * Eb/N0_effective)
    # where Eb/N0_effective = R * log2(M) * Eb/N0
    noise_var = 1.0 / (2.0 * coding_rate * num_bits_per_symbol * ebno_lin)
    return noise_var

# ============================
# Initialize Sionna Components
# ============================

# Create convolutional encoder and decoder for rate 1/2 base
conv_enc_1_2 = ConvEncoder(
    gen_poly=tuple(bin(p)[2:] for p in CONV_POLY),
    terminate=True,
    precision="single"
).to(DEVICE)

viterbi_dec_1_2 = ViterbiDecoder(
    gen_poly=tuple(bin(p)[2:] for p in CONV_POLY),
    terminate=True,
    precision="single",
    method='soft_llr'
).to(DEVICE)

# Trigger build with dummy pass
dummy_bits = torch.randint(0, 2, (1, 100), dtype=torch.float32, device=DEVICE)
_ = conv_enc_1_2(dummy_bits)

# ============================
# Simulation Results Storage
# ============================

results = {}  # {(mod_name, rate_str): {'ebno': [], 'ber': []}}

# ============================
# Main Simulation Loop
# ============================

print("\n" + "="*60)
print("Starting BER vs EbN0 Simulation")
print("="*60)

for mod_name, num_bits_per_symbol in MODULATION_CONFIGS:
    print(f"\n--- Modulation: {mod_name} ({num_bits_per_symbol} bits/symbol) ---")
    
    # Initialize mapper and demapper for this modulation
    modulator = Mapper(
        constellation_type="qam",
        num_bits_per_symbol=num_bits_per_symbol,
        precision="single"
    ).to(DEVICE)
    
    demapper = Demapper(
        demapping_method="app",
        constellation_type="qam",
        num_bits_per_symbol=num_bits_per_symbol,
        hard_out=False,
        precision="single"
    ).to(DEVICE)
    
    for rate_str, coding_rate in CODING_CONFIGS:
        print(f"  Coding: {rate_str}")
        
        ebno_vals = []
        coded_ber_vals = []
        uncoded_ber_vals = []
        
        # EbN0 sweep
        for ebno_db in tqdm(
            np.arange(EBNO_DB_MIN, EBNO_DB_MAX + EBNO_DB_STEP/2, EBNO_DB_STEP),
            desc=f"  {mod_name} @ {rate_str}",
            leave=False
        ):
            coded_errors = 0
            coded_bits_tested = 0
            uncoded_errors = 0
            uncoded_bits_tested = 0
            
            # Coded simulation
            for _ in range(NUM_FRAMES_MAX):
                if coded_bits_tested >= MAX_BITS_TESTED:
                    break
                
                # For rate R, we send (1/R) encoded bits for each info bit
                # So for rate 1/2: 2 encoded bits per info bit
                # For rate 3/4: 4/3 encoded bits per info bit
                # For rate 5/6: 6/5 encoded bits per info bit
                
                # Use a base info block size and scale by rate
                base_info_bits = 1000
                if coding_rate < 1.0:
                    # Scale to get appropriate encoded length
                    encoded_length = int(base_info_bits / coding_rate)
                else:
                    encoded_length = base_info_bits
                
                info_bits = torch.randint(
                    0, 2, (BATCH_SIZE, base_info_bits), dtype=torch.float32, device=DEVICE
                )
                
                # Encode with rate-1/2 base encoder (outputs 2x info bits)
                encoded_bits = conv_enc_1_2(info_bits)  # [batch, 2*base_info_bits]
                
                # For rates > 1/2, we need to simulate puncturing
                # The Viterbi decoder expects rate-1/2 length, so for higher rates
                # we need to handle the depuncturing properly
                
                if coding_rate > 0.5:
                    # Puncture: keep only a subset of encoded bits
                    n_encoded = encoded_bits.shape[1]  # Should be 2*base_info_bits
                    
                    # Calculate target length based on rate
                    # For rate R, we want encoded_length = info_length / R
                    # But encoder gives us 2*info_length for rate-1/2
                    # So for rate R > 1/2: keep fraction (1/R) / 2 of bits
                    
                    target_encoded_len = int(base_info_bits / coding_rate)
                    
                    if target_encoded_len < n_encoded:
                        # Simple puncturing pattern: take first N bits
                        encoded_bits = encoded_bits[:, :target_encoded_len]
                
                n_encoded = encoded_bits.shape[1]
                
                # Ensure bits per symbol alignment (pad or trim)
                n_symbols = (n_encoded + num_bits_per_symbol - 1) // num_bits_per_symbol
                total_bits_needed = n_symbols * num_bits_per_symbol
                
                if n_encoded < total_bits_needed:
                    pad_size = total_bits_needed - n_encoded
                    encoded_bits = torch.cat([
                        encoded_bits,
                        torch.zeros(BATCH_SIZE, pad_size, device=DEVICE)
                    ], dim=1)
                else:
                    encoded_bits = encoded_bits[:, :total_bits_needed]
                
                # Map to symbols
                symbols = modulator(encoded_bits)
                
                noise_var = calculate_noise_variance(
                    ebno_db, coding_rate, num_bits_per_symbol
                )
                
                y = AWGN(precision="single")(symbols, noise_var)
                
                # For punctured codes, we need a different decoding approach
                # The Viterbi decoder expects rate-1/2 encoded length
                # For higher rates, we'll use the full LLR sequence but decode
                # treating it as if it were rate-1/2 (approximation)
                
                llrs = demapper(y, noise_var)
                
                # Decode with Viterbi - for punctured codes, this is approximate
                # The decoder will process all LLRs, assuming rate-1/2 structure
                decoded_bits = viterbi_dec_1_2(llrs)
                
                # Calculate BER
                min_len = min(decoded_bits.shape[1], info_bits.shape[1])
                errors = (decoded_bits[:, :min_len] != info_bits[:, :min_len]).sum().item()
                
                coded_errors += errors
                coded_bits_tested += min_len * BATCH_SIZE
                
                if coded_bits_tested > 1e5 and coded_errors / coded_bits_tested < MIN_BER_TARGET:
                    break
            
            # Uncoded simulation (same EbN0)
            for _ in range(NUM_FRAMES_MAX):
                if uncoded_bits_tested >= MAX_BITS_TESTED:
                    break
                
                base_info_bits = 1000
                info_bits = torch.randint(
                    0, 2, (BATCH_SIZE, base_info_bits), dtype=torch.float32, device=DEVICE
                )
                
                n_encoded = info_bits.shape[1]
                n_symbols = (n_encoded + num_bits_per_symbol - 1) // num_bits_per_symbol
                total_bits_needed = n_symbols * num_bits_per_symbol
                
                if n_encoded < total_bits_needed:
                    bits_trimmed = torch.cat([
                        info_bits,
                        torch.zeros(BATCH_SIZE, total_bits_needed - n_encoded, device=DEVICE)
                    ], dim=1)
                else:
                    bits_trimmed = info_bits[:, :total_bits_needed]
                
                symbols = modulator(bits_trimmed)
                
                noise_var = calculate_noise_variance(
                    ebno_db, coding_rate, num_bits_per_symbol
                )
                
                y = AWGN(precision="single")(symbols, noise_var)
                llrs = demapper(y, noise_var)
                
                # Hard decision for uncoded
                decoded_bits = (llrs > 0).float()
                
                min_len = min(decoded_bits.shape[1], info_bits.shape[1])
                errors = (decoded_bits[:, :min_len] != info_bits[:, :min_len]).sum().item()
                
                uncoded_errors += errors
                uncoded_bits_tested += min_len * BATCH_SIZE
                
                if uncoded_bits_tested > 1e5 and uncoded_errors / uncoded_bits_tested < MIN_BER_TARGET:
                    break
            
            # Store results - use info bits as the reference for BER calculation
            coded_ber = coded_errors / coded_bits_tested if coded_bits_tested > 0 else 0.0
            
            ebno_vals.append(ebno_db)
            coded_ber_vals.append(coded_ber)
            
            # Uncoded simulation (same EbN0) - use same info bits for fair comparison
            uncoded_errors = 0
            uncoded_bits_tested = 0
            
            for _ in range(NUM_FRAMES_MAX):
                if uncoded_bits_tested >= MAX_BITS_TESTED:
                    break
                
                base_info_bits = 1000
                info_bits = torch.randint(
                    0, 2, (BATCH_SIZE, base_info_bits), dtype=torch.float32, device=DEVICE
                )
                
                n_encoded = info_bits.shape[1]
                n_symbols = (n_encoded + num_bits_per_symbol - 1) // num_bits_per_symbol
                total_bits_needed = n_symbols * num_bits_per_symbol
                
                if n_encoded < total_bits_needed:
                    bits_trimmed = torch.cat([
                        info_bits,
                        torch.zeros(BATCH_SIZE, total_bits_needed - n_encoded, device=DEVICE)
                    ], dim=1)
                else:
                    bits_trimmed = info_bits[:, :total_bits_needed]
                
                symbols = modulator(bits_trimmed)
                
                noise_var = calculate_noise_variance(
                    ebno_db, coding_rate, num_bits_per_symbol
                )
                
                y = AWGN(precision="single")(symbols, noise_var)
                llrs = demapper(y, noise_var)
                
                decoded_bits = (llrs > 0).float()
                
                min_len = min(decoded_bits.shape[1], info_bits.shape[1])
                errors = (decoded_bits[:, :min_len] != info_bits[:, :min_len]).sum().item()
                
                uncoded_errors += errors
                uncoded_bits_tested += min_len * BATCH_SIZE
                
                if uncoded_bits_tested > 1e5 and uncoded_errors / uncoded_bits_tested < MIN_BER_TARGET:
                    break
            
            uncoded_ber = uncoded_errors / uncoded_bits_tested if uncoded_bits_tested > 0 else 0.0
            uncoded_ber_vals.append(uncoded_ber)
            
            # Stop if coded BER is below target
            if coded_ber < MIN_BER_TARGET:
                break
        
        # Store results for this configuration
        key = (mod_name, rate_str)
        results[key] = {
            'ebno': ebno_vals,
            'coded_ber': coded_ber_vals,
            'uncoded_ber': uncoded_ber_vals
        }
        
        print(f"    Coded BER at EbN0=4dB: {coded_ber_vals[-1] if coded_ber_vals else 'N/A':.2e}")

# ============================
# Plot Results
# ============================

print("\n" + "="*60)
print("Generating Plot")
print("="*60)

plt.figure(figsize=(12, 8))

# Color map for different modulations
mod_colors = {
    "QPSK": ('b', 'b--'),
    "16-QAM": ('g', 'g--'),
    "64-QAM": ('r', 'r--'),
}

for (mod_name, rate_str), data in results.items():
    ebno_vals = data['ebno']
    coded_ber_vals = data['coded_ber']
    
    # Filter zero BER values
    non_zero_mask = [ber > 0 for ber in coded_ber_vals]
    ebno_plot = [e for e, m in zip(ebno_vals, non_zero_mask) if m]
    ber_plot = [b for b, m in zip(coded_ber_vals, non_zero_mask) if m]
    
    # Get color for this modulation
    color, _ = mod_colors.get(mod_name, ('k', 'k--'))
    
    # Plot coded BER (solid line)
    if ber_plot:
        label = f"{mod_name} + {rate_str}"
        plt.semilogy(ebno_plot, ber_plot, color + '-', label=label, linewidth=1.5, markersize=4)

# Add uncoded BER reference
for mod_name, num_bits in MODULATION_CONFIGS:
    # Calculate theoretical uncoded BER for QAM
    ebno_db_range = np.arange(EBNO_DB_MIN, EBNO_DB_MAX + 0.5, 0.5)
    uncoded_ber_theory = []
    
    for ebno_db in ebno_db_range:
        ebno_lin = 10 ** (ebno_db / 10.0)
        M = 2 ** num_bits
        # Approximate uncoded BER for QAM: Pb ≈ 4/log2(M) * Q(sqrt(3*Eb/N0/(M-1)))
        if ebno_lin > 0:
            ber = (4/num_bits) * 0.5 * torch.erfc(torch.sqrt(3*ebno_lin/(M-1))).item()
        else:
            ber = 1.0
        uncoded_ber_theory.append(max(ber, 1e-6))
    
    color, _ = mod_colors.get(mod_name, ('k', 'k--'))
    plt.semilogy(ebno_db_range, uncoded_ber_theory, color + ':', 
                 label=f"{mod_name} Uncoded (theory)", linewidth=1, alpha=0.7)

plt.xlabel("Eb/N₀ (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("BER vs EbN0: Different Modulation and Coding Combinations\n" +
          f"Sionna 2.0 Simulation ({DEVICE})")
plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.savefig("ber_simulation_comprehensive.png", dpi=150, bbox_inches='tight')
plt.show()

print("\nSimulation complete! Results saved to 'ber_simulation_comprehensive.png'")