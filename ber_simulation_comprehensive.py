# Comprehensive BER vs EbN0 Simulation for Sionna 2.0
# Simulates different modulation and coding rate combinations with proper puncturing support

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

MODULATION_CONFIGS = [
    ("QPSK", 2),      # QPSK: 2 bits per symbol
    ("16-QAM", 4),    # 16-QAM: 4 bits per symbol
    ("64-QAM", 6),    # 64-QAM: 6 bits per symbol
]

CODING_CONFIGS = [
    ("1/2", 0.5, "No puncturing"),      
    ("3/4", 0.75, "Systematic-first"),   
    ("5/6", 0.833, "Systematic-first"),  
]

CONV_POLY = [0o17, 0o15]  # Octal polynomials for rate-1/2 convolutional code

EBNO_DB_MIN = -6.0
EBNO_DB_MAX = 8.0
EBNO_DB_STEP = 1.0
MAX_BITS_TESTED = int(1e7)
MIN_BER_TARGET = 1e-4
BATCH_SIZE = 256
NUM_FRAMES_MAX = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ============================
# ConvPuncturer Class - Systematic-first puncturing pattern
# ============================

class ConvPuncturer:
    """
    卷积码 puncturing/depuncturing 工具类
    
    Puncture Pattern Strategy (Systematic-first):
    - Always preserve all systematic bits (information bits)
    - Selectively remove parity bits based on target rate
    """
    
    @staticmethod
    def get_pattern(target_rate, k, pattern_type="systematic_first", encoded_length=None, device=None):
        """Generate puncturing mask for given target rate"""
        return ConvPuncturer._systematic_first_pattern(target_rate, k, encoded_length, device)
    
    @staticmethod
    def _systematic_first_pattern(target_rate, k, encoded_length=None, device=None):
        """
        Generate puncturing mask using systematic-first strategy
        
        For rate-1/2 output of length 2k (or more with termination):
        - R=0.5: Keep all bits (no puncturing)
        - R=0.75: Keep all systematic + every other parity bit
        - R=0.833: Keep all systematic + one in three parity bits
        """
        total_len = encoded_length if encoded_length is not None else 2 * k
        keep_mask = torch.zeros(total_len, dtype=torch.bool, device=device)
        
        # Always preserve systematic bits (even indices: 0, 2, 4, ...)
        keep_mask[::2] = True
        
        if target_rate == 0.5:
            keep_mask[1::2] = True
            
        elif target_rate == 0.75:
            for i in range(1, total_len, 2):
                if ((i - 1) // 2) % 2 == 0:
                    keep_mask[i] = True
            
        elif target_rate >= 5/6 - 0.01:
            for i in range(1, total_len, 2):
                if ((i - 1) // 2) % 3 == 0:
                    keep_mask[i] = True
        
        return keep_mask
    
    @staticmethod
    def puncture(encoded_bits, keep_mask):
        """Puncture encoded bits according to the mask.
        
        Returns:
            punctured_bits: Bits that will be transmitted (length = sum(keep_mask))
            n_punctured: Number of bits after puncturing
        """
        punctured_bits = encoded_bits[:, keep_mask]
        n_punctured = punctured_bits.shape[1]
        return punctured_bits, n_punctured
    
    @staticmethod
    def depuncture(llrs_punctured, keep_mask, valid_count, input_device):
        """
        Depuncture LLRs by inserting zeros at punctured positions.
        
        The received LLRs (llrs_punctured) may have symbol-alignment padding at the end.
        Only the first `valid_count` LLRs correspond to actual transmitted bits.
        
        Args:
            llrs_punctured: Received LLRs - length = sum(keep_mask) + symbol_alignment_padding
                           The first valid_count elements are from actual transmission,
                           remaining elements are padding (should be treated as erasures)
            keep_mask: Boolean mask indicating which positions were transmitted vs punctured.
                      Length must equal total expected LLR length before symbol alignment.
                      True positions will receive the valid_count LLRs in order.
            valid_count: Number of valid LLRs from actual transmission (should be <= sum(keep_mask))
                        These correspond to positions where keep_mask is True, in order
            input_device: Device for output tensor
        
        Returns:
            depunct_llrs: Full-length LLRs with zeros at punctured/erased positions
                         and received values at transmitted positions
        """
        batch_size = llrs_punctured.shape[0]
        total_len = keep_mask.shape[-1]
        
        # Create full-length zero tensor (LLR=0 means complete uncertainty)
        depunct_llrs = torch.zeros(
            batch_size, total_len,
            dtype=llrs_punctured.dtype,
            device=input_device
        )
        
        if keep_mask.sum() > 0:
            # Get indices of positions that were transmitted (True in mask)
            # Handle both 1D and 2D masks correctly for torch.nonzero with as_tuple=True
            nonzero_result = torch.nonzero(keep_mask, as_tuple=True)
            
            if len(nonzero_result) == 1:
                # 1D mask case - the result is (indices,) tuple
                true_indices = nonzero_result[0]
            else:
                # 2D or higher mask case - get indices along last dimension
                true_indices = nonzero_result[-1]
            
            if valid_count > 0 and len(true_indices) > 0:
                # Copy only the valid LLRs to their original positions
                # Ensure we don't copy more than available or more than True positions
                n_to_copy = min(valid_count, int(keep_mask.sum()), llrs_punctured.shape[-1])
                depunct_llrs[:, true_indices[:n_to_copy]] = llrs_punctured[:, :n_to_copy]
        
        return depunct_llrs


# ============================
# Helper Functions
# ============================

def calculate_noise_variance(ebno_db, coding_rate, num_bits_per_symbol):
    """Calculate noise variance for AWGN channel."""
    ebno_lin = 10 ** (ebno_db / 10.0)
    noise_var = 1.0 / (2.0 * coding_rate * num_bits_per_symbol * ebno_lin)
    return noise_var


# ============================
# Initialize Sionna Components
# ============================

print("\nInitializing convolutional encoder and decoder...")

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

dummy_bits = torch.randint(0, 2, (1, 100), dtype=torch.float32, device=DEVICE)
_ = conv_enc_1_2(dummy_bits)

print("Initialization complete.")


# ============================
# Simulation Results Storage
# ============================

results = {}


# ============================
# Main Simulation Loop
# ============================

print("\n" + "="*60)
print("Starting BER vs EbN0 Simulation")
print("="*60)

for mod_name, num_bits_per_symbol in MODULATION_CONFIGS:
    print(f"\n--- Modulation: {mod_name} ({num_bits_per_symbol} bits/symbol) ---")
    
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
    
    for rate_str, coding_rate, puncture_desc in CODING_CONFIGS:
        print(f"\n  Coding: {rate_str} ({puncture_desc})")
        
        ebno_vals = []
        coded_ber_vals = []
        
        k_info = 1000
        
        for ebno_db in tqdm(
            np.arange(EBNO_DB_MIN, EBNO_DB_MAX + EBNO_DB_STEP/2, EBNO_DB_STEP),
            desc=f"    {mod_name} @ R={rate_str}",
            leave=False
        ):
            coded_errors = 0
            coded_bits_tested = 0
            
            for _ in range(NUM_FRAMES_MAX):
                if coded_bits_tested >= MAX_BITS_TESTED:
                    break
                
                info_bits = torch.randint(
                    0, 2, (BATCH_SIZE, k_info), dtype=torch.float32, device=DEVICE
                )
                
                # Step 1: Encode with rate-1/2 encoder
                encoded_bits = conv_enc_1_2(info_bits)
                n_encoded = encoded_bits.shape[1]
                
                # Step 2: Generate puncturing mask for encoded length
                keep_mask_base = ConvPuncturer.get_pattern(
                    coding_rate, k_info, "systematic_first", 
                    encoded_length=n_encoded, device=DEVICE
                )
                
                # Step 3: Puncture according to pattern
                punctured_bits, n_punctured = ConvPuncturer.puncture(encoded_bits, keep_mask_base)
                
                # Step 4: Pad for symbol alignment (padded bits treated as erasures)
                n_symbols_tx = (n_punctured + num_bits_per_symbol - 1) // num_bits_per_symbol
                total_bits_tx = n_symbols_tx * num_bits_per_symbol
                
                if n_punctured < total_bits_tx:
                    pad_size_tx = total_bits_tx - n_punctured
                    punctured_bits = torch.cat([
                        punctured_bits,
                        torch.zeros(BATCH_SIZE, pad_size_tx, device=DEVICE)
                    ], dim=1)
                    
                    # Create transmission mask for symbol-aligned bits only
                    # True = transmitted (after puncturing), False = padded (erasure)
                    tx_mask_aligned = torch.cat([keep_mask_base, torch.zeros(pad_size_tx, dtype=torch.bool, device=DEVICE)])
                else:
                    punctured_bits = punctured_bits[:, :total_bits_tx]
                    if total_bits_tx > n_encoded:
                        # Pad the mask to match symbol alignment
                        pad_size = total_bits_tx - n_encoded
                        tx_mask_aligned = torch.cat([keep_mask_base, torch.zeros(pad_size, dtype=torch.bool, device=DEVICE)])
                    else:
                        tx_mask_aligned = keep_mask_base[:total_bits_tx]  # Fixed: use 1D indexing for 1D mask
                
                # Step 5: Map to symbols and transmit through AWGN channel
                symbols = modulator(punctured_bits)
                noise_var = calculate_noise_variance(ebno_db, coding_rate, num_bits_per_symbol)
                y = AWGN(precision="single")(symbols, noise_var)
                
                # Step 6: Demap to LLRs (length matches total_bits_tx after symbol alignment)
                llrs = demapper(y, noise_var)
                
                # Step 7: Depuncture - map received LLRs back to original encoded length
                # llrs from demapper has length total_bits_tx (after symbol alignment padding)
                # The first n_punctured LLRs correspond to positions where keep_mask_base is True
                depunct_llrs = ConvPuncturer.depuncture(llrs, keep_mask_base, n_punctured, input_device=DEVICE)
                
                # Step 8: Decode with Viterbi decoder (expects original encoded length)
                decoded_bits = viterbi_dec_1_2(depunct_llrs)
                
                # Step 9: Calculate BER (compare to original info bits)
                min_len = min(decoded_bits.shape[1], info_bits.shape[1])
                errors = (decoded_bits[:, :min_len] != info_bits[:, :min_len]).sum().item()
                
                coded_errors += errors
                coded_bits_tested += min_len * BATCH_SIZE
                
                if coded_bits_tested > 1e5 and coded_errors / coded_bits_tested < MIN_BER_TARGET:
                    break
            
            coded_ber = coded_errors / coded_bits_tested if coded_bits_tested > 0 else 0.0
            
            ebno_vals.append(ebno_db)
            coded_ber_vals.append(coded_ber)
            
            print(f"      EbN0={ebno_db:4.1f}dB -> BER = {coded_ber:.2e}")
            
            if coded_ber < MIN_BER_TARGET and ebno_db > 0:
                break
        
        key = (mod_name, rate_str)
        results[key] = {'ebno': ebno_vals, 'coded_ber': coded_ber_vals}


# ============================
# Plot Results
# ============================

print("\n" + "="*60)
print("Generating Plot")
print("="*60)

plt.figure(figsize=(12, 8))

mod_colors = {"QPSK": 'b', "16-QAM": 'g', "64-QAM": 'r'}
rate_markers = {"0.5": 'o', "0.75": '^', "0.833": 's'}

for (mod_name, rate_str), data in results.items():
    ebno_vals = data['ebno']
    coded_ber_vals = data['coded_ber']
    
    non_zero_mask = [ber > 0 for ber in coded_ber_vals]
    ebno_plot = [e for e, m in zip(ebno_vals, non_zero_mask) if m]
    ber_plot = [b for b, m in zip(coded_ber_vals, non_zero_mask) if m]
    
    color = mod_colors.get(mod_name, 'k')
    marker = rate_markers.get(rate_str, 'o')
    
    if ber_plot:
        label = f"{mod_name} R={rate_str}"
        fmt_string = color + marker + '-'  # 'bo-' means blue line with circle markers
        plt.semilogy(ebno_plot, ber_plot, fmt_string, 
                    label=label, linewidth=1.5, markersize=6)

# Add uncoded BER theoretical curves for reference
ebno_db_range = np.arange(EBNO_DB_MIN, EBNO_DB_MAX + 0.5, 0.5)
for mod_name, num_bits in MODULATION_CONFIGS:
    uncoded_ber_theory = []
    
    for ebno_db in ebno_db_range:
        ebno_lin = 10 ** (ebno_db / 10.0)
        M = 2 ** num_bits
        
        if ebno_lin > 0 and M > 1:
            # Convert to tensor for torch operations
            ebno_tensor = torch.tensor(ebno_lin, dtype=torch.float32, device=DEVICE)
            M_tensor = torch.tensor(M, dtype=torch.float32, device=DEVICE)
            ber_tensor = (4/num_bits) * 0.5 * torch.erfc(torch.sqrt(3*ebno_tensor/(M_tensor-1)))
            ber = max(ber_tensor.item(), 1e-6)
        else:
            ber = 1.0
        
        uncoded_ber_theory.append(max(ber, 1e-6))
    
    color = mod_colors.get(mod_name, 'k')
    fmt_string = color + ':'  # e.g., 'b:' for blue dashed line
    plt.semilogy(ebno_db_range, uncoded_ber_theory, fmt_string, 
                label=f"{mod_name} Uncoded (theory)", linewidth=1.5)

plt.xlabel("Eb/N₀ (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("BER vs EbN0: Convolutional Codes with Puncturing\n" +
          "Sionna 2.0 Simulation - Systematic-first Puncture Pattern")
plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.savefig("ber_simulation_comprehensive.png", dpi=150, bbox_inches='tight')

print("\nSimulation complete! Results saved to 'ber_simulation_comprehensive.png'")