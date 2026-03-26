# simulate_ber_conv_qam.py

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

# ----------------------------
# 配置参数
# ----------------------------
QAM_ORDER = 4                  # 4-QAM → bits per symbol = log2(4) = 2
EBNO_DB_MIN = -6              # EbN0 起始点 (dB)
EBNO_DB_MAX = 4.0              # 结束点（可扩展）
EBNO_DB_STEP = 1.0             # 步长
MAX_EBNO_POINTS = int((EBNO_DB_MAX - EBNO_DB_MIN) / EBNO_DB_STEP) + 1

# 卷积码配置：rate=1/2, K=5, generator polynomials: [17, 15] octal → ["1111", "10101"]
CODE_RATE = 1.0 / 2.0          # Coding rate (R)
POLY = [0o17, 0o15]            # standard (5,1/2) convolutional code
K = 5                          # constraint length
M = K - 1                      # memory order

# Convert octal polynomials to binary string format for Sionna 2.0
def octal_to_binary_str(octal_val):
    """Convert octal polynomial to binary string format (e.g., 0o17 -> '1111')"""
    return bin(octal_val)[2:]  # Remove '0b' prefix

POLY_STR = tuple(octal_to_binary_str(p) for p in POLY)

# Noise variance calculation:
# For AWGN channel with complex signals, noise_var = N0/2 per real dimension
# Sionna's Mapper normalizes Es (symbol energy) to 1 by default
# Relationship: Eb/N0 = (Es/R) / N0 where R is coding rate
# Since Es=1 and N0 = 2 * noise_var (for complex AWGN):
#   Eb/N0 = 1 / (R * N0) = 1 / (R * 2 * noise_var)
#   => noise_var = 1 / (2 * R * Eb/N0)

# 仿真终止条件
MAX_BITS_TESTED = int(1e8)     # 最大测试比特数，超过即停止当前EbN0点
MIN_BER_TARGET = 1e-4          # 目标BER：低于此值自动结束该 EbN0 点

BATCH_SIZE = 256               # 并行帧数（可调）
NUM_FRAMES_PER_EBN0 = 200      # 每个EbN0点最多运行多少轮 batch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ----------------------------
# 初始化模块（GPU/CPU 统一）
# ----------------------------
bits_per_symbol = int(np.log2(QAM_ORDER))

conv_enc = ConvEncoder(
    gen_poly=POLY_STR,
    terminate=True,  # Equivalent to term="TERMINATED"
    precision="single"
)

viterbi_dec = ViterbiDecoder(
    gen_poly=POLY_STR,
    terminate=True,  # Must match encoder
    precision="single",
    method='soft_llr'
)

# Trigger build() by running a dummy forward pass
# In Sionna 2.0, k/n properties are only available after first call
dummy_bits = torch.randint(0, 2, (1, 100), dtype=torch.float32)
_ = conv_enc(dummy_bits)

modulator = Mapper(constellation_type="qam", num_bits_per_symbol=bits_per_symbol, precision="single")
demapper = Demapper(
    demapping_method="app",  # Required in Sionna 2.0
    constellation_type="qam",
    num_bits_per_symbol=bits_per_symbol,
    hard_out=False,  # Changed from 'hard_output'
    precision="single"
)

awgn_ch = AWGN(precision="single")

# ----------------------------
# 辅助函数：计算 BER
# ----------------------------
def calc_ber(decoded_bits, true_bits):
    """计算逐位错误率（标量）"""
    errors = (decoded_bits != true_bits).sum().item()
    total_bits = true_bits.numel()
    return errors / total_bits if total_bits > 0 else 0.0

# ----------------------------
# Uncoded simulation function
# ----------------------------
def run_uncoded_simulation(ebno_db, info_bits_per_frame, batch_size, num_frames_max):
    """Run uncoded BER simulation for a given EbN0 point"""
    bits_info = torch.randint(0, 2, (batch_size, info_bits_per_frame), dtype=torch.float32).to(DEVICE)
    
    # Map directly without coding
    n_frames = bits_info.shape[1]
    n_symbols = n_frames // bits_per_symbol
    codeword_bits_trimmed = bits_info[:, :n_symbols * bits_per_symbol]
    symbols = modulator(codeword_bits_trimmed)
    
    # Calculate noise variance with coding rate R=1 for uncoded
    ebno_lin = 10 ** (ebno_db / 10.0)
    noise_var = 1.0 / (2.0 * CODE_RATE * ebno_lin)  # Use CODE_RATE for fair comparison
    y = awgn_ch(symbols, noise_var)
    
    llrs = demapper(y, noise_var)
    
    # Hard decision on LLRs for uncoded
    decoded_bits = (llrs > 0).float()
    
    min_len = min(decoded_bits.shape[1], bits_info.shape[1])
    decoded_bits = decoded_bits[:, :min_len]
    bits_info_trimmed = bits_info[:, :min_len]
    
    batch_errors = (decoded_bits != bits_info_trimmed).sum().item()
    batch_total = min_len * batch_size
    
    return batch_errors, batch_total

# ----------------------------
# 主仿真循环
# ----------------------------
ebno_db_vals = []
coded_ber_vals = []
uncoded_ber_vals = []

for ebno_db in tqdm(np.arange(EBNO_DB_MIN, EBNO_DB_MAX + EBNO_DB_STEP/2, EBNO_DB_STEP),
                    desc="Simulating EbN0 points",
                    total=MAX_EBNO_POINTS):

    # ========== Coded BER Simulation ==========
    coded_total_bits_tested = 0
    coded_total_errors = 0
    num_frames_run = 0

    while coded_total_errors < MAX_BITS_TESTED * MIN_BER_TARGET and num_frames_run < NUM_FRAMES_PER_EBN0:
        info_bits_per_frame = 2000
        
        bits_info = torch.randint(0, 2, (BATCH_SIZE, info_bits_per_frame), dtype=torch.float32).to(DEVICE)
        
        codeword_bits = conv_enc(bits_info)
        n_frames = codeword_bits.shape[1]

        if n_frames % bits_per_symbol != 0:
            n_symbols = n_frames // bits_per_symbol
            codeword_bits_trimmed = codeword_bits[:, :n_symbols * bits_per_symbol]
        else:
            n_symbols = n_frames // bits_per_symbol
            codeword_bits_trimmed = codeword_bits

        symbols = modulator(codeword_bits_trimmed)
        
        ebno_lin = 10 ** (ebno_db / 10.0)
        noise_var = 1.0 / (2.0 * CODE_RATE * ebno_lin)  # Corrected: accounts for coding rate
        y = awgn_ch(symbols, noise_var)
        
        llrs = demapper(y, noise_var)
        decoded_bits = viterbi_dec(llrs)

        min_len = min(decoded_bits.shape[1], bits_info.shape[1])
        decoded_bits = decoded_bits[:, :min_len]
        bits_info_trimmed = bits_info[:, :min_len]

        batch_errors = (decoded_bits != bits_info_trimmed).sum().item()
        batch_total = min_len * BATCH_SIZE

        coded_total_errors += batch_errors
        coded_total_bits_tested += batch_total
        num_frames_run += 1

        if coded_total_bits_tested > 1e5 and (coded_total_errors / coded_total_bits_tested) < MIN_BER_TARGET:
            break

    coded_ber = coded_total_errors / coded_total_bits_tested if coded_total_bits_tested > 0 else 0.0
    ebno_db_vals.append(ebno_db)
    coded_ber_vals.append(coded_ber)

    # ========== Uncoded BER Simulation (same EbN0, same random bits) ==========
    uncoded_total_bits_tested = 0
    uncoded_total_errors = 0
    num_frames_run = 0

    while uncoded_total_errors < MAX_BITS_TESTED * MIN_BER_TARGET and num_frames_run < NUM_FRAMES_PER_EBN0:
        info_bits_per_frame = 2000
        
        bits_info = torch.randint(0, 2, (BATCH_SIZE, info_bits_per_frame), dtype=torch.float32).to(DEVICE)
        
        n_frames = bits_info.shape[1]
        n_symbols = n_frames // bits_per_symbol
        codeword_bits_trimmed = bits_info[:, :n_symbols * bits_per_symbol]
        symbols = modulator(codeword_bits_trimmed)
        
        ebno_lin = 10 ** (ebno_db / 10.0)
        noise_var = 1.0 / (2.0 * CODE_RATE * ebno_lin)  # Same noise for fair comparison
        y = awgn_ch(symbols, noise_var)
        
        llrs = demapper(y, noise_var)
        decoded_bits = (llrs > 0).float()

        min_len = min(decoded_bits.shape[1], bits_info.shape[1])
        decoded_bits = decoded_bits[:, :min_len]
        bits_info_trimmed = bits_info[:, :min_len]

        batch_errors = (decoded_bits != bits_info_trimmed).sum().item()
        batch_total = min_len * BATCH_SIZE

        uncoded_total_errors += batch_errors
        uncoded_total_bits_tested += batch_total
        num_frames_run += 1

        if uncoded_total_bits_tested > 1e5 and (uncoded_total_errors / uncoded_total_bits_tested) < MIN_BER_TARGET:
            break

    uncoded_ber = uncoded_total_errors / uncoded_total_bits_tested if uncoded_total_bits_tested > 0 else 0.0
    uncoded_ber_vals.append(uncoded_ber)

# ----------------------------
# 输出结果 & 绘图
# ----------------------------
print("\n=== Simulation Results ===")
print(f"{'EbN0 (dB)':<12} {'Coded BER':<15} {'Uncoded BER':<15}")
print("-" * 42)
for ebno, coded_ber, uncoded_ber in zip(ebno_db_vals, coded_ber_vals, uncoded_ber_vals):
    print(f"{ebno:<12.1f} {coded_ber:<15.2e} {uncoded_ber:<15.2e}")

# Filter out zero BER values for plotting
non_zero_coded_mask = [ber > 0 for ber in coded_ber_vals]
non_zero_uncoded_mask = [ber > 0 for ber in uncoded_ber_vals]

ebno_db_plot = ebno_db_vals  # Use all EbN0 points for x-axis
coded_ber_plot = [ber for ber, mask in zip(coded_ber_vals, non_zero_coded_mask) if mask]
uncoded_ber_plot = [ber for ber, mask in zip(uncoded_ber_vals, non_zero_uncoded_mask) if mask]

# Create filtered versions for plotting (only show points with actual errors)
coded_ebno_plot = [ebno for ebno, mask in zip(ebno_db_vals, non_zero_coded_mask) if mask]
uncoded_ebno_plot = [ebno for ebno, mask in zip(ebno_db_vals, non_zero_uncoded_mask) if mask]

plt.figure(figsize=(10, 6))
if coded_ber_plot:
    plt.semilogy(coded_ebno_plot, coded_ber_plot, 'bo-', label=f"Coded ({QAM_ORDER}-QAM, Rate={CODE_RATE})")
else:
    plt.plot([], [], 'bo-', label="Coded (No data)")
    
if uncoded_ber_plot:
    plt.semilogy(uncoded_ebno_plot, uncoded_ber_plot, 'rs--', label=f"Uncoded ({QAM_ORDER}-QAM)")

plt.xlabel("Eb/N₀ (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title(f"AWGN Channel: Coded vs Uncoded {QAM_ORDER}-QAM with Convolutional Coding")
plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
