import time
import ctypes
import os
import sys

# Add the hardware bridges folder to path so we can import the core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Hardware_Agnostic_Bridges")))
from mecan_core import mecan_engine

def run_qsbits_benchmark():
    print("  MECANTENSOR : PYTHON TEST BENCH 2 (QSBits)    ")
    print("=================================================\n")

    BATCH_SIZE = 4096
    DIM = 16384
    PACKED_DIM = DIM // 8

    print(">>> Initializing 200T Core Scale Simulation...")
    print(f"    [+] Input Tensor: {BATCH_SIZE} x {DIM} (Uint8 Packed Activations)")
    print(f"    [+] Weight Tensor: {DIM} x {PACKED_DIM} (Uint8 Packed 1-Bit Weights)\n")

    # Allocate through the C++ Wrapper
    in_ptr = mecan_engine.allocate_ssd_tensor([BATCH_SIZE, PACKED_DIM], is_binary=True)
    w_ptr = mecan_engine.allocate_ssd_tensor([DIM, PACKED_DIM], is_binary=True)
    out_ptr = mecan_engine.allocate_ssd_tensor([BATCH_SIZE, DIM], is_binary=False)

    print(">>> Running QSBits Hardware-Agnostic Matrix Calculation...")

    start_time = time.perf_counter()
    mecan_engine.qsbits_forward(in_ptr, w_ptr, out_ptr)
    end_time = time.perf_counter()

    duration_ms = (end_time - start_time) * 1000.0
    print(f"[QSBits BENCHMARK] Forward Pass -> {duration_ms:.4f} ms")

    # Calculate TOPS
    total_ops = float(BATCH_SIZE * DIM * DIM)
    tops = (total_ops / (end_time - start_time)) / 1e12

    print(f"    [!] Estimated Hardware Throughput: {tops:.2f} TOPS (Bitwise)")
    print("\n=================================================")
    print(" QSBits BENCHMARK COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    run_qsbits_benchmark()
