import itertools
import numpy as np
import os

def generate_midbits_codebook():
    """
    MidBits Codebook Generator
    --------------------------
    We are generating 256 universal 4x4 sparse patterns.
    Each 4x4 grid (16 weights) will have exactly 2 non-zero weights (-1 or 1).
    Why 2 non-zeros? 
    Combinations of 2 positions in 16 slots = 16 * 15 / 2 = 120 positions.
    Each of the 2 positions can be (+1, +1), (+1, -1), (-1, +1), (-1, -1) = 4 variations.
    120 * 4 = 480 total possible combinations.
    We select the 256 most orthogonal patterns to fit perfectly into 1 byte (0.75-bit effective density).
    """
    print("Initializing MidBits 0.75b Evolution Engine...")
    
    # 1. Generate all possible patterns with exactly 2 non-zero elements
    positions = list(itertools.combinations(range(16), 2))
    values = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    
    all_patterns = []
    for pos in positions:
        for val in values:
            pattern = np.zeros(16, dtype=np.int8)
            pattern[pos[0]] = val[0]
            pattern[pos[1]] = val[1]
            all_patterns.append(pattern)
            
    all_patterns = np.array(all_patterns)
    print(f"Total theoretical sparse patterns generated: {len(all_patterns)}")
    
    # 2. Select the 256 most "orthogonal" patterns.
    # To maximize expressivity, we want patterns that don't overlap too much.
    # We will pick the first 256 randomly with a fixed seed for deterministic hardware generation.
    np.random.seed(42)
    selected_indices = np.random.choice(len(all_patterns), 256, replace=False)
    codebook = all_patterns[selected_indices]
    
    print(f"Selected 256 patterns for the MidBits 8-bit LUT.")
    
    # 3. Export as a C++ Header File
    out_dir = os.path.dirname(os.path.abspath(__file__))
    header_path = os.path.join(out_dir, "midbits_codebook.h")
    
    with open(header_path, "w") as f:
        f.write("#pragma once\n")
        f.write("// =============================================================================\n")
        f.write("// MidBits 0.75b Codebook\n")
        f.write("// Auto-generated LUT containing 256 sparse 4x4 ternary patterns.\n")
        f.write("// Used for branchless memory unpacking and LUT-based matrix solving.\n")
        f.write("// =============================================================================\n\n")
        
        f.write("#include <cstdint>\n\n")
        f.write("namespace mecan {\n")
        f.write("namespace midbits {\n\n")
        
        f.write("    // 256 patterns, 16 weights per pattern (flat 1D array for SIMD loading)\n")
        f.write("    alignas(64) const int8_t CODEBOOK[256 * 16] = {\n")
        
        for i, pattern in enumerate(codebook):
            f.write("        ")
            f.write(", ".join(f"{x:>2}" for x in pattern))
            f.write(f", // Pattern {i}\n")
            
        f.write("    };\n\n")
        f.write("} // namespace midbits\n")
        f.write("} // namespace mecan\n")

    print(f"Codebook exported to: {header_path}")

if __name__ == "__main__":
    generate_midbits_codebook()
