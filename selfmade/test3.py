def rebuild_code(flat_bits, layers):
    segs, pos = [], 0
    # SEG_LENS = []
    # for l in range(1, layers + 1):
    #     SEG_LENS.append(l)
    # for L in SEG_LENS:
    for L in range(1, layers + 1):
        segs.append("".join(flat_bits[pos:pos + L]))
        pos += L
    return "-".join(segs)

def flatten_code(code):
    return list(code.replace("-", ""))

b = ['0', '1', '1', '1', '0', '1', '0', '0', '1', '0', '0', '0', '1', '0', '1', '0', '0', '1', '0', '0', '0']

print(rebuild_code(b, 6))