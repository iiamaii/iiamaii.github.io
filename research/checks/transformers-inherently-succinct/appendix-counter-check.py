#!/usr/bin/env python3
"""Local audit of arXiv:2510.19315v3, Appendix A.2, Eqs. (14a–e).

This interprets the counter gadget, not the whole B-RASP/UHAT compiler.
The optional OR changes only the default branch of (14e).
"""

import json
from pathlib import Path


def counter_gadget(word, n, repair=False):
    size = len(word)
    # B[k][i] is the (k+1)-th previous BIT at position i (LSB first at '#').
    bits = [[False] * size for _ in range(n)]
    for i in range(size):
        previous_bits = [j for j in range(i) if word[j] in "01"]
        if previous_bits:
            j = previous_bits[-1]
            bits[0][i] = word[j] == "1"
            for k in range(1, n):
                bits[k][i] = bits[k - 1][j]

    increment, rollover = [False] * size, [False] * size
    for i in range(size):
        separators = [j for j in range(i) if word[j] == "#"]
        if separators:
            j = separators[-1]
            increment[i] = any(
                all(not bits[r][i] and bits[r][j] for r in range(k))
                and bits[k][i] and not bits[k][j]
                and all(bits[r][i] == bits[r][j] for r in range(k + 1, n))
                for k in range(n)
            )
            rollover[i] = all(not bits[k][i] and bits[k][j] for k in range(n))
        else:
            increment[i] = False  # literal default of (14c)
            rollover[i] = all(not bits[k][i] for k in range(n))

    output = []
    for i in range(size):
        bad_predecessors = [j for j in range(i) if word[j] == "#"
                            and not rollover[j] and not increment[j]]
        default = ((rollover[i] or increment[i]) if repair
                   else (rollover[i] and increment[i]))
        output.append(False if bad_predecessors else default)
    return {
        "accepted_by_B": output[-1],
        "separators": [
            {"position_1_based": i + 1,
             "increment": increment[i], "rollover_or_initial_zero": rollover[i],
             "B": output[i]}
            for i, char in enumerate(word) if char == "#"
        ],
    }


def encode(addresses, n):
    return "".join(format(address, "0%db" % n) + "t#" for address in addresses)


def oracle(addresses, n):
    return all(address == i % (2 ** n) for i, address in enumerate(addresses))


def run():
    minimal = "0t#1t#"
    result = {
        "source": "https://arxiv.org/pdf/2510.19315v3#page=15",
        "target": "Appendix A.2, Eqs. (14a–e), counter gadget B only",
        "minimal_valid_tiling": {
            "N": 1, "rows": 1, "tile": "t=(0,0,0,0)=t_fin", "word": minimal,
            "literal": counter_gadget(minimal, 1),
            "candidate_or_repair": counter_gadget(minimal, 1, repair=True),
        },
        "valid_cases": [], "single_bit_mutations": {},
        "limitations": [
            "No execution of gadgets A,C,D,E or the B-RASP-to-UHAT compiler.",
            "The valid all-zero-edge two-column tiling is checked by inspection.",
            "No formal proof of the full reduction, no author confirmation, no softmax training.",
            "Finite checks of OR do not establish its global correctness.",
        ],
    }
    checked = mismatches = original_accepts = repaired_accepts = 0
    for n in range(1, 6):
        for rows in (1, 2):
            addresses = list(range(2 ** n)) * rows
            word = encode(addresses, n)
            before = counter_gadget(word, n)["accepted_by_B"]
            after = counter_gadget(word, n, True)["accepted_by_B"]
            result["valid_cases"].append({"N": n, "rows": rows,
                                           "literal_B": before, "OR_B": after})
            assert not before and after
            # Preserves cell shape and changes exactly one address bit.
            for cell in range(len(addresses)):
                for bit in range(n):
                    mutated = list(addresses)
                    mutated[cell] ^= 1 << bit
                    expected = oracle(mutated, n)
                    encoded = encode(mutated, n)
                    literal = counter_gadget(encoded, n)["accepted_by_B"]
                    repaired = counter_gadget(encoded, n, True)["accepted_by_B"]
                    checked += 1
                    original_accepts += literal
                    repaired_accepts += repaired
                    mismatches += repaired != expected
    result["single_bit_mutations"] = {
        "checked": checked, "oracle_valid": 0, "literal_accepts": original_accepts,
        "OR_accepts": repaired_accepts, "OR_oracle_mismatches": mismatches,
    }
    assert mismatches == 0
    path = Path(__file__).with_name("counter-check-results.json")
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"minimal": result["minimal_valid_tiling"],
                      "valid_cases": len(result["valid_cases"]),
                      "mutations": result["single_bit_mutations"]}, ensure_ascii=False))


if __name__ == "__main__":
    run()
