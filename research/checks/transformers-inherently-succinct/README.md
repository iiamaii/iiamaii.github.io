# Local audit of the counter gadget in Eq. (14e)

Source: Bergsträßer, Cotterell, Lin, *Transformers are Inherently Succinct*, [arXiv:2510.19315v3, Appendix A.2, Eqs. (14a–e)](https://arxiv.org/pdf/2510.19315v3#page=15).

This standard-library Python script interprets the counter gadget B only. It compares the printed AND in the default branch of (14e) with a candidate OR replacement; the earlier-error search condition is unchanged.

From the blog repository root:

```sh
python3 research/checks/transformers-inherently-succinct/appendix-counter-check.py
```

The script writes [counter-check-results.json](counter-check-results.json) next to itself. Expected: the literal version rejects all 10 valid addressed histories, the OR candidate accepts all 10, and rejects all 774 single-address-bit mutations with zero address-order oracle disagreements.

The minimal example is N=1 and `0t#1t#`, where t has four zero boundary colors and is the designated final tile. The validity of this two-cell tiling is checked by inspection. The script does **not** execute gadgets A,C,D,E, a B-RASP-to-UHAT compiler, training, or a formal proof. The finite check does not establish the global correctness of the candidate repair or disprove the paper's theorems. The authors have not confirmed the issue.

- [Script](appendix-counter-check.py)
- [Recorded results](counter-check-results.json)
- [Korean review](https://iiamaii.github.io/reviews/transformers-inherently-succinct/)
- [English review](https://iiamaii.github.io/en/reviews/transformers-inherently-succinct/)
