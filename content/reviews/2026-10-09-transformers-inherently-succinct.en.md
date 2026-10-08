---
title: "Why can a Transformer describe the same rules more compactly?"
description: "A guide to Transformers are Inherently Succinct through long computation histories and small verifiers: size separations, UHAT assumptions, verification complexity, and an appendix connective issue."
date: "2026-10-09"
publishedAt: "2026-10-09T00:06:00+09:00"
updatedAt: "2026-10-09T00:31:33+09:00"
topics: ["language models", "transformer theory", "formal languages", "computational complexity", "formal verification"]
translationKey: "transformers-inherently-succinct"
paperTitle: "Transformers are Inherently Succinct"
authors: "Pascal Bergsträßer, Ryan Cotterell, Anthony W. Lin"
year: "2026"
paperUrl: "https://arxiv.org/abs/2510.19315v3"
visibility: "public"
lang: "en"
thumbnail: "/assets/reviews/transformers-inherently-succinct/transformers-core.svg"
thumbnailAlt: "Attention retrieves horizontal neighbors and the previous row at the same address from a long computation history for a small UHAT verifier; comparison labels show worst-case description-size gaps."
---

Recognizing a rule and **describing that rule compactly** are different abilities. This paper shows that certain Transformers can describe the same language exponentially more compactly than temporal logic or fixed-precision RNNs, and doubly exponentially more compactly than finite automata. Its central construction places a long computation history in the input and uses attention to retrieve the positions needed for verification. The comparison concerns **minimum binary description length** under precise model assumptions, rather than measured LLM speed or accuracy. [§§1–4, Theorems 15 and 17, Corollary 18](https://arxiv.org/pdf/2510.19315v3#page=9)

**Reading the cover diagram.** The verifier on the right retrieves positions from the long input on the left and checks their relationships. This is an original illustration of the proof construction, based on [§§3–4](https://arxiv.org/pdf/2510.19315v3#page=6). [Open full size](/assets/reviews/transformers-inherently-succinct/transformers-core.svg).

1. **The numbers inside cells are column addresses.** The two-bit addresses `00, 01, 10, 11` represent decimal 0, 1, 2, and 3. The two rows are consecutive, reusing the same addresses in the next row. `tile t` and `tile u` are abbreviated labels for tiles in each row; they do not mean that all tiles in a row are identical. The proof uses N-bit addresses for `2ᴺ` columns; only four are shown here.
2. **The two arrows connect information needed for comparison.** The `neighbor` arrow passes information from cell `01` to cell `10` in the same row to check horizontal colors. The `same address` arrow means that the current row's cell `10` retrieves information from cell `10` in the preceding row to check vertical colors. Attention searches from the current cell toward an earlier position; the diagram's arrows show the retrieved information flowing into the current check. Their lengths encode neither runtime nor actual token distance.
3. **The right-hand box represents reusable verification rules.** The left-hand table is supplied data; the UHAT on the right decides whether it is valid. The same rules are shared across positions, so each cell does not require new rules or weights. The input and per-position intermediate representations still require memory. Neighbor checks alone are insufficient: the verifier also checks format, address order, boundaries, and termination. The five gadgets below explain this division.
4. **The two boxes below compare description costs for the same task.** For particular witness language families, equivalent LTL and fixed-precision RNN descriptions must be exponentially larger, while finite automata must be doubly exponentially larger. Box areas are not drawn to those ratios, and the labels do not compare training accuracy or inference speed. [Theorems 15 and 17, Corollary 18](https://arxiv.org/pdf/2510.19315v3#page=9)

**“Long history. Small verifier.”** means keeping the description of the checker small while the long history remains in the input. It does not mean shortening the input. This distinction connects the size lower bounds and verification complexity discussed below.

## What question does the paper ask?

Here, a **language** is a set of strings satisfying a condition. A model receives a string and decides membership. “Do addresses increase correctly, and do adjacent tiles have matching colors?” defines one such recognition task.

There are two distinct comparisons.

| Question | What is compared? |
| --- | --- |
| Expressivity | Which kinds of languages can the model recognize? |
| Succinctness | What is the minimum size needed to recognize **the same language**? |

A model recognizing more languages need not be more compact on shared tasks. Under the relevant assumptions, UHAT recognizes **star-free languages**, a subclass of the regular languages. Fixed-precision RNNs can recognize all regular languages, yet may need much larger descriptions for particular shared languages. Star-free languages can be defined using operations including union, concatenation, and complement. The important distinction is that **language-class inclusion does not determine representation cost**. [§§1–2, Proposition 1, Definitions 2–3](https://arxiv.org/pdf/2510.19315v3#page=3)

Size includes the binary encoding of the structure and weights, rather than just a parameter count. RNN precision also counts. Otherwise, an arbitrarily precise number could conceal information while appearing to cost one parameter. [§§2.4–2.5](https://arxiv.org/pdf/2510.19315v3#page=5)

## The core idea: a small verifier for a long history

### Which Transformer is being studied?

The model is a **unique-hard attention transformer (UHAT)**. It retrieves the vector at one maximum-scoring position, with a fixed rule for breaking ties. This differs from softmax attention, which combines values from multiple positions. [§2.2](https://arxiv.org/pdf/2510.19315v3#page=4)

| Component | Setting in this paper | Why it matters |
| --- | --- | --- |
| Attention | One maximum-scoring position | Results do not automatically transfer to softmax models |
| Position information | No separate positional embeddings | This is not the same model as a RoPE-based LLM |
| Masks and ties | Restrictions on past/future access; leftmost/rightmost ties | These affect whether the most recent matching position is accessible |
| Arithmetic and precision | Affine maps and ReLU; upper bounds allow rational weights | The lower-bound construction is also stated for input-length-independent fixed-precision integers |
| Input length | Finite strings with no fixed maximum length | This differs from accuracy within a bounded context window |

A strict past-access mask permits `j < i`, excluding the current position. The paper calls this “future masking” because it **hides the future**. Choosing the rightmost tied candidate then retrieves the most recent matching past position. Neither the strictness nor the tie rule is incidental. [§2; p.8, Corollaries 11 and 14](https://arxiv.org/pdf/2510.19315v3#page=8)

### Turn a computation history into an addressed table

The proof replaces running a long computation with checking an already supplied history. It uses a tiling table of width `2ᴺ`, giving every cell an N-bit column address. Each row starts at address zero, proceeds to the final address, and then wraps to zero for the next row. The table is encoded as a string in row order. [§3, Lemma 8; Appendix A.2](https://arxiv.org/pdf/2510.19315v3#page=14)

```text
Illustration with N = 2

Row r:      00 t₀ #   01 t₁ #   10 t₂ #   11 t₃ #
Row r+1:    00 u₀ #   01 u₁ #   10 u₂ #   11 u₃ #

Horizontal: does the right color of u₁ match the left color of u₂?
Vertical:   does the upper color of t₂ match the lower color of u₂?
```

For horizontal adjacency, retrieve the preceding cell. For vertical adjacency, retrieve the **most recent earlier cell with the same address**. Address matching and recency avoid writing the enormous distance between rows explicitly into the verifier. The tiling convention builds rows from bottom to top, hence the upper/lower edge comparison above. [Appendix A.2, Eq. (17)](https://arxiv.org/pdf/2510.19315v3#page=16)

Five gadgets divide the checks.

| Gadget | Condition |
| --- | --- |
| A | Correct arrangement of address bits, tile symbols, and separators |
| B | Addresses start at zero, increment, and wrap around correctly |
| C | Correct final address and designated final tile |
| D | Correct colors at the four boundaries |
| E | Matching colors between horizontal and vertical neighbors |

The authors express these conditions in the Boolean programming language **B-RASP** and compile them into UHAT. The polynomial-size compilation applies to the special search conditions used here, not to every B-RASP program. Even without a valid candidate, attention can still return an argmax position; the construction must recheck the condition and use a default when the selected position is invalid. [Lemma 9; Appendix A.3](https://arxiv.org/pdf/2510.19315v3#page=17)

The saving is in the **description of the verification rules**. The long history remains in the input. The construction does not place the entire history in one small hidden vector or generate a doubly exponentially long output quickly.

## Results and evidence: where do the size gaps come from?

The central tool is the **shortest accepted string**. If a small verifier accepts only extremely long strings, how large must another representation be to impose the same requirement?

The paper uses a computation that increments a `2ⁿ`-bit counter through its values. For n=3, this is an eight-bit counter with 256 possible values. In general there are `2^(2ⁿ)` values. A UHAT with a description polynomial in n and the tile-set encoding can check such a history while requiring a shortest accepted string of doubly exponential length. [§4, Theorem 15](https://arxiv.org/pdf/2510.19315v3#page=9)

```text
Verifier description: poly(n)
Length of its shortest accepted string: at least 2^(2ⁿ)
```

The comparison representations guarantee relatively short accepted strings when their descriptions are small. Combining those guarantees with the construction yields the separations.

| Representation | Short accepted string guaranteed by a small representation | Minimum-size gap reported in the paper |
| --- | --- | --- |
| LTL: linear temporal logic | At most exponential in formula size | Some language families require exponentially larger LTL descriptions |
| Finite automata | With q states, an accepted path shorter than q exists | Some language families require doubly exponentially many states |
| Fixed-precision RNNs | k bits per coordinate and D dimensions give at most 2<sup>kD</sup> states | An exponential gap when precision is included in the description cost |

LTL describes temporal relationships such as what held earlier or what holds until another condition becomes true. A finite automaton updates a finite state as it reads each symbol. Removing a repeated-state cycle from an accepted path gives the short-path property used above. [§2; Theorems 15 and 17, Corollary 18](https://arxiv.org/pdf/2510.19315v3#page=9)

**The quantifiers matter.** There exist witness language families for which every equivalent comparison representation must be large. This does not say that Transformers are smaller on every task. The exponents are first obtained in the family index n; translating them into UHAT description size depends on the polynomial relating that size to n. The result does not assert one exact factor of two raised to the model size for all models. [Definition 2, Theorem 15](https://arxiv.org/pdf/2510.19315v3#page=5)

## An upper bound: unfold the model into a larger formula

The paper also gives a direct translation from **every UHAT to an exponentially sized LTL formula**. The reverse translation, LTL→UHAT, has polynomial size. Confusing the two directions would reverse the central conclusion. [Propositions 13 and 16](https://arxiv.org/pdf/2510.19315v3#page=8)

For the upper bound, hard attention retrieves **one existing vector** rather than accumulating all input positions. Once the model and rational weights are fixed, the bit length of values at each layer can be bounded polynomially in the model description. The proof enumerates possible vectors and describes “this position has this vector” using temporal formulas. [Proposition 12; Appendices A.4–A.5](https://arxiv.org/pdf/2510.19315v3#page=18)

Following UHAT→LTL with a standard translation gives a doubly exponential upper bound for **nondeterministic finite automata (NFAs)**. The lower bound applies to both NFAs and deterministic finite automata (DFAs), but this route alone does not give the same upper bound for DFAs: determinization can incur additional cost. [§4, Theorem 17](https://arxiv.org/pdf/2510.19315v3#page=9)

## The other side of compactness: verifying all behavior is hard

A small description can compress an enormous set of behaviors. How difficult is checking that two models agree on **all strings of all lengths**?

The paper establishes **EXPSPACE-completeness** for UHAT nonemptiness—whether any accepted string exists—and language equivalence. These are worst-case decision problems measured in the binary description length of the models. This is not a claim that a forward pass on one string consumes exponential space. [Theorems 4 and 19; Appendices A.1 and A.7](https://arxiv.org/pdf/2510.19315v3#page=10)

Masks and ties matter here too. Past-only access with the most recent tied candidate can encode the difficult tiling checks. For the restriction using past-only access with the oldest tied candidate, the paper gives an **NEXP upper bound**; it does not establish NEXP-completeness. [Corollaries 11 and 14](https://arxiv.org/pdf/2510.19315v3#page=8)

This need not end practical verification. Bounded lengths and simpler model subclasses define different problems. The authors discuss looking for subclasses unable to express the large counters behind the hard instances. [§6](https://arxiv.org/pdf/2510.19315v3#page=10)

## An appendix issue: the AND in Eq. (14e)

The preceding results are the **paper's stated theorems and proof structure**. During this review, I found a local issue in the address-counter construction used by the proof. In Appendix A.2, Eq. (14e) requires both a normal increment and a rollover at the current position when there is no earlier error. The same connective appears in the PDF and TeX.

<figure class="review-figure" id="paper-equation-14e">
<a href="/assets/reviews/transformers-inherently-succinct/paper-equation-14e.png" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/transformers-inherently-succinct/paper-equation-14e.png" width="2400" height="108" alt="Original Appendix A.2 Eq. (14e), whose default branch conjoins the rollover condition B₁→₀(i) and increment condition B₊₁(i)." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Paper excerpt · Counter check, Eq. (14e)</span>Look at ∧ in the default branch on the right. A valid address transition is an increment or a rollover. This branch must be distinguished from the preceding search for an earlier error. <span class="figure-links"><a href="https://arxiv.org/pdf/2510.19315v3#page=15">Bergsträßer et al., arXiv v3, p.15</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · Cropped equation; original notation and fonts preserved · <a href="/assets/reviews/transformers-inherently-succinct/paper-equation-14e.png">Full size</a></span></figcaption>
</figure>

Read the equation from left to right. `Q#(j)` means that j is a separator `#` position, and `j < i` restricts the search to earlier positions. The black-triangle search looks for an earlier transition that is **neither a rollover nor a valid increment**. If such an error is found, it returns zero. Otherwise, the default branch after the colon checks the current transition. Here, `B₊₁` tests an ordinary +1 increment; `B₁→₀` tests rollover from the final address to zero and also handles the initial-zero condition. [Eqs. (14c–e)](https://arxiv.org/pdf/2510.19315v3#page=15)

The connective under examination is therefore **the final ∧ in the default branch**. The earlier ∧ joins the conditions “not a rollover” and “not an increment” to identify an error. Replacing both locations with OR is not the candidate tested in this review.

The smallest example uses N=1 and a tile t with all four boundary colors zero: `0t#1t#`. The addresses correctly increase from zero to one. At the final position:

| Condition | Value |
| --- | --- |
| Normal increment `B₊₁` | True |
| Rollover `B₁→₀` | False |
| Printed default: AND | False → valid input rejected |
| Candidate replacement: OR | True |

The two conditions cannot both hold for N≥1. Since the full verifier requires B, the printed construction rejects this valid tiling. [Appendix A.2, Eqs. (14c–e)](https://arxiv.org/pdf/2510.19315v3#page=15)

I implemented **only counter Eqs. (14a–e)** in Python. Across ten valid cases with N=1…5 and one or two rows, the printed connective rejected every case; replacing only the default AND with OR accepted all ten. The candidate also rejected all 774 variants obtained by flipping one address bit, with zero disagreements against an independent address-order oracle. This is a finite local check, not an exhaustive proof or a measurement of learning accuracy. [Check script](https://github.com/iiamaii/iiamaii.github.io/blob/main/research/checks/transformers-inherently-succinct/appendix-counter-check.py) · [Recorded output](https://github.com/iiamaii/iiamaii.github.io/blob/main/research/checks/transformers-inherently-succinct/counter-check-results.json).

**The limit of this observation matters.** A typographical error is plausible, but the authors have not confirmed it. I did not validate all five gadgets, B-RASP→UHAT compilation, or precision semantics, so the OR candidate does not certify the complete proof. The issue creates a verification task for the lower bounds relying on this construction; it is not itself a disproof of the theorems.

## My interpretation: a compact verifier rather than a compact executor

The useful perspective is to view attention as a **shared rule for retrieving positions**. Instead of retaining the entire table in one recurrent hidden state, the verifier retrieves the earlier per-position representations needed for a local relation. This exposes representation costs invisible in a hierarchy based only on expressivity. That is my interpretation of the proof construction.

Three claims remain separate: suitable small weights **exist**, learning can **find** them, and hardware can **execute** them efficiently. This paper concerns the first. Section 6 also leaves succinctness for fixed-precision softmax and average-hard attention to future work. The results therefore do not establish that practical LLMs are always more efficient than RNNs. [§6](https://arxiv.org/pdf/2510.19315v3#page=10)

## Open questions and further reading

The first follow-up is to compare the **complete tiling verifier against an independent oracle**, including the candidate counter repair. Next, check that the special B-RASP→UHAT compilation handles absent candidates and ties correctly. These steps keep a local test from being mistaken for certification of the entire theorem.

The following three sources clarify background and boundaries. This is a selected reading path, not an exhaustive survey or a ranking by recency.

| Source | Connection | Scope read for this review |
| --- | --- | --- |
| [Yang et al., NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/hash/13d7f172259b11b230cc5da8768abc5f-Abstract-Conference.html) | Expressivity correspondence between UHAT, star-free languages, and LTL | Abstract; PDF §§5.1–5.2, Theorems 5–6 |
| [Jerad et al., ACL 2025, Unique Hard Attention: A Tale of Two Sides](https://aclanthology.org/2025.acl-short.76/) | Why leftmost and rightmost ties matter | Abstract, §§2–3, Table 1 |
| [Yang et al., 2026, Length Generalization Bounds for Transformers](https://arxiv.org/abs/2603.02238v2) | How the placement of precision restrictions changes length-generalization bounds | Abstract; §5 theorem/proposition statements and discussion, not all proofs |

The last source depends on assumptions including attention-internal precision and BOS/EOS. Its results cannot simply be substituted into the no-positional-embedding UHAT setting here. A follow-up comparison should first align the model assumptions.

## Sources and reading scope

- Main paper: Pascal Bergsträßer, Ryan Cotterell, Anthony W. Lin, **Transformers are Inherently Succinct**, ICLR 2026. [arXiv:2510.19315v3](https://arxiv.org/abs/2510.19315v3), version dated 2026-05-15, 21 PDF pages. First submitted to arXiv on 2025-10-22.
- Reading: on 2026-10-08, I read §§1–6 and Appendices A.1–A.7 through TeX and extracted PDF text, and compared rendered PDF pp.8–10 and 15 for central statements and equations. I did not visually inspect every page.
- Version limitation: I could not directly obtain the requested [OpenReview PDF](https://openreview.net/pdf?id=Yxz92UuPLQ). This review uses the official arXiv version. Exact PDF identity, OpenReview reviews, and errata remain **unverified**.
- Cover: one original explanatory SVG based on §§3–4, using Light 300 with Semi Bold 600 emphasis. Space Grotesk is under SIL OFL 1.1. Arrow lengths and table dimensions do not encode measured quantities.
- Paper excerpt: arXiv v3, p.15, Eq. (14e), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Surrounding whitespace was cropped; notation and fonts were not modified.
- Executed check: finite counter tests with ten valid cases and 774 single-bit variants. Full A–E validation, UHAT compiler execution, machine-checked proofs of the complete theorems, training, and GPU benchmarks were **not executed**.
