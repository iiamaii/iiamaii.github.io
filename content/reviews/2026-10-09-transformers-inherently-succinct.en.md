---
title: "Why can a Transformer describe the same rules more compactly?"
description: "A guide to Transformers are Inherently Succinct through long computation histories and small verifiers: size separations, UHAT assumptions, verification complexity, and an appendix connective issue."
date: "2026-10-09"
updatedAt: "2026-10-23T15:26:14+00:00"
publishedAt: "2026-10-09T00:06:00+09:00"
topics: ["language models", "transformer theory", "formal languages", "computational complexity", "formal verification"]
translationKey: "transformers-inherently-succinct"
paperTitle: "Transformers are Inherently Succinct"
paperPublishedDate: "2025-10-22"
authors: ["Pascal Bergsträßer", "Ryan Cotterell", "Anthony W. Lin"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2510.19315v3"
visibility: "public"
lang: "en"
thumbnail: "/assets/reviews/transformers-inherently-succinct/transformers-core.svg"
thumbnailAlt: "Attention retrieves horizontal neighbors and the previous row at the same address from a long computation history for a small UHAT verifier; comparison labels show worst-case description-size gaps."
---

This paper aims to show that unique-hard attention transformers (UHATs) can be exponentially more succinct than linear temporal logic (LTL) or fixed-precision recurrent neural networks (RNNs) recognizing the same language, and doubly exponentially more succinct than finite automata. The core is a tiling verifier that lets a UHAT of size polynomial in $n$ recognize a language whose shortest accepted string has length at least $2^{2^n}$; all three size lower bounds and the EXPSPACE-hardness of non-emptiness and equivalence pass through this construction. [§4, Theorems 15 and 17, Corollary 18](https://arxiv.org/pdf/2510.19315v3#page=9) **Review interpretation.** The upper bounds based on the exponential UHAT→LTL translation hold independently of this construction, but the construction's address check, Eq. (14e), as printed accepts no input at all, so the printed proofs do not establish the lower bounds of Theorems 4, 15, 17, and 19 or Corollary 18; replacing the default branch's AND with OR repairs this check ([inspection below](#paper-equation-14e)).

**Reading the cover diagram.** The verifier on the right retrieves positions from the long input on the left and checks their relationships. This is an original illustration of the proof construction, based on [§§3–4](https://arxiv.org/pdf/2510.19315v3#page=6). [Open full size](/assets/reviews/transformers-inherently-succinct/transformers-core.svg).

1. **The numbers inside cells are column addresses.** The two-bit addresses `00, 01, 10, 11` represent decimal 0, 1, 2, and 3. The two rows are consecutive, reusing the same addresses in the next row. `tile t` and `tile u` are abbreviated labels for the tiles in each row. The proof uses N-bit addresses for $2^N$ columns; only four are shown here.
2. **The two arrows connect information needed for comparison.** The `neighbor` arrow passes information from cell `01` to cell `10` in the same row to check horizontal colors. The `same address` arrow means that the current row's cell `10` retrieves information from cell `10` in the preceding row to check vertical colors. Attention searches from the current cell toward an earlier position; the diagram's arrows show the retrieved information flowing into the current check.
3. **The right-hand box represents reusable verification rules.** The left-hand table is supplied data; the UHAT on the right decides whether it is valid. The same rules are shared across positions, so each cell does not require new rules or weights. Beyond neighbor checks, the verifier also checks format, address order, boundaries, and termination, which the five gadgets below divide among themselves.
4. **The two boxes below compare description costs for the same task.** For particular witness language families, equivalent LTL and fixed-precision RNN descriptions must be exponentially larger, while finite automata must be doubly exponentially larger. Box areas are not drawn to scale. [Theorems 15 and 17, Corollary 18](https://arxiv.org/pdf/2510.19315v3#page=9)

**“Long history. Small verifier.”** means that the long history stays in the input while only the description of the model checking it is kept small.

The article follows the paper's §§1–6 and treats the problem in Appendix A.2, Eq. (14e), in a separate section after §6.

## 1. Introduction

Fixed-precision transformers recognize various subregular classes depending on the assumptions, whereas fixed-precision RNNs recognize all regular languages, making RNNs more expressive. The authors propose **succinctness**, the size of the smallest representation recognizing a given language, as an alternative lens. As in the classic case of LTL, which is expressively equivalent to the star-free languages yet exponentially more succinct than finite automata and pays for it with harder decision problems, succinctness goes hand in hand with the difficulty of decision problems. The key technical ingredient behind the lower bounds is that UHATs can implement a counter from $0$ to $2^{2^N}$ using attention; the authors also give matching upper bounds and the complexity of verification problems. [Original §1](https://arxiv.org/pdf/2510.19315v3#page=1)

## 2. Preliminaries

First fix the compared models and their size measures.

### 2.1 Linear Temporal Logic

The paper builds LTL formulas from propositions $Q_a$ (the current symbol is $a$), Boolean connectives, and the since and until operators, which look into the past and the future; abbreviations such as past, future, and yesterday are defined from these two. A formula recognizes the strings on which it holds at the **last position**. [§2.1](https://arxiv.org/pdf/2510.19315v3#page=3)

### 2.2 Unique-hard attention transformers

A UHAT's attention retrieves the vector at one maximum-scoring position, with a fixed rule for breaking ties. This differs from softmax attention, which combines values from multiple positions. [§2.2](https://arxiv.org/pdf/2510.19315v3#page=4)

| Component | Setting in this paper | Relation to the results |
| --- | --- | --- |
| Attention | One maximum-scoring position | Succinctness for softmax and average-hard attention is left to future work in §6 |
| Position information | No separate positional embeddings; masking only | A different setting from prior work allowing arbitrary positional encodings (§6) |
| Masks and ties | Restrictions on past/future access; leftmost/rightmost ties | These affect whether the most recent matching position is accessible |
| Arithmetic and precision | Affine maps and ReLU; upper bounds allow rational weights | The lower-bound construction is also stated for input-length-independent fixed-precision integers |
| Input length | Finite strings with no fixed maximum length | The decision problems range over strings of every length |

The paper's strict future masking permits $j < i$, so attention sees only the past and excludes the current position, unlike the usual causal mask that includes it. Combined with rightmost tie-breaking, it retrieves the most recent past position satisfying a condition. [§2.2, §3](https://arxiv.org/pdf/2510.19315v3#page=4)

### 2.3 Boolean RASP

B-RASP sequentially defines Boolean values at each position and attention-based selection of another position; Yang et al. (2024) showed it to be expressively equivalent to UHATs. Here it serves as an intermediate representation for writing logical checks before translating them into attention layers. Defaults for absent candidates and tie-breaking belong to its semantics. [§2.3](https://arxiv.org/pdf/2510.19315v3#page=4)

### 2.4–2.5 RNNs, size measures, and succinctness

Here, a **language** is a set of strings satisfying a condition. A model receives a string and decides membership. “Do addresses increase correctly, and do adjacent tiles have matching colors?” defines one such recognition task.

There are two distinct comparisons.

| Question | What is compared? |
| --- | --- |
| Expressivity | Which kinds of languages can the model recognize? |
| Succinctness | What is the minimum size needed to recognize **the same language**? |

A model recognizing more languages need not be more compact on shared tasks. Under the relevant assumptions, UHAT recognizes **star-free languages**, a subclass of the regular languages. Fixed-precision RNNs can recognize all regular languages, yet may need much larger descriptions for particular shared languages. Star-free languages can be defined using operations including union, concatenation, and complement. The important distinction is that **language-class inclusion does not determine representation cost**. [§§1–2, Proposition 1, Definitions 2–3](https://arxiv.org/pdf/2510.19315v3#page=3)

Size is not a parameter count but the length of the minimal binary encoding of structure and weights. RNN precision $k$ enters the size in unary; the authors explain that this prevents comparing a transformer of fixed precision $k$ with an RNN allowed precision $2^k$. Definition 2 ($f$-more succinct) is an existential lower bound requiring one witness family on which the other class must be $f$ times larger, while Definition 3 ($g$-bounded expansion) is a universal upper bound requiring a translation of size at most $g$ for every language. “Exponentially more succinct” means a lower bound $f(m)\in\Omega(2^{cm^d})$ and “doubly exponentially” means $f(m)\in\Omega(2^{2^{cm^d}})$; together the two definitions pin down the size gap. [§§2.4–2.5](https://arxiv.org/pdf/2510.19315v3#page=5)

## 3. The size of smallest witness via non-emptiness problem

Theorem 4 states that non-emptiness for UHATs and B-RASP programs, whether **any** string is accepted, is EXPSPACE-complete. The lower bound transfers EXPSPACE-completeness of the width-$2^N$ tiling problem (Proposition 7) to Lemma 8’s B-RASP verifier and then to Lemma 9’s polynomial-size UHAT (Propositions 5 and 10). Because valid tilings and accepted strings must correspond in both directions, an error in any address, boundary, or adjacency check breaks the reduction. The upper bound comes from translating UHATs into exponentially sized LTL, and the two directions together give EXPSPACE-completeness. [§3, Theorem 4, Propositions 5, 7, and 10; Appendix A.1](https://arxiv.org/pdf/2510.19315v3#page=6)

### From tiling to B-RASP and UHAT: Lemmas 8–9

The proof replaces running a long computation with checking an already supplied history. It uses a tiling table of width $2^N$, giving every cell an N-bit column address. Each row starts at address zero, proceeds to the final address, and then wraps to zero for the next row. The table is encoded as a string in row order. [§3, Lemma 8; Appendix A.2](https://arxiv.org/pdf/2510.19315v3#page=14)

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

The authors express these conditions in the Boolean programming language **B-RASP** and compile them into UHAT. The polynomial-size compilation applies to the special search conditions used here, not to every B-RASP program. Even without a valid candidate, attention can still return an argmax position, so the translation rechecks the condition after selection and uses the default when it fails. [Lemma 9; Appendix A.3](https://arxiv.org/pdf/2510.19315v3#page=17) Because the construction uses only strict future masking and rightmost tie-breaking, the authors note that non-emptiness remains EXPSPACE-hard for UHATs restricted in this way (Corollary 11).

### Upper bounds: Propositions 12–13 and Corollary 14

The intuition for the upper bound is that hard attention retrieves **one existing vector**. Once the model and its rational weights are fixed, the bit length of every value arising in the computation is polynomial in the model size (Proposition 12). There are exponentially many possible values, but finitely many that can be enumerated, so the LTL formula only has to simulate masking and the selected position, not the computation of values. This yields a direct translation from **every UHAT to an exponentially sized LTL formula** (Proposition 13); since LTL non-emptiness is decidable in space polynomial in formula size, the space is exponential in the original UHAT size. The authors note that this improves the doubly exponential construction of Yang et al. (2024), which first translates UHATs into B-RASP. [Propositions 12–13; Appendices A.4–A.5](https://arxiv.org/pdf/2510.19315v3#page=18)

Masks and ties matter here too. A UHAT using strict future masking with leftmost tie-breaking (or strict past masking with rightmost) translates into LTL using only the past (or future) operator, whose non-emptiness is NP-complete, giving an **NEXP upper bound** (Corollary 14). This restriction differs from the EXPSPACE-hard one in Corollary 11 only in the tie-breaking direction. [Corollaries 11 and 14](https://arxiv.org/pdf/2510.19315v3#page=8)

## 4. Succinctness across representations

The authors point to the gap between EXPSPACE-complete UHAT non-emptiness and PSPACE-complete LTL non-emptiness as a first hint of a succinctness gap. The central tool is the **shortest accepted string**: if a small verifier accepts only extremely long strings, any other representation of the same language must be large enough to force the same length. [§4](https://arxiv.org/pdf/2510.19315v3#page=9)

### Theorem 15: exponentially more succinct than LTL

The witness family for Theorem 15 is built in three steps. [§4, Theorem 15](https://arxiv.org/pdf/2510.19315v3#page=9)

1. A deterministic Turing machine $\mathcal M_n$ writes $2^n$ zeros on an empty tape and increments this $2^n$-bit binary counter until it reaches $1^{2^n}$, then accepts. Initialization needs a number of states linear in $n$, and incrementing needs a constant-size machine. The machine uses exponentially many tape cells in $n$, and its unique accepting run has length at least $2^{2^n}$. For $n=3$, for example, this is an eight-bit counter with 256 values.
2. van Emde Boas's (1997) reduction turns $\mathcal M_n$ into a tiling instance $\mathcal I_n$ of width $2^{p(n)}$ for a polynomial $p$. Each row encodes one machine configuration, so correct tilings correspond to valid runs; $\mathcal I_n$ has size polynomial in $n$, and its smallest correct tiling has at least $2^{2^n}$ rows.
3. Lemmas 8–9 give a UHAT $\mathcal T_n$, polynomial in the size of $\mathcal I_n$, recognizing encodings of correct tilings of $\mathcal I_n$. The language it recognizes is $L_n$.

$$
\begin{aligned}
\text{verifier description length}&=\operatorname{poly}(n),\\
\text{shortest accepted string length}&\ge2^{2^n}.
\end{aligned}
$$

The comparison representations guarantee relatively short accepted strings when their descriptions are small. Combining those guarantees with the construction yields the separations.

| Representation | Short accepted string guaranteed by a small representation | Minimum-size gap reported in the paper |
| --- | --- | --- |
| LTL | At most exponential in formula size | Some language families require exponentially larger LTL descriptions |
| Finite automata | With q states, an accepted path shorter than q exists | Some language families require doubly exponentially many states |
| Fixed-precision RNNs | k bits per coordinate and D dimensions give at most $2^{kD}$ states | An exponential gap when precision is included in the description cost |

Because the shortest string accepted by an LTL formula is at most exponential in the formula size (via an LTL-to-automaton conversion in the style of Vardi and Wolper), an LTL formula $\varphi_n$ recognizing $L_n$ must have size exponential in $n$. As a review supplement, with a constant $c>0$:

$$
2^{2^n}\leq\min_{w\in L_n}|w|\leq 2^{c|\varphi_n|}
\Longrightarrow |\varphi_n|\geq 2^n/c.
$$

This bound is stated in the family index $n$. The proof converts it into a bound in model size: from $|\mathcal T_n|\le c_0n^d$ it gets $n\ge(|\mathcal T_n|/c_0)^{1/d}$ and hence $|\varphi_n|\ge c_1 2^{c_2'|\mathcal T_n|^{1/d}}$. Following Definition 2, the result asserts that such a witness family **exists**.

### Proposition 16: the reverse translation

LTL translates to UHAT in polynomial time with polynomial size, so UHATs have polynomially bounded expansion over LTL. The authors note that, combined with Theorem 15, this pins the UHAT–LTL gap: at most polynomial on every language and at least exponential on a witness family. [§4, Proposition 16; Appendix A.6](https://arxiv.org/pdf/2510.19315v3#page=20)

### Theorem 17 and Corollary 18: finite automata and RNNs

Theorem 17 reuses the same witness family. Any automaton recognizing a non-empty language accepts a word of length at most linear in its size, so the smallest automaton recognizing $L_n$ has size doubly exponential in $n$. Corollary 18 combines this with Proposition 1, which views an RNN of precision $k$ and dimension $D$ as an automaton with $2^{kD}$ states. As a review supplement in terms of the state count $q_n$: a shortest accepting path repeats no state and has length below $q_n$, giving the following. [§4, Theorem 17, Corollary 18](https://arxiv.org/pdf/2510.19315v3#page=9)

$$
q_n>2^{2^n},\qquad
2^{kD}\geq q_n\Longrightarrow kD>2^n.
$$

The upper bounds match. In a §1 footnote, the authors compose Proposition 13's exponential LTL translation with the standard exponential LTL-to-automaton conversion and state that every fixed-precision transformer has an equivalent finite automaton of doubly exponential size. [§1, footnote 1](https://arxiv.org/html/2510.19315v3#footnote1) In the other direction, the best known translation from counter-free automata, the finite automata equivalent to LTL, into LTL is exponential, so combined with Proposition 16, UHATs have exponentially bounded expansion over finite automata on the star-free languages. [§4](https://arxiv.org/pdf/2510.19315v3#page=9) **Review interpretation.** The standard LTL-to-automaton conversion yields nondeterministic finite automata (NFAs), so this doubly exponential upper bound is for NFAs; deterministic finite automata (DFAs) may incur additional determinization cost. Theorem 17's lower-bound argument, by contrast, applies to both NFAs and DFAs.

## 5. Applications

A small description can compress an enormous set of behaviors. How difficult is checking that two models agree on **all strings of all lengths**?

The paper establishes **EXPSPACE-completeness** for UHAT nonemptiness—whether any accepted string exists—and language equivalence. These are worst-case decision problems measured in the binary description length of the models, which the authors contrast with deterministic finite automata, where the same problems take polynomial time. [Theorems 4 and 19; Appendices A.1 and A.7](https://arxiv.org/pdf/2510.19315v3#page=10)

**Proof of Theorem 19.** For the lower bound, compare a given UHAT with a fixed UHAT recognizing the empty language. They are equivalent exactly when the given language is empty. Theorem 4 and closure of deterministic space classes under complement transfer hardness. For the upper bound, Proposition 13 translates each UHAT into LTL, then decides equivalence in space polynomial in formula size. Since formula size is exponential in original model size, total space is exponential. This concerns the relation between models over **all strings**, not the forward complexity of one supplied string. [§5, Theorem 19; Appendix A.7](https://arxiv.org/pdf/2510.19315v3#page=21)

## 6. Concluding remarks

In the related-work paragraph, the authors note that prior results establishing the close connection between UHATs, LTL, and the star-free languages (Yang et al. 2024 and others) do not address succinctness or the computational complexity of verification. Closest in angle, Sälzer et al. (2025) studied verification for transformers of various precisions and showed it to be at least NEXP-hard for fixed-precision transformers. Their technique implies that transformers can be singly exponentially more succinct than finite automata but says nothing about LTL or RNNs. The authors write that their results improve on this with doubly exponential gaps over automata and exponential gaps over LTL and RNNs. Their model is also simpler: it uses only unique-hard attention rather than Sälzer et al.'s combination of soft and hard attention, and it restricts positional information to positional masking rather than arbitrary fixed-precision positional encodings. They also mention a recent result that, without positional encodings, verification is undecidable for average-hard and softmax attention transformers with finite but unbounded precision. [Original §6](https://arxiv.org/pdf/2510.19315v3#page=10)

In the formal-verification paragraph, the authors pose as a challenge bringing automated verification techniques such as symbolic methods and simulation to practical transformer verification despite the EXPSPACE-complete worst case. Because their EXPSPACE-hardness proof requires transformers encoding large counters, they also suggest identifying subclasses that cannot encode such counters and therefore admit easier verification. They treat the learnability of succinct transformers as open, with mixed empirical evidence, and leave the succinctness of fixed-precision softmax and average-hard attention transformers to future work. [§6](https://arxiv.org/pdf/2510.19315v3#page=10)

## Review inspection of Appendix A.2: the AND in Eq. (14e)

For the address-checking Gadget B, the authors write “$B(L) = 1$ iff the binary blocks count up correctly.” Appendix A.2, Eq. (14e), requires both a normal increment and rollover at the current position when there is no earlier error; the PDF and TeX use the same connective.

<figure class="review-figure" id="paper-equation-14e">
<a href="/assets/reviews/transformers-inherently-succinct/paper-equation-14e.png" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/transformers-inherently-succinct/paper-equation-14e.png" width="2400" height="108" alt="Original Appendix A.2 Eq. (14e), whose default branch conjoins the rollover condition B₁→₀(i) and increment condition B₊₁(i)." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Paper excerpt · Counter check, Eq. (14e)</span>Look at ∧ in the default branch on the right. A valid address transition is an increment or a rollover. The earlier ∧ belongs to the search condition for a past error. <span class="figure-links"><a href="https://arxiv.org/pdf/2510.19315v3#page=15">Bergsträßer et al., arXiv v3, p.15</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · Cropped equation; original notation and fonts preserved · <a href="/assets/reviews/transformers-inherently-succinct/paper-equation-14e.png">Full size</a></span></figcaption>
</figure>

Read the equation from left to right. $Q_{\#}(j)$ means that j is a separator `#` position, and $j < i$ restricts the search to earlier positions. The black-triangle search looks for an earlier transition that is **neither a rollover nor a valid increment**. If such an error is found, it returns zero. Otherwise, the default branch after the colon checks the current transition. Here, $B_{+1}$ tests an ordinary +1 increment; $B_{1\to0}$ tests rollover from the final address to zero and also handles the initial-zero condition. [Eqs. (14c–e)](https://arxiv.org/pdf/2510.19315v3#page=15)

The connective under examination is therefore **the final ∧ in the default branch**. The earlier ∧ joins the conditions “not a rollover” and “not an increment” to identify an error.

The two default conditions can never hold together for $N\ge1$: $B_{+1}(i)$ requires some bit $B_k(i)$ to be 1, whereas $B_{1\to0}(i)$, in both its value and its default, requires every bit $B_k(i)$ to be 0. In the printed equation, therefore, $B(L)=0$ on every input, through the search result when an earlier error exists and through the default otherwise, and the output $Y(L)=A(L)\wedge B(L)\wedge C(L)\wedge D(L)\wedge E(L)$ is always 0. The printed program of Lemma 8 recognizes the empty language for every tiling instance. [Appendix A.2, Eqs. (14c–e)](https://arxiv.org/pdf/2510.19315v3#page=15)

The smallest example uses N=1 and a tile t with all four boundary colors zero: `0t#1t#`. The addresses correctly increase from zero to one, yet at the final position:

| Condition | Value |
| --- | --- |
| Normal increment $B_{+1}$ | True |
| Rollover $B_{1\to0}$ | False |
| Printed default: AND | False → valid input rejected |
| Candidate replacement: OR | True |

The [counter-only checking code](https://github.com/iiamaii/iiamaii.github.io/blob/main/research/checks/transformers-inherently-succinct/appendix-counter-check.py) provides supplementary evidence for this argument.

**Review interpretation.** As printed, the program of Lemma 8 always recognizes the empty language, so the printed proofs do not establish the lower bounds that pass through it: Propositions 5 and 10 and Theorem 4, Theorems 15 and 17 and Corollary 18, which reuse its witness family, and Theorem 19, which relies on Theorem 4. The upper bounds of Propositions 12, 13, and 16 and Corollary 14 do not depend on this construction. Replacing the default AND with OR makes Gadget B check as intended that the first block is zero and that each later block is an increment or a rollover, so a typographical error is likely, though unconfirmed by the authors. The remaining gadgets, the Lemma 9 translation, and the precision handling were not checked, so whether the corrected construction restores the theorems remains open. Confidence that the correction is needed is high.

<span id="review-interpretation-and-follow-up-checks" aria-hidden="true"></span>

## Review interpretation and open questions

**Review interpretation.** The paper's strength is that it pairs upper and lower bounds: on the LTL side, Proposition 13's exponential upper bound meets Theorem 15's exponential lower bound, and on the automata side, the doubly exponential upper bound meets Theorem 17's lower bound, fixing the scale of size gaps invisible in an expressivity hierarchy. The proof also offers a useful perspective. Using attention as a **shared rule for retrieving positions**, the verifier need not hold the whole table in one recurrent hidden state; it retrieves the earlier per-position representations needed to check a local relation. The upper bounds rest only on Proposition 12's bit-length bound and Proposition 13's translation, so they are unaffected by the Appendix A.2 issue. The lower bounds, by contrast, all rest on Lemma 8, so whether the other gadgets and the Lemma 9 translation hold after correcting (14e) to OR decides whether the claimed improvement over Sälzer et al. (2025) stands.

The scope of the results is that suitable small weights **exist**. Whether learning finds such weights, and whether fixed-precision softmax or average-hard attention shows the same gaps, are open questions left by §6.

Two questions remain. First, the **complete tiling verifier with the corrected address check could be compared against an independent oracle** to confirm that gadgets A–E together recognize the intended language. Second, one could check that the special B-RASP→UHAT translation handles absent candidates and ties correctly.

The following three sources show the background and boundaries.

| Source | Connection | Scope read for this review |
| --- | --- | --- |
| [Yang et al., NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/hash/13d7f172259b11b230cc5da8768abc5f-Abstract-Conference.html) | Expressivity correspondence between UHAT, star-free languages, and LTL | Abstract; PDF §§5.1–5.2, Theorems 5–6 |
| [Jerad et al., ACL 2025, Unique Hard Attention: A Tale of Two Sides](https://aclanthology.org/2025.acl-short.76/) | Why leftmost and rightmost ties matter | Abstract, §§2–3, Table 1 |
| [Yang et al., 2026, Length Generalization Bounds for Transformers](https://arxiv.org/abs/2603.02238v2) | How the placement of precision restrictions changes length-generalization bounds | Abstract; §5 theorem/proposition statements and discussion, not all proofs |

The last source is the work that §6 of the paper cites as an initial attempt at succinctness for softmax and average-hard attention; it relies on assumptions such as attention-internal precision and BOS/EOS, which differ from the UHAT setting without positional embeddings.

## Sources and reading scope

- Paper: *Transformers are Inherently Succinct* · Pascal Bergsträßer et al. · first public date 2025-10-22 · [reviewed version](https://arxiv.org/abs/2510.19315v3) · reviewed 2026-10-23.
- Scope read: Version 3, §§1–6 and Appendix A.
- Figure sources: The thumbnail is a review-authored explanatory diagram. The Eq. (14e) excerpt is cropped from the arXiv v3 PDF (CC BY 4.0), equation only. [Paper](https://arxiv.org/abs/2510.19315v3).
- Not verified: the proof of the tiling theorem cited by Proposition 7, and the details of gadgets other than B and of the Appendix A.3 translation.
