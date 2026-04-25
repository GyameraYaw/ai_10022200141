# Manual Experiment Log
**Student:** Yaw Acheampong Ahenkora Gyamera | **Index:** 10022200141  
**Course:** CS4241 — Introduction to Artificial Intelligence  
**Date:** 2026-04-25  
**System:** Manual RAG chatbot (no LangChain / LlamaIndex), gpt-4o-mini, all-MiniLM-L6-v2

---

## Part A — Chunking Strategy Comparison

### Setup
Three chunking strategies were evaluated against the same 10 test queries (5 election, 5 budget) using MRR@5 and Hit@3 as metrics. The index was rebuilt fresh for each strategy.

### Results

| Strategy   | # Chunks | MRR@5  | Hit@3 |
|------------|----------|--------|-------|
| fixed_char | 2173     | 0.5500 | 0.600 |
| sentence   | 1010     | 0.5333 | 0.600 |
| structure  | 1097     | 0.4754 | 0.500 |

### Detailed Observations

**1. The three election constituency queries failed across ALL strategies (RR = 0.0 for "Ablekuma Central", "Adenta", "Ayawaso West Wuogon").**

This was the most important finding of this experiment. At first I thought this was a chunking failure, but after inspecting the CSV it is actually a data limitation: the election CSV contains **regional presidential vote totals** (e.g., "NDC received X votes in Greater Accra Region"), not constituency-by-constituency breakdowns. No chunking strategy can retrieve data that was never in the source. This is a known gap documented in the project — the dataset covers 1992–2020 regional results only.

**2. fixed_char produced 2173 chunks — more than double sentence strategy's 1010.**

When I reviewed the top-1 preview for fixed_char budget queries, many chunks ended mid-sentence (e.g., "ant 2025 Budget \n\nAppendix 8B: 2025 Internally Generated Funds Retention (Expend…"). This is a direct consequence of character-boundary splitting ignoring sentence structure. Despite this noise, fixed_char achieved the highest MRR (0.55) because more chunks means more candidate passages — the relevant one is more likely to appear somewhere in the top 5.

**3. sentence strategy achieved the same Hit@3 as fixed_char (0.6) with far fewer chunks (1010).**

The sentence-grouped chunks are cleaner and more semantically coherent. For example, the top-1 for "What is Ghana's GDP growth target for 2025?" under sentence strategy retrieved a chunk starting with "GDP in purchasers' value 356,544 391,941…" — a complete data table, undivided. Under fixed_char the same query retrieved a chunk beginning mid-table. Although sentence scored slightly lower MRR (0.5333 vs 0.55), it is a better production choice because:
- Fewer chunks = faster retrieval and cheaper embedding
- Chunks align with semantic units, improving LLM comprehension
- The MRR difference of 0.017 is within noise given the 10-query test set

**4. structure strategy performed worst (MRR 0.475, Hit@3 0.5).**

The structural chunker uses CSV row boundaries and PDF section headings. This worked well for direct lookup queries ("What revenue measures were introduced in 2025?" RR = 1.0 in all three strategies) but collapsed on thematic queries that span multiple sections ("What are the key priorities of the 2025 Ghana budget?" — RR = 0.0 for fixed_char and sentence, RR = 0.111 for structure). The section-level chunks for the budget are too long and contain mixed topics, degrading cosine similarity.

**5. Decision rationale for choosing sentence strategy:**

The sentence strategy offers the best balance of retrieval quality and index size. Hit@3 is identical to fixed_char, MRR is only marginally lower, and the chunks are semantically coherent, which helps the LLM produce grounded answers. The structured CSV metadata (region, year, candidate, party) is preserved in each chunk's `metadata` field regardless of the strategy, so no information is lost.

---

## Part B — Retrieval Failure Analysis

### Setup
Three queries known to stress-test the retriever were run against both pure vector search and hybrid search (BM25 α=0.70 + vector α=0.30). Hit@3 was recorded for each.

### Results

| Query | Vector Hit@3 | Hybrid Hit@3 |
|-------|-------------|-------------|
| How many votes did the NDC win in Amenfi East? | ❌ NO | ❌ NO |
| What is the 2025 ABFA allocation in Ghana cedis? | ✅ YES | ✅ YES |
| Bunkpurugu-Nakpayili constituency NPP candidate | ❌ NO | ❌ NO |

### Detailed Observations

**Case 1 — Amenfi East (NDC votes):**

Vector-only returned five 1992 NDC chunks from unrelated regions (scores 0.539–0.556). None mentioned Amenfi East. Hybrid reranked slightly: the top combined score jumped from 0.556 to 0.656 because BM25 boosted all five results equally (BM25 = 0.8878 for all). The uniform BM25 score is the tell — BM25 could match the token "NDC" across many chunks but found no chunk containing the exact string "Amenfi East". This confirms the constituency-level data gap: the source CSV simply does not contain Amenfi East data. Hybrid search cannot recover data that does not exist.

**Case 2 — ABFA (2025 budget):**

Vector-only succeeded (Hit@3 = YES, top score 0.661). Hybrid also succeeded and improved ranking: the correct chunk (`pdf_sw0132`, containing "ABFA" and petroleum receipt data) moved to rank 1 (combined 0.720) due to a BM25 score of 1.0 — a perfect keyword match on the rare acronym "ABFA". This is exactly the failure mode hybrid search was designed to address: the embedding model treats "ABFA" as an out-of-vocabulary sub-word token and ranks it lower than semantically adjacent budget chunks, but BM25 finds the exact character sequence.

**Case 3 — Bunkpurugu-Nakpayili:**

This case illustrates the difference between a vocabulary problem and a data problem. Vector-only retrieved five NPP chunks (scores 0.554–0.572) — all related to NPP vote totals but from entirely different constituencies. Hybrid produced identical top-5 results with minimal score adjustment. Crucially, all hybrid BM25 scores were identical at 0.5085. This means BM25 matched on "NPP" and "candidate" but found nothing matching "Bunkpurugu" — again confirming the data does not contain this specific constituency record.

**Key insight from Part B:**

Hybrid search (BM25 + vector) is effective for vocabulary-mismatch failures where the relevant document *does* exist in the corpus but the embedding model loses specificity on rare tokens (as in Case 2 — ABFA). However, hybrid search cannot compensate for missing data (Cases 1 and 3). The practical implication is that the system should be transparent with users about its data coverage — it does not have constituency-level presidential election results.

---

## Part C — Prompt Engineering Comparison

### Setup
The same two queries were passed through all three prompt versions (v1_naive, v2_grounded, v3_grounded_cite) with identical retrieved chunks. Responses were compared for grounding, hallucination, and citation behaviour.

### Detailed Observations

**Query 1: "How many votes did the NPP candidate get in Assin North constituency?"**

All three prompt versions produced refusals, but for different reasons and with different formulations:

- **v1_naive**: "The provided context does not include specific information about the votes received by the NPP candidate…in the Assin North constituency." This is a reasonable refusal, but the phrasing ("The provided context does not include") reveals that v1 correctly read the context — it just had no instructions, so it relied on its own judgement to refuse. There is no guarantee it would always refuse. In an alternative run with a different query where the context is ambiguous, v1 might hallucinate.
- **v2_grounded**: "I don't have enough information in the provided documents to answer this." Short, template-matching the exact refusal phrase from the RULES block. Correct, but rigid — every edge case gets the same boilerplate response.
- **v3_grounded_cite**: "The provided documents do not contain enough information to answer this question." Slightly more explanatory than v2, follows the structured RULES format. Same correct refusal.

For this query, all three behave similarly because the correct answer is genuinely absent from the data.

**Query 2: "What is the 2025 budget allocation for the health sector?"**

This query produced the most revealing difference between versions:

- **v1_naive** actually *answered*: "The 2025 budget allocation for the health sector includes an amount of GH¢9.93 billion programmed for the National Health Insurance Scheme (NHIS)...". It also mentioned the Ghana Medical Care Trust Fund and free primary healthcare. These numbers appear to come from the retrieved chunks — the top chunk (pdf_sw0213) discusses the Ministry of Health milestones. However, there are **no citations**: I cannot verify which claim came from which chunk. The answer may also blend retrieved text with the model's prior knowledge, which is the hallucination risk.
- **v2_grounded**: "I don't have enough information in the provided documents to answer this." This is **factually incorrect** — the top retrieved chunk (score 0.6861, `pdf_sw0213`) clearly contains health sector data from the 2025 budget. v2 over-refused. The RULES block ("Answer ONLY using the context") appeared to make the model too conservative — it applied the refusal clause when the context was actually sufficient.
- **v3_grounded_cite**: "The provided documents do not contain enough information to answer this question." Same over-refusal as v2. This was the most disappointing result: v3 should be the best version, but both v2 and v3 refused a question the data *can* answer.

**Why v3 over-refused on the health query:**

Inspecting the retrieved chunks, they contain narrative budget paragraphs about the Ministry of Health milestones (e.g., "Health Financing Strategy", "National Infection Prevention and Control Strategy") but the specific numerical allocation (GH¢ total) is not explicitly stated as a single figure in the top-5 chunks. v3's strict rules ("Do NOT invent statistics not present in the chunks") caused it to refuse rather than synthesise a partial answer with caveats. This is the correct conservative behaviour — better to refuse than to hallucinate a number — but it shows a limitation of overly strict rules when the relevant data is spread across multiple chunks.

**Conclusion:**

v3_grounded_cite remains the correct production choice because:
1. It is the only version that can produce cited, verifiable answers when the data *is* present
2. Its conservative refusal on ambiguous queries prevents hallucination
3. v1's willingness to answer without citation is more dangerous than over-refusal — an uncited wrong number is worse than an explicit "I don't know"

v2 is not clearly better than v1 for this data: it over-refuses and its refusal phrasing is mechanical.

---

## Part D — Adversarial Evaluation

### Setup
Two adversarial queries were tested: one maximally ambiguous query ("Who won?") and one misleading-year query ("2026 budget allocation for education"). Each was run through the full RAG pipeline and through a pure LLM call (no retrieval). Consistency was measured across 5 runs.

### Detailed Observations

**ADV1 — Ambiguous query: "Who won?"**

- **RAG response**: "The provided documents do not contain enough information to answer this question."
- **Pure LLM response**: "Could you please provide more context or specify what event or competition you are referring to regarding 'who won' in Ghana?"
- **RAG consistency**: 1.0 (identical response all 5 runs)
- **LLM consistency**: 1.0 (identical response all 5 runs)
- **Hallucination rate**: 0.0 for both (no numeric claims made)

Both systems handled this case well. The RAG pipeline grounded the query against retrieved chunks (which were unrelated budget passages, not election win data), found no match, and refused cleanly. The LLM correctly identified the ambiguity and asked for clarification — it did not guess "NDC won" or fabricate election results.

The perfect consistency (1.0 for both) reflects the determinism of a refusal response: when neither system has anything useful to say, they say the same thing every time.

**ADV2 — Misleading year: "What was the 2026 budget allocation for education in Ghana?"**

- **RAG response**: "The provided documents do not contain enough information to answer this question."
- **Pure LLM response** (representative): "As of my last knowledge update in October 2023, the specific budget allocation for education in Ghana for the year 2026 had not been publicly released…"
- **RAG consistency**: 1.0 (identical refusal all 5 runs)
- **LLM consistency**: 0.6703 (Jaccard mean, min=0.529, max=0.900)
- **LLM hallucination rate**: 1.0 (2 numeric claims, both unsupported)

This is the most important adversarial result. The RAG system correctly refused: the index contains 2025 budget data, not 2026, and the v3 prompt rules prevented the model from inventing 2026 figures. The refusal is consistent 100% of the time.

The pure LLM, by contrast, gave inconsistent answers across runs (Jaccard 0.67). Across the 5 runs it sometimes admitted uncertainty ("had not been publicly released"), sometimes fabricated specific allocation percentages, and sometimes cited a 2023 knowledge cutoff. The logged numeric claims (2 unsupported claims) came from a run where the LLM asserted figures like "approximately 18% of Ghana's total budget" — a plausible-sounding hallucination that has no grounding in any real document. This is precisely the failure mode RAG is designed to prevent.

**Summary — RAG vs pure LLM on adversarial queries:**

| Metric | RAG | Pure LLM |
|--------|-----|---------|
| ADV1 consistency | 1.0 | 1.0 |
| ADV2 consistency | 1.0 | 0.67 |
| ADV2 hallucination rate | 0.0 | 1.0 |

The RAG pipeline is demonstrably more reliable for this domain. Its grounding mechanism forces the model to only use retrieved text, making it deterministic and hallucination-free on out-of-scope queries. The pure LLM has no such constraint — it fills knowledge gaps with plausible-sounding but fabricated details, and it does so inconsistently.
