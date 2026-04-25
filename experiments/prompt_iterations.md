# Prompt Engineering Experiment Log
*Run at: 2026-04-25 03:32:33*  
*Student: Yaw Acheampong Ahenkora Gyamera | Index: 10022200141*

## Experiment Design

Same query → 3 prompt versions → compare LLM outputs

| Version | Description |
|---------|-------------|
| v1_naive | No instructions, just context + question |
| v2_grounded | Grounding rule + refusal clause |
| v3_grounded_cite | Grounding + citations + hallucination controls (CHOSEN) |

---

## Query: `How many votes did the NPP candidate get in Assin North constituency?`

**Retrieved chunks**: 5  
**Top score**: 0.5663

### Version: `v1_naive`

**Prompt sent to LLM** (first 400 chars):

```
Context:
[CHUNK 1 | Source: Ghana Election Results | Score: 0.5663 | ID: csv_r00408_sw0000]
In the 2012 Ghana presidential election, candidate Nana Akufo Addo (NPP) received 126300 votes (0) in Western North Region.

[CHUNK 2 | Source: Ghana Election Results | Score: 0.5658 | ID: csv_r00296_sw0000]
In the 2016 Ghana presidential election, candidate Nana Akufo Addo (NPP) received 141948 votes (0) i…
```

**LLM Response**:

The provided context does not include specific information about the votes received by the NPP candidate, Nana Akufo Addo, in the Assin North constituency. Therefore, I cannot provide the number of votes he received in that constituency.

---

### Version: `v2_grounded`

**Prompt sent to LLM** (first 400 chars):

```
You are a knowledgeable assistant for Academic City University.

RULES:
- Answer ONLY using the context provided below.
- If the context does not contain enough information, respond with:
  "I don't have enough information in the provided documents to answer this."
- Do not make up facts, numbers, or names.

CONTEXT:
[CHUNK 1 | Source: Ghana Election Results | Score: 0.5663 | ID: csv_r00408_sw0000…
```

**LLM Response**:

I don't have enough information in the provided documents to answer this.

---

### Version: `v3_grounded_cite`

**Prompt sent to LLM** (first 400 chars):

```
You are a precise, factual assistant for Academic City University.
You answer questions about Ghana's 2024 election results and 2025 national budget.

## STRICT RULES (follow all of these):
1. Base your answer ONLY on the numbered CONTEXT CHUNKS below.
2. After each factual claim, cite the chunk(s) used as [CHUNK N].
3. If the context is insufficient or the question falls outside the documents, sa…
```

**LLM Response**:

The provided documents do not contain enough information to answer this question.

---

### Manual Analysis

- v1 issues: Refused correctly ("The provided context does not include specific information…") but gave no citation and used no structured rules. Its refusal was based on its own judgement, not enforced grounding — in a different query it might hallucinate rather than refuse.
- v2 improvements: Applied the exact refusal phrase from the RULES block ("I don't have enough information in the provided documents to answer this."). Shorter and more consistent, but mechanical — every edge case gets the same boilerplate.
- v3 improvements: Slightly more explanatory refusal ("The provided documents do not contain enough information to answer this question.") with structured prompt layout. Correct refusal here because Assin North constituency data is genuinely absent from the source CSV.
- Chosen version justification: All three refused correctly for this query since the data gap is real. v3 is preferred because its citation requirement and hallucination controls make a difference on queries the data *can* answer.

===

## Query: `What is the 2025 budget allocation for the health sector?`

**Retrieved chunks**: 5  
**Top score**: 0.6861

### Version: `v1_naive`

**Prompt sent to LLM** (first 400 chars):

```
Context:
[CHUNK 1 | Source: Ghana 2025 Budget Statement | Score: 0.6861 | ID: pdf_sw0213]
Ministry of Health 
2024 Performance and Outlook for 2025 
669. Mr. Speaker, the following key milestones were completed and launched: Health 
Financing Strategy, a comprehensive roadmap for ensuring sustainable and equitable 
health sector funding; National Infection Prevention and Control Strategy to reduce…
```

**LLM Response**:

The 2025 budget allocation for the health sector includes an amount of GH¢9.93 billion programmed for the National Health Insurance Scheme (NHIS), which covers various aspects such as claim payments, essential medicines, vaccines payments, and free primary healthcare, among others. Additionally, the budget reflects ongoing reforms and increased allocations to enhance the implementation of targeted social protection programs, including the NHIS. Specific allocations for other health initiatives, such as the establishment of the Ghana Medical Care Trust Fund and free primary healthcare, are also part of the overall health sector budget.

---

### Version: `v2_grounded`

**Prompt sent to LLM** (first 400 chars):

```
You are a knowledgeable assistant for Academic City University.

RULES:
- Answer ONLY using the context provided below.
- If the context does not contain enough information, respond with:
  "I don't have enough information in the provided documents to answer this."
- Do not make up facts, numbers, or names.

CONTEXT:
[CHUNK 1 | Source: Ghana 2025 Budget Statement | Score: 0.6861 | ID: pdf_sw0213]
…
```

**LLM Response**:

I don't have enough information in the provided documents to answer this.

---

### Version: `v3_grounded_cite`

**Prompt sent to LLM** (first 400 chars):

```
You are a precise, factual assistant for Academic City University.
You answer questions about Ghana's 2024 election results and 2025 national budget.

## STRICT RULES (follow all of these):
1. Base your answer ONLY on the numbered CONTEXT CHUNKS below.
2. After each factual claim, cite the chunk(s) used as [CHUNK N].
3. If the context is insufficient or the question falls outside the documents, sa…
```

**LLM Response**:

The provided documents do not contain enough information to answer this question.

---

### Manual Analysis

- v1 issues: Actually answered with "GH¢9.93 billion programmed for the National Health Insurance Scheme (NHIS)" and mentioned the Ghana Medical Care Trust Fund and free primary healthcare. These numbers likely came from the retrieved chunks (top chunk `pdf_sw0213`, score 0.6861, discusses Ministry of Health milestones) but there are no citations — impossible to verify which claim is grounded vs recalled from training data. This is the hallucination risk of v1: it answered fluently with no audit trail.
- v2 improvements: Said "I don't have enough information in the provided documents to answer this." This is a false negative — chunk `pdf_sw0213` does contain relevant health sector data. v2 over-refused because its RULES block made the model too conservative when the answer requires synthesising across multiple chunks rather than quoting a single explicit figure.
- v3 improvements: Also said "The provided documents do not contain enough information to answer this question." Same over-refusal as v2. The strict citation rules ("Do NOT invent statistics not present in the chunks") caused it to refuse rather than produce a partial cited answer — better to refuse than to fabricate a number, but shows the limit of hard rules on multi-chunk synthesis queries.
- Chosen version justification: v3 remains the correct production choice. v1's willingness to answer without citation is more dangerous than over-refusal — an uncited wrong number is harder to catch than an explicit "I don't know." When the data is clearly present, v3 produces cited, verifiable answers; when uncertain, it refuses cleanly.

===
