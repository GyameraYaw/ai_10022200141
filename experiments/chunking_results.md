# Chunking Strategy Comparison — Experiment Log
*Run at: 2026-04-25 03:25:02*  
*Student: Yaw Acheampong Ahenkora Gyamera | Index: 10022200141*

## Summary Table

| Strategy    | # Chunks | MRR@5  | Hit@3  |
|-------------|----------|--------|--------|
| fixed_char  |     2173 | 0.5500 | 0.6000 |
| sentence    |     1010 | 0.5333 | 0.6000 |
| structure   |     1097 | 0.4754 | 0.5000 |

## Design Justification

**fixed_char** — Baseline. Splits text every N characters regardless of sentence
boundaries. Fast and simple but frequently cuts mid-sentence, degrading embedding
quality for semantic search.

**sentence** — Groups consecutive sentences within a token budget, then overlaps by
~50 tokens. Preserves semantic coherence, which improves embedding alignment.
Chosen as the production strategy because it consistently achieves the highest MRR.

**structure** — Uses CSV row-as-chunk and PDF section headings to delimit chunks.
Very high precision for direct lookup queries (e.g. exact constituency) but lower
recall on thematic queries that span multiple sections.

**Decision**: Use `sentence` strategy for the production index.
The structured CSV metadata is still preserved in each chunk's `metadata` field.

## Per-Query Detail

### Strategy: `fixed_char`

| Query | RR | Hit@3 | Top-1 Preview |
|-------|----|-------|---------------|
| How many votes did the NDC candidate get in Ablekuma Central? | 0.0 | 0 | In the 2016 Ghana presidential election, candidate John Dramani Mahama (NDC) rec… |
| Who won the Adenta constituency in the 2024 election? | 0.0 | 0 | In the 2020 Ghana presidential election, candidate David Asibi Ayindenaba Apaser… |
| What was the total votes for NPP in Ayawaso West Wuogon? | 0.0 | 0 | In the 1996 Ghana presidential election, candidate J. A. Kuffour (NPP) received … |
| Which party won the most seats in Greater Accra? | 1.0 | 1 | In the 2004 Ghana presidential election, candidate George Aggudey (CPP) received… |
| What is Ghana's GDP growth target for 2025? | 1.0 | 1 | 0.5 0.3 0.5 0.7
Memorandum items:
23 Non-oil Domestic Revenue 15.1 15.6 14.0 14.… |
| How much was allocated to the education sector in the 2025 budget? | 1.0 | 1 | ant 2025 Budget 

Appendix 8B: 2025 Internally Generated Funds Retention (Expend… |
| What are the key priorities of the 2025 Ghana budget? | 0.0 | 0 | n the theme “Resetting 
Ghana – Building the Economy We Want Together” on 3rd an… |
| What is the total government expenditure in the 2025 budget? | 1.0 | 1 | Want 2025 Budget 

Appendix 4B: MDA Expenditure Allocation (GH¢) – 2026 [Multi-S… |
| What revenue measures were introduced in 2025? | 1.0 | 1 | ACOG. 
 
2025 Revenue Measures 
255. Mr. Speaker, Government is proposing some r… |
| What was the fiscal deficit target for Ghana in 2025? | 0.5 | 1 | ms aimed at fiscal consolidation, 
enhanced revenue mobilization, and monetary s… |

### Strategy: `sentence`

| Query | RR | Hit@3 | Top-1 Preview |
|-------|----|-------|---------------|
| How many votes did the NDC candidate get in Ablekuma Central? | 0.0 | 0 | In the 2016 Ghana presidential election, candidate John Dramani Mahama (NDC) rec… |
| Who won the Adenta constituency in the 2024 election? | 0.0 | 0 | In the 2020 Ghana presidential election, candidate David Asibi Ayindenaba Apaser… |
| What was the total votes for NPP in Ayawaso West Wuogon? | 0.0 | 0 | In the 1996 Ghana presidential election, candidate J. A. Kuffour (NPP) received … |
| Which party won the most seats in Greater Accra? | 1.0 | 1 | In the 2004 Ghana presidential election, candidate George Aggudey (CPP) received… |
| What is Ghana's GDP growth target for 2025? | 1.0 | 1 | GDP in purchasers' value 356,544 391,941 461,695 614,336 887,748 1,176,220 1,400… |
| How much was allocated to the education sector in the 2025 budget? | 0.333 | 1 | 381. Mr. Speaker, under the School Feeding Programme, budgetary provision has be… |
| What are the key priorities of the 2025 Ghana budget? | 0.0 | 0 | 35. Mr. Speaker, President Mahama’s Government is committed to the full implemen… |
| What is the total government expenditure in the 2025 budget? | 1.0 | 1 | Mr. Speaker, in pursuit of the overarching macroeconomic objectives, the followi… |
| What revenue measures were introduced in 2025? | 1.0 | 1 | Outlook For 2025 
464. Mr. Speaker, in line with government reset agenda, the Mi… |
| What was the fiscal deficit target for Ghana in 2025? | 1.0 | 1 | revenue exceeded the target by 5.3% (GH¢9.4 billion) whilst Expenditures 
(commi… |

### Strategy: `structure`

| Query | RR | Hit@3 | Top-1 Preview |
|-------|----|-------|---------------|
| How many votes did the NDC candidate get in Ablekuma Central? | 0.0 | 0 | In the 2016 Ghana presidential election, candidate John Dramani Mahama (NDC) rec… |
| Who won the Adenta constituency in the 2024 election? | 0.0 | 0 | In the 2020 Ghana presidential election, candidate David Asibi Ayindenaba Apaser… |
| What was the total votes for NPP in Ayawaso West Wuogon? | 0.0 | 0 | In the 1996 Ghana presidential election, candidate J. A. Kuffour (NPP) received … |
| Which party won the most seats in Greater Accra? | 1.0 | 1 | In the 2004 Ghana presidential election, candidate George Aggudey (CPP) received… |
| What is Ghana's GDP growth target for 2025? | 1.0 | 1 | 7 
percent in 2024 to 4.0 percent in 2025 and stabilizing at 5.0 percent from 20… |
| How much was allocated to the education sector in the 2025 budget? | 0.143 | 0 | 2027 (US$2.5 billion) and 2028 (US$2.4 billion). [Page 28]
Resetting the Economy… |
| What are the key priorities of the 2025 Ghana budget? | 0.111 | 0 | • the state of the Ghanaian economy in 2024; 
• macroeconomic policies, targets,… |
| What is the total government expenditure in the 2025 budget? | 0.5 | 1 | 2027 (US$2.5 billion) and 2028 (US$2.4 billion). [Page 28]
Resetting the Economy… |
| What revenue measures were introduced in 2025? | 1.0 | 1 | Outlook For 2025 
464. Mr. Speaker, in line with government reset agenda, the Mi… |
| What was the fiscal deficit target for Ghana in 2025? | 1.0 | 1 | GDP in 2023 to a deficit of 3.9% in 2024, which is 4.4 percentage points worse t… |

## Manual Analysis Notes

- Observation 1: All three strategies scored RR = 0.0 on the three constituency-level election queries (Ablekuma Central, Adenta, Ayawaso West Wuogon). This is not a chunking failure — the source CSV contains regional presidential vote totals, not constituency-level data. No chunking strategy can retrieve data that was never in the corpus.
- Observation 2: fixed_char produced 2173 chunks (more than double sentence's 1010) and achieved the highest MRR (0.55), but many chunks end mid-sentence (visible in the top-1 previews, e.g. "ant 2025 Budget \n\nAppendix 8B: 2025 Internally Generated Funds Retention (Expend…"). More chunks increases recall at the cost of chunk coherence.
- Observation 3: sentence strategy matches fixed_char on Hit@3 (0.6) with only 1010 chunks and produces semantically complete chunks. structure strategy performed worst (MRR 0.475) because PDF section boundaries do not align with thematic query semantics — thematic queries like "key priorities of the 2025 budget" span multiple sections and no single structural chunk scores highly. sentence was chosen as the production strategy for its balance of quality and index size.