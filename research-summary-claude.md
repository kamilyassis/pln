# Research Summary: Persuasion Technique Detection in Malicious Brazilian-Portuguese Content

> **Status:** Work in progress — notes synthesized from research fragments  
> **Core objective:** Detect and explain textual persuasion techniques in fake news, AI-generated political propaganda, and biased blogs in Brazilian Portuguese.

---

## 1. Overview

This project targets the automatic detection of **persuasion techniques** in malicious online content, with a strong emphasis on **explainability**. The working definition of "malicious" covers fake news, politically-motivated AI-generated content, and ideologically biased blogs. The research connects classical rhetoric (ethos/pathos/logos) with modern computational propaganda taxonomies, and proposes using LLMs as scalable annotation judges to build a labeled dataset adapted to the Brazilian Portuguese context.

---

## 2. Core Research Problem

Manual fact-checking and annotation cannot scale to the volume of online misinformation. Two compounding challenges exist in the Brazilian context:

- **Data scarcity:** Existing Brazilian datasets (e.g., FactPolCheckBr, FakeBR) focus on fact-check verdicts or article-level labels, but do not annotate the **original suspicious text** with persuasion technique spans. This limits their utility for fine-grained detection.
- **Language gap:** The dominant persuasion technique taxonomies (SemEval-2020/2023) are multilingual but not specifically adapted to Portuguese or the Brazilian political discourse domain.

---

## 3. Key Concepts

### 3.1 Persuasion Technique Taxonomy

The project will adapt the **SemEval-2023 Task 3 taxonomy**, which defines **23 fine-grained techniques** organized under 6 coarse-grained categories:

| Category | Examples of Fine-Grained Techniques |
|---|---|
| Attack on Reputation | Name Calling/Labeling, Ad Hominem |
| Justification | Appeal to Authority, Appeal to Fear |
| Simplification | False Dilemma, Oversimplification |
| Distraction | Whataboutism, Red Herring |
| Call | Bandwagon, Appeal to Popularity |
| Manipulative Wording | Loaded Language, Exaggeration/Minimization |

> **Project decision:** Use the SemEval taxonomy as a base, pruning techniques with low applicability to Brazilian political content.

### 3.2 Rhetorical Framing (Ethos / Pathos / Logos)

Complementary to the SemEval taxonomy, the classical triad provides a higher-level lens:
- **Ethos** — credibility-based appeals (authority, personal attacks)
- **Pathos** — emotion-based appeals (fear, outrage, solidarity)
- **Logos** — logic-based appeals (statistics, false analogies, oversimplification)

### 3.3 Classification Tasks

Two tasks from the propaganda detection literature are considered:

- **SLC (Sentence-Level Classification):** Binary — does a sentence contain at least one propaganda technique? Lower complexity, good entry point.
- **FLC (Fragment-Level Classification):** Multi-label span detection — identify the exact text fragment and its technique type. Higher complexity and closer to explainability goals.

> **Project decision:** Prioritize **SLC** initially, with FLC as a stretch goal.

---

## 5. Annotation Strategy: LLM-as-Judge

Instead of costly multi-human annotation pipelines, the project proposes using **multiple LLMs as annotation judges** (a "jury" approach), cross-validating their outputs before training downstream classifiers.

**Rationale from literature:**
- GPT-4-class models achieve ~83.6% accuracy in text annotation tasks, marginally outperforming crowdsource workers (MTurk ~81.5%).
- Averaging scores across diverse LLMs (e.g., GPT-4, Claude, Mistral) produces judgments closer to human consensus than any single model alone — the "LLM jury" pattern.
- Known biases to mitigate: position bias, verbosity bias, self-enhancement bias.

**Recommended practice:**
- Use ensemble agreement as a proxy for Fleiss' Kappa (baseline from related work: κ = 0.86 with 4 human annotators).
- Validate a sample with human reviewers as a calibration checkpoint.
- Break annotation rubrics into atomic sub-tasks per technique to reduce ambiguity.

> **Reference benchmark:** The *Spotting Persuasion* paper used 4 independent annotators achieving Fleiss' κ = 0.86, and selected **XLNet-base** as the final classifier.

---

## 6. Model Architecture Candidates

| Model | Notes |
|---|---|
| **XLNet-base** | Selected in *Spotting Persuasion* paper; strong on sequential classification |
| **XLM-RoBERTa large** | Best performer in SemEval-2023 Task 3 persuasion subtask (KInIT team, 6/9 languages); strong multilingual baseline |
| **DistilBERT** | Lighter-weight option; viable for resource-constrained setups |
| **docBERT** | Document-level embedding; useful if classifying full articles |

**Training guidance:** Models should be fine-tuned on domain-specific data (Brazilian political content) rather than generic corpora, given distributional differences.

---

## 7. Datasets

### 7.1 Available Brazilian Datasets

| Dataset | Description | Limitation |
|---|---|---|
| **FactPolCheckBr** | Fact-checker texts, verdict labels | No original suspicious text; only fact-checker output |
| **FakeBR** (HuggingFace) | Fake news corpus in PT-BR | Article-level labels only |
| **Central de Fatos CSV** | Fact-check texts | Same issue as FactPolCheckBr — source text missing |
| **SemEval-2023 Task 3** | Multilingual (no PT), 23-label persuasion techniques | No Portuguese |

### 7.2 Proposed Data Strategy

Since labeled datasets with both the **original suspicious text** and **persuasion technique annotations** do not exist for Portuguese, the project can pursue one of two paths:

**Option A — Leverage existing fake news corpora as proxy:**  
Treat fake news articles as implicitly using persuasion techniques (supported by the literature connecting misinformation with propaganda tactics). Apply LLM-judge annotation on top.

**Option B — Scrape and annotate from controversial sources directly:**  
The following sites have been identified as ideologically biased or controversial (from Wikipédia's list of unreliable sources):
- `diariodocentrodomundo.com.br`
- `conexaopolitica.com.br`
- `brasilparalelo.com.br`
- `revistaoeste.com`
- `jornaldacidadeonline.com.br`

Combine with legitimate news sources as negative examples.

> **Recommended approach:** Combine both options — start with FakeBR as the base corpus, augment with scraped content, and annotate via LLM jury.

### 7.3 Pending Data Access

- **"Persuasion strategies of misinformation-containing posts"** (DOI: 10.1016/j.ipm.2021.102665) — email request sent to authors for dataset access.
- **SemEval-2023 Task 3** — access requested.

---

## 8. Open Questions

1. **Taxonomy pruning:** Which of the 23 SemEval techniques are most relevant to Brazilian political discourse? Should any be added (e.g., religious appeals, regional/cultural populism)?
2. **LLM judge calibration:** What is the acceptable agreement threshold between LLM judges before a label is accepted without human review?
3. **Task scope:** Is SLC (binary, sentence-level) sufficient for a first publication, or is span-level FLC required to demonstrate meaningful explainability?
4. **Domain adaptation:** How much in-domain fine-tuning data is needed for XLM-RoBERTa to perform reliably on Brazilian Portuguese political content?
5. **Evaluation:** Without a gold-standard PT-BR persuasion dataset, how will model performance be validated beyond internal agreement metrics?

---

## 9. Conclusion

The project occupies a meaningful gap at the intersection of computational propaganda detection, Brazilian NLP, and explainable AI. The core pipeline — scraping ideologically charged content, annotating persuasion techniques via LLM jury, and training an explainable classifier — is technically feasible and methodologically grounded. The SemEval-2023 taxonomy adapted to Portuguese provides a ready-made label schema, XLM-RoBERTa large is the strongest available backbone, and the LLM-as-judge approach offers a scalable path to annotation without excessive human labeling cost.

The immediate priorities are: (1) finalizing the taxonomy adaptation, (2) securing dataset access, and (3) designing the LLM annotation rubric.

---

## References

1. Piskorski, J., Stefanovitch, N., Da San Martino, G., & Nakov, P. (2023). *SemEval-2023 Task 3: Detecting the Category, the Framing, and the Persuasion Techniques in Online News in a Multi-lingual Setup.* ACL Anthology. https://aclanthology.org/2023.semeval-1.317/

2. Da San Martino, G., et al. (2019). *Fine-Grained Analysis of Propaganda in News Articles.* EMNLP-IJCNLP 2019. (Referenced in notes as `*1`)

3. Hromadka et al. / KInITVeraAI (2023). *Simple yet Powerful Multilingual Fine-Tuning for Persuasion Techniques Detection.* SemEval-2023. https://www.researchgate.net/publication/370227611

4. CLEF-2024 CheckThat! Lab Task 3. *Persuasion Techniques Detection (span-level).* https://ceur-ws.org/Vol-3740/paper-26.pdf

5. FactPolCheckBr Dataset. https://github.com/Interfaces-UFSCAR/Dataset-FactPolCheckBr

6. FakeBR Dataset. https://huggingface.co/datasets/fake-news-UFG/fakebr

7. Central de Fatos Dataset. https://journals-sol.sbc.org.br/index.php/jidm/article/view/2354

8. Persuasion strategies in misinformation posts. *Information Processing & Management.* https://doi.org/10.1016/j.ipm.2021.102665 *(dataset access pending)*

9. LLMs-as-Judges Survey (2024). https://arxiv.org/html/2412.05579v2

10. "Replacing Judges with Juries" — ensemble LLM evaluation. Referenced in Label Studio webinar. https://labelstud.io/videos/using-llm-as-a-judge-helpful-or-harmful/

11. SemEval-2023 Task 3 shared task page. https://propaganda.math.unipd.it/semeval2023task3/