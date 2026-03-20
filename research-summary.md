# Technical Summary: Persuasion-Technique Detection in Malicious Portuguese Content

## Overview

The research direction emerging from the notes is not generic fake-news detection. The central idea is to build an **explainable system for identifying textual persuasion techniques** in malicious or strongly biased content, especially in Brazilian Portuguese.

In this project, "malicious content" includes at least four families of material:

- fake news and misinformation
- AI-generated political propaganda
- hyperpartisan or tendentious blogs
- political or commercial campaign material designed to manipulate perception

The main differentiator is **explainability**. Instead of only predicting whether a text is false, biased, or propagandistic, the system should indicate **which persuasive technique appears in the text and where it appears**.

## Core Research Thesis

The notes support the following thesis:

> Malicious or misleading content often relies on recurring persuasion techniques. If these techniques can be labeled and detected in Portuguese texts, they can serve as interpretable signals for downstream tasks such as misinformation analysis, political-ad analysis, and source credibility assessment.

This framing is stronger than a simple truthfulness classifier because it focuses on **how manipulation is performed**, not only on whether a claim is true or false.

## Key Concepts

### 1. Persuasion as the primary object of analysis

The project should focus on **textual persuasion techniques** rather than treating propaganda as a purely source-level property. This avoids the weak assumption that all content from a suspicious source is uniformly manipulative.

The literature reviewed in the notes supports moving from coarse document labels to more granular annotation:

- fragment-level detection makes the prediction explainable
- sentence-level detection is simpler and more feasible as a first stage
- document-level labels alone are too noisy for this goal

### 2. Ethos, pathos, logos as a high-level frame

One of the user’s original ideas is to use the classical rhetoric frame of **ethos, pathos, and logos** as a conceptual umbrella, while operational labels come from modern propaganda-taxonomy papers.

This is a good compromise:

- ethos, pathos, logos provide an intuitive top-level explanation layer
- SemEval-style persuasion labels provide concrete annotatable categories

The social-media misinformation paper is especially relevant here because it explicitly analyzes persuasion through an Aristotelian lens and finds that **pathos-oriented strategies are especially common** in misinformation-containing posts.

### 3. Explainability should shape task design

The notes repeatedly emphasize explainability, and the reviewed papers support that choice. Fine-grained propaganda work shows that span- or sentence-level predictions are useful because they reveal **why** a text was flagged.

For this reason, the project should treat explainability as a design requirement, not a later add-on.

## Findings From the Reviewed Materials

### A. Fine-grained propaganda detection is the strongest methodological base

The most important methodological reference is *Fine-Grained Analysis of Propaganda in News Articles*. It defines 18 propaganda techniques and explicitly frames fine-grained detection as a way to support explainable AI. It also distinguishes:

- `SLC` (Sentence-level Classification): predict whether a sentence contains at least one technique
- `FLC` (Fragment-level Classification): identify the span and the technique type

This aligns closely with the notes, especially the current preference for starting with `SLC` and possibly expanding later to `FLC`.

### B. SemEval-style persuasion taxonomies are useful, but should be adapted

The SemEval 2023 persuasion task expands the label space to 23 persuasion techniques in a multilingual setting. This is useful as a **starting ontology**, but probably too broad to adopt unchanged for a Brazilian Portuguese corpus.

The notes already point in the right direction: use a **curated subset** of the existing taxonomy and remove labels that are too rare, too ambiguous, or too dependent on external world knowledge.

### C. Efficient models are viable

The notes mention XLNet, DistilBERT, APatt, and KInIT. The reviewed materials support a pragmatic interpretation:

- high-performing SemEval systems used large multilingual transformers such as XLM-RoBERTa
- lightweight alternatives can still be competitive in persuasion detection
- XLNet-base appears to offer a reasonable performance/efficiency tradeoff

This suggests a practical modeling path: begin with a strong but manageable encoder, and avoid overcommitting to very large models before the annotation scheme stabilizes.

### D. LLMs are more promising for annotation support than for final detection

The notes propose an "LLM judge" approach with multiple models agreeing on labels. The literature does not support using zero-shot GPT-style models as the main detector. Recent evidence instead suggests:

- span-level propaganda annotation is difficult and subjective
- fine-tuned task-specific models outperform GPT-4 in fine-grained detection
- a structured multi-step annotation process improves consistency

This means the strongest use of LLMs here is probably:

- draft labeling
- disagreement surfacing
- annotator support
- label consolidation

not replacing a supervised detector trained on curated in-domain data.

### E. Existing Brazilian fact-check datasets are useful, but insufficient on their own

The notes correctly identify a major limitation in datasets such as FactCenter / FactPolCheckBr: they are valuable as repositories of verified misinformation cases, but often emphasize the **fact-check narrative** rather than the **original text being judged**.

For persuasion-technique detection, the original text matters more than the later verification article. This is an important project-level insight.

One nuance is important here: a **2025 FactPolCheckBr data paper** describes retrieval of the full texts of fake-news items from the web. That makes FactPolCheckBr potentially more useful than earlier fact-check-story repositories that mainly expose checker-authored narratives. Even so, the broader conclusion still holds: these resources are most valuable when they preserve or help recover the **original manipulated text**.

Therefore, fact-check datasets are best treated as:

- a source of topics, claims, and suspicious URLs
- a way to identify recurring misinformation themes
- a weak supervision or collection aid

not as the final training corpus by themselves.

## Proposed Project Direction

### Recommended task formulation

The clearest first milestone is:

1. Build a **sentence-level classifier** for persuasion-technique presence in Brazilian Portuguese malicious content.
2. Predict either:
   - binary persuasive/non-persuasive labels, or
   - a reduced multi-label set of persuasion techniques.
3. Preserve the sentence evidence so the system remains explainable.

After that, the project can move to fragment-level span extraction if the annotation process proves stable enough.

### Recommended data strategy

The data strategy implied by the notes is strong and should be preserved:

- crawl original texts from suspicious or hyperpartisan Brazilian sources
- collect campaign and advertising material, especially political ads
- include AI-generated video transcripts when available
- use fact-check repositories to discover disputed claims and candidate collection targets

This supports a corpus centered on **original malicious discourse**, which is the right unit for persuasion analysis.

An adjacent Portuguese benchmark, HateBRXplain, is also relevant as methodological inspiration: it shows the value of human-annotated rationales for explainable classification, even though its task is hate speech rather than persuasion.

### Recommended annotation workflow

The reviewed materials suggest a practical workflow:

1. Curate guidelines with a reduced set of labels and examples in Portuguese.
2. Run pilot annotation on a small sample.
3. Use multi-annotator labeling plus adjudication or consolidation.
4. Optionally use multiple LLMs as pre-annotators or disagreement detectors.
5. Train a smaller supervised model on the finalized corpus.

This is consistent with the user’s idea of agreement-based labeling while staying aligned with evidence from recent propaganda-annotation studies.

## Open Questions

- Which exact subset of persuasion techniques should be kept for Portuguese annotation?
- Should the first release target only political content, or a broader set including fake news and commercial persuasion?
- How will AI-generated videos be converted into analyzable text: transcript only, transcript plus metadata, or multimodal later?
- What is the minimum acceptable agreement threshold for the annotation pipeline?
- Will the first benchmark optimize for sentence-level classification only, or also include rationale spans?

## Conclusion

The project is best understood as an **explainable persuasion-analysis pipeline for malicious Portuguese content**, not as a standard fake-news classifier. The strongest version of the idea is to adapt fine-grained propaganda and persuasion taxonomies to Brazilian Portuguese, begin with sentence-level detection, use LLMs to support annotation rather than replace supervised modeling, and prioritize original source texts over fact-check summaries.

If executed this way, the project can produce a corpus and model that are both technically useful and analytically interpretable.

## References

1. Da San Martino, G., Yu, S., Barrón-Cedeño, A., Petrov, R., and Nakov, P. *Fine-Grained Analysis of Propaganda in News Articles*. EMNLP-IJCNLP 2019. Local copy: `articles/D19-1565.pdf`. https://aclanthology.org/D19-1565/
2. Hromadka, T. et al. *KInITVeraAI at SemEval-2023 Task 3: Simple yet Powerful Multilingual Fine-Tuning for Persuasion Techniques Detection*. Local copy: `articles/SemEval_2023_KInIT_at_SemEval_2023_Task_3.pdf`.
3. Wu, B. et al. *SheffieldVeraAI at SemEval-2023 Task 3: Mono and multilingual approaches for news genre, topic and persuasion technique classification*. Local copy: `articles/sheffieldveraai_semeval_3.pdf`.
4. Meguellati, E. et al. *Spotting Persuasion: A Low-cost Model for Persuasion Detection in Political Ads on Social Media*. Local copy: `articles/Spotting Persuasion: A Low-cost Model for Persuasion Detection.pdf`.
5. Hasanain, M., Ahmed, F., and Alam, F. *Can GPT-4 Identify Propaganda? Annotation and Detection of Propaganda Spans in News Articles*. LREC-COLING 2024. Local copy: `articles/2024.lrec-main.244.pdf`.
6. Ahmad, P. N. et al. *Robust Benchmark for Propagandist Text Detection and Mining High-Quality Data*. *Mathematics* 2023. Local copy: `articles/Robust_Benchmark_for_Propagandist_Text_Detection_a.pdf`.
7. D’Ulizia, A., Caschera, M. C., Ferri, F., and Grifoni, P. *Fake news detection: a survey of evaluation datasets*. *PeerJ Computer Science* 2021. Local copy: `articles/peerj-cs-07-518.pdf`. https://doi.org/10.7717/peerj-cs.518
8. Marques, I. et al. *A Comprehensive Dataset of Brazilian Fact-Checking Stories*. *Journal of Information and Data Management* 2022. https://journals-sol.sbc.org.br/index.php/jidm/article/view/2354
9. Chen, S., Xiao, L., and Mao, J. *Persuasion strategies of misinformation-containing posts in the social media*. *Information Processing & Management* 58(5), 2021. https://doi.org/10.1016/j.ipm.2021.102665
10. Iasulaitis, S. et al. *FactPolCheckBr: a dataset of fake news fact-checked during the 2022 Brazilian presidential elections*. *Latin American Data in Science* 2025. https://ojs.datainscience.com.br/lads/article/view/76
11. Salles, I., Vargas, F., and Benevenuto, F. *HateBRXplain: A Benchmark Dataset with Human-Annotated Rationales for Explainable Hate Speech Detection in Brazilian Portuguese*. COLING 2025. https://aclanthology.org/2025.coling-main.446/
