# Related Work Digest (research notes for the Anima design paper)

Compiled July 2026 via web research. Verified items only; leads flagged where not fully read.

## 1. MemGPT / Letta — virtual context management
Packer, Wooders, Lin, Fang, Patil, Gonzalez (2023). "MemGPT: Towards LLMs as Operating Systems." arXiv:2310.08560. https://arxiv.org/abs/2310.08560
OS virtual-memory metaphor: context window = main memory, external stores = disk; LLM function calls page information in/out and self-edit context. Now the open-source Letta framework. https://docs.letta.com/letta-memgpt

## 2. Generative Agents — memory stream, weighted retrieval, reflection
Park, O'Brien, Cai, Morris, Liang, Bernstein (2023). "Generative Agents: Interactive Simulacra of Human Behavior." UIST '23. arXiv:2304.03442.
Memory stream = append-only timestamped log; retrieval = recency (exp decay) + importance (LLM-scored) + relevance (cosine). Periodic reflections synthesized and written back. Canonical template.

## 3. HippoRAG — hippocampal-index retrieval
Gutiérrez, Shu, Gu, Yasunaga, Su (2024). NeurIPS 2024. arXiv:2405.14831. LLM + knowledge graph as neocortex; Personalized PageRank as hippocampal index; single-step multi-hop retrieval, ~20% multi-hop QA gains, 6-13x faster than IRCoT.
HippoRAG 2: "From RAG to Memory: Non-Parametric Continual Learning for LLMs." ICML 2025. arXiv:2502.14802.

## 4. Sleep-time compute
Lin, Snell, Wang, Packer, Wooders, Stoica, Gonzalez et al. (2025). "Sleep-time Compute: Beyond Inference Scaling at Test-time." arXiv:2504.13171. https://www.letta.com/blog/sleep-time-compute/
Background computation over context before queries arrive; transforms raw context into learned context; explicit sleep-consolidation analogy.
Lead (not fully read): "Learning to Forget: Sleep-Inspired Memory Consolidation for Resolving Proactive Interference in LLMs" arXiv:2603.14517 (2026).

## 5. MemoryBank — Ebbinghaus forgetting
Zhong, Guo, Gao, Ye, Wang, Wang (2024). AAAI 2024. arXiv:2305.10250.
External memory bank with Ebbinghaus-forgetting-curve update: strength decays with time, reinforced on recall, weighted by significance. SiliconFriend companion bot. Most-cited biologically-motivated *forgetting* in agent memory.

## 6. Titans and successors — architectural neural LTM
Behrouz, Zhong, Mirrokni (2025). "Titans: Learning to Memorize at Test Time." NeurIPS 2025. arXiv:2501.00663.
Deep neural memory module updated at test time; gradient "surprise" signal with momentum; adaptive weight-decay forgetting; >2M token contexts. https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/
- ATLAS: arXiv:2505.23735 (window-optimal memorization).
- MIRAS: theoretical framing (same blog).
- Nested Learning / HOPE: arXiv:2512.24695, NeurIPS 2025 — continuum memory system, modules updating at different frequencies; targets catastrophic forgetting.

## 7. Production memory layers
- Mem0: Chhikara, Khant, Aryan, Singh, Yadav (2025). arXiv:2504.19413. Extract/consolidate/retrieve salient facts; graph variant Mem0g; ~26% over OpenAI memory on LOCOMO.
- Zep/Graphiti: Rasmussen, Paliychuk, Beauvais, Ryan, Chalef (2025). arXiv:2501.13956. Bi-temporal knowledge-graph edges (event time + ingestion time); hybrid semantic+BM25+graph retrieval; 94.8% DMR.
- LangMem SDK (LangChain, early 2025, not peer-reviewed): episodic/semantic/procedural primitives, background extraction. https://langchain-ai.github.io/langmem/

## 8. Dreaming / replay / catastrophic forgetting
- McCloskey & Cohen (1989). Catastrophic interference. Psychology of Learning and Motivation.
- Kirkpatrick et al. (2017). EWC. PNAS 114(13):3521-3526.
- Shin, Lee, Kim, Kim (2017). "Continual Learning with Deep Generative Replay." NeurIPS 2017. arXiv:1705.08690 — generative model synthesizes pseudo-samples of past tasks ("dreamed" data); direct ML analogue of dreaming.
- van de Ven, Siegelmann, Tolias (2020). "Brain-inspired replay..." Nature Communications 11:4069 — internal generative replay from hidden representations, modeled on hippocampal replay.
- As of mid-2026: no canonical "dreaming for LLM agents" paper. Leads: MemGen arXiv:2509.24704, Memory-R1 arXiv:2508.19828, survey arXiv:2506.13045 (not vouched).

## 9. Ambient / proactive agents
LangChain "ambient agents" (Chase, early 2025): act on event streams rather than prompts; always-on, multi-threaded, proactive. Industry framing, not peer-reviewed. ProactiveEval arXiv:2508.20973 (evaluation of proactive dialogue agents; lead).

## 10. Cognitive-science anchors
- CLS: McClelland, McNaughton, O'Reilly (1995). Psychological Review 102(3):419-457. Fast hippocampus (episodic, pattern-separated) + slow neocortex (generalized), interleaved replay avoids interference. Updated: Kumaran, Hassabis, McClelland (2016). TiCS 20(7):512-534.
- Hippocampal replay: Wilson & McNaughton (1994). Science 265:676-679. Causal: Girardeau et al. (2009). Nat Neurosci 12:1222-1223. Review: Joo & Frank (2018). Nat Rev Neurosci 19:744-757.
- Synaptic homeostasis (SHY): Tononi & Cirelli (2014). Neuron 81(1):12-34. Wake potentiates synapses; sleep down-scales toward baseline — sleep as renormalization/forgetting, "the price of plasticity."

## Framing note
Three recurring threads: (a) dual-store fast/slow memory (CLS, HippoRAG, MemGPT, Titans); (b) offline consolidation (sleep-time compute, reflection, generative replay, SWR transfer); (c) principled forgetting/renormalization (MemoryBank/Ebbinghaus, Titans adaptive forgetting, SHY down-scaling). The under-explored gap as of mid-2026: an explicit always-on dreaming loop unifying (b) and (c) for LLM agents — Anima's target territory.
