---
title: Idempotent Systems Frontier AI Showcase
emoji: ⚡
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Zero-Copy KV Compaction, 0-FLOPs Tropical Attention & MCTS Pruning
---

# Idempotent Systems: Frontier AI Acceleration Showcase

> **Inventor & Author:** Dr. A. Emre ÇETİN (`aemre.cetin@gmail.com`)  
> **Patents:** U.S. Patent Applications `64/148,668`, `64/148,679`, `64/149,516`, `64/149,518`, `64/149,520` (USPTO Patent Pending)  
> **PyPI Ecosystem:** `pip install idempotent-core idempotent-kv idempotent-attention`

This interactive showcase demonstrates:
1. **Zero-Copy In-Place LLM KV-Cache Compaction:** Eliminates 100% of auxiliary VRAM overhead during context eviction in vLLM and Hugging Face inference.
2. **Zero-Multiplication Tropical Attention (0 FLOPs):** Converts floating-point multiplications into idempotent addition and maximum operations over the max-plus semiring.
3. **Tarski Invariant Reasoning Verifier:** In-situ 75% thought pruning for test-time compute scaling in LLM reasoning engines (o1/R1).
