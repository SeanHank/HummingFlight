# Credits

**Last updated:** 2026-09-12 — Project version 2026.9.0

HummingFlight is an independent, from-scratch C++ inference engine. The credit
below recognizes the people and projects that made it possible.

## Author and maintainer

- **Sean Hank** — design, implementation, and maintenance of the HummingFlight
  engine. Copyright © 2026 Sean Hank. Licensed under
  [GNU AGPL-3.0](LICENSE).

## Model

- **GLM-5.2** — the reference large language model this engine serves, created
  by **Z.ai (Zhipu AI)** and its research teams. The model architecture, config
  semantics, and tokenizer behavior implemented here follow the published
  references. The weights remain the property of their respective owners and
  are obtained independently; see [DISCLAIMER.md](DISCLAIMER.md).

## Reference implementations and libraries

- **Hugging Face `transformers`** — the official tokenizer and chat-template
  behavior that the Python subprocess bridge delegates to, guaranteeing
  byte-identical tokenization.
- **The DeepSeek-V3 technical report line** — the compressed Multi-head Latent
  Attention (MLA) design that GLM-5.2-style attention builds on; this project
  independently re-implements it from the model config.
- **colibri** — inspiration for the "run very large models on small hardware
  by streaming weights from disk" approach.
- **CMake / Visual Studio / GCC / Clang toolchains** — build system and
  compilers used to produce the engine.
- **OpenMP** — parallelization of the BF16 GEMV kernels on CPU.
- **CUDA toolkit (optional)** — the GPU kernels (BF16 matvec, softmax,
  argmax) verified on an NVIDIA RTX 3060 Laptop GPU.
- **Python** and the **standard library** — tokenizer bridge, gate scripts,
  and validation tooling.

## Icons and media

No bundled icons, fonts, or media are distributed with this project. The
glow-in-the-dark lab-night aesthetic is plain text markup.

## Recognition policy

If you contributed code, tests, documentation, or infrastructure and you
believe you belong here, raise an issue or send a pull request that updates
this file.

## License notice

This file is part of HummingFlight and, unless otherwise noted, all
project files are licensed under the **GNU Affero General Public License
v3.0**. Third-party components retain their own licenses; see the project
copyright lines and dependency metadata where applicable.