# Disclaimer

**Last updated:** 2026-09-12 — Project version 2026.9.0

Please read this disclaimer carefully before using, copying, modifying, or
distributing HummingFlight or any part of it.

## Not affiliated or endorsed

HummingFlight is an independent, from-scratch software project. It is not
affiliated with, endorsed by, sponsored by, or connected to Z.ai (Zhipu AI),
the creators or maintainers of GLM-5.2, or any of their subsidiaries or
partners, unless explicitly stated otherwise in writing.

## Model weights

- The **GLM-5.2 model weights are the property of their respective owners**
  and are **not** included in this repository.
- You are responsible for obtaining the weights independently and for
  complying with the license and terms of their owner.
- HummingFlight does not redistribute, host, or bundle model weights. Any use
  of the weights with this engine is at your own discretion and risk.

## No warranty

THIS PROJECT IS PROVIDED **"AS IS"**, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM,
OUT OF, OR IN CONNECTION WITH THE PROJECT OR THE USE OR OTHER DEALINGS IN THE
PROJECT. See the [GNU AGPL-3.0](LICENSE) for the full legal text.

## Experimental nature

- HummingFlight is an experimental research-oriented engine designed for
  fidelity, not throughput. Generation is intentionally very slow on
  consumer-grade hardware.
- Outputs are **not** guaranteed to be accurate, complete, safe, or suitable
  for any purpose. Model output can be incorrect, biased, or harmful.
  Always review AI-generated content before acting on it.
- Do not use this project, or the models it runs, for critical decisions
  (medical, legal, financial, safety, life-support, autonomous control, or any
  decision where error could cause harm) without independent validation.

## Performance claims

Benchmark figures quoted in this repository (per-token times, throughput,
cache hit rates) reflect specific hardware, storage, and configuration on
which they were recorded. Your results will differ. Treat all numbers as
reference measurements only.

## Security

This project processes local model data and spawns a Python subprocess for
tokenization. Keep inputs trustworthy; do not point it at untrusted data while
granting elevated privileges. You are responsible for the security of your own
environment. The project is provided for research purposes and does not
provide a security guarantee of any kind.

## Trademarks

All trademarks, service marks, and product names referenced in this repository
belong to their respective owners. Their use here does not imply any
affiliation with or endorsement by the owners.

## Jurisdiction

You are solely responsible for compliance with the laws of your jurisdiction
when obtaining weights, running the engine, or distributing any outputs or
modifications.

---

*By using HummingFlight you acknowledge that you have read this disclaimer and
accept it in full. If you do not agree, do not use the project.*