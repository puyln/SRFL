# Implementation details

| Component | Location | Implementation |
|---|---|---|
| SAM encoder | `srfl/models/encoder.py` | ViT-B/L image encoder with frozen pretrained parameters |
| Adapters | `BottleneckAdapter`, `AdaptedOperation` | Down projection, ReLU, up projection; adaptation after attention and MLP outputs |
| Decoder | `srfl/models/decoder.py` | Image-derived attention queries and pixel upsampling; configurable feature fusion |
| USGR | `srfl/usgr.py` | L2-normalized features, normalized Gaussian reference directions, nonnegative transport coupling, quadratic marginal penalties |
| GUA | `srfl/aggregation.py` | Parameter-deviation gradients, simplex-constrained weights, global trainable-parameter update |
| Supervision | `srfl/losses.py`, `srfl/regions.py` | Marginalized class loss, ignored-pixel loss, or independent region BCE and Dice |
| Training | `srfl/training.py` | Local client updates, server aggregation, validation, and checkpoint resume |
| Evaluation | `srfl/evaluation.py`, `srfl/metrics.py` | Native-grid volume reconstruction and Dice/HD/HD95 |

The SAM configurations select fused decoder features for USGR. Feature location, reference count, solver iterations, and marginal penalties remain explicit configuration fields.

GUA offers normalized or raw directions and an optional finite-step safeguard. Its fixed-client step conditions describe a server update rather than full-training convergence.

The standard pipeline uses its configured local optimizer; `convergence.py` implements a separate SGD comparison including FedProx and SCAFFOLD. Benchmark configuration files specify 64 batches per client per round. That sampling budget differs from a full local epoch. Benchmark seeds, metric definitions, and simulated-client settings are retained in those configurations.

Adapter placement, decoder dimensions, MRI normalization, and solver defaults are implementation choices. Original experiment checkpoints and complete historical run records are not included. The supplied implementation and benchmark defaults are not independently verified reproductions of the manuscript's numerical results.
