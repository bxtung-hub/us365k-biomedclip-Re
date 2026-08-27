# Checkpoint provenance

| Variant | Preserved checkpoint epoch | Checkpoint SHA256 |
|---|---:|---|
| FT1k | 1 | `05871a0f447e1cc535761ee1dd36b2f1c7aa9fd82e2bb32cb3e16655740f3a17` |
| FT5k | 3 | `6adc47b0e60c64b8a177eba95f69b7b02935a96ae5d607b0651c1eb6ac809425` |
| FT10k | 3 | `1cbbb24b812022f1513870b71d91ce12fdfdd227a39830fdf2ffa7c553fabdb0` |
| FT50k best | 2 | `3b7cb059523804c75cbbfc5a3ca16ad43c7910ea4f66c348f7968a64c6e1799f` |

The historical controlled 5k FT50k row was produced from the final epoch-3 in-memory model state. The separate 71,918-pair full-gallery evaluation used the validation-selected saved FT50k epoch-2 checkpoint. This distinction is preserved rather than retrospectively rewritten.

`LARGE_ARTIFACTS.csv` records hashes of the multi-volume checkpoint archives kept outside the ordinary Git repository because each volume is hundreds of megabytes.
