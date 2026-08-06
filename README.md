# Qwen3-VL Quantization on Jetson Orin Nano

A reproducible comparison of five Qwen3-VL-4B GGUF
quantizations for 19-class HaGRID hand-gesture
classification on an 8 GB NVIDIA Jetson Orin Nano Super.

| Quantization | Exact accuracy | Family accuracy | Wall latency | Peak RAM | Energy/image |
| ------------ | -------------: | --------------: | -----------: | -------: | -----------: |
| Q8_0         |         65.00% |          77.63% |      1.583 s | 7,244 MB |      16.73 J |
| Q6_K         |         64.47% |          77.37% |      1.873 s | 6,427 MB |      19.02 J |
| Q5_K_M       |         65.00% |          77.11% |      1.767 s | 6,006 MB |      17.21 J |
| Q4_K_M       |         61.32% |          73.42% |      1.685 s | 5,654 MB |      16.10 J |
| Q3_K_M       |         62.89% |          74.21% |      1.860 s | 5,290 MB |      17.22 J |


Maximum performance: Q8
Best practical tradeoff: Q5
Lowest energy: Q4
Lowest memory: Q3
Q6 is dominated by Q5