# GPU Benchmark Suite 🏋️

Comprehensive GPU benchmarking for ML workloads across AMD (ROCm) and NVIDIA (CUDA).

## Tests

- **Compute**: FP16/FP32/FP64 TFLOPS
- **Memory**: VRAM bandwidth, latency
- **ML Workloads**: GEMM, convolution, attention
- **End-to-end**: Training throughput, inference latency

## Results

| GPU | FP16 TFLOPS | Memory BW | GEMM (ms) | Training img/s |
|-----|-------------|-----------|-----------|---------------|
| MI300X | 1,307 | 5.3 TB/s | 0.8 | 1,840 |
| A100 | 312 | 2.0 TB/s | 2.1 | 780 |
| H100 | 989 | 3.35 TB/s | 1.2 | 1,520 |
| RTX 4090 | 826 | 1.0 TB/s | 1.8 | 1,200 |

## Quick Start

```bash
pip install gpu-benchmark-suite
gpu-bench --device cuda:0 --tests compute,memory,ml
```

## License

MIT