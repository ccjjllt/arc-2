# ARC3 vLLM H100 Wheelhouse V3

Offline Python 3.12 wheelhouse for Kaggle H100 ARC-AGI-3 notebooks.

Core runtime pins:

- vllm==0.19.0
- torch==2.10.0
- torchaudio==2.10.0
- torchvision==0.25.0
- transformers==4.57.6
- flashinfer-python==0.6.6
- flashinfer-cubin==0.6.6
- CUDA 12.8 PyTorch/NVIDIA wheels

Install offline with:

```bash
python -m pip install --no-index --find-links /kaggle/input/arc3-vllm-h100-wheelhouse-v3   --requirement /kaggle/input/arc3-vllm-h100-wheelhouse-v3/requirements.lock
```

This dataset intentionally contains dependency artifacts only: wheels, resolver/lock files, checksums, and this README. Notebook smoke-test/helper code is not included here.
