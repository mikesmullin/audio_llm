# Music LLM

Trying to generate music .wav files locally.

## Installation

- install latest Python (w/ `py` installer)
- install [CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local) Toolkit 13.0 (backward-compatible)
- install PyTorch 2.8.0 for Windows Pip Python w/ CUDA 12.9
  ```
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
  ```
- verify it can see your GPU
  ```
  python

  import torch;
  print(torch.__version__, 'cuda_ok=', torch.cuda.is_available());

  import sys;
  if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
  ```
- install misc libs
  ```
  pip install --upgrade huggingface_hub accelerate
  pip install transformers==4.41.2 diffusers==0.29.2
  pip install scipy pydub
  pip install librosa
  ```

## Usage

*NOTE*: On first run, this will auto-download the HuggingFace models.

```
python gen.py
```