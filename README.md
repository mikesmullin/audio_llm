# Audio LLM

Generate SFX, VOX, and Music locally w/ your GPU.

## Installation (on Windows 11)

- install latest Python (w/ `py` installer)
- install [CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local) Toolkit 13.0 (backward-compatible)
- install PyTorch 2.8.0 for Windows pip Python w/ CUDA 12.9
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
python gen_sfx.py
```

## Related Work

- [github:facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)
- [huggingface:cvssp/audioldm2](https://huggingface.co/cvssp/audioldm2)