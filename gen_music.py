from diffusers import AudioLDM2Pipeline
import torch

# repo_id = "cvssp/audioldm2"  # Text-to-audio 	350M 	1.1B 	1150k
# repo_id = "cvssp/audioldm2-large"  # Text-to-audio 	750M 	1.5B 	1150k
repo_id = "cvssp/audioldm2-music"  # Text-to-music 	350M 	1.1B 	665k
# repo_id = "cvssp/audioldm2-gigaspeech"  # Text-to-speech 	350M 	1.1B 	10k
# repo_id = "cvssp/audioldm2-ljspeech"  # Text-to-speech 	350M 	1.1B 	10k

pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Descriptive prompt inputs work best:
# - use adjectives to describe the sound (e.g. “high quality” or “clear”) and
# - make the prompt context specific (e.g. “water stream in a forest” instead of “stream”).
prompt = """
saxophone playing a smooth jazz melody
"""
filename = "sax_jazz"
# Using a negative prompt can significantly improve the quality of the generated waveform,
# by guiding the generation away from terms that correspond to poor quality audio.
# negative_prompt = "Low quality."
# rate = 44100
rate = 16000  # default for AudioLDM2
audio_length_seconds = 60.0  # Length of generated audio in seconds
# steps = 200
steps = 200
audio = pipe(
    prompt, num_inference_steps=steps, audio_length_in_s=audio_length_seconds
).audios[0]

import scipy
from datetime import datetime

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
scipy.io.wavfile.write(f"out/music/{filename}_{ts}.wav", rate=rate, data=audio)
