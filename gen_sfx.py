import numpy as np
import torch
from diffusers import AudioLDM2Pipeline
from scipy.io import wavfile

# repo_id = "cvssp/audioldm2"  # Text-to-audio 	350M 	1.1B 	1150k
repo_id = "cvssp/audioldm2-large"  # Text-to-audio 	750M 	1.5B 	1150k
# repo_id = "cvssp/audioldm2-music"  # Text-to-music 	350M 	1.1B 	665k
# repo_id = "cvssp/audioldm2-gigaspeech"  # Text-to-speech 	350M 	1.1B 	10k
# repo_id = "cvssp/audioldm2-ljspeech"  # Text-to-speech 	350M 	1.1B 	10k

pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Descriptive prompt inputs work best:
# - use adjectives to describe the sound (e.g. “high quality” or “clear”) and
# - make the prompt context specific (e.g. “water stream in a forest” instead of “stream”).
prompt = "player picks up an item, adding it to their inventory"
filename = "pickup.wav"
# Using a negative prompt can significantly improve the quality of the generated waveform,
# by guiding the generation away from terms that correspond to poor quality audio.
negative_prompt = "Low quality."
# rate = 44100
rate = 16000  # default for AudioLDM2
audio_length_seconds = 10.0  # Length of generated audio in seconds

# set the seed for generator
generator = torch.Generator("cuda").manual_seed(0)

# run the generation
audio = pipe(
    prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=200,
    audio_length_in_s=audio_length_seconds,
    num_waveforms_per_prompt=3,
    generator=generator,
).audios

# Convert to numpy and proper format for saving
# NOTE: choosing audio[0] is important because the first is ranked as the best of the 3
if isinstance(audio[0], torch.Tensor):
    audio_data = audio[0].squeeze().cpu().numpy()
else:
    audio_data = audio[0].squeeze()

# Debug: Print actual audio information
print(f"Generated audio shape: {audio_data.shape}")
print(f"Sample rate: {rate}")
print(f"Actual duration: {len(audio_data) / rate:.2f} seconds")
print(f"Expected duration: {audio_length_seconds} seconds")

audio_data = np.clip(audio_data, -1.0, 1.0)
audio_data = (audio_data * 32767).astype(np.int16)

# save the best audio sample (index 0) as a .wav file
wavfile.write("out/sfx/" + filename, rate=rate, data=audio_data)
