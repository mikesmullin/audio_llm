import numpy as np
import torch
from diffusers import AudioLDM2Pipeline
from scipy.io import wavfile

repo_id = "cvssp/audioldm2"
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# define the prompts
prompt = "The sound of a hammer hitting a wooden surface."
negative_prompt = "Low quality."

# set the seed for generator
generator = torch.Generator("cuda").manual_seed(0)

# run the generation
audio = pipe(
    prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=200,
    audio_length_in_s=10.0,
    num_waveforms_per_prompt=3,
    generator=generator,
).audios

# Convert to numpy and proper format for saving
if isinstance(audio[0], torch.Tensor):
    audio_data = audio[0].squeeze().cpu().numpy()
else:
    audio_data = audio[0].squeeze()

audio_data = np.clip(audio_data, -1.0, 1.0)
audio_data = (audio_data * 32767).astype(np.int16)

# save the best audio sample (index 0) as a .wav file
wavfile.write("out/hammer.wav", rate=16000, data=audio_data)
