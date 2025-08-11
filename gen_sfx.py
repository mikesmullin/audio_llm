from diffusers import AudioLDM2Pipeline
import torch
import numpy as np

repo_id = "cvssp/audioldm2"  # Text-to-audio 	350M 	1.1B 	1150k
# repo_id = "cvssp/audioldm2-large"  # Text-to-audio 	750M 	1.5B 	1150k
# repo_id = "cvssp/audioldm2-music"  # Text-to-music 	350M 	1.1B 	665k
# repo_id = "cvssp/audioldm2-gigaspeech"  # Text-to-speech 	350M 	1.1B 	10k
# repo_id = "cvssp/audioldm2-ljspeech"  # Text-to-speech 	350M 	1.1B 	10k

pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Descriptive prompt inputs work best:
# - use adjectives to describe the sound (e.g. “high quality” or “clear”) and
# - make the prompt context specific (e.g. “water stream in a forest” instead of “stream”).
prompt = """
short retro-inspired sound effect for a player health pickup or powerup in a video game.
Style: Bright, cheerful, and satisfying, with a clear upward pitch progression to indicate positivity.
Tone & Timbre: Use chiptune-style synths or clean sine/square waves, with light harmonic overtones. Avoid harsh distortion.
Intended Use: In-game pickup sound triggered when the player collects a health item or powerup.
"""
filename = "health"
# Using a negative prompt can significantly improve the quality of the generated waveform,
# by guiding the generation away from terms that correspond to poor quality audio.
negative_prompt = """
Constraints: No background noise, no voice samples, no percussion that could be mistaken for damage or error.
"""
# rate = 44100
rate = 16000  # default for AudioLDM2

from datetime import datetime
import scipy

for i in range(30):
    audio_length_seconds = 5.0  # np.random.uniform(1.0, 2.0)

    # run the generation
    audio = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=200,
        audio_length_in_s=audio_length_seconds,
        num_waveforms_per_prompt=1,
    ).audios[0]

    audio = audio.squeeze()
    audio = np.clip(audio, -1.0, 1.0)
    audio = (audio * 32767).astype(np.int16)

    # Trim trailing silence from the end of the audio
    def trim_trailing_silence(audio, threshold=500, min_silence_len=160):
        # threshold: amplitude below which is considered silence (int16)
        # min_silence_len: minimum number of samples to consider as silence
        abs_audio = np.abs(audio)
        # Find last index where audio is above threshold
        non_silence = np.where(abs_audio > threshold)[0]
        if len(non_silence) == 0:
            return audio[:1]  # all silence, keep at least 1 sample
        last_idx = non_silence[-1]
        # Optionally, pad a little after the last sound
        return audio[: last_idx + 1 + min_silence_len]

    audio = trim_trailing_silence(audio)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"out/sfx/{filename}_{ts}_{i+1}.wav"

    scipy.io.wavfile.write(out_path, rate=rate, data=audio)
