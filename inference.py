import numpy as np
import pyaudio
import torch
from voxws import model, util

sample_rate = 16000
frames_per_buffer = 512
support_examples = ["./test_clips/aandachtig.wav", "./test_clips/stroom.wav",
    "./test_clips/persbericht.wav", "./test_clips/klinkers.wav",
    "./test_clips/zinsbouw.wav"]
classes = ["aandachtig", "stroom", "persbericht", "klinkers", "zinsbouw"]
int_indices = [0,1,2,3,4]

support = {
    "paths": support_examples,
    "classes": classes,
    "labels": torch.tensor(int_indices)
}
support["audio"] = torch.stack([util.load_clip(path) for path in support["paths"]])
support = util.batch_device(support, device="cpu")

fws_model = model.load(encoder_name="small", language="nl", device="cpu")

p = pyaudio.PyAudio()
stream = p.open(format = pyaudio.paInt16, channels=1, 
    rate=sample_rate, input=True, frames_per_buffer=frames_per_buffer)

frames = []
while True:  
    data = stream.read(frames_per_buffer)
    buffer = np.frombuffer(data, dtype=np.int16)
    frames.append(buffer)
    if len(frames) * frames_per_buffer / sample_rate >= 1:
        audio = np.concatenate(frames)
        audio = audio.astype(float) / np.iinfo(np.int16).max 
        query = {"audio":torch.tensor(audio[np.newaxis, np.newaxis,:], dtype=torch.float32)}
        query = util.batch_device(query, device="cpu")
        with torch.no_grad():
            predictions = fws_model(support, query)
            print(classes[predictions.item()])
        frames = []
