import streamlit as st
import numpy as np
import torch
from voxws import model, util
import sounddevice as sd

# Initialize the PLiX model
@st.cache_resource
def load_model():
    return model.load(encoder_name="small", language="en", device="cpu")

fws_model = load_model()

# Set up support examples
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

# Streamlit app
st.title("Audio Keyword Detection")

# Audio recording function
def record_audio(duration, samplerate):
    recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    return recording.flatten()

# Record audio button
if st.button("Record Audio (5 second)"):
    st.write("Recording...")
    audio = record_audio(duration=5, samplerate=16000)
    st.write("Recording complete!")
    
    # Process audio
    audio_tensor = torch.tensor(audio[np.newaxis, np.newaxis, :], dtype=torch.float32)
    query = {"audio": audio_tensor}
    query = util.batch_device(query, device="cpu")
    
    # Run model prediction
    with torch.no_grad():
        predictions = fws_model(support, query)
    
    # Display result
    st.write(f"Detected keyword: {classes[predictions.item()]}")

st.write("Note: Make sure you have a microphone connected and you've granted microphone permissions to your browser.")