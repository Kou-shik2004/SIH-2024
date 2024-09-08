# VoxVersa: Few Shot Language Agnostic Keyword Spotting (FSLAKWS) System

## Overview

**VoxVersa** is an advanced system designed to efficiently detect and classify keywords across multiple languages using few training samples per keyword. The system leverages cutting-edge meta-learning techniques and audio signal processing to create a flexible, scalable, and adaptable keyword spotting model that works across diverse linguistic environments. 

The system processes audio at various sample rates (8k-48k) and is capable of quickly learning new keywords and adapting to different audio conditions, making it highly effective for applications in voice-controlled technologies, multilingual customer service, and more.

## Features

- **Few-Shot Learning**: Efficient detection and classification of keywords using very few training samples.
- **Language Agnostic**: Capable of handling keywords in multiple languages without requiring extensive language-specific training data.
- **Audio Flexibility**: Processes audio at multiple sample rates (8kHz to 48kHz).
- **Meta-Learning**: Uses model-agnostic meta-learning techniques for rapid adaptation to new keywords and environments.
- **On-Device Processing**: Enhances user privacy and security by enabling on-device processing.

## Technologies Used

- **Programming Language**: Python
- **Framework**: PyTorch

## Installation

To set up the environment for **VoxVersa**, follow the steps below:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Kou-shik2004/SIH-2024.git
   cd SIH-2024
   ```

2. **Install dependencies**:
   Create a virtual environment and install the required Python packages:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # For Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Start with the project**:
    ```bash
    python setup.py install
    ```

## Usage

Once the environment is set up, you can start training the model on your dataset or testing it on new audio samples.

### 1. Training the Model
To train the model using a custom dataset, use the following command:
```bash
python test_model.py
```

### 2. Inference
To get inference from the model:
```bash
python inference.py
```

## Customizing for Your Own Few-Shot Data

To train the model on your own few-shot data and use it for inference, you'll need to make changes to the `test_model.py` and `inference.py` files. Here are specific instructions based on the current implementation:

### Modifying `test_model.py`:

1. Update the support set:
   - Replace the file paths in `support_examples` with your own audio files.
   - Update the `classes` list with your own keyword classes.
   - Adjust the `int_indices` if necessary.

```python
support_examples = ["./your_clips/keyword1.wav", "./your_clips/keyword2.wav", ...]
classes = ["keyword1", "keyword2", ...]
int_indices = [0, 1, 2, ...]
```

2. Modify the model loading if needed:
   - Change the `encoder_name` or `language` parameters to match your use case.

```python
fws_model = model.load(encoder_name="your_encoder", language="your_language", device="cpu")
```

3. Adjust audio processing parameters if necessary:
   - Modify `sample_rate` and `frames_per_buffer` to match your audio data.

### Modifying `inference.py`:

1. Update the support set:
   - Replace the file paths in `support["paths"]` with your own audio files.
   - Update the `support["classes"]` list with your own keyword classes.
   - Adjust the `support["labels"]` tensor if necessary.

```python
support = {
    "paths": ["./your_clips/keyword1.wav", "./your_clips/keyword2.wav", ...],
    "labels": torch.tensor([0, 1, 2, ...]),
    "classes": ["keyword1", "keyword2", ...],
}
```

2. Modify the model loading if needed:
   - Change the `encoder_name` or `language` parameters to match your use case.

```python
fws_model = model.load(encoder_name="your_encoder", language="your_language", device="cpu")
```

3. Adjust the query processing:
   - If you're using different test clips, update the paths in the `query` dictionary.

```python
query = {
    "paths": ["./your_test_clips/query1.wav", "./your_test_clips/query2.wav"]
}
```

4. Fine-tune the inference process:
   - You may need to adjust the audio processing parameters or prediction threshold based on your specific use case.

Remember to thoroughly test your modifications to ensure they work correctly with your specific dataset and use case. You may also need to update the `requirements.txt` file if you introduce any new dependencies.

## Running the Customized Model

After making the necessary modifications:

1. To train and test the model:
   ```bash
   python test_model.py
   ```

2. To run inference:
   ```bash
   python inference.py
   ```

Make sure you have the required audio files in the correct directories before running these scripts.
