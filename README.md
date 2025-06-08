## Project Motivation

This project was developed to help my friend improve his Arabic recitation. The project tries to solve the challenge of accurately transcribing Quranic Arabic audio, providing automated segmentation, transcription, and feedback tools tailored to users needs.

## Quranic Arabic ASR Pipeline

This project implements an Automatic Speech Recognition (ASR) pipeline for Quranic Arabic audio using the Whisper model and Hugging Face Transformers. It features silence-based audio segmentation, batch transcription, objective accuracy metrics (WER, CER), and user-friendly color-coded feedback.

## Features

- **Automatic Speech Recognition:** Transcribes Quranic Arabic audio using the Whisper deep learning model.
- **Audio Segmentation:** Splits long audio files into manageable segments based on silence detection for improved transcription accuracy.
- **Batch Transcription:** Processes and transcribes each segment, then combines results into a complete transcript.
- **Accuracy Metrics:** Calculates Word Error Rate (WER) and Character Error Rate (CER) using `jiwer` for objective evaluation.
- **Text Similarity & Normalization:** Utilizes `pyarabic`, `rapidfuzz`, and `Levenshtein` for advanced text comparison and Arabic script normalization.
- **Color-Coded Feedback:** Highlights transcription accuracy to help users quickly identify errors and areas for improvement.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com//.git
    cd 
    ```
2. **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3. **Install ffmpeg:**  
   `pydub` requires ffmpeg for audio processing.  
   - **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html), add to PATH.
   - **macOS:** `brew install ffmpeg`
   - **Linux:** `sudo apt-get install ffmpeg`

## Usage

1. Place your audio file (e.g., `sample_audio.ogg`) in the project directory.
2. Run the script or notebook:
    ```bash
    python asr_pipeline.py --audio sample_audio.ogg
    ```
    *Or open the notebook in Colab/Jupyter and run all cells.*
3. View the generated transcript and feedback.

## Requirements

All required Python packages are listed in `requirements.txt`:
```
transformers
pydub
torch
rapidfuzz
pyarabic
python-Levenshtein
jiwer
```
Install them with:
```bash
pip install -r requirements.txt
```

## Model

Uses the [tarteel-ai/whisper-base-ar-quran](https://huggingface.co/tarteel-ai/whisper-base-ar-quran) model from Hugging Face.

## Results

- **Random Forest**: WER: XX%, CER: XX% (replace with your actual results)
- **RNN**: WER: XX%, CER: XX% (replace with your actual results)

## Example

```python
from transformers import pipeline
from pydub import AudioSegment, silence

# Load the model
pipe = pipeline("automatic-speech-recognition", model="tarteel-ai/whisper-base-ar-quran")

# Transcribe audio file
transcription = transcribe_audio("sample_audio.ogg")
print("Transcription:", transcription)
```

## License

MIT License

## Acknowledgements

- [Hugging Face](https://huggingface.co/)
- [Tarteel AI](https://tarteel.ai/)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [jiwer](https://github.com/jitsi/jiwer)
- [pyarabic](https://github.com/arabic-tools/pyarabic)
- [rapidfuzz](https://github.com/maxbachmann/RapidFuzz)

---

*Feel free to contribute or raise issues!*

---

**Tip:**  
If you have a demo audio file or screenshots, add them to your repo and reference them in the README for better presentation.

Let me know if you want further customization or a section for contributors!
