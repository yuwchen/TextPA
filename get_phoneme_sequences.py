import json
import os

import pandas as pd
import torch
import torchaudio
from Charsiu import charsiu_predictive_aligner
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
charsiu = charsiu_predictive_aligner(aligner="charsiu/en_w2v2_fc_10ms")


def phone_recognition(wavpath):

    waveform, sample_rate = torchaudio.load(wavpath)
    waveform = waveform.squeeze(0)
    input_values = processor(
        waveform, sampling_rate=sample_rate, return_tensors="pt"
    ).input_values
    # retrieve logits
    with torch.no_grad():
        logits = model(input_values).logits

    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return " ".join(transcription)


def create_dir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")


def process_alignment(lst):
    # Remove [SIL] at the start
    while lst and lst[0][1] == "[SIL]":
        lst.pop(0)
    # Remove [SIL] at the end
    while lst and lst[-1][1] == "[SIL]":
        lst.pop()

    # Build the result string in the desired format
    result = ""
    for time, word in lst:

        if word == "[UNK]":
            result += "? "

        elif word == "[SIL]":
            result += f"({time:.2f}s pause) "
        else:
            result += f"{word} "

    return result.strip()


wav_dir = "/path/to/wav/dir"
df = pd.read_csv("/path/to/whisper_transcript.csv")
outputdir = "/path/to/output/dir"
create_dir(outputdir)

for index, row in df.iterrows():
    wavname = row["wavname"]
    transcript = row["transcript"]
    wavpath = os.path.join(wav_dir, wavname)

    alignment = charsiu.align(audio=wavpath)
    alignment_ipa = phone_recognition(wavpath)

    outputpath = os.path.join(outputdir, wavname.replace(".wav", ".json"))
    if os.path.exists(outputpath):
        continue

    json_data = {}
    try:
        json_data["alignment_cmu"] = alignment
        json_data["wavname"] = wavname
        json_data["alignment_ipa"] = alignment_ipa
        json_data["transcript"] = transcript
        with open(outputpath, "w") as file:
            json.dump(json_data, file, indent=4)
    except Exception as e:
        print(e, wavname)
