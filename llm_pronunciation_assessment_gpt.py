import json
import os

import pandas as pd
import torch
import torchaudio
from Charsiu import charsiu_predictive_aligner
from openai import OpenAI
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

client = OpenAI(api_key="API-Key")

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")

charsiu = charsiu_predictive_aligner(aligner="charsiu/en_w2v2_fc_10ms")

prompt = """
You are an expert evaluator of English pronunciation. Assess the accuracy and fluency of the given text input on a scale of 1 to 5, with higher scores indicating better performance. A score of 5 represents native-speaker-level proficiency.

Input format: 
{
  "Transcript": "<Recognized ASR sentence>",
  "Phonemes_CMU": "<Recognized CMU pronouncing phoneme sequence, with (time.s pause) indicating pauses in speech.>",
  "Phonemes_IPA": "<Recognized IPA pronouncing phoneme sequence.>",
  }


Task: Return a dictionary with the following format:
{
  "Accuracy": <the assessment accuracy score>, 
  "Fluency": <the assessment fluency score>,
  "Reasoning": <detailed reasoning for the assigned score>
}

Note: Do not include any other text other than the json object. 

Input: 
"""


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


def llm_assessment_gpt(content):
    completion = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": content}]
    )
    result = completion.choices[0].message.content
    return result


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
df = pd.read_csv("/path/to/whisper/transcription.csv")
outputdir = "/path/to/output/dir"
create_dir(outputdir)

for index, row in df.iterrows():

    wavname = row["wavname"]
    transcript = row["transcript"]
    wavpath = os.path.join(wav_dir, wavname)
    outputpath = os.path.join(outputdir, wavname.replace(".wav", ".json"))
    if os.path.exists(outputpath):
        continue

    alignment = charsiu.align(audio=wavpath)
    alignment = [(round(end - start, 5), word) for start, end, word in alignment]
    alignment_cmu = process_alignment(alignment)

    alignment_ipa = phone_recognition(wavpath)
    input_data = {}
    input_data["Transcript"] = transcript
    input_data["Phonemes_CMU"] = alignment_cmu
    input_data["Phonemes_IPA"] = alignment_ipa

    input_content = prompt + str(input_data)
    result = llm_assessment_gpt(input_content)
    result = result.replace("json", "").replace("`", "")

    try:
        json_data = json.loads(result)
        json_data["alignment"] = alignment
        json_data["wavname"] = wavname
        json_data["input_content"] = input_content
        with open(outputpath, "w") as file:
            json.dump(json_data, file, indent=4)
    except Exception as e:
        print(e, wavname)
