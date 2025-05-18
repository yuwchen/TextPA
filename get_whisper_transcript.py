import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def get_filepaths(directory, format=".wav"):
    file_paths = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(format):
                file_paths.append(filename)
    return file_paths


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir",
        default="/path/to/input/wav",
        type=str,
        help="Path of your DATA/ directory",
    )
    args = parser.parse_args()
    input_dir = args.datadir

    file_list = get_filepaths(input_dir, format=".wav")  # loop all the .wav file in dir
    file_list = set(file_list)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3"
    model_name = os.path.basename(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, use_safetensors=True
    )

    model.to(device)

    outputname = "{}-{}.csv".format(os.path.basename(input_dir), model_name)
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs={"language": "english"},
    )

    try:
        print("Number of files:", len(file_list))
        df = pd.read_csv()
        exist_list = set(df["wavname"].to_list())
        print("Number of already processed files:", len(exist_list))
        file_list = file_list - exist_list
        print("Number of unprocessed files:", len(file_list))
        file_list = list(file_list)
    except Exception as e:
        print(e)
        print("Create new file")
        df = pd.DataFrame(columns=["wavname", "transcript"])
        df.to_csv(
            outputname,
            sep=",",
            index=False,
            header=True,
        )

    for filename in tqdm(file_list):
        try:
            path = os.path.join(input_dir, filename)
            result = pipe(path)
            transcript = result["text"]
            results = pd.DataFrame([{"wavname": filename, "transcript": transcript}])
            results.to_csv(
                outputname,
                mode="a",
                sep=",",
                index=False,
                header=False,
            )
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()
