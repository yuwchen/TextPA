import base64
import json
import os

from openai import OpenAI

client = OpenAI(api_key="API-KEY")

prompt = """You are an expert evaluator of English pronunciation. Assess the accuracy and fluency of the given input on a scale of 1 to 5, with higher scores indicating better performance. A score of 5 represents native-speaker-level proficiency.

Task: Return a dictionary with the following format:
{
  "Accuracy": <the assessment accuracy score>, 
  "Fluency": <the assessment fluency score>,
  "Reasoning": <detailed reasoning for the assigned score>
}

Note: Do not include any other text other than the json object. 

Input: 

"""


def create_dir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")


def get_file_list(rootdir, format=".wav"):
    file_list = []
    for subdir, _, files in os.walk(rootdir):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(format):
                file_list.append(filepath)
    return file_list


def call_openai(filepath):

    with open(filepath, "rb") as audio_file:
        encoded_string = base64.b64encode(audio_file.read()).decode("utf-8")

    completion = client.chat.completions.create(
        model="gpt-4o-mini-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {"data": encoded_string, "format": "wav"},
                    },
                ],
            },
        ],
    )
    return completion


file_list = get_file_list("/path/to/input/wav", format=".wav")

outputdir = "/path/to/output/file"
create_dir(outputdir)

for filepath in file_list:
    wavname = os.path.basename(filepath)
    outputpath = os.path.join(outputdir, wavname.replace(".wav", ".json"))
    if os.path.exists(outputpath):
        continue

    result = call_openai(filepath)
    result = result.choices[0].message.audio.transcript

    result = (
        result.replace("json", "")
        .replace("`", "")
        .replace("Accuracy:", "Accuracy-")
        .replace("Fluency:", "Fluency-")
        .replace('\\"', "**")
    )

    try:
        json_data = json.loads(result)
        with open(outputpath, "w") as file:
            json.dump(json_data, file, indent=4)
    except Exception as e:
        print(e, filepath)
        print(result)
