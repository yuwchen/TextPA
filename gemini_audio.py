import json
import os
import re

from google import genai
from google.genai import types

client = genai.Client(api_key="API-Key")

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


def call_gemini(filepath):

    with open(filepath, "rb") as f:
        image_bytes = f.read()

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            prompt,
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="audio/wav",
            ),
        ],
    )

    return response.text


file_list = get_file_list("/path/to/wav/dir", format=".wav")

outputdir = "/path/to/output/json"
create_dir(outputdir)

for filepath in file_list:
    wavname = os.path.basename(filepath)
    outputpath = os.path.join(outputdir, wavname.replace(".wav", ".json"))
    if os.path.exists(outputpath):
        continue

    result = call_gemini(filepath)

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
