import json
import os
import string

import matplotlib.pyplot as plt
import num2words
import numpy as np
import pandas as pd
import scipy
import textdistance
from nltk.corpus import cmudict
from phonemizer.backend import EspeakBackend
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# nltk.download("cmudict")
# Load the CMU dictionary
cmu_dict = cmudict.dict()

separator = Separator(phone=" ", word=None)
backend = EspeakBackend("en-us")


def get_filepaths(directory, format=".json"):
    file_paths = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(format):
                file_paths.append(filename)
    return file_paths


def read_json(path):
    with open(path, "r") as file:
        data = json.load(file)
    return data


def print_result(pred, gt, score_name):
    mse = mean_squared_error(pred, gt)
    corr, _ = scipy.stats.pearsonr(pred, gt)
    spearman, _ = scipy.stats.spearmanr(pred, gt)
    # print('mse:', mse)
    # print('corr:', round(corr,4))
    # print('srcc:', round(spearman,4))
    print(score_name, round(corr, 4), len(pred))


def remove_pun_except_apostrophe(input_string):
    """
    remove punctuations (except for ' ) of the inupt string.
    """
    translator = str.maketrans("", "", string.punctuation.replace("'", ""))
    output_string = input_string.translate(translator).replace("  ", " ")
    return output_string


def convert_num_to_word(sen):
    """
    convert digit in a sentence to word. e.g. "7112" to "seven one one two".
    """
    try:  # for 4 digit samples of speechocean data.
        int(sen.replace(" ", ""))
        sen = " ".join([char for char in sen])
        sen = " ".join(
            [num2words.num2words(i) if i.isdigit() else i for i in sen.split()]
        )
        sen = sen.replace("  ", " ")
    except:
        sen = " ".join(
            [num2words.num2words(i) if i.isdigit() else i for i in sen.split()]
        )
    return sen


def get_cmu_label(sentence):

    cmu_list = []
    sentence = sentence.lower()
    for word in sentence.split(" "):
        if word in cmu_dict:
            # There may be multiple pronunciations
            pronunciations = cmu_dict[word]
            for i, pron in enumerate(pronunciations):
                cmu_list.extend(pron)

    cmu_list = [
        phoneme.replace("0", "").replace("1", "").replace("2", "")
        for phoneme in cmu_list
    ]
    return " ".join(cmu_list)


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def process_cmu_alignment(alignment):
    cmu_sequence = ""
    for parts in alignment:
        phoneme = parts[-1]
        if phoneme == "[SIL]":
            pass
        else:
            cmu_sequence += phoneme + " "
    return cmu_sequence


def get_ipa_label(sentence):

    ipa_list = []
    sentence = sentence.lower()
    sentence = Punctuation(';:,.!"?()-').remove(sentence)

    for word in sentence.split(" "):
        try:
            phonemes = backend.phonemize([word], separator=separator, strip=True)[0]
            ipa_list.append(phonemes)
        except:
            continue
    return " ".join(ipa_list)


def normalized_result(input_pred):
    min_value = np.min(input_pred)
    max_value = np.max(input_pred)
    normalized_pred = ((input_pred) - min_value) / (max_value - min_value)
    return normalized_pred


def extract_number(text):
    return "".join([char for char in text if char.isdigit()])


file_list = get_filepaths("/path/to/phoneme_sequence_dir")

for filepath in file_list:
    the_alignment = load_json(filepath)
    cmu_sequence = process_cmu_alignment(the_alignment["alignment_cmu"])
    asr_cmu_sequence = get_cmu_label(the_alignment["transcript"])
    ipa_sequence = the_alignment["alignment_ipa"]
    asr_ipa_sequence = get_ipa_label(the_alignment["transcript"])

    cmu_score = textdistance.smith_waterman.normalized_similarity(
        cmu_sequence, asr_cmu_sequence
    )

    ipa_score = textdistance.smith_waterman.normalized_similarity(
        ipa_sequence, asr_ipa_sequence
    )

    print(filepath, cmu_score, ipa_score)
