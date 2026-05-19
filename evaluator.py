import argparse, sys, os, copy, time, random, json, pickle, re, collections
from itertools import combinations
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime


def get_instruction_suffix(args):

    if args.data in ['gsm8k']:

        if args.bae :
            return ' Make sure to state your answer at the end of the response.'

        elif args.cot :
            return " Make sure to state your final answer in curly brackets at the very end of your response, just like: '{final answer: 123}'. Let's think step by step."

        else :
            return ' Make sure to state your final answer in curly brackets at the very end of your response, just like: "{final answer: 123}".'


def evaluate_arithmetics(responses, answer):
    # Returns True if corret, False if incorrect

    final_answers = []

    for _, response in responses.items():

        try:
            pred = re.findall(r"\{(.*?)\}", response)[-1]
            pred = float(pred.replace("final answer:", "").strip())

            final_answers.append(np.round(pred, 1))

        except :
            final_answers.append("")


    if len(set(final_answers)) == 1 and list(set(final_answers))[0] == "":

        final_answers = [""] * len(final_answers)
        debate_answer = ""

    else :

        counter = collections.Counter(
            [x for x in final_answers if x != ""]
        )

        max_count = max(counter.values())

        most_common = [
            key for key, value in counter.items()
            if value == max_count
        ]

        debate_answer = random.choice(most_common)


    return (
        final_answers,
        debate_answer,
        debate_answer == np.round(answer, 1)
    )


def base_evaluate_arithmetics(responses, answer):

    final_answers = []

    for _, sentence in responses.items():

        parts = sentence.split(" ")

        for part in parts[::-1]:

            try:
                ans = float(part)

                final_answers.append(ans)
                break

            except:
                continue


    counter = collections.Counter(
        [x for x in final_answers if x != ""]
    )

    try:

        max_count = max(counter.values())

        most_common = [
            key for key, value in counter.items()
            if value == max_count
        ]

        debate_answer = random.choice(most_common)

    except :

        debate_answer = ""


    return (
        final_answers,
        debate_answer,
        debate_answer == np.round(answer, 1)
    )