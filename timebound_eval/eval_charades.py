"""Evaluates text score for aubset of Charades."""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
from glob import glob
from tqdm import tqdm
import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import pandas as pd

import sys
sys.path.append("../")
from video_llama.common.config import Config
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import (
    Chat,
    Conversation,
    default_conversation,
    SeparatorStyle,
    conv_llava_llama_2,
)
import decord
decord.bridge.set_bridge('torch')

from video_llama.datasets.builders import *
from video_llama.models import *
from video_llama.processors import *
from video_llama.runners import *
from video_llama.tasks import *

from timebound_eval.misc import setup_seeds, AttrDict, get_world_size, get_rank
from model_api import (
    load_config, load_model, setup_seeds, ask_about_video,
)


def llm_answer_plain(
        chat, video_path=None, text_options=None, num_beams=1, temperature=1.0,
    ):
    if video_path is None:
        video_path = "../TimeBound.v1/sample_data/folding_paper.mp4"
    if text_options is None:
        text_options = [
            "Someone folding a paper.",
            "Someone unfolding a paper.",
        ]
    assert os.path.exists(video_path)
    assert len(text_options) == 2


    # Check normal prompt
    chat_state = conv_llava_llama_2.copy()
    chat_state.system =  "You are able to understand the visual content that "\
        "the user provides."\
        "Follow the instructions carefully and explain your answers in detail."
    img_list = []
    llm_message = chat.upload_video_without_audio(
        video_path, chat_state, img_list, video_loader="load_video_cv2",
    )


    user_message = f"""
    Given this video, you have to select which is the text sentence
    that correctly describes the video.
    (a) {text_options[0]} (b) {text_options[1]}

    You have to only answer (a) or (b).
    """
    chat.ask(user_message, chat_state)


    llm_message = chat.answer(
        conv=chat_state,
        img_list=img_list,
        num_beams=num_beams,
        temperature=temperature,
        max_new_tokens=300,
        max_length=2000,
    )[0]

    correct_answer = f"(a)"
    is_correct = int(correct_answer in llm_message)


    # Check reversed prompt
    chat_state = conv_llava_llama_2.copy()
    chat_state.system =  "You are able to understand the visual content that "\
        "the user provides."\
        "Follow the instructions carefully and explain your answers in detail."
    img_list = []
    llm_message = chat.upload_video_without_audio(
        video_path, chat_state, img_list, video_loader="load_video_cv2",
    )


    user_message = f"""
    Given this video, you have to select which is the text sentence
    that correctly describes the video.
    (a) {text_options[1]} (b) {text_options[0]}

    You have to only answer (a) or (b).
    """
    chat.ask(user_message, chat_state)


    llm_message = chat.answer(
        conv=chat_state,
        img_list=img_list,
        num_beams=num_beams,
        temperature=temperature,
        max_new_tokens=300,
        max_length=2000,
    )[0]

    correct_answer = f"(b)"
    # NOTE: both (a) vs (b) and (b) vs (a) should be correct
    is_correct *= int(correct_answer in llm_message)

    return is_correct


def gather_results(df_pair):
    N = len(df_pair)
    results = []
    iterator = su.log.tqdm_iterator(range(N), desc="Evaluating on Charades")
    for i in iterator:
        row = df_pair.iloc[i].to_dict()

        id_a = row["id_a"]
        id_b = row["id_b"]

        row_a = df_main[df_main["item_id"] == id_a].iloc[0]
        video_path_a = charades.get_clips_path(row_a['item_id'])
        assert os.path.exists(video_path_a)
        label_a = row_a["cls_name"]

        row_b = df_main[df_main["item_id"] == id_b].iloc[0]
        video_path_b = charades.get_clips_path(row_b['item_id'])
        assert os.path.exists(video_path_b)
        label_b = row_b["cls_name"]

        # Get text score
        try:
            a = llm_answer_plain(
                chat, video_path=video_path_a, text_options=[label_a, label_b],
            )
            b = llm_answer_plain(
                chat, video_path=video_path_b, text_options=[label_b, label_a],
            )
            text_flag = a and b
            results.append({"text": text_flag})
        except:
            print("Skipping video: ", id_a, id_b)
            continue
    results = pd.DataFrame(results)
    scores = results.mean()
    print("Result: ", scores)

    # Save result
    save_path = "./results/videollama_charades_text_score.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    scores.to_csv(save_path)



if __name__ == "__main__":

    # Get config
    args, cfg = load_config()

    # Load model
    chat, model, vis_processor = load_model(args, cfg, low_resource=False)

    # Load data
    sys.path.append("../TimeBound.v1/")
    import shared.utils as su
    import video_language.datasets.charades as charades
    split = "test"
    paths = charades.get_paths()
    _, df_main = charades.load_main_csv(paths, split)
    df_time = charades.load_time_csv(paths, split)
    df_pair = charades.load_pair_csv(paths, split)

    # Gather results
    gather_results(df_pair)

