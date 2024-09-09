"""Evaluates text score for aubset of SSv2."""
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

# def load_model():
#     # Load config
#     args = dict(
#         cfg_path="./eval_configs/video_llama_eval_only_vl_edited.yaml",
#         model_type="llama_v2",
#         gpu_id=0,
#         options=[],
#         seed=0,
#     )
#     args = AttrDict(args)
#     cfg = Config(args)

#     # Set up seeds
#     setup_seeds(args.seed)

#     model_config = cfg.model_cfg
#     model_config.device_8bit = args.gpu_id
#     model_cls = registry.get_model_class(model_config.arch)


#     # Load model
#     print("[:::] Load model.")
#     device_str = 'cuda:{}'.format(args.gpu_id)
#     model = model_cls.from_config(model_config).to(device_str)
#     model = model.eval()
#     n_params = np.sum([p.numel() for p in model.parameters()]) / 1e9
#     print(f"Model has {n_params:.3f}B parameters.")

#     vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
#     vis_processor = registry.get_processor_class(
#         vis_processor_cfg.name,
#     ).from_config(vis_processor_cfg)

#     # Load chat
#     chat = Chat(model, vis_processor, device=device_str)
#     return cfg, model, chat


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
    iterator = su.log.tqdm_iterator(range(N), desc="Evaluating on SSv2")
    for i in iterator:
        row = df_pair.iloc[i].to_dict()
        id_a = row["id_a"]
        id_b = row["id_b"]

        video_path_a = ssv2.get_video_path_basic(id_a)
        assert os.path.exists(video_path_a)
        row_a = df_main[df_main.id == str(id_a)].iloc[0].to_dict()
        video_path_b = ssv2.get_video_path_basic(id_b)
        assert os.path.exists(video_path_b)
        row_b = df_main[df_main.id == str(id_b)].iloc[0].to_dict()

        # Get text score
        try:
            a = llm_answer_plain(
                chat, video_path=video_path_a, text_options=[row_a["label"], row_b["label"]],
            )
            b = llm_answer_plain(
                chat, video_path=video_path_b, text_options=[row_b["label"], row_a["label"]],
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
    save_path = "./results/videollama_ssv2_text_score.csv"
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
    import video_language.datasets.ssv2 as ssv2
    paths = ssv2.get_paths()
    split = "validation"
    df_main = ssv2.load_main_csv(paths, split)
    df_time = ssv2.load_time_csv(paths, split)
    df_pair = ssv2.load_pair_csv(paths, split)

    # Gather results
    gather_results(df_pair)
