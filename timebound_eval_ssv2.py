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
from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank
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


def setup_seeds(seed):
    seed = seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_video_path(video_dir, video_id, ext="webm"):
    paths = glob(os.path.join(video_dir, f"*/{video_id}.{ext}"))
    assert len(paths) == 1
    return paths[0]


def get_llm_answer(
        video_path=None, text_options=None, num_beams=1, temperature=1.0,
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
    Given this video, you have to select which is the option
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

    correct_answer = f"(a) {text_options[0]}"
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
    Given this video, you have to select which is the option
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

    correct_answer = f"(b) {text_options[0]}"
    is_correct += int(correct_answer in llm_message)

    accuracy = is_correct / 2.
    return accuracy


if __name__ == "__main__":

    # Load config
    args = dict(
        cfg_path="./eval_configs/video_llama_eval_only_vl_edited.yaml",
        model_type="llama_v2",
        gpu_id=0,
        options=[],
        seed=0,
    )
    args = AttrDict(args)
    cfg = Config(args)

    # Set up seeds
    setup_seeds(args.seed)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)


    # Load model
    print("[:::] Load model.")
    device_str = 'cuda:{}'.format(args.gpu_id)
    model = model_cls.from_config(model_config).to(device_str)
    model = model.eval()
    n_params = np.sum([p.numel() for p in model.parameters()]) / 1e9
    print(f"Model has {n_params:.3f}B parameters.")

    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor = registry.get_processor_class(
        vis_processor_cfg.name,
    ).from_config(vis_processor_cfg)

    # Load chat
    chat = Chat(model, vis_processor, device=device_str)

    # Debug
    debug = True
    if debug:
        is_correct = get_llm_answer()
        print(f"Accuracy: {is_correct:.2f}")
    

    # Load data
    print("[:::] Load data.")

    csv_path = "/scratch/shared/nfs2/piyush/datasets/SSv2/metadata/time_antonyms-validation.csv"
    df = pd.read_csv(csv_path)

    data_dir = "/scratch/shared/beegfs/shared-datasets/SomethingSomething-V2/"
    video_dir = os.path.join(data_dir, "videos")

    iterator = tqdm(df.iterrows(), total=len(df))
    text_corrects = []
    failed = []
    for i, row in iterator:
        row = row.to_dict()
        video_path_x = get_video_path(video_dir, row["id_x"])
        video_path_y = get_video_path(video_dir, row["id_y"])
        label_x = row["label_x"]
        label_y = row["label_y"]

        try:
            # Check for first video
            is_correct = get_llm_answer(video_path_x, [label_x, label_y])
            text_corrects.append(is_correct)

            # Check for second video
            is_correct = get_llm_answer(video_path_y, [label_y, label_x])
            text_corrects.append(is_correct)
        except:
            failed.append(i)

        if debug:
            if i == 10:
                break
    
    print("Number of failed: ", len(failed))

    text_corrects = np.array(text_corrects)
    print("Video to text accuracy: ", text_corrects.mean())
