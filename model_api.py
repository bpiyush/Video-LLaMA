"""Simplified API access to the model."""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
from glob import glob
from tqdm import tqdm
import argparse
import os
import random
from termcolor import colored

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



def load_config():
    curr_filepath = os.path.abspath(__file__)
    curr_dirpath = os.path.dirname(curr_filepath)
    cfg_path = os.path.join(
        curr_dirpath, "eval_configs/video_llama_eval_only_vl_edited.yaml"
    )
    assert os.path.exists(cfg_path), f"Config file not found at {cfg_path}."

    # Load config
    args = dict(
        cfg_path=cfg_path,
        model_type="llama_v2",
        gpu_id=0,
        options=[],
        seed=0,
    )
    args = AttrDict(args)
    cfg = Config(args)
    return args, cfg


def load_model(args, cfg, low_resource=False):
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)

    print(colored("[:::] Loading model.", "green"))
    device_str = 'cuda:{}'.format(args.gpu_id)
    model_config.low_resource = low_resource
    if low_resource:
        print("Loading in low resource mode.")
        print("Note: this needs accelerate==0.16.0 installed.")
        print("Also, make sure to have right bitsandbytes installed compatible with CUDA.")
        # https://github.com/TimDettmers/bitsandbytes/issues/1059
        raise NotImplementedError("Currently, the bitandbytes package is not compatible.")
    model = model_cls.from_config(model_config).to(device_str)
    model = model.eval()
    n_params = np.sum([p.numel() for p in model.parameters()]) / 1e9
    print(f"[:::] Model has {n_params:.3f}B parameters.")
    print(colored("[:::] Model loaded.", "green"))

    # Load vision processor
    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor = registry.get_processor_class(
        vis_processor_cfg.name,
    ).from_config(vis_processor_cfg)

    # Load chat
    chat = Chat(model, vis_processor, device=device_str)

    return chat, model, vis_processor


def ask_about_video(chat, video_path, user_message, num_beams=1, temperature=1.0):
    # Check normal prompt
    chat_state = conv_llava_llama_2.copy()
    chat_state.system =  "You are able to understand the visual content that "\
        "the user provides."\
        "Follow the instructions carefully and explain your answers in detail."
    img_list = []
    llm_message = chat.upload_video_without_audio(
        video_path, chat_state, img_list, video_loader="load_video_cv2",
    )


    # user_message = f"""
    # Given this video, you have to select which is the option
    # that correctly describes the video.
    # (a) {text_options[0]} (b) {text_options[1]}

    # You have to only answer (a) or (b).
    # """
    chat.ask(user_message, chat_state)


    llm_message = chat.answer(
        conv=chat_state,
        img_list=img_list,
        num_beams=num_beams,
        temperature=temperature,
        max_new_tokens=300,
        max_length=2000,
    )[0]
    return llm_message


if __name__ == "__main__":

    # Get config
    args, cfg = load_config()

    # Load model
    chat, model, vis_processor = load_model(args, cfg, low_resource=False)
