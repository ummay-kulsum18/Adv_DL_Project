import argparse
import copy
import torch
import warnings
from decord import VideoReader, cpu
import numpy as np
import json
import multiprocessing as mp
import os
from multiprocessing import Pool
import functools
import itertools
import random
from tqdm import tqdm
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import rank0_print


warnings.filterwarnings("ignore")


def load_video(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time

def load_video_frames(video_file):
    images = sorted(os.listdir(video_file))
    video = []
    for image in images:
        frame_path = os.path.join(video_file, image)
        with Image.open(frame_path) as img:
            frame = img.convert("RGB")
            video.append(frame)
    return video

def get_prompt(sample, conv_template="qwen_1_5", video_time=None, num_frames=None, frame_time=None):
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    if video_time:
        prompt = f"The video lasts for {video_time:.2f} seconds, and {num_frames} frames are uniformly sampled from it. These frames are located at {frame_time}.\n"
    else:
        prompt = ""

    prompt += sample["question"]
    
    question = DEFAULT_IMAGE_TOKEN + prompt
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

def run(rank, world_size, args):
    torch.cuda.set_device(rank)

    rank0_print("Loadind dataset from", args.data_path)
    with open(args.data_path, "r") as f:
        dataset = json.load(f)
    
    if args.max_frames_num==64:
        with open('/mnt/meg/mmiemon/datasets/YTScam/YTscam_metadata_w_crypto.json', 'r') as f:
            yt_metadata = json.load(f)
    elif args.max_frames_num==100:
        with open('/mnt/meg/mmiemon/datasets/YTScam/YTscam_metadata_frames_100.json', 'r') as f:
            yt_metadata = json.load(f)       
 
    random.shuffle(dataset)

    num_samples = int(len(dataset) * args.test_ratio)
    dataset = dataset[rank:num_samples:world_size]
    rank0_print(f"Total samples: {num_samples}")
    print(f"Samples in rank {rank}: {len(dataset)}")

    tokenizer, model, image_processor, max_length = load_pretrained_model(
                                                        model_path = args.model_path, 
                                                        model_base = args.model_base, 
                                                        model_name = args.model_name, 
                                                        torch_dtype="bfloat16", 
                                                        device_map="auto",  
                                                        #device_map = {"": torch.device(f"cuda:{rank}")},
                                                    )
    model.eval()
    #model = model.to(torch.device(rank))

    result_list = []
    for cnt, sample in enumerate(tqdm(dataset)):
        sample_save_path = f"{args.results_dir}/outputs/{sample['video_id']}.json"
        if os.path.exists(sample_save_path):
            continue
        if args.max_frames_num==64:
            frames_path = sample["source"] + "_frames"
        elif args.max_frames_num==100:
            frames_path = sample["source"] + "_frames_100"

        video_path = os.path.join(args.video_root, frames_path, sample['video_id'])
        video = load_video_frames(video_path)
        video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
        video = [video]
        if args.use_time_ins:
            metadata = yt_metadata[sample["video_id"]]
            video_time = metadata['video_time']
            frame_time = metadata['frame_time']
            num_frames_to_sample = metadata['num_frames_to_sample']
            prompt_question = get_prompt(sample, video_time=video_time, num_frames=args.max_frames_num, frame_time=frame_time)
        else:
            prompt_question = get_prompt(sample)

        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
        cont = model.generate(
            input_ids,
            images=video,
            modalities= ["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        sample["prediction"] = text_outputs

        with open(sample_save_path, "w") as f:
            json.dump(sample, f, indent=4)
        
        result_list.append(sample)
        print(cnt, "GT:", sample["label"], "Pred:", sample["prediction"])
    
    return result_list


def main():
    parser = argparse.ArgumentParser(description="Run Inference")

    # Model
    parser.add_argument("--model_name", type=str, default="llava_qwen_lora")
    parser.add_argument("--model_base", type=str, default="lmms-lab/LLaVA-Video-7B-Qwen2")
    parser.add_argument("--model_path", type=str, default="/mnt/bum/mmiemon/LLaVA-NeXT/work_dirs/llava_video_yt_scam_64f")
    parser.add_argument("--max_frames_num", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--conv_template", type=str, default="qwen_1_5")
    parser.add_argument("--use_time_ins", action="store_true")

    # Data
    parser.add_argument("--data_path", type=str, default="/mnt/meg/mmiemon/datasets/YTScam/eval_data_YTscam.json")
    parser.add_argument("--video_root", type=str, default="/mnt/meg/mmiemon/datasets/YTScam")
    parser.add_argument("--results_dir", type=str, default="/mnt/bum/mmiemon/LLaVA-NeXT/results/yt_scam/llava_video_64f_time_ins")
    parser.add_argument("--test_ratio", type=float, default=1)
    parser.add_argument("--multiprocess", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(f"{args.results_dir}/outputs", exist_ok=True)

    if args.multiprocess:
        mp.set_start_method("spawn")
        print(f"started benchmarking")
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus
        print("World size", world_size)
        with Pool(world_size) as pool:
            func = functools.partial(run, args=args, world_size=world_size)
            result_lists = pool.map(func, range(world_size))

        print("finished running")
        result_list = [res for res in itertools.chain(*result_lists)]
    else:
        result_list = run(0, world_size=1, args=args)

    


if __name__ == "__main__":
    main()