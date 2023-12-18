import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


model_name_var = None
tokenizer_var = None
model_var = None
image_processor_var = None
context_len_var = None
roles_var = None
conv_var = None

def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    global model_name_var
    model_name_var = model_name

    global tokenizer_var
    tokenizer_var = tokenizer

    global model_var
    model_var = model

    global image_processor_var
    image_processor_var = image_processor

    global context_len_var
    context_len_var = context_len

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    global roles_var

    global conv_var
    conv_var = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles_var = ('user', 'assistant')
    else:
        roles_var = conv_var.roles

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
parser.add_argument("--model-base", type=str, default=None)
parser.add_argument("--image-file", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--conv-mode", type=str, default=None)
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--max-new-tokens", type=int, default=512)
parser.add_argument("--load-8bit", action="store_true")
parser.add_argument("--load-4bit", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--image-aspect-ratio", type=str, default='pad')
args = parser.parse_args()
main(args)


def get_llava():
    image = load_image(args.image_file)
    # Similar operation in model_worker.py
    image_tensor = process_images([image], image_processor_var, args)
    if type(image_tensor) is list:
        image_tensor = [image.to(model_var.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model_var.device, dtype=torch.float16)

    global roles_var
    try:
        inp = input(f"{roles_var[0]}: ")
    except EOFError:
        inp = ""

    print(f"{roles_var[1]}: ", end="")

    if image is not None:
        # first message
        if model_var.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv_var.append_message(conv_var.roles[0], inp)
        image = None
    else:
        # later messages
        conv_var.append_message(conv_var.roles[0], inp)
    conv_var.append_message(conv_var.roles[1], None)
    prompt = conv_var.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer_var, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model_var.device)
    stop_str = conv_var.sep if conv_var.sep_style != SeparatorStyle.TWO else conv_var.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer_var, input_ids)
    streamer = TextStreamer(tokenizer_var, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        output_ids = model_var.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer_var.decode(output_ids[0, input_ids.shape[1]:]).strip()

    return outputs


if __name__ == "__main__":
    print("Hello")