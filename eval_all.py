import argparse
import base64
import io
import json
import os

import torch
from PIL import Image
from accelerate import load_checkpoint_and_dispatch, init_empty_weights
from transformers import AutoTokenizer, AutoModel

from omnilmm.model.omnilmm import OmniLMMForCausalLM
from omnilmm.model.utils import build_transform
from omnilmm.train.train_utils import omni_preprocess
from omnilmm.utils import disable_torch_init

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def init_omni_lmm(model_path):
    torch.backends.cuda.matmul.allow_tf32 = True
    disable_torch_init()
    model_name = os.path.expanduser(model_path)
    print(f'Load omni_lmm model and tokenizer from {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, model_max_length=2048)

    if False:
        # model on multiple devices for small size gpu memory (Nvidia 3090 24G x2) 
        with init_empty_weights():
            model = OmniLMMForCausalLM.from_pretrained(model_name, tune_clip=True, torch_dtype=torch.bfloat16)
        model = load_checkpoint_and_dispatch(model, model_name, dtype=torch.bfloat16,
                                             device_map="auto",
                                             no_split_module_classes=['Eva', 'MistralDecoderLayer', 'ModuleList',
                                                                      'Resampler']
                                             )
    else:
        model = OmniLMMForCausalLM.from_pretrained(
            model_name, tune_clip=True, torch_dtype=torch.bfloat16
        ).to(device='cuda', dtype=torch.bfloat16)

    image_processor = build_transform(
        is_train=False, input_size=model.model.config.image_size, std_mode='OPENAI_CLIP')

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    assert mm_use_im_start_end

    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN,
                          DEFAULT_IM_END_TOKEN], special_tokens=True)

    vision_config = model.model.vision_config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
        [DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids(
        [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    image_token_len = model.model.config.num_query

    return model, image_processor, image_token_len, tokenizer


def expand_question_into_multimodal(question_text, image_token_len, im_st_token, im_ed_token, im_patch_token):
    if '<image>' in question_text[0]['content']:
        question_text[0]['content'] = question_text[0]['content'].replace(
            '<image>', im_st_token + im_patch_token * image_token_len + im_ed_token)
    else:
        question_text[0]['content'] = im_st_token + im_patch_token * \
                                      image_token_len + im_ed_token + '\n' + question_text[0]['content']
    return question_text


def wrap_question_for_omni_lmm(question, image_token_len, tokenizer):
    question = expand_question_into_multimodal(
        question, image_token_len, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN)

    conversation = question
    data_dict = omni_preprocess(sources=[conversation],
                                tokenizer=tokenizer,
                                generation=True)

    data_dict = dict(input_ids=data_dict["input_ids"][0],
                     labels=data_dict["labels"][0])
    return data_dict


class OmniLMM12B:
    def __init__(self, model_path) -> None:
        model, img_processor, image_token_len, tokenizer = init_omni_lmm(model_path)
        self.model = model
        self.image_token_len = image_token_len
        self.image_transform = img_processor
        self.tokenizer = tokenizer
        self.model.eval()

    def decode(self, image, input_ids):
        with torch.inference_mode():
            output = self.model.generate_vllm(
                input_ids=input_ids.unsqueeze(0).cuda(),
                images=image.unsqueeze(0).half().cuda(),
                temperature=0.6,
                max_new_tokens=1024,
                # num_beams=num_beams,
                do_sample=True,
                output_scores=True,
                return_dict_in_generate=True,
                repetition_penalty=1.1,
                top_k=30,
                top_p=0.9,
            )

            response = self.tokenizer.decode(
                output.sequences[0], skip_special_tokens=True)
            response = response.strip()
            return response

    def chat(self, input):
        try:
            image = Image.open(io.BytesIO(base64.b64decode(input['image']))).convert('RGB')
        except Exception as e:
            return "Image decode error"

        msgs = json.loads(input['question'])
        input_ids = wrap_question_for_omni_lmm(
            msgs, self.image_token_len, self.tokenizer)['input_ids']
        input_ids = torch.as_tensor(input_ids)
        # print('input_ids', input_ids)
        image = self.image_transform(image)

        out = self.decode(image, input_ids)

        return out


def img2base64(file_name):
    with open(file_name, 'rb') as f:
        encoded_string = base64.b64encode(f.read())
        return encoded_string


class MiniCPMV:
    def __init__(self, model_path) -> None:
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval().cuda()

    def chat(self, input):
        try:
            image = Image.open(io.BytesIO(base64.b64decode(input['image']))).convert('RGB')
        except Exception as e:
            return "Image decode error"

        msgs = json.loads(input['question'])

        answer, context, _ = self.model.chat(
            image=image,
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            sampling=True,
            temperature=0.7
        )
        return answer


class MiniCPMV2_5:
    def __init__(self, model_path) -> None:
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval().cuda()

    def chat(self, input):
        try:
            image = Image.open(io.BytesIO(base64.b64decode(input['image']))).convert('RGB')
        except Exception as e:
            return "Image decode error"

        msgs = json.loads(input['question'])

        answer = self.model.chat(
            image=image,
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=True,
            temperature=0.7
        )
        return answer


class MiniCPMVChat:
    def __init__(self, model_path) -> None:
        if '12B' in model_path:
            self.model = OmniLMM12B(model_path)
        elif 'MiniCPM-Llama3-V' in model_path:
            self.model = MiniCPMV2_5(model_path)
        else:
            self.model = MiniCPMV(model_path)

    def chat(self, input):
        return self.model.chat(input)


def determine_position(center, w=1280, h=720, center_rate=0.25):
    center_x, center_y = center
    center_w, center_h = int(w * center_rate), int(h * center_rate)

    center_x_rect = w / 2
    center_y_rect = h / 2

    # 确定中部区域的边界
    mid_left = center_x_rect - center_w / 2
    mid_right = center_x_rect + center_w / 2
    mid_top = center_y_rect - center_h / 2
    mid_bottom = center_y_rect + center_h / 2

    # 判断是否在中部区域
    if mid_left <= center_x <= mid_right and mid_top <= center_y <= mid_bottom:
        return 'center'
    else:
        # 判断水平位置
        if center_x > mid_right:
            horizontal = 'right'
        else:
            horizontal = 'left'

        # 判断垂直位置
        if center_y > mid_bottom:
            vertical = 'down'
        else:
            vertical = 'top'

        # 组合水平和垂直位置
        return f'{horizontal} {vertical}'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="MiniCPM-V")
    parser.add_argument("--model_path", type=str,
                        default='/nfs/ofs-902-vlm/jiangjing/MiniCPM-Llama3-V-2_5/official_ckpts')
    parser.add_argument("--split", type=str, default='ROOT_TO_GT', choices=['ROOT_TO_GT', 'Test', 'NEW_Mini'])
    parser.add_argument("--provide_bbox", action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()

    # changeable para
    DEV_MODE = False
    model_path = args.model_path
    split = args.split
    provide_bbox = args.provide_bbox

    # fixed para
    answer_keys = {'question_id', 'image', 'question', 'answer'}
    save_dir = '/nfs/ofs-902-1/object-detection/jiangjing/experiments/MiniCPM-V/ans'
    save_dir = f'{save_dir}/{model_path.split("/")[-1]}/{split}'
    data_ann_root_dir = '/nfs/ofs-902-1/object-detection/tangwenbo/vlm/data/CODA-LM'
    data_img_root_dir = '/nfs/ofs-902-1/object-detection/tangwenbo/vlm/data/'
    data_ann_dir = f'{data_ann_root_dir}/{split}/vqa_anno'
    task_list = ['general_perception', 'region_perception', 'driving_suggestion']

    # load model
    if not DEV_MODE:
        chat_model = MiniCPMVChat(model_path)

    # start chat
    task_list_json = []
    for t in task_list:
        task_path = f'{data_ann_dir}/{t}.jsonl'
        with open(task_path, mode='r', encoding='utf-8') as f:
            task_list_json.append(f.read().strip().split('\n'))
    num_samples = len(task_list_json[0])

    answer_list = {
        'general_perception': [],
        'region_perception': [],
        'driving_suggestion': []
    }

    region_i = 0
    for i in range(num_samples):
        print(f'evaluating [{i + 1}/{num_samples}]')

        msgs = []
        answer = None

        # 1. general_perception
        task_id, task_name = 0, 'general_perception'
        task_json = json.loads(task_list_json[task_id][i])
        ori_img_name = task_json['image'].split('/')[-1].split('.')[0]

        im_64 = img2base64(f"{data_img_root_dir}/{task_json['image']}")
        # chat
        if answer is not None:
            msgs.append({"role": "assistant", "content": answer})
        msgs.append({"role": "user", "content": task_json['question']})
        input = {"image": im_64, "question": json.dumps(msgs, ensure_ascii=True)}
        if not DEV_MODE:
            answer = chat_model.chat(input)
        # save_ans
        answer_list[task_name].append(answer)

        # 2. region_perception
        task_id, task_name = 1, 'region_perception'
        while True:
            try:
                task_json = json.loads(task_list_json[task_id][region_i])
            except IndexError:
                break
            img_name = task_json['image'].split('/')[-1].split('.')[0]
            if img_name.find(ori_img_name) == -1:
                # not found
                break

            region_i += 1
            im_64 = img2base64(f"{data_img_root_dir}/{task_json['image']}")
            # chat
            if answer is not None:
                msgs.append({"role": "assistant", "content": answer})

            if provide_bbox:
                question = task_json['question']
                bbox = task_json['bbox']
                center = [bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2]
                location = determine_position(center)
                question += f' Note that the red rectangle is at the {location} of the provided image, exact rectangle with [x, y, width, height] = {bbox}.'
            else:
                question = task_json['question']

            msgs.append({"role": "user", "content": question})
            input = {"image": im_64, "question": json.dumps(msgs, ensure_ascii=True)}
            if not DEV_MODE:
                answer = chat_model.chat(input)
            # save_ans
            answer_list[task_name].append(answer)

        # 3. driving_suggestion
        task_id, task_name = 2, 'driving_suggestion'
        task_json = json.loads(task_list_json[task_id][i])

        im_64 = img2base64(f"{data_img_root_dir}/{task_json['image']}")
        # chat
        if answer is not None:
            msgs.append({"role": "assistant", "content": answer})
        msgs.append({"role": "user", "content": task_json['question']})
        input = {"image": im_64, "question": json.dumps(msgs, ensure_ascii=True)}
        if not DEV_MODE:
            answer = chat_model.chat(input)
        # save_ans
        answer_list[task_name].append(answer)

    # dump ans
    task_list_json_dump = {
        'general_perception': [],
        'region_perception': [],
        'driving_suggestion': []
    }
    for task_id, task_name in enumerate(task_list):
        assert len(task_list_json[task_id]) == len(answer_list[task_name])

        for i in range(len(answer_list[task_name])):
            task_json = json.loads(task_list_json[task_id][i])
            task_json['answer'] = answer_list[task_name][i]

            # only answer keys are dumped
            new_task_json = {}
            for key in task_json:
                if key in answer_keys:
                    new_task_json[key] = task_json[key]
            task_list_json_dump[task_name].append(json.dumps(new_task_json))

    os.makedirs(save_dir, exist_ok=True)
    for task_name in task_list:
        save_path = f'{save_dir}/{task_name}_answer.jsonl'
        with open(save_path, mode='w', encoding='utf-8') as f:
            f.write('\n'.join(task_list_json_dump[task_name]))
