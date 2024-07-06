import json
import os
import random
import shutil

from tqdm import tqdm

random.seed(1234)


def prepare_all_tasks(name='train_all_tasks'):
    # fixed para
    save_dir = '/nfs/ofs-902-1/object-detection/jiangjing/experiments/MiniCPM-V/finetune_data'
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/{name}.json'

    task_list = ['general_perception', 'region_perception', 'driving_suggestion']
    data_ann_root_dir = '/nfs/ofs-902-1/object-detection/tangwenbo/vlm/data/CODA-LM'
    data_img_root_dir = '/nfs/ofs-902-1/object-detection/tangwenbo/vlm/data/'

    if name.find('train') != -1:
        splits = ['Train', 'Val']
    elif name.find('eval') != -1:
        splits = ['Mini']
    else:
        raise Exception

    json_samples = []
    for split in splits:
        data_ann_dir = f'{data_ann_root_dir}/{split}/vqa_anno'

        # start chat
        task_list_json = []
        for t in task_list:
            task_path = f'{data_ann_dir}/{t}.jsonl'
            with open(task_path, mode='r', encoding='utf-8') as f:
                text_list = f.read().strip().split('\n')

                if name.find('eval') != -1 and t == 'general_perception':
                    # process error
                    text_list = [text_list[0] + text_list[1]] + text_list[2:]

                task_list_json.append(text_list)

        num_samples = len(task_list_json[0])

        region_i = 0
        for i in tqdm(range(num_samples)):
            # 1. general_perception
            task_id, task_name = 0, 'general_perception'
            task_json = json.loads(task_list_json[task_id][i])
            ori_img_name = task_json['image'].split('/')[-1].split('.')[0]

            # chat
            msgs = []
            msgs.append({"role": "user", "content": task_json['question']})
            msgs.append({"role": "assistant", "content": task_json['answer']})
            # save_ans
            sample = {
                "id": len(json_samples),
                "image": f"{data_img_root_dir}/{task_json['image']}",
                "conversations": msgs
            }
            json_samples.append(sample)

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

                # chat
                msgs = []
                msgs.append({"role": "user", "content": task_json['question']})
                msgs.append({"role": "assistant", "content": task_json['answer']})
                # save_ans
                sample = {
                    "id": len(json_samples),
                    "image": f"{data_img_root_dir}/{task_json['image']}",
                    "conversations": msgs
                }
                json_samples.append(sample)

            # 3. driving_suggestion
            task_id, task_name = 2, 'driving_suggestion'
            task_json = json.loads(task_list_json[task_id][i])

            # chat
            msgs = []
            msgs.append({"role": "user", "content": task_json['question']})
            msgs.append({"role": "assistant", "content": task_json['answer']})
            # save_ans
            sample = {
                "id": len(json_samples),
                "image": f"{data_img_root_dir}/{task_json['image']}",
                "conversations": msgs
            }
            json_samples.append(sample)
    # dump ans
    with open(save_path, mode='w', encoding='utf-8') as f:
        f.write(json.dumps(json_samples))


def check_split_region_class(splits=None):
    if splits is None:
        splits = ['Mini']
    label_dict = {
        "vehicle": ["car", "truck", "tram", "tricycle", "bus", "trailer", "construction_vehicle",
                    "recreational_vehicle"],
        "vru": ["pedestrian", "cyclist", "bicycle", "moped", "motorcycle", "stroller", "wheelchair", "cart"],
        "traffic_sign": ["warning_sign", "traffic_sign"],
        "traffic_light": ["traffic_light"],
        "traffic_cone": ["traffic_cone"],
        "barrier": ["barrier", "bollard"],
        "miscellaneous": ["dog", "cat", "sentry_box", "traffic_box", "traffic_island", "debris", "suitcace",
                          "dustbin", "concrete_block", "machinery", "chair", "phone_booth", "basket", "cardboard",
                          "carton", "garbage", "garbage_bag", "plastic_bag", "stone", "tire", "misc"],
    }

    root_dir = '/nfs/ofs-902-1/object-detection/tangwenbo/vlm/data/CODA-LM/'
    category_info = {}
    total_category_info = {
        "vehicle": 0,
        "vru": 0,
        "traffic_sign": 0,
        "traffic_light": 0,
        "traffic_cone": 0,
        "barrier": 0,
        "miscellaneous": 0
    }

    for split in splits:
        anno_dir = f'{root_dir}/{split}'
        for anno_name in tqdm(os.listdir(anno_dir)):
            if not anno_name.endswith('.json'):
                continue
            with open(f'{anno_dir}/{anno_name}', mode='r', encoding='utf-8') as f:
                json_data = json.loads(f.read())
                for key in json_data['region_perception']:
                    category_name = json_data['region_perception'][key]['category_name']
                    category_info[category_name] = category_info.get(category_name, 0) + 1
    print(category_info)
    total_num = sum(v for k, v in category_info.items())
    print(total_num)

    for k, v in category_info.items():
        find = False
        for key in label_dict:
            if k in set(label_dict[key]):
                total_category_info[key] += v
                find = True
                break
        if not find:
            print(f'not find {k}')
    print(total_category_info)
    total_num = sum(v for k, v in total_category_info.items())
    print(total_num)


def split_val_test(splits, save_dir='/nfs/ofs-902-1/object-detection/tangwenbo/vlm/data/CODA-LM/NEW_Mini',
                   samples_per_category=10):
    if splits is None:
        splits = ['Mini']
    os.makedirs(save_dir, exist_ok=True)

    label_dict = {
        "vehicle": ["car", "truck", "tram", "tricycle", "bus", "trailer", "construction_vehicle",
                    "recreational_vehicle"],
        "vru": ["pedestrian", "cyclist", "bicycle", "moped", "motorcycle", "stroller", "wheelchair", "cart"],
        "traffic_sign": ["warning_sign", "traffic_sign"],
        "traffic_light": ["traffic_light"],
        "traffic_cone": ["traffic_cone"],
        "barrier": ["barrier", "bollard"],
        "miscellaneous": ["dog", "cat", "sentry_box", "traffic_box", "traffic_island", "debris", "suitcace",
                          "dustbin", "concrete_block", "machinery", "chair", "phone_booth", "basket", "cardboard",
                          "carton", "garbage", "garbage_bag", "plastic_bag", "stone", "tire", "misc"],
    }

    root_dir = '/nfs/ofs-902-1/object-detection/tangwenbo/vlm/data/CODA-LM/'
    category_info = {}
    total_category_info = {
        "vehicle": [],
        "vru": [],
        "traffic_sign": [],
        "traffic_light": [],
        "traffic_cone": [],
        "barrier": [],
        "miscellaneous": []
    }

    for split in splits:
        anno_dir = f'{root_dir}/{split}'
        for anno_name in tqdm(os.listdir(anno_dir)):
            if not anno_name.endswith('.json'):
                continue
            anno_full_path = f'{anno_dir}/{anno_name}'
            with open(f'{anno_dir}/{anno_name}', mode='r', encoding='utf-8') as f:
                json_data = json.loads(f.read())
                for key in json_data['region_perception']:
                    category_name = json_data['region_perception'][key]['category_name']
                    if category_name not in category_info:
                        category_info[category_name] = []
                    category_info[category_name].append(anno_full_path)

    for k, v in category_info.items():
        find = False
        for key in label_dict:
            if k in set(label_dict[key]):
                total_category_info[key] += v
                find = True
                break
        if not find:
            print(f'not find {k}')

    total_category_info_count = {k: len(v) for k, v in total_category_info.items()}
    print(total_category_info_count)

    val_sample_dict = {}
    for key in total_category_info:
        data_path_list = total_category_info[key]
        if key in ['traffic_sign', 'traffic_light']:
            sample_num = samples_per_category
        else:
            sample_num = samples_per_category // 2

        try:
            val_sample_dict[key] = random.sample(data_path_list, sample_num)
        except ValueError:
            val_sample_dict[key] = data_path_list

    print(val_sample_dict)
    val_sample_dict_count = {k: len(v) for k, v in val_sample_dict.items()}
    print(val_sample_dict_count)

    # dump
    for class_name in tqdm(val_sample_dict):
        for sample in val_sample_dict[class_name]:
            sample_name = sample.split('/')[-1]
            shutil.copyfile(sample, f'{save_dir}/{sample_name}')
    print(len(os.listdir(save_dir)))


if __name__ == '__main__':
    # prepare_all_tasks(name='train_all_tasks')
    # prepare_all_tasks(name='eval_all_tasks')

    # split_val_test(splits=['Val'], samples_per_category=10)
    check_split_region_class(splits=['NEW_Mini'])
