import json
import os

from tqdm import tqdm


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


if __name__ == '__main__':
    prepare_all_tasks(name='train_all_tasks')
    prepare_all_tasks(name='eval_all_tasks')
