import os
import sys
import json

sys.path.insert(0, '/content/visual_genome_python_driver')
from visual_genome import api

with open('data/captions_val2014.json') as g:  # COCO_PATH is from git
    caption_data = json.load(g)
    name_to_id = [(x['file_name'], x['id']) for x in caption_data['images']]
    name_to_id = dict(name_to_id)

with open('data/vg_test.txt', 'r') as g:
    with open('data/vg_train.txt', 'a', encoding='utf-8') as f:
        text = [x.strip() + '\n' for x in g]
        f.writelines(text)

# read coco_train
with open('data/coco_train.txt', 'r') as g:
    coco_train = []
    for name in g:
        name = os.path.splitext(name)[0]
        name = name.split("_")[-1]
        coco_train.append(int(str(name)))

# read coco_val
with open('data/coco_val.txt', 'r') as g:
    coco_val = [name_to_id[x.strip()] for x in g]
    # coco_val = g.readlines()

# read coco_test
with open('data/coco_test.txt', 'r') as g:
    coco_test = [name_to_id[x.strip()] for x in g]
    # coco_test=[]
    # for name in g:
    #   name = os.path.splitext(name)[0]
    #   name = name.split("_")[-1]
    #   coco_test.append(int(str(name)))

# read vg
with open('data/vg_train.txt', 'r') as g:
    vg_coco = []  # vg 数据集中有coco_id
    vg_coco_val = []
    vg_coco_test = []
    vg_coco_train = []

    for name in g:
        name = name.strip()
        id = os.path.splitext(name)[0]
        kk = api.get_image_data(id).coco_id
        if kk != 'null':
            if kk in coco_val:
                vg_coco_val.append(name)
            elif kk in coco_test:
                vg_coco_test.append(name)
            elif kk in coco_train:
                vg_coco_train.append(name)
            else:
                vg_coco.append(name)
        else:
            vg_coco.append(name)

    print('number of vg has not coco_id :', len(vg_coco))
    print('number of coco train :', len(vg_coco_train))
    print('number of coco val :', len(vg_coco_val))
    print('number of coco test :', len(vg_coco_test))
    # with open('data/vg_coco_id.txt', 'w', encoding='utf-8') as f:
    #     text = [x + '\n' for x in vg_coco]
    #     f.writelines(text)

    # with open('data/vg_coco_train.txt', 'w', encoding='utf-8') as f:
    #     text = [x + '\n' for x in vg_coco_train]
    #     f.writelines(text)

    # with open('data/vg_coco_val.txt', 'w', encoding='utf-8') as f:
    #     text = [x + '\n' for x in vg_coco_val]
    #     f.writelines(text)

    # with open('data/vg_coco_test.txt', 'w', encoding='utf-8') as f:
    #     text = [x + '\n' for x in vg_coco_test]
    #     f.writelines(text)




