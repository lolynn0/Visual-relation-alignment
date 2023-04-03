"""Convert the images and the detected object labels to tfrecords."""
import os
import sys
import pickle as pkl
import random
import json

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from functools import partial
from config import TF_MODELS_PATH

from misc_fn import _int64_feature_list

sys.path.insert(0, TF_MODELS_PATH + '/research/im2txt/im2txt')
sys.path.append(TF_MODELS_PATH + '/research')
sys.path.append(TF_MODELS_PATH + '/research/object_detection')
from inference_utils import vocabulary

# from utils import label_map_util


sys.path.insert(0, '/home/xzheng10/workspace/unsupervised_captioning')

from misc_fn import _bytes_feature
from misc_fn import _float_feature_list
from misc_fn import _int64_feature_list

tf.enable_eager_execution()

flags.DEFINE_string('image_path', './dataset/all_images/VG_100K', 'Path to all coco images.')
# flags.DEFINE_string('f', '', 'kernel')
tf.flags.DEFINE_string("word_counts_output_file", "data/word_counts.txt",
                       "Text file containing the vocabulary.")

FLAGS = flags.FLAGS


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    value = tf.compat.as_bytes(value)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj

    def read(self, size):
        return self.fileobj.read(size).encode()

    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()


def load_info(info_file):
    """
  Loads the file containing the visual genome label meanings
  :param info_file: JSON
  :return: ind_to_classes: sorted list of classes
           ind_to_predicates: sorted list of predicates
  """
    info = json.load(open(info_file, 'r'))
    info['label_to_idx']['__background__'] = 0
    info['predicate_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])

    # return ind_to_classes, ind_to_predicates
    return predicate_to_ind


def get_vocab(category_name):
    with open('data/glove_vocab.pkl', 'rb') as f:  ## 全部词向量矩阵 glove.word_vectors
        glove = pkl.load(f, encoding='bytes')
        glove.append('<S>')
        glove.append('</S>')
        glove = set(glove)
    with open(FLAGS.word_counts_output_file, 'r') as ff:
        vocab = list(ff)
        vocab = [i.strip() for i in vocab]
        vocab = [i.split() for i in vocab]
        # [['<S>', '2322635'], ['</S>', '2322635']]....
        vocab = [(i, int(j)) for i, j in vocab]
    category_set = set(category_name)
    word_counts = [i for i in vocab if i[0] in category_set or i[1] >= 40]
    words = set([i[0] for i in word_counts])
    for i in category_name:
        if i not in words:
            word_counts.append((i, 0))
    # with open(new_vocab.txt, 'w') as f:
    #   f.write('\n'.join(['%s %d' % (w, c) for w, c in word_counts]))

    vocab = vocabulary.Vocabulary(FLAGS.word_counts_output_file)

    return vocab


def get_list_relation(vocab, x):
    return [vocab.word_to_id(y) for y in x]


def image_generator(split):
    with open('/home/xzheng10/workspace/data_release/objects.json', 'r') as f:
        all_objects = json.load(f)

    with open('/home/xzheng10/workspace/data_release/relationships.json', 'r') as f:
        all_rel = json.load(f)

    with open('/home/xzheng10/workspace/data_release/VG-SGG-dicts.json', 'r') as f:
        vg_dict = json.load(f)
        obj_label = list(vg_dict['object_count'].keys())
        pred_label = list(vg_dict['predicate_count'].keys())

    imageid_ind = {}
    for jj in range(len(all_objects)):
        keyyy = all_objects[jj]['image_id']
        imageid_ind[keyyy] = jj

    pred_to_ind = load_info('/home/xzheng10/workspace/data_release/VG-SGG-dicts.json')

    with open('data/vg_%s.txt' % split, 'r') as ff:
        filename = list(ff)
        filename = [i.strip() for i in filename]
    if split == 'train':
        random.shuffle(filename)

    vocab = get_vocab(obj_label)

    for i in filename:  # i is filename e.g. 00000.jpg
        name = os.path.splitext(i)[0]  # 将文件名和扩展名分开
        ori_objects = all_objects[imageid_ind[int(name)]]['objects']
        classes = [x['names'] for x in ori_objects]
        classes = np.unique(classes)
        classes = [vocab.word_to_id(xx) for xx in classes if xx in obj_label]

        ori_rel = all_rel[imageid_ind[int(name)]]['relationships']
        rel = [(cc['object']['name'], cc['predicate'], cc['subject']['name']) for cc in ori_rel]
        rel = list(set(rel))
        relationships = []
        objects = []
        for xx in rel:
            # print(xx)
            if xx[0] in obj_label and xx[1] in pred_label and xx[2] in obj_label:
                relationships.append(xx[1].lower())
                objects.append(vocab.word_to_id(xx[0]))
                objects.append(vocab.word_to_id(xx[2]))
                # rel = [xx[1].lower() for xx in rel if xx[0] in obj_label and xx[1] in pred_label and xx[2] in obj_label]
        relationships = [pred_to_ind[x] for x in relationships]

        image_path = FLAGS.image_path + '/' + i

        # use tf.gfile.GFile                                              
        with tf.gfile.FastGFile(image_path,
                                'rb') as g:  # 实现对图片的读取 image_jpg = tf.gfile.FastGFile('dog.jpg','rb').read()
            image = g.read()  # 图片解码 (‘r’:UTF-8编码; ‘rb’:非UTF-8编码)
        context = tf.train.Features(feature={
            'image/name': _bytes_feature(i),
            'image/data': _bytes_feature(image),
        })
        feature_lists = tf.train.FeatureLists(feature_list={
            'objects': _int64_feature_list(objects),
            'relationships': _int64_feature_list(relationships),
            'classes': _int64_feature_list(classes)
        })
        sequence_example = tf.train.SequenceExample(
            context=context, feature_lists=feature_lists)
        # 创建Example对象，并且将Feature（img_raw，label) 一一对应填充进去。并保存到writer中

        yield sequence_example.SerializeToString()

        # yield 是一个类似 return 的关键字，只是这个函数返回的是个生成器
        # 当你调用这个函数的时候，函数内部的代码并不立马执行 ，这个函数只是返回一个生成器对象
        # 当你使用for进行迭代的时候，函数中的代码才会执行


def gen_tfrec(split):
    ds = tf.data.Dataset.from_generator(partial(image_generator, split=split),
                                        output_types=tf.string, output_shapes=())
    tfrec = tf.data.experimental.TFRecordWriter(
        '/home/xzheng10/workspace/unsupervised_captioning/data/GT_image_%s.tfrec' % split)
    tfrec.write(ds)


def main(_):
    for i in ['train']:
        gen_tfrec(i)


if __name__ == '__main__':
    app.run(main)

