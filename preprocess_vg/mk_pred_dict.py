import os
import sys
import pickle as pkl
import json

import h5py
import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from functools import partial



sys.path.insert(0, '/Users/macbook/PycharmProjects/Tencent_unsupervised/'+'/tf_models' + '/research/im2txt/im2txt')
sys.path.append('/Users/macbook/PycharmProjects/Tencent_unsupervised/'+'/tf_models' + '/research')
sys.path.append('/Users/macbook/PycharmProjects/Tencent_unsupervised/'+'/tf_models'+ '/research/object_detection')
from inference_utils import vocabulary
# from utils import label_map_util

os.chdir('/Users/macbook/PycharmProjects/Tencent_unsupervised')





tf.enable_eager_execution()

# tf.flags.DEFINE_string("word_counts_output_file", "data/word_counts.txt",
#                        "Text file containing the vocabulary.")

FLAGS = flags.FLAGS


def get_vocab(category_name):
  with open('data/glove_vocab.pkl', 'rb') as f:  ## 全部词向量矩阵 glove.word_vectors
    glove = pkl.load(f,encoding ='bytes')
    glove.append('<S>')
    glove.append('</S>')
    glove = set(glove)
  with open('data/word_counts.txt', 'r') as ff:
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

  vocab = vocabulary.Vocabulary('data/word_counts.txt')

  return vocab

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
    return ind_to_predicates

ind_to_pred = load_info('data_release/VG-SGG-dicts.json')

with open('data_release/VG-SGG-dicts.json', 'r') as f:
    vg_dict = json.load(f)
    # obj_label = list(vg_dict['object_count'].keys())
    pred_label = list(vg_dict['predicate_count'].keys())
category_name = np.unique([y for x in pred_label for y in x.split(' ')])
vocab = get_vocab(category_name)
i=0
pred_vocab = []
for kk in range(51):
    x = ind_to_pred[kk]
    pred = ''.join(x).split(' ')   #['on','front','of'] ['', '', 'background', '', '']
    pred = [vocab.word_to_id(jj) for jj in pred]
    if len(pred) == 1:
        pred.extend([99999,99999])
    elif len(pred) == 2:
        pred.extend([99999])
    pred_vocab.append(pred)
    i += 1

pred_vocab[0] = [11,99999,99999]
np.save('pred_vocab.npy',np.array(pred_vocab))

# import numpy as np
# import os
# os.chdir('/Users/macbook/PycharmProjects/Tencent_unsupervised')
# pred = np.load('pred_vocab.npy',allow_pickle=True)

# import numpy as np
# import os
# os.chdir('/Users/macbook/PycharmProjects/Tencent_unsupervised')
# pred = np.load('data/anno_vg.npy',allow_pickle=True).item()
# # with open('pred_vocab_2.json', 'w') as g:
# #     json.dump(pred_vocab, g)
# # #
# #
# # import json
# #
# # with open('pred_vocab_2.json') as g:  # COCO_PATH is from git
# #     predicate_vocab = json.load(g)
#
