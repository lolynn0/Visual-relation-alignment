import json

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
import json
with open('data_release/objects.json', 'r') as f:
    all_objects = json.load(f)

import json
with open('data_release/relationships.json', 'r') as f:
    all_rel = json.load(f)

import json
with open('data_release/VG-SGG-dicts.json', 'r') as f:
    vg_dict = json.load(f)
    obj_label = list(vg_dict['object_count'].keys())
    pred_label = list(vg_dict['predicate_count'].keys())

pred_to_ind = load_info('data_release/VG-SGG-dicts')


def image_generator(split):
  with open('data/vg_%s.txt' % split, 'r') as ff:
    filename = list(ff)
    filename = [i.strip() for i in filename]
  if split == 'train':
    random.shuffle(filename)

  #load classes words
  # object_classes = list(classes['object_classes'])
  # object_classes[0] = 'background'
  # predicate_classes = list(classes['predicate_classes'])
  # category_name = np.unique([y for x in predicate_classes for y in x.split(' ')])
  vocab = get_vocab(obj_label)

  for i in filename:  # i is filename e.g. 00000.jpg
    name = os.path.splitext(i)[0]   #将文件名和扩展名分开
    ori_objects = all_objects[int(name)-1]['objects']
    classes = [x['names'] for x in ori_objects]
    classes = np.unique(classes)
    classes = [vocab.word_to_id(xx) for xx in classes if xx in obj_label].astype(np.int32)

    ori_rel = all_rel[int(name)-1]['relationships']
    rel = [(x['object']['name'], x['predicate'], x['subject']['name']) for x in ori_rel]
    rel = np.unique(rel)
    relationships = []
    obejcts = []
    for xx in rel:
        if xx[0] in obj_label and xx[1] in pred_label and xx[2] in obj_label:
            relationships.append(xx[1].lower())
            obejcts.append(vocab.word_to_id(xx[0]),vocab.word_to_id(xx[2]))
    # rel = [xx[1].lower() for xx in rel if xx[0] in obj_label and xx[1] in pred_label and xx[2] in obj_label]
    relationships = [pred_to_ind[x] for x in relationships]

    image_path = FLAGS.image_path + '/' + i

    # use tf.gfile.GFile
    with tf.gfile.FastGFile(image_path, 'rb') as g:  # 实现对图片的读取 image_jpg = tf.gfile.FastGFile('dog.jpg','rb').read()
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
