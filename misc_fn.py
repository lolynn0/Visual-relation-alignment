import tensorflow as tf
import numpy as np
import json


def get_len(sequence, end):
    """Gets the length of a generated caption.

    Args:
      sequence: A tensor of size [batch, max_length].
      end: The <EOS> token.

    Returns:
      length: The length of each caption.
    """

    def body(x):
        idx = tf.to_int32(tf.where(tf.equal(x, end)))
        idx = tf.cond(tf.shape(idx)[0] > 0, lambda: idx[0] + 1, lambda: tf.shape(x))
        return idx[0]

    length = tf.map_fn(body, sequence, tf.int32)
    return length


def variable_summaries(var, mask, name):
    """Attaches a lot of summaries to a Tensor.

    Args:
      var: A tensor to summary.
      mask: The mask indicating the valid elements in var.
      name: The name of the tensor in summary.
    """
    var = tf.boolean_mask(var, mask)
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean/' + name, mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.summary.scalar('sttdev/' + name, stddev)
    tf.summary.scalar('max/' + name, tf.reduce_max(var))
    tf.summary.scalar('min/' + name, tf.reduce_min(var))
    tf.summary.histogram(name, var)


def transform_grads_fn(grads):
    """Gradient clip."""
    grads, vars = zip(*grads)
    grads, _ = tf.clip_by_global_norm(grads, 10)
    return list(zip(grads, vars))


def crop_sentence(sentence, end):
    """Sentence cropping for logging. Remove the tokens after <EOS>."""
    idx = tf.to_int32(tf.where(tf.equal(sentence, end)))
    idx = tf.cond(tf.shape(idx)[0] > 0, lambda: idx[0] + 1,
                  lambda: tf.shape(sentence))
    sentence = sentence[:idx[0]]
    return sentence


def validate_batch_size_for_multi_gpu(batch_size):
    """For multi-gpu, batch-size must be a multiple of the number of GPUs.

    Note that this should eventually be handled by replicate_model_fn
    directly. Multi-GPU support is currently experimental, however,
    so doing the work here until that feature is in place.

    Args:
      batch_size: the number of examples processed in each training batch.

    Raises:
      ValueError: if no GPUs are found, or selected batch_size is invalid.
    """
    from tensorflow.python.client import \
        device_lib  # pylint: disable=g-import-not-at-top

    local_device_protos = device_lib.list_local_devices()
    num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
    if not num_gpus:
        raise ValueError('Multi-GPU mode was specified, but no GPUs '
                         'were found. To use CPU, run without --multi_gpu.')

    remainder = batch_size % num_gpus
    if remainder:
        err = ('When running with multiple GPUs, batch size '
               'must be a multiple of the number of available GPUs. '
               'Found {} GPUs with a batch size of {}; try --batch_size={} instead.'
               ).format(num_gpus, batch_size, batch_size - remainder)
        raise ValueError(err)


def find_obj(sentence, s_mask, classes, num):
  """Computes the object reward for one sentence."""
  shape = tf.shape(sentence)
  sentence = tf.boolean_mask(sentence, s_mask)

  def body(x):
    idx = tf.to_int32(tf.where(tf.equal(sentence, x)))
    idx = tf.cond(tf.shape(idx)[0] > 0, lambda: idx[0, 0],
                  lambda: tf.constant(999, tf.int32))
    return idx

  classes = classes[:num]
  scores = tf.ones_like(classes, dtype=tf.float32)
  ind = tf.map_fn(body, classes, tf.int32)
  mask = tf.not_equal(ind, 999)
  miss, detected = tf.dynamic_partition(scores, tf.to_int32(mask), 2)
  ind = tf.boolean_mask(ind, mask)
  ret = tf.scatter_nd(tf.expand_dims(ind, 1), detected, shape)
  return ret


def obj_rewards(sentence, mask, classes, num):
  """Computes the object reward.
  Args:
    sentence: A tensor of size [batch, max_length].
    mask: The mask indicating the valid elements in sentence.
    classes: [batch, padded_size] int32 tensor of detected objects.
    scores: [batch, padded_size] float32 tensor of detection scores.
    num: [batch] int32 tensor of number of detections.
  Returns:
    rewards: [batch, max_length] float32 tensor of rewards.
  """

  def body(x):
    ret = find_obj(x[0], x[1], x[2], x[3])
    return ret

  rewards = tf.map_fn(body, [sentence, mask, classes, num], tf.float32)
  return rewards


def find_relation1(sentence, s_mask, classes, num_class, relationships, num_rl):
    """Computes the object reward and predicate reward for one sentence."""

    shape = tf.shape(sentence)
    sentence = tf.boolean_mask(sentence, s_mask)

    def body_o(x):
        idx = tf.to_int32(tf.where(tf.equal(sentence, x)))  # [[1] [3]]
        idx = tf.cond(tf.shape(idx)[0] > 0, lambda: idx[0, 0],
                      lambda: tf.constant(999, tf.int32))
        return idx

    def cond(tensor1, tensor2):
        bool_tensor = tf.equal(tensor1, tensor2)  # 返回bool类型的tensor，如([True, false],[False, True])
        result = tf.reduce_all(bool_tensor)  # 默认keep_dim = False,若bool_tensor中全为True则返回tensor类型的True(与普通的True相区别)
        return result

    def body_r(y):
        y = tf.gather(pred_vocab, y)
        mask = tf.not_equal(y, 99999)
        y = tf.boolean_mask(y, mask)
        idy = tf.constant(999, tf.int32)
        i = tf.constant(0, tf.int32)
        n = tf.shape(sentence)[0] - tf.shape(y)[0] + 1

        def cond1(i, n, idy):
            return i < n

        def body(i, n, idy):
            eq = cond(sentence[i:i + tf.shape(y)[0]], y)
            idy = tf.cond(tf.equal(eq, True), lambda: tf.convert_to_tensor(i, dtype=tf.int32), lambda: idy)
            i = tf.cond(tf.equal(eq, True), lambda: n, lambda: i+1)
            return i, n, idy

        i, n, idy = tf.while_loop(cond1, body, [i, n, idy])
        # def body(ii):
        #   eq = cond(sentence[ii:ii + tf.shape(y)[0]], y)
        #   return eq

        # i = tf.range(tf.shape(sentence)[0] - tf.shape(y)[0] + 1)
        # idy = tf.map_fn(body, i, dtype = tf.bool)
        # idy = tf.to_int32(tf.where(tf.equal(idy, True)))
        # idy = tf.cond(tf.shape(idy)[0] > 0, lambda: idy[0,0],
        #               lambda: tf.constant(999, tf.int32))

        return idy

    objects = classes[:num_class]
    relationships = relationships[: num_rl]
    ind_obj = tf.map_fn(body_o, objects, tf.int32)
    ind_pred = tf.map_fn(body_r, relationships, tf.int32)
    ind = tf.concat([ind_obj, ind_pred], axis=0)
#     ind = ind_pred
    scores = tf.ones_like(ind, dtype=tf.float32)
    mask = tf.not_equal(ind, 999)
    miss, detected = tf.dynamic_partition(scores, tf.to_int32(mask), 2)
    ind = tf.boolean_mask(ind, mask)
    ret = tf.scatter_nd(tf.expand_dims(ind, 1), detected, shape)
    return ret


def relation_rewards1(sentence, mask, classes, num_class, relationships, num_rl):
    """Computes the object reward.

    Args:
      sentence: A tensor of size [batch, max_length].
      mask: The mask indicating the valid elements in sentence.
      classes: [batch, padded_size] int32 tensor of detected objects.
      objects: [batch,padded_size] int32 tensor of grouped objects.
      relationships: [batch,padded_size] int32 tensor of predicates
      # scores: [batch, padded_size] float32 tensor of detection scores.
      num: [batch] int32 tensor of number of detections.

    Returns:
      rewards: [batch, max_length] float32 tensor of rewards.
    """

    def body(x):
        ret = find_relation(x[0], x[1], x[2], x[3], x[4], x[5])
        return ret

    global pred_vocab
    pred_vocab = np.load('data/pred_vocab.npy', allow_pickle=True)
    pred_vocab = tf.convert_to_tensor(pred_vocab, tf.int32)
    rewards = tf.map_fn(body, [sentence, mask, classes, num_class, relationships, num_rl], tf.float32)
    return rewards

def find_relation(sentence, s_mask, classes, num_class, relationships, num_rl, objects):
    """Computes the object reward and predicate reward for one sentence."""

    shape = tf.shape(sentence)
    sentence = tf.boolean_mask(sentence, s_mask)

    def body_o(x):
        idx = tf.to_int32(tf.where(tf.equal(sentence, x)))  # [[1] [3]]
        idx = tf.cond(tf.shape(idx)[0] > 0, lambda: idx[0, 0],
                      lambda: tf.constant(999, tf.int32))
        return idx

    def cond(tensor1, tensor2):
        bool_tensor = tf.equal(tensor1, tensor2)  # 返回bool类型的tensor，如([True, false],[False, True])
        result = tf.reduce_all(bool_tensor)  # 默认keep_dim = False,若bool_tensor中全为True则返回tensor类型的True(与普通的True相区别)
        return result

    def body_r(y):
        y = tf.gather(pred_vocab, y)
        mask = tf.not_equal(y, 99999)
        y = tf.boolean_mask(y, mask)
        idy = tf.constant(999, tf.int32)
        i = tf.constant(0, tf.int32)
        n = tf.shape(sentence)[0] - tf.shape(y)[0] + 1

        def cond1(i, n, idy):
            return i < n

        def body(i, n, idy):
            eq = cond(sentence[i:i + tf.shape(y)[0]], y)
            idy = tf.cond(tf.equal(eq, True), lambda: tf.convert_to_tensor(i, dtype=tf.int32), lambda: idy)
            i = tf.cond(tf.equal(eq, True), lambda: n, lambda: i+1)
            return i, n, idy

        i, n, idy = tf.while_loop(cond1, body, [i, n, idy])

        return idy

#     def mix_body(x):
#         obj, pred = x
#         rel = tf.cond(obj[0] < pred and pred < obj[1], lambda: pred, lambda: 999)
#         rel = tf.cond(rel == 999 or obj[0] == 999 or pred == 999 or obj[1] == 999, lambda: 999, lambda: pred)
#         return rel

    def mix_body(x):
        obj, pred = x
    
        iff = tf.less([obj[0],pred], [pred,obj[1]])
        iff = tf.reduce_all(iff)
        rel = tf.cond(tf.equal(iff, True), lambda: pred, lambda: tf.constant(999, tf.int32))
    
        iff1 = tf.less([rel,obj[0],pred,obj[1]], tf.constant(999, tf.int32))
        iff1 = tf.reduce_all(iff1)
        rel = tf.cond(tf.equal(iff1, True), lambda: pred, lambda: tf.constant(999, tf.int32))
        return rel

#    objects = objects[:2*num_rl]
    relationships = relationships[:num_rl]
#    ind_obj = tf.map_fn(body_o, objects, tf.int32)
#    ind_obj = tf.reshape(ind_obj, [-1,2])
    ind_pred = tf.map_fn(body_r, relationships, tf.int32)
#    ind = tf.map_fn(mix_body, (ind_obj, ind_pred), tf.int32)
    ind = ind_pred
    scores = tf.ones_like(ind, dtype=tf.float32)
    mask = tf.not_equal(ind, 999)
    miss, detected = tf.dynamic_partition(scores, tf.to_int32(mask), 2)
    ind = tf.boolean_mask(ind, mask)
    ret = tf.scatter_nd(tf.expand_dims(ind, 1), detected, shape)
    return ret


def relation_rewards(sentence, mask, classes, num_class, relationships, num_rl, objects):
    """Computes the object reward.

    Args:
      sentence: A tensor of size [batch, max_length].
      mask: The mask indicating the valid elements in sentence.
      classes: [batch, padded_size] int32 tensor of detected objects.
      objects: [batch,padded_size] int32 tensor of grouped objects.
      relationships: [batch,padded_size] int32 tensor of predicates
      # scores: [batch, padded_size] float32 tensor of detection scores.
      num: [batch] int32 tensor of number of detections.

    Returns:
      rewards: [batch, max_length] float32 tensor of rewards.
    """

    def body(x):
        ret = find_relation(x[0], x[1], x[2], x[3], x[4], x[5], x[6])
        return ret

    global pred_vocab
    pred_vocab = np.load('data/pred_vocab.npy', allow_pickle=True)
    pred_vocab = tf.convert_to_tensor(pred_vocab, tf.int32)
    rewards = tf.map_fn(body, [sentence, mask, classes, num_class, relationships, num_rl, objects], tf.float32)
    return rewards


def random_drop(sentence):
    """Randomly drops some tokens."""
    length = tf.shape(sentence)[0]
    rnd = tf.random_uniform([length]) + 0.9
    mask = tf.cast(tf.floor(rnd), tf.bool)
    sentence = tf.boolean_mask(sentence, mask)
    return sentence


def controlled_shuffle(sentence, d=3.0):
    """Shuffles the sentence as described in https://arxiv.org/abs/1711.00043"""
    length = tf.shape(sentence)[0]
    rnd = tf.random_uniform([length]) * (d + 1) + tf.to_float(tf.range(length))
    _, idx = tf.nn.top_k(rnd, length)
    idx = tf.reverse(idx, axis=[0])
    sentence = tf.gather(sentence, idx)
    return sentence


def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _float_feature(value):
    """Wrapper for inserting an float Feature into a SequenceExample proto."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _float_feature_list(values):
    """Wrapper for inserting an float FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_float_feature(v) for v in values])


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))
