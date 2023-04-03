import tensorflow as tf
from misc_fn import controlled_shuffle
from misc_fn import random_drop
import functools

FLAGS = tf.flags.FLAGS

AUTOTUNE = tf.data.experimental.AUTOTUNE


def batching_func(x, batch_size):
  """Forms a batch with dynamic padding."""
  return x.padded_batch(
    batch_size,
    padded_shapes=((
                     tf.TensorShape([299, 299, 3]),
                     tf.TensorShape([None]),
                     tf.TensorShape([]),
                     tf.TensorShape([None]),
                     tf.TensorShape([]),
                     tf.TensorShape([None])),
                   (tf.TensorShape([None]),
                    tf.TensorShape([]),
                    tf.TensorShape([None]),
                    tf.TensorShape([]))),
    drop_remainder=True)


def preprocess_image(encoded_image, classes, relationships, objects):
  """Decodes an image."""
  image = tf.image.decode_jpeg(encoded_image, 3)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize_images(image, [346, 346])
  image = tf.random_crop(image, [299, 299, 3])
  image = image * 2 - 1
  return image, classes, tf.shape(relationships)[0], relationships, tf.shape(relationships)[0], objects


def parse_image(serialized):
  """Parses a tensorflow.SequenceExample into an image and detected objects.

  Args:
    serialized: A scalar string Tensor; a single serialized SequenceExample.

  Returns:
    encoded_image: A scalar string Tensor containing a JPEG encoded image.
    classes: A 1-D int64 Tensor containing the detected objects.
    scores: A 1-D float32 Tensor containing the detection scores.
  """
  context, sequence = tf.parse_single_sequence_example(
    serialized,
    context_features={
      'image/data': tf.FixedLenFeature([], dtype=tf.string)
    },
    sequence_features={
      'objects': tf.FixedLenSequenceFeature([], dtype=tf.int64),
      'relationships': tf.FixedLenSequenceFeature([], dtype=tf.int64),
      'classes': tf.FixedLenSequenceFeature([], dtype=tf.int64)
    })

  encoded_image = context['image/data']
  objects = tf.to_int32(sequence['objects'])
  relationships = tf.to_int32(sequence['relationships'])
  classes = tf.to_int32(sequence['classes'])
  return encoded_image, classes, relationships, objects


def parse_sentence(serialized):
  """Parses a tensorflow.SequenceExample into an caption.

  Args:
    serialized: A scalar string Tensor; a single serialized SequenceExample.

  Returns:
    key: The keywords in a sentence.
    num_key: The number of keywords.
    sentence: A description.
    sentence_length: The length of the description.
  """
  context, sequence = tf.parse_single_sequence_example(
    serialized,
    context_features={},
    sequence_features={
      'sentence': tf.FixedLenSequenceFeature([], dtype=tf.int64),
    })
  sentence = tf.to_int32(sequence['sentence'])
  key = controlled_shuffle(sentence[1:-1])
  key = random_drop(key)
  key = tf.concat([key, [FLAGS.end_id]], axis=0)
  return key, tf.shape(key)[0], sentence, tf.shape(sentence)[0]


def input_fn(batch_size):
  """Input function."""
  image_ds = tf.data.TFRecordDataset('data/GT_image_full_train.tfrec')
  image_ds = image_ds.map(parse_image, num_parallel_calls=AUTOTUNE)
  image_ds = image_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
  image_ds = image_ds.shuffle(8192).repeat()

  sentence_ds = tf.data.TFRecordDataset('data/sentence.tfrec')
  sentence_ds = sentence_ds.map(parse_sentence, num_parallel_calls=AUTOTUNE)
  sentence_ds = sentence_ds.shuffle(65536).repeat()

  dataset = tf.data.Dataset.zip((image_ds, sentence_ds))

  dataset = batching_func(dataset, batch_size)
  dataset = dataset.prefetch(AUTOTUNE)
  iterator = dataset.make_one_shot_iterator()
  image, sentence = iterator.get_next()
  im, classes, num_class, relationships, num_rl, objects = image       #image, classes, objects,relationships,tf.shape(classes)[0]
  key, lk, sentence, ls = sentence
  return {'im': im, 'classes': classes, 'num_class': num_class, 'relationships': relationships, 'num_rl': num_rl, 'objects': objects, 
          'key': key, 'lk': lk}, {'sentence': sentence, 'len': ls}

