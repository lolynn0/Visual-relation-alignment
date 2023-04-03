"""Evaluates the performance of all the checkpoints on validation set."""
import glob
import json
import multiprocessing
import os
import sys

os.environ['NO_PROXY'] = 'http://visualgenome.org/api/v0/images'

from absl import app
from absl import flags

from config import COCO_PATH
from config import VG_PATH

sys.path.insert(0, VG_PATH)
from visual_genome import api

flags.DEFINE_integer('threads', 2, 'num of threads')

from caption_infer import Infer


# sys.path.insert(0, COCO_PATH)
# from pycocotools.coco import COCO
# from pycocoevalcap.eval import COCOEvalCap

FLAGS = flags.FLAGS


def initializer():
    """Decides which GPU is assigned to a worker.
  If your GPU memory is large enough, you may put several workers in one GPU.
  """
    global tf, no_gpu
    no_gpu = False
    devices = os.getenv('CUDA_VISIBLE_DEVICES')
    if devices is None:
        print('Please set CUDA_VISIBLE_DEVICES')
        no_gpu = True
        return
    devices = devices.split(',')
    if len(devices) == 0:
        print('You should assign some gpus to this program.')
        no_gpu = True
        return
    current = multiprocessing.current_process()
    id = (current._identity[0] - 1) % len(devices)
    os.environ['CUDA_VISIBLE_DEVICES'] = devices[id]
    import tensorflow as tf


def run(inp):
    if no_gpu:
        return
    out = FLAGS.job_dir + '/val_%s.json' % inp
    if not os.path.exists(out):
        # with open(COCO_PATH + '/annotations/captions_val2014.json') as g:
        #     caption_data = json.load(g)
        #     name_to_id = [(x['file_name'], x['id']) for x in caption_data['images']]
        #     name_to_id = dict(name_to_id)

        ret = []
        with tf.Graph().as_default():
            infer = Infer(job_dir='%s/model.ckpt-%s' % (FLAGS.job_dir, inp))
            with open('data/vg_test.txt', 'r') as g:
              for name in g:
                vg_name = name.strip()
                id = os.path.splitext(vg_name)[0]
                kk = api.get_image_data(id)
                sentences = infer.infer(vg_name)
                cur = {}
                cur['image_id'] = kk.coco_id
                cur['caption'] = sentences[0][0]
                ret.append(cur)
        with open(out, 'w') as g:
            json.dump(ret, g)

#     coco = COCO(COCO_PATH + '/annotations/captions_train2014.json')
#     cocoRes = coco.loadRes(out)
#     # create cocoEval object by taking coco and cocoRes
#     cocoEval = COCOEvalCap(coco, cocoRes)
#     # evaluate on a subset of images by setting
#     # cocoEval.params['image_id'] = cocoRes.getImgIds()
#     # please remove this line when evaluating the full validation set
#     cocoEval.params['image_id'] = cocoRes.getImgIds()
#     # evaluate results
#     cocoEval.evaluate()
#     return (inp, cocoEval.eval['CIDEr'], cocoEval.eval['METEOR'],
#             cocoEval.eval['Bleu_4'], cocoEval.eval['Bleu_3'],
#             cocoEval.eval['Bleu_2'])
    return inp


def main(_):
    results = glob.glob(FLAGS.job_dir + '/model.ckpt-*')
    results = [os.path.splitext(i)[0] for i in results]
    results = set(results)
    gs_list = [i.split('-')[-1] for i in results]
#    gs_list = [3000,2000]

    pool = multiprocessing.Pool(FLAGS.threads, initializer)
    ret = pool.map(run, gs_list)
    pool.close()
    pool.join()
    if not ret or ret[0] is None:
        return

#     ret = sorted(ret, key=lambda x: x[1])
#     with open(FLAGS.job_dir + '/cider.json', 'w') as f:
#         json.dump(ret, f)
#     ret = sorted(ret, key=lambda x: x[2])
#     with open(FLAGS.job_dir + '/meteor.json', 'w') as f:
#         json.dump(ret, f)
#     ret = sorted(ret, key=lambda x: x[3])
#     with open(FLAGS.job_dir + '/b4.json', 'w') as f:
#         json.dump(ret, f)
#     ret = sorted(ret, key=lambda x: x[4])
#     with open(FLAGS.job_dir + '/b3.json', 'w') as f:
#         json.dump(ret, f)
#     ret = sorted(ret, key=lambda x: x[5])
#     with open(FLAGS.job_dir + '/b2.json', 'w') as f:
#         json.dump(ret, f)
#     ret = sorted(ret, key=lambda x: x[3] + x[4])
#     with open(FLAGS.job_dir + '/b34.json', 'w') as f:
#         json.dump(ret, f)


if __name__ == '__main__':
    app.run(main)
