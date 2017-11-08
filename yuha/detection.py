import numpy as np
import tensorflow as tf
from PIL import Image

from util import CKPT_PATH


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(CKPT_PATH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph)


def sort_by_score(boxes, classes, scores):
    index = np.argsort(scores[::-1])[:1]
    if scores[index] < 0.7 and classes[index] != 1:
        return False, False, False
    return boxes[index], classes[index], scores[index]


def load_image_into_numpy_array(image, channel='RGB'):
    im_width, im_height = image.size
    num_channel = 3 if channel == 'RGB' else 1
    return np.array(image.getdata()).reshape((im_height, im_width, num_channel)).astype(np.uint8)


def detect_objects(image_np, sess_, detection_graph_):
    image = Image.fromarray(image_np)
    im_width, im_height = image.size
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph_.get_tensor_by_name('image_tensor:0')

    boxes = detection_graph_.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph_.get_tensor_by_name('detection_scores:0')
    classes = detection_graph_.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph_.get_tensor_by_name('num_detections:0')

    (boxes, scores, classes, num_detections) = sess_.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded}
    )

    boxes, classes, scores = sort_by_score(np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores))
    crop_box = (boxes[0][1] * im_width, boxes[0][0] * im_height, boxes[0][3] * im_width, boxes[0][2] * im_height)
    # TODO add execption
    image = image.crop(crop_box)

    return image


def detect(image):
    image = Image.open(image)
    image_np = load_image_into_numpy_array(image)
    image = detect_objects(image_np, sess, detection_graph)
    return image
