import tensorflow as tf
import numpy as np
import cleverhans
from cleverhans.model import Model
from cleverhans.attacks import *
from utils import load_model, read_tensor_from_image_file
import base64
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc


class InceptionCNNModel(Model):
    model_path = './models/output_graph_2.pb'

    def __init__(self):
        super(InceptionCNNModel, self).__init__()

        # Load trained model
        load_model(self.model_path)
        # Save input and output tensors references
        graph = tf.get_default_graph()
        self.input_tensor = graph.get_tensor_by_name('Mul:0')
        self.output_tensor = graph.get_tensor_by_name('final_result:0')

    def convert_to_classifier(self):
        # Save softmax layer
        self.layer_names = []
        self.layers = []
        self.layers.append(self.output_tensor)
        self.layer_names.append('probs')

    def fprop(self, x, set_ref=False):
        return dict(zip(self.layer_names, self.layers))


with tf.Graph().as_default():
    with tf.Session() as sess:
        # Load model
        model = InceptionCNNModel()
        model.convert_to_classifier()

        # Load faces
        origin_image = read_tensor_from_image_file('./data/yyy.jpg')
        # with open("./data/kotori.jpg", "rb") as image_file:
        #     origin_image = base64.encodebytes(image_file.read())

        # Define target label
        array = np.zeros((1, 100))
        array[0][24] = 1.
        y_target = tf.convert_to_tensor(array, np.float32)

        # Craft adversarial examples
        steps = 5
        eps = 0.1
        alpha = eps / steps
        fgsm = FastGradientMethod(model, back='tf', sess=sess)
        fgsm_params = {'eps': alpha,
                       'y_target': y_target,
                       'clip_min': 0.,
                       'clip_max': 1.}
        adv_x = fgsm.generate(model.input_tensor, **fgsm_params)

        adv = origin_image
        for i in range(steps):
            print("FGSM step " + str(i + 1))
            adv = sess.run(adv_x, feed_dict={model.input_tensor: adv})


        # # Craft adversarial examples
        # bis_params = {
        #     'eps': 0.01,
        #     'eps_iter': 3,
        #     'nb_iter': 10
        # }
        # bis = BasicIterativeMethod(model, back='tf', sess=sess)
        # adv_x = bis.generate(model.input_tensor, **bis_params)
        # adv = sess.run(adv_x, feed_dict={model.input_tensor: origin_image})

        def resize_and_to_int(image_array):
            resize = image_array.reshape((299, 299, 3)) * 255
            int_img = resize.astype(np.int)
            return int_img


        scipy.misc.imsave('./data/outfile.jpg', resize_and_to_int(adv))
        scipy.misc.imsave('./data/infile.jpg', resize_and_to_int(origin_image))

        label_lines = [line.rstrip() for line
                       in tf.gfile.GFile("./models/output_labels_2.txt")]
        predictions = sess.run(model.output_tensor, feed_dict={model.input_tensor: adv})
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        results = []
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            results.append((human_string, score))

        print(results)
