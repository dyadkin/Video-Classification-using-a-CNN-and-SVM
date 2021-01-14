# from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3  # , preprocess_input
from tensorflow.keras.models import Model  # , load_model
from processor import preprocess_multires

from models import ResearchModels

# import numpy as np


class Extractor:
    def __init__(self, weights=None):
        """Either load pretrained from imagenet, or load our saved
        weights from our own training."""

        self.weights = weights  # so we can check elsewhere which model

        if weights is None:
            # Get model with pretrained weights.
            base_model = InceptionV3(weights="imagenet", include_top=True)

            # We'll extract features at the final pool layer.
            self.model = Model(
                inputs=base_model.input,
                outputs=base_model.get_layer("avg_pool").output
            )
        else:
            print(self.weights)
            # Load the model first.
            base_model = ResearchModels(101, "sf_multires", "sgd", 40, weights).model
            # base_model = load_model(weights)

            # Then remove the top so we get features not predictions.
            # From: https://github.com/fchollet/keras/issues/2371
            # self.model.layers.pop()
            # self.model.layers.pop()
            # self.model.layers.pop()
            # self.model.layers.pop()
            # self.model.layers.pop()  # Five pops to get to flatten layer
            outputs = [base_model.layers[-5].output]
            print(outputs)
            self.model = Model(inputs=base_model.input, outputs=outputs)
            print("Modified model for feature extraction:\n",
                  self.model.summary())
            # self.model.output_layers = [self.model.layers[-1]]
            # self.model.layers[-1].outbound_nodes = []

    def extract(self, image_path):
        # img = image.load_img(image_path, target_size=(299, 299))
        # x = image.img_to_array(img)
        # x = np.expand_dims(x, axis=0)
        # x = preprocess_input(x)

        inputs = preprocess_multires(image_path, (89, 89))

        # Get the prediction.
        features = self.model.predict(inputs)

        if self.weights is None:
            # For imagenet/default network:
            features = features[0]
        else:
            # For loaded network:
            features = features[0]

        return features
