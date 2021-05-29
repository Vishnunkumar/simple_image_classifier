# simple-image-classification
Simple Image classification using transfer learning 

```
# loading the package
from simple_image_classifier import tr_learner_classification
import tensorflow as tf

# defining metrics and model weights
met = tf.keras.metrics.Accuracy()
los = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# please look into tf_hub for more details
model_weights = 'https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/classification/4'

# preprocessing and augmentation
preprocess = tr_learner_classification.Preprocessing(224, 224, 8, 'train/', 'valid/')
train_generator, valid_generator = tr_learner_classification.Preprocessing.generator(preprocess)

# model compilation
image_class_model = tr_learner_classification.Model(train_generator, valid_generator, met, los, model_weights, 224, 224)

# params for compiler: compiled model, activation function, penultimate dense layer nodes, dropout value
tl_model = tr_learner_classification.Model.compiler(image_class_model, 'sigmoid', 1280, 0.5)
 

# model training
img_model, img_hist = tr_learner_classification.Model.train(image_class_model, 80, tl_model)

# prediction
predictor = tr_learner_classification.Predictions(img_path, img_model, 224, 224)
predictions = predictor.predict()

```

- Please note: the library is under development, prediction module will be updated sooner
