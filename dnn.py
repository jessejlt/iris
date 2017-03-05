import numpy as np
from tensorflow.contrib.learn import DNNClassifier
from tensorflow import logging
from tensorflow.contrib.learn import MetricSpec, PredictionKey, RunConfig, datasets, monitors
from tensorflow.contrib.metrics import streaming_accuracy, streaming_precision, streaming_recall
from tensorflow.contrib.layers import real_valued_column

# Enable logging
logging.set_verbosity(logging.INFO)

# Adaptation of https://www.tensorflow.org/get_started/tflearn

# Both our training and test datasets contain headers for the types of iris.
# Targets are boolean 0 or 1, so our target types are just np.int, and
# the features are measurements as floats, so our target type of np.float32.

# Load both our training and test datasets
training_set = datasets.base.load_csv_with_header(
    filename='iris_training.csv',
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = datasets.base.load_csv_with_header(
    filename='iris_test.csv', target_dtype=np.int, features_dtype=np.float32)

# Let TF know how many of the columns are to be used for features
feature_cols = [real_valued_column("", dimension=4)]

# Periodically evaluate our accuracy. This is useful for visualizing
# our model's performance with tensorboard.
validation_metrics = {
    "accuracy":
    MetricSpec(
        metric_fn=streaming_accuracy, prediction_key=PredictionKey.CLASSES),
    "precision":
    MetricSpec(
        metric_fn=streaming_precision, prediction_key=PredictionKey.CLASSES),
    "recall":
    MetricSpec(
        metric_fn=streaming_recall, prediction_key=PredictionKey.CLASSES)
}
validation_monitor = monitors.ValidationMonitor(
    test_set.data,
    test_set.target,
    every_n_steps=50,
    metrics=validation_metrics)

# Build our model.
# n_classes=3 means we have a 3 layer deep neural net with
# 10, 20, and 10 neurons per layer. Why 10, 20, 10? It's just something
# to start with. We can tune these as hyperparmeters.
#
# Our config option is for periodically saving monitoring information for
# later use in tensorboard.
classifier = DNNClassifier(
    feature_columns=feature_cols,
    hidden_units=[10, 20, 10],
    n_classes=3,
    model_dir='/tmp/iris_model',
    config=RunConfig(save_checkpoints_secs=1))

# TF will perform gradient descent on our model over 2k epochs.
classifier.fit(
    x=training_set.data,
    y=training_set.target,
    steps=2000,
    monitors=[validation_monitor])

# Evaluate the accuracy of our DNN.
accuracy_score = classifier.evaluate(
    x=test_set.data, y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))
