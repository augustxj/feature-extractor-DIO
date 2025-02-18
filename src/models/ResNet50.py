import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50 # type: ignore
from tensorflow.keras.layers import GlobalMaxPooling2D # type: ignore
from tensorflow.keras.models import Sequential # type: ignore

def load_model():
    print(f"🔍 TensorFlow runnning on: {tf.config.experimental.list_logical_devices()}")
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = Sequential([base_model, GlobalMaxPooling2D()])
    return model
