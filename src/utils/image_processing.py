import numpy as np
from tensorflow.keras.preprocessing import image #type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input #type: ignore
from numpy.linalg import norm

def extract_features(img_path, model):
    """Extrai características de uma imagem usando o modelo carregado."""
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()

    return result / norm(result)
