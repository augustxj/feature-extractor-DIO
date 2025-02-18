import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pickle
import time
import tensorflow as tf
from tqdm import tqdm
from src.utils.config import IMAGES_PATH, EMBEDDINGS_FILE, FILENAMES_FILE
from src.models.ResNet50 import load_model
from src.utils.image_processing import extract_features
from src.utils.recommender import recommend_similar_images
from src.utils.header import print_header
from tkinter.filedialog import askopenfilename

if __name__ == "__main__":
    print_header()
    time.sleep(1)
    print("\n")
    print("Starting feature extraction...")
    # Verifica se há GPUs disponíveis e imprime qual está sendo usada
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Ativa o uso da memória sob demanda
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"✅ TensorFlow is running on {len(gpus)} GPU(s): {logical_gpus}")
        except RuntimeError as e:
            print(f"❌ Error with GPU: {e}")
    else:
        print("⚠️ No GPU detected. Code will run on CPU.")

    # Carregar o modelo
    model = load_model()

    # Obter lista de imagens
    filenames = [os.path.join(IMAGES_PATH, file) for file in os.listdir(IMAGES_PATH) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Extrair características e armazenar
    feature_list = [extract_features(file, model) for file in tqdm(filenames)]
    
    # Salvar embeddings e filenames na pasta data
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, 'embeddings.pkl'), 'wb') as f:
        pickle.dump(feature_list, f)
    with open(os.path.join(data_dir, 'filenames.pkl'), 'wb') as f:
        pickle.dump(filenames, f)
    print(f"Feature extraction done successfully! All files saved in: {os.path.abspath(os.path.join(data_dir, 'embeddings.pkl'))}")
    
    start_time = time.time()
    while True:
        # Perguntar se o usuário deseja obter recomendações
        proceed = input("Would you like to get recommendations right now? [Y/N] ")
        current_time = time.time()

        if proceed.lower() == "y":
            query_img_path = askopenfilename()
            recommend_similar_images(query_img_path, model)
            break

        elif proceed.lower() == "n" or current_time - start_time > 20:
            print("Goodbye!")
            time.sleep(1)
            exit()
        else:
            input("please answer with 'Y' or 'N':")

