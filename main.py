import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pickle
import time
import tensorflow as tf
from colorama import Fore, Style, init
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
    print("Checking if GPU is available...")
    # Verifica se há GPUs disponíveis e imprime qual está sendo usada
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Ativa o uso da memória sob demanda
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(Fore.YELLOW + f"✅ TensorFlow is running on {len(gpus)} GPU(s): {logical_gpus}"+ Style.RESET_ALL)
        except RuntimeError as e:
            print(f"❌ Error with GPU: {e}")
    else:
        print(Fore.RED + "⚠️ No GPU detected. Code will run on CPU." + Style.RESET_ALL)

    # Verificar se os arquivos de embeddings e filenames já existem
    print("\nChecking if embeddings and filenames files already exist...")
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(FILENAMES_FILE):
        proceed_extraction = input("It seems that embeddings and filenames already exist. Do you want to"+ Fore.RED + " re-extract "+ Style.RESET_ALL + "features?" + Fore.RED + "[Y/N] " + Style.RESET_ALL)
        if proceed_extraction.lower() == "n":
            print(Fore.YELLOW + "\nSkipping to recommendation step.\n " + Style.RESET_ALL)
            # Load the model
            model = load_model()
        else:
            # Carregar o modelo
            model = load_model()

            # Obter lista de imagens
            filenames = [os.path.join(IMAGES_PATH, file) for file in os.listdir(IMAGES_PATH) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

            # Extrair características e armazenar
            feature_list = [extract_features(file, model) for file in tqdm(filenames)]
            
            # Salvar embeddings e filenames na pasta data
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
            os.makedirs(data_dir, exist_ok=True)
            with open(EMBEDDINGS_FILE, 'wb') as f:
                pickle.dump(feature_list, f)
            with open(FILENAMES_FILE, 'wb') as f:
                pickle.dump(filenames, f)
            print(f"Feature extraction done successfully! All files saved in: {os.path.abspath(EMBEDDINGS_FILE)}")
    else:
        # Carregar o modelo
        model = load_model()

        # Obter lista de imagens
        filenames = [os.path.join(IMAGES_PATH, file) for file in os.listdir(IMAGES_PATH) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # Extrair características e armazenar
        feature_list = [extract_features(file, model) for file in tqdm(filenames)]
        
        # Salvar embeddings e filenames na pasta data
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(data_dir, exist_ok=True)
        with open(EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(feature_list, f)
        with open(FILENAMES_FILE, 'wb') as f:
            pickle.dump(filenames, f)
        print(f"Feature extraction done successfully! All files saved in: {os.path.abspath(EMBEDDINGS_FILE)}")
    
    while True:
        query_img_path = askopenfilename()
        recommend_similar_images(query_img_path, model)
        proceed = input("\nWould you like to analyze another image? " + Fore.RED + "[Y/N] " + Style.RESET_ALL)

        if proceed.lower() == "n":
            print(Fore.CYAN + "\nGoodbye! \n" + Style.RESET_ALL)
            time.sleep(1)
            break
        elif proceed.lower() == "y":
            continue 
    exit()
