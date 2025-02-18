import pickle
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sklearn.metrics.pairwise import cosine_similarity
from tkinter.filedialog import askopenfilename
from image_processing import extract_features
from config import FILENAMES_FILE, EMBEDDINGS_FILE
from header import print_header
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.ResNet50 import load_model

# Define BASE_DIR
BASE_DIR = "C:/Users/joao_/OneDrive/Documents/GitHub/Fashion"

# Load saved embeddings and filenames
features_path = os.path.join(BASE_DIR, "data")
feature_list = pickle.load(open(os.path.join(features_path, "embeddings.pkl"), 'rb'))
filenames = pickle.load(open(os.path.join(features_path, "filenames.pkl"), 'rb'))

def recommend_similar_images(query_img_path, model, top_n=5):
    """Dado uma imagem de entrada, recomenda imagens visualmente semelhantes."""
    if not query_img_path:
        query_img_path = askopenfilename()
    
    if not query_img_path:
        print("Nenhuma imagem foi selecionada.")
        return

    if len(feature_list) == 0:
        print("A lista de features está vazia. Execute `main.py` primeiro para gerar os embeddings.")
        return

    try:
        query_features = extract_features(query_img_path, model).reshape(1, -1)
        similarities = cosine_similarity(query_features, feature_list)[0]
        top_indices = np.argsort(similarities)[::-1][1:top_n+1]  # Exclui a própria imagem da busca

        fig, axes = plt.subplots(1, top_n+1, figsize=(15, 5))
        
        # Exibir a imagem de consulta
        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(query_img)
        axes[0].set_title("Imagem de Consulta")
        axes[0].axis("off")

        # Exibir imagens recomendadas
        for i, idx in enumerate(top_indices):
            similar_img = cv2.imread(filenames[idx])
            similar_img = cv2.cvtColor(similar_img, cv2.COLOR_BGR2RGB)
            axes[i+1].imshow(similar_img)
            axes[i+1].set_title(f"Similar {i+1}")
            axes[i+1].axis("off")

        plt.show()
    except Exception as e:
        print(f"Erro ao recomendar imagens: {e}")

if __name__ == "__main__":
    print_header()
    selected_image = askopenfilename()
    model = load_model()  # Replace with your actual model
    recommend_similar_images(selected_image, model)