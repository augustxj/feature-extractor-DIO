# Dataset Feature Extraction and Recommendation System 🛍️👗

A deep learning-based system that extracts features from a source dataset and recommends visually similar images using ResNet50 and cosine similarity. Built with TensorFlow and OpenCV.

Originally, it was trained on a fashion dataset. The code automatically resizes the images to (224, 224), the optimal size for ResNet50.

![Header Art](https://raw.githubusercontent.com/augustxj/feature-extractor-DIO/refs/heads/main/results/interface.png)
## Result example:
![Result Art](https://raw.githubusercontent.com/augustxj/feature-extractor-DIO/refs/heads/main/results/final%20result.png)
## Features ✨
- **GPU Support**: Accelerates feature extraction using TensorFlow GPU capabilities.
- **ResNet50 Embeddings**: Extracts image features using a pre-trained ResNet50 model.
- **Cosine Similarity**: Finds similar images based on extracted embeddings.
- **Interactive CLI**: Colorful terminal interface with progress tracking.
- **Batch Processing**: Pre-computes and stores embeddings for efficient recommendations.

## Installation 🛠️

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/augustxj-feature-extractor-dio.git
   cd augustxj-feature-extractor-dio
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset Setup**:
   - Create a `dataset/images` folder in the project root.
   - Add your fashion images to `dataset/images`.
  
   or
   - Change the `images_path` var inside the `config.py` file to your dataset path.

## Usage 🚀

### 1. Feature Extraction
Run the main script:
```bash
python main.py
```
- The system will:
  - Check for GPU availability
  - Extract features from all images in `dataset/images`
  - Save embeddings to `data/embeddings.pkl`

### 2. Get Recommendations
After feature extraction:
1. A file dialog will prompt you to select a query image
2. The system will display:
   - Input image
   - Top 5 similar images from your dataset

### 3. Reuse Existing Embeddings
If embeddings already exist:
```bash
python main.py
```
- Choose `N` to skip re-extraction and go directly to recommendation

### Utility Script
If you face any erros with not found directories, go into `\src\utils` and please run:
```bash
python directories_resolve.py
```

## Configuration ⚙️
Modify `src/utils/config.py` to adjust:
- Image size (`IMAGE_SIZE`)
- Dataset/data paths
- Model weights source

## Dependencies 📦
- tensorflow>=2.18.0
- numpy>=1.26.0
- scikit-learn>=1.6.0
- matplotlib>=3.3.0
- opencv-python>=4.10.0
- tqdm>=4.67.0

**Note**: For GPU support, ensure CUDA and cuDNN are installed.
## Next Updates 🚧
### 🔄 Dynamic Dataset Location Selection
- **Current Limitation**: The dataset path is hardcoded to `dataset/images`
- **Planned Feature**: Implement `tkinter.filedialog.askdirectory()` to let users:
  - Interactively choose their dataset folder location
  - Store custom paths in configuration
  - Validate selected directories for image files
- **Benefits**: 
  - No need to edit config files manually
  - Better support for multiple dataset locations
  - More flexible workflow for different projects

## Contributing 🤝
Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---
🖼️ Built with passion by [@augustxj](https://github.com/augustxj) | 🔗 Inspired by Deep Learning-based recommendation systems
