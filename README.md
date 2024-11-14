# Pneumonia Prediction using Pretrained CNN Models

This repository contains a deep learning project for pneumonia prediction using chest X-ray images. Leveraging transfer learning with pretrained CNN models such as VGG16, VGG19, and InceptionV3, this project aims to identify pneumonia cases with high accuracy.

## Project Structure

- **Source Code Folder**: Contains the primary code files:
  - **`Preprocessing Input and Model Train.ipynb`**: A Jupyter Notebook for preprocessing the input data and training the CNN models.
  - **`Read_Model.py`**: A script for loading and evaluating the trained model on test images, displaying predictions based on the model output.

## Models Used

The following pretrained CNN architectures are utilized in this project:

- **VGG16**: A 16-layer convolutional neural network well-suited for image classification tasks.
- **VGG19**: Similar to VGG16 but with a deeper architecture, providing enhanced feature extraction capabilities.
- **InceptionV3**: A model known for handling a broader set of features and capturing more detailed patterns within images.

## Dataset

This project uses a pneumonia dataset comprising X-ray images, which should be structured as follows for preprocessing and training:
- `train/`: Contains labeled X-ray images for training.
- `test/`: Contains labeled X-ray images for testing.

> Note: Ensure the dataset is placed in the correct directory before running the notebook or model loading script.

## Requirements

To run this project, install the necessary packages by running:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Preprocess and Train the Model

Use the `Preprocessing Input and Model Train.ipynb` notebook to preprocess the data and train the model:

1. Open the notebook:
   ```bash
   jupyter notebook "Preprocessing Input and Model Train.ipynb"
   ```
2. Follow the steps within the notebook to:
   - Load and preprocess images.
   - Train a CNN model using one of the pretrained architectures (VGG16, VGG19, or InceptionV3).
3. Save the trained model for later use.

### 2. Load and Evaluate the Trained Model

To use the trained model for predictions on new images, run the `Read_Model.py` script:

```bash
python Read_Model.py
```

This script will:
- Load the specified trained model.
- Perform predictions on test images.
- Display results, showing whether the model predicts pneumonia.

## Example

```python
# Example usage in Read_Model.py
from keras.models import load_model
from utils import load_and_preprocess_image  # Hypothetical helper function

# Load the trained model
model = load_model("path_to_your_saved_model.h5")

# Load an example image and make a prediction
image = load_and_preprocess_image("path_to_test_image.jpg")
prediction = model.predict(image)
print("Pneumonia Detected" if prediction[0] > 0.5 else "No Pneumonia Detected")
```

## Results

The trained model demonstrates robust performance on the test set, with accuracy metrics that vary slightly depending on the CNN architecture used. Check the output of the `Read_Model.py` script for visualizations of the predictions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
