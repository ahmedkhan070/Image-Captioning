# Image-Captioning
Used CNN ( DenseNet201 ) and LSTM to make text of images

Here's the updated README file with the reference to the "Show and Tell: A Neural Image Caption Generator" paper and mentioning the use of DenseNet201:

```markdown
# Flickr8k Image Captioning using CNNs and LSTMs

This project demonstrates how to create an image captioning model using Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks. The dataset used for this project is the Flickr8k dataset.

## Project Structure

- `flickr8k-image-captioning-using-cnns-lstms.ipynb`: The main Jupyter notebook containing the code for data preprocessing, model building, training, and evaluation.
- `requirements.txt`: A file containing the list of dependencies required to run the notebook.

## Installation

To run this project, you need to have Python installed on your system. You can install the required dependencies using `pip`:

```sh
pip install -r requirements.txt
```

## Dataset

The Flickr8k dataset contains 8,000 images with corresponding captions. You can download the dataset from [here](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip).

After downloading, extract the contents into a directory named `Flickr8k_Dataset`.

## How to Run

1. Clone the repository and navigate to the project directory:

    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Ensure you have all the dependencies installed:

    ```sh
    pip install -r requirements.txt
    ```

3. Place the extracted `Flickr8k_Dataset` directory in the project directory.

4. Open the Jupyter notebook:

    ```sh
    jupyter notebook flickr8k-image-captioning-using-cnns-lstms.ipynb
    ```

5. Run the cells in the notebook sequentially to preprocess the data, build the model, train it, and evaluate its performance.

## Model Overview

The model consists of two main parts:
- **CNN Encoder**: This part uses a pre-trained DenseNet201 to extract features from images.
- **LSTM Decoder**: This part uses LSTM networks to generate captions based on the image features extracted by the CNN.

The approach used in this project is inspired by the "Show and Tell: A Neural Image Caption Generator" paper. You can refer to the paper [here](https://arxiv.org/pdf/1411.4555.pdf) for more details on the encoder-decoder architecture.

## Results

The model is evaluated on the Flickr8k test set. The performance can be measured using metrics such as BLEU scores.

## Dependencies

- TensorFlow
- Keras
- NumPy
- Matplotlib
- NLTK
- Pillow
- scikit-learn

