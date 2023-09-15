# Transformer and BiLSTM Models

This repository contains Python code for implementing Transformer and Bidirectional LSTM (BiLSTM) models for natural language processing (NLP) tasks. These models can be used for tasks such as text classification, sentiment analysis, and more.

## Transformer Model
The Transformer model is implemented using Keras. It includes custom layers for token and position embedding, as well as a Transformer block. You can build the model with different configurations using the `Transformer.build` method. Here are the key components of the Transformer model:

- `TransformerBlock`: This is the core building block of the Transformer model, consisting of multi-head self-attention and feedforward layers.

- `TokenAndPositionEmbedding`: Custom layer for combining token and position embeddings.

- `PreTrainedTokenAndPositionEmbedding`: Custom layer for using pretrained embeddings and combining them with position embeddings.

- `Transformer`: A class to build the complete Transformer model with various configuration options for embedding dimensions, number of attention heads, hidden layer size, dropout rate, and more.

## Bidirectional LSTM Models
Two variations of BiLSTM models are provided in this repository:

1. `BiLSTM`: A BiLSTM model with moderate complexity, suitable for a wide range of NLP tasks.

2. `BiLSTM_heavy`: A more complex BiLSTM model with larger LSTM units, ideal for tasks requiring a deeper network.

Both BiLSTM models accept pretrained word embeddings as input and allow you to specify whether to train the embedding layer. Additionally, you can configure dropout rates and regularization factors.

## Usage
You can use these models for your NLP tasks by following these steps:

1. Import the desired model class (Transformer, BiLSTM, BiLSTM_heavy).
2. Build the model by specifying the required hyperparameters and input data shapes.
3. Compile the model and train it on your dataset.
4. Evaluate the model's performance and use it for inference.

Refer to the code and documentation within the repository for detailed usage examples and customization options.

## Dependencies
- TensorFlow
- Keras

Please make sure to install these dependencies before using the code.
