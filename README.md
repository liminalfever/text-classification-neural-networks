# Neural Networks for Text Classification

This module contains two text classification models. You can choose between a Transformer-based model and a Bidirectional LSTM (BiLSTM) model for your text classification needs. 

## Transformer Model

The Transformer model in this module is a powerful deep learning architecture for text classification tasks. It utilizes self-attention mechanisms and multiple layers of Transformer blocks to capture contextual information within input text data.

### Usage

```python
from text_classification_models import Transformer

# Build a Transformer model for text classification
model = Transformer.build(embed_dim, num_heads, num_blocks, ff_dim, dropout_rate, embedding_matrix, trainable, max_len, vocab_size)
```

### Arguments

- `embed_dim` (int): The dimension of the token embeddings.
- `num_heads` (int): The number of self-attention heads in the Transformer.
- `num_blocks` (int): The number of Transformer blocks (layers) in the model.
- `ff_dim` (int): The dimension of the feed-forward network in each Transformer block.
- `dropout_rate` (float): The dropout rate to apply within the model.
- `embedding_matrix` (numpy.ndarray, optional): A pre-trained embedding matrix. If provided, it is used for token embeddings.
- `trainable` (bool, optional): Whether the embedding layer should be trainable if using a pre-trained embedding matrix.
- `max_len` (int, optional): The maximum sequence length.
- `vocab_size` (int, optional): The vocabulary size.

## BiLSTM Model

The Bidirectional LSTM (BiLSTM) model in this module is a traditional but effective choice for text classification tasks. It leverages bidirectional recurrent layers to capture contextual information within the input text data.

### Usage

```python
from text_classification_models import BiLSTM

# Build a BiLSTM model for text classification
model = BiLSTM.build(embedding_matrix, layer_1_units, layer_2_units, dense_units, dropout_rate, recurrent_dropout_rate, regularization_factor, trainable, input_length, activation)
```

### Arguments

- `embedding_matrix` (numpy.ndarray): A pre-trained embedding matrix.
- `layer_1_units` (int): Number of units in the first LSTM layer.
- `layer_2_units` (int): Number of units in the second LSTM layer.
- `dropout_rate` (float): Dropout rate.
- `recurrent_dropout_rate` (float): Recurrent dropout rate.
- `trainable` (bool): Whether to make the embedding layer trainable.
- `regularization_factor` (float): Regularization factor for the LSTM layers.
- `input_length` (int): Length of input sequences.
- `activation` (str): Activation function for the second last layer.

## Installation

To use these models, you need to import them into your Python code and then you can build and train them for your specific text classification task.

```python
from text_classification_models import Transformer, BiLSTM
```

## Requirements

- Python 3.6+
- TensorFlow
- NumPy
- Keras