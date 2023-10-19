import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential
from keras.layers import Dense, Input, Bidirectional, LSTM, Embedding, LayerNormalization, GlobalMaxPooling1D, Dropout, MultiHeadAttention
from keras.models import Model
from keras.regularizers import l1

# -------- TRANSFORMER MODEL ------------------------------------------------------
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim=32, num_heads=2, ff_dim=16, dropout_rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [Dense(ff_dim, activation='relu'),
            Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)

        return x + positions

class PreTrainedTokenAndPositionEmbedding(layers.Layer):
    def __init__(self, embedding_matrix, trainable=False):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], trainable=trainable)
        self.pos_emb = layers.Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1])

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        out = x + positions

        return out

class Transformer:
    """Transformer Model for text classification."""
    @staticmethod
    def build(embed_dim: int,
              num_heads: int, 
              num_blocks: int, 
              ff_dim: int, 
              dropout_rate: float = 0.1, 
              embedding_matrix=None, 
              trainable=False, 
              max_len=None, 
              vocab_size=None):
        
        """Builds the model.
        Args:
            embed_dim (int): The dimension of the token embeddings.
            num_heads (int): The number of self-attention heads in the Transformer.
            num_blocks (int): The number of Transformer blocks (layers) in the model.
            ff_dim (int): The dimension of the feed-forward network in each Transformer block.
            dropout_rate (float): The dropout rate to apply within the model.
            embedding_matrix (numpy.ndarray, optional): A pre-trained embedding matrix. If provided, it is used for token embeddings.
            trainable (bool, optional): Whether the embedding layer should be trainable if using a pre-trained embedding matrix.
            max_len (int, optional): The maximum sequence length.
            vocab_size (int, optional): The vocabulary size.

        Returns:
            tensorflow.keras.Model: A Transformer-based text classification model.
        """

        inputs = layers.Input(shape=(max_len, ))
        if embedding_matrix is not None:
            embedding_layer = PreTrainedTokenAndPositionEmbedding(embedding_matrix, trainable)
        else:
            embedding_layer = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim)
        X = embedding_layer(inputs)
        for _ in range(num_blocks):
            X = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)(X)        
        X = layers.GlobalMaxPooling1D()(X)
        X = layers.Dropout(dropout_rate)(X)
        X = layers.Dense(20, activation='relu')(X)
        X = layers.Dropout(dropout_rate)(X)
        outputs = layers.Dense(2, activation='softmax')(X)

        model = keras.Model(inputs=inputs, outputs=outputs)

        return model
# ---------------------------------------------------------------------------------

# -------- BILSTM MODEL -----------------------------------------------------------
class BiLSTM:
    """BiLSTM model for text classification"""
    @staticmethod
    def build(embedding_matrix: np.ndarray,
                layer_1_units: int = 128,
                layer_2_units: int = 32,
                dense_units: int = 1,
                dropout_rate: float = 0.5,
                recurrent_dropout_rate: float = 0,
                regularization_factor: float = 0.001,
                trainable: bool = False,
                input_length: int = 100,
                activation: str = 'sigmoid'):

        """Builds the model.
        Args:
            embedding_matrix: Embedding matrix.
            layer_1_units: Number of units in the first LSTM layer.
            layer_2_units: Number of units in the second LSTM layer.
            dropout_rate: Dropout rate.
            recurrent_dropout_rate: Recurrent dropout rate.
            trainable: Whether to make the embedding layer trainable.
            regularization_factor: Regularization factor for the LSTM layers.
            activation: Activation function for the second last layer.

        Returns:
            tensorflow.keras.Model: A BiLSTM-based text classification model.
        """

        # Embedding layer
        embeddings = Embedding(input_dim=embedding_matrix.shape[0],
                               output_dim=embedding_matrix.shape[1],
                               weights=[embedding_matrix],
                               input_length=input_length,
                               trainable=trainable)
        inputs = Input(shape=(None, ), dtype='int64')
        embedded_sequences = embeddings(inputs)

        # Hidden layers
        X = Bidirectional(LSTM(layer_1_units, kernel_regularizer=l1(regularization_factor), activation = "tanh", return_sequences=True, dropout=dropout_rate, recurrent_activation = "sigmoid", recurrent_dropout = recurrent_dropout_rate, unroll = False, use_bias = True))(embedded_sequences)
        X = Bidirectional(LSTM(layer_2_units, kernel_regularizer=l1(regularization_factor), activation = "tanh", return_sequences=True, dropout=dropout_rate, recurrent_activation = "sigmoid", recurrent_dropout = recurrent_dropout_rate, unroll = False, use_bias = True))(X)
        X = GlobalMaxPooling1D()(X)

        # Output layer
        if dense_units > 1:
            X = Dense(dense_units, activation=activation)(X)
        outputs = Dense(1, activation='sigmoid')(X)
        model = Model(inputs, outputs)

        return model
# ---------------------------------------------------------------------------------