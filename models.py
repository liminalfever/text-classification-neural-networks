import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential
from keras.layers import Dense, Input, Bidirectional, LSTM, Embedding, LayerNormalization, Reshape, GlobalMaxPooling1D, Dropout, MultiHeadAttention
from keras.models import Model
from keras.regularizers import l1

class TransformerBlock(layers.Layer):
    """Transformer Block"""
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
    """Token and Position Embedding"""
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
    """Pretrained Token and Position Embedding"""  
    def __init__(self, embedding_matrix):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False)
        self.pos_emb = layers.Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1])

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        out = x + positions

        return out

class Transformer:
    """Transformer Model"""

    @staticmethod
    def build(embed_dim, num_heads, ff_dim, dropout_rate, embedding_matrix=None, max_len=None, vocab_size=None):
        """Build Transformer Model
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Hidden layer size in feed forward network inside transformer
            dropout_rate: Dropout rate
            embedding_matrix: Embedding matrix
            max_len: Maximum length of sequence
            vocab_size: Vocabulary size
        """
        inputs = layers.Input(shape=(max_len, ))
        if embedding_matrix is not None:
            embedding_layer = PreTrainedTokenAndPositionEmbedding(embedding_matrix)
        else:
            embedding_layer = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim)
        X = embedding_layer(inputs)
        transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
        X = transformer_block(X)
        X = layers.GlobalMaxPooling1D()(X)
        X = layers.Dropout(dropout_rate)(X)
        X = layers.Dense(20, activation='relu')(X)
        X = layers.Dropout(dropout_rate)(X)
        outputs = layers.Dense(1, activation='sigmoid')(X)

        model = keras.Model(inputs=inputs, outputs=outputs)

        return model

class BiLSTM:
    """Bidirectional LSTM Model
    Args:  
        embedding_matrix: Embedding matrix
        trainable: Whether to train embedding layer
        dropout: Dropout rate
        regularization_factor: Regularization factor
    """
    @staticmethod
    def build_model(embedding_matrix, trainable=False, dropout=0.5, regularization_factor=1e-6):
        sentence_indices = Input(shape=(None, ), dtype='int32')
        embeddings = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], trainable=trainable)(sentence_indices)
        X1 = Bidirectional(LSTM(128, kernel_regularizer=l1(regularization_factor), return_sequences=True))(embeddings)
        X1 = Dropout(dropout)(X1)
        X2 = Bidirectional(LSTM(32, kernel_regularizer=l1(regularization_factor)))(X1)
        X = Dropout(dropout)(X2)
        X = Reshape((-1, 32))(X)
        X = GlobalMaxPooling1D()(X)
        X = Dense(1, activation='sigmoid')(X)
        
        model = Model(inputs=sentence_indices, outputs=X)

        return model


class BiLSTM_heavy:
    """Bidirectional LSTM Model
    Args:  
        embedding_matrix: Embedding matrix
        trainable: Whether to train embedding layer
        dropout: Dropout rate
        regularization_factor: Regularization factor
    """
    @staticmethod
    def build_model(embedding_matrix, dropout=0.5, regularization_factor=1e-6):
        sentence_indices = Input(shape=(None, ), dtype='int32')

        embeddings = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix])(sentence_indices)

        X1 = Bidirectional(LSTM(units=512, kernel_regularizer=l1(regularization_factor), return_sequences=True))(embeddings)
        X1 = Dropout(rate=dropout)(X1)
        X2 = Bidirectional(LSTM(units=256, kernel_regularizer=l1(regularization_factor), return_sequences=True))(X1)
        X2 = Dropout(rate=dropout)(X2)
        X3 = Bidirectional(LSTM(units=128, kernel_regularizer=l1(regularization_factor)))(X2)
        X = Dropout(rate=dropout)(X3)
        X = Reshape((-1, 256))(X)
        X = GlobalMaxPooling1D()(X)
        X = Dense(128, activation='relu')(X)
        X = Dense(1, activation='sigmoid')(X)

        model = Model(inputs=sentence_indices, outputs=X)

        return model