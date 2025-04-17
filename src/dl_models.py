from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Embedding, GlobalMaxPooling1D, Reshape, Concatenate
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Flatten, Conv1D, MaxPooling1D, Embedding, Input, GlobalMaxPooling1D, Convolution1D
from keras_self_attention import SeqSelfAttention
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Add
)
from tensorflow.keras import regularizers

from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.decomposition import PCA
import numpy as np


from tensorflow.keras.layers import (Input, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D,
                                     Dense, Dropout, Concatenate, Add, Activation, BatchNormalization, Multiply,
                                     Flatten, RepeatVector, Permute)


from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dropout, GlobalMaxPooling1D, Dense, BatchNormalization, Attention, Flatten
from tensorflow.keras.optimizers import Adam



from tensorflow.keras import Model
from tensorflow.keras.layers import (Input, Embedding, Conv1D, MaxPooling1D, Dropout,
                                     GlobalMaxPooling1D, Bidirectional, LSTM, Dense,
                                     Concatenate, Attention, Flatten, LayerNormalization)
from tensorflow.keras import regularizers
from tensorflow.keras.layers import MultiHeadAttention



class DlModels:

    def __init__(self, categories, embed_dim, sequence_length, weight_decay):

        self.categories = categories
        self.embed_dim = embed_dim
        self.sequence_length = sequence_length
        self.weight_decay = weight_decay

    def rnn_base(self, char_index):

        model = Sequential()
        voc_size = len(char_index.keys())

        model.add(Embedding(voc_size + 1, self.embed_dim))
        model.add(LSTM(128))
        model.add(Dense(len(self.categories), activation='sigmoid'))

        return model

    def brnn_base(self, char_index):

        model = Sequential()
        voc_size = len(char_index.keys())

        model.add(Embedding(voc_size + 1, self.embed_dim))
        model.add(Bidirectional(LSTM(128)))
        model.add(Dense(len(self.categories), activation='sigmoid'))

        return model

    def cnn_base(self, char_index):

        model = Sequential()
        voc_size = len(char_index.keys())
        model.add(
            Embedding(voc_size + 1, self.embed_dim, input_length=self.sequence_length + 9))
        model.add(Convolution1D(128, 3, activation='tanh'))
        model.add(Flatten())

        model.add(Dense(len(self.categories), activation='sigmoid'))

        return model

    def ann_base(self, char_index):

        model = Sequential()
        voc_size = len(char_index.keys())

        model.add(
            Embedding(voc_size + 1, self.embed_dim, input_length=self.sequence_length))
        model.add(Dense(128, activation='tanh'))
        model.add(Flatten())

        model.add(Dense(len(self.categories), activation='sigmoid'))

        return model

    def att_base(self, char_index):

        model = Sequential()
        voc_size = len(char_index.keys())

        model.add(Embedding(voc_size + 1, self.embed_dim, input_length=self.sequence_length))
        # model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
        model.add(SeqSelfAttention(attention_activation='sigmoid'))
        model.add(Flatten())

        model.add(Dense(len(self.categories), activation='sigmoid'))

        return model

    def rnn_complex(self, char_index):

        model = Sequential()
        voc_size = len(char_index.keys())
        print("voc_size: {}".format(voc_size))

        model.add(Embedding(voc_size + 1, self.embed_dim))
        model.add(LSTM(128, return_sequences=True))

        model.add(LSTM(128, return_sequences=True))

        model.add(LSTM(128, return_sequences=True))

        model.add(LSTM(128, return_sequences=True))

        model.add(LSTM(128, return_sequences=True))

        model.add(LSTM(128, return_sequences=True))

        model.add(LSTM(128))

        model.add(Dense(len(self.categories), activation='sigmoid'))

        return model

    def brnn_complex(self, char_index):

        model = Sequential()
        voc_size = len(char_index.keys())
        print("voc_size: {}".format(voc_size))
        model.add(Embedding(voc_size + 1, self.embed_dim))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))

        model.add(Bidirectional(LSTM(64, return_sequences=True)))

        model.add(Bidirectional(LSTM(64, return_sequences=True)))

        model.add(Bidirectional(LSTM(64, return_sequences=True)))

        model.add(Bidirectional(LSTM(64, return_sequences=True)))

        model.add(Bidirectional(LSTM(64, return_sequences=True)))

        model.add(Bidirectional(LSTM(128)))

        model.add(Dense(len(self.categories), activation='sigmoid'))

        return model

    def ann_complex(self, char_index):

        model = Sequential()
        voc_size = len(char_index.keys())
        print("voc_size: {}".format(voc_size))

        model.add(Embedding(voc_size + 1, self.embed_dim, input_length=self.sequence_length))

        model.add(Dense(128, activation='tanh'))

        model.add(Dense(128, activation='tanh'))

        model.add(Dense(128, activation='tanh'))

        model.add(Dense(128, activation='tanh'))

        model.add(Dense(128, activation='tanh'))

        model.add(Dense(128, activation='tanh'))

        model.add(Dense(128, activation='tanh'))
        model.add(Flatten())

        model.add(Dense(len(self.categories), activation='sigmoid'))

        return model

    def att_complex(self, char_index):

        model = Sequential()
        voc_size = len(char_index.keys())

        model.add(Embedding(voc_size + 1, self.embed_dim, input_length=self.sequence_length))

        model.add(LSTM(units=128, return_sequences=True))

        model.add(Bidirectional(LSTM(units=128, return_sequences=True)))

        model.add(SeqSelfAttention(attention_activation='sigmoid'))

        model.add(Bidirectional(LSTM(units=128, return_sequences=True)))

        model.add(LSTM(units=128, return_sequences=True))

        model.add(SeqSelfAttention(attention_activation='sigmoid'))

        model.add(LSTM(units=128, return_sequences=True))

        model.add(Flatten())

        model.add(Dense(len(self.categories), activation='sigmoid'))

        return model

    def cnn_complex(self, char_index):

        model = Sequential()
        voc_size = len(char_index.keys())
        print("voc_size: {}".format(voc_size))
        model.add(Embedding(voc_size + 1, self.embed_dim, input_length=self.sequence_length))

        model.add(Convolution1D(128, 3, activation='tanh'))
        model.add(MaxPooling1D(3))
        model.add(Dropout(0.2))

        model.add(Convolution1D(128, 7, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(128, 5, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(128, 3, activation='tanh', padding='same'))
        model.add(MaxPooling1D(3))
        model.add(Dropout(0.2))

        model.add(Convolution1D(128, 5, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(128, 3, activation='tanh', padding='same'))
        model.add(MaxPooling1D(3))
        model.add(Dropout(0.2))

        model.add(Convolution1D(128, 3, activation='tanh', padding='same'))
        model.add(MaxPooling1D(3))
        model.add(Dropout(0.2))

        model.add(Flatten())

        model.add(Dense(len(self.categories), activation='sigmoid'))

        return model

    def cnn_complex2(self, char_index):
        model = Sequential()
        voc_size = len(char_index.keys())
        print("voc_size: {}".format(voc_size))
        model.add(Embedding(voc_size + 1, self.embed_dim, input_length=self.sequence_length))

        model.add(Convolution1D(128, 3, activation='tanh'))
        model.add(MaxPooling1D(3))
        model.add(Dropout(0.2))

        model.add(Convolution1D(256, 7, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(96, 5, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(128, 3, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(196, 5, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(128, 3, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(96, 5, activation='tanh', padding='same'))
        model.add(MaxPooling1D(3))
        model.add(Dropout(0.2))

        model.add(Convolution1D(128, 3, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(196, 5, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(128, 7, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(96, 3, activation='tanh', padding='same'))
        model.add(MaxPooling1D(3))
        model.add(Dropout(0.2))

        model.add(Flatten())

        model.add(Dense(len(self.categories), activation='sigmoid'))

        return model

    def cnn_complex3(self, char_index):
        model = Sequential()
        voc_size = len(char_index.keys())
        print("voc_size: {}".format(voc_size))
        model.add(Embedding(voc_size + 1, self.embed_dim, input_length=self.sequence_length, name="inputLayer"))

        model.add(Convolution1D(128, 3, activation='tanh'))
        model.add(MaxPooling1D(3))

        model.add(Convolution1D(256, 7, activation='tanh', padding='same'))
        model.add(Convolution1D(96, 5, activation='tanh', padding='same'))
        model.add(Convolution1D(128, 3, activation='tanh', padding='same'))
        model.add(MaxPooling1D(3))
        model.add(Convolution1D(196, 5, activation='tanh', padding='same'))
        model.add(Convolution1D(128, 3, activation='tanh', padding='same'))
        model.add(Convolution1D(96, 5, activation='tanh', padding='same'))
        model.add(Convolution1D(128, 3, activation='tanh', padding='same'))
        model.add(Convolution1D(196, 5, activation='tanh', padding='same'))
        model.add(Convolution1D(128, 7, activation='tanh', padding='same'))
        model.add(Convolution1D(96, 3, activation='tanh', padding='same'))
        model.add(MaxPooling1D(3))

        model.add(Convolution1D(196, 5, activation='tanh', padding='same'))
        model.add(Convolution1D(128, 7, activation='tanh', padding='same'))
        model.add(MaxPooling1D(3))

        model.add(Convolution1D(196, 5, activation='tanh', padding='same'))
        model.add(Convolution1D(128, 7, activation='tanh', padding='same'))
        model.add(Convolution1D(96, 3, activation='tanh', padding='same'))

        model.add(Flatten())

        model.add(Dense(2, activation='sigmoid', name="inferenceLayer"))

        return model

    def custom_model(self, char_index):

        model = None

        return model
    

    def DeepCNN_LSTM_Attention(self, char_index):
        model = Sequential()
        voc_size = len(char_index.keys())
        print("voc_size: {}".format(voc_size))

        # Embedding layer
        model.add(Embedding(voc_size + 1, self.embed_dim, input_length=self.sequence_length))

        # First convolutional block
        model.add(Conv1D(256, 5, activation='relu', padding='same'))
        model.add(MaxPooling1D(3))
        model.add(Dropout(0.3))

        # Second convolutional block
        model.add(Conv1D(512, 5, activation='relu', padding='same'))
        model.add(MaxPooling1D(3))
        model.add(Dropout(0.4))

        # Third convolutional block
        model.add(Conv1D(1024, 3, activation='relu', padding='same'))
        model.add(MaxPooling1D(3))
        model.add(Dropout(0.5))

        # Fourth convolutional block with GlobalMaxPooling
        model.add(Conv1D(2048, 3, activation='relu', padding='same'))
        model.add(MaxPooling1D(3))
        model.add(Dropout(0.5))

        # LSTM layer to capture sequential dependencies (return_sequences=False)
        model.add(LSTM(256))  # Output is a 2D tensor of shape (batch_size, 256)
        model.add(LSTM(128))  # Output is a 2D tensor of shape (batch_size, 128)

        # Attention mechanism to focus on important parts of the sequence
        model.add(SeqSelfAttention(attention_activation='sigmoid'))

        # Fully connected layer for classification
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))

        # Output layer
        model.add(Dense(len(self.categories), activation='sigmoid'))

        return model
    
    def DeepCNN_V2(self, char_index):
        model = Sequential()
        voc_size = len(char_index.keys())
        print("voc_size: {}".format(voc_size))

        # Embedding layer
        model.add(Embedding(voc_size + 1, self.embed_dim, input_length=self.sequence_length, name="inputLayer"))

        # First convolutional block
        model.add(Convolution1D(128, 3, activation='relu'))
        model.add(MaxPooling1D(2))  # Change pool size from 3 to 2
        model.add(Dropout(0.3))

        # Second convolutional block with larger filters
        model.add(Convolution1D(256, 5, activation='relu', padding='same'))
        model.add(MaxPooling1D(2))  # Change pool size from 3 to 2
        model.add(Dropout(0.3))

        # Third convolutional block with even larger filters
        model.add(Convolution1D(512, 7, activation='relu', padding='same'))
        model.add(MaxPooling1D(2))  # Change pool size from 3 to 2
        model.add(Dropout(0.4))

        # Fourth convolutional block with larger kernel size and more filters
        model.add(Convolution1D(1024, 9, activation='relu', padding='same'))
        model.add(MaxPooling1D(2))  # Change pool size from 3 to 2
        model.add(Dropout(0.4))

        # Fifth convolutional block with larger kernel size and more filters
        model.add(Convolution1D(2048, 11, activation='relu', padding='same'))
        model.add(MaxPooling1D(2))  # Change pool size from 3 to 2
        model.add(Dropout(0.5))

        # Sixth convolutional block with GlobalMaxPooling
        model.add(Convolution1D(4096, 13, activation='relu', padding='same'))
        model.add(MaxPooling1D(2))  # Change pool size from 3 to 2
        model.add(Dropout(0.5))

        # Add GlobalMaxPooling to reduce dimensionality after convolutions
        model.add(GlobalMaxPooling1D())

        # Fully connected layer for classification
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))

        # Output layer
        model.add(Dense(len(self.categories), activation='sigmoid'))

        return model


    def DeepCNN_V3(self, char_index):
        model = Sequential()
        voc_size = len(char_index.keys())
        print("voc_size: {}".format(voc_size))
        
        weight_decay = self.weight_decay 

        # Embedding layer
        model.add(Embedding(voc_size + 1, self.embed_dim, input_length=self.sequence_length, name="inputLayer"))

        # Block 1
        model.add(Convolution1D(128, 3, activation='relu',
                                kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.3))

        # Block 2
        model.add(Convolution1D(256, 5, activation='relu', padding='same',
                                kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.3))

        # Block 3
        model.add(Convolution1D(512, 7, activation='relu', padding='same',
                                kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.4))

        # Block 4
        model.add(Convolution1D(1024, 9, activation='relu', padding='same',
                                kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.4))

        # Block 5
        model.add(Convolution1D(2048, 11, activation='relu', padding='same',
                                kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.5))

        # Block 6
        model.add(Convolution1D(4096, 13, activation='relu', padding='same',
                                kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.5))

        model.add(GlobalMaxPooling1D())

        # Dense layer
        model.add(Dense(512, activation='relu',
                        kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Dropout(0.5))

        # Output layer
        # model.add(Dense(len(self.categories), activation='sigmoid'))
        model.add(Dense(2, activation='softmax'))
        
        return model


    def DeepCNN_Light(self, char_index):
        
        model = Sequential()
        voc_size = len(char_index.keys())
        print("voc_size: {}".format(voc_size))
        
        weight_decay = self.weight_decay  # 

        # Embedding Layer
        model.add(Embedding(voc_size + 1, self.embed_dim, input_length=self.sequence_length, name="inputLayer"))

        # Convolution Block 1
        model.add(Conv1D(64, 3, activation='relu', padding='same',
                        kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.3))

        # Convolution Block 2
        model.add(Conv1D(128, 5, activation='relu', padding='same',
                        kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.4))

        # Global Pooling
        model.add(GlobalMaxPooling1D())

        # Dense Layer
        model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Dropout(0.5))

        # Output Layer
        model.add(Dense(2, activation='softmax'))  # 二分类用 softmax + one-hot 标签

        return model

    def DeepCNN_Light_V2(self, char_index):
        
        model = Sequential()
        voc_size = len(char_index.keys())
        print("voc_size: {}".format(voc_size))
        
        weight_decay = self.weight_decay  # weight_decay

        # Embedding Layer
        model.add(Embedding(voc_size + 1, self.embed_dim, input_length=self.sequence_length, name="inputLayer"))

        # Convolution Block 1
        model.add(Conv1D(64, 3, activation='relu', padding='same',
                        kernel_regularizer=regularizers.l2(weight_decay)))  # Decrease regularization
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.3))  # Reduced dropout

        # Convolution Block 2
        model.add(Conv1D(128, 5, activation='relu', padding='same',
                        kernel_regularizer=regularizers.l2(weight_decay)))  # Decrease regularization
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.3))  # Reduced dropout

        # Global Pooling
        model.add(GlobalMaxPooling1D())

        # Dense Layer
        model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))  # Slight regularization
        model.add(Dropout(0.4))  # Reduced dropout

        # Output Layer
        model.add(Dense(2, activation='softmax'))  # 二分类用 softmax + one-hot 标签

        return model


    def DeepCNN_Light_V2_1(self, x_train, char_index=None):
        model = Sequential()
        
        if char_index is not None:  # 如果使用字符或词汇级特征
            # 获取词汇大小
            voc_size = len(char_index.keys())  
            print("voc_size: {}".format(voc_size))

            # Embedding Layer
            model.add(Embedding(voc_size + 1, self.embed_dim, input_length=self.sequence_length, name="inputLayer"))
            
            # Convolution Block 1
            model.add(Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(self.weight_decay)))  
            model.add(BatchNormalization())
            model.add(MaxPooling1D(2))
            model.add(Dropout(0.3))

            # Convolution Block 2
            model.add(Conv1D(128, 5, activation='relu', padding='same', kernel_regularizer=regularizers.l2(self.weight_decay)))  
            model.add(BatchNormalization())
            model.add(MaxPooling1D(2))
            model.add(Dropout(0.3))

            # Global Pooling
            model.add(GlobalMaxPooling1D())
            
        else:  # 如果使用 n-grams 或 TF-IDF 特征
           # 使用 InputLayer 来明确输入的形状（稀疏矩阵）
            model.add(Input(shape=(x_train.shape[1],)))
            model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay)))
            model.add(Dropout(0.3))  # Dropout层在dense层之后

        # Dense Layer
        model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay)))  
        model.add(Dropout(0.4))  # Reduced dropout

        # Output Layer
        model.add(Dense(2, activation='softmax'))  # 二分类用 softmax + one-hot 标签

        return model
    

    def DeepCNN_Light_V2_2(self, x_train, char_index=None):
        model = Sequential()
        
        if char_index is not None:  # If using character-level or word-level features
            # Get vocabulary size
            voc_size = len(char_index.keys())  
            print("voc_size: {}".format(voc_size))

            # Embedding Layer
            model.add(Embedding(voc_size + 1, self.embed_dim, input_length=self.sequence_length + 9, name="inputLayer"))
            
            # Convolution Block 1
            model.add(Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(self.weight_decay)))  
            model.add(BatchNormalization())
            model.add(MaxPooling1D(2))
            model.add(Dropout(0.3))

            # Convolution Block 2
            model.add(Conv1D(128, 5, activation='relu', padding='same', kernel_regularizer=regularizers.l2(self.weight_decay)))  
            model.add(BatchNormalization())
            model.add(MaxPooling1D(2))
            model.add(Dropout(0.3))

            # Global Pooling
            model.add(GlobalMaxPooling1D())
            
        else:  # If using n-grams or TF-IDF features
            # The new shape should be the number of features after manual feature addition
            input_shape = x_train.shape[1]  # Assuming x_train is now (num_samples, num_features)

            # Adjust the input shape to account for added manual features
            model.add(Input(shape=(input_shape,)))
            model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay)))
            model.add(Dropout(0.3))  # Dropout layer after dense

        # Dense Layer
        model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay)))  
        model.add(Dropout(0.4))  # Reduced dropout

        # Output Layer
        model.add(Dense(2, activation='softmax'))  # Binary classification with softmax

        return model


    def DeepCNN_Light_V2_3(self, x_train, char_index=None):
        model = Sequential()
        
        if char_index is not None:  # If using character-level or word-level features
            # Get vocabulary size
            voc_size = len(char_index.keys())  
            print("voc_size: {}".format(voc_size))

            # Embedding Layer
            model.add(Embedding(voc_size + 1, self.embed_dim, input_length=self.sequence_length + 5, name="inputLayer"))
            
            # Convolution Block 1
            model.add(Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(self.weight_decay)))  
            model.add(BatchNormalization())
            model.add(MaxPooling1D(2))
            model.add(Dropout(0.3))

            # Convolution Block 2
            model.add(Conv1D(128, 5, activation='relu', padding='same', kernel_regularizer=regularizers.l2(self.weight_decay)))  
            model.add(BatchNormalization())
            model.add(MaxPooling1D(2))
            model.add(Dropout(0.3))

            # Global Pooling
            model.add(GlobalMaxPooling1D())
            
        else:  # If using n-grams or TF-IDF features
            # The new shape should be the number of features after manual feature addition
            input_shape = x_train.shape[1]  # Assuming x_train is now (num_samples, num_features)

            # Adjust the input shape to account for added manual features
            model.add(Input(shape=(input_shape,)))
            model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay)))
            model.add(Dropout(0.3))  # Dropout layer after dense

        # Dense Layer
        model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay)))  
        model.add(Dropout(0.3))  # Reduced dropout

        # Output Layer
        model.add(Dense(2, activation='softmax'))  # Binary classification with softmax

        return model
    
    def DeepCNN_Light_Hybrid(self, text_input_shape, manual_input_shape, char_index=None):
        #     """
        # - text_input_shape: e.g., (200,)
        # - manual_input_shape: e.g., (9,)
        # """
        # ----------------------- 文本输入支路 -----------------------
        text_input = Input(shape=text_input_shape, name="Text_Input")

        if char_index is not None:
            vocab_size = len(char_index) + 1  # +1 for padding/oov
            x = Embedding(input_dim=vocab_size,
                        output_dim=self.embed_dim,
                        input_length=text_input_shape[0],
                        name="Embedding_Layer")(text_input)

            x = Conv1D(64, kernel_size=3, activation='relu', padding='same',
                    kernel_regularizer=regularizers.l2(self.weight_decay))(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = Dropout(0.3)(x)

            x = Conv1D(128, kernel_size=5, activation='relu', padding='same',
                    kernel_regularizer=regularizers.l2(self.weight_decay))(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = Dropout(0.3)(x)

            x = GlobalMaxPooling1D()(x)
        else:
            raise ValueError("char_index cannot be None in this architecture")

        # ----------------------- 手动特征子网络 -----------------------
        manual_input = Input(shape=manual_input_shape, name="Manual_Features")
        m = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay))(manual_input)
        m = Dropout(0.3)(m)
        m = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay))(m)
        m = Dropout(0.3)(m)

        # ----------------------- 拼接融合 -----------------------
        merged = Concatenate(name="Concatenated")([x, m])

        # ----------------------- 全连接分类器 -----------------------
        z = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay))(merged)
        z = Dropout(0.4)(z)
        output = Dense(2, activation='softmax', name="Output")(z)

        # ----------------------- 构建模型 -----------------------
        model = Model(inputs=[text_input, manual_input], outputs=output)
        
        return model


    def AMR_CNN(self, x_train, char_index=None):
        # from tensorflow.keras.layers import (Input, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D,
        #                              Dense, Dropout, Concatenate, Add, Activation, BatchNormalization, Multiply,
        #                              Flatten, RepeatVector, Permute)

        input_shape = (x_train.shape[1],)
        sequence_input = Input(shape=input_shape, name='text_input')

        if char_index is not None:
            vocab_size = len(char_index) + 1
            x = Embedding(vocab_size, self.embed_dim, input_length=input_shape[0],
                          name="embedding", embeddings_regularizer=regularizers.l2(self.weight_decay))(sequence_input)
        else:
            x = sequence_input  # in case TF-IDF/n-grams, directly Dense input

        # Multi-Kernel Convs
        convs = []
        for k in [3, 5, 7]:
            conv = Conv1D(128, kernel_size=k, activation='relu', padding='same',
                          kernel_regularizer=regularizers.l2(self.weight_decay))(x)
            conv = BatchNormalization()(conv)
            conv = MaxPooling1D(pool_size=2)(conv)
            conv = Dropout(0.3)(conv)
            convs.append(conv)

        # 多尺度卷积合并
        merged = Concatenate()(convs) if len(convs) > 1 else convs[0]

        # 投影降维使形状匹配（残差连接）
        merged_proj = Conv1D(128, 1, padding='same')(merged)

        # 残差卷积块
        res = Conv1D(128, 3, padding='same', activation='relu')(merged_proj)
        res = BatchNormalization()(res)
        res = Add()([merged_proj, res])


        # Attention Layer
        attention = Dense(1, activation='tanh')(res)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(128)(attention)
        attention = Permute([2, 1])(attention)
        attended = Multiply()([res, attention])
        x = GlobalMaxPooling1D()(attended)

        x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        x = Dropout(0.4)(x)
        output = Dense(2, activation='softmax')(x)

        model = Model(inputs=sequence_input, outputs=output)
        return model

    def AMR_CNN2(self, x_train, char_index=None):
        input_shape = (x_train.shape[1],)
        sequence_input = Input(shape=input_shape, name='text_input')

        # Step 1: Embedding Layer (if character index is available)
        if char_index is not None:
            vocab_size = len(char_index) + 1
            x = Embedding(vocab_size, self.embed_dim, input_length=input_shape[0],
                          embeddings_regularizer=regularizers.l2(self.weight_decay))(sequence_input)
        else:
            x = sequence_input  # in case TF-IDF/n-grams, directly Dense input
        
        # Step 2: Adaptive Convolutions with Multi-Scale Learning
        convs = []
        for k in [3, 5, 7]:  # Different kernel sizes for multi-scale learning
            conv = Conv1D(128, kernel_size=k, activation='relu', padding='same',
                          kernel_regularizer=regularizers.l2(self.weight_decay))(x)
            conv = BatchNormalization()(conv)
            conv = MaxPooling1D(pool_size=2)(conv)
            conv = Dropout(0.3)(conv)
            convs.append(conv)
        
        # Multi-scale feature aggregation
        merged = Concatenate()(convs) if len(convs) > 1 else convs[0]

        # 投影降维使形状匹配（残差连接）
        merged_proj = Conv1D(128, 1, padding='same')(merged)

        # 残差卷积块
        res = Conv1D(128, 3, padding='same', activation='relu')(merged_proj)
        res = BatchNormalization()(res)
        res = Add()([merged_proj, res])
        
        # Step 4: Multi-Head Attention Mechanism to enhance long-range dependencies
        attention = MultiHeadAttention(num_heads=8, key_dim=128)(res, res)
        attention = Dropout(0.3)(attention)

        # Step 5: Final feature extraction and classification
        x = GlobalMaxPooling1D()(attention)
        x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        x = Dropout(0.4)(x)
        output = Dense(2, activation='softmax')(x)  # Binary classification

        model = Model(inputs=sequence_input, outputs=output)
        return model


    def Hybrid_cnn_brnn_att(self, x_train, char_index=None):
        input_shape = (x_train.shape[1],)  # input shape for the text data
        sequence_input = Input(shape=input_shape, name='text_input')

        if char_index is not None:
            vocab_size = len(char_index) + 1  # Vocabulary size based on char_index
            x = Embedding(vocab_size, self.embed_dim, input_length=input_shape[0], name="embedding", embeddings_regularizer=regularizers.l2(self.weight_decay))(sequence_input)
        else:
            x = sequence_input  # If using TF-IDF or N-grams, directly use Dense input

        # CNN Layer for Local Feature Extraction
        cnn = Conv1D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        cnn = MaxPooling1D(pool_size=2)(cnn)
        cnn = Dropout(0.3)(cnn)

        # Bidirectional LSTM Layer for Sequential Feature Extraction
        rnn = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(self.weight_decay)))(cnn)
        rnn = Dropout(0.3)(rnn)

        # Attention Layer to Focus on Important Features
        attention = Attention(use_scale=True)([rnn, rnn])  # Using self-attention
        attention = GlobalMaxPooling1D()(attention)  # Apply global max pooling to attention output
        attention = Dropout(0.3)(attention)

        # Global Max Pooling on CNN output
        cnn_pool = GlobalMaxPooling1D()(cnn)

        # Concatenate the CNN and Attention features
        merged = layers.Concatenate()([cnn_pool, attention])  # Concatenate along the feature axis

        # Dense Layers
        x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay))(merged)
        x = Dropout(0.4)(x)
        output = Dense(2, activation='softmax')(x)  # Binary classification (phishing vs. legitimate)

        model = Model(inputs=sequence_input, outputs=output)

        # Compile the model
        
        # model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        
        return model
        
       
    def MGCF_Net(self, text_input_shape, manual_input_shape, char_index=None):
        # 参数
        vocab_size = len(char_index) + 1 if char_index else 50000
        embed_dim = self.embed_dim

        # 输入层
        text_input = Input(shape=text_input_shape, name="text_input")
        manual_input = Input(shape=manual_input_shape, name="manual_input")

        # 嵌入层
        x = Embedding(input_dim=vocab_size,
                    output_dim=embed_dim,
                    input_length=text_input_shape[0],
                    embeddings_regularizer=regularizers.l2(self.weight_decay))(text_input)

        # CNN 分支（局部特征）
        conv = Conv1D(128, 3, activation='relu', padding='same')(x)
        conv = MaxPooling1D(2)(conv)
        conv = GlobalMaxPooling1D()(conv)

        # BiLSTM 分支（全局上下文）
        lstm = Bidirectional(LSTM(128, return_sequences=True))(x)
        lstm = GlobalMaxPooling1D()(lstm)

        # 合并 URL 编码表示
        url_feat = Concatenate()([conv, lstm])
        url_feat = Dense(128, activation='relu')(url_feat)

        # 手动特征处理
        manual_proj = Dense(64, activation='relu')(manual_input)

        query = Dense(64)(manual_proj)
        key = Dense(64)(url_feat)
        value = Dense(64)(url_feat)

        # cross_att = Attention()([query, value])
        cross_att = Attention()([query, value, key])
	cross_att = Flatten()(cross_att)

        # 合并所有特征
        merged = Concatenate()([cross_att, url_feat, manual_proj])
        merged = Dense(128, activation='relu')(merged)
        merged = Dropout(0.4)(merged)
        output = Dense(2, activation='softmax')(merged)

        model = Model(inputs=[text_input, manual_input], outputs=output)
        
        return model


    def MGCF_Net_NoCNN(self, text_input_shape, manual_input_shape, char_index=None):
         # 参数
        vocab_size = len(char_index) + 1 if char_index else 50000
        embed_dim = self.embed_dim

        # 输入层
        text_input = Input(shape=text_input_shape, name="text_input")
        manual_input = Input(shape=manual_input_shape, name="manual_input")

        # 嵌入层
        x = Embedding(input_dim=vocab_size,
                    output_dim=embed_dim,
                    input_length=text_input_shape[0],
                    embeddings_regularizer=regularizers.l2(self.weight_decay))(text_input)

        # BiLSTM 分支（全局上下文）
        lstm = Bidirectional(LSTM(128, return_sequences=True))(x)
        lstm = GlobalMaxPooling1D()(lstm)

        # 合并 URL 编码表示
        url_feat = lstm
        url_feat = Dense(128, activation='relu')(url_feat)

        # 手动特征处理
        manual_proj = Dense(64, activation='relu')(manual_input)

        query = Dense(64)(manual_proj)
        key = Dense(64)(url_feat)
        value = Dense(64)(url_feat)

        # cross_att = Attention()([query, value])
	cross_att = Attention()([query, value, key])
        cross_att = Flatten()(cross_att)

        # 合并所有特征
        merged = Concatenate()([cross_att, url_feat, manual_proj])
        merged = Dense(128, activation='relu')(merged)
        merged = Dropout(0.4)(merged)
        output = Dense(2, activation='softmax')(merged)

        model = Model(inputs=[text_input, manual_input], outputs=output)
        return model


    def MGCF_Net_NoBiLSTM(self, text_input_shape, manual_input_shape, char_index=None):
        # 参数
        vocab_size = len(char_index) + 1 if char_index else 50000
        embed_dim = self.embed_dim

        # 输入层
        text_input = Input(shape=text_input_shape, name="text_input")
        manual_input = Input(shape=manual_input_shape, name="manual_input")

        # 嵌入层
        x = Embedding(input_dim=vocab_size,
                    output_dim=embed_dim,
                    input_length=text_input_shape[0],
                    embeddings_regularizer=regularizers.l2(self.weight_decay))(text_input)

        # CNN 分支（局部特征）
        conv = Conv1D(128, 3, activation='relu', padding='same')(x)
        conv = MaxPooling1D(2)(conv)
        conv = GlobalMaxPooling1D()(conv)

        # 合并 URL 编码表示
        url_feat = conv
        url_feat = Dense(128, activation='relu')(url_feat)

        # 手动特征处理
        manual_proj = Dense(64, activation='relu')(manual_input)

        # Cross Attention 简单实现：query 由手动特征生成
        query = Dense(64)(manual_proj)
        key = Dense(64)(url_feat)
        value = Dense(64)(url_feat)

        cross_att = Attention()([query, value, key])
        cross_att = Flatten()(cross_att)

        # 合并所有特征
        merged = Concatenate()([cross_att, url_feat, manual_proj])
        merged = Dense(128, activation='relu')(merged)
        merged = Dropout(0.4)(merged)
        output = Dense(2, activation='softmax')(merged)

        model = Model(inputs=[text_input, manual_input], outputs=output)
        return model


    def MGCF_Net_NoAttention(self, text_input_shape, manual_input_shape, char_index=None):
        # 参数
        vocab_size = len(char_index) + 1 if char_index else 50000
        embed_dim = self.embed_dim

        # 输入层
        text_input = Input(shape=text_input_shape, name="text_input")
        manual_input = Input(shape=manual_input_shape, name="manual_input")

        # 嵌入层
        x = Embedding(input_dim=vocab_size,
                    output_dim=embed_dim,
                    input_length=text_input_shape[0],
                    embeddings_regularizer=regularizers.l2(self.weight_decay))(text_input)

        # CNN 分支（局部特征）
        conv = Conv1D(128, 3, activation='relu', padding='same')(x)
        conv = MaxPooling1D(2)(conv)
        conv = GlobalMaxPooling1D()(conv)

        # BiLSTM 分支（全局上下文）
        lstm = Bidirectional(LSTM(128, return_sequences=True))(x)
        lstm = GlobalMaxPooling1D()(lstm)

        # 合并 URL 编码表示
        url_feat = Concatenate()([conv, lstm])
        url_feat = Dense(128, activation='relu')(url_feat)

        # 手动特征处理
        manual_proj = Dense(64, activation='relu')(manual_input)

        # 合并所有特征
        merged = Concatenate()([url_feat, manual_proj])
        merged = Dense(128, activation='relu')(merged)
        merged = Dropout(0.4)(merged)
        output = Dense(2, activation='softmax')(merged)

        model = Model(inputs=[text_input, manual_input], outputs=output)
        return model


    def MGCF_Net_NoCNN_BiLSTM(self, text_input_shape, manual_input_shape, char_index=None):
    # 参数
        vocab_size = len(char_index) + 1 if char_index else 50000
        embed_dim = self.embed_dim

        # 输入层
        text_input = Input(shape=text_input_shape, name="text_input")
        manual_input = Input(shape=manual_input_shape, name="manual_input")

        # 手动特征处理
        manual_proj = Dense(64, activation='relu')(manual_input)

        query = Dense(64)(manual_proj)
        key = Dense(64)(manual_proj)
        value = Dense(64)(manual_proj)

        cross_att = Attention()([query, value, key])
        cross_att = Flatten()(cross_att)

        merged = Concatenate()([cross_att, manual_proj])
        merged = Dense(128, activation='relu')(merged)
        merged = Dropout(0.4)(merged)
        output = Dense(2, activation='softmax')(merged)

        model = Model(inputs=[text_input, manual_input], outputs=output)
        return model




    def MGCF_Net_Improved(self, text_input_shape, manual_input_shape, char_index=None):
        # 参数
        vocab_size = len(char_index) + 1 if char_index else 50000
        embed_dim = self.embed_dim

        # 输入层
        text_input = Input(shape=text_input_shape, name="text_input")
        manual_input = Input(shape=manual_input_shape, name="manual_input")

        # 嵌入层
        x = Embedding(input_dim=vocab_size,
                    output_dim=embed_dim,
                    input_length=text_input_shape[0],
                    embeddings_regularizer=regularizers.l2(self.weight_decay))(text_input)

        # 多尺度卷积分支（局部特征）
        convs = []
        for kernel_size in [3, 5, 7]:
            conv = Conv1D(128, kernel_size, activation='relu', padding='same')(x)
            conv = MaxPooling1D(2)(conv)
            conv = GlobalMaxPooling1D()(conv)
            convs.append(conv)
        conv_merged = Concatenate()(convs)

        # BiLSTM 分支（全局上下文）
        lstm = Bidirectional(LSTM(128, return_sequences=True))(x)
        lstm = GlobalMaxPooling1D()(lstm)

        # 合并卷积特征和LSTM特征
        url_feat = Concatenate()([conv_merged, lstm])
        url_feat = Dense(128, activation='relu')(url_feat)

        # 手动特征处理
        manual_proj = Dense(64, activation='relu')(manual_input)

        # 多头注意力机制：增强特征间的关系
        query = Dense(64)(manual_proj)
        key = Dense(64)(url_feat)
        value = Dense(64)(url_feat)

        multihead_attention = MultiHeadAttention(num_heads=4, key_dim=64)(query, key, value)
        multihead_attention = Flatten()(multihead_attention)

        # 合并所有特征
        merged = Concatenate()([multihead_attention, url_feat, manual_proj])
        merged = Dense(128, activation='relu')(merged)
        merged = Dropout(0.4)(merged)

        # 修改 softmax 层的输入，确保输入形状正确
        output = Dense(2, activation='softmax')(merged)

        model = Model(inputs=[text_input, manual_input], outputs=output)
        return model


    def DeepCNN_Light_TFIDF(self, x_train, char_index=None):
        model = Sequential()

        # 如果 x_train 是稀疏矩阵，我们可以直接处理它
        input_shape = x_train.shape[1]  # 这里假设 x_train 是一个稀疏矩阵或一个高维的 TF-IDF 特征
        
        # PCA降维： 如果输入维度过大，可以考虑先做PCA降维。这里设置为维度减少到500维作为示例
        pca = PCA(n_components=500)
        x_train_pca = pca.fit_transform(x_train)
        x_val_pca = pca.transform(x_val)
        x_test_pca = pca.transform(x_test)
        
        print(f"PCA 压缩后的维度: {x_train_pca.shape[1]}")

        # Dense 层 1：用于学习压缩后的特征
        model.add(Dense(128, activation='relu', input_dim=x_train_pca.shape[1], kernel_regularizer=regularizers.l2(self.weight_decay)))
        model.add(Dropout(0.3))  # 添加dropout层防止过拟合

        # Dense 层 2：增加模型的非线性
        model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay)))
        model.add(Dropout(0.4))

        # 输出层：二分类
        model.add(Dense(2, activation='softmax'))

        return model


    def TransformerLight(self, char_index):
        weight_decay = self.weight_decay
        vocab_size = len(char_index) + 1

        embed_dim = self.embed_dim
        seq_len = self.sequence_length

        inputs = Input(shape=(seq_len,), name="input")
        
        # Embedding Layer
        x = Embedding(vocab_size, embed_dim)(inputs)
        
        # Optional: Positional Encoding (可加可不加，根据表现微调)

        # Transformer Encoder Block (轻量版)
        # Multi-head Self Attention
        attn_output = MultiHeadAttention(
            num_heads=2,
            key_dim=embed_dim // 2,
            dropout=0.1
        )(x, x)

        # Add & Norm
        x = Add()([x, attn_output])
        x = LayerNormalization(epsilon=1e-6)(x)

        # FeedForward
        ff = Dense(embed_dim * 2, activation='relu',
                kernel_regularizer=regularizers.l2(weight_decay))(x)
        ff = Dropout(0.3)(ff)
        ff = Dense(embed_dim, kernel_regularizer=regularizers.l2(weight_decay))(ff)

        # Add & Norm again
        x = Add()([x, ff])
        x = LayerNormalization(epsilon=1e-6)(x)

        # Pooling
        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.4)(x)

        # Output
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.4)(x)
        outputs = Dense(2, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model


    def TransformerLightV2(self, char_index):
        import tensorflow as tf
        from tensorflow.keras.layers import (
            Input, Embedding, Dense, Dropout, LayerNormalization,
            MultiHeadAttention, GlobalAveragePooling1D, Add
        )
        from tensorflow.keras.models import Model
        from tensorflow.keras import regularizers

        weight_decay = self.weight_decay
        vocab_size = len(char_index) + 1
        embed_dim = self.embed_dim
        seq_len = self.sequence_length

        # Input
        inputs = Input(shape=(seq_len,), name="input")

        # Word Embedding
        token_embed = Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)

        # Positional Embedding
        pos_inputs = tf.range(start=0, limit=seq_len, delta=1)
        pos_embed = Embedding(input_dim=seq_len, output_dim=embed_dim)(pos_inputs)
        x = Add()([token_embed, pos_embed])

        # Transformer Block 1
        attn1 = MultiHeadAttention(num_heads=2, key_dim=embed_dim)(x, x)
        x = Add()([x, attn1])
        x = LayerNormalization(epsilon=1e-6)(x)

        ffn1 = Dense(embed_dim * 2, activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(x)
        ffn1 = Dense(embed_dim, kernel_regularizer=regularizers.l2(weight_decay))(ffn1)
        x = Add()([x, ffn1])
        x = LayerNormalization(epsilon=1e-6)(x)

        # Transformer Block 2
        attn2 = MultiHeadAttention(num_heads=2, key_dim=embed_dim)(x, x)
        x = Add()([x, attn2])
        x = LayerNormalization(epsilon=1e-6)(x)

        ffn2 = Dense(embed_dim * 2, activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(x)
        # ffn2 = Dense(embed_dim, kernel_regularizer=regularizers.l2(weight_decay))(ffn2)
        ffn2 = Dense(embed_dim, kernel_regularizer=regularizers.l2(weight_decay))(attention_out)

        x = Add()([x, ffn2])
        x = LayerNormalization(epsilon=1e-6)(x)

        # Pooling
        x = GlobalAveragePooling1D()(x)

        # Classifier Head
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)

        outputs = Dense(2, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model
