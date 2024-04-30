import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, LayerNormalization, MultiHeadAttention, Layer
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('powerball.csv')

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Function to convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # Input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # Forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # Put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# Prepare the data
reframed = series_to_supervised(scaled_data, 3, 1)
values = reframed.values
train = values[:-10, :]
test = values[-10:, :]
X_train, y_train = train[:, :-6], train[:, -6:]
X_test, y_test = test[:, :-6], test[:, -6:]

# Reshape input to be 3D [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], 3, 6))
X_test = X_test.reshape((X_test.shape[0], 3, 6))

# Transformer block
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Dense(ff_dim, activation="relu"), Dense(embed_dim)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Build the model
def build_model(input_shape):
    inputs = Input(shape=input_shape)
    x = TransformerBlock(6, 4, 32)(inputs)  # Adjust the embed_dim to match input features
    x = Dense(20, activation='relu')(x)
    outputs = Dense(6, activation='sigmoid')(x)  # Adjust activation to match the scaling of output
    model = Model(inputs=inputs, outputs=outputs)
    return model

model = build_model((3, 6))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2)

# Make predictions
yhat = model.predict(X_test)
yhat = scaler.inverse_transform(yhat)  # Inverse scaling to get actual predictions

# Print multiple sets of predictions
print('Predicted sets of numbers:')
for i in range(3):  # Change this to generate more or fewer predictions
    print(f'Set {i+1}: {yhat[i]}')
