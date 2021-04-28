# Basic Flask Python Web App
from flask import Flask
from flask_socketio import SocketIO, emit

import numpy as np
from transformers import AutoTokenizer
import tensorflow as tf
import keras
import sys



seq_len = 50
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

def tokenize(sentence):
    tokens = tokenizer.encode_plus(sentence, max_length=seq_len,
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_attention_mask=True,
                                   return_token_type_ids=False, return_tensors='tf')
    return tokens['input_ids'], tokens['attention_mask']


from transformers import TFAutoModel

bert = TFAutoModel.from_pretrained('bert-base-cased')


input_ids = tf.keras.layers.Input(shape=(seq_len,), name='input_ids', dtype='int32')
mask = tf.keras.layers.Input(shape=(seq_len,), name='attention_mask', dtype='int32')

embeddings = bert(input_ids, attention_mask=mask)[0]

X = tf.keras.layers.GlobalMaxPool1D()(embeddings)
X = tf.keras.layers.BatchNormalization()(X)
X = tf.keras.layers.Dense(128, activation='elu')(X)
X = tf.keras.layers.Dropout(0.1)(X)
X = tf.keras.layers.Dense(64, activation='elu')(X)
X = tf.keras.layers.Dropout(0.1)(X)
X = tf.keras.layers.Dense(32, activation='elu')(X)
y = tf.keras.layers.Dense(1, activation='elu', name='outputs')(X)

model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)

model.layers[2].trainable = False

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss=keras.losses.mean_squared_error)

model.load_weights("moddy.h5")








app = Flask(__name__, static_url_path='', static_folder='', template_folder='')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)


@socketio.on('text_to_server')
def test_message(message):
	print(message, file=sys.stderr)
	print(message['data'], file=sys.stderr)


	output = model.predict(tokenize(message['data']))
	
	print(output, file=sys.stderr)
	print(output[0][0], file=sys.stderr)

	emit('text_to_client', {'data': float(output[0][0])})
                                                  
@app.route("/")
def hello():
	return app.send_static_file('index.html')

if __name__ == "__main__":
    app.run()
    socketio.run(app)