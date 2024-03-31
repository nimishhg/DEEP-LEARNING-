# DEEP-LEARNING-
MEDICAL DIAGNOSTIC ASSISTANCE TOOL 
BY- ARYAN SINGHAL (6629) AND NIMISH GOYAL (6634)

CODE - 

//Importing Libraries
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

// Loading Dataset
data = pd.read_csv('https://raw.githubusercontent.com/adil200/Medical-Diagnoser/main/medical_data.csv')
data.head()

// Data Preprocessing and Preparation
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(data['Patient_Problem'])

sequences = tokenizer.texts_to_sequences(data['Patient_Problem'])

// Padding Sequences
max_length = max(len(x) for x in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

// Encoding the labels
label_encoder_disease = LabelEncoder()
label_encoder_prescription = LabelEncoder()

disease_labels = label_encoder_disease.fit_transform(data['Disease'])
prescription_labels = label_encoder_prescription.fit_transform(data['Prescription'])

// Converting labels to categorical
disease_labels_categorical = to_categorical(disease_labels)
prescription_labels_categorical = to_categorical(prescription_labels)

// Combining Labels into a Multi-label Target Variable
Y = np.hstack((disease_labels_categorical, prescription_labels_categorical))

// Model Building
input_layer = Input(shape=(max_length,))

embedding = Embedding(input_dim=5000, output_dim=64)(input_layer)
lstm_layer = LSTM(64)(embedding)

disease_output = Dense(len(label_encoder_disease.classes_), activation='softmax',
                       name='disease_output')(lstm_layer)

prescription_output = Dense(len(label_encoder_prescription.classes_),
                            activation='softmax', name='prescription_output')(lstm_layer)

// Compiling the model
model = Model(inputs=input_layer, outputs=[disease_output, prescription_output])

model.compile(
    loss={'disease_output': 'categorical_crossentropy',
          'prescription_output': 'categorical_crossentropy'},
    optimizer='adam',
    metrics={'disease_output': ['accuracy'], 'prescription_output': ['accuracy']}
)

model.summary()

// Training the model
model.fit(padded_sequences, {'disease_output': disease_labels_categorical, 'prescription_output':
    prescription_labels_categorical}, epochs=100, batch_size=32)


// Making Predictions
def make_prediction(patient_problem):
    # Preprocessing the input
    sequence = tokenizer.texts_to_sequences([patient_problem])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

    # Making prediction
    prediction = model.predict(padded_sequence)

    # Decoding the prediction
    disease_index = np.argmax(prediction[0], axis=1)[0]
    prescription_index = np.argmax(prediction[1], axis=1)[0]

    disease_predicted = label_encoder_disease.inverse_transform([disease_index])[0]
    prescription_predicted = label_encoder_prescription.inverse_transform([prescription_index])[0]

    print(f"Predicted Disease: {disease_predicted}")
    print(f"Suggested Prescription: {prescription_predicted}")


patient_input = "I've experienced a loss of appetite and don't enjoy food anymore."
make_prediction(patient_input)

