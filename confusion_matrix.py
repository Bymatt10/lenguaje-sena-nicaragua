import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from helpers import get_word_ids, get_sequences_and_labels
from constants import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def generate_confusion_matrix():
    '''
    ### MATRIZ DE CONFUSIÓN
    Evalúa el modelo con todos los keypoints guardados en `data/keypoints`
    y muestra la matriz de confusión por palabra.
    '''
    word_ids = get_word_ids(WORDS_JSON_PATH)
    model = load_model(MODEL_PATH)

    sequences, labels = get_sequences_and_labels(word_ids)
    sequences = pad_sequences(sequences, maxlen=int(MODEL_FRAMES), padding='pre', truncating='post', dtype='float32')

    predictions = model.predict(np.array(sequences), verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    conf_matrix = confusion_matrix(labels, predicted_labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=word_ids)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    generate_confusion_matrix()
