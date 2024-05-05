from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import uvicorn

# Créer une instance FastAPI
app = FastAPI()

# Charger le modèle et les mappings
model = load_model('entity_recognition_model.keras')
model.load_weights('entity_recognition.weights.h5')

with open('tokens_to_index.pickle', 'rb') as f:
    token2idx = pickle.load(f)

with open('tag_to_index.pickle', 'rb') as f:
    tag2idx = pickle.load(f)

with open('index_to_tag.pickle', 'rb') as f:
    idx2tag = pickle.load(f)

# Définir la structure des données attendues dans la requête
class InputData(BaseModel):
    sentence: str

# Définir la route pour la prédiction
@app.post('/predict')
async def predict(data: InputData):

    # Récupérer la phrase depuis les données de la requête
    sentence = data.sentence

    # Prétraiter la phrase
    tokens, processed_sentence = preprocess_sentence(sentence)
    # Prédire les étiquettes
    predicted_tags = model.predict(processed_sentence)
    # Convertir les indices en étiquettes
    predicted_tags = np.argmax(predicted_tags, axis=-1)

    sentence_length = len(tokens)

    predicted_tags = predicted_tags[0][:sentence_length]

    tags = [idx2tag[idx] for idx in predicted_tags]

    print("{:15}{:5}\n".format("Word","Pred"))
    print("-"*30)

    result = ""

    for w,pred in zip(tokens, tags):
        result+=str(w) + "\t" + str(pred)
        
    return result

# Fonction de prétraitement de la phrase
def preprocess_sentence(sentence):
    tokens = sentence.split()
    token_ids = [token2idx.get(token, 0) for token in tokens]
    token_ids_padded = np.pad(token_ids, (0, 104 - len(token_ids)), 'constant')
    return tokens, np.array([token_ids_padded])

if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8000)