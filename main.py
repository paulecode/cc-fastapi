import mido
import pickle
from sklearn.ensemble import RandomForestClassifier
from fastapi import BackgroundTasks, FastAPI, UploadFile

app = FastAPI()

model_filename = 'random_forest_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

@app.get("/")
def read_root():
    return {"message": "Hello World"}


@app.get("/express")
def read_express():
    print("Test")
    return {"message": "Live v3"}

def preprocessFile(file: UploadFile):
    print(file.content_type)
    note_count = 0
    midi_file = mido.MidiFile(file=file.file)
    for msg in midi_file:
        if msg.type == 'note_on':
            note_count += 1
    prediction = model.predict([[note_count]])

    print(prediction)
    return prediction

@app.post("/predict")
async def predict(file: UploadFile, background_tasks: BackgroundTasks):
    if not file:
        raise Exception("File missing")

    background_tasks.add_task(preprocessFile, file)
    return {"message": file.content_type}
