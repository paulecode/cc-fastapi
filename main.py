from helper.chord_extract import chord_extract
from helper.midi_processor import midi_preprocess
import librosa
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse

from helper.note_name import get_note_name

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello World"}


def preprocessFile(file: UploadFile):
    note_df, meta_df = midi_preprocess(file)
    print("Done processing")
    note_df = get_note_name(note_df)
    chord_df = chord_extract(note_df)

    return chord_df


@app.post("/predictMidi")
async def predict(file: UploadFile):
    chord_df = preprocessFile(file)

    return {"chords": chord_df.reset_index().to_dict(orient="records")}

@app.post("/predictWav")
async def predictWav(file: UploadFile):
    print("Received file in wav")
    y, sr = librosa.load(file.file)

    rms = librosa.feature.rms(y=y)

    return {"rms": rms.flatten().tolist()}
