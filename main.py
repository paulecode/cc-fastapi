import io
import pickle

import httpx
import numpy as np
import pandas as pd
from helper.chord_extract import chord_extract
from helper.midi_processor import midi_preprocess
import librosa
from fastapi import FastAPI, File, Form, Response, UploadFile, BackgroundTasks

from helper.note_name import get_note_name

ml_models = {}


async def lifespan(app: FastAPI):
    # Load the models
    with open("models/rf_midi.pkl", "rb") as f:
        models = pickle.load(f)
        ml_models["form_midi"] = models["form_model"]
        ml_models["composer_midi"] = models["composer_model"]

    with open("models/rf_wav.pkl", "rb") as f:
        models = pickle.load(f)
        ml_models["form_wav"] = models["form_model"]
        ml_models["composer_wav"] = models["composer_model"]

    yield
    f.close()


app = FastAPI(lifespan=lifespan)  # type: ignore


@app.get("/")
def read_root():
    return {"message": "Hello World"}


def chordProcess(df: pd.DataFrame):
    print("Done processing")
    note_df = get_note_name(df)
    chord_df = chord_extract(df)

    return chord_df


def midiBGTASK(file: bytes, userId: int, filename: str | None):
    note_df, _ = midi_preprocess(file)
    chord_df = chordProcess(note_df)

    note_df = note_df[note_df["velocity"] != 0]

    X = np.array(note_df["note"].values).reshape(1, -1)
    X = np.array(librosa.util.fix_length(X, size=50152).reshape(1, -1))

    composer = ml_models["composer_midi"].predict(X)
    genre = ml_models["form_midi"].predict(X)

    response = httpx.post(
        "http://localhost:3000/api/postMidiResult",
        json={
            "meta": {"userId": userId, "filename": filename},
            "visualization": {
                "chords": chord_df.reset_index().to_dict(orient="records"),
                "notes": note_df["note"].to_list(),
                "timestamps": note_df["time"].tolist(),
                "velocity": note_df["velocity"].tolist(),
            },
            "classification": {"genre": genre[0], "composer": composer[0]},
        },
    )


@app.post("/predictMidi")
async def predict(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    userId: int = Form(...),
):
    file_content = await file.read()
    filename = file.filename
    background_tasks.add_task(midiBGTASK, file_content, userId, filename)

    return Response(status_code=200)


def wavBGTASK(file: bytes, filename: str | None, userId: int):
    with io.BytesIO(file) as f:
        y, sr = librosa.load(f)
        rms = librosa.feature.rms(y=y)
        spectogram = librosa.feature.melspectrogram(y=y, sr=sr)
        X = np.array(librosa.util.fix_length(rms, size=113222).reshape(1, -1))
        composer = ml_models["composer_wav"].predict(X)
        genre = ml_models["form_wav"].predict(X)

    response = httpx.post(
        "http://localhost:3000/api/postWavResult",
        json={
            "meta": {"filename": filename, "userId": userId},
            "visualization": {
                "rms": rms[0].flatten().tolist(),
                "spectogram": spectogram.flatten().tolist(),
            },
            "classification": {"composer": composer[0], "genre": genre[0]},
        },
    )


@app.post("/predictWav")
async def predictWav(
    background_tasks: BackgroundTasks, file: UploadFile, userId: int = Form(...)
):
    file_content = await file.read()
    filename = file.filename
    background_tasks.add_task(wavBGTASK, file_content, filename, userId)

    return Response(status_code=200)
