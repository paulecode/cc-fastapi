import io
import os
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

NEXT_URL = os.environ.get("NEXT_PUBLIC_API_URL", "http://localhost:3000")


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
    """
    Process a DataFrame of MIDI data using mido to extract chord information.

    Args:
        df (pd.DataFrame): DataFrame containing MIDI data.

    Returns:
        pd.DataFrame: DataFrame containing chord information.
    """
    print("Done processing")
    note_df = get_note_name(df)
    chord_df = chord_extract(df)

    return chord_df


def midiBGTASK(file: bytes, userId: int, filename: str | None):
    """
    Background task for processing MIDI files and sending results back to nextjs.

    Args:
        file (bytes): MIDI file content.
        userId (int): ID of the user who uploaded the file.
        filename (str | None): Name of the file.
    """
    note_df, _ = midi_preprocess(file)
    chord_df = chordProcess(note_df)

    note_df = note_df[note_df["velocity"] != 0]

    X = np.array(note_df["note"].values).reshape(1, -1)
    X = np.array(librosa.util.fix_length(X, size=50152).reshape(1, -1))

    composer = ml_models["composer_midi"].predict(X)
    genre = ml_models["form_midi"].predict(X)

    response = httpx.post(
        f"{NEXT_URL}/api/postMidiResult",
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
    """
    Endpoint to handle MIDI uploads and start background processing.

    Args:
        background_tasks (BackgroundTasks): FastAPI background tasks manager.
        file (UploadFile): Uploaded MIDI file.
        userId (int): ID of the user.

    Returns:
        Response: HTTP response with status code 200.
    """
    file_content = await file.read()
    filename = file.filename
    print("File received: ", filename)
    background_tasks.add_task(midiBGTASK, file_content, userId, filename)

    return Response(status_code=200)


def trim_audio(y: np.ndarray, sr: int, start_time: int, end_time: int) -> np.ndarray:
    """
    Trim wav file

    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of the audio.
        start_time (int): Start time in seconds.
        end_time (int): End time in seconds.

    Returns:
        np.ndarray: Trimmed audio signal.
    """
    start_at = int(start_time * sr)
    stop_at = int(end_time * sr)
    return y[start_at:stop_at]


def wavBGTASK(file: bytes, filename: str | None, userId: int):
    """
    Background task for processing WAV files and sending results to the Next APi.

    Args:
        file (bytes): WAV file content.
        filename (str | None): Name of the file.
        userId (int): ID of user who uploaded the file.
    """
    with io.BytesIO(file) as f:
        y, sr = librosa.load(f)
        rms = librosa.feature.rms(y=y)
        spectogram = librosa.feature.melspectrogram(y=y, sr=sr)
        X = np.array(librosa.util.fix_length(rms, size=113222).reshape(1, -1))
        composer = ml_models["composer_wav"].predict(X)
        genre = ml_models["form_wav"].predict(X)

    response = httpx.post(
        f"{NEXT_URL}/api/postWavResult",
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
    """
    Endpoint to handle WAV file uploads and start background processing.

    Args:
        background_tasks (BackgroundTasks): FastAPI background tasks manager.
        file (UploadFile): Uploaded WAV file.
        userId (int): ID of the user.
    """
    file_content = await file.read()
    filename = file.filename
    print("File received: ", filename)
    background_tasks.add_task(wavBGTASK, file_content, filename, userId)

    return Response(status_code=200)
