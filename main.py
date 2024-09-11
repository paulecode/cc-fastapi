from helper.chord_extract import chord_extract
from helper.midi_processor import midi_preprocess

from fastapi import FastAPI, UploadFile

from helper.note_name import get_note_name

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello World"}


def preprocessFile(file: UploadFile):
    print(file.content_type)

    note_df, meta_df = midi_preprocess(file)
    print("Done processing")
    note_df = get_note_name(note_df)
    chord_df = chord_extract(note_df)

    return chord_df


@app.post("/predict")
async def predict(file: UploadFile):
    chord_df = preprocessFile(file)
    return {"chords": chord_df.to_dict()}
