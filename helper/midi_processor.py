import mido
import os

from fastapi import UploadFile
import pandas as pd


def midi_preprocess(file: UploadFile):
    print("Got here")
    mid = mido.MidiFile(file=file.file)

    meta_data_list = []
    note_list = []

    for track_index, track in enumerate(mid.tracks):
        absolute_time = 0
        for message_index, message in enumerate(track):
            absolute_time += message.time
            if message.is_meta:
                meta_message = message.dict()
                if len(meta_message) == 2:
                    meta_data_list.append(
                        [track_index, absolute_time, str(message.type), "-"]
                    )
                elif len(meta_message) == 3:
                    meta_message.pop("type")
                    meta_message.pop("time")
                    for key, value in meta_message.items():
                        meta_data_list.append(
                            [
                                track_index,
                                absolute_time,
                                str(message.type),
                                str(value),
                            ]
                        )
                else:
                    meta_message.pop("type")
                    meta_message.pop("time")
                    for key, value in meta_message.items():
                        meta_data_list.append(
                            [
                                track_index,
                                absolute_time,
                                f"{message.type}_{key}",
                                str(value),
                            ]
                        )
            elif message.type == "control_change":
                if "control=64" in str(message):
                    meta_message = message.dict()
                    meta_data_list.append(
                        [
                            track_index,
                            absolute_time,
                            "pedal",
                            meta_message["value"],
                        ]
                    )
            elif message.type == "note_on" or message.type == "note_off":
                meta_message = message.dict()
                note_list.append(
                    [
                        track_index,
                        absolute_time,
                        message.type,
                        meta_message["note"],
                        meta_message["velocity"],
                    ]
                )
        print("Done")
        note_df = pd.DataFrame(
            note_list,
            columns=["track", "time", "type", "note", "velocity"],
        )
        meta_df = pd.DataFrame(
            meta_data_list,
            columns=["track", "time", "key", "value"],
        )
        print(note_df.head())
    return note_df, meta_df
