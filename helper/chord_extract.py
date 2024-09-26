import pandas as pd
from music21 import chord


def chord_extract(df: pd.DataFrame):
    """
    Extracts chords from a dataframe of notes
    """

    chord_df = df.groupby("time")["note_name"].agg(list)

    chord_df = chord_df[chord_df.apply(lambda x: len(x) > 2)].to_frame()

    chord_df.rename(columns={"note_name": "chord"}, inplace=True)

    chord_df["chord_name"] = chord_df["chord"].apply(
        lambda x: chord.Chord(x).commonName
    )

    chord_df["root"] = chord_df["chord"].apply(lambda x: chord.Chord(x).root().name)

    chord_df = chord_df

    # print(chord_df.reset_index())

    return chord_df
