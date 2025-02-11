from .utils import payload_to_melvecs
import click
import common
from pathlib import Path
from typing import Optional
from auth import PRINT_PREFIX
from common.env import load_dotenv
from common.logging import logger
import numpy as np
import os
import matplotlib.pyplot as plt
from keras import models

pathFile = Path(__file__).resolve()
mel_dir = str(pathFile)[:-11]+"/../../data/melspecs/"
model_dir = str(pathFile)[:-11]+"/../../data/models/"
model = models.load_model(model_dir + "two.keras")

@click.command()
@click.option(
    "-i",
    "--input",
    "_input",
    default="-",
    type=click.File("r"),
    help="Where to read the input stream. Default to '-', a.k.a. stdin.",
)
@click.option(
    "-m",
    "--model",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the trained classification model.",
)
@common.click.melvec_length
@common.click.n_melvecs
@common.click.verbosity

def record(
    _input: Optional[click.File],
    model: Optional[Path],
    melvec_length: int,
    n_melvecs: int,
) -> None:
    
    """Record the recieved melspectrogram into a file"""

    for payload in _input:
        if PRINT_PREFIX in payload:
            payload = payload[len(PRINT_PREFIX) :]

            melvecs = payload_to_melvecs(payload, melvec_length, n_melvecs)
            
            # Save the melvecs into a file
            filename = mel_dir + payload[:10] + ".npy"
            np.save(filename, melvecs)


def classify():
    """Classify the recieved melspectrogram"""

    classes = ["birds", "chainsaw", "fire", "handsaw", "helicopter"]

    for file in os.listdir(mel_dir):
        melvec = np.load(mel_dir + file)

        # First model
        melvecs = np.rot90(melvec, k=1, axes=(0, 1))
        melvecs = melvecs.reshape(20,20,1).T
        plt.imshow(melvecs[0])
        plt.show()

        y_pred = model.predict(melvecs) # [[0.1, 0.2, 0.3, 0.4, 0.5]]
        guess = classes[int(np.argmax(y_pred))]

        print(f"Prediction: {guess}")
        print(f"Probabilities: {y_pred}")

        # Second model

record()
classify()