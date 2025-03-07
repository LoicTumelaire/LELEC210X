import pickle
from keras import models
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

import click

import common
from auth import PRINT_PREFIX
from common.env import load_dotenv
from common.logging import logger

from .utils import payload_to_melvecs

import requests
import json

pathFile = Path(__file__).resolve()
model_dir = str(pathFile)[:-11]+"/../../data/models/"
mel_dir = str(pathFile)[:-11]+"/../../data/melspecs/"

load_dotenv()

send = True
#hostname = "http://localhost:5000"
hostname = "http://lelec210x.sipr.ucl.ac.be" # Contest: http://lelec210x.sipr.ucl.ac.be/lelec210x/leaderboard
#key = "dhdCGK4Xq7EKm-U9Ji1MAHYvPyWBqoimYAU4pknY"
key = "EPHNDFX0Y_aie6lb6trPdTrw_ob8Gc8yNzIpusWF" # Contest


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
def main(
    _input: Optional[click.File],
    model: Optional[Path],
    melvec_length: int,
    n_melvecs: int,
) -> None:
    """
    Extract Mel vectors from payloads and perform classification on them.
    Classify MELVECs contained in payloads (from packets).

    Most likely, you want to pipe this script after running authentification
    on the packets:

        rye run auth | rye run classify

    This way, you will directly receive the authentified packets from STDIN
    (standard input, i.e., the terminal).
    """
    model = models.load_model(model_dir + "two.keras")
    for payload in _input:
        if PRINT_PREFIX in payload:
            payload = payload[len(PRINT_PREFIX) :]

            melvecs = payload_to_melvecs(payload, melvec_length, n_melvecs)
            logger.info(f"Parsed payload into Mel vectors: {melvecs}")

            classify = True
            # If the melvecs has too much noise, we don't classify it, TODO: add in the MCU
            if np.max(melvecs) < 1e-3:
                classify = False

            if classify:
                # Perform classification

                # Flip the melvecs for the frequencies and reshape it for the model
                #melvecs = np.flipud(melvecs)
                melvecs = melvecs[None,...]

                # Plot the melvecs !! Plot à l'envers
                plt.imshow(melvecs[0,:], origin='lower')
                plt.show()

                # Save the melvecs into a file
                filename = mel_dir + payload[:10] + ".npy"
                #np.save(filename, melvecs)
                print(melvecs)
                y_pred = model.predict(melvecs) # [[0.1, 0.2, 0.3, 0.4, 0.5]]

                classes = ["birds", "chainsaw", "fire", "handsaw", "helicopter"]

                guess = classes[int(np.argmax(y_pred))]

                print(f"Prediction: {guess}")
                print(f"Probabilities: {y_pred}")

                # Send to the server if the probabilities are high enough
                if send and np.max(y_pred) > 0.5:
                    response = requests.post(f"{hostname}/lelec210x/leaderboard/submit/{key}/{guess}", timeout=1)
                    # All responses are JSON dictionaries
                    response_as_dict = json.loads(response.text)
                    print(response_as_dict)
