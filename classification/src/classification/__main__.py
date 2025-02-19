from keras import models #import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib.pyplot import imshow, show #import matplotlib.pyplot as plt

import click

import common
from auth import PRINT_PREFIX
from common.env import load_dotenv
from common.logging import logger

from .utils import payload_to_melvecs

from requests import post
from json import loads

pathFile = Path(__file__).resolve()
model_dir = str(pathFile)[:-11]+"/../../data/models/"
mel_dir = str(pathFile)[:-11]+"/../../data/melspecs/"

load_dotenv()

###################################
# Variables globales
###################################
send = True
save = False
DEBUG = False

# Adresses + keys
#hostname = "http://localhost:5000"
hostname = "http://lelec210x.sipr.ucl.ac.be" # Contest: http://lelec210x.sipr.ucl.ac.be/lelec210x/leaderboard
#key = "dhdCGK4Xq7EKm-U9Ji1MAHYvPyWBqoimYAU4pknY"
key = "EPHNDFX0Y_aie6lb6trPdTrw_ob8Gc8yNzIpusWF" # Contest

model = models.load_model(model_dir + "two.keras")
classes = ["birds", "chainsaw", "fire", "handsaw", "helicopter"]
history = np.zeros((5,5)) # History of max len of 5 y_pred = [0.1, 0.2, 0.3, 0.4, 0.5], if we add more, the first one is removed

def exponentialWeight(number, expo=0.1):
    """
    Calculate the exponential weight of a given number.
    Parameters:
        number (float): The input number to be weighted.
        expo (float, optional): The exponent value to be used in the calculation. Default is 0.1.
    Returns:
        float: The exponential weight of the input number.
    """
    return np.exp(-expo*number)

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
    for payload in _input:
        if PRINT_PREFIX in payload:
            payload = payload[len(PRINT_PREFIX) :]

            melvecs = payload_to_melvecs(payload, melvec_length, n_melvecs)
            logger.info(f"Parsed payload into Mel vectors: {melvecs}")

            # If the melvecs has too much noise, we don't classify it, 
            # TODO: add in the MCU: don't send it
            noise = False
            if np.max(melvecs) < 1e-3:
                noise = True

            if not noise:
                # Perform classification

                melvecs = melvecs[None,...] # Add a dimension to the melvecs for the model

                if DEBUG:
                    # Plot the melvecs !! Plot à l'envers
                    imshow(melvecs[0,:], origin='lower')
                    show()

                # Save the melvecs into a file
                if save:
                    filename = mel_dir + payload[:10] + ".npy"
                    np.save(filename, melvecs)

                # Predict the class probabilities
                y_pred = model.predict(melvecs) # [0.1, 0.2, 0.3, 0.4, 0.5]

                # Add the prediction to the history and delete the last one
                history = np.roll(history, 1, axis=0)
                history[0] = y_pred
                
                # Smooth the prediction
                smoothed_pred = np.zeros(5)
                for i in range(5):
                    smoothed_pred += history[i]*exponentialWeight(i)
                smoothed_pred /= np.sum(smoothed_pred)

                # Get the most probable class
                guess = classes[int(np.argmax(smoothed_pred))]

                if DEBUG:
                    print(f"Probabilities: {y_pred}")
                    print(f"Prediction: {guess}")

                # Send to the server if the probabilities are high enough
                if send and np.max(y_pred) > 0.5:
                    print("Sending to the server")
                    response = post(f"{hostname}/lelec210x/leaderboard/submit/{key}/{guess}", timeout=1)
                    # All responses are JSON dictionaries
                    response_as_dict = loads(response.text)
                    print(response_as_dict)

