from pathlib import Path
from typing import Optional

import tensorflow as tf

import numpy as np  
from matplotlib.pyplot import imshow, show #import matplotlib.pyplot as plt

from requests import post
from json import loads
import os
import struct

PRINT_PREFIX = "DF:HEX:"
MELVEC_LENGTH = 30
N_MELVECS = 20

def normalize(x):
    return x / tf.reduce_max(x)

def payload_to_melvecs(
    payload: str, melvec_length: int = MELVEC_LENGTH, n_melvecs: int = N_MELVECS
) -> np.ndarray:
    """Convert a payload string to a melvecs array."""
    fmt = f"!{melvec_length}h"
    buffer = bytes.fromhex(payload.strip())
    unpacked = struct.iter_unpack(fmt, buffer)
    melvecs_q15int = np.asarray(list(unpacked), dtype=np.int16)
    melvecs = melvecs_q15int.astype(float)
    melvecs = np.rot90(melvecs, k=-1, axes=(0, 1))
    return melvecs

pathFile = Path(__file__).resolve()
model_dir = str(pathFile)[:-11]+'..\\..\\data\\models\\'
mel_dir = str(pathFile)[:-11]+'..\\..\\data\\melspecs\\'

###################################
# Variables globales
###################################
send = True
save = False
DEBUG = True

# Adresses + keys
#hostname = "http://localhost:5000"
hostname = "http://lelec210x.sipr.ucl.ac.be" # Contest: http://lelec210x.sipr.ucl.ac.be/lelec210x/leaderboard
#key = "dhdCGK4Xq7EKm-U9Ji1MAHYvPyWBqoimYAU4pknY"
key = "EPHNDFX0Y_aie6lb6trPdTrw_ob8Gc8yNzIpusWF" # Contest


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


def main(
    _input: list[str],
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
    
    print(os.path.exists(model_dir + "four.keras"))

    model = tf.keras.models.load_model(model_dir + "four.keras", custom_objects={'normalize': normalize}) # BUG four.keras can't load
    classes = ['chainsaw', 'fire', 'fireworks', 'gunshot']
    history = np.zeros((5,5)) # History of max len of 5 y_pred = [0.1, 0.2, 0.3, 0.4, 0.5], if we add more, the first one is removed

    for payload in _input:
        if PRINT_PREFIX in payload:
            payload = payload[len(PRINT_PREFIX) :]

            melvecs = payload_to_melvecs(payload, melvec_length, n_melvecs)

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
                    print(f"Sending to the server:{guess}")
                    response = post(f"{hostname}/lelec210x/leaderboard/submit/{key}/{guess}", timeout=1)
                    # All responses are JSON dictionaries
                    response_as_dict = loads(response.text)
                    print(response_as_dict)

