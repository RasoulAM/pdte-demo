"""Server that will listen for GET and POST requests from the client."""

import time
from typing import List

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response
from utils import DEPLOYMENT_DIR, SERVER_DIR  # pylint: disable=no-name-in-module
from utils import PARAM_FILE_PATH, KEYS_AND_CT_FILE_PATH, CT_SIZE_PATH, PK_FILE_PATH, SK_FILE_PATH

from concrete.ml.deployment import FHEModelServer
import os
import subprocess

# Load the FHE server
FHE_SERVER = FHEModelServer(DEPLOYMENT_DIR)

# Initialize an instance of FastAPI
app = FastAPI()

# Define the default route
@app.get("/")
def root():
    """
    Root endpoint of the health prediction API.

    Returns:
        dict: The welcome message.
    """
    return {"message": "Welcome to your disease prediction with FHE!"}


@app.post("/send_input")
def send_input(
    user_id: str = Form(),
    files: List[UploadFile] = File(),
):
    """Send the inputs to the server."""

    print("\nSend the data to the server ............\n")

    # Receive the Client's files (Evaluation key + Encrypted symptoms)
    evaluation_key_path = SERVER_DIR / f"{user_id}_valuation_key"
    encrypted_input_path = SERVER_DIR / f"{user_id}_encrypted_input"
    
    pdte_params_path = SERVER_DIR / f"{PARAM_FILE_PATH}_{user_id}"
    keys_and_ct_path = SERVER_DIR / f"{KEYS_AND_CT_FILE_PATH}_{user_id}"
    ct_size_path = SERVER_DIR / f"{CT_SIZE_PATH}_{user_id}"
    pk_path = SERVER_DIR / f"{PK_FILE_PATH}_{user_id}"
    # sk_file_path = SERVER_DIR / f"{SK_FILE_PATH}_{user_id}"

    # Save the files using the above paths
    with encrypted_input_path.open("wb") as encrypted_input,\
            evaluation_key_path.open("wb") as evaluation_key,\
            pdte_params_path.open("wb") as pdte_params,\
            keys_and_ct_path.open("wb") as keys_and_ct,\
            ct_size_path.open("wb") as ct_size,\
            pk_path.open("wb") as pk:
            # sk_file_path.open("wb") as sk_file:
        encrypted_input.write(files[0].file.read())
        evaluation_key.write(files[1].file.read())
        pdte_params.write(files[2].file.read())
        keys_and_ct.write(files[3].file.read())
        ct_size.write(files[4].file.read())
        pk.write(files[5].file.read())
        # sk_file.write(files[6].file.read())


@app.post("/run_fhe")
def run_fhe(
    user_id: str = Form(),
):
    """Inference in FHE."""

    print("\nRun in FHE in the server ............\n")
    evaluation_key_path = SERVER_DIR / f"{user_id}_valuation_key"
    encrypted_input_path = SERVER_DIR / f"{user_id}_encrypted_input"

    # Read the files (Evaluation key + Encrypted symptoms) using the above paths
    with encrypted_input_path.open("rb") as encrypted_output_file, evaluation_key_path.open(
        "rb"
    ) as evaluation_key_file:
        encrypted_output = encrypted_output_file.read()
        evaluation_key = evaluation_key_file.read()

    subprocess.run(["cp", "server_permanent/eval", f"{SERVER_DIR}"], cwd=DEPLOYMENT_DIR, check=True)
    subprocess.run(["cp", "server_permanent/tree.json", f"{SERVER_DIR}"], cwd=DEPLOYMENT_DIR, check=True)

    # Retrieve the encrypted output path
    encrypted_output_path = SERVER_DIR / f"{user_id}_encrypted_output"

    # Run the FHE execution
    start = time.time()
    encrypted_output = FHE_SERVER.run(encrypted_output, evaluation_key)
    ####################
    # HERE
    result = subprocess.run(["./eval", f"{user_id}"], cwd=SERVER_DIR, check=True)
    ####################
    assert isinstance(encrypted_output, bytes)
    fhe_execution_time = round(time.time() - start, 2)

    # Write the file using the above path
    with encrypted_output_path.open("wb") as f:
        f.write(encrypted_output)

    return JSONResponse(content=fhe_execution_time)


@app.post("/get_output")
def get_output(user_id: str = Form()):
    """Retrieve the encrypted output from the server."""

    print("\nGet the output from the server ............\n")

    # Path where the encrypted output is saved
    # encrypted_output_path = SERVER_DIR / f"{user_id}_encrypted_output"
    encrypted_output_path = SERVER_DIR / f"server_ostream_{user_id}"

    # Read the file using the above path
    with encrypted_output_path.open("rb") as f:
        encrypted_output = f.read()

    time.sleep(1)

    # Send the encrypted output
    return Response(encrypted_output)
