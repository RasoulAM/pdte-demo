import subprocess
import time
from typing import Dict, List, Tuple

import gradio as gr  # pylint: disable=import-error
import numpy as np
import pandas as pd
import requests
from symptoms_categories import SYMPTOMS_LIST
from utils import (
    CLIENT_DIR,
    CURRENT_DIR,
    DEPLOYMENT_DIR,
    INPUT_BROWSER_LIMIT,
    KEYS_DIR,
    SERVER_URL,
    TARGET_COLUMNS,
    TRAINING_FILENAME,
    clean_directory,
    get_disease_name,
    load_data,
    pretty_print,
)
import sys
from utils import PARAM_FILE_PATH, KEYS_AND_CT_FILE_PATH, CT_SIZE_PATH, PK_FILE_PATH, OUTPUT_FILE_PATH
import ast

from concrete.ml.deployment import FHEModelClient

# pylint: disable=c-extension-no-member,invalid-name


def is_none(obj) -> bool:
    """
    Check if the object is None.

    Args:
        obj (any): The input to be checked.

    Returns:
        bool: True if the object is None or empty, False otherwise.
    """
    return obj is None or (obj is not None and len(obj) < 1)


def display_default_symptoms_fn(default_disease: str) -> Dict:
    """
    Displays the symptoms of a given existing disease.

    Args:
        default_disease (str): Disease
    Returns:
        Dict: The according symptoms
    """
    df = pd.read_csv(TRAINING_FILENAME)
    df_filtred = df[df[TARGET_COLUMNS[1]] == default_disease]

    return {
        default_symptoms: gr.update(
            visible=True,
            value=pretty_print(
                df_filtred.columns[df_filtred.eq(1).any()].to_list(), delimiter=", "
            ),
        )
    }


def get_user_symptoms_from_checkboxgroup(checkbox_symptoms: List) -> np.array:
    """
    Convert the user symptoms into a binary vector representation.

    Args:
        checkbox_symptoms (List): A list of user symptoms.

    Returns:
        np.array: A binary vector representing the user's symptoms.

    Raises:
        KeyError: If a provided symptom is not recognized as a valid symptom.

    """
    symptoms_vector = {key: 0 for key in valid_symptoms}
    for pretty_symptom in checkbox_symptoms:
        original_symptom = "_".join((pretty_symptom.lower().split(" ")))
        if original_symptom not in symptoms_vector.keys():
            raise KeyError(
                f"The symptom '{original_symptom}' you provided is not recognized as a valid "
                f"symptom.\nHere is the list of valid symptoms: {symptoms_vector}"
            )
        symptoms_vector[original_symptom] = 1

    user_symptoms_vect = np.fromiter(symptoms_vector.values(), dtype=float)[np.newaxis, :]

    assert all(value == 0 or value == 1 for value in user_symptoms_vect.flatten())

    return user_symptoms_vect


def get_features_fn(*checked_symptoms: Tuple[str]) -> Dict:
    """
    Get vector features based on the selected symptoms.

    Args:
        checked_symptoms (Tuple[str]): User symptoms

    Returns:
        Dict: The encoded user vector symptoms.
    """

    return {
        error_box1: gr.update(visible=False),
        one_hot_vect: gr.update(
            visible=False,
            value=list(checked_symptoms) #get_user_symptoms_from_checkboxgroup(pretty_print(checked_symptoms)),
        ),
        submit_btn: gr.update(value="Data submitted ✅"),
    }


def key_gen_fn(user_symptoms: List[str]) -> Dict:
    """
    Generate keys for a given user.

    Args:
        user_symptoms (List[str]): The vector symptoms provided by the user.

    Returns:
        dict: A dictionary containing the generated keys and related information.

    """
    clean_directory()

    for file in ["keygen", "encrypt", "decrypt"]:
        subprocess.run(["cp", f"client_permanent/{file}", CLIENT_DIR], cwd=DEPLOYMENT_DIR, check=True)

    # Generate a random user ID
    # user_id = np.random.randint(0, 2**32)
    user_id = np.random.randint(1, 10**6)
    print(f"Your user ID is: {user_id}....")

    result = subprocess.run(["./keygen", f"{user_id}"], cwd=CLIENT_DIR, check=True)

    return {
        error_box2: gr.update(visible=False),
        # key_box: gr.update(visible=False, value="aaa"),
        user_id_box: gr.update(visible=False, value=user_id),
        # key_len_box: gr.update(
        #     visible=False, value=f"?? MB"
        # ),
        gen_key_btn: gr.update(value="Keys have been generated ✅")
    }


def encrypt_fn(user_symptoms: np.ndarray, user_id: str) -> None:
    """
    Encrypt the user symptoms vector in the `Client Side`.

    Args:
        user_symptoms (List[str]): The vector symptoms provided by the user
        user_id (user): The current user's ID
    """

    if is_none(user_id) or is_none(user_symptoms):
        print("Error in encryption step: Provide your symptoms and generate the evaluation keys.")
        return {
            error_box3: gr.update(
                visible=True,
                value="⚠️ Please ensure that your symptoms have been submitted and "
                "that you have generated the evaluation key.",
            )
        }

    # input_choice = np.random.randint(0, 5)
    input_choice = 0
    # print(f"Your input choice is: {input_choice}....")
    # for file in [f"input_{input_choice}"]:
    #     subprocess.run(["cp", f"client_permanent/{file}", CLIENT_DIR], cwd=DEPLOYMENT_DIR, check=True)



    with open(f'{CLIENT_DIR}/input_0', 'w') as file:
        # Write each integer on a new line
        features = ast.literal_eval(user_symptoms)
        for integer in features:
            file.write(f'{integer}\n')
    result = subprocess.run(["./encrypt", f"{user_id}", f"{input_choice}"], cwd=CLIENT_DIR, check=True)

    return {
        error_box3: gr.update(visible=False),
        one_hot_vect_box: gr.update(visible=True, value=user_symptoms),
        # enc_vect_box: gr.update(visible=True, value=encrypted_quantized_user_symptoms_shorten_hex),
    }


def send_input_fn(user_id: str, user_symptoms: np.ndarray) -> Dict:
    """Send the encrypted data and the evaluation key to the server.

    Args:
        user_id (str): The current user's ID
        user_symptoms (np.ndarray): The user symptoms
    """

    # Define the data and files to post
    data = {
        "user_id": user_id,
        # "input": user_symptoms,
    }

    files = [
        ("files", open(f"{CLIENT_DIR}/{PARAM_FILE_PATH}_{user_id}", "rb")),
        ("files", open(f"{CLIENT_DIR}/{KEYS_AND_CT_FILE_PATH}_{user_id}", "rb")),
        ("files", open(f"{CLIENT_DIR}/{CT_SIZE_PATH}_{user_id}", "rb")),
        ("files", open(f"{CLIENT_DIR}/{PK_FILE_PATH}_{user_id}", "rb")),
    ]

    # Send the encrypted input and evaluation key to the server
    url = SERVER_URL + "send_input"
    with requests.post(
        url=url,
        data=data,
        files=files,
    ) as response:
        print(f"Sending Data: {response.ok=}")
    return {
        error_box4: gr.update(visible=False),
        srv_resp_send_data_box: "Data sent",
    }


def run_fhe_fn(user_id: str) -> Dict:
    """Send the encrypted input and the evaluation key to the server.

    Args:
        user_id (int): The current user's ID.
    """
    if is_none(user_id):
        return {
            error_box5: gr.update(
                visible=True,
                value="⚠️ Please check your connectivity \n"
                "⚠️ Ensure that the symptoms have been submitted, the evaluation "
                "key has been generated and the server received the data "
                "before processing the data.",
            ),
            fhe_execution_time_box: None,
        }

    data = {
        "user_id": user_id,
    }

    url = SERVER_URL + "run_fhe"

    with requests.post(
        url=url,
        data=data,
    ) as response:
        if not response.ok:
            return {
                error_box5: gr.update(
                    visible=True,
                    value=(
                        "⚠️ An error occurred on the Server Side. "
                        "Please check connectivity and data transmission."
                    ),
                ),
                fhe_execution_time_box: gr.update(visible=False),
            }
        else:
            time.sleep(1)
            print(f"response.ok: {response.ok}, {response.json()} - Computed")

    return {
        error_box5: gr.update(visible=False),
        fhe_execution_time_box: gr.update(visible=True, value=f"{response.json():.2f} seconds"),
    }


def get_output_fn(user_id: str, user_symptoms: np.ndarray) -> Dict:
    """Retreive the encrypted data from the server.

    Args:
        user_id (str): The current user's ID
        user_symptoms (np.ndarray): The user symptoms
    """

    if is_none(user_id) or is_none(user_symptoms):
        return {
            error_box6: gr.update(
                visible=True,
                value="⚠️ Please check your connectivity \n"
                "⚠️ Ensure that the server has successfully processed and transmitted the data to the client.",
            )
        }

    data = {
        "user_id": user_id,
    }

    # Retrieve the encrypted output
    url = SERVER_URL + "get_output"
    with requests.post(
        url=url,
        data=data,
    ) as response:
        if response.ok:
            print(f"Receive Data: {response.ok=}")

            encrypted_output = response.content

            # Save the encrypted output to bytes in a file as it is too large to pass through
            # regular Gradio buttons (see https://github.com/gradio-app/gradio/issues/1877)
            # encrypted_output_path = CLIENT_DIR / f"{user_id}_encrypted_output"
            encrypted_output_path = CLIENT_DIR / f"{OUTPUT_FILE_PATH}_{user_id}"

            with encrypted_output_path.open("wb") as f:
                f.write(encrypted_output)
    return {error_box6: gr.update(visible=False), srv_resp_retrieve_data_box: "Data received"}


def decrypt_fn(
    user_id: str, user_symptoms: np.ndarray
) -> Dict:
    """Dencrypt the data on the `Client Side`.

    Args:
        user_id (str): The current user's ID
        user_symptoms (np.ndarray): The user symptoms
        threshold (float): Probability confidence threshold

    Returns:
        Decrypted output
    """

    if is_none(user_id) or is_none(user_symptoms):
        return {
            error_box7: gr.update(
                visible=True,
                value="⚠️ Please check your connectivity \n"
                "⚠️ Ensure that the client has successfully received the data from the server.",
            )
        }

    # Get the encrypted output path
    # encrypted_output_path = CLIENT_DIR / f"{user_id}_encrypted_output"
    encrypted_output_path = CLIENT_DIR / f"{OUTPUT_FILE_PATH}_{user_id}"

    if not encrypted_output_path.is_file():
        print("Error in decryption step: Please run the FHE execution, first.")
        return {
            error_box7: gr.update(
                visible=True,
                value="⚠️ Please ensure that: \n"
                "- the connectivity \n"
                "- the symptoms have been submitted \n"
                "- the evaluation key has been generated \n"
                "- the server processed the encrypted data \n"
                "- the Client received the data from the Server before decrypting the prediction",
            ),
            decrypt_box: None,
        }

    APPROVED_MESSAGE = "Your tumor is benign ✅"
    DENIED_MESSAGE = "You tumor is malignant ❌"

    result = subprocess.run(["./decrypt", f"{user_id}"], cwd=CLIENT_DIR, stdout=subprocess.PIPE, text=True)
    output = result.stdout
    if "Response: 1" in output:
        out = (APPROVED_MESSAGE)
    else:
        out = (DENIED_MESSAGE)

    return {
        error_box7: gr.update(visible=False),
        decrypt_box: out,
        submit_btn: gr.update(value="Submit"),
    }


def reset_fn():
    """Reset the space and clear all the box outputs."""

    clean_directory()

    return {
        one_hot_vect: None,
        one_hot_vect_box: None,
        enc_vect_box: gr.update(visible=True, value=None),
        quant_vect_box: gr.update(visible=False, value=None),
        user_id_box: gr.update(visible=False, value=None),
        # default_symptoms: gr.update(visible=True, value=None),
        # default_disease_box: gr.update(visible=True, value=None),
        key_box: gr.update(visible=True, value=None),
        key_len_box: gr.update(visible=False, value=None),
        fhe_execution_time_box: gr.update(visible=True, value=None),
        decrypt_box: None,
        submit_btn: gr.update(value="Submit"),
        error_box7: gr.update(visible=False),
        error_box1: gr.update(visible=False),
        error_box2: gr.update(visible=False),
        error_box3: gr.update(visible=False),
        error_box4: gr.update(visible=False),
        error_box5: gr.update(visible=False),
        error_box6: gr.update(visible=False),
        srv_resp_send_data_box: None,
        srv_resp_retrieve_data_box: None,
        **{box: None for box in check_boxes},
    }


if __name__ == "__main__":

    SERVER_PORT = 8000
    _public = False

    if len(sys.argv) > 1:
        SERVER_PORT = int(sys.argv[1])
    if len(sys.argv) > 2:
        _public = True if sys.argv[2].lower() == "public" else False
    
    subprocess.Popen(["uvicorn", "server:app", "--port", f"{SERVER_PORT}"], cwd=CURRENT_DIR)
    time.sleep(3)


    print("Starting demo ...")

    clean_directory()

    (X_train, X_test), (y_train, y_test), valid_symptoms, diseases = load_data()

    with gr.Blocks() as demo:

        # Link + images
        gr.Markdown()
        # gr.Markdown(
        #     """
        #     <p align="center">
        #         <img width=200 src="https://user-images.githubusercontent.com/5758427/197816413-d9cddad3-ba38-4793-847d-120975e1da11.png">
        #     </p>
        #     """)
        # gr.Markdown()
        gr.Markdown("""<h2 align="center">Health Prediction On Encrypted Data Using Fully Homomorphic Encryption</h2>""")
        # gr.Markdown()
        # gr.Markdown(
        #     """
        #     <p align="center">
        #         <a href="https://github.com/zama-ai/concrete-ml"> <img style="vertical-align: middle; display:inline-block; margin-right: 3px;" width=15 src="https://user-images.githubusercontent.com/5758427/197972109-faaaff3e-10e2-4ab6-80f5-7531f7cfb08f.png">Concrete-ML</a>
        #         —
        #         <a href="https://docs.zama.ai/concrete-ml"> <img style="vertical-align: middle; display:inline-block; margin-right: 3px;" width=15 src="https://user-images.githubusercontent.com/5758427/197976802-fddd34c5-f59a-48d0-9bff-7ad1b00cb1fb.png">Documentation</a>
        #         —
        #         <a href="https://zama.ai/community"> <img style="vertical-align: middle; display:inline-block; margin-right: 3px;" width=15 src="https://user-images.githubusercontent.com/5758427/197977153-8c9c01a7-451a-4993-8e10-5a6ed5343d02.png">Community</a>
        #         —
        #         <a href="https://twitter.com/zama_fhe"> <img style="vertical-align: middle; display:inline-block; margin-right: 3px;" width=15 src="https://user-images.githubusercontent.com/5758427/197975044-bab9d199-e120-433b-b3be-abd73b211a54.png">@zama_fhe</a>
        #     </p>
        #     """)
        gr.Markdown()
        gr.Markdown(
            """
            <p align="center">
            <img width="65%" height="25%" src="https://raw.githubusercontent.com/kcelia/Img/main/healthcare_prediction.jpg">
            </p>
            """
        )
        gr.Markdown("## Notes")
        gr.Markdown(
            """
            - The private key is used to encrypt and decrypt the data and shall never be shared.
            - The evaluation key is a public key that the server needs to process encrypted data.
            """
        )

        # ------------------------- Step 1 -------------------------
        gr.Markdown("\n")
        gr.Markdown("## Step 1: Selecting inputs")
        gr.Markdown("<hr />")
        gr.Markdown("<span style='color:grey'>Client Side</span>")
        
        min_max = {
            "minimum": 0,
            "maximum": 2**16-1, 
        }

        features = []
        with gr.Row():
            with gr.Column():
                for i in range(10):
                    features.append(gr.Slider(
                        minimum=0,
                        maximum=2**16-1,
                        step=1, 
                        label=f"Feature {i}"
                    ))
            with gr.Column():
                for i in range(10):
                    features.append(gr.Slider(
                        minimum=0,
                        maximum=2**16-1,
                        step=1, 
                        label=f"Feature {10+i}"
                    ))
            with gr.Column():
                for i in range(10):
                    features.append(gr.Slider(
                        minimum=0,
                        maximum=2**16-1,
                        step=1, 
                        label=f"Feature {20+i}"
                    ))

        error_box1 = gr.Textbox(label="Error ❌", visible=False)

        # User vector symptoms encoded in oneHot representation
        one_hot_vect = gr.Textbox(visible=False)
        # Submit button
        submit_btn = gr.Button("Submit")
        # Clear botton
        clear_button = gr.Button("Reset Space 🔁", visible=False)

        submit_btn.click(
            fn=get_features_fn,
            inputs=[*features],
            outputs=[error_box1, one_hot_vect, submit_btn],
        )

        # ------------------------- Step 2 -------------------------
        gr.Markdown("\n")
        gr.Markdown("## Step 2: Encrypt data")
        gr.Markdown("<hr />")
        gr.Markdown("<span style='color:grey'>Client Side</span>")
        # Step 2.1: Key generation
        gr.Markdown(
            "### Key Generation\n\n"
            "In FHE schemes, a secret (enc/dec)ryption keys are generated for encrypting and decrypting data owned by the client. \n\n"
            "Additionally, a public evaluation key is generated, enabling external entities to perform homomorphic operations on encrypted data, without the need to decrypt them. \n\n"
            "The evaluation key will be transmitted to the server for further processing."
        )

        gen_key_btn = gr.Button("Generate the private and evaluation keys.")
        error_box2 = gr.Textbox(label="Error ❌", visible=False)
        user_id_box = gr.Textbox(label="User ID:", visible=False)
        key_len_box = gr.Textbox(label="Evaluation Key Size:", visible=False)
        key_box = gr.Textbox(label="Evaluation key (truncated):", max_lines=3, visible=False)

        gen_key_btn.click(
            key_gen_fn,
            inputs=one_hot_vect,
            outputs=[
                key_box,
                user_id_box,
                key_len_box,
                error_box2,
                gen_key_btn,
            ],
        )

        # Step 2.2: Encrypt data locally
        gr.Markdown("### Encrypt the data")
        encrypt_btn = gr.Button("Encrypt the data using the private secret key")
        error_box3 = gr.Textbox(label="Error ❌", visible=False)
        quant_vect_box = gr.Textbox(label="Quantized Vector:", visible=False)

        with gr.Row():
            with gr.Column():
                one_hot_vect_box = gr.Textbox(label="User Symptoms Vector:", max_lines=10)
            with gr.Column():
                enc_vect_box = gr.Textbox(label="Encrypted Vector:", max_lines=10)

        encrypt_btn.click(
            encrypt_fn,
            inputs=[one_hot_vect, user_id_box],
            outputs=[
                one_hot_vect_box,
                enc_vect_box,
                error_box3,
            ],
        )
        # Step 2.3: Send encrypted data to the server
        gr.Markdown(
            "### Send the encrypted data to the <span style='color:grey'>Server Side</span>"
        )
        error_box4 = gr.Textbox(label="Error ❌", visible=False)

        with gr.Row().style(equal_height=False):
            with gr.Column(scale=4):
                send_input_btn = gr.Button("Send data")
            with gr.Column(scale=1):
                srv_resp_send_data_box = gr.Checkbox(label="Data Sent", show_label=False)

        send_input_btn.click(
            send_input_fn,
            inputs=[user_id_box, one_hot_vect],
            outputs=[error_box4, srv_resp_send_data_box],
        )

        # ------------------------- Step 3 -------------------------
        gr.Markdown("\n")
        gr.Markdown("## Step 3: Run the FHE evaluation")
        gr.Markdown("<hr />")
        gr.Markdown("<span style='color:grey'>Server Side</span>")
        gr.Markdown(
            "Once the server receives the encrypted data, it can process and compute the output without ever decrypting the data just as it would on clear data.\n\n"
            "This server employs a [Logistic Regression](https://github.com/zama-ai/concrete-ml/tree/release/1.1.x/use_case_examples/disease_prediction) model that has been trained on this [data-set](https://github.com/anujdutt9/Disease-Prediction-from-Symptoms/tree/master/dataset)."
        )

        run_fhe_btn = gr.Button("Run the FHE evaluation")
        error_box5 = gr.Textbox(label="Error ❌", visible=False)
        fhe_execution_time_box = gr.Textbox(label="Total FHE Execution Time:", visible=True)
        run_fhe_btn.click(
            run_fhe_fn,
            inputs=[user_id_box],
            outputs=[fhe_execution_time_box, error_box5],
        )

        # ------------------------- Step 4 -------------------------
        gr.Markdown("\n")
        gr.Markdown("## Step 4: Decrypt the data")
        gr.Markdown("<hr />")
        gr.Markdown("<span style='color:grey'>Client Side</span>")
        gr.Markdown(
            "### Get the encrypted data from the <span style='color:grey'>Server Side</span>"
        )

        error_box6 = gr.Textbox(label="Error ❌", visible=False)

        # Step 4.1: Data transmission
        with gr.Row().style(equal_height=True):
            with gr.Column(scale=4):
                get_output_btn = gr.Button("Get data")
            with gr.Column(scale=1):
                srv_resp_retrieve_data_box = gr.Checkbox(label="Data Received", show_label=False)

        get_output_btn.click(
            get_output_fn,
            inputs=[user_id_box, one_hot_vect],
            outputs=[srv_resp_retrieve_data_box, error_box6],
        )

        # Step 4.1: Data transmission
        gr.Markdown("### Decrypt the output")
        decrypt_btn = gr.Button("Decrypt the output using the private secret key")
        error_box7 = gr.Textbox(label="Error ❌", visible=False)
        decrypt_box = gr.Textbox(label="Decrypted Output:")

        decrypt_btn.click(
            decrypt_fn,
            inputs=[user_id_box, one_hot_vect],
            outputs=[decrypt_box, error_box7, submit_btn],
        )

        # ------------------------- End -------------------------

        # gr.Markdown(
        #     """**Please Note**: This space is intended solely for educational and demonstration purposes. 
        #    It should not be considered as a replacement for professional medical counsel, diagnosis, or therapy for any health or related issues. 
        #    Any questions or concerns about your individual health should be addressed to your doctor or another qualified healthcare provider.
        #     """
        # )

        clear_button.click(
            reset_fn,
            outputs=[
                one_hot_vect_box,
                one_hot_vect,
                submit_btn,
                error_box1,
                error_box2,
                error_box3,
                error_box4,
                error_box5,
                error_box6,
                error_box7,
                # default_disease_box,
                # default_symptoms,
                user_id_box,
                key_len_box,
                key_box,
                quant_vect_box,
                enc_vect_box,
                srv_resp_send_data_box,
                srv_resp_retrieve_data_box,
                fhe_execution_time_box,
                decrypt_box,
                *features,
            ],
        )

        demo.launch(share=_public)
