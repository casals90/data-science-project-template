import os
from typing import List, Optional

import numpy as np
import pandas as pd
import unidecode
from sentence_transformers import SentenceTransformer
from tensorboard.plugins import projector
from tqdm import tqdm


def generate_average_embeddings_content(
        model_name: str, contents: List[dict], text: str, content_id:
        Optional[str] = 'id', embeddings_folder: Optional[str] =
        '/data/processed/embeddings') \
        -> np.ndarray:
    """
    Generate embeddings using the SentenceTransformer class by using the model
    specified as argument. It loads the embeddings from the specified file if
    it exists. Otherwise, it generates the embeddings. In addition, it computes
    the average embeddings for each content.

    Args:
        model_name (str): the model name to load as a SentenceTransformer.
        contents (List[dict]): the list of dicts with the content to generate
            the embedding
        text (str): the key of dict that contains the text to compute
            the embedding.
        content_id (Optional, str): the key of dict that contains the id
            of content. Defaults to 'id'.
        embeddings_folder (Optional, str): the folder to store the embeddings.

    Returns:
        (np.ndarray): the generated embeddings for each content.
    """
    # Get the model name for output file name
    name = model_name.split('/')[-1]

    # Create embeddings folder if it does not exist.
    if not os.path.exists(embeddings_folder):
        os.makedirs(embeddings_folder)

    model = None
    embeddings = []
    with tqdm(contents, unit="iter", desc=f'Computing embedding') as pbar:
        for content in pbar:
            content_texts = content[text]

            output_filename = os.path.join(
                embeddings_folder,
                f'{content[content_id]}_'
                f'{name}.npy')

            if output_filename and os.path.exists(output_filename):
                content_embedding = np.load(output_filename)
            else:
                if model is None:
                    model = SentenceTransformer(model_name)
                # Compute the average embedding vector.
                content_embedding = np.mean(
                    model.encode(content_texts), axis=0)
                np.save(output_filename, content_embedding)

            embeddings.append(content_embedding)

    return np.array(embeddings)


def create_embeddings_metadata(
        df: pd.DataFrame, columns: List[str], log_dir: str, filename: str) \
        -> None:
    """
    Generate a metadata file for the embeddings. It should be a tsv file with
    the columns associated to the embeddings and specified by parameter.

    Args:
        df (pd.DataFrame): a pandas dataframe containing information related
            to the embeddings.
        columns (List[str]): the list of the dataframe columns to be stored.
        log_dir (str): the destination directory.
        filename (str): the name for the metadata file.
    """
    # Convert all metadata columns to a string.
    metadata_df = df.astype("string").fillna('').copy(deep=True)
    with open(os.path.join(log_dir, filename), "w") as f:
        f.write('\t'.join(columns) + '\n')
        for i, row in metadata_df.iterrows():
            row_values = ''
            for c in columns:
                row_values += unidecode.unidecode(row[c]) + '\t'
            f.write(f"{row_values[:-1]}\n")


def add_embedding_to_projector_config(
        config: projector.ProjectorConfig,
        tensor_name: str,
        metadata_path: str) -> None:
    """
    Given a Projector Config object add an embedding to the projection with
    the given name and metadata associated.

    Args:
        config (projector.ProjectorConfig): the Projector Config object.
        tensor_name (str): the name of the tensor holding the embeddings.
        metadata_path (str): the path to the metadata associated to
            the embeddings.
    """
    embedding = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
    embedding.tensor_name = f'{tensor_name}/.ATTRIBUTES/VARIABLE_VALUE'
    embedding.metadata_path = metadata_path
