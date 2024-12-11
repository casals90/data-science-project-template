import importlib
import os
from datetime import timedelta
from typing import List, Dict

import nltk
import pandas as pd
from nltk.langnames import langname

from src.tools.startup import logger, settings


def get_lang_name_from_lang_code(lang_code: str) -> str:
    """
    Given a language code, this function gets the corresponding language name.

    Args:
        lang_code (str): the lang code to get the language name.

    Returns:
        (str): the language name.
    """
    return langname(lang_code)


def get_stopwords(languages: List[str]) -> Dict[str, List[str]]:
    """
    Given a list of language codes, this function try to download the
    stopwords from 'nltk' package. If the language does not exist in 'nltk'
    package, it tries to read from disk.

    Args:
        languages (List[str]): a list of language to load the stopwords.

    Notes:
        I manually downloaded the stopwords of Estonian language.
        Source: https://github.com/kristel-/estonian-stopwords

    Returns:
        (Dict[List[str]]): a list of dicts where the key is the language and
        the value the list of stop words.
    """
    stop_words = {}
    for lang in languages:
        lang_name = get_lang_name_from_lang_code(lang).lower()
        try:
            stop_words[lang] = nltk.corpus.stopwords.words(lang_name)
        except Exception as e:
            logger.warning(f'Error when downloading stopwords '
                           f'for language {lang}: {e}.')

            file_path = os.path.join(
                settings['volumes']['raw'], settings['stop_words'][lang])
            logger.info(f'Loading {lang} stop words from local '
                        f'file {file_path}')

            content = read_file(file_path, encoding='utf8', mode='r')
            stop_words[lang] = content.split('\n')

    return stop_words


def read_file(file_path: str, **kwargs) -> str:
    """
    Given a file path, this function reads the content and returns it as a
    string.

    Args:
        file_path (str): file path to read.
        **kwargs: additional parameters for open function.

    Returns:
        (str): the content of the file.
    """
    try:
        with open(file_path, **kwargs) as file:
            content = file.read()
    except FileNotFoundError:
        logger.warning(f'File {file_path} does not exists.')
        content = ''

    return content


def compute_columns_correlation_with_target(
        df: pd.DataFrame, columns: List[str],
        target_column: str, corr_method: callable) -> pd.DataFrame:
    """
    Given a pd.DataFrame, a list of columns and a target column,
    this function computes the correlation between columns and target.

    Args:
        df (pd.DataFrame): a pd.DataFrame to compute correlation.
        columns (ct.StrList): a list of columns to compute correlation with
            the target.
        target_column (str): target's column name.
        corr_method (Callable): correlation method to apply.

    Returns:
        (pd.DataFrame): a pd.DataFrame with correlations values between
        columns and target.
    """
    correlation_values = []
    for col in columns:
        try:
            corr = corr_method(df[col], df[target_column])
            correlation_values.append(
                {'column': col, 'target': target_column, 'value': corr})
        except TypeError as _:
            logger.warning(f'Error when compute the correlation between '
                           f'{col} and {target_column}')

    corr_df = pd.DataFrame(correlation_values) \
        .pivot(values='value', index='column', columns='target') \
        .dropna()

    corr_df.columns = ['correlation']
    corr_df.sort_values(by='correlation', ascending=False, inplace=True)

    return corr_df


def get_string_chunks(string: str, length: int) -> List[str]:
    """
    Given a string and a length, it splits the string into chunks of length.

    Args:
        string (str): The string to split.
        length (int): The length of each chunk.

    Returns:
        (List[str]): A list of string chunks.
    """
    return list(
        (string[0 + i:length + i] for i in range(0, len(string), length)))


def import_library(module: str, params: dict = None) -> callable:
    """
    Given a module name and dict params, this function imports the module and
    creates a new callable with specific parameters.

    Args:
        module (str): module name.
        params (str): dict that contains params.

    Returns:
        callable of imported module with parameters.
    """
    library = '.'.join(module.split('.')[:-1])
    imported_module = importlib.import_module(library)
    name = module.split('.')[-1]

    if params is None:
        params = dict()

    return getattr(imported_module, name)(**params)


def format_time(elapsed) -> str:
    """
    Takes a time in seconds and returns a string hh:mm:ss

    Args:
        elapsed:

    Returns:
        (str): A string with the time in 'hh:mm:ss' format.
    """

    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(timedelta(seconds=elapsed_rounded))
