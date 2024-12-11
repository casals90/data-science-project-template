import string
from typing import Optional

import langdetect
import nltk
import pandas as pd

from src.tools import utils as tools_utils
from src.tools.startup import logger


def compute_features(
        df: pd.DataFrame, text_column: Optional[str] = 'text') -> pd.DataFrame:
    """
    Given a pd.DataFrame and a text column name, this function computes the
    following features:
    - digits_number
    - punctuations_number
    - original text length
    - cleaned text length
    - cleaned text without stopwords length
    - original text tokens number
    - cleaned text tokens number
    - cleaned text without stopwords tokens number
    - original text length and cleaned text length difference
    - original text length and cleaned text without stopwords length difference
    - cleaned text length and cleaned text without stopwords length difference
    - original text tokens number and cleaned text tokens number difference
    - original text tokens number and cleaned text without stopwords tokens
        number difference
    - original text tokens number and cleaned text without stopwords tokens
        number difference

    Args:
        df (pd.DataFrame): a pd.dataFrame to compute features.
        text_column (Optional[str]): column name of text column. Default value
            is 'text'.

    Returns:
        (pd.DataFrame): a pd.DataFrame with new features.
    """
    df = compute_lang(df, text_column)
    df = compute_same_consultancy_value(df)
    df = text_cleaning_and_feature_engineering(df, text_column)

    return df


def compute_lang(
        df: pd.DataFrame, text_column: Optional[str] = 'text') -> pd.DataFrame:
    """
    This function checks the language of text column. It creates a new column
    'lang' with the language of the text column.

    Args:
        df (pd.DataFrame): a pd.DataFrame to compute the feature.
        text_column (Optional): column name of text column. Default value is
            'text'.

    Returns:
        (pd.DataFrame): a pd.DataFrame with new feature.
    """
    df['lang'] = df[text_column].apply(langdetect.detect)

    return df


def compute_same_consultancy_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function creates a new column 'same_consultancy_value' that it is True
    when the 'is_consultancy' and 'predicted_consultancy' have the same value.
    Otherwise, False.

    Args:
        df (pd.DataFrame): a pd.DataFrame to compute the feature.

    Returns:
        (pd.DataFrame): a pd.DataFrame with new feature.
    """
    cond_same_value = df['is_consultancy'] == df['predicted_consultancy']
    new_column = 'same_consultancy_value'
    df.loc[cond_same_value, new_column] = True
    df[new_column].fillna(False, inplace=True)

    return df


# Clean text by language
def text_cleaning_and_feature_engineering(
        df: pd.DataFrame, text_column: Optional[str] = 'text') -> pd.DataFrame:
    """
    Given a dataframe and the text column, this function computes different
    features. It cleans the text column for different purposes: for analyze
    text data: removing stop words (for each specific language), punctuations
    ets. On the other hand, it applies a "simple" clean as input of
    Transformer model.

    In addition, it computes the following features:
    - digits_number
    - punctuations_number
    - original text length
    - cleaned text length
    - cleaned text without stopwords length
    - original text tokens number
    - cleaned text tokens number
    - cleaned text without stopwords tokens number
    - original text length and cleaned text length difference
    - original text length and cleaned text without stopwords length difference
    - cleaned text length and cleaned text without stopwords length difference
    - original text tokens number and cleaned text tokens number difference
    - original text tokens number and cleaned text without stopwords tokens
        number difference
    - original text tokens number and cleaned text without stopwords tokens
        number difference

    Args:
        df (pd.DataFrame): a pd.DataFrame to compute the feature.
        text_column (Optional[str]): name of text column.

    Returns:
        (pd.DataFrame): a pd.DataFrame with new features.
    """
    langs = df.lang.unique()
    stop_words_dict = tools_utils.get_stopwords(langs)

    # Convert column to string
    df[text_column] = df[text_column].astype('string')
    # Convert text to lower case
    df[f'{text_column}_lower'] = df[text_column].str.lower()

    # Simple cleaning for transformer input.
    # Remove html tags.
    df[f'{text_column}_simple_cleaned'] = df[text_column].str.replace(
        r'<.*?>', '', regex=True)
    # Remove urls.
    df[f'{text_column}_simple_cleaned'] = df[text_column].str.replace(
        r'https?://[A-Za-z0-9./]+', ' ', regex=True)
    # Remove emails.
    email_reg_exp = r'[A-Za-z0-9]*@[A-Za-z]*\.?[A-Za-z0-9]*'
    df[f'{text_column}_simple_cleaned'] = df[f'{text_column}_simple_cleaned'] \
        .str.replace(email_reg_exp, ' ', regex=True)
    # Remove special characters.
    df[f'{text_column}_simple_cleaned'] = df[f'{text_column}_simple_cleaned'] \
        .str.replace(r'[\+\-\(\)\:\&\|]', ' ', regex=True)
    # Remove numbers.
    df[f'{text_column}_simple_cleaned'] = df[f'{text_column}_simple_cleaned'] \
        .str.replace(r'[0-9]', ' ', regex=True)
    # Remove extra blank spaces.
    df[f'{text_column}_simple_cleaned'] = df[f'{text_column}_simple_cleaned'] \
        .str.replace(r'\s+', ' ', regex=True)

    # Clean text by language.
    grouped = df.groupby('lang')
    for lang, grouped_df in grouped:
        logger.info(f'Cleaning {lang}...')

        # Get index of grouped rows.
        idx = grouped_df.index

        language = tools_utils.get_lang_name_from_lang_code(lang).lower()
        df.loc[idx, f'{text_column}_tokens'] = df.loc[idx, text_column] \
            .apply(nltk.tokenize.word_tokenize, language)

        # Get only words (remove numbers, special characters,
        # punctuation, etc.)
        df.loc[idx, f'{text_column}_cleaned_tokens'] = \
            df.loc[idx, f'{text_column}_lower'] \
                .str.replace('[^a-zA-Z]', ' ', regex=True) \
                .str.strip() \
                .apply(nltk.tokenize.word_tokenize, language)

        # Remove language stop words.
        stop_words = stop_words_dict[lang]
        df.loc[idx, f'{text_column}_cleaned_without_stopwords_tokens'] = \
            df.loc[idx, f'{text_column}_cleaned_tokens'].apply(
                lambda ws: [w for w in ws if w not in stop_words])

        logger.info('Done.')

    # Join tokens into the same string.
    df[f'{text_column}_cleaned'] = df[f'{text_column}_cleaned_tokens'] \
        .str.join(' ')
    df[f'{text_column}_cleaned_without_stopwords'] = \
        df[f'{text_column}_cleaned_without_stopwords_tokens'] \
            .str.join(' ')

    # Compute count features (only from original text).
    df[f'{text_column}_digits_number'] = df[text_column] \
        .str.count(r'\d')
    df[f'{text_column}_punctuations_number'] = df[text_column] \
        .str.count(f'[{string.punctuation}]')

    # Compute len features.
    df[f'{text_column}_len'] = df[text_column] \
        .str.len() \
        .fillna(0)
    df[f'{text_column}_cleaned_len'] = df[f'{text_column}_cleaned'] \
        .str.len() \
        .fillna(0)
    df[f'{text_column}_cleaned_without_stopwords_len'] = \
        df[f'{text_column}_cleaned_without_stopwords'] \
            .str.len() \
            .fillna(0)

    # Compute tokens number.
    df[f'{text_column}_tokens_number'] = df[f'{text_column}_tokens'] \
        .str.len() \
        .fillna(0)
    df[f'{text_column}_cleaned_tokens_number'] = \
        df[f'{text_column}_cleaned_tokens'] \
            .str.len() \
            .fillna(0)
    df[f'{text_column}_cleaned_without_stopwords_tokens_number'] = \
        df[f'{text_column}_cleaned_without_stopwords_tokens'] \
            .str.len() \
            .fillna(0)

    # Compute diff features
    df['original_cleaned_diff'] = \
        df[f'{text_column}_len'] - df[f'{text_column}_cleaned_len']
    df['original_cleaned_without_sw_diff'] = \
        df[f'{text_column}_len'] - \
        df[f'{text_column}_cleaned_without_stopwords_len']
    df['cleaned_cleaned_without_sw_diff'] = \
        df[f'{text_column}_cleaned_len'] - \
        df[f'{text_column}_cleaned_without_stopwords_len']

    df['original_cleaned_tokens_diff'] = \
        df[f'{text_column}_tokens_number'] - \
        df[f'{text_column}_cleaned_tokens_number']
    df['original_cleaned_without_sw_tokens_diff'] = \
        df[f'{text_column}_tokens_number'] - \
        df[f'{text_column}_cleaned_without_stopwords_tokens_number']
    df['cleaned_cleaned_without_sw_tokens_diff'] = \
        df[f'{text_column}_cleaned_tokens_number'] - \
        df[f'{text_column}_cleaned_without_stopwords_tokens_number']

    return df
