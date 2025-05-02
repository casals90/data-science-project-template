from typing import Tuple, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.tools.startup import logger


def create_pie_chart_with_grouped_threshold(
        input_df: pd.DataFrame, column_name: str, title: str = None,
        ax: plt.Axes = None, threshold: float = 0.01,
        grouped_label: str = 'Others', x_label: str = '', y_label: str = '',
        font_size: int = 14, start_angle: int = 90,
        fig_size: Tuple[int, int] = (8, 8), **kwargs) -> None:
    """
    Create Pie chart from dataframe and grouped values by threshold.

    Args:
        input_df (pd.DataFrame): dataframe to plot.
        column_name (str): column to plot.
        title (optional, str): plot's title.
        ax (optional, plt.Axes): Matplotlib axes.
        threshold (optional, float): threshold to apply. Default it is 0.01.
        grouped_label (optional, str): label value to put when grouped
            values are greater than a threshold.
            Default value is 'Others'
        x_label (optional, str): label for x axis.
        y_label (optional, str): label for y axis.
        font_size (optional, str): font size of plot.
        start_angle (optional, int): start angle of Pie chart.
        fig_size (optional, Tuple[int, int]): plot's size.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    if title is None:
        title = f'{column_name}'

    grouped_df = input_df \
        .groupby(column_name) \
        .size() \
        .to_frame() \
        .reset_index() \
        .rename(columns={0: 'sizes'})

    if not grouped_df.empty:
        grouped_df['perc'] = grouped_df.sizes / grouped_df.sizes.sum()

        # If only exists 2 different values, it shows the both labels.
        if grouped_df.shape[0] == 2:
            grouped_df['label_cleaned'] = grouped_df[column_name]
        else:
            cond_lt_threshold = grouped_df.perc < threshold
            grouped_df.loc[cond_lt_threshold, 'label_cleaned'] = grouped_label
            grouped_df['label_cleaned'] = grouped_df \
                .label_cleaned \
                .fillna(grouped_df[column_name])

        grouped_df \
            .groupby('label_cleaned') \
            .sizes \
            .sum() \
            .sort_values() \
            .plot(kind='pie', autopct='%1.1f%%', title=title, ax=ax,
                  legend=None, xlabel=x_label, ylabel=y_label,
                  fontsize=font_size, startangle=start_angle, **kwargs)
    else:
        logger.warning(f'Empty grouped dataframe for column {column_name}')
        ax.remove()


def create_consulting_non_consulting_histogram_plot(
        df: pd.DataFrame, target_column: str, column_name: str,
        fig_size: Tuple[int, int] = (8, 8)) \
        -> None:
    """
    This function creates a histogram from specific dataset column for
    each class.

    Args:
        df (pd.DataFrame): dataframe to plot.
        target_column (str): the dataframe's target column.
        column_name (str): column to generate the plot by target class.
        fig_size (optional, Tuple[int, int]): plot's size.
    """
    cond_is_consultancy = df[target_column] == 1
    consulting_feature = df[cond_is_consultancy][column_name]
    non_consulting_feature = df[~cond_is_consultancy][column_name]

    plt.subplots(figsize=fig_size)

    kwargs = dict(alpha=0.5, bins=10)
    plt.hist(
        consulting_feature, **kwargs, color='tab:blue', label='Consulting')
    plt.hist(
        non_consulting_feature, **kwargs, color='tab:orange',
        label='Non-consulting')

    plt.gca().set(title=f'Histogram of {column_name}', ylabel='Value')
    plt.legend()


def generate_heat_map(
        corr_df: pd.DataFrame, title: Optional[str] = 'Heatmap',
        color_map: Optional[str] = 'BrBG',
        fig_size: Tuple[int, int] = (8, 8)) -> None:
    """
    Given a correlation dataframe, this function generates the heatmap plot.

    Args:
        corr_df (pd.DataFrame): a pd.DataFrame of correlations.
        title (Optional[str]): the plot's title. Default value is 'Heatmap'.
        color_map (Optional[str]): the color map to use in the plot.
        fig_size (optional, Tuple[int, int]): plot's size.
    """
    plt.figure(figsize=fig_size)
    heatmap = sns.heatmap(corr_df, vmin=-1, vmax=1, annot=True, cmap=color_map)
    heatmap.set_title(title, fontdict={'fontsize': 18}, pad=16)
