import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd 


def dataentry_to_dataframe(entry):
    df = pd.DataFrame(
        entry["target"],
        columns=[entry.get("item_id")],
        index=pd.period_range(
            start=entry["start"], periods=len(entry["target"]), freq=entry["start"].freq
        ),
    )

    return df


def plot_quantiles(quantiles, x_axis, predict, observe, predict_quantile, ax=None,):
    # x_axis = np.array([i for i in range(20)])
    # create figure
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    prop_cycle = iter(plt.rcParams["axes.prop_cycle"])

    obs_color = next(prop_cycle)["color"]
    pred_color = next(prop_cycle)["color"]

    ax.plot(range(len(observe)), observe,
            label="observed", c=obs_color, marker='o')
    ax.plot(range(len(observe)), predict,
            label="predicted", c=pred_color, marker='^')

    for i in range(predict_quantile.shape[1] // 2):
        interval = 1 - quantiles[i] * 2
        ax.fill_between(x_axis, predict_quantile[:, i], predict_quantile[:, -i - 1], alpha=(0.15 + i / 20),
                        fc=pred_color, label='%s%%Interval' % int(interval * 100))

    # ax.set_xlabel("Time index")
    fig.legend()
    plt.show()

    return fig
