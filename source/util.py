import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def plot_sample(df, cols, figsize=(15, 4), grid=True):
    ax = df[cols].plot(figsize=figsize, grid=grid)
    return ax
    
def plot_comparison(models, y_true, X_input, ylim=None):
    plt.figure(figsize=(15, 10))
    plt.plot(y_true, marker='.', label="True")

    color = iter(cm.rainbow(np.linspace(0, 1, len(models) + 1)))

    for name, model in models:
        c = next(color)
        y_pred = model.predict(X_input)
        # need to clip minus values
        y_pred = np.clip(y_pred, a_min=0, a_max=None)
        plt.plot(y_pred, c=c, label=name)

    if ylim is not None:
        plt.ylim(ylim)

    plt.legend()
    plt.show()
    
def report_feature_importance(f_columns, importances):
    feature_importances = [(feature, round(importance, 2))
                           for feature, importance in zip(f_columns, importances)]

    feature_importances = sorted(feature_importances,
                                 key=lambda x: x[1],
                                 reverse=True)

    for pair in feature_importances:
        print('Variable: {:20} Importance: {}'.format(*pair))
        
