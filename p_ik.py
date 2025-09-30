import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def get_p_ik(train_embeddings, is_false, eval_embeddings=None, eval_is_false=None):
    """
        Fit linear classifier to embeddings to predict model correctness.
    """

    print('Accuracy of model on Task: %f.', 1 - torch.tensor(is_false).mean())

    # Convert the list of tensors to a 2D tensor.
    train_embeddings_tensor = torch.cat(train_embeddings, dim=0)
    embeddings_array = train_embeddings_tensor.cpu().numpy()

    # Split the data into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(embeddings_array, is_false,
                                                        test_size=0.2, random_state=42)

    # Fit a logistic regression model.
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict deterministically and probabilistically and compute accuracy and auroc for all splits.
    X_eval = torch.cat(eval_embeddings, dim=0).cpu().numpy()
    y_eval = eval_is_false

    Xs = [X_train, X_test, X_eval]
    ys = [y_train, y_test, y_eval]
    suffixes = ['train_train', 'train_test', 'eval']

    metrics, y_preds_proba = {}, {}
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle('Log Probabilities Predicted by p_ik when the true label is [False/True]')

    for ax, suffix, X, y_true in zip(axes, suffixes, Xs, ys):

        if suffix == 'eval':
            model = LogisticRegression()
            model.fit(embeddings_array, is_false)
            convergence = {'n_iter': model.n_iter_[0], 'converged': (model.n_iter_ < model.max_iter)[0]}

        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)
        y_preds_proba[suffix] = y_pred_proba
        acc_p_ik_train = accuracy_score(y_true, y_pred)
        auroc_p_ik_train = roc_auc_score(y_true, y_pred_proba[:, 1])
        split_metrics = {
            f'acc_p_ik_{suffix}': acc_p_ik_train,
            f'auroc_p_ik_{suffix}': auroc_p_ik_train}
        metrics.update(split_metrics)

        # Plotting.
        probabilities_of_false_points = y_pred_proba[:, 1][np.array(y_true) == 1.0]
        probabilities_of_true_points = y_pred_proba[:, 1][np.array(y_true) == 0.0]
        ax.hist(probabilities_of_false_points, bins=20, alpha=0.5, label='False')
        ax.hist(probabilities_of_true_points, bins=20, alpha=0.5, label='True')
        ax.legend(loc='upper right', title='True Label')
        fmt = {k: f"{v:.2f}" for k, v in split_metrics.items()}
        ax.set_title(f'Set: {suffix} \n {fmt}')

    # Plotting.
    axes[0].set_ylabel('Counts')
    axes[1].set_xlabel('Predicted Probabilities')
    os.system('mkdir -p figures')
    plt.savefig('figures/p_ik.png')
    plt.savefig('figures/p_ik.pdf')

    print('Metrics for p_ik classifier: %s.', metrics)

    return y_preds_proba['eval'][:, 1]