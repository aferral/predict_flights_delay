import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, PrecisionRecallDisplay, \
    roc_curve,f1_score
import numpy as np


def evaluate_predicted_df(df, name,plot=False):
    y_true = df['atraso_15'].astype(int).values
    y_pred = df['y_pred'].astype(int).values
    y_score = df['y_score'].values

    print("=========== model: {0} ==============".format(name))
    print(classification_report(y_true=y_true, y_pred=y_pred, target_names=['oka', 'atraso']))
    roc_area = roc_auc_score(y_true=y_true, y_score=y_score)
    f1_val = f1_score(y_true=y_true, y_pred=y_pred,pos_label=0) * 0.5 + f1_score(y_true=y_true, y_pred=y_pred,pos_label=1)*0.5
    print("ROC area {0}".format(roc_area))

    if plot:
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        disp = PrecisionRecallDisplay(precision=precision, recall=recall)
        disp.plot()
        plt.show()

        plot_roc(y_true, y_score)

        fig, ax = plt.subplots()
        ax.plot(recall, precision, color='purple')
        ax.set_title('Precision-Recall Curve')
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        # display plot
        plt.show()
        print('=========================================')
    return roc_area,f1_val


def get_metric_and_best_threshold_from_roc_curve(y_train, y_score, calc='f1'):
    fpr, tpr, thresholds = roc_curve(y_train, y_score)
    n_positivos = y_train.sum()
    n_negativos = y_train.shape[0] - n_positivos

    tp = tpr * n_positivos
    tn = (1 - fpr) * n_negativos
    fp = fpr * n_negativos
    fn = (1 - tpr) * n_positivos

    if calc == 'f1':
        precision_pos = (tp) * 1.0 / (tp + fp)
        recall_pos = tpr
        F1_pos = np.nan_to_num(2 * (precision_pos * recall_pos) / (precision_pos + recall_pos + 0.00000001), 0)

        precision_neg = (tn) * 1.0 / (tn + fn)
        recall_neg = (1 - fpr)
        F1_neg = np.nan_to_num(2 * (precision_neg * recall_neg) / (precision_neg + recall_neg + 0.00000001), 0)

        score = F1_pos * 0.5 + F1_neg * 0.5
        best_threshold = thresholds[np.argmax(score)]
        best_score = np.amax(score)

    elif calc == 'f1_pos':
        precision_pos = (tp) * 1.0 / (tp + fp)
        recall_pos = tpr
        F1_pos = np.nan_to_num(2 * (precision_pos * recall_pos) / (precision_pos + recall_pos + 0.00000001), 0)
        score = F1_pos
        best_threshold = thresholds[np.argmax(score)]
        best_score = np.amax(score)

    elif calc == 'prec_pos':
        precision_pos = (tp) * 1.0 / (tp + fp)
        precision_pos = np.nan_to_num(precision_pos, 0)
        score = precision_pos
        best_threshold = thresholds[np.argmax(score)]
        best_score = np.amax(score)

    elif calc == 'acc':
        score = (tp + tn) * 1.0 / (tp + tn + fp + fn)
        best_threshold = thresholds[np.argmax(score)]
        best_score = np.amax(score)

    elif calc == 'f1_neg':
        precision_neg = (tn) * 1.0 / (tn + fn)
        recall_neg = (1 - fpr)
        F1_neg = np.nan_to_num(2 * (precision_neg * recall_neg) / (precision_neg + recall_neg + 0.00000001), 0)

        score = F1_neg
        best_threshold = thresholds[np.argmax(score)]
        best_score = np.amax(score)

    elif calc == 'mean_recall':
        positive_recall = tpr
        negative_recall = (1 - fpr)
        score = negative_recall * 0.5 + positive_recall * 0.5  # mean recall dado positivo y negativo
        best_threshold = thresholds[np.argmax(score)]
        best_score = np.amax(score)
    print("Threshold {0} maximizando {2} con valor {1}".format(best_threshold, best_score, calc))
    return best_score, best_threshold


def plot_roc(y_train, y_score):
    # plot roc test
    fpr, tpr, thresholds = roc_curve(y_train, y_score)
    plt.plot(fpr, tpr, label='ROC')
    plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), label="diagonal")
    max_anotations = 40
    jump = max(int(len(fpr) / max_anotations), 1)
    for x, y, txt in zip(fpr[::jump], tpr[::jump], thresholds[::jump]):
        plt.annotate(np.round(txt, 2), (x, y - 0.04))
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc="upper left")
    plt.show()
