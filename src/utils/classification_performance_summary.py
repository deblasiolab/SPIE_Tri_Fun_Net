import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def classification_performance_summary(model, x_data, y_data, y_encoder, learn=False, save_path=None):
    y_pred = None
    if learn: y_pred = model.predict(x_data)
    else:
        y_pred_probs = model.predict(x_data, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

    # Compute overall accuracy
    acc = accuracy_score(y_data, y_pred)
    num_samples = len(y_data)
    num_correct = int(acc * num_samples)
    num_incorrect = num_samples - num_correct
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_data, y_pred, average=None, zero_division=0
    )

    # Add per-class accuracy
    class_accuracies = []
    for cls in np.unique(y_data):
        class_mask = y_data == cls
        class_accuracy = accuracy_score(y_data[class_mask], y_pred[class_mask])
        class_accuracies.append(class_accuracy)

    class_names = y_encoder.classes_
    table_data = [
        [cls, f"{p:.2f}", f"{r:.2f}", f"{f1s:.2f}", f"{ca:.2f}"]
        for cls, p, r, f1s, ca in zip(class_names, precision, recall, f1, class_accuracies)
    ]
    col_labels = ['Class', 'Precision', 'Recall', 'F1-Score', 'Accuracy']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Pie chart for overall accuracy
    ax1.pie([num_correct, num_incorrect],
            labels=['Correct', 'Incorrect'],
            autopct='%1.1f%%', startangle=90,
            colors=['#4CAF50', '#F44336'])
    ax1.set_title(f'Overall Accuracy: {acc*100:.2f}%')

    # Table for class-wise metrics
    ax2.set_axis_off()
    tbl = ax2.table(cellText=table_data, colLabels=col_labels, loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.2)
    ax2.set_title('Class-wise Metrics', pad=20)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.show()
    plt.clf()