from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def plot_history(
    history: dict,
    output_path: Path
) -> None:

    train = history["train"]
    val = history["val"]

    epochs = [
        item["epoch"] for item in train
    ]

    metrics = [
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(15, 9)
    )

    for ax, metric in zip(
        axes.flat, metrics
    ):

        train_values = [
            item[metric] for item in train
        ]

        val_values = [
            item[metric] for item in val
        ]

        ax.plot(
            epochs,
            train_values,
            marker="o",
            label="Train"
        )

        ax.plot(
            epochs,
            val_values,
            marker="o",
            label="Validation"
        )

        ax.axvline(
            history["best_epoch"],
            linestyle="--",
            label="Best epoch"
        )

        ax.set_title(metric.upper())
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        ax.legend()

        if metric != "loss":
            ax.set_ylim(0, 1.05)

    axes.flat[-1].axis("off")

    fig.tight_layout()

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


def plot_confusion_matrix(
    metrics: dict,
    output_path: Path
) -> None:

    matrix = np.array([
        [metrics["tn"], metrics["fp"]],
        [metrics["fn"], metrics["tp"]]
    ], dtype=int)

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    im = ax.imshow(
        matrix,
        cmap="Blues"
    )

    for i in range(2):
        for j in range(2):

            ax.text(
                j, i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=16
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Horse", "Human"])
    ax.set_yticklabels(["Horse", "Human"])

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    fig.colorbar(im, ax=ax)

    fig.tight_layout()

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)