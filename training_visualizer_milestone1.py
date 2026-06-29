import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

from micrograd.nn import MLP


OUT_DIR = Path("visualizer_outputs")
SNAPSHOT_STEPS = (0, 1, 5, 10, 25, 50, 75, 100)


def build_dataset():
    """Create the fixed 2D classification dataset used by the visualizer.

    Parameters:
        None.

    Returns:
        A tuple ``(X, y)`` where ``X`` is a NumPy array of 2D input points and
        ``y`` is a NumPy array of class labels converted from ``0/1`` to
        ``-1/+1`` for the SVM-style margin loss.
    """
    X, y = make_moons(n_samples=100, noise=0.1, random_state=1337)
    y = y * 2 - 1
    return X, y


def evaluate_loss(model, X, y, alpha=1e-4):
    """Run a forward pass and compute loss plus classification accuracy.

    Parameters:
        model: The ``micrograd.nn.MLP`` model to evaluate. Its parameters are
            ``Value`` objects, so the returned loss can later call
            ``backward()``.
        X: A NumPy array of input points. Each row is one 2D training example.
        y: A NumPy array of target labels. Labels are expected to be ``-1`` or
            ``+1``.
        alpha: L2 regularization strength. Higher values penalize large model
            parameters more strongly.

    Returns:
        A tuple ``(total_loss, accuracy)``. ``total_loss`` is a micrograd
        ``Value`` that includes data loss and regularization loss. ``accuracy``
        is a Python float between ``0`` and ``1``.
    """
    scores = [model(x) for x in X]
    losses = [(1 + -yi * scorei).relu() for yi, scorei in zip(y, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))
    reg_loss = alpha * sum((p * p for p in model.parameters()))
    total_loss = data_loss + reg_loss
    accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(y, scores)]
    return total_loss, sum(accuracy) / len(accuracy)


def predict_array(model, points):
    """Evaluate the trained MLP on many points using a vectorized NumPy pass.

    Parameters:
        model: The trained ``micrograd.nn.MLP`` model. This function reads the
            current scalar ``Value.data`` weights and biases from its layers.
        points: A NumPy array where each row is one 2D point to classify.

    Returns:
        A flat NumPy array of model scores, one score per input point. Positive
        scores are interpreted as class ``+1`` and negative scores as class
        ``-1``.

    Notes:
        Training still uses micrograd. This helper is only for faster
        visualization of dense decision-boundary grids.
    """
    activations = points
    for layer in model.layers:
        W = np.array([[w.data for w in neuron.w] for neuron in layer.neurons]).T
        b = np.array([neuron.b.data for neuron in layer.neurons])
        activations = activations @ W + b
        if layer.neurons[0].nonlin:
            activations = np.maximum(0, activations)
    return activations.reshape(-1)


def predict_grid(model, X, grid_res=120):
    """Evaluate the model over a rectangular 2D grid for contour plotting.

    Parameters:
        model: The trained ``micrograd.nn.MLP`` model to visualize.
        X: The original training inputs. Their min/max coordinates define the
            plotted region.
        grid_res: Number of grid samples along each axis. Larger values produce
            smoother decision-boundary plots but take more time to render.

    Returns:
        A tuple ``(xx, yy, zz)``. ``xx`` and ``yy`` are coordinate grids, and
        ``zz`` contains the model score at each grid location.
    """
    pad = 0.35
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res),
    )
    points = np.c_[xx.ravel(), yy.ravel()]
    scores = predict_array(model, points)
    return xx, yy, scores.reshape(xx.shape)


def layer_preactivations(model, points):
    """Compute each layer's pre-activation values with a NumPy forward pass.

    Parameters:
        model: The trained or currently-training ``micrograd.nn.MLP`` model.
        points: A NumPy array where each row is one input point.

    Returns:
        A list of NumPy arrays. Each array contains the pre-activation values
        for one layer before any ReLU is applied.
    """
    activations = points
    preactivations = []
    for layer in model.layers:
        W = np.array([[w.data for w in neuron.w] for neuron in layer.neurons]).T
        b = np.array([neuron.b.data for neuron in layer.neurons])
        z = activations @ W + b
        preactivations.append(z)
        activations = np.maximum(0, z) if layer.neurons[0].nonlin else z
    return preactivations


def dead_relu_stat_keys(model):
    """Build metric keys for dead-ReLU counts in nonlinear layers.

    Parameters:
        model: The ``micrograd.nn.MLP`` model whose hidden layers should be
            tracked.

    Returns:
        A list of CSV/metric keys, one for each nonlinear layer.
    """
    keys = []
    for layer_index, layer in enumerate(model.layers, start=1):
        if layer.neurons[0].nonlin:
            keys.append(f"layer_{layer_index}_dead_relu_count")
    return keys


def collect_dead_relu_stats(model, X):
    """Count dead ReLU neurons for each nonlinear layer on the dataset.

    Parameters:
        model: The ``micrograd.nn.MLP`` model to inspect.
        X: A NumPy array of training inputs. The dead-ReLU count is measured on
            these points.

    Returns:
        A dictionary mapping ``layer_N_dead_relu_count`` names to integer
        counts. A neuron is counted as dead when its pre-activation is never
        positive for any point in ``X``.
    """
    stats = {}
    preactivations = layer_preactivations(model, X)
    for layer_index, (layer, z) in enumerate(
        zip(model.layers, preactivations), start=1
    ):
        if layer.neurons[0].nonlin:
            dead_count = int(np.sum(np.all(z <= 0, axis=0)))
            stats[f"layer_{layer_index}_dead_relu_count"] = dead_count
    return stats


def first_layer_boundary_snapshot(model, X):
    """Capture first-layer ReLU boundary lines for one training step.

    Parameters:
        model: The ``micrograd.nn.MLP`` model to inspect.
        X: A NumPy array of training inputs used to decide which first-layer
            neurons are dead on the visible dataset.

    Returns:
        A dictionary with ``lines`` and ``dead_mask`` entries. ``lines`` is a
        list of ``(w1, w2, b)`` tuples for boundaries ``w1*x + w2*y + b = 0``.
        ``dead_mask`` is a boolean array marking neurons that never activate on
        ``X``.
    """
    first_layer = model.layers[0]
    lines = [
        (neuron.w[0].data, neuron.w[1].data, neuron.b.data)
        for neuron in first_layer.neurons
    ]
    z = layer_preactivations(model, X)[0]
    dead_mask = np.all(z <= 0, axis=0)
    return {"lines": lines, "dead_mask": dead_mask}


def plot_decision_snapshots(X, y, snapshots, output_path):
    """Render decision-boundary snapshots from multiple training steps.

    Parameters:
        X: A NumPy array of 2D training inputs to draw as scatter points.
        y: A NumPy array of ``-1/+1`` labels used to color the scatter points.
        snapshots: A list of dictionaries. Each dictionary must contain
            ``step``, ``loss``, ``accuracy``, and ``grid`` keys. The ``grid``
            value is the ``(xx, yy, zz)`` tuple returned by ``predict_grid``.
        output_path: File path where the PNG image should be written.

    Returns:
        None. The function writes the plot to ``output_path``.
    """
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for ax, snapshot in zip(axes.ravel(), snapshots):
        xx, yy, zz = snapshot["grid"]
        ax.contourf(xx, yy, zz, levels=30, cmap="RdBu", alpha=0.72)
        ax.contour(xx, yy, zz, levels=[0], colors="black", linewidths=1.3)
        ax.scatter(
            X[:, 0],
            X[:, 1],
            c=y,
            cmap="bwr",
            edgecolors="black",
            linewidths=0.5,
            s=34,
        )
        ax.set_title(
            f"step {snapshot['step']}  "
            f"loss {snapshot['loss']:.3f}  "
            f"acc {snapshot['accuracy'] * 100:.0f}%"
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("micrograd MLP decision boundary during training", fontsize=15)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_training_curves(metrics, output_path):
    """Render loss and accuracy curves across the full training run.

    Parameters:
        metrics: A list of dictionaries with ``step``, ``loss``, and
            ``accuracy`` keys. Each dictionary records one training step.
        output_path: File path where the PNG image should be written.

    Returns:
        None. The function writes the plot to ``output_path``.
    """
    steps = [row["step"] for row in metrics]
    losses = [row["loss"] for row in metrics]
    accuracies = [row["accuracy"] for row in metrics]

    fig, ax_loss = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax_acc = ax_loss.twinx()

    loss_line = ax_loss.plot(steps, losses, color="#1f77b4", label="loss")
    acc_line = ax_acc.plot(steps, accuracies, color="#d62728", label="accuracy")

    ax_loss.set_xlabel("training step")
    ax_loss.set_ylabel("loss", color="#1f77b4")
    ax_acc.set_ylabel("accuracy", color="#d62728")
    ax_acc.set_ylim(0, 1.05)
    ax_loss.grid(alpha=0.25)

    lines = loss_line + acc_line
    labels = [line.get_label() for line in lines]
    ax_loss.legend(lines, labels, loc="center right")
    fig.suptitle("training progress")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def layer_stat_keys(model):
    """Build the metric column names used for per-layer training diagnostics.

    Parameters:
        model: The ``micrograd.nn.MLP`` model whose layers should be tracked.

    Returns:
        A flat list of CSV/metric keys. For each layer, the list contains a
        gradient-norm key and an update-norm key.
    """
    keys = []
    for layer_index, _ in enumerate(model.layers, start=1):
        keys.append(f"layer_{layer_index}_grad_norm")
        keys.append(f"layer_{layer_index}_update_norm")
    return keys


def collect_layer_stats(model, learning_rate):
    """Measure gradient and SGD update sizes for each layer.

    Parameters:
        model: The ``micrograd.nn.MLP`` model after ``backward()`` has filled
            each parameter's ``grad`` field.
        learning_rate: The scalar learning rate for the current SGD step. The
            update norm is computed from ``learning_rate * parameter.grad``.

    Returns:
        A dictionary mapping metric names to floats. For each layer, it records
        the L2 norm of all parameter gradients and the L2 norm of the parameter
        update that will be applied by SGD.
    """
    stats = {}
    for layer_index, layer in enumerate(model.layers, start=1):
        params = layer.parameters()
        grad_norm = sum((p.grad ** 2 for p in params)) ** 0.5
        update_norm = sum(((learning_rate * p.grad) ** 2 for p in params)) ** 0.5
        stats[f"layer_{layer_index}_grad_norm"] = grad_norm
        stats[f"layer_{layer_index}_update_norm"] = update_norm
    return stats


def plot_layer_norms(metrics, output_path):
    """Render per-layer gradient and update norms across training.

    Parameters:
        metrics: A list of dictionaries with one entry per training step. Rows
            should include ``layer_N_grad_norm`` and ``layer_N_update_norm``
            values for steps where an optimizer update occurred.
        output_path: File path where the PNG image should be written.

    Returns:
        None. The function writes the plot to ``output_path``.
    """
    grad_keys = [
        key for key in metrics[0] if key.startswith("layer_") and key.endswith("_grad_norm")
    ]
    update_keys = [
        key
        for key in metrics[0]
        if key.startswith("layer_") and key.endswith("_update_norm")
    ]

    fig, (ax_grad, ax_update) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, constrained_layout=True
    )

    for key in grad_keys:
        rows = [row for row in metrics if row[key] is not None]
        ax_grad.plot(
            [row["step"] for row in rows],
            [row[key] for row in rows],
            label=key.replace("_grad_norm", ""),
        )

    for key in update_keys:
        rows = [row for row in metrics if row[key] is not None]
        ax_update.plot(
            [row["step"] for row in rows],
            [row[key] for row in rows],
            label=key.replace("_update_norm", ""),
        )

    ax_grad.set_ylabel("gradient L2 norm")
    ax_grad.set_title("gradient size by layer")
    ax_grad.grid(alpha=0.25)
    ax_grad.legend(loc="upper right")

    ax_update.set_xlabel("training step")
    ax_update.set_ylabel("SGD update L2 norm")
    ax_update.set_title("parameter update size by layer")
    ax_update.grid(alpha=0.25)
    ax_update.legend(loc="upper right")

    fig.suptitle("milestone 2: gradient and update diagnostics")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_first_layer_boundaries(X, y, snapshots, output_path):
    """Render first-layer ReLU boundary lines from multiple training steps.

    Parameters:
        X: A NumPy array of 2D training inputs to draw as scatter points.
        y: A NumPy array of ``-1/+1`` labels used to color the scatter points.
        snapshots: A list of dictionaries. Each dictionary must contain
            ``step``, ``accuracy``, ``loss``, and ``first_layer`` keys. The
            ``first_layer`` value comes from ``first_layer_boundary_snapshot``.
        output_path: File path where the PNG image should be written.

    Returns:
        None. The function writes the plot to ``output_path``.
    """
    pad = 0.35
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xs = np.linspace(x_min, x_max, 120)

    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    colors = plt.cm.tab20(np.linspace(0, 1, len(snapshots[0]["first_layer"]["lines"])))

    for ax, snapshot in zip(axes.ravel(), snapshots):
        ax.scatter(
            X[:, 0],
            X[:, 1],
            c=y,
            cmap="bwr",
            edgecolors="black",
            linewidths=0.5,
            s=30,
            zorder=3,
        )

        lines = snapshot["first_layer"]["lines"]
        dead_mask = snapshot["first_layer"]["dead_mask"]
        for neuron_index, ((w1, w2, b), is_dead) in enumerate(zip(lines, dead_mask)):
            color = "0.6" if is_dead else colors[neuron_index]
            linestyle = "--" if is_dead else "-"
            alpha = 0.45 if is_dead else 0.8
            if abs(w2) > 1e-12:
                ys = -(w1 * xs + b) / w2
                ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=1, alpha=alpha)
            elif abs(w1) > 1e-12:
                x = -b / w1
                ax.axvline(x, color=color, linestyle=linestyle, linewidth=1, alpha=alpha)

        dead_count = int(np.sum(dead_mask))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(
            f"step {snapshot['step']}  "
            f"dead {dead_count}/{len(lines)}  "
            f"acc {snapshot['accuracy'] * 100:.0f}%"
        )
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("milestone 3: first-layer ReLU activation boundaries", fontsize=15)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_dead_relu_counts(metrics, output_path):
    """Render dead-ReLU counts for nonlinear layers over training.

    Parameters:
        metrics: A list of dictionaries with one entry per training step. Rows
            should include ``layer_N_dead_relu_count`` values.
        output_path: File path where the PNG image should be written.

    Returns:
        None. The function writes the plot to ``output_path``.
    """
    dead_keys = [
        key
        for key in metrics[0]
        if key.startswith("layer_") and key.endswith("_dead_relu_count")
    ]
    steps = [row["step"] for row in metrics]

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for key in dead_keys:
        ax.step(
            steps,
            [row[key] for row in metrics],
            where="post",
            label=key.replace("_dead_relu_count", ""),
        )

    ax.set_xlabel("training step")
    ax.set_ylabel("dead ReLU count")
    ax.set_title("dead ReLU neurons by layer")
    ax.set_ylim(bottom=-0.25)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.suptitle("milestone 3: dead-ReLU diagnostics")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def html_snapshot_payload(snapshot):
    """Convert one training snapshot into JSON-friendly data for the browser.

    Parameters:
        snapshot: A snapshot dictionary captured during training. It contains
            scalar metrics, a decision grid, and first-layer boundary data.

    Returns:
        A dictionary made only from JSON-serializable Python values.
    """
    xx, yy, zz = snapshot["grid"]
    first_layer = snapshot["first_layer"]
    return {
        "step": snapshot["step"],
        "loss": snapshot["loss"],
        "accuracy": snapshot["accuracy"],
        "grid": {
            "xMin": float(xx[0, 0]),
            "xMax": float(xx[0, -1]),
            "yMin": float(yy[0, 0]),
            "yMax": float(yy[-1, 0]),
            "rows": int(zz.shape[0]),
            "cols": int(zz.shape[1]),
            "scores": np.round(zz, 6).reshape(-1).tolist(),
        },
        "firstLayer": {
            "lines": [
                [float(w1), float(w2), float(b)]
                for w1, w2, b in first_layer["lines"]
            ],
            "deadMask": [bool(value) for value in first_layer["dead_mask"]],
        },
    }


def write_interactive_data(X, y, metrics, snapshots, data_path):
    """Write the JSON payload consumed by the standalone milestone 4 HTML app.

    Parameters:
        X: A NumPy array of 2D training inputs.
        y: A NumPy array of ``-1/+1`` labels.
        metrics: A list of dictionaries with full per-step training metrics.
        snapshots: A list of snapshot dictionaries captured at selected steps.
        data_path: File path where the JSON data payload should be written.

    Returns:
        None. The function writes the visualizer data to ``data_path``.
    """
    data = {
        "points": np.round(X, 6).tolist(),
        "labels": [int(value) for value in y],
        "metrics": metrics,
        "snapshots": [html_snapshot_payload(snapshot) for snapshot in snapshots],
    }
    data_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_metrics(metrics, output_path):
    """Write the per-step training metrics to a CSV file.

    Parameters:
        metrics: A list of dictionaries. Every dictionary should have the same
            keys, including core training metrics and any optional diagnostic
            metrics.
        output_path: File path where the CSV file should be written.

    Returns:
        None. The function writes rows to ``output_path``.
    """
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)


def train_and_capture():
    """Train the micrograd MLP and save milestone visualization artifacts.

    Parameters:
        None.

    Returns:
        The final metrics dictionary from the last training step. It contains
        ``step``, ``loss``, and ``accuracy`` values.

    Notes:
        This is the main pipeline for milestones 1, 2, 3, and 4. It creates the
        dataset, trains the model with micrograd backprop, captures
        decision-boundary snapshots, records layer-wise gradient/update norms,
        tracks dead ReLUs, then writes plots, CSV metrics, and milestone 4 JSON
        data under ``OUT_DIR``. The milestone 4 HTML file is a standalone static
        app that reads that JSON file.
    """
    random.seed(1337)
    np.random.seed(1337)
    OUT_DIR.mkdir(exist_ok=True)

    X, y = build_dataset()
    model = MLP(2, [16, 16, 1])
    stat_keys = layer_stat_keys(model) + dead_relu_stat_keys(model)

    metrics = []
    snapshots = []

    for step in range(101):
        total_loss, accuracy = evaluate_loss(model, X, y)
        metric_row = {
            "step": step,
            "loss": total_loss.data,
            "accuracy": accuracy,
            "learning_rate": None,
        }
        metric_row.update({key: None for key in stat_keys})
        metric_row.update(collect_dead_relu_stats(model, X))
        metrics.append(metric_row)

        if step in SNAPSHOT_STEPS:
            snapshots.append(
                {
                    "step": step,
                    "loss": total_loss.data,
                    "accuracy": accuracy,
                    "grid": predict_grid(model, X),
                    "first_layer": first_layer_boundary_snapshot(model, X),
                }
            )

        if step == 100:
            break

        model.zero_grad()
        total_loss.backward()
        learning_rate = 1.0 - 0.9 * step / 100
        metric_row["learning_rate"] = learning_rate
        metric_row.update(collect_layer_stats(model, learning_rate))
        for p in model.parameters():
            p.data -= learning_rate * p.grad

    plot_decision_snapshots(
        X, y, snapshots, OUT_DIR / "milestone1_decision_snapshots.png"
    )
    plot_training_curves(metrics, OUT_DIR / "milestone1_training_curves.png")
    plot_layer_norms(metrics, OUT_DIR / "milestone2_layer_norms.png")
    plot_first_layer_boundaries(
        X, y, snapshots, OUT_DIR / "milestone3_first_layer_boundaries.png"
    )
    plot_dead_relu_counts(metrics, OUT_DIR / "milestone3_dead_relu_counts.png")
    write_interactive_data(X, y, metrics, snapshots, OUT_DIR / "milestone4_data.json")
    write_metrics(metrics, OUT_DIR / "milestone1_metrics.csv")
    write_metrics(metrics, OUT_DIR / "milestone2_metrics.csv")
    write_metrics(metrics, OUT_DIR / "milestone3_metrics.csv")

    return metrics[-1]


if __name__ == "__main__":
    final = train_and_capture()
    print(
        "wrote visualizer_outputs; "
        f"final step={final['step']} "
        f"loss={final['loss']:.4f} "
        f"accuracy={final['accuracy'] * 100:.1f}%"
    )
