"""Six-plane metrics and device groups, independent of ML dependencies."""

from collections import Counter


def confusion_metrics(confusion):
    if len(confusion) != 6 or any(len(row) != 6 for row in confusion):
        raise ValueError("Expected a 6x6 confusion matrix")
    if any(type(v) is not int or v < 0 for row in confusion for v in row):
        raise ValueError("Confusion entries must be nonnegative integers")
    support = list(map(sum, confusion))
    predicted = [sum(row[c] for row in confusion) for c in range(6)]
    count = sum(support)
    if not count:
        raise ValueError("Empty confusion matrix")
    tp = [confusion[i][i] for i in range(6)]
    f1 = [2 * tp[i] / max(1, support[i] + predicted[i]) for i in range(6)]
    recall = [tp[i] / max(1, support[i]) for i in range(6)]
    present = [i for i, n in enumerate(support) if n]
    return dict(
        accuracy=sum(tp) / count,
        macro_f1=sum(f1) / 6,
        balanced_accuracy=sum(recall[i] for i in present) / len(present),
        per_class_f1=f1,
        per_class_recall=recall,
        per_class_precision=[tp[i] / max(1, predicted[i]) for i in range(6)],
        per_class_support=support,
        n_classes_present=len(present),
        confusion=confusion,
    )


def evaluate_predictions(labels, predictions, records):
    if len(labels) != len(predictions) or len(labels) != len(records) or not labels:
        raise ValueError("Prediction records must align with a nonempty dataset")
    if labels != [r["label"] for r in records]:
        raise ValueError("Prediction order does not match image metadata")

    def measure(indices):
        matrix = [[0] * 6 for _ in range(6)]
        for i in indices:
            if predictions[i] not in range(6):
                raise ValueError("Prediction outside the six-class vocabulary")
            matrix[labels[i]][predictions[i]] += 1
        return (
            dict(
                n_images=len(indices),
                n_patients=len({records[i]["patient_id"] for i in indices}),
                **confusion_metrics(matrix),
            )
            if indices
            else None
        )

    groups = {
        name: [i for i, r in enumerate(records) if r["machine"] == name]
        for name in sorted({r["machine"] for r in records})
    }
    groups["seen_E6_Aloka"] = [i for i, r in enumerate(records) if r["machine"] in ("Voluson E6", "Aloka")]
    groups["near_unseen_S10"] = [i for i, r in enumerate(records) if r["machine"] == "Voluson S10"]
    return dict(
        overall=measure(list(range(len(records)))),
        devices={name: measure(ids) for name, ids in groups.items()},
    )


def dataset_composition(records, indices):
    rows = []
    for split, ids in indices.items():
        counts = Counter(records[i]["machine"] for i in ids)
        for machine in sorted({r["machine"] for r in records}):
            rows.append(
                dict(
                    split=split,
                    machine=machine,
                    images=counts[machine],
                    patients=len({records[i]["patient_id"] for i in ids if records[i]["machine"] == machine}),
                    percent_of_split=100 * counts[machine] / len(ids) if ids else None,
                    total_split_images=len(ids),
                )
            )
    return rows
