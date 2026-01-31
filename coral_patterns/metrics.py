import math

NEIGHBORS8 = [
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (-1, 1), (1, -1), (-1, -1)
]


def tip_count(cluster):
    """Tips = sites with exactly 1 occupied neighbor in 8-neighborhood."""
    tips = 0
    for (x, y) in cluster:
        n = 0
        for dx, dy in NEIGHBORS8:
            if (x + dx, y + dy) in cluster:
                n += 1
        if n == 1:
            tips += 1
    return tips


def bbox_metrics(cluster):
    xs = [p[0] for p in cluster]
    ys = [p[1] for p in cluster]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    w = (xmax - xmin) + 1
    h = (ymax - ymin) + 1
    return xmin, xmax, ymin, ymax, w, h


def compute_metrics(cluster, origin, max_r2):
    M = len(cluster)
    _, _, _, _, w, h = bbox_metrics(cluster)

    verticality = h / (w + 1e-12)
    compactness = M / ((w * h) + 1e-12)

    tips = tip_count(cluster)
    tips_frac = tips / max(M, 1)

    R = math.sqrt(float(max_r2))

    return {
        "M": int(M),
        "R": float(R),
        "width": int(w),
        "height": int(h),
        "verticality": float(verticality),
        "compactness": float(compactness),
        "tips": int(tips),
        "tips_frac": float(tips_frac),
    }
