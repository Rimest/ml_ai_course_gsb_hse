"""Открытый интерфейс моделей и единые экономические правила задания 3.

Кейс: выходной контроль партий продукции. Модели экспортированы из
scikit-learn в JSON; для их применения достаточно numpy и pandas.
Этот файл под именем hw3_runtime.py выдаётся студентам в homeworks/hw3_quality_control/ репозитория курса.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ID = "batch_id"
FEATURES = ["supplier_score", "tool_wear", "solder_temp_dev", "humidity",
            "seal_pressure_dev", "night_shift", "new_product", "batch_size",
            "complexity", "inline_alarms"]
TYPE_NAMES = ["Размеры", "Электрика", "Герметичность"]
METHOD_NAMES = ["Замер размеров", "Электротест", "Тест герметичности"]
TASKS = ["defect", "type", "duration", "detect"]
K_MAX = 24
LABOR_PER_HOUR = 1800.0
# Расходные материалы одной проверки методом 0/1/2, рубли.
METHOD_COST = np.array([500.0, 1400.0, 900.0])
# Во сколько раз потери от отгрузки дефектной партии больше её стоимости.
LOSS_MULTIPLIER = np.array([1.0, 1.4, 3.5])


def overload_cost(k):
    """Дневная надбавка C(K): организация и сверхурочная работа лаборатории при k проверках, рубли."""
    n = np.asarray(k, dtype=float)
    return 900.0 * n + 1800.0 * np.maximum(n - 6, 0) ** 2


def _tree(tree, x):
    feature = np.asarray(tree["feature"])
    threshold = np.asarray(tree["threshold"])
    left, right = np.asarray(tree["left"]), np.asarray(tree["right"])
    nodes = np.zeros(len(x), dtype=int)
    while True:
        active = np.flatnonzero(feature[nodes] >= 0)
        if not len(active):
            break
        at = nodes[active]
        go_left = x[active, feature[at]] <= threshold[at]
        nodes[active] = np.where(go_left, left[at], right[at])
    return np.asarray(tree["value"])[nodes]


class FrozenModel:
    """Готовая модель; predict/predict_proba без обучения и без pickle.

    Модели времени (duration) и обнаружения (detect) получают, кроме признаков
    партии, код метода проверки method: 0, 1 или 2 для каждой строки.
    """
    def __init__(self, spec):
        self.spec = spec
        self.name = spec["name"]
        self.task = spec["task"]
        self.note = spec.get("note", "")
        self.classes_ = np.asarray(spec.get("classes", []))

    def _input(self, features, method=None):
        x = features[FEATURES].to_numpy(dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("В признаках есть пропуски или бесконечности")
        if self.task in ("duration", "detect"):
            if method is None:
                raise ValueError("Нужен method: массив кодов метода 0, 1, 2 длиной в число строк")
            m = np.asarray(method)
            if m.ndim == 0:
                m = np.full(len(x), int(m))
            if m.shape != (len(x),):
                raise ValueError("Длина method должна совпадать с числом строк features")
            if not np.isin(m, [0, 1, 2]).all():
                raise ValueError("Коды method: 0, 1, 2")
            x = np.column_stack([x, np.eye(3)[m.astype(int)]])
        return x[:, self.spec["columns"]]

    def _raw(self, features, method=None):
        x = self._input(features, method)
        s = self.spec
        if s["kind"] == "linear":
            x = (x - np.asarray(s["mean"])) / np.asarray(s["scale"])
            return x @ np.asarray(s["coef"]).T + np.asarray(s["intercept"])
        # scikit-learn использует float32 при проходе по деревьям.
        x = x.astype(np.float32)
        if s["kind"] == "forest":
            return sum(_tree(t, x) for t in s["trees"]) / len(s["trees"])
        result = np.full(len(x), s["initial"], dtype=float)
        for t in s["trees"]:
            result += s["learning_rate"] * _tree(t, x).ravel()
        return result

    def predict_proba(self, features, method=None):
        if self.task == "duration":
            raise ValueError("Модель времени возвращает часы через predict")
        z = self._raw(features, method)
        if self.spec["kind"] == "forest":
            return z
        if z.ndim == 1 or z.shape[1] == 1:
            p = 1 / (1 + np.exp(-np.clip(z.ravel(), -40, 40)))
            return np.column_stack([1 - p, p])
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, features, method=None):
        if self.task == "duration":
            raw = self._raw(features, method).ravel()
            if self.spec.get("target") == "log":
                raw = np.exp(raw)
            return np.clip(raw, 0.25, 30)
        return self.classes_[self.predict_proba(features, method).argmax(axis=1)]


def load_models(path):
    """Словарь models[задача][вариант]; варианты M1, M2, M3."""
    specs = json.loads(Path(path).read_text(encoding="utf-8"))
    return {task: {s["name"]: FrozenModel(s) for s in specs if s["task"] == task}
            for task in TASKS}


def validate_actions(features, actions):
    """Проверяет полный CSV; возвращает его в порядке строк features."""
    if list(actions.columns) != [ID, "action"]:
        raise ValueError(f"Нужны ровно два столбца: {ID},action")
    if actions[ID].isna().any() or actions[ID].duplicated().any():
        raise ValueError(f"{ID} не должен содержать пропуски или повторения")
    if len(actions) != len(features) or set(actions[ID]) != set(features[ID]):
        raise ValueError("Нужна одна строка на каждую партию исходной выборки")
    if not pd.api.types.is_numeric_dtype(actions.action) or not np.isfinite(actions.action).all():
        raise ValueError("action должен содержать конечные числа")
    if not actions.action.isin([-1, 0, 1, 2]).all():
        raise ValueError("action: -1 — не проверять; 0, 1, 2 — метод проверки")
    aligned = actions.set_index(ID).loc[features[ID]].reset_index()
    counts = pd.Series((aligned.action.to_numpy() >= 0).astype(int)).groupby(
        features.day.to_numpy()).sum()
    if (counts > K_MAX).any():
        raise ValueError(f"Превышен дневной лимит {K_MAX}: {counts[counts > K_MAX].to_dict()}")
    aligned["action"] = aligned.action.astype(int)
    return aligned


def evaluate(features, outcomes, actions):
    """Одинаковый расчёт на открытых данных и закрытом тесте.

    outcomes содержит ожидаемые предотвращённые потери и средние часы для каждого
    метода. Возвращает итог и таблицу по дням; изменять оценщик нельзя.
    """
    aligned = validate_actions(features, actions)
    if outcomes.duplicated([ID, "action"]).any():
        raise ValueError("Повторяющиеся исходы")
    expected = pd.MultiIndex.from_product([features[ID], range(3)])
    indexed = outcomes.set_index([ID, "action"])
    if len(indexed) != len(expected) or len(expected.difference(indexed.index)):
        raise ValueError("Нужны исходы всех трёх методов для каждой партии")
    selected = aligned[aligned.action >= 0].merge(outcomes, on=[ID, "action"],
                                                validate="one_to_one")
    selected = selected.merge(features[[ID, "day"]], on=ID, validate="one_to_one")
    selected["consumables"] = METHOD_COST[selected.action.to_numpy()]
    selected["labor"] = selected.hours.to_numpy() * LABOR_PER_HOUR
    daily = selected.groupby("day").agg(k=("action", "size"),
        prevented=("prevented", "sum"), consumables=("consumables", "sum"),
        labor=("labor", "sum"))
    daily = daily.reindex(sorted(features.day.unique()), fill_value=0)
    daily["overload"] = overload_cost(daily.k)
    daily["effect"] = daily.prevented - daily.consumables - daily.labor - daily.overload
    totals = {key: float(daily[key].sum())
              for key in ["prevented", "consumables", "labor", "overload", "effect"]}
    totals.update(days=len(daily), effect_per_day=float(daily.effect.mean()),
                  mean_k=float(daily.k.mean()))
    return totals, daily


def plot_economics(daily):
    """Готовое оформление графиков результата; возвращает matplotlib Figure."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    axes[0].hist(daily.k, bins=np.arange(-0.5, K_MAX + 1.5), color="#1764ab")
    axes[0].set(xlabel="Проверок за день", ylabel="Дней", title="Загрузка лаборатории")
    axes[1].plot(daily.index, daily.effect / 1000)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(xlabel="День", ylabel="Экономия J, тыс. руб.", title="Экономия J по дням")
    axes[2].bar(["Предотвращено", "Расходники", "Работа", "Надбавка C(K)"],
                [daily.prevented.mean(), -daily.consumables.mean(),
                 -daily.labor.mean(), -daily.overload.mean()])
    axes[2].set(ylabel="В среднем, руб./день", title="Составляющие экономии J")
    axes[2].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig


def baseline_actions(features, models, kind="defect_value", k=6):
    """Простые базовые правила с моделями M1.

    none — не проверять; defect — первые k партий дня по вероятности дефекта;
    defect_value — по произведению вероятности дефекта и стоимости партии.
    Метод проверки — самый вероятный тип дефекта по модели type M1.
    """
    if kind not in ("none", "defect", "defect_value") or not 0 <= k <= K_MAX:
        raise ValueError("Неизвестное базовое правило или недопустимое k")
    action = np.full(len(features), -1)
    if kind != "none":
        score = models["defect"]["M1"].predict_proba(features)[:, 1]
        if kind == "defect_value":
            score = score * features.batch_value_rub.to_numpy()
        method = models["type"]["M1"].predict(features)
        days = features.day.to_numpy()
        for day in sorted(features.day.unique()):
            ids = np.flatnonzero(days == day)
            chosen = ids[np.argsort(-score[ids], kind="stable")[:k]]
            action[chosen] = method[chosen]
    return pd.DataFrame({ID: features[ID], "action": action})


def plot_metric_table(table, pairs):
    """Рисует выбранные пары (задача, метрика) из длинной таблицы результатов."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 3.5), squeeze=False)
    for ax, (task, metric) in zip(axes[0], pairs):
        data = table[(table.task == task) & (table.metric == metric)]
        ax.bar(data.model, data.value, color="#1764ab")
        ax.set(title=f"{task}: {metric}", xlabel="Модель", ylabel=metric)
    fig.tight_layout()
    return fig


def stability_score(strategy_daily, baseline_daily, seed=42):
    """5-й процентиль средних парных разностей дневной экономии, рубли за день.

    На вход — дневные таблицы evaluate на одних и тех же контрольных днях.
    10 000 повторных выборок целых дней с возвращением; фиксированный seed.
    """
    for table in (strategy_daily, baseline_daily):
        if not table.index.is_unique or table.index.hasnans:
            raise ValueError("Индекс должен содержать уникальные дни без пропусков")
        if "effect" not in table:
            raise ValueError("Нужен столбец effect: экономия за каждый день")
    if len(strategy_daily) < 2:
        raise ValueError("Для оценки устойчивости нужны хотя бы два дня")
    if set(strategy_daily.index) != set(baseline_daily.index):
        raise ValueError("Ваше правило и базовое правило сравнивают на одинаковых днях")
    left = strategy_daily.sort_index()
    right = baseline_daily.reindex(left.index)
    values = left.effect.to_numpy(dtype=float) - right.effect.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Экономия должна содержать конечные числа")
    rng = np.random.default_rng(seed)
    means = np.empty(10_000)
    for start in range(0, len(means), 500):
        indices = rng.integers(0, len(values), size=(500, len(values)))
        means[start:start + 500] = values[indices].mean(axis=1)
    return float(np.percentile(means, 5))
