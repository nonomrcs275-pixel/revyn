#!/usr/bin/env python3
"""Mini IA de trading (didactique) en Python standard.

Ce script entraîne une petite régression logistique maison pour prédire
la direction du prochain mouvement de prix, puis simule une stratégie
long/short basique sur un jeu de test.

⚠️  Ce code est éducatif, pas un conseil financier.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass
class Model:
    weights: List[float]
    bias: float


def sigmoid(x: float) -> float:
    x = max(min(x, 60.0), -60.0)
    return 1.0 / (1.0 + math.exp(-x))


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def load_close_prices(csv_path: str) -> List[float]:
    prices: List[float] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "Close" not in reader.fieldnames:
            raise ValueError("Le CSV doit contenir une colonne 'Close'.")
        for row in reader:
            try:
                prices.append(float(row["Close"]))
            except (ValueError, TypeError):
                continue
    if len(prices) < 50:
        raise ValueError("Pas assez de données: au moins 50 lignes de prix 'Close'.")
    return prices


def generate_synthetic_prices(length: int = 800, seed: int = 42) -> List[float]:
    random.seed(seed)
    prices = [100.0]
    for _ in range(length - 1):
        regime = 0.0004 if random.random() > 0.45 else -0.0002
        noise = random.gauss(0.0, 0.008)
        prices.append(prices[-1] * (1.0 + regime + noise))
    return prices


def signal_from_proba(proba: float, buy_th: float = 0.55, sell_th: float = 0.45) -> int:
    if proba > buy_th:
        return 1
    if proba < sell_th:
        return -1
    return 0


def build_samples(prices: Sequence[float], lookback: int = 8) -> Tuple[List[List[float]], List[int], List[float]]:
    """Construit X, y et les rendements futurs pour le backtest."""
    returns = [(prices[i] / prices[i - 1]) - 1.0 for i in range(1, len(prices))]
    x_data: List[List[float]] = []
    y_data: List[int] = []
    future_returns: List[float] = []

    for t in range(lookback, len(returns) - 1):
        window = returns[t - lookback : t]
        ret_1 = returns[t]
        ret_3 = sum(returns[t - 2 : t + 1])
        momentum = prices[t + 1] / prices[t + 1 - lookback] - 1.0
        volatility = std(window)
        ma_short = mean(prices[t - 3 : t + 1])
        ma_long = mean(prices[t - lookback : t + 1])
        ma_gap = (ma_short / ma_long) - 1.0 if ma_long else 0.0

        x_data.append([ret_1, ret_3, momentum, volatility, ma_gap])

        fwd = returns[t + 1]
        y_data.append(1 if fwd > 0 else 0)
        future_returns.append(fwd)

    return x_data, y_data, future_returns


def standardize(x_data: List[List[float]]) -> Tuple[List[List[float]], List[float], List[float]]:
    n_features = len(x_data[0])
    means = [mean([row[i] for row in x_data]) for i in range(n_features)]
    stds = [std([row[i] for row in x_data]) or 1.0 for i in range(n_features)]
    normed = [[(row[i] - means[i]) / stds[i] for i in range(n_features)] for row in x_data]
    return normed, means, stds


def apply_standardization(x_data: List[List[float]], means: Sequence[float], stds: Sequence[float]) -> List[List[float]]:
    return [[(row[i] - means[i]) / (stds[i] or 1.0) for i in range(len(means))] for row in x_data]


def train_logistic_regression(x_train: List[List[float]], y_train: List[int], epochs: int = 250, lr: float = 0.05) -> Model:
    n_features = len(x_train[0])
    weights = [0.0] * n_features
    bias = 0.0

    for _ in range(epochs):
        for x, y in zip(x_train, y_train):
            pred = sigmoid(dot(weights, x) + bias)
            error = pred - y
            for i in range(n_features):
                weights[i] -= lr * error * x[i]
            bias -= lr * error

    return Model(weights=weights, bias=bias)


def predict_proba(model: Model, x_data: List[List[float]]) -> List[float]:
    return [sigmoid(dot(model.weights, x) + model.bias) for x in x_data]


def accuracy(probas: Sequence[float], y_true: Sequence[int], threshold: float = 0.5) -> float:
    preds = [1 if p >= threshold else 0 for p in probas]
    correct = sum(int(a == b) for a, b in zip(preds, y_true))
    return correct / len(y_true) if y_true else 0.0


def backtest(probas: Sequence[float], future_returns: Sequence[float], buy_th: float = 0.55, sell_th: float = 0.45) -> Tuple[float, float, float]:
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    strat_returns: List[float] = []

    for p, r in zip(probas, future_returns):
        position = signal_from_proba(p, buy_th=buy_th, sell_th=sell_th)

        trade_r = position * r
        strat_returns.append(trade_r)
        equity *= 1.0 + trade_r
        peak = max(peak, equity)
        dd = (equity / peak) - 1.0
        max_drawdown = min(max_drawdown, dd)

    sharpe = 0.0
    s = std(strat_returns)
    if s > 0:
        sharpe = (mean(strat_returns) / s) * math.sqrt(252)

    return equity - 1.0, max_drawdown, sharpe


def export_signals(path: str, probas: Sequence[float], future_returns: Sequence[float], buy_th: float, sell_th: float) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "proba_up", "signal", "future_return"])
        for idx, (p, r) in enumerate(zip(probas, future_returns), start=1):
            writer.writerow([idx, f"{p:.6f}", signal_from_proba(p, buy_th, sell_th), f"{r:.6f}"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini IA de trading (régression logistique maison).")
    parser.add_argument("--csv", help="Chemin CSV avec colonne Close", default=None)
    parser.add_argument("--split", type=float, default=0.7, help="Part train entre 0 et 1")
    parser.add_argument("--buy-th", type=float, default=0.55, help="Seuil d'achat")
    parser.add_argument("--sell-th", type=float, default=0.45, help="Seuil de vente")
    parser.add_argument("--export-signals", default=None, help="Export CSV des signaux de test")
    args = parser.parse_args()

    if not (0.0 <= args.sell_th < args.buy_th <= 1.0):
        raise ValueError("Les seuils doivent respecter 0 <= sell_th < buy_th <= 1.")

    prices = load_close_prices(args.csv) if args.csv else generate_synthetic_prices()
    x_data, y_data, future_returns = build_samples(prices)

    split_idx = int(len(x_data) * args.split)
    if split_idx < 30 or split_idx >= len(x_data) - 10:
        raise ValueError("Split invalide: pas assez de données train/test.")

    x_train_raw, x_test_raw = x_data[:split_idx], x_data[split_idx:]
    y_train, y_test = y_data[:split_idx], y_data[split_idx:]
    fwd_test = future_returns[split_idx:]

    x_train, means, stds = standardize(x_train_raw)
    x_test = apply_standardization(x_test_raw, means, stds)

    model = train_logistic_regression(x_train, y_train)
    probas_test = predict_proba(model, x_test)

    acc = accuracy(probas_test, y_test)
    total_return, max_dd, sharpe = backtest(
        probas_test,
        fwd_test,
        buy_th=args.buy_th,
        sell_th=args.sell_th,
    )

    last_proba = probas_test[-1]
    next_signal = signal_from_proba(last_proba, buy_th=args.buy_th, sell_th=args.sell_th)

    print("=== Résultats IA Trading ===")
    print(f"Données: {len(prices)} prix | train={len(x_train)} | test={len(x_test)}")
    print(f"Seuils: achat>{args.buy_th:.2f} | vente<{args.sell_th:.2f}")
    print(f"Accuracy directionnelle (test): {acc:.2%}")
    print(f"Performance stratégie (test): {total_return:.2%}")
    print(f"Max drawdown: {max_dd:.2%}")
    print(f"Sharpe approx.: {sharpe:.2f}")
    print(f"Dernière proba UP: {last_proba:.2%} | Signal suivant: {next_signal} (1=long, 0=flat, -1=short)")

    if args.export_signals:
        export_signals(args.export_signals, probas_test, fwd_test, args.buy_th, args.sell_th)
        print(f"Signaux exportés vers: {args.export_signals}")

    print("\n⚠️ Usage éducatif uniquement. Valider avec frais, slippage, et gestion du risque réelle.")


if __name__ == "__main__":
    main()
