# revyn

## IA de trading (exemple éducatif)

Le fichier `trading_ai.py` contient une mini IA de trading en Python (sans dépendances externes) qui :

- charge des prix de clôture (`Close`) depuis un CSV, ou génère des données synthétiques,
- construit des features techniques simples,
- entraîne une régression logistique "maison",
- backteste une stratégie long/short basique,
- affiche des métriques (accuracy, performance, drawdown, sharpe approx.),
- donne un **signal suivant** (`1=long`, `0=flat`, `-1=short`),
- peut exporter un CSV de signaux de test.

## Je fais quoi maintenant ?

### 1) Lance une première exécution

```bash
python3 trading_ai.py
```

### 2) Branche tes propres données

```bash
python3 trading_ai.py --csv chemin/vers/prices.csv
```

Le CSV doit contenir au minimum une colonne `Close` (au moins 50 lignes).

### 3) Ajuste les seuils de décision

```bash
python3 trading_ai.py --csv chemin/vers/prices.csv --buy-th 0.60 --sell-th 0.40
```

- `buy-th`: au-dessus, le modèle prend un long.
- `sell-th`: en dessous, le modèle prend un short.
- entre les deux: pas de position.

### 4) Exporte les signaux pour les analyser

```bash
python3 trading_ai.py --csv chemin/vers/prices.csv --export-signals signals.csv
```

Le fichier exporté contient : `idx`, `proba_up`, `signal`, `future_return`.

---

> ⚠️ Ce projet est fourni à but pédagogique uniquement (pas un conseil en investissement).
