# OES Model Advisor

The Model Advisor is a transparent, deterministic screening layer. It profiles the
uploaded data, uses the stated research objective and variable roles, and ranks only
models that are registered and executable in the current OES engine.

## Python API

```python
import pandas as pd
from oes_engine.advisor import profile_data, recommend_models

df = pd.read_csv("my_panel.csv")
profile = profile_data(df, time="year", unit="country")
result = recommend_models(
    df,
    objective="causal",
    y="emissions",
    x=["income", "energy_use"],
    time="year",
    unit="country",
    treatment="policy",
)
```

Supported objectives: `explanatory`, `prediction`, `forecasting`, `causal`,
and `descriptive`.

## Recommendation principles

- Recommendations are restricted to IDs returned by `oes_engine.catalog()`.
- The score is a transparent rule-based ranking, not a probability that a model is correct.
- User-supplied integration and cointegration assessments are not inferred from the
  raw data. The advisor recommends preliminary tests and states the relevant conditions.
- A recommendation includes the reasons, conditions to check, guide URL, input fields,
  and a built-in example when available.
- The advisor never claims that an instrument is valid, parallel trends hold, regressors
  are exogenous, or a model is identified merely because columns were selected.

## Run tests

```bash
cd oes-engine
python -m pytest tests/test_advisor.py -q
```

## Roadmap to 1,000 models

The current engine catalogue is the executable source of truth. Do not inflate the
executable count with aliases, lag orders, covariance options, or parameter grids.
New estimators should be added as separately documented implementations with:
1. an explicit model specification and input schema;
2. numerical/reference tests;
3. diagnostics and failure handling;
4. guide/example data;
5. browser-runtime compatibility checks.

The 1,000-model target is a staged engineering programme. Only models that pass the
validation suite should appear as executable in the public catalogue.
