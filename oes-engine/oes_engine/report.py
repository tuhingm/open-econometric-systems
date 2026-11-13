"""Print OES result tables and draw OES charts in a script or notebook.

    r = oes_engine.run("bvar", open("money.csv").read(), spec)
    oes_engine.show(r, "Equation: log_gdp")       # every table whose title starts with the text
    oes_engine.plot(r, "Response of log_gdp")     # charts whose title starts with the text (matplotlib)
"""
import pandas as pd


def tables(result, prefix=""):
    if "error" in result:
        raise ValueError(result["error"])
    return [t for t in result["tables"] if t["title"].startswith(prefix)]


def frame(table):
    df = pd.DataFrame(table["rows"], columns=table["columns"])
    return df.set_index(df.columns[0])


def show(result, prefix="", rows=None):
    """Print the tables whose titles start with prefix (all tables if prefix is empty)."""
    found = tables(result, prefix)
    if not found:
        raise KeyError(f"No table starts with {prefix!r}. Titles: " + "; ".join(t["title"] for t in result["tables"]))
    with pd.option_context("display.width", 140, "display.max_columns", 20, "display.max_colwidth", 60):
        for t in found:
            df = frame(t)
            if rows:
                df = df.head(rows)
            print(t["title"])
            print(df.to_string())
            if t.get("note"):
                print("Note:", t["note"])
            print()


def summary(result):
    for s in result["summary"]:
        print(f"{s['label']}: {s['value']}")


def plot(result, prefix="", ncols=3, size=(3.3, 2.4)):
    import matplotlib.pyplot as plt
    found = [c for c in result["charts"] if c["title"].startswith(prefix)]
    if not found:
        raise KeyError(f"No chart starts with {prefix!r}.")
    nrow = (len(found) + ncols - 1) // ncols
    ncol = min(ncols, len(found))
    fig, axes = plt.subplots(nrow, ncol, figsize=(size[0] * ncol, size[1] * nrow), squeeze=False)
    for ax, c in zip(axes.ravel(), found):
        x = list(range(len(c["x"])))
        if c.get("band"):
            lo = [v if v is not None else float("nan") for v in c["band"]["lo"]]
            hi = [v if v is not None else float("nan") for v in c["band"]["hi"]]
            ax.fill_between(x, lo, hi, color="#9fd3f0", alpha=.5, lw=0)
        for s in c["series"]:
            y = [v if v is not None else float("nan") for v in s["y"]]
            if c.get("kind") == "bar":
                ax.bar(x, y, label=s["name"])
            elif c.get("kind") == "scatter":
                ax.plot(x, y, ".", ms=3, label=s["name"])
            else:
                ax.plot(x, y, lw=1.3, label=s["name"])
        step = max(1, len(x) // 5)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([str(v) for v in c["x"][::step]], fontsize=7)
        ax.set_title(c["title"], fontsize=8)
        if len(c["series"]) > 1:
            ax.legend(fontsize=6)
    for ax in axes.ravel()[len(found):]:
        ax.axis("off")
    fig.tight_layout()
    return fig
