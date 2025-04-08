import marimo

__generated_with = "0.12.4"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
def _(pd):
    cpi = pd.read_parquet("workspace/cpi_fred_factors.pq")
    return (cpi,)


@app.cell
def _(pd):
    tlcpi = pd.read_parquet("workspace/cpi_turnleaf_factors.pq")
    return (tlcpi,)


@app.cell
def _(tlcpi):
    print(tlcpi.head())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
