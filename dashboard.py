"""
Streamlit app para indicadores 'M' (GATS 2009, 2015, 2023) con matplotlib
"""

# ============================
# 0. IMPORTS & CONFIG
# ============================
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Indicadores M – GATS 2009/2015/2023", layout="wide")

# ============================
# 1. DATA
# ============================
@st.cache_data
def load_data(path="Datos_GATS_Completo.xlsx"):
    df = pd.read_excel(path)
    expected = {"indicador","grupo","categoria","anio","valor","inferior","superior"}
    missing = expected - set(df.columns)
    return df, missing

df, missing = load_data()
if missing:
    st.error(f"Faltan columnas: {missing}")
    st.stop()

@st.cache_data
def compute_deltas(df):
    wide = df.pivot_table(index=["indicador","grupo","categoria"],
                          columns="anio",
                          values=["valor","inferior","superior"])
    for comp in ["valor","inferior","superior"]:
        for y in [2009,2015,2023]:
            if (comp,y) not in wide.columns:
                wide[(comp,y)] = np.nan
    wide.columns = [f"{m}_{y}" for m,y in wide.columns]
    wide = wide.reset_index()
    wide["abs_change"] = wide["valor_2023"] - wide["valor_2009"]
    wide["rel_change"] = 100*wide["abs_change"] / wide["valor_2009"]
    def no_overlap(r):
        return not (r["inferior_2023"] <= r["superior_2009"] and r["superior_2023"] >= r["inferior_2009"])
    wide["cambio_relevante"] = wide.apply(no_overlap, axis=1)
    return wide

wide = compute_deltas(df)

# ============================
# 2. PLOTS (matplotlib)
# ============================

def plot_bars_grouped(sub_long):
    """Barras verticales agrupadas por año con IC95%."""
    data = sub_long.copy()
    years = sorted(data["anio"].unique())
    cats  = data["categoria"].unique().tolist()

    # Ordena por 2023 si existe
    if 2023 in years:
        order_vals = (data[data["anio"]==2023]
                      .set_index("categoria")["valor"]
                      .reindex(cats))
        cats = [c for c in order_vals.sort_values(ascending=False).index if c in cats]

    x = np.arange(len(cats))
    width = 0.25  # ancho de cada barra

    fig, ax = plt.subplots(figsize=(0.8*len(cats)*len(years)+2, 6))

    for i, y in enumerate(years):
        d   = data[data["anio"]==y].set_index("categoria")
        vals = [d.loc[c,"valor"] if c in d.index else np.nan for c in cats]
        li   = [d.loc[c,"inferior"] if c in d.index else np.nan for c in cats]
        ui   = [d.loc[c,"superior"] if c in d.index else np.nan for c in cats]

        ax.bar(x + (i - (len(years)-1)/2)*width, vals, width,
               label=str(y),
               yerr=[np.subtract(vals, li), np.subtract(ui, vals)],
               capsize=3, alpha=.9)

    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=45, ha='right')
    ax.set_ylabel("% (IC95%)")
    ax.grid(axis='y', alpha=.2)
    ax.legend(title="Año", ncol=len(years))
    fig.tight_layout()
    return fig

def plot_lines_facets(df_long, indicador):
    """Un subplot por grupo; líneas por categoría con IC como barras."""
    data = df_long[df_long["indicador"]==indicador].copy()
    years = sorted(data["anio"].unique())
    grupos = data["grupo"].unique()

    n = len(grupos)
    fig, axes = plt.subplots(n, 1, figsize=(9, 3*n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, g in zip(axes, grupos):
        d_g = data[data["grupo"]==g]
        for cat, d_cat in d_g.groupby("categoria"):
            d_cat = d_cat.sort_values("anio")
            ax.plot(d_cat["anio"], d_cat["valor"], marker="o", linewidth=2, label=cat)
            ax.vlines(d_cat["anio"], d_cat["inferior"], d_cat["superior"], linewidth=1)
        ax.set_ylabel("%")
        ax.set_title(g)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, ncol=2, loc="best")
    axes[-1].set_xlabel("Año")
    fig.tight_layout()
    return fig

# --- Avanzados (opcionales) ---
def plot_dumbbell(sub_wide):
    cats = sub_wide["categoria"].tolist()
    y = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(8, 0.4*len(cats)+2))
    ax.hlines(y, sub_wide["valor_2009"], sub_wide["valor_2023"], lw=2, alpha=.6)
    ax.plot(sub_wide["valor_2009"], y, 'o', label="2009")
    if "valor_2015" in sub_wide:
        ax.plot(sub_wide["valor_2015"], y, 'o', label="2015")
    ax.plot(sub_wide["valor_2023"], y, 'o', label="2023")
    ax.set_yticks(y)
    ax.set_yticklabels(cats)
    ax.invert_yaxis()
    ax.set_xlabel("%")
    ax.set_title("Cambios 2009–2023")
    ax.legend()
    fig.tight_layout()
    return fig

def plot_slopegraph(sub_wide):
    years = [c.split("_")[1] for c in sub_wide.columns if c.startswith("valor_")]
    years = sorted(set(map(int, years)))
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in sub_wide.iterrows():
        vals = [row[f"valor_{y}"] for y in years]
        ax.plot(years, vals, marker="o")
        ax.text(years[-1]+0.1, vals[-1], row["categoria"], va='center', fontsize=8)
    ax.set_xticks(years)
    ax.set_ylabel("%")
    ax.set_title("Slopegraph")
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    return fig

# ============================
# 3. UI
# ============================

st.title("Indicadores M del MPOWER – México (GATS 2009, 2015, 2023)")

st.sidebar.header("Filtros")
indicadores = sorted(df["indicador"].unique())
indicador_sel = st.sidebar.selectbox("Indicador", indicadores)

sub_df = df[df["indicador"]==indicador_sel]
grupos = sorted(sub_df["grupo"].unique())
grupo_sel = st.sidebar.selectbox("Grupo (para barras)", grupos)

sub_long = sub_df[sub_df["grupo"]==grupo_sel].copy()

# Hallazgos clave
st.subheader("Hallazgos clave")
msg_lines = []
for cat, g in sub_long.groupby("categoria"):
    try:
        v09 = float(g.loc[g.anio==2009,"valor"].values[0])
        v23 = float(g.loc[g.anio==2023,"valor"].values[0])
        delta = v23 - v09
        msg_lines.append((cat, v23, delta))
    except Exception:
        continue
msg_lines = sorted(msg_lines, key=lambda x: abs(x[2]), reverse=True)
if msg_lines:
    bullets = [f"- {cat}: {v23:.1f}% ({'+' if d>=0 else ''}{d:.1f} pts vs 2009)" for cat, v23, d in msg_lines[:5]]
    st.markdown("\n".join(bullets))
else:
    st.info("No se pudieron calcular cambios (faltan datos 2009/2023).")

# Visualización sencilla
st.subheader("Visualización sencilla")
viz_tipo = st.radio("Tipo de visualización", ["Barras agrupadas", "Líneas facetadas"], horizontal=True)

if viz_tipo == "Barras agrupadas":
    st.caption("Barras por categoría y año con IC95%")
    fig = plot_bars_grouped(sub_long)
    st.pyplot(fig)
else:
    st.caption("Líneas por categoría; un subplot por grupo")
    fig = plot_lines_facets(df, indicador_sel)
    st.pyplot(fig)

with st.expander("Ver datos crudos (grupo seleccionado)"):
    st.dataframe(sub_long.sort_values(["categoria","anio"]))

# ============================
# 4. MODO AVANZADO
# ============================
adv = st.checkbox("Mostrar modo avanzado", value=False)
if adv:
    st.markdown("---")
    st.subheader("Gráficos avanzados (matplotlib)")

    sub_wide = wide[(wide["indicador"]==indicador_sel) & (wide["grupo"]==grupo_sel)].copy()

    tipo_adv = st.radio("Tipo de gráfico avanzado", ["Dumbbell","Slopegraph"], horizontal=True)
    if tipo_adv == "Dumbbell":
        fig2 = plot_dumbbell(sub_wide)
    else:
        fig2 = plot_slopegraph(sub_wide)
    st.pyplot(fig2)

    st.markdown("**Resumen de cambios (2009→2023)**")
    cols_show = ["categoria","valor_2009","valor_2023","abs_change","rel_change","cambio_relevante"]
    st.dataframe(sub_wide[cols_show].sort_values("abs_change", ascending=False), use_container_width=True)
    csv2 = sub_wide.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar CSV (indicador/grupo)", data=csv2,
                       file_name="resumen_indicador_grupo.csv", mime="text/csv")
