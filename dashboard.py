"""
Streamlit app to explore and summarize MPOWER "M" indicators (GATS 2009, 2015, 2023)
Author: (you)

How this app is organized
-------------------------
1. Data prep & helper functions (delta calculations, significance flags)
2. Plot functions (dumbbell, slopegraph, heatmap) using matplotlib/altair
3. Streamlit UI with three tabs:
   - Resumen (tabla inteligente + descargas)
   - Cambios relevantes (top ups/downs por indicador)
   - Explora (elige indicador/grupo/gráfico)

Run locally:
------------
$ pip install streamlit pandas numpy matplotlib altair openpyxl
$ streamlit run app.py

If deploying on Streamlit Cloud, include a requirements.txt with those packages.
"""

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from io import BytesIO

# ============================
# 1. DATA LOADING & PREP
# ============================
@st.cache_data
def load_data(path: str = "Datos_GATS_Completo.xlsx") -> pd.DataFrame:
    df = pd.read_excel(path)
    # sanity: expected columns
    expected = {"indicador","grupo","categoria","anio","valor","inferior","superior"}
    missing = expected - set(df.columns)
    if missing:
        st.error(f"Faltan columnas: {missing}")
    return df

df = load_data()

# Pivot wide by year to compute deltas easily

def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    wide = (df.pivot_table(index=["indicador","grupo","categoria"],
                           columns="anio",
                           values=["valor","inferior","superior"]) )
    wide.columns = [f"{m}_{y}" for m,y in wide.columns]  # flatten
    wide = wide.reset_index()
    # absolute & relative change 2009->2023
    wide["abs_change"] = wide["valor_2023"] - wide["valor_2009"]
    wide["rel_change"] = 100 * wide["abs_change"] / wide["valor_2009"]
    # IC overlap flag (simple heuristic)
    def no_overlap(r):
        return not (r["inferior_2023"] <= r["superior_2009"] and r["superior_2023"] >= r["inferior_2009"])
    wide["cambio_relevante"] = wide.apply(no_overlap, axis=1)
    return wide

wide = compute_deltas(df)

# ============================
# 2. PLOT FUNCTIONS
# ============================

def alt_dumbbell(sub: pd.DataFrame) -> alt.Chart:
    """Dumbbell plot: categoría en Y, 2009-2015-2023 en X."""
    # Melt to long for altair
    long = sub.melt(id_vars=["categoria"], value_vars=["valor_2009","valor_2015","valor_2023"],
                    var_name="anio", value_name="valor")
    long["anio"] = long["anio"].str.extract(r"(\d{4})").astype(int)

    base = alt.Chart(long).encode(
        y=alt.Y("categoria:N", sort='-x', title="Categoría")
    )
    # line connecting 2009-2023
    line = base.mark_rule().encode(
        x=alt.X("min(valor):Q", title="%"),
        x2="max(valor):Q"
    )
    pts = base.mark_point(filled=True, size=60).encode(
        x="valor:Q",
        color=alt.Color("anio:N", legend=alt.Legend(title="Año"))
    )
    return (line + pts).properties(height=25*len(sub), width=600)


def alt_slopegraph(sub_df: pd.DataFrame) -> alt.Chart:
    """Slopegraph: líneas por categoría a través de años."""
    long = sub_df.melt(id_vars=["categoria"], value_vars=["valor_2009","valor_2015","valor_2023"],
                       var_name="anio", value_name="valor")
    long["anio"] = long["anio"].str.extract(r"(\d{4})").astype(int)

    chart = alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("anio:O", title="Año"),
        y=alt.Y("valor:Q", title="%"),
        detail="categoria",
        color=alt.Color("categoria", legend=None)
    ).properties(width=600, height=300)

    # Add end labels (2023)
    end_labels = long[long['anio']==2023]
    label_chart = alt.Chart(end_labels).mark_text(align='left', dx=5).encode(
        x=alt.X("anio:O"), y="valor:Q", text="categoria"
    )
    return chart + label_chart


def alt_heatmap(df_heat: pd.DataFrame) -> alt.Chart:
    """Heatmap of relative change 2009-2023 by indicador x categoria (or group)."""
    # Expect columns: indicador, categoria, rel_change
    chart = alt.Chart(df_heat).mark_rect().encode(
        x=alt.X("indicador:N", title="Indicador"),
        y=alt.Y("categoria:N", title="Categoría", sort='-x'),
        color=alt.Color("rel_change:Q", title="Δ% 09-23", scale=alt.Scale(scheme='redblue', domainMid=0)),
        tooltip=["indicador","categoria","rel_change"]
    ).properties(width=600, height=25*len(df_heat['categoria'].unique()))
    return chart

# ============================
# 3. STREAMLIT UI
# ============================

st.set_page_config(page_title="Indicadores M – GATS 2009/2015/2023", layout="wide")
st.title("Indicadores M del MPOWER – México (GATS 2009, 2015, 2023)")

with st.expander("Descripción del dataset", expanded=False):
    st.write("Columnas:")
    st.code("indicador, grupo, categoria, anio, valor, inferior, superior")

st.sidebar.header("Filtros globales")
indicadores = sorted(df["indicador"].unique())
sel_indic = st.sidebar.multiselect("Indicadores", indicadores, default=indicadores)

# Filter by selected indicators
f_df = df[df["indicador"].isin(sel_indic)]
f_wide = wide[wide["indicador"].isin(sel_indic)]

# Tabs
resumen_tab, cambios_tab, explora_tab = st.tabs(["Resumen", "Cambios relevantes", "Explora"])

# -------- TAB 1: RESUMEN ---------
with resumen_tab:
    st.subheader("Tabla resumen de cambios 2009→2023")
    st.caption("Ordena por cambio absoluto o relativo para detectar los hallazgos clave.")
    st.dataframe(f_wide.sort_values("abs_change", ascending=False), use_container_width=True)

    # Download button
    csv = f_wide.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar CSV", data=csv, file_name="resumen_cambios.csv", mime="text/csv")

    # Heatmap opcional (si hay más de 1 indicador)
    if len(sel_indic) > 1:
        st.subheader("Heatmap de cambios relativos (%) 2009-2023")
        hm = alt_heatmap(f_wide[["indicador","categoria","rel_change"]])
        st.altair_chart(hm, use_container_width=True)

# -------- TAB 2: CAMBIOS RELEVANTES ---------
with cambios_tab:
    st.subheader("Top cambios por indicador")
    top_n = st.number_input("Top N por indicador (↑ y ↓)", 1, 10, 5)

    for ind in sel_indic:
        st.markdown(f"### {ind}")
        sub = f_wide[f_wide["indicador"]==ind].copy()
        # top up/down
        top_up = sub.sort_values("abs_change", ascending=False).head(top_n)
        top_down = sub.sort_values("abs_change", ascending=True).head(top_n)

        if len(top_up)>0:
            st.markdown("**Aumentos más grandes**")
            st.dataframe(top_up[["grupo","categoria","valor_2009","valor_2023","abs_change","rel_change","cambio_relevante"]], use_container_width=True)
            st.altair_chart(alt_dumbbell(top_up), use_container_width=True)

        if len(top_down)>0:
            st.markdown("**Disminuciones más grandes**")
            st.dataframe(top_down[["grupo","categoria","valor_2009","valor_2023","abs_change","rel_change","cambio_relevante"]], use_container_width=True)
            st.altair_chart(alt_dumbbell(top_down), use_container_width=True)

# -------- TAB 3: EXPLORA ---------
with explora_tab:
    st.subheader("Explora un indicador y grupo específicos")
    indicador_sel = st.selectbox("Indicador", sorted(df["indicador"].unique()))
    grupos = sorted(df[df["indicador"]==indicador_sel]["grupo"].unique())
    grupo_sel = st.selectbox("Grupo", grupos)

    sub_long = df[(df["indicador"]==indicador_sel) & (df["grupo"]==grupo_sel)].copy()
    sub_wide = f_wide[(f_wide["indicador"]==indicador_sel) & (f_wide["grupo"]==grupo_sel)].copy()

    tipo = st.radio("Tipo de gráfico", ["Dumbbell","Slopegraph"], horizontal=True)

    if tipo == "Dumbbell":
        st.altair_chart(alt_dumbbell(sub_wide), use_container_width=True)
    else:
        st.altair_chart(alt_slopegraph(sub_wide), use_container_width=True)

    # Show raw table for this subset
    with st.expander("Ver datos crudos de este subset"):
        st.dataframe(sub_long.sort_values(["categoria","anio"]))

# End of file
