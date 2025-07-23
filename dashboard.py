"""
Streamlit app sencilla para indicadores 'M' (GATS 2009, 2015, 2023)
- Visualización por defecto: barras + IC95%
- Opción: líneas facetadas por grupo con Altair
- Modo avanzado opcional
"""

# ============================
# 0. IMPORTS & PAGE CONFIG
# ============================
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

st.set_page_config(page_title="Indicadores M – GATS 2009/2015/2023", layout="wide")

# ============================
# 1. DATA LOADING & PREP
# ============================
@st.cache_data
def load_data(path: str = "Datos_GATS_Completo.xlsx"):
    df = pd.read_excel(path)
    expected = {"indicador","grupo","categoria","anio","valor","inferior","superior"}
    missing = expected - set(df.columns)
    return df, missing

df, missing_cols = load_data()
if missing_cols:
    st.error(f"Faltan columnas obligatorias: {missing_cols}")
    st.stop()

@st.cache_data
def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot_table(index=["indicador","grupo","categoria"],
                          columns="anio",
                          values=["valor","inferior","superior"])
    for comp in ["valor","inferior","superior"]:
        for y in [2009,2015,2023]:
            if (comp, y) not in wide.columns:
                wide[(comp, y)] = np.nan
    wide.columns = [f"{m}_{y}" for m,y in wide.columns]
    wide = wide.reset_index()
    wide["abs_change"] = wide["valor_2023"] - wide["valor_2009"]
    wide["rel_change"] = 100 * wide["abs_change"] / wide["valor_2009"]
    def no_overlap(r):
        return not (r["inferior_2023"] <= r["superior_2009"] and r["superior_2023"] >= r["inferior_2009"])
    wide["cambio_relevante"] = wide.apply(no_overlap, axis=1)
    return wide

wide = compute_deltas(df)

# ============================
# 2. PLOT FUNCTIONS
# ============================

def alt_barras_ci(sub_long: pd.DataFrame) -> alt.Chart:
    """Barras horizontales (año en color) + IC95%."""
    data = sub_long.copy()
    data["anio"] = data["anio"].astype(str)
    if "2023" in data["anio"].unique():
        order = (data[data["anio"]=="2023"]
                 .sort_values("valor", ascending=False)["categoria"].tolist())
    else:
        order = data["categoria"].unique().tolist()

    bars = alt.Chart(data).mark_bar().encode(
        y=alt.Y('categoria:N', title='Categoría', sort=order),
        x=alt.X('valor:Q', title='%'),
        color=alt.Color('anio:N', title='Año')
    )

    errs = alt.Chart(data).mark_errorbar().encode(
        y=alt.Y('categoria:N', sort=order),
        x='inferior:Q',
        x2='superior:Q',
        color=alt.Color('anio:N', legend=None)
    )

    height = max(220, 28 * data['categoria'].nunique())
    return (bars + errs).properties(width=650, height=height).resolve_scale(y='shared')

def alt_line_facets(df_long: pd.DataFrame, indicador: str) -> alt.Chart:
    """
    Líneas por categoría, facet por grupo. 
    Se usa layer(linea + banda de error) y luego facet al final para evitar errores de esquema.
    """
    data = df_long[df_long["indicador"] == indicador].copy()
    data["anio"] = data["anio"].astype(str)

    # Línea + puntos
    line = alt.Chart(data).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("anio:O", title="Año"),
        y=alt.Y("valor:Q", title="%"),
        color=alt.Color("categoria:N", title="Categoría"),
        tooltip=["grupo","categoria","anio","valor","inferior","superior"]
    )

    # Banda de error (IC95%). Misma codificación de x/y para evitar conflictos
    band = alt.Chart(data).mark_errorband(opacity=0.25).encode(
        x="anio:O",
        y="inferior:Q",
        y2="superior:Q",
        color=alt.Color("categoria:N", legend=None)
    )

    layered = alt.layer(band, line)

    # Facet al final (row por grupo)
    chart = layered.facet(
        row=alt.Row("grupo:N", title=None, header=alt.Header(labelAngle=0, labelPadding=5))
    ).properties(width=350, height=220, spacing=10)

    # Escalas independientes en Y para cada facet (evita warnings de dominio)
    return chart.resolve_scale(y="independent")

# --- Avanzados opcionales ---
def _prep_long(sub: pd.DataFrame, cols):
    long = sub.melt(id_vars=["categoria"], value_vars=cols,
                    var_name="anio", value_name="valor")
    long["anio"] = long["anio"].str.extract(r"(\d{4})").astype(int)
    return long.dropna(subset=["valor"])

def alt_dumbbell(sub: pd.DataFrame) -> alt.Chart:
    cols = [c for c in ["valor_2009","valor_2015","valor_2023"] if c in sub.columns]
    if len(cols) < 2:
        return alt.Chart(pd.DataFrame()).mark_text(text="Sin datos suficientes").encode()
    long = _prep_long(sub, cols)
    base = alt.Chart(long).encode(y=alt.Y("categoria:N", sort='-x', title="Categoría"))
    line = base.mark_rule().encode(x=alt.X("min(valor):Q", title="%"), x2="max(valor):Q")
    pts  = base.mark_point(filled=True, size=60).encode(x="valor:Q", color=alt.Color("anio:N", legend=alt.Legend(title="Año")))
    return (line + pts).properties(height=max(200, 25*len(sub)), width=650)

def alt_slopegraph(sub: pd.DataFrame) -> alt.Chart:
    cols = [c for c in ["valor_2009","valor_2015","valor_2023"] if c in sub.columns]
    if len(cols) < 2:
        return alt.Chart(pd.DataFrame()).mark_text(text="Sin datos suficientes").encode()
    long = _prep_long(sub, cols)
    chart = alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("anio:O", title="Año"),
        y=alt.Y("valor:Q", title="%"),
        detail="categoria",
        color=alt.Color("categoria", legend=None)
    ).properties(width=650, height=350)
    end_labels = long[long["anio"]==long["anio"].max()]
    labels = alt.Chart(end_labels).mark_text(align='left', dx=5).encode(x="anio:O", y="valor:Q", text="categoria")
    return chart + labels

def alt_heatmap(df_heat: pd.DataFrame) -> alt.Chart:
    chart = alt.Chart(df_heat).mark_rect().encode(
        x=alt.X("indicador:N", title="Indicador"),
        y=alt.Y("categoria:N", title="Categoría", sort='-x'),
        color=alt.Color("rel_change:Q", title="Δ% 09-23", scale=alt.Scale(scheme='redblue', domainMid=0)),
        tooltip=["indicador","categoria","rel_change"]
    ).properties(width=700, height=25*len(df_heat['categoria'].unique()))
    return chart

# ============================
# 3. UI
# ============================

st.title("Indicadores M del MPOWER – México (GATS 2009, 2015, 2023)")

st.sidebar.header("Filtros")
indicadores = sorted(df["indicador"].unique())
indicador_sel = st.sidebar.selectbox("Indicador", indicadores)

sub_df = df[df["indicador"] == indicador_sel]
grupos = sorted(sub_df["grupo"].unique())
grupo_sel = st.sidebar.selectbox("Grupo (para barras)", grupos)

sub_long = sub_df[sub_df["grupo"] == grupo_sel].copy()

# Hallazgos clave
msg_lines = []
for cat, g in sub_long.groupby('categoria'):
    try:
        v09 = float(g.loc[g.anio==2009, 'valor'].values[0])
        v23 = float(g.loc[g.anio==2023, 'valor'].values[0])
        delta = v23 - v09
        msg_lines.append((cat, v23, delta))
    except Exception:
        continue
msg_lines = sorted(msg_lines, key=lambda x: abs(x[2]), reverse=True)

st.subheader("Hallazgos clave")
if msg_lines:
    top_k = min(5, len(msg_lines))
    bullets = [f"- {cat}: {v23:.1f}% ({'+' if d>=0 else ''}{d:.1f} pts vs 2009)" for cat, v23, d in msg_lines[:top_k]]
    st.markdown("\n".join(bullets))
else:
    st.info("No se pudieron calcular cambios (faltan datos 2009/2023 para este grupo)")

# Visualización sencilla: elegir tipo
st.subheader("Visualización sencilla")
viz_tipo = st.radio("Tipo de visualización", ["Barras agrupadas", "Líneas facetadas"], horizontal=True)

if viz_tipo == "Barras agrupadas":
    st.caption("Barras por categoría y año con intervalos de confianza")
    st.altair_chart(alt_barras_ci(sub_long), use_container_width=True)
else:
    st.caption("Líneas por categoría; cada fila es un grupo")
    st.altair_chart(alt_line_facets(df, indicador_sel), use_container_width=True)

with st.expander("Ver datos crudos (grupo seleccionado)"):
    st.dataframe(sub_long.sort_values(["categoria","anio"]))

# ============================
# 4. MODO AVANZADO
# ============================
af = st.checkbox("Mostrar modo avanzado", value=False)
if af:
    st.markdown("---")
    st.subheader("Opciones avanzadas")
    sub_wide = wide[(wide["indicador"]==indicador_sel) & (wide["grupo"]==grupo_sel)].copy()
    tipo = st.radio("Tipo de gráfico avanzado", ["Dumbbell","Slopegraph","Heatmap cambios % (para todos)"], horizontal=True)
    if tipo == "Dumbbell":
        st.altair_chart(alt_dumbbell(sub_wide), use_container_width=True)
    elif tipo == "Slopegraph":
        st.altair_chart(alt_slopegraph(sub_wide), use_container_width=True)
    else:
        st.altair_chart(alt_heatmap(wide[["indicador","categoria","rel_change"]]), use_container_width=True)

    st.markdown("**Resumen de cambios (2009→2023)**")
    cols_show = ["categoria","valor_2009","valor_2023","abs_change","rel_change","cambio_relevante"]
    st.dataframe(sub_wide[cols_show].sort_values("abs_change", ascending=False), use_container_width=True)
    csv2 = sub_wide.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar CSV (indicador/grupo)", data=csv2,
                       file_name="resumen_indicador_grupo.csv", mime="text/csv")

