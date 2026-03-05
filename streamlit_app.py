import io
from statistics import NormalDist

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st


st.set_page_config(page_title="Online Linear Regression", layout="wide")
query = st.query_params


MENU_OPTIONS = [
    ("file_upload", "File Upload", "Cargar Archivo"),
    ("analyze_observations", "Analyze Observations", "Analizar Observaciones"),
    ("variable_selection", "Variable Selection", "Selección de Variables"),
    ("regression_results", "Regression Results", "Resultados de Regresión"),
    ("residuals", "Residuals", "Residuos"),
    ("prediction", "Prediction", "Predicción"),
]


def _qp_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return default
    return str(value).lower() in {"1", "true", "yes", "on"}

st.markdown(
    """
    <style>
    :root {
        --app-bg: #ffffff;
        --app-surface: #f6f4ef;
        --app-text: #111111;
        --app-accent: #1f77b4;
        --app-border: #c9c3b8;
    }
    .stApp { background-color: var(--app-bg); }
    section[data-testid="stSidebar"] { background-color: var(--app-surface); }
    body, p, span, label, div { color: var(--app-text); }
    [data-testid="stFileUploaderDropzone"] {
        background-color: var(--app-bg);
        border: 1px dashed var(--app-border);
        color: var(--app-text);
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: var(--app-text) !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background-color: var(--app-accent) !important;
        color: #ffffff !important;
        border: none;
    }
    header[data-testid="stHeader"] {
        background-color: var(--app-bg);
    }
    [data-testid="stToolbar"] {
        background-color: var(--app-bg);
        color: var(--app-text);
    }
    [data-testid="stToolbar"] * {
        color: var(--app-text) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stButton"] button {
        background: none !important;
        border: none !important;
        padding: 2px 0 !important;
        color: var(--app-text) !important;
        text-align: left !important;
        justify-content: flex-start !important;
    }
    section[data-testid="stSidebar"] [data-testid="stButton"] button:hover {
        background: none !important;
        text-decoration: underline;
    }
    section[data-testid="stSidebar"] .st-key-lang_en button,
    section[data-testid="stSidebar"] .st-key-lang_es button {
        line-height: 1 !important;
        min-height: 5.4rem !important;
        border-radius: 10px !important;
        text-align: center !important;
        justify-content: center !important;
        padding: 0.2rem !important;
    }
    section[data-testid="stSidebar"] .st-key-lang_en button p,
    section[data-testid="stSidebar"] .st-key-lang_es button p {
        font-size: 2.5rem !important;
        line-height: 1 !important;
    }
    section[data-testid="stSidebar"] .st-key-lang_en button[kind="secondary"],
    section[data-testid="stSidebar"] .st-key-lang_es button[kind="secondary"] {
        filter: grayscale(100%);
        opacity: 0.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "lang" not in st.session_state:
    lang_from_query = query.get("lang")
    if isinstance(lang_from_query, list):
        lang_from_query = lang_from_query[0] if lang_from_query else None
    st.session_state["lang"] = lang_from_query if lang_from_query in {"en", "es"} else "en"

lang = st.session_state.get("lang", "en")
if lang not in {"en", "es"}:
    lang = "en"
    st.session_state["lang"] = lang


def tr(en_text, es_text):
    return es_text if lang == "es" else en_text


st.title(tr("Online Linear Regression", "Regresión Lineal Online"))

with st.sidebar:
    lang_en, lang_es = st.columns(2)
    with lang_en:
        if st.button("🇬🇧", key="lang_en", type="primary" if lang == "en" else "secondary", use_container_width=True):
            st.session_state["lang"] = "en"
            query["lang"] = "en"
            st.rerun()
    with lang_es:
        if st.button("🇪🇸", key="lang_es", type="primary" if lang == "es" else "secondary", use_container_width=True):
            st.session_state["lang"] = "es"
            query["lang"] = "es"
            st.rerun()

    st.markdown(f"<span style='font-size:22px; font-weight:600;'>{tr('Options', 'Opciones')}</span>", unsafe_allow_html=True)
    valid_sections = {key for key, _, _ in MENU_OPTIONS}
    if "section" not in st.session_state:
        st.session_state["section"] = MENU_OPTIONS[0][0]
    if st.session_state["section"] not in valid_sections:
        st.session_state["section"] = MENU_OPTIONS[0][0]
    for key, label_en, label_es in MENU_OPTIONS:
        option_label = tr(label_en, label_es)
        if st.session_state["section"] == key:
            st.markdown(f"**▶ {option_label}**")
        else:
            if st.button(option_label, key=f"menu_{key}"):
                st.session_state["section"] = key
                st.rerun()
    st.divider()
    current_name = st.session_state.get("uploaded_name")
    if current_name:
        st.caption(f"{tr('Current file', 'Archivo actual')}: {current_name}")
    else:
        st.caption(tr("No file uploaded yet.", "Todavía no has cargado ningún archivo."))
    if st.button(tr("Clear file", "Limpiar archivo")):
        st.session_state.pop("uploaded_bytes", None)
        st.session_state.pop("uploaded_name", None)
        st.session_state.pop("uploaded_signature", None)
        st.session_state["uploader"] = None


def load_dataframe():
    bytes_data = st.session_state.get("uploaded_bytes")
    if not bytes_data:
        return None, None
    try:
        return pd.read_excel(io.BytesIO(bytes_data)), None
    except Exception as exc:
        return None, exc


def apply_transforms(base_df, transforms=None):
    if transforms is None:
        transforms = st.session_state.get("transforms", [])
    if not transforms:
        return base_df.copy(), None
    df = base_df.copy()
    for item in transforms:
        source = item.get("source")
        kind = item.get("kind")
        name = item.get("name")
        if not source or not kind or not name:
            continue
        if source not in df.columns:
            return None, tr(
                f"Source column missing for transform: {source}",
                f"Falta la columna fuente para la transformación: {source}",
            )
        series = df[source]
        try:
            if kind == "square":
                df[name] = series**2
            elif kind == "log":
                if (series <= 0).any():
                    return None, tr(
                        f"Log transform requires positive values in {source}.",
                        f"La transformación log requiere valores positivos en {source}.",
                    )
                df[name] = np.log(series)
            elif kind == "log(1+x)":
                if (series < 0).any():
                    return None, tr(
                        f"log(1+x) requires non-negative values in {source}.",
                        f"log(1+x) requiere valores no negativos en {source}.",
                    )
                df[name] = np.log1p(series)
            elif kind == "sqrt":
                if (series < 0).any():
                    return None, tr(
                        f"sqrt requires non-negative values in {source}.",
                        f"sqrt requiere valores no negativos en {source}.",
                    )
                df[name] = np.sqrt(series)
        except Exception as exc:
            return None, tr(
                f"Failed to create {name}: {exc}",
                f"No se pudo crear {name}: {exc}",
            )
    return df, None


def build_outlier_mask(df):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        low = df[col].median() - 1.5 * iqr
        high = df[col].median() + 1.5 * iqr
        mask[col] = (df[col] < low) | (df[col] > high)
    return mask


def sync_row_selection_state(df):
    expected = len(df)
    current = st.session_state.get("row_selection")
    if not isinstance(current, dict):
        current = {}
    # Keep existing selections when possible, default new rows to selected.
    st.session_state["row_selection"] = {
        i: bool(current.get(i, True)) for i in range(expected)
    }


df, df_error = load_dataframe()
section = st.session_state.get("section", "file_upload")

if section == "file_upload":
    st.write(tr("Upload an Excel file, select variables, and fit a linear regression model.", "Carga un archivo de Excel, selecciona variables y ajusta un modelo de regresión lineal."))
    uploaded_file = st.file_uploader(tr("Upload Excel file", "Cargar archivo de Excel"), type=["xlsx", "xls"], key="uploader")
    transform_error = None
    working_df = None
    if uploaded_file is not None:
        upload_signature = (uploaded_file.name, uploaded_file.size, uploaded_file.type)
        if st.session_state.get("uploaded_signature") != upload_signature:
            st.session_state["uploaded_bytes"] = uploaded_file.getvalue()
            st.session_state["uploaded_name"] = uploaded_file.name
            st.session_state["uploaded_signature"] = upload_signature
        df, df_error = load_dataframe()
        working_df, transform_error = apply_transforms(df) if df is not None else (None, None)
    else:
        df, df_error = load_dataframe()
        working_df, transform_error = apply_transforms(df) if df is not None else (None, None)
        if df_error is None:
            st.success(tr("File uploaded successfully.", "Archivo cargado correctamente."))
    if df_error is not None:
        st.error(tr(f"Failed to read Excel file: {df_error}", f"No se pudo leer el archivo Excel: {df_error}"))
    elif transform_error is not None:
        st.error(transform_error)
    elif df is not None:
        st.subheader(tr("Data Preview", "Vista Previa de Datos"))
        preview_df = working_df if working_df is not None else df
        if "show_all_upload" not in st.session_state:
            st.session_state["show_all_upload"] = _qp_bool(query.get("show_all_upload"), False)
        def _update_show_all_upload():
            query["show_all_upload"] = str(st.session_state["show_all_upload"]).lower()
        show_all = st.checkbox(
            tr("Show all rows", "Mostrar todas las filas"),
            key="show_all_upload",
            on_change=_update_show_all_upload,
        )
        st.dataframe(preview_df if show_all else preview_df.head(50), use_container_width=True)
    else:
        st.info(tr("Upload an Excel file to get started.", "Carga un archivo de Excel para comenzar."))
    st.stop()

if df_error is not None:
    st.error(tr(f"Failed to read Excel file: {df_error}", f"No se pudo leer el archivo Excel: {df_error}"))
    st.stop()

if df is None:
    st.info(tr("Upload an Excel file in the File Upload section.", "Carga un archivo de Excel en la sección Cargar Archivo."))
    st.stop()

if df.empty:
    st.error(tr("The uploaded file is empty.", "El archivo cargado está vacío."))
    st.stop()

working_df, transform_error = apply_transforms(df)
if transform_error is not None:
    st.error(transform_error)
    st.stop()

if "applied_clean" not in st.session_state:
    st.session_state["applied_clean"] = False
if "applied_df" not in st.session_state:
    st.session_state["applied_df"] = None
if "row_selection" not in st.session_state:
    st.session_state["row_selection"] = {}

sync_row_selection_state(working_df)

if st.session_state["applied_clean"] and st.session_state["applied_df"] is not None:
    active_df, active_transform_error = apply_transforms(st.session_state["applied_df"])
    if active_transform_error is not None:
        st.error(active_transform_error)
        st.stop()
else:
    active_df = working_df

all_cols = active_df.columns.tolist()
numeric_cols = active_df.select_dtypes(include="number").columns.tolist()

if section not in ["file_upload", "analyze_observations"]:
    if st.session_state["applied_clean"] and st.session_state["applied_df"] is not None:
        st.info(
            tr(
                f"Using subset of rows: {len(active_df)} of {len(working_df)} observations.",
                f"Usando subconjunto de filas: {len(active_df)} de {len(working_df)} observaciones.",
            )
        )
    else:
        st.info(tr(f"Using full dataset: {len(working_df)} observations.", f"Usando conjunto completo: {len(working_df)} observaciones."))

if section == "analyze_observations":
    st.subheader(tr("Analyze Observations", "Analizar Observaciones"))
    st.write(tr("Select observations to include in downstream analysis.", "Selecciona observaciones para incluir en el análisis posterior."))
    st.divider()

    page_size = 25
    total_rows = len(working_df)
    total_pages = max(1, int(np.ceil(total_rows / page_size)))
    if "obs_page" not in st.session_state:
        st.session_state["obs_page"] = 0
    st.session_state["obs_page"] = max(0, min(st.session_state["obs_page"], total_pages - 1))

    nav_left, nav_mid, nav_right = st.columns([1, 2, 1])
    with nav_left:
        if st.button(tr("Previous page", "Página anterior"), key="obs_prev_page", disabled=st.session_state["obs_page"] == 0):
            st.session_state["obs_page"] -= 1
            st.rerun()
    with nav_mid:
        start_row = st.session_state["obs_page"] * page_size
        end_row = min(start_row + page_size, total_rows)
        st.caption(
            tr(
                f"Page {st.session_state['obs_page'] + 1} of {total_pages} (rows {start_row + 1}-{end_row} of {total_rows})",
                f"Página {st.session_state['obs_page'] + 1} de {total_pages} (filas {start_row + 1}-{end_row} de {total_rows})",
            )
        )
    with nav_right:
        if st.button(
            tr("Next page", "Página siguiente"),
            key="obs_next_page",
            disabled=st.session_state["obs_page"] >= total_pages - 1,
        ):
            st.session_state["obs_page"] += 1
            st.rerun()

    selection_df = working_df.copy()
    selection_df.insert(
        0,
        "selected",
        [st.session_state["row_selection"].get(i, True) for i in range(len(selection_df))],
    )
    page_selection_df = selection_df.iloc[start_row:end_row].copy()
    nan_mask = working_df.isna().iloc[start_row:end_row]
    outlier_mask = build_outlier_mask(working_df).iloc[start_row:end_row]

    def _style_selection_table(_):
        styles = pd.DataFrame("", index=page_selection_df.index, columns=page_selection_df.columns)
        for col in page_selection_df.columns:
            if col == "selected":
                continue
            is_nan = nan_mask[col]
            is_outlier = outlier_mask[col] & (~is_nan)
            styles.loc[is_outlier, col] = "background-color: #6b0000; color: #ffffff; font-weight: 700;"
            styles.loc[is_nan, col] = "background-color: #000000; color: #ffffff; font-weight: 700;"
        return styles

    styled_selection = page_selection_df.style.apply(_style_selection_table, axis=None)

    edited = st.data_editor(
        styled_selection,
        use_container_width=True,
        hide_index=False,
        num_rows="fixed",
        height=920,
        column_config={"selected": st.column_config.CheckboxColumn("")},
        disabled=[col for col in page_selection_df.columns if col != "selected"],
    )
    for offset, value in enumerate(edited["selected"].tolist()):
        st.session_state["row_selection"][start_row + offset] = bool(value)

    action_left, action_mid, action_mid_right, action_right = st.columns(4)
    with action_left:
        if st.button(tr("Unmark all nan", "Desmarcar todos los nan"), key="obs_unmark_all_nan"):
            rows_with_nan = working_df.isna().any(axis=1).tolist()
            st.session_state["row_selection"] = {
                i: (not rows_with_nan[i]) for i in range(len(working_df))
            }
            st.rerun()
    with action_mid:
        if st.button(tr("mark all rows", "marcar todas las filas"), key="obs_mark_all_rows"):
            st.session_state["row_selection"] = {
                i: True for i in range(len(working_df))
            }
            st.rerun()
    with action_mid_right:
        selected_positions_for_download = [
            i for i, keep in st.session_state["row_selection"].items() if keep
        ]
        subset_for_download = working_df.iloc[selected_positions_for_download].copy()
        download_buffer = io.BytesIO()
        subset_for_download.to_excel(download_buffer, index=False)
        download_buffer.seek(0)
        st.download_button(
            tr("download subset", "descargar subconjunto"),
            data=download_buffer.getvalue(),
            file_name="subset.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="obs_download_subset",
            disabled=subset_for_download.empty,
        )
    with action_right:
        if st.button(tr("use current selection", "usar seleccion actual"), key="obs_use_current_selection"):
            selected_positions = [
                i for i, keep in st.session_state["row_selection"].items() if keep
            ]
            if not selected_positions:
                st.error(tr("No observations selected.", "No hay observaciones seleccionadas."))
            else:
                st.session_state["applied_df"] = working_df.iloc[selected_positions].copy()
                st.session_state["applied_clean"] = True
                st.success(
                    f"Selection applied: {len(selected_positions)} of {len(working_df)} observations."
                    if lang == "en"
                    else f"Selección aplicada: {len(selected_positions)} de {len(working_df)} observaciones."
                )
                st.rerun()

    st.divider()
    st.caption(tr("Unmark rows by condition", "Desmarcar filas por condición"))
    cond_input_cols = st.columns([2, 1, 2, 1.5])
    with cond_input_cols[0]:
        cond_col = st.selectbox(
            tr("Variable", "Variable"),
            working_df.columns.tolist(),
            key="obs_cond_col",
        )
    cond_series = working_df[cond_col]
    is_numeric = pd.api.types.is_numeric_dtype(cond_series)

    if is_numeric:
        with cond_input_cols[1]:
            cond_op = st.selectbox(tr("Operator", "Operador"), ["<", "<=", ">", ">="], key="obs_cond_op")
        with cond_input_cols[2]:
            cond_value = st.number_input(tr("Value", "Valor"), key="obs_cond_value")
    else:
        cond_op = "=="
        cond_options = cond_series.dropna().unique().tolist()
        with cond_input_cols[1]:
            st.text_input(tr("Operator", "Operador"), value="=", disabled=True, key="obs_cond_op_cat")
        with cond_input_cols[2]:
            if not cond_options:
                cond_value = None
                st.text_input(tr("Value", "Valor"), value=tr("No options", "Sin opciones"), disabled=True, key="obs_cond_value_empty")
            else:
                cond_value = st.selectbox(tr("Value", "Valor"), cond_options, key="obs_cond_value_cat")

    with cond_input_cols[3]:
        st.markdown("<div style='height: 1.9rem;'></div>", unsafe_allow_html=True)
        run_unmark_condition = st.button(tr("Unmark rows by condition", "Desmarcar filas por condición"), key="obs_unmark_by_condition")

    if run_unmark_condition:
        if cond_value is None:
            st.warning(tr("Choose a variable with at least one non-missing categorical value.", "Elige una variable con al menos un valor categórico no faltante."))
        else:
            if is_numeric:
                if cond_op == "<":
                    cond_mask = cond_series < cond_value
                elif cond_op == "<=":
                    cond_mask = cond_series <= cond_value
                elif cond_op == ">":
                    cond_mask = cond_series > cond_value
                else:
                    cond_mask = cond_series >= cond_value
            else:
                cond_mask = cond_series == cond_value

            updated = st.session_state["row_selection"].copy()
            for i, should_unmark in enumerate(cond_mask.fillna(False).tolist()):
                if should_unmark:
                    updated[i] = False
            st.session_state["row_selection"] = updated
            st.success(
                tr(
                    f"Unmarked {int(cond_mask.fillna(False).sum())} rows.",
                    f"Se desmarcaron {int(cond_mask.fillna(False).sum())} filas.",
                )
            )
            st.rerun()

    st.divider()
    st.caption(tr("Observation scatter plot", "Gráfico de dispersión de observaciones"))
    selected_positions_for_plot = [
        i for i, keep in st.session_state["row_selection"].items() if keep
    ]
    numeric_columns = working_df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_columns:
        st.info(
            tr(
                "No numeric columns available for scatter plotting.",
                "No hay columnas numéricas disponibles para el gráfico de dispersión.",
            )
        )
    elif not selected_positions_for_plot:
        st.info(
            tr(
                "Select at least one observation to display the scatter plot.",
                "Selecciona al menos una observación para mostrar el gráfico de dispersión.",
            )
        )
    else:
        default_x = 0
        default_y = 1 if len(numeric_columns) > 1 else 0
        if st.session_state.get("obs_scatter_x_applied") not in numeric_columns:
            st.session_state["obs_scatter_x_applied"] = numeric_columns[default_x]
        if st.session_state.get("obs_scatter_y_applied") not in numeric_columns:
            st.session_state["obs_scatter_y_applied"] = numeric_columns[default_y]
        if st.session_state.get("obs_scatter_x_ui") not in numeric_columns:
            st.session_state["obs_scatter_x_ui"] = st.session_state["obs_scatter_x_applied"]
        if st.session_state.get("obs_scatter_y_ui") not in numeric_columns:
            st.session_state["obs_scatter_y_ui"] = st.session_state["obs_scatter_y_applied"]

        scatter_cols = st.columns([2, 2, 1])
        with scatter_cols[0]:
            st.selectbox(
                tr("X axis", "Eje X"),
                numeric_columns,
                key="obs_scatter_x_ui",
            )
        with scatter_cols[1]:
            st.selectbox(
                tr("Y axis", "Eje Y"),
                numeric_columns,
                key="obs_scatter_y_ui",
            )
        with scatter_cols[2]:
            st.markdown("<div style='height: 1.9rem;'></div>", unsafe_allow_html=True)
            if st.button(tr("Update plot", "Actualizar gráfico"), key="obs_scatter_update"):
                st.session_state["obs_scatter_x_applied"] = st.session_state["obs_scatter_x_ui"]
                st.session_state["obs_scatter_y_applied"] = st.session_state["obs_scatter_y_ui"]

        scatter_x = st.session_state["obs_scatter_x_applied"]
        scatter_y = st.session_state["obs_scatter_y_applied"]

        scatter_df = working_df.iloc[selected_positions_for_plot].copy()
        scatter_df["__observation__"] = [i + 1 for i in selected_positions_for_plot]
        scatter_df = scatter_df[[scatter_x, scatter_y, "__observation__"]].dropna()

        if scatter_df.empty:
            st.warning(
                tr(
                    "No selected rows with non-missing values for the chosen axes.",
                    "No hay filas seleccionadas con valores no faltantes para los ejes elegidos.",
                )
            )
        else:
            x_values = scatter_df[scatter_x].to_numpy(dtype=float)
            y_values = scatter_df[scatter_y].to_numpy(dtype=float)
            fig_obs_scatter = go.Figure(
                data=go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="markers",
                    marker={"color": "#1f77b4", "size": 9},
                    customdata=scatter_df["__observation__"],
                    hovertemplate=(
                        tr("Observation", "Observación")
                        + " %{customdata}<br>"
                        + f"{scatter_x}: %{{x}}<br>{scatter_y}: %{{y}}<extra></extra>"
                    ),
                )
            )
            if len(x_values) >= 2 and np.unique(x_values).size >= 2:
                slope, intercept = np.polyfit(x_values, y_values, 1)
                x_line = np.array([float(np.min(x_values)), float(np.max(x_values))])
                y_line = slope * x_line + intercept
                fig_obs_scatter.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        line={"color": "#b22222", "width": 2},
                        name=tr("Regression line", "Línea de regresión"),
                        hovertemplate=f"{scatter_y} = {slope:.4g}*{scatter_x} + {intercept:.4g}<extra></extra>",
                    )
                )
            fig_obs_scatter.update_layout(
                title=tr("Selected observations", "Observaciones seleccionadas"),
                xaxis_title=scatter_x,
                yaxis_title=scatter_y,
                template="plotly_white",
                height=520,
            )
            st.plotly_chart(fig_obs_scatter, use_container_width=True)

    st.stop()

if section == "variable_selection":
    st.subheader(tr("Variable Selection", "Selección de Variables"))
    if "dependent_var" not in st.session_state:
        st.session_state["dependent_var"] = None
    if "independent_num" not in st.session_state:
        st.session_state["independent_num"] = []
    if "independent_cat" not in st.session_state:
        st.session_state["independent_cat"] = []
    if "transforms" not in st.session_state:
        st.session_state["transforms"] = []
    if "create_error" not in st.session_state:
        st.session_state["create_error"] = None

    selected_var = st.selectbox(tr("Choose a variable", "Elige una variable"), all_cols)
    examples = active_df[selected_var].dropna()
    if not examples.empty:
        sample_n = min(5, len(examples))
        st.caption(tr("Examples", "Ejemplos"))
        st.write(examples.sample(sample_n, random_state=42).tolist())

    col_actions = st.columns(3)
    with col_actions[0]:
        if st.button(tr("Set as Dependent (Y)", "Definir como Dependiente (Y)")):
            st.session_state["dependent_var"] = selected_var
    with col_actions[1]:
        if st.button(tr("Add as Categorical X", "Agregar como X Categórica")):
            if selected_var != st.session_state.get("dependent_var"):
                if selected_var not in st.session_state["independent_cat"]:
                    st.session_state["independent_cat"].append(selected_var)
            if selected_var in st.session_state["independent_num"]:
                st.session_state["independent_num"].remove(selected_var)
    with col_actions[2]:
        if st.button(tr("Add as Numerical X", "Agregar como X Numérica")):
            if selected_var != st.session_state.get("dependent_var"):
                if selected_var not in st.session_state["independent_num"]:
                    st.session_state["independent_num"].append(selected_var)
            if selected_var in st.session_state["independent_cat"]:
                st.session_state["independent_cat"].remove(selected_var)

    st.divider()
    st.subheader(tr("Create New Variable", "Crear Nueva Variable"))
    col_base, col_kind, col_name = st.columns(3)
    with col_base:
        base_var = st.selectbox(tr("Base variable", "Variable base"), all_cols, key="transform_base")
    with col_kind:
        transform_kind = st.selectbox(
            tr("Transformation", "Transformación"),
            ["square", "log", "log(1+x)", "sqrt"],
            key="transform_kind",
        )
    default_name = f"{base_var}_{transform_kind}"
    with col_name:
        new_name = st.text_input(tr("New column name", "Nombre de nueva columna"), value=default_name, key="transform_name")
    if st.button(tr("Create Variable", "Crear Variable")):
        if new_name in all_cols:
            st.session_state["create_error"] = tr("Column name already exists.", "El nombre de la columna ya existe.")
        else:
            candidate = {"source": base_var, "kind": transform_kind, "name": new_name}
            preview_df, preview_error = apply_transforms(
                df, transforms=st.session_state["transforms"] + [candidate]
            )
            if preview_error:
                st.session_state["create_error"] = preview_error
            else:
                st.session_state["transforms"].append(candidate)
                st.session_state["create_error"] = None
                st.success(tr(f"Created {new_name}.", f"Se creó {new_name}."))
                st.rerun()

    if st.session_state.get("create_error"):
        st.error(st.session_state["create_error"])
        if st.button(tr("Return to Variable Selection", "Volver a Selección de Variables")):
            st.session_state["create_error"] = None
            st.rerun()

    st.divider()
    st.subheader(tr("Current Equation", "Ecuación Actual"))
    dependent_var = st.session_state.get("dependent_var")
    independent_num = st.session_state.get("independent_num", [])
    independent_cat = st.session_state.get("independent_cat", [])
    terms = []
    terms.extend([f"C({var})" for var in independent_cat])
    terms.extend(independent_num)
    if dependent_var and terms:
        st.code(f"{dependent_var} ~ " + " + ".join(terms))
    else:
        st.info(tr("Select a dependent variable and at least one independent variable.", "Selecciona una variable dependiente y al menos una independiente."))

    st.subheader(tr("Remove Variables", "Eliminar Variables"))
    remove_items = (
        [("cat", var, tr(f"Remove C({var})", f"Eliminar C({var})")) for var in independent_cat]
        + [("num", var, tr(f"Remove {var}", f"Eliminar {var}")) for var in independent_num]
    )
    if remove_items:
        for i in range(0, len(remove_items), 5):
            row = remove_items[i : i + 5]
            cols = st.columns(len(row))
            for col, (kind, var, label) in zip(cols, row):
                with col:
                    if st.button(label, key=f"remove_{kind}_{var}"):
                        if kind == "cat" and var in st.session_state["independent_cat"]:
                            st.session_state["independent_cat"].remove(var)
                        if kind == "num" and var in st.session_state["independent_num"]:
                            st.session_state["independent_num"].remove(var)
                        st.rerun()
            st.markdown("<div style='height:5px'></div>", unsafe_allow_html=True)
    if dependent_var:
        if st.button(tr("Clear dependent variable", "Limpiar variable dependiente")):
            st.session_state["dependent_var"] = None
            st.rerun()
    st.stop()

dependent_var = st.session_state.get("dependent_var")
independent_num = st.session_state.get("independent_num", [])
independent_cat = st.session_state.get("independent_cat", [])
if not dependent_var or dependent_var not in all_cols:
    st.info(tr("Select variables in the Variable Selection section.", "Selecciona variables en la sección Selección de Variables."))
    st.stop()

independent_num = [col for col in independent_num if col in all_cols and col != dependent_var]
independent_cat = [col for col in independent_cat if col in all_cols and col != dependent_var]
terms = [f"C({var})" for var in independent_cat] + independent_num
if not terms:
    st.info(tr("Select variables in the Variable Selection section.", "Selecciona variables en la sección Selección de Variables."))
    st.stop()

model_df = active_df[[dependent_var] + independent_num + independent_cat].dropna()
if model_df.empty:
    st.error(tr("No rows available after dropping missing values.", "No hay filas disponibles después de eliminar valores faltantes."))
    st.stop()

try:
    formula = f"{dependent_var} ~ " + " + ".join(terms)
    model = smf.ols(formula=formula, data=model_df).fit()
except Exception as exc:
    st.error(tr(f"Failed to fit model: {exc}", f"No se pudo ajustar el modelo: {exc}"))
    st.stop()

if section == "regression_results":
    st.subheader(tr("Regression Results", "Resultados de Regresión"))
    st.caption(tr("Model formula", "Fórmula del modelo"))
    st.code(formula)
    st.caption(tr(f"Observations used: {len(model_df)}", f"Observaciones usadas: {len(model_df)}"))
    st.code(model.summary().as_text())
    st.stop()

if section == "residuals":
    residuals = model.resid
    fitted = model.fittedvalues
    plot_df = model_df.copy()
    plot_df["__residuals__"] = residuals
    plot_df["__fitted__"] = fitted

    st.subheader(tr("Residuals Plot", "Gráfico de Residuos"))
    st.caption(tr("Model formula", "Fórmula del modelo"))
    st.code(formula)
    st.caption(tr(f"Observations used: {len(model_df)}", f"Observaciones usadas: {len(model_df)}"))

    def render_two_per_row(figures):
        for i in range(0, len(figures), 2):
            col_left, col_right = st.columns(2)
            with col_left:
                st.plotly_chart(figures[i], use_container_width=True)
            if i + 1 < len(figures):
                with col_right:
                    st.plotly_chart(figures[i + 1], use_container_width=True)

    fig = go.Figure(
        data=go.Scatter(
            x=plot_df["__fitted__"],
            y=plot_df["__residuals__"],
            mode="markers",
            marker={"color": "#1f77b4"},
        )
    )
    fig.update_layout(
        title=tr("Residuals vs Fitted", "Residuos vs Ajustados"),
        xaxis_title=tr("Fitted values", "Valores ajustados"),
        yaxis_title=tr("Residuals", "Residuos"),
        template="plotly_white",
        height=400,
    )

    edges = np.histogram_bin_edges(plot_df["__residuals__"], bins="auto")
    fig_hist = go.Figure(
        data=go.Histogram(x=plot_df["__residuals__"], xbins={"start": edges[0], "end": edges[-1], "size": edges[1] - edges[0]})
    )
    fig_hist.update_layout(
        title=tr("Residuals Histogram", "Histograma de Residuos"),
        xaxis_title=tr("Residuals", "Residuos"),
        yaxis_title=tr("Count", "Conteo"),
        template="plotly_white",
        height=400,
    )

    residual_values = np.sort(plot_df["__residuals__"].to_numpy())
    n_residuals = len(residual_values)
    empirical_probs = (np.arange(1, n_residuals + 1) - 0.5) / n_residuals
    residual_std = float(np.std(residual_values, ddof=1))
    if residual_std > 0:
        dist = NormalDist(mu=float(np.mean(residual_values)), sigma=residual_std)
        theoretical_probs = np.array([dist.cdf(value) for value in residual_values])
    else:
        theoretical_probs = np.full(n_residuals, 0.5)

    fig_pp = go.Figure()
    fig_pp.add_trace(
        go.Scatter(
            x=theoretical_probs,
            y=empirical_probs,
            mode="markers",
            marker={"color": "#1f77b4"},
            name=tr("Residual probabilities", "Probabilidades de residuos"),
        )
    )
    fig_pp.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"color": "#b22222", "dash": "dash"},
            name=tr("Reference line", "Línea de referencia"),
        )
    )
    fig_pp.update_layout(
        title=tr("P-P Plot of Residuals", "Gráfico P-P de Residuos"),
        xaxis_title=tr("Theoretical probabilities", "Probabilidades teóricas"),
        yaxis_title=tr("Empirical probabilities", "Probabilidades empíricas"),
        template="plotly_white",
        height=420,
    )
    render_two_per_row([fig, fig_hist, fig_pp])

    st.subheader(tr("Residuals by Variable", "Residuos por Variable"))
    variable_figures = []
    for var in independent_num:
        fig_var = go.Figure(
            data=go.Scatter(
                x=plot_df[var],
                y=plot_df["__residuals__"],
                mode="markers",
                marker={"color": "#1f77b4"},
            )
        )
        fig_var.update_layout(
            title=tr(f"Residuals vs {var}", f"Residuos vs {var}"),
            xaxis_title=var,
            yaxis_title=tr("Residuals", "Residuos"),
            template="plotly_white",
            height=350,
        )
        variable_figures.append(fig_var)

    for var in independent_cat:
        fig_box = go.Figure(
            data=go.Box(
                x=plot_df[var].astype(str),
                y=plot_df["__residuals__"],
                marker={"color": "#1f77b4"},
            )
        )
        fig_box.update_layout(
            title=tr(f"Residuals by {var}", f"Residuos por {var}"),
            xaxis_title=var,
            yaxis_title=tr("Residuals", "Residuos"),
            template="plotly_white",
            height=350,
        )
        variable_figures.append(fig_box)

    if variable_figures:
        render_two_per_row(variable_figures)

    st.subheader(tr("Observations Used for Regression", "Observaciones Usadas para la Regresión"))
    residuals_table = model_df.copy()
    residuals_table["expected_value"] = plot_df["__fitted__"]
    residuals_table["residual_error"] = plot_df["__residuals__"]
    abs_error = residuals_table["residual_error"].abs()
    residuals_table = residuals_table.loc[abs_error.sort_values(ascending=False).index]
    st.dataframe(residuals_table, use_container_width=True, hide_index=False)
    st.stop()

if section == "prediction":
    st.subheader(tr("Prediction", "Predicción"))
    st.caption(tr("Model formula", "Fórmula del modelo"))
    st.code(formula)
    input_values = {}
    for var in independent_num:
        input_values[var] = st.number_input(
            f"{var}",
            value=st.session_state.get(f"pred_{var}", 0.0),
            key=f"pred_{var}",
        )
    for var in independent_cat:
        categories = working_df[var].dropna().unique().tolist()
        if categories:
            input_values[var] = st.selectbox(
                f"{var}",
                categories,
                key=f"pred_{var}",
            )
    if st.button(tr("Predict", "Predecir")):
        input_df = pd.DataFrame([input_values])
        try:
            prediction = model.predict(input_df).iloc[0]
            st.success(tr(f"Predicted {dependent_var}: {prediction:.4f}", f"{dependent_var} predicho: {prediction:.4f}"))
        except Exception as exc:
            st.error(tr(f"Prediction failed: {exc}", f"Falló la predicción: {exc}"))
