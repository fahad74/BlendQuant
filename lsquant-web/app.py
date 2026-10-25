import pathlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lsquant import METHODS, analyse, read_spectrum

ROOT = pathlib.Path(__file__).parent
PHOTO = ROOT / "assets" / "kim.jpg"   # drop a local copy here to avoid hot-linking
PHOTO_URL = "https://static.wixstatic.com/media/2539f8_151e9dd50d0246b8949104463c3a3ad3~mv2.jpg"
SITE = "https://www.k-sentech.com/"

st.set_page_config(page_title="LSQUANT-Web", page_icon="🧪", layout="wide")


def load(src):
    if isinstance(src, pathlib.Path):
        return read_spectrum(src.read_bytes(), src.name)
    return read_spectrum(src.getvalue(), src.name)


def analysis_tab():
    st.title("LSQUANT-Web")
    st.caption("Quantitative IR analysis of polymer blends by least-squares curve fitting "
               "(web port of Prof. Kap Jin Kim's 1994 LSQUANT program)")
    demo = st.toggle("Use demo data (PVDF / PMMA blends, 10–90 wt% PVDF)", value=True)
    nominal = {}
    if demo:
        names, d = ["PVDF", "PMMA"], ROOT / "sample_data"
        refs_src = [d / "VDF.ASP", d / "MMA.ASP"]
        mix_src = [d / f"V{k}M{10 - k}.ASP" for k in range(1, 10)]
        nominal = {p.name: 10.0 * k for k, p in zip(range(1, 10), mix_src)}
        st.caption("Nominal compositions are inferred from the file names (V1M9 = 1 : 9).")
    else:
        n = st.slider("Number of components", 2, 5, 2)
        names, refs_src = [], []
        for i, col in enumerate(st.columns(n)):
            with col:
                names.append(st.text_input(f"Component {i + 1} name", f"Polymer {chr(65 + i)}", key=f"n{i}"))
                refs_src.append(st.file_uploader(f"Pure spectrum {i + 1}", key=f"r{i}"))
        mix_src = st.file_uploader("Mixture spectra (one or more) — .ASP or two-column .PRN/.TXT/.CSV",
                                   accept_multiple_files=True)
        if not mix_src or any(s is None for s in refs_src):
            st.info("Upload every pure-component spectrum and at least one mixture spectrum.")
            return
    try:
        refs = [load(s) for s in refs_src]
        mixes = {s.name: load(s) for s in mix_src}
    except ValueError as e:
        st.error(str(e)); return

    lo0, hi0 = max(r[0].min() for r in refs), min(r[0].max() for r in refs)
    with st.expander("Fit settings", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        lo = c1.number_input("Lower wavenumber (cm⁻¹)", value=float(round(lo0)))
        hi = c2.number_input("Upper wavenumber (cm⁻¹)", value=float(round(hi0)))
        amin = c3.number_input("Min. threshold absorbance", value=0.0)
        amax = c4.number_input("Max. threshold absorbance", value=2.0)

    rows, results = [], {}
    for name, (xm, ym) in mixes.items():
        try:
            res = analyse(xm, ym, refs, names, lo, hi, amin, amax)
        except ValueError as e:
            st.error(f"{name}: {e}"); continue
        results[name] = res
        for key, r in res.items():
            row = {"Mixture": name, "Method": METHODS[key]}
            for nm in names:
                row[f"{nm} wt%"] = r["weight_pct"][nm]
                row[f"{nm} content"] = r["content"][nm]
                row[f"{nm} content ±SE"] = r["content_se"][nm]
            row.update({"Std. error": r["std_error"], "R": r["corr"], "N points": r["n_points"]})
            if name in nominal:
                row[f"Nominal {names[0]} wt%"] = nominal[name]
            rows.append(row)
    if not rows:
        return
    df = pd.DataFrame(rows)
    method = st.radio("Method", list(METHODS), format_func=METHODS.get, horizontal=True)
    if method != "ols":
        st.caption("⚠ Weighting formulas for methods 2 and 3 are reconstructed from the cited papers "
                   "(original source unavailable) — see the Method tab.")
    st.dataframe(df[df["Method"] == METHODS[method]].drop(columns="Method").round(4), hide_index=True)

    pick = st.selectbox("Show fit for", list(results))
    r = results[pick][method]
    fig = go.Figure()
    fig.add_scatter(x=r["x"], y=r["y"], name="Mixture (observed)")
    fig.add_scatter(x=r["x"], y=r["fit"], name="Least-squares fit")
    fig.add_scatter(x=r["x"], y=r["y"] - r["fit"], name="Residual")
    fig.add_vrect(x0=lo, x1=hi, fillcolor="grey", opacity=0.08, line_width=0)
    fig.update_xaxes(autorange="reversed", title="Wavenumber (cm⁻¹)")
    fig.update_yaxes(title="Absorbance")
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("Download all results (CSV)", df.to_csv(index=False), "lsquant_results.csv", "text/csv")


def method_tab():
    st.header("Method")
    st.markdown("The mixture spectrum is modelled as a linear combination of the pure-component "
                "(reference) spectra over the chosen wavenumber window, using every data point "
                "whose absorbance lies between the minimum and maximum thresholds:")
    st.latex(r"A_{mix}(\nu_i)\;=\;\sum_{j=1}^{p} c_j\,A_j(\nu_i)\;+\;e_i,\qquad "
             r"\min\sum_i w_i\,e_i^{2},\qquad \mathrm{wt\%}_j=\frac{100\,c_j}{\sum_k c_k}")
    st.markdown("""
| Method | Weights *wᵢ* | Status |
|---|---|---|
| 1. Non-weighted (Haaland) | 1 | exact |
| 2. Weighted (Haaland) | 10^(−2·A) (noise in absorbance grows with A) | reconstructed |
| 3. Weighted (Antoon) | 1 / A² (equal relative error) | reconstructed, experimental |

The original BASIC source code was not available, only the compiled DOS program, which prints
these three results. Methods 2 and 3 therefore follow the cited papers rather than the original
code. Compare against the DOS output and edit `WEIGHTS` in `lsquant/core.py` to match it exactly.

**Reported:** content *cⱼ* and its standard error, weight %, standard error of the estimate
(√(Σwe²/(n−p)), weights normalised to mean 1), and the correlation between fitted and observed spectrum.

**References:** M. K. Antoon, J. H. Koenig, J. L. Koenig, *Appl. Spectrosc.* **31**, 518 (1977);
D. M. Haaland, R. G. Easterling, *Appl. Spectrosc.* **36**, 665 (1982).

**Demo check:** on the nine PVDF/PMMA blends (10–90 wt%), method 1 recovers the nominal
composition within about 1 wt%.""")


def about_tab():
    st.header("About")
    left, right = st.columns([1, 2.4])
    with left:
        st.image(str(PHOTO) if PHOTO.exists() else PHOTO_URL, caption="Prof. Kap Jin Kim", use_container_width=True)
        st.link_button("Personal website", SITE)
    with right:
        st.subheader("Prof. Kap Jin Kim, Ph.D.")
        st.markdown("""
Professor Emeritus, Dept. of Advanced Materials Engineering for Information & Electronics,
College of Engineering, **Kyung Hee University**, Korea.

- Ph.D. in Textile Engineering, Seoul National University; worked at Kolon Research Institute before joining Kyung Hee University in 1980.
- Visiting professor at the University of Massachusetts Amherst (1988–89), the University of Akron (1997–98) and the University of Washington (2008–09).
- President of the Korean Fiber Society (2010); editor of *Fibers and Polymers* since 2004.
- 164 peer-reviewed papers, 13 books/chapters and 30 patents.
- Recent research: piezoelectric PVDF and PLA films and nanowebs, and flexible sensors for smart textiles.

*(Source: his personal website.)*""")
    st.subheader("History of this program")
    st.markdown("""
- **LSQUANT.BAS, 19 June 1994** — written by Prof. Kap Jin Kim as a DOS program (compiled QuickBASIC) to determine the content of each component, 2 to 5, in a mixture from IR spectra by least-squares curve fitting, following Antoon–Koenig–Koenig (1977) and Haaland–Easterling (1982).
- It read ASCII spectrum files (`.ASP`), asked for a wavenumber window and minimum/maximum absorbance thresholds, and printed three analyses: non-weighted, weighted (Haaland) and weighted (Antoon).
- It was distributed alongside the *Spectra Calc* spectroscopy package (Galactic Industries) with example data from PVDF/PMMA and PVDF-based blends.
- **This web version** keeps the same inputs and outputs in Python so it runs in any browser, with no DOS emulator.""")
    extra = ROOT / "assets" / "history_extra.md"
    if extra.exists():
        st.markdown(extra.read_text(encoding="utf-8"))


tab_run, tab_method, tab_about = st.tabs(["Analysis", "Method", "About"])
with tab_run:
    analysis_tab()
with tab_method:
    method_tab()
with tab_about:
    about_tab()
