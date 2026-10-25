# LSQUANT-Web
Browser version of LSQUANT (Prof. Kap Jin Kim, 1994): least-squares IR quantitative analysis of polymer blends.

Run locally:  `pip install -r requirements.txt && streamlit run app.py`

Share with others (free): push this folder to GitHub, then deploy at share.streamlit.io (main file `app.py`),
or create a Hugging Face Space (SDK: Streamlit).

- Photo: save a copy as `assets/kim.jpg` (otherwise it is loaded from k-sentech.com).
- Extra history text: put Markdown in `assets/history_extra.md`; it appears on the About tab.
- Methods 2 and 3 weights are in `lsquant/core.py` (`WEIGHTS`).
