"""Wiki-game RL analysis app.   uv run streamlit run app/streamlit_app.py

Reads runs and episode artifacts from W&B (aiwizards/wiki-game), so runs from the Mac and the rented
CUDA box show up together. Needs `wandb login` on this machine.
"""
import streamlit as st

st.set_page_config(page_title="Wiki-game RL", page_icon="🧭", layout="wide")

pages = [
    st.Page("views/runs.py", title="Runs", icon="📊", default=True),
    st.Page("views/episodes.py", title="Episodes", icon="🔍"),
]
st.navigation(pages).run()
