"""Play: you play the same game the agent plays (same env, rules, observations) to get a human baseline."""
import pandas as pd
import streamlit as st

from wikigame.eval import summarize
from wikigame.webapp import st_helpers as H
from wikigame.webapp.play import human_text, new_game, pair_for, upload_human_run

st.title("🎮 Play the game")
st.caption("You see exactly what the agent sees: the target's title and summary, and the *titles* of the links on "
           "the current page. No going back to pages you've visited; the game ends at the target, a dead end, or "
           "the step limit.")

S = st.session_state
S.setdefault("pl_games", [])
S.setdefault("pl_game_no", 0)

graphs = H.available_graphs()
if not graphs:
    st.info("No graph files in `data/`. Build one with `uv run python -I -m wikigame.data wikispeedia`.")
    st.stop()

with st.sidebar:
    graph_name = st.selectbox("Graph", graphs)
    n_hops = st.slider("Hops to target", 1, 4, 2)
    max_steps = st.number_input("Step limit", 3, 30, 10, help="10 matches the baseline runs.")
    held_out = st.toggle("Use the held-out eval pairs", value=True,
                         help="Same pairs the baselines were evaluated on, so your results are directly comparable.")
graph = H.graph(graph_name)


def start_game() -> None:
    pair = pair_for(graph, n_hops, S.pl_game_no, held_out, seed=int(S.pl_game_no))
    S.pl_env, S.pl_rec, S.pl_obs = new_game(graph, n_hops, int(max_steps), pair)
    S.pl_done, S.pl_graph, S.pl_game_no = None, graph_name, S.pl_game_no + 1


def move(title: str) -> None:
    env, rec = S.pl_env, S.pl_rec
    _, reward, terminated, truncated, info = env.step(title)
    rec.record(human_text(title), title, reward, info)
    if terminated or truncated:
        finish()


def finish(reason: str | None = None) -> None:
    ep = S.pl_rec.finish(reason)
    S.pl_done = ep
    S.pl_games.append(ep)


st.sidebar.button("▶ New game", on_click=start_game, width="stretch", type="primary")

env = S.get("pl_env")
if env is None or S.get("pl_graph") != graph_name:
    st.info("Pick a graph and difficulty in the sidebar, then press **New game**.")
else:
    g, rec, done = env.graph, S.pl_rec, S.get("pl_done")
    ep = rec.ep
    st.markdown(f"### 🎯 Target: **{ep.target}**")
    st.markdown(f"> {g.summaries[env.target] or '_(no summary)_'}")
    st.markdown("**Path so far:** " + " → ".join(f"`{t}`" for t in ep.path))
    if not done:
        left = env.max_steps - env.steps
        st.markdown(f"## 📍 You are on: **{g.titles[env.cur]}**  ·  {left} step{'s' if left != 1 else ''} left")
        q = st.text_input("Filter links", placeholder="type to narrow the list…", key=f"flt_{env.steps}")
        links = [t for t in env.links if q.lower() in t.lower()]
        shown = links[:60]
        st.caption(f"{len(env.links)} links on this page" + (f" · showing {len(shown)} of {len(links)} matching" if len(links) > 60 or q else ""))
        cols = st.columns(3)
        for i, t in enumerate(sorted(shown)):
            cols[i % 3].button(t, key=f"lnk_{env.steps}_{t}", on_click=move, args=(t,), width="stretch")
        st.button("🏳️ Give up", on_click=finish, args=("gave_up",))
    else:
        if done.success:
            st.success(f"🎉 Reached **{done.target}** in {len(done.path) - 1} steps (optimal {done.optimal_steps}, "
                       f"{done.extra_steps} extra).")
        else:
            st.error(f"Game over: {done.end_reason.replace('_', ' ')}. The target was {done.optimal_steps} steps away.")
        opt = g.shortest_path(g.id_of(done.source), g.id_of(done.target))
        st.markdown("**A shortest path:** " + " → ".join(f"`{g.titles[i]}`" for i in opt))

# ---- results & upload ------------------------------------------------------------------------------
games = S.pl_games
if games:
    st.divider()
    st.subheader(f"Your games ({len(games)})")
    m = summarize(games)
    c = st.columns(4)
    c[0].metric("Success", f"{m['success']:.0%}")
    c[1].metric("Mean steps", f"{m['path_len']:.1f}")
    ex = [e.extra_steps for e in games if e.success]
    c[2].metric("Extra steps (wins)", f"{sum(ex) / len(ex):.2f}" if ex else "–")
    c[3].metric("Games", len(games))
    st.dataframe(pd.DataFrame([{"hops": e.n_hops, "game": f"{e.source} → {e.target}", "result": "✅" if e.success else e.end_reason,
                                "steps": len(e.path) - 1, "optimal": e.optimal_steps, "extra": e.extra_steps} for e in games]),
                 hide_index=True, width="stretch")
    st.markdown("**Upload as a human baseline.** Creates one W&B run in `aiwizards/wiki-game` "
                "(`job_type=human`, tag `human`) with these games; nothing is uploaded until you press the button.")
    if st.button(f"☁️ Upload {len(games)} game{'s' if len(games) != 1 else ''} to W&B"):
        url = upload_human_run(games, graph, S.get("pl_graph", graph_name), int(max_steps))
        st.success(f"Uploaded: {url}")
    st.button("Clear my games", on_click=lambda: S.update(pl_games=[], pl_game_no=0))
