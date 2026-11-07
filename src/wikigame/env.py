"""Gymnasium-style Wikipedia game environment over an offline WikiGraph.

Observation (text): target title+summary, current title, visited trail, and the
outgoing link titles (titles only, as in the paper). Action: a link title.
Rules from the paper: no revisiting, dead end ends the episode, step cap.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces

from .graph import WikiGraph

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def parse_action(text: str) -> str | None:
    """Extract the chosen title from '<think>..</think><answer>{"title": "..."}</answer>'.

    Returns None if the output is malformed (the env treats that as an invalid action).
    """
    m = ANSWER_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1).strip())
    except json.JSONDecodeError:
        return None
    title = obj.get("title") if isinstance(obj, dict) else None
    return title if isinstance(title, str) else None


SYSTEM_PROMPT = (
    "You are playing the Wikipedia Game. Starting from the current article, choose ONE link to click "
    "so that you reach the target article in as few steps as possible. Think inside <think></think>, "
    'then answer with JSON inside <answer></answer>, e.g. <answer>{"title": "Exact Link Title"}</answer>. '
    "The title MUST be copied exactly from the list of links."
)


def classify_invalid(graph: WikiGraph, cur: int, visited: set[int], target: int, action: str | None) -> str | None:
    """Why an action is invalid, or None if it is a legal move.

    Causes: unparseable | unknown_title | current_page | revisit | target_not_linked | not_linked.
    Pure (no env state) so the analysis app can also backfill episodes recorded before causes were stored.
    """
    if action is None:
        return "unparseable"
    node = graph.id_of(action)
    if node is None:
        return "unknown_title"
    if node == cur:
        return "current_page"
    if node in visited:
        return "revisit"
    if node not in graph.adj[cur]:
        return "target_not_linked" if node == target else "not_linked"
    return None


_FEEDBACK = {
    "unparseable": "I could not read an answer. Reply with <answer>{\"title\": \"Exact Link Title\"}</answer>.",
    "unknown_title": "'{a}' is not an article title. Copy a title exactly from the Links list.",
    "current_page": "You are already on '{a}'. Choose a different link.",
    "revisit": "'{a}' was already visited. Choose an unvisited link.",
    "target_not_linked": "'{a}' is the target, but it is not linked from this page. Choose a title from the Links list.",
    "not_linked": "'{a}' is not linked from this page. Choose a title from the Links list.",
}


@dataclass
class RewardConfig:
    success: float = 1.0
    step_cost: float = 0.02
    invalid: float = -0.1  # unknown title / revisit / unparseable
    dead_end: float = -0.2


class WikiGameEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        graph: WikiGraph,
        n_hops: int = 2,
        max_steps: int = 10,
        max_invalid: int = 3,
        max_links_shown: int = 200,
        reward: RewardConfig | None = None,
        seed: int | None = None,
        feedback: bool = False,
    ):
        self.graph, self.n_hops, self.max_steps = graph, n_hops, max_steps
        self.max_invalid, self.max_links_shown = max_invalid, max_links_shown
        self.feedback = feedback  # if True, the observation after an invalid action says why it was invalid
        self._feedback_msg: str | None = None
        self.rc = reward or RewardConfig()
        self.rng = random.Random(seed)
        self.observation_space = spaces.Text(max_length=100_000)
        self.action_space = spaces.Text(max_length=1_000)

    # -- helpers ---------------------------------------------------------
    def _links(self) -> list[str]:
        """Unvisited outgoing links, capped at max_links_shown (target always kept if linked)."""
        cand = [v for v in self.graph.adj[self.cur] if v not in self.visited]
        if len(cand) > self.max_links_shown:
            keep = self.rng.sample(cand, self.max_links_shown)
            if self.target in cand and self.target not in keep:
                keep[0] = self.target
            cand = keep
        self.rng.shuffle(cand)
        return [self.graph.titles[v] for v in cand]

    def _obs(self) -> str:
        g = self.graph
        self.links = self._links()
        trail = " -> ".join(g.titles[v] for v in self.path)
        links = "\n".join(f"- {t}" for t in self.links)
        note = f"Your last answer was invalid: {self._feedback_msg}\n" if self.feedback and self._feedback_msg else ""
        return (
            f"Target: {g.titles[self.target]}\nTarget summary: {g.summaries[self.target]}\n\n"
            f"Path so far: {trail}\nCurrent article: {g.titles[self.cur]}\n"
            f"Steps left: {self.max_steps - self.steps}\n{note}\nLinks:\n{links}"
        )

    # -- gym API ---------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = random.Random(seed)
        options = options or {}
        if "source" in options and "target" in options:
            self.source, self.target = options["source"], options["target"]
        else:
            self.source, self.target = self.graph.sample_pair(options.get("n_hops", self.n_hops), self.rng)
        self.cur, self.path, self.visited = self.source, [self.source], {self.source}
        self.steps = self.invalid = 0
        self._feedback_msg = None
        self.optimal = self.graph.shortest_dist(self.source, self.target)
        return self._obs(), {"optimal_steps": self.optimal, "source": self.source, "target": self.target}

    def step(self, action: str | None):
        """action = chosen link title (use parse_action on raw model text first). None = unparseable."""
        info: dict = {"valid": False}
        reward = -self.rc.step_cost
        cause = classify_invalid(self.graph, self.cur, self.visited, self.target, action)
        if cause is not None:
            self.invalid += 1
            reward += self.rc.invalid
            truncated = self.invalid >= self.max_invalid or self.steps + self.invalid >= self.max_steps
            info["reason"] = "invalid_action"
            info["cause"] = cause
            left = max(0, self.max_invalid - self.invalid)
            self._feedback_msg = _FEEDBACK[cause].replace("{a}", str(action)) + f" ({left} invalid answer{'s' if left != 1 else ''} left)"
            return self._obs(), reward, False, truncated, info

        node = self.graph.id_of(action)
        self._feedback_msg = None
        info["valid"] = True
        self.cur = node
        self.path.append(node)
        self.visited.add(node)
        self.steps += 1
        terminated = node == self.target
        if terminated:
            reward += self.rc.success
            info["success"] = True
            info["extra_steps"] = self.steps - self.optimal
        elif not [v for v in self.graph.adj[node] if v not in self.visited]:
            reward += self.rc.dead_end
            terminated = True
            info["reason"] = "dead_end"
        truncated = (not terminated) and self.steps >= self.max_steps
        if truncated:
            info["reason"] = "step_limit"
        return self._obs(), reward, terminated, truncated, info
