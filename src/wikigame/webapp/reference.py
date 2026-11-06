"""Reference numbers from the paper, for orientation only (NOT comparable to our hop-bucketed results).

Source: Hickey, Knight & Bishop, "Can Language Models Play the Wikipedia Game?" (arXiv 2610.13948v1),
Table 4 (Appendix I), English Wikipedia column. Status: values were read through a model-extracted
summary of the table, NOT verbatim: check against the paper before quoting them anywhere.
Left out on purpose: GPT-o3 mini and the human success rate, which could not be pinned down (the text
says o3 mini was best and comparable to humans, yet Table 4 as extracted has GPT-4o highest).
"""

PAPER_ENGLISH = {  # agent -> (success rate, stated ± error)
    "GPT-4o (100 games only)": (0.71, 0.05),
    "DeepSeek-R1 70B (reasoning)": (0.64, 0.03),
    "E5-Large embeddings": (0.53, 0.00),
    "Llama 3.1 405B": (0.47, 0.03),
    "TF-IDF": (0.47, 0.02),
    "GPT-4o mini": (0.36, 0.05),
}

CAVEATS = (
    "Paper: full English Wikipedia (~6.9M articles), random navigable start/target pairs, 100-step limit, "
    "success = reached the target within 100 steps. Ours: a 4.6k-article graph, pairs at exact hop distances 1–4, "
    "10-step limit. Treat the paper as a rough sense of scale only."
)
