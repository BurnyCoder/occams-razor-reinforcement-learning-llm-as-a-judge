"""Pure reward algebra and representation-dependent baseline measurements."""

import ast
import io
import re
import tokenize

from .schema import Judgment


def regions(text: str, tag: str) -> list[str]:
    """Keep all marked regions for length scoring, not just the submitted solution."""
    return re.findall(fr"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)


def code_regions(text: str) -> list[str]:
    """Count both explicit submission tags and fenced code in reasoning."""
    return regions(text, "code") + re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)


def formal_length(text: str) -> int:
    """Lexical math tokens plus non-comment/non-space code characters.

    Python tokenize distinguishes '#' in a string from a comment:
    https://docs.python.org/3/library/tokenize.html
    """
    codes = code_regions(text)
    total = 0
    for code in codes:
        try:
            tokens = tokenize.generate_tokens(io.StringIO(code).readline)
            total += sum(len(re.sub(r"\s", "", tok.string)) for tok in tokens
                         if tok.type not in (tokenize.COMMENT, tokenize.ENCODING))
        except (tokenize.TokenError, IndentationError):
            total += len(re.sub(r"\s", "", code))
    math = re.findall(r"\$\$?(.*?)\$\$?|\\\[(.*?)\\\]|\\\((.*?)\\\)", text, re.DOTALL)
    for match in math:
        total += len(re.findall(r"\\[A-Za-z]+|[A-Za-z]+|\d+(?:\.\d+)?|[^\s]", "".join(match)))
    return total


def code_structure(text: str) -> dict:
    """AST metrics are diagnostics; they are not used as unregistered rewards."""
    nodes, depth, branches = 0, 0, 0
    for code in code_regions(text):
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        stack = [(tree, 0)]
        while stack:
            node, level = stack.pop()
            nodes += 1
            depth = max(depth, level)
            branches += isinstance(node, (ast.If, ast.For, ast.While, ast.IfExp, ast.Match))
            stack.extend((child, level + 1) for child in ast.iter_child_nodes(node))
    return {"ast_nodes": nodes, "ast_depth": depth, "ast_branches": branches}


def reward(correct: bool, judgment: Judgment | None, arm: str, token_length: int,
           symbol_length: int, normalizers: dict | None = None, bonus: float = 0.2,
           shuffled_score: float | None = None) -> float:
    """Correctness dominates each sample; sufficiency gates every simplicity arm."""
    if arm not in ("accuracy", "occam", "token", "formal", "shuffled"):
        raise ValueError(f"Unknown reward arm {arm}")
    if not correct:
        return 0.0
    if arm == "accuracy" or judgment is None or not judgment.adequate:
        return 1.0
    if arm in ("token", "formal"):
        if not normalizers or normalizers[arm] <= 0:
            raise ValueError("Length normalizers must be frozen after baseline calibration")
        k = normalizers[arm]
        length = token_length if arm == "token" else symbol_length
        if length < 0:
            raise ValueError("Negative length")
        score = k / (k + length)
    else:
        score = judgment.simplicity if arm == "occam" else shuffled_score
        if score is None or not 0 <= score <= 1:
            raise ValueError("Shuffled score must be a valid eligible-group permutation")
    return 1.0 + bonus * score
