"""Agent-R (arxiv 2501.11425): MCTS over tau2 runs, then revision trajectories for SFT.

Agent-R teaches the agent to notice its own mistake mid-run and recover. It has three parts:

1. MCTS (Agent-R section 3.1). A tree over one task. A node is a conversation prefix that
   ends where the agent acts next; an edge is one agent step (one assistant message plus
   the tool results / user reply it got back). Selection uses UCT (Eq. 5,
   ``Q + c * sqrt(ln N_parent / N)``, c = 0.25); expansion samples ``children`` (4)
   rollouts at temperature 1 from the selected node; each rollout runs to the end of the
   conversation and tau2's evaluator gives its reward, which is backed up the path.
   Every rollout path is kept in the tree, so every rollout is a leaf with a final reward.
2. Pairs (Eq. 4). A bad leaf (low reward) and a good leaf (reward >= alpha, at least
   beta above the bad one) that share the longest common prefix (the "adjacent correct
   path"). They diverge at step ``t``.
3. Revision trajectory (Eq. 3). The actor itself judges each bad step from ``t`` on
   (model-guided critique) and the first step judged Bad is ``t'``. The revision run is:
   shared prefix + bad steps ``t..t'`` + a revision thought + good steps ``t..end``.
   Only the revision thought and the good suffix are trained; the shared prefix and the
   bad steps are context (loss-masked), as in Eq. 6.

This module has no tau2 or network dependency: the tree, pairing, critique parsing and
splicing work on plain tau2 message dicts (the ``messages`` of a trace). The scripts
``scripts/agent_r_search.py`` and ``scripts/agent_r_revise.py`` call tau2 and the models.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field

# Paper values (Agent-R section 4.2 and Table 1).
C_UCT = 0.25
CHILDREN = 4
MAX_DEPTH = 20
ROLLOUT_TEMPERATURE = 1.0
BETA = 0.2  # the "distinguishable gap" between a good and a bad reward
ALPHA_BY_ITERATION = {1: 0.5, 2: 0.7, 3: 1.0}

USER_STOP_TOKENS = ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###")

# Revision thoughts. The paper samples one of ten short reflections; the first two are the
# paper's own wording, the rest follow its style. With the inline format the thought may
# reach the customer (when the next good action is a text reply), so they stay neutral.
REVISION_THOUGHTS = [
    "I realize my approach was flawed. I need to revise it.",
    "I overlooked something important. I need to address it.",
    "I made a mistake in my previous step. Let me correct it.",
    "My previous action was not right for this request. Let me fix that.",
    "I went in the wrong direction. Let me get back on track.",
    "That step did not follow the right procedure. I will correct it now.",
    "I should not have done that. Let me take the correct step instead.",
    "I need to reconsider what I just did. Here is the right next step.",
    "My last step was an error. I will handle this properly now.",
    "Looking back, my previous action was wrong. Let me revise my approach.",
]


# ---------------------------------------------------------------------------
# Steps: a tau2 conversation cut at each assistant message
# ---------------------------------------------------------------------------


def split_steps(messages: list[dict]) -> list[list[dict]]:
    """Cut a tau2 message list into agent steps.

    A step is one assistant message plus every message after it up to the next assistant
    message (tool results, the user's reply). Messages before the first assistant message,
    if any, form step 0. So the concatenation of the first k steps is always a prefix
    where the agent acts next (or the end of the run).
    """
    cut_indices = [index for index, message in enumerate(messages) if message.get("role") == "assistant"]
    if not cut_indices or cut_indices[0] != 0:
        cut_indices = [0, *cut_indices]
    cut_indices = sorted(set(cut_indices))

    steps = []
    step_ends = [*cut_indices[1:], len(messages)]
    for step_start, step_end in zip(cut_indices, step_ends):
        step = messages[step_start:step_end]
        if step:
            steps.append(step)
    return steps


def _call_key(tool_call: dict) -> list:
    function = tool_call.get("function") or {}
    name = tool_call.get("name", function.get("name"))
    arguments = tool_call.get("arguments", function.get("arguments"))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            pass
    return [name, arguments]


def message_key(message: dict) -> list:
    """What identifies a message across rollouts (no ids, timestamps or usage)."""
    call_keys = [_call_key(tool_call) for tool_call in (message.get("tool_calls") or [])]
    return [message.get("role"), (message.get("content") or "").strip(), call_keys]


def step_key(step: list[dict]) -> str:
    serialized_step = json.dumps(
        [message_key(message) for message in step], sort_keys=True, ensure_ascii=False, default=str
    )
    return hashlib.sha256(serialized_step.encode()).hexdigest()[:16]


def is_user_stop(message: dict) -> bool:
    content = message.get("content") or ""
    return message.get("role") == "user" and any(token in content for token in USER_STOP_TOKENS)


# ---------------------------------------------------------------------------
# MCTS tree
# ---------------------------------------------------------------------------


@dataclass
class Node:
    id: int
    parent: int | None
    depth: int  # number of steps from the root
    key: str = ""  # step_key of the step that leads here ("" for the root)
    step: list[dict] = field(default_factory=list)
    children: dict[str, int] = field(default_factory=dict)
    visits: int = 0
    value: float = 0.0  # sum of backed-up rewards
    expansions: int = 0  # rollouts started from this node
    terminal: bool = False
    reward: float | None = None  # final reward when terminal

    @property
    def q(self) -> float:
        if self.visits:
            return self.value / self.visits
        return 0.0


class Tree:
    """The MCTS tree of one task (Agent-R section 3.1)."""

    def __init__(
        self,
        task_id: str,
        *,
        c_uct: float = C_UCT,
        children: int = CHILDREN,
        max_depth: int = MAX_DEPTH,
    ):
        self.task_id = str(task_id)
        self.c_uct = c_uct
        self.n_children = children
        self.max_depth = max_depth
        self.nodes: list[Node] = [Node(id=0, parent=None, depth=0)]
        self.leaves: list[dict] = []  # one per rollout: {"path", "reward", "simulation_id"}

    @property
    def root(self) -> Node:
        return self.nodes[0]

    def path_to(self, node_id: int) -> list[int]:
        """Node ids from the root's first child down to node_id (the root itself is left out)."""
        path = []
        current_id: int | None = node_id
        while current_id is not None and current_id != 0:
            path.append(current_id)
            current_id = self.nodes[current_id].parent
        return path[::-1]

    def prefix(self, node_id: int) -> list[dict]:
        """The conversation that leads to a node (empty for the root)."""
        prefix_messages = []
        for path_node_id in self.path_to(node_id):
            prefix_messages += self.nodes[path_node_id].step
        return prefix_messages

    def expandable(self, node: Node) -> bool:
        return not node.terminal and node.depth < self.max_depth and node.expansions < self.n_children

    def uct(self, parent: Node, child: Node) -> float:
        if child.visits == 0:
            return math.inf
        exploration = self.c_uct * math.sqrt(math.log(max(parent.visits, 1)) / child.visits)
        return child.q + exploration

    def select(self) -> Node:
        """Descend by UCT to the first node that can still be expanded.

        Returns a terminal (or max-depth) node when the descent reaches one; the caller
        backs up its known value again without a new rollout.
        """
        node = self.root
        while not self.expandable(node):
            if node.terminal or not node.children:
                return node
            child_nodes = [self.nodes[child_id] for child_id in node.children.values()]
            node = max(child_nodes, key=lambda child: (self.uct(node, child), -child.id))
        return node

    def insert(
        self,
        from_node: int,
        messages: list[dict],
        reward: float,
        simulation_id: str | None = None,
        ended: bool = True,
    ) -> list[int]:
        """Add one rollout (its full message list) below ``from_node`` and back up its reward.

        ``messages`` is the whole conversation, prefix included; the prefix must match
        the path to ``from_node``. Returns the node path of the rollout from the root.
        """
        expected_prefix = self.prefix(from_node)
        prefix_length = len(expected_prefix)
        rollout_prefix_keys = [message_key(message) for message in messages[:prefix_length]]
        if rollout_prefix_keys != [message_key(message) for message in expected_prefix]:
            raise ValueError("rollout does not start with the node's prefix")

        node = self.nodes[from_node]
        for step in split_steps(messages[prefix_length:]):
            key = step_key(step)
            if key in node.children:
                node = self.nodes[node.children[key]]
                continue
            child = Node(id=len(self.nodes), parent=node.id, depth=node.depth + 1, key=key, step=step)
            self.nodes.append(child)
            node.children[key] = child.id
            node = child

        if ended:
            node.terminal = True
            if node.reward is None:
                node.reward = reward
            else:
                node.reward = max(node.reward, reward)
        path = self.path_to(node.id)
        self.backup(node.id, reward)
        self.leaves.append({"path": path, "reward": reward, "simulation_id": simulation_id})
        return path

    def backup(self, node_id: int, reward: float) -> None:
        current_id: int | None = node_id
        while current_id is not None:
            self.nodes[current_id].visits += 1
            self.nodes[current_id].value += reward
            current_id = self.nodes[current_id].parent

    def to_json(self) -> dict:
        return {
            "task_id": self.task_id,
            "c_uct": self.c_uct,
            "children": self.n_children,
            "max_depth": self.max_depth,
            "nodes": [
                {
                    "id": node.id,
                    "parent": node.parent,
                    "depth": node.depth,
                    "key": node.key,
                    "visits": node.visits,
                    "q": round(node.q, 4),
                    "expansions": node.expansions,
                    "terminal": node.terminal,
                    "reward": node.reward,
                    "children": list(node.children.values()),
                    "action": step_summary(node.step),
                }
                for node in self.nodes
            ],
            "leaves": self.leaves,
        }


def run_search(
    tree: Tree,
    rollout: Callable[[list[dict], int], list[dict]],
    simulations: int,
    on_rollout: Callable[[dict, list[int]], None] | None = None,
) -> dict:
    """MCTS loop: select, expand with ``children`` rollouts, back up. Returns counters.

    ``rollout(prefix_messages, n)`` runs n rollouts from a prefix and returns one dict per
    finished rollout: {"messages", "reward", "simulation_id", "ended"}; failed rollouts
    (infra errors) are left out. ``simulations`` caps the number of rollouts.
    """
    rollouts_started = 0
    revisit_count = 0
    failed_rollouts = 0
    idle_iterations = 0
    while rollouts_started < simulations:
        node = tree.select()
        if not tree.expandable(node):
            # Terminal or max depth: back up what is known, no new rollout.
            known_value = node.reward if node.reward is not None else node.q
            tree.backup(node.id, known_value)
            revisit_count += 1
            idle_iterations += 1
            if idle_iterations > 4 * tree.n_children or node is tree.root:
                break  # the reachable tree is exhausted
            continue

        idle_iterations = 0
        rollout_count = min(tree.n_children - node.expansions, simulations - rollouts_started)
        node.expansions += rollout_count
        rollout_results = rollout(tree.prefix(node.id), rollout_count)
        rollouts_started += rollout_count
        failed_rollouts += rollout_count - len(rollout_results)
        for rollout_result in rollout_results:
            path = tree.insert(
                node.id,
                rollout_result["messages"],
                float(rollout_result["reward"]),
                rollout_result.get("simulation_id"),
                rollout_result.get("ended", True),
            )
            if on_rollout:
                on_rollout(rollout_result, path)
    return {
        "rollouts": rollouts_started,
        "failed_rollouts": failed_rollouts,
        "revisits": revisit_count,
        "nodes": len(tree.nodes),
        "leaves": len(tree.leaves),
    }


# ---------------------------------------------------------------------------
# Pairs (Eq. 4)
# ---------------------------------------------------------------------------


def common_prefix(path_a: list[int], path_b: list[int]) -> int:
    """Length of the shared start of two node paths."""
    shared_length = 0
    for node_a, node_b in zip(path_a, path_b):
        if node_a != node_b:
            break
        shared_length += 1
    return shared_length


def select_pairs(
    leaves: list[dict],
    alpha: float,
    beta: float = BETA,
    max_pairs: int | None = None,
    seed: int = 123,
) -> list[dict]:
    """Bad/good leaf pairs of one task.

    ``leaves``: {"path": [node ids], "reward", ...}. A good leaf has reward >= alpha; a bad
    leaf has reward at least ``beta`` below it. Each bad leaf gets the good leaf with the
    longest shared path (the adjacent correct path), ties to the shorter good path. One
    pair per distinct bad path; ``max_pairs`` caps them (random sample, fixed seed).
    """
    good_leaves = [leaf for leaf in leaves if leaf["reward"] >= alpha]
    pairs = []
    seen_bad_paths = set()
    for bad_leaf in leaves:
        bad_path_key = tuple(bad_leaf["path"])
        if bad_path_key in seen_bad_paths:
            continue
        candidate_good_leaves = [
            good_leaf for good_leaf in good_leaves
            if good_leaf["reward"] - bad_leaf["reward"] >= beta and good_leaf["path"] != bad_leaf["path"]
        ]
        if not candidate_good_leaves:
            continue

        good_leaf = max(
            candidate_good_leaves,
            key=lambda candidate: (common_prefix(candidate["path"], bad_leaf["path"]), -len(candidate["path"])),
        )
        divergence_step = common_prefix(good_leaf["path"], bad_leaf["path"])
        if divergence_step >= len(bad_leaf["path"]) or divergence_step >= len(good_leaf["path"]):
            continue  # one path is a prefix of the other: nothing to splice
        seen_bad_paths.add(bad_path_key)
        pairs.append({"bad": bad_leaf, "good": good_leaf, "t": divergence_step})

    if max_pairs is not None and len(pairs) > max_pairs:
        pairs = random.Random(seed).sample(pairs, max_pairs)
    return pairs


def select_good(leaves: list[dict], alpha: float, max_good: int | None = None) -> list[dict]:
    """Distinct successful leaves (Eq. 6 also trains on good trajectories)."""
    good_leaves = []
    seen_paths = set()
    for leaf in sorted(leaves, key=lambda leaf: len(leaf["path"])):
        if leaf["reward"] >= alpha and tuple(leaf["path"]) not in seen_paths:
            seen_paths.add(tuple(leaf["path"]))
            good_leaves.append(leaf)
    if max_good is not None:
        return good_leaves[:max_good]
    return good_leaves


# ---------------------------------------------------------------------------
# Model-guided critique: find t'
# ---------------------------------------------------------------------------

CRITIC_SYSTEM = """You are a good verifier of interactive environments.
You check one step of a customer-support agent that talks to a customer and calls tools under a policy.
You get the agent's policy, the conversation so far, the agent's current action and what happened after it.
You need to verify whether the current action is good or bad or uncertain.
- Good: the action moves the customer's request forward and follows the policy.
- Bad: the action is harmful to solving the task, for example the wrong tool, wrong or invented arguments, \
a policy violation, a change the customer did not confirm, wrong information to the customer, \
or giving up or transferring when the task could be solved.
- Uncertain: it cannot be judged from the current information.
You must give reasons first and then give the response with the format: Judgment: <Good or Bad or Uncertain>"""

_JUDGMENT = re.compile(r"judge?ment\s*[:：]\s*[*_`\"'<\[]*\s*(good|bad|uncertain)", re.IGNORECASE)


def parse_judgment(text: str | None) -> str:
    """'good', 'bad' or 'uncertain' (the last Judgment line; unparseable counts as uncertain)."""
    judgments = _JUDGMENT.findall(text or "")
    if judgments:
        return judgments[-1].lower()
    return "uncertain"


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut_count = len(text) - limit + 20
    return text[: limit - 20] + f" ...[{cut_count} chars cut]"


def render_message(message: dict, limit: int = 600) -> str:
    role = message.get("role")
    content = (message.get("content") or "").strip()
    if role == "assistant":
        lines = [f"Agent: {_clip(content, limit)}"] if content else []
        for tool_call in message.get("tool_calls") or []:
            tool_name, arguments = _call_key(tool_call)
            lines.append(f"Agent action: {tool_name}({json.dumps(arguments, ensure_ascii=False)})")
        return "\n".join(lines) or "Agent: (empty reply)"
    if role == "tool":
        return f"Tool result: {_clip(content, limit)}"
    return f"Customer: {_clip(content, limit)}"


def render_steps(steps: list[list[dict]], limit: int = 600) -> str:
    return "\n".join(render_message(message, limit) for step in steps for message in step)


def critic_messages(
    policy: str,
    history: list[list[dict]],
    step: list[dict],
    policy_chars: int = 9000,
    history_chars: int = 9000,
) -> list[dict]:
    """The verifier prompt for one step (sized for Tiny Aya's 8k context)."""
    history_text = render_steps(history)
    if len(history_text) > history_chars:
        history_text = "[earlier turns omitted]\n" + history_text[-history_chars:]
    action_text = render_message(step[0]) if step else ""
    after_text = "\n".join(render_message(message) for message in step[1:]) or "(nothing yet)"
    user_message = (
        f"Policy:\n{_clip(policy, policy_chars)}\n\nConversation so far:\n{history_text or '(start)'}\n\n"
        f"Current action:\n{action_text}\n\nWhat happened after it:\n{after_text}\n\n"
        "Is the current action good, bad or uncertain? Give reasons first, then the line "
        "'Judgment: <Good or Bad or Uncertain>'."
    )
    return [{"role": "system", "content": CRITIC_SYSTEM}, {"role": "user", "content": user_message}]


def find_transition(
    bad_steps: list[list[dict]],
    t: int,
    mode: str,
    judge: Callable[[list[list[dict]], list[dict]], str] | None = None,
) -> dict:
    """The transition step t' in the bad run (index into ``bad_steps``, t <= t').

    mode "model": the actor judges steps t, t+1, ... and t' is the first one judged Bad
      (Agent-R). If none is judged Bad, t' is the last step (as in the paper's code, this
      falls back to a direct revision) and ``found`` is False.
    mode "direct": t' is the last step (the paper's Direct-Revision baseline).
    mode "divergence": t' = t, the first step after the shared prefix (no model calls;
      our cheap ablation, not in the paper).
    """
    last_step_index = len(bad_steps) - 1
    if mode == "direct":
        return {"t_prime": last_step_index, "found": True, "judgments": []}
    if mode == "divergence":
        return {"t_prime": t, "found": True, "judgments": []}
    if mode != "model":
        raise ValueError(f"unknown transition mode {mode!r}")
    if judge is None:
        raise ValueError("mode 'model' needs a judge")

    judgments = []
    for step_index in range(t, len(bad_steps)):
        raw_reply = judge(bad_steps[:step_index], bad_steps[step_index])
        verdict = parse_judgment(raw_reply)
        judgments.append({"step": step_index, "verdict": verdict, "raw": raw_reply})
        if verdict == "bad":
            return {"t_prime": step_index, "found": True, "judgments": judgments}
    return {"t_prime": last_step_index, "found": False, "judgments": judgments}


# ---------------------------------------------------------------------------
# Revision trajectory (Eq. 3)
# ---------------------------------------------------------------------------


def _strip_stop(step: list[dict]) -> list[dict]:
    """A bad step without a final user stop message: in the revision run the customer stays."""
    return [message for message in step if not is_user_stop(message)]


def build_revision(
    bad_steps: list[list[dict]],
    good_steps: list[list[dict]],
    t: int,
    t_prime: int,
    thought: str,
    fmt: str = "inline",
) -> dict:
    """Splice shared prefix + bad steps t..t' + revision thought + good steps t..end.

    Returns {"messages": tau2 message dicts, "n_context_assistant": the number of leading
    assistant messages that are context only (loss-masked)}. Formats:

    inline (default): the thought is put in front of the first good-suffix assistant
      message. The prompted protocol renders it as "<thought>\\n```json {...}```", and the
      live parser already accepts text before the JSON block, so the student can emit
      this at deployment with no extra turn.
    turn: the paper's literal signal: an assistant turn with the thought, then a user
      turn "OK.", then the good suffix.
    """
    if not (0 <= t <= t_prime < len(bad_steps)) or t >= len(good_steps):
        raise ValueError(f"bad split t={t} t'={t_prime} for {len(bad_steps)} bad / {len(good_steps)} good steps")

    context_messages: list[dict] = [message for step in bad_steps[:t] for message in step]
    for step_index in range(t, t_prime + 1):
        context_messages += _strip_stop(bad_steps[step_index])
    suffix_messages = [dict(message) for step in good_steps[t:] for message in step]

    if fmt == "inline":
        first_suffix_message = suffix_messages[0]
        if first_suffix_message.get("role") != "assistant":
            raise ValueError("good suffix must start with an assistant message")
        content = (first_suffix_message.get("content") or "").strip()
        if content:
            first_suffix_message["content"] = f"{thought}\n{content}"
        else:
            first_suffix_message["content"] = thought
        signal_messages: list[dict] = []
    elif fmt == "turn":
        signal_messages = [{"role": "assistant", "content": thought}, {"role": "user", "content": "OK."}]
    else:
        raise ValueError(f"unknown revision format {fmt!r}")

    context_assistant_count = sum(1 for message in context_messages if message.get("role") == "assistant")
    return {
        "messages": context_messages + signal_messages + suffix_messages,
        "n_context_assistant": context_assistant_count,
    }


def step_summary(step: list[dict]) -> str:
    """One short line for a tree node (tree files and the control room)."""
    if not step:
        return ""
    return render_message(step[0], 160).replace("\n", " | ")
