"""Synthetic tau2 retail tasks for training: new customers and orders from tau2's retail database, in the shapes of the
74 train tasks, each with expected actions computed in code and verified by replaying them on the database.

    python scripts/make_synthetic_tasks.py            # writes data/synthetic/retail_tasks.json and a report

Training only. The gate stays on the official held-out tasks and the hidden split is never touched: no customer, order or
item that any held-out or hidden task mentions may appear in a synthetic task (``Forbidden``, enforced in ``generate`` and
re-checked by ``leaks``). A run uses them through the tau2 split ``synthetic`` (``register_split``).

Task kinds (the train tasks' shapes):
  cancel, modify_address, modify_items, return, exchange, user_address   one write
  multi                                                                  two or three writes, across two orders
  refuse_*                                                               a request the policy refuses: no write at all
"""

from __future__ import annotations

import copy
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RETAIL_DATA = ROOT / "third_party" / "tau2-bench" / "data" / "tau2" / "domains" / "retail"
SYNTHETIC_PATH = ROOT / "data" / "synthetic" / "retail_tasks.json"
SPLIT_NAME = "synthetic"
WRITE_TOOLS = ("cancel_pending_order", "modify_pending_order_address", "modify_pending_order_items",
               "return_delivered_order_items", "exchange_delivered_order_items", "modify_user_address")
ORDER_ID = re.compile(r"#W\d{7}")
ITEM_ID = re.compile(r"\b\d{10}\b")


def load_db() -> dict:
    return json.loads((RETAIL_DATA / "db.json").read_text())


def official_tasks() -> list[dict]:
    return json.loads((RETAIL_DATA / "tasks.json").read_text())


# ---------------------------------------------------------------- the leak guard

@dataclass
class Forbidden:
    """Everything a held-out or hidden task mentions: none of it may appear in a synthetic task."""
    user_ids: set[str] = field(default_factory=set)
    order_ids: set[str] = field(default_factory=set)
    item_ids: set[str] = field(default_factory=set)


def forbidden_from(tasks: list[dict], db: dict) -> Forbidden:
    """Users, orders and items named anywhere in ``tasks`` (actions or customer text), plus every order of a named user."""
    forbidden = Forbidden()
    by_email = {user["email"]: user_id for user_id, user in db["users"].items()}
    by_name_zip = {(user["name"]["first_name"], user["name"]["last_name"], user["address"]["zip"]): user_id
                   for user_id, user in db["users"].items()}
    for task in tasks:
        text = json.dumps(task)
        forbidden.order_ids |= set(ORDER_ID.findall(text))
        forbidden.item_ids |= set(ITEM_ID.findall(text))
        for action in (task.get("evaluation_criteria") or {}).get("actions") or []:
            arguments = action.get("arguments") or {}
            if "user_id" in arguments:
                forbidden.user_ids.add(arguments["user_id"])
            if "email" in arguments and arguments["email"] in by_email:
                forbidden.user_ids.add(by_email[arguments["email"]])
            key = (arguments.get("first_name"), arguments.get("last_name"), arguments.get("zip"))
            if key in by_name_zip:
                forbidden.user_ids.add(by_name_zip[key])
    for order_id in list(forbidden.order_ids):
        if order_id in db["orders"]:
            forbidden.user_ids.add(db["orders"][order_id]["user_id"])
    for user_id in forbidden.user_ids:
        forbidden.order_ids |= set(db["users"].get(user_id, {}).get("orders", []))
    return forbidden


def protected_forbidden(db: dict) -> Forbidden:
    """The leak guard for training data: everything in the held-out and hidden tasks (configs/splits.yaml)."""
    from shipwright.heal import splits

    protected = set(splits.heldout_ids()) | set(splits.hidden_ids())
    return forbidden_from([task for task in official_tasks() if str(task["id"]) in protected], db)


def leaks(task: dict, forbidden: Forbidden) -> list[str]:
    """What a task shares with the forbidden sets (empty when it is clean)."""
    text = json.dumps(task)
    found = [f"order {o}" for o in ORDER_ID.findall(text) if o in forbidden.order_ids]
    found += [f"item {i}" for i in ITEM_ID.findall(text) if i in forbidden.item_ids]
    found += [f"user {u}" for u in forbidden.user_ids if u in text]
    return found


# ---------------------------------------------------------------- wording

STYLES = ["You are polite but in a hurry.", "You are detail-oriented and want everything handled in one go.",
          "You are a bit frustrated but cooperative.", "You are friendly and chatty.",
          "You are cautious and ask the agent to confirm before anything changes.",
          "You keep answers short and only share details when asked."]
CANCEL = ["You want to cancel order {order} because {why}.", "Please cancel your order {order}: {why}.",
          "You placed order {order} and now want it cancelled, since {why}."]
REASONS = {"no longer needed": "you no longer need it", "ordered by mistake": "you ordered it by mistake"}
RETURN = ["You received order {order} and want to return the {items}. Refund it to {pay}.",
          "You'd like to send back the {items} from order {order}; the refund should go to {pay}."]
EXCHANGE = ["You received order {order} and want to exchange the {old} for one with {new}.",
            "In order {order}, swap the {old} for the version with {new}."]
MODIFY = ["Your order {order} hasn't shipped yet; change the {old} to the one with {new}.",
          "Before order {order} ships, you want the {old} switched to {new}."]
ADDRESS = ["You moved: send order {order} to {address} instead.",
           "Please update the shipping address of order {order} to {address}."]
USER_ADDRESS = ["You moved and want your account's default address changed to {address}.",
                "Update the address on your profile to {address}."]
REFUSE = {"refuse_cancel": "You want to cancel order {order} (it has already been {status}). If the agent says it "
                           "cannot, accept that and end the conversation.",
          "refuse_return": "You want to return the {items} from order {order}, which has not been delivered yet. If the "
                           "agent says it cannot, accept that and end the conversation."}
NEW_ADDRESSES = [("17 Elm Street", "Apt 4", "Austin", "TX", "USA", "78701"),
                 ("902 Lakeview Drive", "", "Columbus", "OH", "USA", "43004"),
                 ("55 Harbor Road", "Unit 12", "Seattle", "WA", "USA", "98101"),
                 ("3310 Pine Avenue", "Suite 7", "Denver", "CO", "USA", "80202")]


def describe(options: dict) -> str:
    return ", ".join(f"{key} {value}" for key, value in options.items())


def option_change(old: dict, new: dict) -> str:
    return ", ".join(f"{key} {new[key]}" for key in new if old.get(key) != new[key])


def address_text(address: tuple) -> str:
    address1, address2, city, state, _, zip_code = address
    return f"{address1}{', ' + address2 if address2 else ''}, {city}, {state} {zip_code}"


def address_arguments(address: tuple) -> dict:
    return dict(zip(("address1", "address2", "city", "state", "country", "zip"), address))


# ---------------------------------------------------------------- building tasks

@dataclass
class Draft:
    kind: str
    user_id: str
    requests: list[str]
    actions: list[dict]
    writes: int


class Builder:
    def __init__(self, db: dict, forbidden: Forbidden, rng: random.Random):
        self.db, self.forbidden, self.rng = db, forbidden, rng

    def usable_orders(self, user_id: str, status: str) -> list[dict]:
        orders = [self.db["orders"][o] for o in self.db["users"][user_id]["orders"] if o in self.db["orders"]]
        return [o for o in orders if o["status"] == status and o["order_id"] not in self.forbidden.order_ids
                and not any(item["item_id"] in self.forbidden.item_ids for item in o["items"])]

    def payment_of(self, order: dict) -> str:
        return order["payment_history"][0]["payment_method_id"]

    def other_variant(self, item: dict) -> dict | None:
        variants = self.db["products"][item["product_id"]]["variants"].values()
        choices = [v for v in variants if v["available"] and v["item_id"] != item["item_id"]
                   and v["item_id"] not in self.forbidden.item_ids]
        return self.rng.choice(choices) if choices else None

    def lookups(self, order: dict) -> list[dict]:
        return [action("get_order_details", order_id=order["order_id"])]

    def cancel(self, user_id):
        order = self.pick(self.usable_orders(user_id, "pending"))
        if not order:
            return None
        reason = self.rng.choice(list(REASONS))
        text = self.rng.choice(CANCEL).format(order=order["order_id"], why=REASONS[reason])
        return Draft("cancel", user_id, [text], self.lookups(order) + [
            action("cancel_pending_order", order_id=order["order_id"], reason=reason)], 1)

    def modify_address(self, user_id):
        order = self.pick(self.usable_orders(user_id, "pending"))
        if not order:
            return None
        new = self.rng.choice(NEW_ADDRESSES)
        text = self.rng.choice(ADDRESS).format(order=order["order_id"], address=address_text(new))
        return Draft("modify_address", user_id, [text], self.lookups(order) + [
            action("modify_pending_order_address", order_id=order["order_id"], **address_arguments(new))], 1)

    def swap(self, user_id, status, kind, templates, tool):
        order = self.pick(self.usable_orders(user_id, status))
        if not order:
            return None
        pairs = [(item, self.other_variant(item)) for item in order["items"]]
        pairs = [(old, new) for old, new in pairs if new is not None]
        if not pairs:
            return None
        pairs = self.rng.sample(pairs, k=min(len(pairs), self.rng.choice([1, 1, 2])))
        texts = [self.rng.choice(templates).format(order=order["order_id"], old=f"{old['name']} ({describe(old['options'])})",
                                                   new=option_change(old["options"], new["options"])) for old, new in pairs]
        products = [action("get_product_details", product_id=old["product_id"]) for old, _ in pairs]
        return Draft(kind, user_id, [" ".join(texts) + " Keep the same payment method."],
                     self.lookups(order) + products + [action(tool, order_id=order["order_id"],
                                                              item_ids=[old["item_id"] for old, _ in pairs],
                                                              new_item_ids=[new["item_id"] for _, new in pairs],
                                                              payment_method_id=self.payment_of(order))], 1)

    def give_back(self, user_id):
        order = self.pick(self.usable_orders(user_id, "delivered"))
        if not order:
            return None
        items = self.rng.sample(order["items"], k=min(len(order["items"]), self.rng.choice([1, 1, 2])))
        text = self.rng.choice(RETURN).format(order=order["order_id"], items=" and the ".join(i["name"] for i in items),
                                              pay="the original payment method")
        return Draft("return", user_id, [text], self.lookups(order) + [
            action("return_delivered_order_items", order_id=order["order_id"], item_ids=[i["item_id"] for i in items],
                   payment_method_id=self.payment_of(order))], 1)

    def user_address(self, user_id):
        new = self.rng.choice(NEW_ADDRESSES)
        text = self.rng.choice(USER_ADDRESS).format(address=address_text(new))
        return Draft("user_address", user_id, [text], [action("get_user_details", user_id=user_id),
                                                       action("modify_user_address", user_id=user_id,
                                                              **address_arguments(new))], 1)

    def refuse(self, user_id):
        if self.rng.random() < 0.5:
            order = self.pick(self.usable_orders(user_id, "delivered") + self.usable_orders(user_id, "processed"))
            if not order:
                return None
            text = REFUSE["refuse_cancel"].format(order=order["order_id"], status=order["status"])
            return Draft("refuse_cancel", user_id, [text], self.lookups(order), 0)
        order = self.pick(self.usable_orders(user_id, "pending"))
        if not order:
            return None
        text = REFUSE["refuse_return"].format(order=order["order_id"], items=order["items"][0]["name"])
        return Draft("refuse_return", user_id, [text], self.lookups(order), 0)

    def multi(self, user_id):
        """Two or three writes in one conversation, on different orders (so no write blocks another)."""
        makers = [self.cancel, self.modify_address, self.exchange, self.give_back, self.modify_items]
        self.rng.shuffle(makers)
        parts, used_orders = [], set()
        for maker in makers:
            draft = maker(user_id)
            if draft is None:
                continue
            orders = {a["arguments"].get("order_id") for a in draft.actions if a["name"] in WRITE_TOOLS}
            if orders & used_orders:
                continue
            parts.append(draft)
            used_orders |= orders
            if len(parts) == self.rng.choice([2, 2, 3]):
                break
        if len(parts) < 2:
            return None
        return Draft("multi", user_id, [p.requests[0] for p in parts],
                     [a for p in parts for a in p.actions], sum(p.writes for p in parts))

    def exchange(self, user_id):
        return self.swap(user_id, "delivered", "exchange", EXCHANGE, "exchange_delivered_order_items")

    def modify_items(self, user_id):
        return self.swap(user_id, "pending", "modify_items", MODIFY, "modify_pending_order_items")

    def pick(self, orders):
        return self.rng.choice(orders) if orders else None


def action(name: str, **arguments) -> dict:
    return {"name": name, "arguments": arguments}


def to_task(draft: Draft, number: int, db: dict, rng: random.Random) -> dict:
    user = db["users"][draft.user_id]
    if rng.random() < 0.5:
        known = f"You are {user['name']['first_name']} {user['name']['last_name']} in zip code {user['address']['zip']}."
        unknown = "You do not remember your email address."
        auth = action("find_user_id_by_name_zip", first_name=user["name"]["first_name"],
                      last_name=user["name"]["last_name"], zip=user["address"]["zip"])
    else:
        known, unknown = f"Your email is {user['email']}.", "You do not know your user id."
        auth = action("find_user_id_by_email", email=user["email"])
    actions = [auth] + draft.actions
    task_id = f"syn-{number:04d}"
    return {
        "id": task_id,
        "description": {"purpose": f"synthetic {draft.kind} task with {draft.writes} write(s)", "relevant_policies": None,
                        "notes": "Shipwright synthetic training task (scripts/make_synthetic_tasks.py)"},
        "user_scenario": {"persona": None, "instructions": {
            "task_instructions": rng.choice(STYLES), "domain": "retail",
            "reason_for_call": " Also, ".join(draft.requests), "known_info": known, "unknown_info": unknown}},
        "initial_state": None,
        "evaluation_criteria": {"actions": [dict(a, action_id=f"{task_id}_{n}", info=None) for n, a in enumerate(actions)],
                                "communicate_info": [], "nl_assertions": None, "reward_basis": ["DB"]},
    }


# ---------------------------------------------------------------- verification

def verify(task: dict) -> str | None:
    """Replay the expected actions on a fresh retail database. None if every call succeeds and each write changed the
    database the way it should; otherwise the reason it failed. For a refusal task, the refused write must fail."""
    from tau2.domains.retail.data_model import RetailDB
    from tau2.domains.retail.tools import RetailTools

    tools = RetailTools(RetailDB.load(str(RETAIL_DATA / "db.json")))
    before = copy.deepcopy(tools.db.model_dump())
    for item in task["evaluation_criteria"]["actions"]:
        try:
            getattr(tools, item["name"])(**item["arguments"])
        except Exception as error:  # noqa: BLE001 - any failure makes the task invalid
            return f"{item['name']} failed: {error}"
    after = tools.db.model_dump()
    writes = [a for a in task["evaluation_criteria"]["actions"] if a["name"] in WRITE_TOOLS]
    if writes and before == after:
        return "the writes did not change the database"
    if not writes:
        if before != after:
            return "a refusal task changed the database"
        return refusal_holds(task, tools)
    return None


def kind_of(task: dict) -> str:
    return task["description"]["purpose"].split()[1]


def writes_of(task: dict) -> int:
    return sum(1 for a in task["evaluation_criteria"]["actions"] if a["name"] in WRITE_TOOLS)


def refusal_holds(task: dict, tools) -> str | None:
    """The write a refusal task asks for must be one the tools reject (so refusing is the policy-correct answer)."""
    kind = kind_of(task)
    order_id = ORDER_ID.search(task["user_scenario"]["instructions"]["reason_for_call"]).group(0)
    order = tools.db.orders[order_id]
    try:
        if kind == "refuse_cancel":
            tools.cancel_pending_order(order_id=order_id, reason="no longer needed")
        else:
            tools.return_delivered_order_items(order_id=order_id, item_ids=[order.items[0].item_id],
                                               payment_method_id=order.payment_history[0].payment_method_id)
    except Exception:  # noqa: BLE001 - the expected outcome
        return None
    return "the refused write would have succeeded"


# ---------------------------------------------------------------- generation

KIND_WEIGHTS = {"multi": 45, "exchange": 10, "return": 10, "modify_items": 10, "cancel": 8, "modify_address": 6,
                "user_address": 2, "refuse": 14}


def generate(count: int, seed: int = 300) -> tuple[list[dict], dict]:
    """``count`` verified, leak-free tasks and a report of what was tried, dropped and kept."""
    rng = random.Random(seed)
    db = load_db()
    forbidden = protected_forbidden(db)
    builder = Builder(db, forbidden, rng)
    makers = {"multi": builder.multi, "exchange": builder.exchange, "return": builder.give_back,
              "modify_items": builder.modify_items, "cancel": builder.cancel, "modify_address": builder.modify_address,
              "user_address": builder.user_address, "refuse": builder.refuse}
    users = sorted(u for u in db["users"] if u not in forbidden.user_ids)
    kinds = [kind for kind, weight in KIND_WEIGHTS.items() for _ in range(weight)]
    tasks, report = [], {"tried": 0, "no_fit": 0, "failed_verification": 0, "leaked": 0, "failures": {}}
    used_users: dict[str, int] = {}
    while len(tasks) < count and report["tried"] < count * 20:
        report["tried"] += 1
        user_id = rng.choice(users)
        if used_users.get(user_id, 0) >= 2:  # spread tasks over many customers
            continue
        draft = makers[rng.choice(kinds)](user_id)
        if draft is None:
            report["no_fit"] += 1
            continue
        task = to_task(draft, len(tasks) + 1, db, rng)
        if leaks(task, forbidden):
            report["leaked"] += 1
            continue
        problem = verify(task)
        if problem:
            report["failed_verification"] += 1
            key = problem.split(":")[0]
            report["failures"][key] = report["failures"].get(key, 0) + 1
            continue
        tasks.append(task)
        used_users[user_id] = used_users.get(user_id, 0) + 1
    report["kept"] = len(tasks)
    report["kinds"] = {}
    for task in tasks:
        kind = kind_of(task)
        report["kinds"][kind] = report["kinds"].get(kind, 0) + 1
    report["writes"] = {n: sum(1 for t in tasks if writes_of(t) == n) for n in range(4)}
    report["forbidden"] = {"users": len(forbidden.user_ids), "orders": len(forbidden.order_ids),
                           "items": len(forbidden.item_ids)}
    return tasks, report


# ---------------------------------------------------------------- the tau2 split

def register_split(path: Path = SYNTHETIC_PATH) -> None:
    """Make tau2's retail task loader also answer the split ``synthetic`` (the tasks in ``path``). Other splits are
    unchanged; tau2's files are not edited."""
    from tau2.data_model.tasks import Task
    from tau2.registry import registry

    official_loader = registry._tasks["retail"]
    official_splits = registry._task_splits.get("retail")
    if getattr(official_loader, "_shipwright_synthetic", False):
        return

    def get_tasks(task_split_name=None):
        if task_split_name == SPLIT_NAME:
            return [Task.model_validate(task) for task in json.loads(Path(path).read_text())]
        return official_loader(task_split_name)

    def get_task_splits():
        splits = dict(official_splits()) if official_splits else {}
        splits[SPLIT_NAME] = [task["id"] for task in json.loads(Path(path).read_text())]
        return splits

    get_tasks._shipwright_synthetic = True
    registry._tasks["retail"] = get_tasks
    registry._task_splits["retail"] = get_task_splits
