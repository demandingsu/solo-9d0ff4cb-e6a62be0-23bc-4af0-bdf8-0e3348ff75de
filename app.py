"""Flask UI for the rehearsal-part pagination solver."""

from __future__ import annotations

import os

from flask import Flask, render_template, request, session

from solver import (
    MAX_SECTIONS,
    MAX_STEPS,
    InputError,
    Page,
    solve,
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-secret-change-in-compose")
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")

DEFAULT_LINES = [8, 8, 8, 8]
DEFAULT_CAPACITY = 16
DEFAULT_STEPS = [1, 2, 3, 4]
DEFAULT_RESTS = [False, True, False]


# ---------------------------------------------------------------------------
# session helpers


def _dump_last(
    lines: list[int],
    capacity: int,
    pages: tuple[Page, ...],
    steps: list[int],
    rests: list[bool],
) -> dict:
    return {
        "lines": list(lines),
        "capacity": int(capacity),
        "pages": [[list(page.sections), page.reprint] for page in pages],
        "steps": list(steps),
        "rests": list(rests),
    }


def _load_last() -> dict | None:
    data = session.get("last")
    if not isinstance(data, dict):
        return None
    try:
        pages = tuple(
            Page(
                tuple(int(section) for section in sections),
                None if reprint is None else int(reprint),
            )
            for sections, reprint in data["pages"]
        )
        lines = [int(value) for value in data["lines"]]
        capacity = int(data["capacity"])
        steps = [int(value) for value in data["steps"]]
        rests = [bool(value) for value in data["rests"]]
    except (KeyError, TypeError, ValueError):
        return None
    if not pages or not lines:
        return None
    return {
        "lines": lines,
        "capacity": capacity,
        "pages": pages,
        "steps": steps,
        "rests": rests,
    }


# ---------------------------------------------------------------------------
# form parsing


def _parse_int(value: str, field: str) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        raise InputError(f"{field}必须是整数。") from None


def _parse_form(
    form,
) -> tuple[list[int], int, list[int], list[bool], int]:
    raw_lines = form.getlist("line")
    if not raw_lines:
        raise InputError("请至少填写一个小节的谱面行数。")
    lines = [
        _parse_int(value, f"第 {index + 1} 小节的谱面行数")
        for index, value in enumerate(raw_lines)
    ]
    capacity = _parse_int(form.get("capacity", ""), "每页行数")

    raw_steps = form.getlist("step")
    if not raw_steps:
        raise InputError("请至少录入一步演奏顺序。")
    steps = [
        _parse_int(value, f"第 {index + 1} 步的小节编号")
        for index, value in enumerate(raw_steps)
    ]
    rests = [form.get(f"rest_{index}") is not None for index in range(1, len(steps))]

    raw_lock = (form.get("lock_count") or "0").strip()
    lock_count = _parse_int(raw_lock or "0", "锁定页数")
    return lines, capacity, steps, rests, lock_count


# ---------------------------------------------------------------------------
# request handling


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/")
def index():
    last = _load_last()
    return render_template(
        "index.html",
        max_sections=MAX_SECTIONS,
        max_steps=MAX_STEPS,
        lines=DEFAULT_LINES,
        capacity=DEFAULT_CAPACITY,
        steps=DEFAULT_STEPS,
        rests=DEFAULT_RESTS,
        lock_count=0,
        last_pages=len(last["pages"]) if last else 0,
        outcome=None,
        error=None,
    )


@app.post("/solve")
def solve_route():
    last = _load_last()
    last_pages = len(last["pages"]) if last else 0

    # Defaults shown if the submitted form cannot be parsed at all.
    view = {
        "lines": DEFAULT_LINES,
        "capacity": DEFAULT_CAPACITY,
        "steps": DEFAULT_STEPS,
        "rests": DEFAULT_RESTS,
        "lock_count": 0,
    }
    error = None
    outcome = None

    try:
        (
            lines,
            capacity,
            steps,
            rests,
            lock_count,
        ) = _parse_form(request.form)
        view = {
            "lines": lines,
            "capacity": capacity,
            "steps": steps,
            "rests": rests,
            "lock_count": lock_count,
        }
        if lock_count < 0:
            raise InputError("锁定页数不能为负数。")

        locked: tuple[Page, ...] = ()
        if lock_count > 0:
            if last is None:
                raise InputError("还没有成功排过页，无法锁定。请先正常排一次。")
            if lock_count > len(last["pages"]):
                raise InputError(f"最多只能锁定上次排法的 {len(last['pages'])} 页。")
            locked = tuple(last["pages"][:lock_count])

        outcome = solve(lines, capacity, steps, rests, locked, last)
        if outcome.status == "solved":
            session["last"] = _dump_last(
                lines, capacity, tuple(_page_objects(outcome.pages)), steps, rests
            )
            last_pages = len(outcome.pages)
            view["lock_count"] = outcome.locked_pages
    except InputError as exc:
        error = str(exc)
        # Keep the user's input visible, even if some fields were invalid.
        raw_lines = request.form.getlist("line")
        raw_steps = request.form.getlist("step")
        if raw_lines:
            view["lines"] = raw_lines
        if raw_steps:
            view["steps"] = raw_steps
            view["rests"] = [
                request.form.get(f"rest_{index}") is not None
                for index in range(1, len(raw_steps))
            ]
        raw_capacity = request.form.get("capacity")
        if raw_capacity:
            view["capacity"] = raw_capacity
        raw_lock = request.form.get("lock_count")
        if raw_lock:
            view["lock_count"] = raw_lock

    return render_template(
        "index.html",
        max_sections=MAX_SECTIONS,
        max_steps=MAX_STEPS,
        lines=view["lines"],
        capacity=view["capacity"],
        steps=view["steps"],
        rests=view["rests"],
        lock_count=view["lock_count"],
        last_pages=last_pages,
        outcome=outcome,
        error=error,
    )


def _page_objects(page_dicts: list[dict]) -> list[Page]:
    return [Page(tuple(page["sections"]), page["reprint"]) for page in page_dicts]


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=False)
