"""Tests for the pagination solver and the Flask UI."""

from __future__ import annotations

import itertools

import pytest
from werkzeug.datastructures import MultiDict

from app import app
from solver import (
    InputError,
    Page,
    _compositions,
    feasible,
    page_total,
    perform,
    select_reprint_pages,
    solve,
)


# ---------------------------------------------------------------------------
# brute-force reference (independent of the fast solver)


def brute_force(lines, capacity, steps, rests, locked=()):
    covered = sum(len(page.sections) for page in locked)
    remaining = list(range(covered + 1, len(lines) + 1))
    best = None
    for chunks in _compositions(remaining):
        if any(sum(lines[s - 1] for s in chunk) > capacity for chunk in chunks):
            continue
        previous_tail = locked[-1].sections[-1] if locked else None
        options = []
        for index in range(len(chunks)):
            tail = previous_tail if index == 0 else chunks[index - 1][-1]
            choices = [None]
            if tail is not None and page_total(Page(chunks[index], tail), lines) <= capacity:
                choices.append(tail)
            options.append(choices)
        for picked in itertools.product(*options):
            pages = list(locked) + [Page(chunk, r) for chunk, r in zip(chunks, picked)]
            if any(page_total(page, lines) > capacity for page in pages):
                continue
            if not feasible(pages, steps, rests, len(lines)):
                continue
            key = (
                len(pages),
                sum(page.reprint is not None for page in pages),
                tuple(len(page.sections) for page in pages[:-1]),
            )
            if best is None or key < best[0]:
                best = (key, pages)
    return best


def layout_snapshot(outcome):
    return [(page["sections"], page["reprint"]) for page in outcome.pages]


# ---------------------------------------------------------------------------
# basics


def test_simple_split():
    outcome = solve([8, 8, 8, 8], 16, [1, 2, 3, 4], [False, True, False])
    assert outcome.status == "solved"
    assert outcome.total_pages == 2
    assert layout_snapshot(outcome) == [([1, 2], None), ([3, 4], None)]
    assert [step["page"] for step in outcome.steps] == [1, 1, 2, 2]
    assert outcome.reprints == 0


def test_reprint_is_used_for_repeat_without_rest():
    # Page 1 holds sections 1+2, page 2 holds 3; the last step returns to
    # section 2 with a rest, so page 2's reprint of section 2 is required.
    outcome = solve([2, 2, 1], 4, [1, 2, 3, 2], [False, True, False])
    assert outcome.status == "solved"
    assert outcome.total_pages == 2
    assert outcome.reprints == 1
    assert layout_snapshot(outcome) == [([1, 2], None), ([3], 2)]
    assert outcome.steps[-1]["page"] == 2
    assert outcome.steps[-1]["copy_kind"] == "reprint"


def test_minimum_pages_beats_more_pages():
    outcome = solve([1] * 6, 2, list(range(1, 7)), [True] * 5)
    assert outcome.total_pages == 3


def test_unsolvable_without_rest():
    # Sections 1 and 2 cannot share a page; moving between them needs a rest.
    outcome = solve([5, 5], 5, [1, 2], [False])
    assert outcome.status == "unsolvable"
    assert "无解" in outcome.message


def test_start_must_be_section_one():
    # No explicit start restriction; the first step simply must be on page 1.
    outcome = solve([5, 5], 5, [2, 1], [True])
    assert outcome.status == "unsolvable"


def test_rest_allows_backward_turn_and_cross_page_repeat():
    outcome = solve([4, 4, 4, 4], 8, [1, 2, 3, 4, 3, 2, 1], [False, True, False, True, True, True])
    assert outcome.status == "solved"
    assert outcome.total_pages == 2
    pages = [step["page"] for step in outcome.steps]
    assert pages == [1, 1, 2, 2, 2, 1, 1]


def test_validation_errors():
    with pytest.raises(InputError):
        solve([5], 4, [1], [])
    with pytest.raises(InputError):
        solve([1], 4, [2], [])
    with pytest.raises(InputError):
        solve([1], 4, [1], [True])
    with pytest.raises(InputError):
        solve([1] * 13, 4, [1], [])
    with pytest.raises(InputError):
        solve([1], 4, [1] * 49, [False] * 48)


# ---------------------------------------------------------------------------
# locking


def test_locked_prefix_is_preserved():
    first = solve([4, 4, 4, 4], 8, [1, 2, 3, 4], [False, True, False])
    locked = tuple(Page(tuple(page["sections"]), page["reprint"]) for page in first.pages[:1])
    outcome = solve([4, 4, 4, 4], 8, [1, 2, 1, 2], [False, True, False], locked)
    assert outcome.status == "solved"
    assert outcome.pages[0]["locked"] is True
    assert layout_snapshot(outcome)[0] == ([1, 2], None)


def test_locked_pages_cause_unsolvable_and_previous_is_marked_unusable():
    first = solve([4, 4, 4, 4], 8, [1, 2, 3, 4], [False, True, False])
    previous = {
        "pages": tuple(Page(tuple(page["sections"]), page["reprint"]) for page in first.pages),
        "lines": [4, 4, 4, 4],
        "capacity": 8,
    }
    locked = previous["pages"][:1]
    # Section 3 must follow section 1 with no rest, but page 1 only has 1+2.
    outcome = solve([4, 4, 4, 4], 8, [1, 3, 2], [False, True], locked, previous)
    assert outcome.status == "unsolvable"
    assert outcome.last_good is not None
    assert "上次排法" in outcome.last_good["message"]
    assert outcome.last_good["total_pages"] == 2


def test_locked_page_that_no_longer_fits_is_unsolvable_not_an_error():
    locked = (Page((1, 2), None),)
    outcome = solve([8, 8, 8, 8], 8, [1, 2, 3, 4], [False, True, False], locked)
    assert outcome.status == "unsolvable"
    assert "第 1 页" in outcome.message
    assert "原样保留" in outcome.message


def test_locked_pages_must_be_a_consecutive_prefix():
    with pytest.raises(InputError):
        solve([1, 1, 1], 3, [1], [], (Page((2,), None),))
    with pytest.raises(InputError):
        solve([1, 1, 1], 3, [1], [], (Page((1,), 1),))


def test_locked_reprint_is_kept_even_if_unused():
    # The locked 2-page layout carries a page-2 reprint of section 2.  The new
    # performance never returns to section 2 on page 2, yet the locked reprint
    # must stay on the page: it still occupies lines and counts as a reprint.
    locked = (Page((1, 2), None), Page((3,), 2))
    outcome = solve([2, 2, 1], 4, [1, 2, 3], [False, True], locked)
    assert outcome.status == "solved"
    assert layout_snapshot(outcome) == [([1, 2], None), ([3], 2)]
    assert outcome.reprints == 1
    page2 = outcome.pages[1]
    assert page2["locked"] is True
    assert page2["reprint"] == 2
    assert page2["lines_used"] == 3  # section 3 (1 line) + reprint (2 lines)
    # The retained reprint remains playable when the route needs it.
    used = solve([2, 2, 1], 4, [1, 2, 3, 2, 3], [False, True, False, False], locked)
    assert used.status == "solved"
    assert layout_snapshot(used) == [([1, 2], None), ([3], 2)]
    assert used.steps[3]["page"] == 2
    assert used.steps[3]["copy_kind"] == "reprint"


def test_same_reprint_reused_many_times_counts_once():
    # Page 2's head reprint of section 2 is used at steps 4 and 6 (both stays
    # without a rest), but it is one physical reprint and counts only once.
    outcome = solve(
        [2, 2, 1], 4, [1, 2, 3, 2, 3, 2], [False, True, False, False, False]
    )
    assert outcome.status == "solved"
    assert outcome.total_pages == 2
    assert outcome.reprints == 1
    reprint_uses = [step for step in outcome.steps if step["copy_kind"] == "reprint"]
    assert len(reprint_uses) >= 2
    assert {step["page"] for step in reprint_uses} == {2}


def test_selector_work_is_linear_in_sections_pages_steps():
    # The work counter must be a small constant multiple of S + P + T and, for
    # a successful selection, exactly the two forward/backward linear sweeps.
    for page_count in range(1, 8):
        layout = [
            Page((i + 1,), i if i >= 1 else None) for i in range(page_count)
        ]
        section_count = page_count

        def bounce(total):
            seq, pos, direction = [], 0, 1
            for _ in range(total):
                seq.append(pos + 1)
                if not (0 <= pos + direction < page_count):
                    direction = -direction
                pos += direction
            return seq

        for step_count in (1, 6, 13, 27):
            steps = bounce(step_count)
            rests = [True] * (step_count - 1)
            result = select_reprint_pages(layout, steps, rests, section_count, 0)
            if result is None:
                continue
            _picked, work = result
            assert work == 2 * (section_count + page_count + step_count) - 1
            assert work <= 3 * (section_count + page_count + step_count)


def test_selector_matches_exhaustive_subset_minimum():
    import random

    from solver import _apply_reprint_choice, _solver_candidates

    rng = random.Random(20240517)
    for _ in range(300):
        count = rng.randint(1, 6)
        lines = [rng.randint(1, 4) for _ in range(count)]
        capacity = rng.randint(max(lines), max(lines) + 4)
        steps = [rng.randint(1, count) for _ in range(rng.randint(1, 12))]
        rests = [rng.random() < 0.5 for _ in range(len(steps) - 1)]
        remaining = list(range(1, count + 1))
        for _signature, maximal in _solver_candidates(lines, capacity, remaining, ()):
            choice = select_reprint_pages(maximal, steps, rests, count, 0)
            optional = [
                i for i, page in enumerate(maximal) if page.reprint is not None
            ]
            exhaustive_min = None
            exhaustive_set = None
            for size in range(len(optional) + 1):
                for combo in itertools.combinations(optional, size):
                    trial = _apply_reprint_choice(maximal, set(combo), 0)
                    if feasible(trial, steps, rests, count):
                        exhaustive_min = size
                        exhaustive_set = frozenset(combo)
                        break
                if exhaustive_min is not None:
                    break
            if exhaustive_min is None:
                assert choice is None
                continue
            assert choice is not None
            picked, _work = choice
            assert len(picked) == exhaustive_min
            # The lowest-path set is the unique minimum and equals the first
            # feasible minimum subset found by the exhaustive search.
            assert picked == exhaustive_set
            assert perform(
                _apply_reprint_choice(maximal, picked, 0), steps, rests, count
            )


def test_solve_does_not_enumerate_reprint_subsets(monkeypatch):
    # The new selector is a single linear pass: solve() must not call
    # itertools.combinations while choosing reprints, even on a full-size run.
    def fail_combinations(*_args, **_kwargs):  # pragma: no cover - guard
        raise AssertionError("solve() must not enumerate reprint subsets")

    monkeypatch.setattr(itertools, "combinations", fail_combinations)
    steps = [((i * 7) % 12) + 1 for i in range(48)]
    rests = [i % 2 == 0 for i in range(47)]
    outcome = solve([1] * 12, 3, steps, rests)
    assert outcome.status in {"solved", "unsolvable"}


# ---------------------------------------------------------------------------
# comparison against the brute-force reference


@pytest.mark.parametrize("seed", range(12))
def test_matches_brute_force(seed):
    import random

    rng = random.Random(seed)
    for _ in range(25):
        count = rng.randint(1, 6)
        lines = [rng.randint(1, 4) for _ in range(count)]
        capacity = rng.randint(max(lines), max(lines) + 5)
        steps = [rng.randint(1, count) for _ in range(rng.randint(1, 8))]
        rests = [rng.random() < 0.5 for _ in range(len(steps) - 1)]
        reference = brute_force(lines, capacity, steps, rests)
        outcome = solve(lines, capacity, steps, rests)
        if reference is None:
            assert outcome.status == "unsolvable", (lines, capacity, steps, rests)
        else:
            assert outcome.status == "solved", (lines, capacity, steps, rests)
            assert outcome.total_pages == reference[0][0]
            assert outcome.reprints == reference[0][1]
            assert layout_snapshot(outcome) == [
                (list(page.sections), page.reprint) for page in reference[1]
            ]


@pytest.mark.parametrize("seed", range(6))
def test_locked_matches_brute_force(seed):
    import random

    rng = random.Random(100 + seed)
    for _ in range(15):
        count = rng.randint(2, 6)
        lines = [rng.randint(1, 3) for _ in range(count)]
        capacity = rng.randint(max(lines), max(lines) + 4)
        first = solve(
            lines,
            capacity,
            list(range(1, count + 1)),
            [True] * (count - 1),
        )
        assert first.status == "solved"
        pages = [Page(tuple(page["sections"]), page["reprint"]) for page in first.pages]
        locked = tuple(pages[: rng.randint(1, len(pages))])
        steps = [rng.randint(1, count) for _ in range(rng.randint(1, 7))]
        rests = [rng.random() < 0.5 for _ in range(len(steps) - 1)]
        reference = brute_force(lines, capacity, steps, rests, locked)
        outcome = solve(lines, capacity, steps, rests, locked)
        if reference is None:
            assert outcome.status == "unsolvable", (lines, capacity, steps, rests, locked)
        else:
            assert outcome.status == "solved", (lines, capacity, steps, rests, locked)
            assert outcome.total_pages == reference[0][0]
            assert outcome.reprints == reference[0][1]
            assert layout_snapshot(outcome) == [
                (list(page.sections), page.reprint) for page in reference[1]
            ]


def test_perform_output_is_consistent():
    outcome = solve([2, 2, 1], 4, [1, 2, 3, 2], [False, True, False])
    layout = [Page(tuple(page["sections"]), page["reprint"]) for page in outcome.pages]
    assert perform(layout, [1, 2, 3, 2], [False, True, False], 3) == outcome.steps


# ---------------------------------------------------------------------------
# HTTP


@pytest.fixture()
def client():
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def form_data(lines, capacity, steps, rests, lock_count=0):
    data = [("capacity", str(capacity)), ("lock_count", str(lock_count))]
    for value in lines:
        data.append(("line", str(value)))
    for index, value in enumerate(steps):
        data.append(("step", str(value)))
    for index, rest in enumerate(rests, start=1):
        if rest:
            data.append((f"rest_{index}", "on"))
    return MultiDict(data)


def test_http_index(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "排练分谱".encode() in response.data


def test_http_solve_and_lock_flow(client):
    response = client.post("/solve", data=form_data([4, 4, 4, 4], 8, [1, 2, 3, 4], [False, True, False]))
    assert response.status_code == 200
    assert "共 2 页".encode() in response.data

    # Lock page 1, change the order so the locked page cannot support it.
    response = client.post(
        "/solve",
        data=form_data([4, 4, 4, 4], 8, [1, 3, 2], [False, True], lock_count=1),
    )
    assert response.status_code == 200
    assert "无解".encode() in response.data
    assert "上次排法不可用".encode() in response.data


def test_http_lock_without_previous_errors(client):
    response = client.post("/solve", data=form_data([4, 4], 8, [1, 2], [True], lock_count=1))
    assert response.status_code == 200
    assert "还没有成功排过页".encode() in response.data


def test_http_locked_reprint_stays_on_page(client):
    # First solve produces page 2 with a head reprint of section 2.
    response = client.post(
        "/solve",
        data=form_data([2, 2, 1], 4, [1, 2, 3, 2], [False, True, False]),
    )
    assert response.status_code == 200
    assert "页首重印第 2 小节".encode() in response.data

    # Lock both pages and re-solve with an order that never needs the reprint.
    response = client.post(
        "/solve",
        data=form_data([2, 2, 1], 4, [1, 2, 3], [False, True], lock_count=2),
    )
    assert response.status_code == 200
    body = response.data
    assert "共 2 页".encode() in body
    assert "页首重印第 2 小节".encode() in body  # retained, even though unused
    assert "锁定保留".encode() in body
    assert "上次排法不可用".encode() not in body


def test_http_healthz(client):
    assert client.get("/healthz").json == {"status": "ok"}
