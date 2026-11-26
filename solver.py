"""Pagination solver for rehearsal parts.

Model
-----
* The score has ``S`` sections (1..12) in playing order; section ``i`` needs
  ``lines[i-1]`` staff lines and is always printed whole on one page.
* A page holds at most ``capacity`` lines of new sections, taken consecutively
  in score order.  Every page therefore contains a non-empty consecutive run
  of sections, and the runs partition 1..S.
* Every page except the first may reprint the last *new* section of the
  previous page at its top.  The reprint costs that section's line count too.
  This gives that section a second printed copy on the following page.
* The performance is a list of steps (section numbers).  It starts on page 1.
  Between two consecutive steps the player may stay on the current page or --
  only when a rest is marked between those steps -- turn exactly one page
  forward or backward.
* Any printed copy of a section may be used: its main copy, or a reprint at
  the top of the following page.
* Minimise the total page count.  Ties prefer fewer reprints, then
  lexicographically earlier page breaks.
* Locked pages are a fixed prefix of the layout (from an earlier solution) and
  are reproduced verbatim.  Only the sections after the locked prefix are
  paginated again.

With at most 12 sections, all compositions of the remaining sections are
enumerated (at most 2^11).  A composition fixes the page contents.  Feasible
page counts are still screened with the maximal-reprint layout using a
(step, page) reachability bitmask test, because reprints only ever *add* an
alternative copy.  For each minimum-page composition,
:func:`select_reprint_pages` then finds the minimum reprint set in one linear
``O(sections + pages + steps)`` backward/forward pass instead of retrying
reprint subsets for the same pagination: the pointwise-lowest viable route
turns onto exactly the reprints every feasible route needs, so repeated use
of one page-head reprint is collected once.  Locked pages and their reprints
are always reproduced verbatim, even when unused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

MAX_SECTIONS = 12
MAX_STEPS = 48


class InputError(ValueError):
    """Structurally invalid solver input."""


@dataclass(frozen=True)
class Page:
    sections: tuple[int, ...]
    reprint: int | None = None


@dataclass
class Outcome:
    status: str  # "solved" | "unsolvable"
    message: str
    pages: list[dict] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    total_pages: int = 0
    reprints: int = 0
    locked_pages: int = 0
    last_good: dict | None = None

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "message": self.message,
            "pages": self.pages,
            "steps": self.steps,
            "total_pages": self.total_pages,
            "reprints": self.reprints,
            "locked_pages": self.locked_pages,
            "last_good": self.last_good,
        }


# ---------------------------------------------------------------------------
# validation


def validate_inputs(
    lines: Sequence[int], capacity: int, steps: Sequence[int], rests: Sequence[bool]
) -> None:
    if isinstance(capacity, bool) or not isinstance(capacity, int):
        raise InputError("每页行数必须是整数。")
    if not 1 <= capacity <= 60:
        raise InputError("每页行数需在 1 到 60 之间。")
    if not 1 <= len(lines) <= MAX_SECTIONS:
        raise InputError(f"小节数需在 1 到 {MAX_SECTIONS} 之间。")
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in lines):
        raise InputError("每个小节的谱面行数都需是正整数。")
    if any(n > capacity for n in lines):
        raise InputError("有单节行数超过每页容量，单独一节也排不进一页。")
    if not 1 <= len(steps) <= MAX_STEPS:
        raise InputError(f"演奏步数需在 1 到 {MAX_STEPS} 之间。")
    if any(isinstance(s, bool) or not isinstance(s, int) or not 1 <= s <= len(lines) for s in steps):
        raise InputError("演奏顺序里的每个小节编号都必须存在。")
    if len(rests) != len(steps) - 1:
        raise InputError("空拍标记数量应为步数减一。")
    if any(not isinstance(flag, bool) for flag in rests):
        raise InputError("空拍标记必须是布尔值。")


def validate_locked(locked: Sequence[Page], lines: Sequence[int], capacity: int) -> int:
    """Validate a locked page prefix and return the number of covered sections.

    Only structural problems raise InputError. A locked page that no longer
    fits the current line counts is a *solver* failure (the locked pages must
    be kept verbatim, so the whole arrangement is unsolvable); that is reported
    by :func:`solve`, not raised here.
    """
    if not locked:
        return 0
    if locked[0].reprint is not None:
        raise InputError("第 1 页不能重印上一页末节。")
    expected_next = 1
    for index, page in enumerate(locked):
        if not page.sections:
            raise InputError("锁定页不能是空页。")
        if tuple(page.sections) != tuple(range(expected_next, expected_next + len(page.sections))):
            raise InputError("锁定页必须从第 1 小节起按谱面顺序连续覆盖。")
        expected_next += len(page.sections)
        if index > 0 and page.reprint is not None:
            if page.reprint != locked[index - 1].sections[-1]:
                raise InputError("锁定页的重印小节必须是上一页的末节。")
    return expected_next - 1


def locked_overflow(locked: Sequence[Page], lines: Sequence[int], capacity: int) -> int | None:
    """Return the first over-full locked page number, or None."""
    for number, page in enumerate(locked, start=1):
        if page_total(page, lines) > capacity:
            return number
    return None


# ---------------------------------------------------------------------------
# layout helpers


def page_total(page: Page, lines: Sequence[int]) -> int:
    total = sum(lines[section - 1] for section in page.sections)
    if page.reprint is not None:
        total += lines[page.reprint - 1]
    return total


def _page_dicts(
    layout: Sequence[Page], lines: Sequence[int], capacity: int, locked_count: int = 0
) -> list[dict]:
    result = []
    for number, page in enumerate(layout, start=1):
        result.append(
            {
                "number": number,
                "sections": list(page.sections),
                "reprint": page.reprint,
                "reprint_lines": lines[page.reprint - 1] if page.reprint else 0,
                "lines_used": page_total(page, lines),
                "capacity": capacity,
                "locked": number <= locked_count,
            }
        )
    return result


def _copy_masks(layout: Sequence[Page], section_count: int) -> list[int]:
    """section -> bitmask of page numbers (0-based) where it is printed."""
    masks = [0] * (section_count + 1)
    for index, page in enumerate(layout):
        bit = 1 << index
        for section in page.sections:
            masks[section] |= bit
        if page.reprint is not None:
            masks[page.reprint] |= bit
    return masks


def feasible(
    layout: Sequence[Page],
    steps: Sequence[int],
    rests: Sequence[bool],
    section_count: int,
) -> bool:
    if not layout:
        return False
    masks = _copy_masks(layout, section_count)
    reach = masks[steps[0]] & 1  # must start on page 1
    if not reach:
        return False
    full = (1 << len(layout)) - 1
    for index in range(1, len(steps)):
        if rests[index - 1]:
            reach = (reach | (reach << 1) | (reach >> 1)) & full
        reach &= masks[steps[index]]
        if not reach:
            return False
    return True


def perform(
    layout: Sequence[Page],
    steps: Sequence[int],
    rests: Sequence[bool],
    section_count: int,
) -> list[dict] | None:
    """Return one valid per-step page/copy assignment, or None."""
    if not layout or len(rests) != len(steps) - 1:
        return None
    page_count = len(layout)
    masks = _copy_masks(layout, section_count)
    first = masks[steps[0]] & 1
    if not first:
        return None
    full = (1 << page_count) - 1
    levels: list[int] = [first]
    for index in range(1, len(steps)):
        if rests[index - 1]:
            reach = (levels[-1] | (levels[-1] << 1) | (levels[-1] >> 1)) & full
        else:
            reach = levels[-1]
        reach &= masks[steps[index]]
        if not reach:
            return None
        levels.append(reach)

    def pages_in(mask: int) -> Iterator[int]:
        index = 0
        while mask:
            if mask & 1:
                yield index
            mask >>= 1
            index += 1

    chosen = [0] * len(steps)
    page = min(pages_in(levels[-1]))
    for index in range(len(steps) - 1, -1, -1):
        chosen[index] = page
        if index == 0:
            break
        if rests[index - 1]:
            predecessors = {page - 1, page + 1, page}
        else:
            predecessors = {page}
        options = [candidate for candidate in pages_in(levels[index - 1]) if candidate in predecessors]
        if not options:
            return None
        # Prefer staying on the page, then turning forward, then backward.
        page = min(
            options,
            key=lambda candidate: (
                candidate != chosen[index],
                candidate != chosen[index] + 1,
                candidate,
            ),
        )

    result = []
    for index, (section, step_page) in enumerate(zip(steps, chosen)):
        if index == 0:
            move = "从第 1 页开始"
        else:
            delta = step_page - chosen[index - 1]
            if delta == 0:
                move = "留在本页"
            elif delta == 1:
                move = "空拍·向前翻 1 页"
            else:
                move = "空拍·向后翻 1 页"
        kind = "main"
        if section not in layout[step_page].sections:
            kind = "reprint"
        if kind == "main":
            copy = "本页正文"
        else:
            copy = f"页首重印（第 {step_page} 页末节）"
        result.append(
            {
                "number": index + 1,
                "section": section,
                "page": step_page + 1,
                "copy": copy,
                "copy_kind": kind,
                "move": move,
                "rest_before": bool(rests[index - 1]) if index else False,
            }
        )
    return result


# ---------------------------------------------------------------------------
# candidate layouts


def _compositions(sequence: Sequence[int]) -> Iterator[tuple[tuple[int, ...], ...]]:
    """All ways to split ``sequence`` into consecutive non-empty chunks."""
    if not sequence:
        yield ()
        return
    for end in range(1, len(sequence) + 1):
        head = tuple(sequence[:end])
        for tail in _compositions(sequence[end:]):
            yield (head,) + tail


def _suffix_options(
    chunks: Sequence[tuple[int, ...]], lines: Sequence[int], capacity: int
) -> list[tuple[Page, ...]]:
    """All layouts for one chunking; reprints are fixed to the maximal set."""
    pages = []
    for index, chunk in enumerate(chunks):
        reprint = None
        if index > 0:
            tail = chunks[index - 1][-1]
            if page_total(Page(chunk, tail), lines) <= capacity:
                reprint = tail
        pages.append(Page(chunk, reprint))
    return pages


def _solver_candidates(
    lines: Sequence[int],
    capacity: int,
    remaining: Sequence[int],
    locked: Sequence[Page],
) -> Iterator[tuple[tuple[int, ...], list[Page]]]:
    """Yield (break signature, maximal-reprint layout) for every chunking."""
    for chunks in _compositions(remaining):
        if any(sum(lines[section - 1] for section in chunk) > capacity for chunk in chunks):
            continue
        suffix = _suffix_options(chunks, lines, capacity)
        if suffix:
            # The first unlocked page may also reprint the locked tail.
            first = suffix[0]
            if first.reprint is None and locked:
                tail = locked[-1].sections[-1]
                if page_total(Page(first.sections, tail), lines) <= capacity:
                    suffix[0] = Page(first.sections, tail)
        yield tuple(len(chunk) for chunk in chunks[:-1]), list(locked) + suffix


def _apply_reprint_choice(
    layout: Sequence[Page], picked: set[int] | frozenset[int], locked_count: int
) -> list[Page]:
    """Keep only ``picked`` optional reprints; locked reprints are always kept.

    Locked pages (indices below ``locked_count``) are reproduced verbatim, so
    their head reprints stay on the page -- occupying lines and remaining
    playable -- even when the new performance never uses them.
    """
    result = []
    for index, page in enumerate(layout):
        keep = page.reprint is None or index < locked_count or index in picked
        result.append(page if keep else Page(page.sections, None))
    return result


def select_reprint_pages(
    layout: Sequence[Page],
    steps: Sequence[int],
    rests: Sequence[bool],
    section_count: int,
    locked_count: int = 0,
) -> tuple[frozenset[int], int] | None:
    """Find the minimum optional head reprints for fixed page bodies.

    Returns ``(picked, work)`` where ``picked`` is the set of 0-based page
    indices (always at or above ``locked_count``) whose optional head reprint
    has to be present.  Returns ``None`` when even the *maximal* layout --
    every reprint that fits enabled, including locked ones -- cannot carry the
    whole performance.

    Why one linear pass is enough
    -----------------------------
    Every section has one body page ``home[s]`` and at most one extra copy, as
    the head reprint of the immediately following page.  A step on section
    ``s`` can therefore sit on ``home[s]`` or on ``home[s] + 1``; a reprint
    only ever extends the usable page *upward*.  Compute backward-reachable
    page bitmasks and reconstruct, from page 1, the pointwise-lowest viable
    page for each step.  Any optional reprint this lowest path turns onto is
    in fact required by **every** feasible choice (a higher alternative can
    only reach that page through the same reprint), so the collected set is
    the unique minimum.  When the same reprint carries several steps its page
    index is added to a set once -- repeated use counts once.

    Cost is ``O(section_count + len(layout) + len(steps))`` fixed-width page
    mask operations (at most :data:`MAX_SECTIONS` pages, so each mask op is a
    constant-width integer operation).  ``work`` counts the linear scan
    iterations for callers/tests that want to verify the bound.
    """
    page_count = len(layout)
    if page_count == 0 or not steps:
        return None

    home = [-1] * (section_count + 1)
    work = 0
    for index, page in enumerate(layout):
        for section in page.sections:
            home[section] = index
            work += 1
        work += 1

    masks = _copy_masks(layout, section_count)
    work += section_count + page_count
    full = (1 << page_count) - 1

    # backward[k]: bit p set iff, starting the k-th step on page p, the tail
    # k..end can still be played under the maximal copy masks.
    backward = [0] * len(steps)
    backward[-1] = masks[steps[-1]]
    work += 1
    for k in range(len(steps) - 2, -1, -1):
        nxt = backward[k + 1]
        if rests[k]:
            predecessors = (nxt | (nxt << 1) | (nxt >> 1)) & full
        else:
            predecessors = nxt
        backward[k] = predecessors & masks[steps[k]]
        work += 1

    # The first step must start on page 1 (bit 0), and the suffix must work.
    if not (backward[0] & 1):
        return None

    picked: set[int] = set()
    current = 0
    for k in range(1, len(steps)):
        bit = 1 << current
        if rests[k - 1]:
            reachable = (bit | (bit << 1) | (bit >> 1)) & full
        else:
            reachable = bit
        candidates = backward[k] & reachable
        if not candidates:
            # Cannot happen when backward[0] bit 0 is set; kept defensive.
            return None
        lowest = candidates & -candidates
        page = lowest.bit_length() - 1
        section = steps[k]
        # A chosen page different from the body page means the head reprint is
        # used.  Locked reprints are retained regardless, so only collect
        # unlocked page indices.
        if home[section] != page and page >= locked_count:
            picked.add(page)
        current = page
        work += 1

    return frozenset(picked), work


# ---------------------------------------------------------------------------
# public entry point


def solve(
    lines: Sequence[int],
    capacity: int,
    steps: Sequence[int],
    rests: Sequence[bool],
    locked_pages: Sequence[Page] = (),
    previous: dict | None = None,
) -> Outcome:
    """Return the minimum-page layout supporting the performance.

    ``previous`` is an optional snapshot of the last successful solve
    (``{"pages": [Page, ...], "lines": [...], "capacity": int, ...}``).  It is
    only used to explain why the old layout is unusable when no solution
    exists.
    """
    validate_inputs(lines, capacity, steps, rests)
    locked = tuple(locked_pages)
    covered = validate_locked(locked, lines, capacity)
    if covered > len(lines):
        message = (
            "无解：锁定页覆盖的小节比当前填写的小节还多，锁定页无法原样保留。"
            "请把小节补回或减少锁页后重试。"
        )
        outcome = Outcome(status="unsolvable", message=message, locked_pages=len(locked))
        if previous:
            outcome.last_good = _previous_snapshot(previous, lines, capacity, locked, steps, rests)
        return outcome
    overflow = locked_overflow(locked, lines, capacity)
    if overflow is not None:
        message = (
            f"无解：锁定的第 {overflow} 页按当前每页行数已放不下，"
            "而锁定页必须原样保留。请调大每页行数或减少锁页后重试。"
        )
        outcome = Outcome(status="unsolvable", message=message, locked_pages=len(locked))
        if previous:
            outcome.last_good = _previous_snapshot(previous, lines, capacity, locked, steps, rests)
        return outcome
    remaining = list(range(covered + 1, len(lines) + 1))
    locked_reprints = sum(page.reprint is not None for page in locked)

    # Phase 1: minimum page count with every fitting reprint enabled.  Reprints
    # only add alternative copies, so a chunking whose maximal layout is
    # infeasible cannot be rescued by any reprint subset.
    candidates: list[tuple[tuple[int, ...], list[Page]]] = []
    for signature, layout in _solver_candidates(lines, capacity, remaining, locked):
        if feasible(layout, steps, rests, len(lines)):
            candidates.append((signature, layout))
    if not candidates:
        return _unsolvable(lines, capacity, locked, steps, rests, previous)

    best_pages = min(len(layout) for _signature, layout in candidates)
    page_candidates = [
        (signature, layout) for signature, layout in candidates if len(layout) == best_pages
    ]
    # Lexicographic chunk sizes compare breaks left to right: a smaller first
    # chunk means an earlier first page break.
    page_candidates.sort(key=lambda item: item[0])

    # Phase 2: one linear pass per minimum-page pagination finds its unique
    # minimum optional-reprint set -- no subset re-trials for the same layout.
    # Rank by total reprints (locked ones count and stay), then earlier break.
    best_key: tuple[int, tuple[int, ...]] | None = None
    best_layout: list[Page] | None = None
    for signature, maximal in page_candidates:
        choice = select_reprint_pages(
            maximal, steps, rests, len(lines), locked_count=len(locked)
        )
        if choice is None:  # should not happen: phase 1 already ruled this out
            continue
        picked, _work = choice
        reprint_total = locked_reprints + len(picked)
        key = (reprint_total, signature)
        if best_key is None or key < best_key:
            best_key = key
            best_layout = _apply_reprint_choice(maximal, picked, len(locked))
        if len(picked) == 0:
            # No added reprint is the global floor; signatures run earliest
            # break first, so the first such layout is the overall winner.
            break

    if best_key is None or best_layout is None:
        return _unsolvable(lines, capacity, locked, steps, rests, previous)
    assignment = perform(best_layout, steps, rests, len(lines))
    if assignment is None:  # the selector already proved feasibility
        return _unsolvable(lines, capacity, locked, steps, rests, previous)
    best_reprints = best_key[0]

    return Outcome(
        status="solved",
        message="已排好。按下面的页码逐页翻谱即可。",
        pages=_page_dicts(best_layout, lines, capacity, locked_count=len(locked)),
        steps=assignment,
        total_pages=best_pages,
        reprints=best_reprints,
        locked_pages=len(locked),
    )


def _unsolvable(
    lines: Sequence[int],
    capacity: int,
    locked: Sequence[Page],
    steps: Sequence[int],
    rests: Sequence[bool],
    previous: dict | None,
) -> Outcome:
    message = "无解：按当前谱面行数、每页容量、演奏顺序和空拍标记，没有任何排法可行。"
    if locked:
        message += " 锁定的页面会原样保留，不会为迁就新顺序而改动。"
    outcome = Outcome(status="unsolvable", message=message, locked_pages=len(locked))
    if previous:
        outcome.last_good = _previous_snapshot(previous, lines, capacity, locked, steps, rests)
    return outcome


def _previous_snapshot(
    previous: dict,
    lines: Sequence[int],
    capacity: int,
    locked: Sequence[Page],
    steps: Sequence[int],
    rests: Sequence[bool],
) -> dict:
    layout: tuple[Page, ...] = tuple(previous["pages"])
    previous_lines = previous.get("lines") or list(lines)
    previous_capacity = previous.get("capacity") or capacity
    pages = _page_dicts(layout, previous_lines, previous_capacity)
    assignment = perform(layout, steps, rests, len(previous_lines))
    keeps_locked = tuple(layout[: len(locked)]) == tuple(locked)
    snapshot = {
        "pages": pages,
        "steps": assignment or [],
        "previous_steps": previous.get("steps") or [],
        "total_pages": len(layout),
        "reprints": sum(page.reprint is not None for page in layout),
        "keeps_locked": keeps_locked,
        "message": "上次排法不可用。",
    }
    if assignment is None:
        snapshot["message"] = "上次排法在当前演奏顺序下不可用。"
    elif not keeps_locked:
        snapshot["message"] = "上次排法本身仍可演奏，但不符合当前锁页要求，不能沿用。"
    else:
        snapshot["message"] = "上次排法按当前的谱面行数或每页容量已无法原样使用。"
    return snapshot
