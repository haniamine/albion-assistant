"""Offline tests for laborer-v4: no game, no screen capture, no input.

Synthetic screens are built by pasting a known template into noise, so the
detect-then-classify path can be exercised for correctness rather than just
for "it did not crash". Run with:

    python scripts\\tests\\test_laborer_v4.py
"""

from __future__ import annotations

import json
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laborer_v4.assets import build_library, laborer_profession, laborer_tier  # noqa: E402
from laborer_v4.config import CONFUSABLE_ELEMENTS, MAPPING_PATH, Config  # noqa: E402
from laborer_v4.feedback import BEEP_PATTERNS, Outcome  # noqa: E402
from laborer_v4.identify import LaborerVerdict, Rejection, classify  # noqa: E402
from laborer_v4.inventory import _resolve_collisions, detect_slots, identify_slot  # noqa: E402
from laborer_v4.state import AnchorModel, AppState, JournalSlot  # noqa: E402
from laborer_v4.vision import Frame, Rect, Template, locate_in, pyramid_locate  # noqa: E402

try:
    from laborer_v4 import winput  # noqa: E402
except ImportError as _error:  # pywin32 absent, or not running on Windows
    winput = None
    print(f"! input tests will be skipped: {_error}")

FAILURES: List[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)
        print(f"  FAIL  {message}")
    else:
        print(f"  ok    {message}")


def degrade(
    bgr: np.ndarray,
    rng: np.random.Generator,
    shift: Tuple[float, float] = (0.0, 0.0),
    gain: float = 1.0,
    bias: float = 0.0,
    noise: float = 0.0,
) -> np.ndarray:
    """Approximate what a real capture does to a pristine template."""
    image = bgr.astype(np.float32)
    if shift != (0.0, 0.0):
        matrix = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
        image = cv2.warpAffine(
            image, matrix, (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
        )
    image = image * gain + bias
    if noise:
        image = image + rng.normal(0, noise, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


# Deliberately harsher than a real capture: sub-pixel misalignment, a 7%
# brightness change and heavy sensor-style noise all at once.
HARSH = dict(shift=(0.7, -0.7), gain=0.93, bias=7.0, noise=5.0)


def synthetic_frame(
    template: Template,
    at: Tuple[int, int],
    size: Tuple[int, int] = (600, 400),
    origin: Tuple[int, int] = (0, 0),
    seed: int = 7,
    **degradation,
) -> Frame:
    """A frame of deterministic noise with ``template`` pasted at ``at``."""
    rng = np.random.default_rng(seed)
    bgr = rng.integers(40, 90, (size[1], size[0], 3), dtype=np.uint8)
    patch = degrade(template.bgr, rng, **degradation) if degradation else template.bgr
    x, y = at
    bgr[y:y + template.height, x:x + template.width] = patch
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return Frame(rect=Rect(origin[0], origin[1], size[0], size[1]), gray=gray, bgr=bgr)


# ---------------------------------------------------------------------------


def test_geometry() -> None:
    print("\n[geometry]")
    rect = Rect(10, 20, 100, 50)
    check(rect.right == 110 and rect.bottom == 70, "Rect right/bottom")
    check(rect.center == (60, 45), "Rect center")
    check(rect.inflate(5) == Rect(5, 15, 110, 60), "Rect inflate")

    bounds = Rect(0, 0, 80, 80)
    clamped = rect.clamp_to(bounds)
    check(clamped == Rect(10, 20, 70, 50), f"Rect clamp inside bounds (got {clamped})")
    check(Rect(200, 200, 10, 10).clamp_to(bounds) is None, "Rect clamp fully outside returns None")

    merged = Rect(0, 0, 10, 10).union(Rect(20, 30, 10, 10))
    check(merged == Rect(0, 0, 30, 40), f"Rect union (got {merged})")


def test_template_loading(library) -> None:
    print("\n[template loading]")
    opaque = [name for name, t in library.laborers.items() if t.mask is None]
    check(
        len(opaque) == len(library.laborers),
        f"fully opaque alphas are dropped ({len(opaque)}/{len(library.laborers)})",
    )
    check(all(t.bgr is not None for t in library.laborers.values()), "laborer templates keep BGR for tier comparison")
    check(len(library.tier_weights) == len(library.laborers), "a tier weight map exists for every laborer")
    check(
        all(float(w.sum()) > 0 for w in library.tier_weights.values()),
        "no tier weight map is degenerate",
    )
    # The tier signal must be sparse: if it were spread over the whole icon the
    # whole-icon score would already separate siblings, and it demonstrably does not.
    coverage = [float((w > 1).mean()) for w in library.tier_weights.values()]
    check(
        max(coverage) < 0.30,
        f"tier weights concentrate on the badge, not the whole icon (max coverage {max(coverage):.1%})",
    )

    # An all-255 mask must not change the score: this is why dropping it is safe.
    sample = next(iter(library.laborers.values()))
    src = synthetic_frame(sample, (100, 80)).gray
    full_mask = np.full(sample.gray.shape, 255, np.uint8)
    a = cv2.matchTemplate(src, sample.gray, cv2.TM_SQDIFF_NORMED, mask=full_mask)
    b = cv2.matchTemplate(src, sample.gray, cv2.TM_SQDIFF_NORMED)
    a = np.nan_to_num(a, nan=1.0, posinf=1.0, neginf=1.0)
    check(float(np.abs(a - b).max()) < 1e-4, "all-255 mask is numerically identical to no mask")


def test_stop_laborers(library, config) -> None:
    print("\n[stop laborers]")
    check(library.is_stop("t6-bs"), "t6-bs is a stop laborer")
    check(not library.is_stop("t5-bs"), "t5-bs is actionable")
    check(
        library.journal_for("t6-bs") is None,
        "stop laborers carry no journal mapping (mapping.json's placeholder is ignored)",
    )
    check(
        "settings" not in library.scannable_journals(),
        "the placeholder 'settings' journal is not scanned for",
    )
    check(laborer_tier("t5-imb") == 5 and laborer_profession("t5-imb") == "imb", "tier/profession parsing")


def test_candidate_reduction(library) -> None:
    print("\n[candidate reduction]")
    active, excluded = library.active_laborers(["t5-bs", "t2-common"])
    check("t5-bs" in active, "t5-bs is active when its journal is present")
    check("t2-bs" in active and "t2-tk" in active, "every t2 profession is active via the shared t2-common journal")
    check("t5-ft" not in active, "t5-ft is excluded when no t5-ft journal is held")
    check("t5-ft" in excluded and "t5-ft" in excluded["t5-ft"] or "no 't5-ft'" in excluded.get("t5-ft", ""),
          f"exclusion carries a reason (got {excluded.get('t5-ft')!r})")
    check(all(library.is_stop(n) or n in active for n in library.stop_laborers),
          "stop laborers stay active with no journal at all")

    empty_active, _ = library.active_laborers([])
    check(
        set(empty_active) == set(library.stop_laborers),
        f"an empty inventory leaves only stop laborers ({sorted(empty_active)})",
    )


def test_classification(library, config) -> None:
    print("\n[detect-then-classify]")
    everything = library.laborers

    for label, degradation in (("clean", {}), ("harsh capture", HARSH)):
        wrong = []
        for seed, name in enumerate(sorted(everything)):
            frame = synthetic_frame(everything[name], (120, 90), origin=(300, 200), seed=seed, **degradation)
            result = classify(frame, None, everything, library.tier_weights, config)
            if not isinstance(result, LaborerVerdict) or result.name != name:
                got = getattr(result, "name", None) or result.describe()
                wrong.append(f"{name} -> {got}")
        check(not wrong, f"every laborer classifies as itself, {label} ({len(everything)} icons) {wrong or ''}")

    # Position must be reported in screen coordinates, not frame-local ones.
    frame = synthetic_frame(everything["t5-bs"], (120, 90), origin=(300, 200))
    result = classify(frame, None, everything, library.tier_weights, config)
    check(
        isinstance(result, LaborerVerdict) and result.top_left == (420, 290),
        f"verdict top-left is in screen coordinates (got {getattr(result, 'top_left', None)})",
    )

    # Nothing present at all must be rejected, never guessed.
    rng = np.random.default_rng(3)
    noise = rng.integers(0, 255, (400, 600, 3), dtype=np.uint8)
    empty = Frame(rect=Rect(0, 0, 600, 400), gray=cv2.cvtColor(noise, cv2.COLOR_BGR2GRAY), bgr=noise)
    result = classify(empty, None, everything, library.tier_weights, config)
    check(isinstance(result, Rejection), f"pure noise is rejected (got {result})")

    # Grayscale-only capture cannot decide a tier and must say so.
    gray_only = Frame(rect=frame.rect, gray=frame.gray, bgr=None)
    result = classify(gray_only, None, everything, library.tier_weights, config)
    check(
        isinstance(result, Rejection) and result.reason == "no_color_capture",
        f"a grayscale capture refuses to decide a tier (got {result})",
    )


def test_tier_margins(library, config) -> None:
    print("\n[tier separation - the failure v3 kept patching]")
    everything = library.laborers
    from laborer_v4.identify import score_candidates

    # The signal v3 relied on: whole-icon score. Show that it is not usable.
    score_gaps = []
    tier_ratios = []
    for seed, name in enumerate(sorted(everything)):
        frame = synthetic_frame(everything[name], (120, 90), seed=seed, **HARSH)
        candidates = score_candidates(frame, None, everything)
        siblings = [c for c in candidates if laborer_profession(c.name) == laborer_profession(name)]
        if len(siblings) > 1:
            score_gaps.append(siblings[0].score - siblings[1].score)
        result = classify(frame, None, everything, library.tier_weights, config)
        if isinstance(result, LaborerVerdict) and result.tier_ratio:
            tier_ratios.append(result.tier_ratio)

    print(f"        whole-icon score gap between siblings: min {min(score_gaps):.4f}")
    print(f"        badge-weighted separation ratio      : min {min(tier_ratios):.2f}x")
    check(
        min(score_gaps) < 0.05,
        f"whole-icon score really is too close to decide a tier (min gap {min(score_gaps):.4f})",
    )
    check(
        min(tier_ratios) >= config.match.tier_ratio_min,
        f"badge-weighted comparison clears the configured margin "
        f"(min {min(tier_ratios):.2f}x >= {config.match.tier_ratio_min}x)",
    )


def test_unknown_tier_is_rejected(library, config) -> None:
    print("\n[unknown tier safety]")
    # t6-imb and t6-tk have no asset. Simulate one by transplanting a t6 badge
    # onto a t5-imb icon: the classifier must refuse rather than hand over a
    # t5 journal. v3 scored this at 0.986 against t4-imb - above its 0.97 gate.
    everything = library.laborers
    if "t5-imb" not in everything or "t6-bs" not in everything:
        print("  skip  (assets unavailable)")
        return

    faked = everything["t5-imb"].bgr.copy()
    faked[0:34, 0:34] = everything["t6-bs"].bgr[0:34, 0:34]
    fake_template = Template(
        name="t6-imb-simulated",
        path=everything["t5-imb"].path,
        gray=cv2.cvtColor(faked, cv2.COLOR_BGR2GRAY),
        mask=None,
        gray_half=everything["t5-imb"].gray_half,
        mask_half=None,
        bgr=faked,
        has_alpha=False,
        align_offset=everything["t5-imb"].align_offset,
    )
    frame = synthetic_frame(fake_template, (120, 90))
    result = classify(frame, None, everything, library.tier_weights, config)
    check(
        isinstance(result, Rejection) and result.reason == "unknown_tier",
        f"an unknown tier is refused, not read as its t5 sibling (got {getattr(result, 'name', result)})",
    )

    # And the whole-icon score alone would have accepted it - the exact hole in v3.
    from laborer_v4.identify import score_candidates
    best = score_candidates(frame, None, everything)[0]
    check(
        best.score > 0.97,
        f"the same icon scores {best.score:.3f} on {best.name}, which v3's 0.97 gate would have passed",
    )


def test_missing_journal_is_not_a_misread(library, config) -> None:
    print("\n[identify first, then check the bag]")
    # A ready t5-bs while only t2 journals are held. Narrowing the comparison
    # set to the inventory - v3's approach - leaves t2-bs as the best available
    # match at 0.996, comfortably past a 0.97 gate, so the t5 laborer is handed
    # a t2 journal. Classifying against every tier identifies it correctly, so
    # the caller can refuse for the right reason.
    everything = library.laborers
    frame = synthetic_frame(everything["t5-bs"], (120, 90))

    from laborer_v4.identify import score_candidates

    reduced, _ = library.active_laborers(["t2-common"])
    check("t5-bs" not in reduced, "t5-bs is not a candidate when only t2 journals are held")
    best_reduced = score_candidates(frame, None, reduced)[0]
    check(
        best_reduced.name != "t5-bs" and best_reduced.score > 0.97,
        f"a reduced candidate set misreads t5-bs as {best_reduced.name} "
        f"at {best_reduced.score:.3f}, past a 0.97 gate (this is v3's bug)",
    )

    result = classify(frame, None, everything, library.tier_weights, config)
    check(
        isinstance(result, LaborerVerdict) and result.name == "t5-bs",
        f"the full candidate set identifies it correctly as t5-bs (got {getattr(result, 'name', result)})",
    )
    check(
        library.journal_for("t5-bs") == "t5-bs",
        "and it maps to its own journal, so the caller reports a supply problem, not a misread",
    )


def test_anchors() -> None:
    print("\n[anchor model]")
    model = AnchorModel()
    check(model.predict_top_left("ready") is None, "an empty model predicts nothing")

    model.observe({"ready": (1000, 500), "take_all": (1040, 640), "laborer": (1010, 520)})
    check(model.origin == (1000, 500), "the reference sets the origin")
    check(model.offsets["take_all"] == (40, 140), "offsets are relative to the reference")
    check(model.predict_top_left("take_all") == (1040, 640), "prediction round-trips")

    # The dialog moves: a later observation must move every prediction with it.
    model.observe({"ready": (1200, 500)})
    check(model.predict_top_left("take_all") == (1240, 640), "the whole dialog follows the reference")

    # The reference was not seen, but a known element pins the origin anyway.
    model.observe({"take_all": (1340, 640)})
    check(model.origin == (1300, 500), f"origin is back-derived from a known element (got {model.origin})")

    rect = model.element_rect("take_all", (124, 52), padding=10)
    check(rect == Rect(1330, 630, 144, 72), f"element_rect applies size and padding (got {rect})")

    dialog = model.dialog_rect({"ready": (100, 133), "take_all": (124, 52), "laborer": (95, 95)}, padding=10)
    check(
        dialog is not None and dialog.contains_point((1340, 640)) and dialog.contains_point((1300, 500)),
        "dialog_rect covers every known element",
    )

    model.clear()
    check(model.origin is None and not model.offsets, "clear() drops everything")


def test_coincident_anchors_are_kept() -> None:
    """take-all, advance-tier and accept share one button slot in the dialog.

    Observed live: take-all at (334,952) and accept at (329,948) - 5 px apart,
    both genuine. An earlier fix read that coincidence as a corrupted anchor and
    deleted one, throwing away a good position. Which button is in the slot is a
    question about pixels, not coordinates.
    """
    print("\n[coincident anchors]")
    model = AnchorModel()
    model.observe({"ready": (24, 201), "take_all": (272, 926), "accept": (268, 922)})
    check(model.offsets["take_all"] == (248, 725), "take-all offset learned")
    check(model.offsets["accept"] == (244, 721), "accept offset learned at nearly the same place")
    check(model.predict_top_left("accept") == (268, 922), "and both survive to be predicted")
    check(not hasattr(model, "sanitize"), "the anchor-deleting heuristic is gone")


def test_advance_tier_not_confused_with_take_all(library, config) -> None:
    """The regression itself: a screen showing take-all must not yield advance-tier.

    Logged symptom was 'clicked take-all at (334,952)' immediately followed by
    'clicked advance-tier at (345,952)' - the same button, 11 px apart, which is
    exactly half the width difference between the two templates.
    """
    print("\n[advance-tier vs take-all]")
    take_all = library.menu["take_all"]
    advance = library.menu["advance_tier"]

    raw = cv2.matchTemplate(
        cv2.copyMakeBorder(take_all.gray, 8, 8, 16, 16, cv2.BORDER_REPLICATE),
        advance.gray, cv2.TM_SQDIFF_NORMED,
    )
    cross = 1.0 - float(raw.min())
    # 0.80 was the shared menu floor when this shipped; the score clears it on a
    # live capture, which is what made the phantom click possible.
    check(cross > 0.78,
          f"advance-tier really does score ~{cross:.3f} on a take-all button (this was the bug)")
    check(cross < config.match.advance_tier_score,
          f"the dedicated floor {config.match.advance_tier_score} rejects it ({cross:.3f})")
    check(cross < config.match.menu_score,
          f"the raised shared floor {config.match.menu_score} rejects it too ({cross:.3f})")

    # Now the end-to-end shape: a frame showing only take-all.
    for label, degradation in (("clean", {}), ("degraded", HARSH)):
        frame = synthetic_frame(take_all, at=(300, 900), size=(800, 1000), **degradation)

        phantom = locate_in(frame, advance, None, config.match.menu_score)
        strict = locate_in(frame, advance, None, config.match.advance_tier_score)
        real = locate_in(frame, take_all, None, config.match.menu_score)

        check(real is not None, f"[{label}] take-all is found on its own button")
        check(strict is None,
              f"[{label}] advance-tier is NOT found at the strict floor "
              f"(scored {phantom.score:.3f})" if phantom is not None else
              f"[{label}] advance-tier is NOT found at the strict floor")

        # And the score-comparison gate, which catches it even if the floor is lowered.
        if phantom is not None:
            rival = locate_in(frame, take_all, None, 0.0)
            check(rival is not None and rival.score >= phantom.score,
                  f"[{label}] take-all fits the spot better ({rival.score:.3f} >= {phantom.score:.3f}), "
                  "so the confusion gate drops it")


def test_dialog_buttons_rank_against_each_other(library, config) -> None:
    """take-all / advance-tier / accept are one button art with three labels.

    Reported: 'clicked take-all at (205,129) (0.810)' then 'clicked accept at
    (201,129) (0.804)' - the same corner of the screen, neither of them a
    button. Meanwhile a real match in the same run scored 1.000 and 0.999.
    Two separate defences have to hold: the floor rejects the corner, and
    ranking rejects one button wearing another's identity.
    """
    print("\n[dialog button ranking]")
    check(config.match.menu_score > 0.81,
          f"the menu floor {config.match.menu_score} rejects the 0.810 corner phantom")

    for name, rivals in CONFUSABLE_ELEMENTS.items():
        for rival_name in rivals:
            subject = library.menu[name]
            rival = library.menu[rival_name]
            pad = cv2.copyMakeBorder(rival.gray, 12, 12, 16, 16, cv2.BORDER_REPLICATE)
            if subject.height > pad.shape[0] or subject.width > pad.shape[1]:
                continue
            impostor = 1.0 - float(cv2.matchTemplate(pad, subject.gray, cv2.TM_SQDIFF_NORMED).min())
            genuine = 1.0 - float(cv2.matchTemplate(pad, rival.gray, cv2.TM_SQDIFF_NORMED).min())
            check(genuine > impostor,
                  f"on a real {rival_name} button, {rival_name} ({genuine:.4f}) outranks "
                  f"{name} ({impostor:.4f}) -> '{name}' is dropped")

    # The pair that actually mattered: accept clears the floor on a take-all
    # button, so the floor alone would not have saved it.
    accept, take_all = library.menu["accept"], library.menu["take_all"]
    pad = cv2.copyMakeBorder(take_all.gray, 12, 12, 16, 16, cv2.BORDER_REPLICATE)
    impostor = 1.0 - float(cv2.matchTemplate(pad, accept.gray, cv2.TM_SQDIFF_NORMED).min())
    check(impostor > config.match.menu_score,
          f"accept scores {impostor:.4f} on a take-all button - above the "
          f"{config.match.menu_score} floor, so ranking is what stops it")


def test_config_migration_reaches_an_existing_file(tmp_path: Path) -> None:
    """A changed default must survive an existing config file.

    The file is applied over the defaults, so the file wins - raising
    menu_score in code did nothing at all for anyone who had ever run v4.
    """
    print("\n[config migration]")
    from laborer_v4.config import CONFIG_VERSION, load_config, save_config

    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps({"match": {"menu_score": 0.80}}), encoding="utf-8")
    config, warnings = load_config(stale)
    check(config.match.menu_score == 0.90,
          f"an untouched old default is moved forward (got {config.match.menu_score})")
    check(config.version == CONFIG_VERSION, "the file is stamped with the new version")
    check(json.loads(stale.read_text())["match"]["menu_score"] == 0.90, "and written back to disk")

    # A deliberately tuned value must NOT be overwritten.
    tuned = tmp_path / "tuned.json"
    tuned.write_text(json.dumps({"match": {"menu_score": 0.97}}), encoding="utf-8")
    config, warnings = load_config(tuned)
    check(config.match.menu_score == 0.97, "a value you set yourself is left alone")
    check(any("left as you set it" in w for w in warnings), "and you are told the default moved")

    # Re-loading an already-migrated file must be a no-op.
    config, warnings = load_config(stale)
    check(not any("->" in w for w in warnings), "migrating twice does nothing")


def test_the_game_window_is_picked_over_windows_that_merely_mention_it(config) -> None:
    """The reported bug: 'v4 detects no journal in the inventory'.

    Nothing was wrong with the inventory scan. The window picker took the
    largest window whose title contained 'Albion', and on this desk that is the
    market tool - 42k px larger than the client, and on the *other* monitor. The
    search region was therefore built from the wrong screen, so every lookup
    failed and the scan reported an empty bag.

    These are the four windows as enumerated live, areas and monitors included.
    """
    print("\n[game window selection]")
    from laborer_v4.winput import GameWindow, select_game_window

    needles = [n.lower() for n in config.window.title_contains]
    processes = [p.lower() for p in config.window.process_contains]

    tool = GameWindow(1, "AlbionOnline - StatisticsAnalysisTool",
                      (-1927, -7, 7, 1087), "statisticsanalysistool.exe")
    game = GameWindow(2, "Albion Online Client", (0, 0, 1920, 1080), "albion-online.exe")
    sheet = GameWindow(3, "Albion gestion - Google Sheets - Google Chrome",
                       (-1928, -8, 8, 1048), "chrome.exe")
    editor = GameWindow(4, "laborer-v4.py - albion-assistant - Visual Studio Code",
                        (-1928, -8, 8, 1048), "code.exe")

    check(tool.area > game.area,
          f"the market tool's window really is the largest ({tool.area} > {game.area})")

    picked = select_game_window([tool, game, sheet, editor], needles, processes)
    check(picked is not None and picked.hwnd == game.hwnd,
          f"the client is picked, not the largest match (got {picked.title if picked else None!r})")

    # Order of enumeration must not decide it.
    picked = select_game_window([editor, sheet, tool, game], needles, processes)
    check(picked is not None and picked.hwnd == game.hwnd, "and enumeration order does not matter")

    # An elevated client refuses a process handle: the name comes back "" and
    # must read as *unknown*, not as *not the game*, or the picker would go
    # blind exactly when the game runs as administrator.
    elevated = GameWindow(5, "Albion Online Client", (0, 0, 1920, 1080), "")
    picked = select_game_window([tool, sheet, editor, elevated], needles, processes)
    check(picked is not None and picked.hwnd == elevated.hwnd,
          "a client whose process cannot be read is still found by title")

    # ...but an unreadable process alone is not enough; the title still has to fit.
    unknown = GameWindow(6, "Some Other Fullscreen Thing", (0, 0, 1920, 1080), "")
    picked = select_game_window([tool, sheet, editor, unknown], needles, processes)
    check(picked is None, "an unrelated window is never adopted as the game")

    # The game not running must be reported as absent rather than substituted.
    picked = select_game_window([tool, sheet, editor], needles, processes)
    check(picked is None, "with the game closed, no window is returned at all")

    # Two windows of the real process (launcher + client): the exact title wins
    # over the bigger one.
    launcher = GameWindow(7, "Albion Online Launcher", (0, 0, 1920, 1080), "albion-online.exe")
    smaller_client = GameWindow(8, "Albion Online Client", (100, 100, 1700, 1000), "albion-online.exe")
    check(launcher.area > smaller_client.area, "the launcher window is the larger of the two")
    picked = select_game_window([launcher, smaller_client], needles, processes)
    check(picked is not None and picked.hwnd == smaller_client.hwnd,
          "the exact title beats the larger window within the same process")


def test_a_new_high_tier_laborer_keeps_its_journal(library, config) -> None:
    """The reported bug: 'the new journals and laborer I added are not scanned'.

    t8-ft/t8-imb were added with templates and mapping.json lines pointing at
    the new t6-ft/t6-imb journals. Nothing scanned for them.

    laborer.max_action_tier is 5, so tier 8 tripped the ceiling and both were
    filed as *stop* laborers - and a stop laborer's mapping is deliberately
    discarded as a placeholder, which threw the journals away with it. The
    ceiling exists to stop an icon whose tier we have no template for being
    read as its lower sibling; it was never meant to overrule an explicit
    mapping to a journal that is sitting on disk.
    """
    print("\n[a new tier above the ceiling]")
    from laborer_v4.assets import is_stop_laborer

    prefixes, ceiling = config.laborer.stop_prefixes, config.laborer.max_action_tier

    check(is_stop_laborer("t6-bs", True, prefixes, ceiling),
          "an explicit stop prefix wins even when a journal exists")
    check(is_stop_laborer("t8-ft", False, prefixes, ceiling),
          "above the ceiling with no journal is still a stop laborer (the guard works)")
    check(not is_stop_laborer("t8-ft", True, prefixes, ceiling),
          "above the ceiling *with* a journal is actionable - the mapping is the intent")
    check(not is_stop_laborer("t5-ft", True, prefixes, ceiling), "below the ceiling is untouched")
    check(not is_stop_laborer("t5-ft", False, prefixes, ceiling),
          "and stays untouched when its journal is missing")

    # The same thing through the real assets, which is where it actually broke.
    for laborer in ("t8-ft", "t8-imb"):
        if laborer not in library.laborers:
            continue
        journal = library.journal_for(laborer)
        check(not library.is_stop(laborer), f"{laborer} is actionable, not a stop laborer")
        check(journal is not None, f"{laborer} keeps its journal mapping (got {journal})")
        check(journal in library.scannable_journals(),
              f"and '{journal}' is actually scanned for")

    # An asset that no mapping points at is invisible however good the capture
    # is, so it has to be said out loud rather than left to be discovered.
    orphans = [w for w in library.warnings if "never scanned for" in w]
    check(bool(orphans) == bool(
        set(p.stem for p in (Path(__file__).resolve().parents[2] / "assets" / "journals").iterdir()
            if p.suffix.lower() == ".png") - set(library.journals_by_name)),
        "unmapped journal assets are reported, and only when there are some")

    promoted = [w for w in library.warnings if "max_action_tier" in w]
    check(bool(promoted) == any(
        laborer_tier(name) is not None and laborer_tier(name) > ceiling
        for name in library.laborer_to_journal),
        "stepping over the ceiling is reported rather than done silently")


def test_assets_load_whatever_the_extension_case(library) -> None:
    """t8-ft.PNG must load exactly like t8-ft.png.

    pathlib's glob is case-insensitive on Windows and case-sensitive
    everywhere else, so a hand-captured .PNG works here and silently vanishes
    on a Linux checkout - the same invisible-asset failure, one platform over.
    """
    print("\n[asset extension case]")
    from laborer_v4.assets import asset_path, asset_paths
    from laborer_v4.config import JOURNAL_ASSETS_DIR, LABORER_ASSETS_DIR

    for directory, label in ((LABORER_ASSETS_DIR, "laborer"), (JOURNAL_ASSETS_DIR, "journal")):
        on_disk = {p.stem for p in directory.iterdir()
                   if p.is_file() and p.suffix.lower() == ".png"}
        found = {p.stem for p in asset_paths(directory)}
        check(found == on_disk,
              f"every {label} asset is listed regardless of case "
              f"(missed: {sorted(on_disk - found) or 'none'})")

    upper = [p for p in LABORER_ASSETS_DIR.iterdir() if p.suffix == ".PNG"]
    for path in upper:
        check(path.stem in library.laborers, f"{path.name} loaded as '{path.stem}'")
        check(asset_path(LABORER_ASSETS_DIR, path.stem) is not None,
              f"and is resolvable by name without its extension's case")


def test_beeps_only_on_attention(library) -> None:
    print("\n[beep patterns]")
    silent = {Outcome.SUCCESS, Outcome.NOT_READY, Outcome.PICKED_UP, Outcome.BLOCKED, Outcome.ABORTED}
    for outcome in silent:
        check(outcome not in BEEP_PATTERNS, f"{outcome.value} is silent")
    for outcome in (Outcome.STOP_LABORER, Outcome.ERROR, Outcome.JOURNAL_FAILED,
                    Outcome.ACCEPT_FAILED, Outcome.PICKUP_FAILED, Outcome.UNKNOWN_LABORER,
                    Outcome.NO_JOURNAL):
        check(outcome in BEEP_PATTERNS, f"{outcome.value} still beeps")
    check(Outcome.PICKUP_FAILED.is_failure, "a failed pick-up counts toward the failure breaker")
    check(not Outcome.PICKED_UP.is_failure, "a successful pick-up does not")


def test_pickup_assets(library) -> None:
    print("\n[pick-up assets]")
    # pick-up.PNG and yes.PNG are stored with upper-case extensions; the loader
    # must find them regardless.
    for element in ("settings", "pick_up", "yes"):
        check(element in library.menu, f"the '{element}' template is loaded")
    check(library.menu["pick_up"].size == (122, 32), f"pick-up size {library.menu['pick_up'].size}")
    check("settings" not in library.scannable_journals(),
          "'settings' is a stop-laborer placeholder, never scanned for as a journal")


def test_journal_scan_finds_every_stack(library, config) -> None:
    """The reported bug: 't2-common is not in the inventory' when it plainly was.

    Journals are the same book in slightly different colours. In grayscale every
    template matches every other journal's slot at 0.87-0.97 against a 1.000
    self-match, so the old per-template argmax regularly landed on a *different*
    journal's slot and the collision resolver then deleted the loser.

    This lays out the exact seven stacks the reported state file held and
    requires every one of them back, at the right place, with the right name.
    """
    print("\n[journal scan]")
    journals = library.scannable_journals()
    held = ["t2-common", "t3-common", "t4-bs", "t4-ft", "t5-bs", "t5-ft", "t5-tk"]

    # Grid mirroring a real inventory: ~81 px pitch, two rows.
    placed = {name: (1560 + 82 * (i % 4), 530 + 81 * (i // 4)) for i, name in enumerate(held)}

    rng = np.random.default_rng(19)
    width, height = 960, 1080
    bgr = rng.integers(38, 74, (height, width, 3), dtype=np.uint8)
    region = Rect(960, 0, width, height)

    for name, center in placed.items():
        template = journals[name]
        patch = degrade(template.bgr, rng, **HARSH)
        x = center[0] - region.left - template.width // 2
        y = center[1] - region.top - template.height // 2
        bgr[y:y + template.height, x:x + template.width] = patch

    frame = Frame(rect=region, gray=cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), bgr=bgr)

    centers = detect_slots(frame, journals, config)
    check(len(centers) >= len(held),
          f"every stack is detected as a slot (found {len(centers)}, placed {len(held)})")

    named = {}
    for center in centers:
        verdict = identify_slot(frame, center, journals, config)
        if verdict is not None and verdict.score >= config.match.journal_score \
                and verdict.margin >= config.match.journal_margin:
            named[verdict.name] = (center, verdict)

    for name in held:
        if name not in named:
            check(False, f"{name} was found (it is in the inventory)")
            continue
        center, verdict = named[name]
        want = placed[name]
        drift = max(abs(center[0] - want[0]), abs(center[1] - want[1]))
        check(drift <= config.match.journal_min_separation,
              f"{name:<10} @ {center} vs placed {want} (drift {drift}px), "
              f"margin over {verdict.runner_up} = {verdict.margin:.4f}")

    check(not (set(held) - set(named)),
          f"nothing in the bag went missing (lost: {sorted(set(held) - set(named)) or 'none'})")

    # The specific pair that broke it: t3-common must not be reported at
    # t2-common's slot, in either direction.
    if "t2-common" in named and "t3-common" in named:
        check(named["t2-common"][0] != named["t3-common"][0],
              "t2-common and t3-common resolve to different slots")


def test_journal_verify_catches_a_reflow(library, config) -> None:
    """journal_at must reject a slot that now holds a lookalike.

    It used to score one template in grayscale against a 0.90 floor. A different
    journal in the slot scores up to 0.97 that way, so the guard passed on
    exactly the reflow it existed to catch - and the shift-click went out on the
    wrong item.
    """
    print("\n[journal slot verification]")
    journals = library.scannable_journals()
    rng = np.random.default_rng(23)

    for occupant, expected in (("t2-common", "t2-common"), ("t3-common", "t2-common")):
        template = journals[occupant]
        region = Rect(1700, 580, 200, 200)
        bgr = rng.integers(38, 74, (region.height, region.width, 3), dtype=np.uint8)
        center = (1800, 680)
        x = center[0] - region.left - template.width // 2
        y = center[1] - region.top - template.height // 2
        bgr[y:y + template.height, x:x + template.width] = degrade(template.bgr, rng, **HARSH)
        frame = Frame(rect=region, gray=cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), bgr=bgr)

        verdict = identify_slot(frame, center, journals, config)
        correct = verdict is not None and verdict.name == occupant
        check(correct, f"a slot holding {occupant} identifies as {verdict.name if verdict else None}")
        if occupant != expected:
            check(verdict is not None and verdict.name != expected,
                  f"looking for {expected} in a slot holding {occupant} is refused")

    # An obscured slot must not be reported as a reflow. Observed live: a check
    # landing while the panel redrew scored 0.573 and was announced as "now
    # holds t2-common", which sent the run off to rescan an inventory that had
    # not changed. The two failures need different words because they need
    # different responses - wait vs rescan.
    region = Rect(1700, 580, 200, 200)
    noise = rng.integers(38, 74, (region.height, region.width, 3), dtype=np.uint8)
    frame = Frame(rect=region, gray=cv2.cvtColor(noise, cv2.COLOR_BGR2GRAY), bgr=noise)
    verdict = identify_slot(frame, (1800, 680), journals, config)
    got = f"{verdict.score:.3f}" if verdict else "no verdict"
    check(verdict is None or verdict.score < config.match.journal_score,
          f"an obscured/redrawing slot lands below the {config.match.journal_score} floor (got {got})")


def test_journal_collisions() -> None:
    print("\n[journal scan collisions]")
    found = {
        "t4-bs": (JournalSlot((1000, 500), 0.97, 2), (0, 0)),
        "t4-ft": (JournalSlot((1004, 502), 0.93, 2), (0, 0)),  # same slot, weaker
        "t5-bs": (JournalSlot((1100, 500), 0.95, 2), (0, 0)),
    }
    kept, collisions = _resolve_collisions(found, min_separation=24)
    check(set(kept) == {"t4-bs", "t5-bs"}, f"the weaker of two colliding journals is dropped (kept {sorted(kept)})")
    check(len(collisions) == 1 and "t4-ft" in collisions[0], "the collision is reported")

    kept, collisions = _resolve_collisions(
        {"a": (JournalSlot((0, 0), 0.9, 1), (0, 0)), "b": (JournalSlot((100, 100), 0.9, 1), (0, 0))},
        min_separation=24,
    )
    check(len(kept) == 2 and not collisions, "well-separated journals both survive")


def test_state_roundtrip(tmp_path: Path) -> None:
    print("\n[state persistence]")
    state = AppState()
    state.anchors.observe({"ready": (10, 20), "accept": (30, 60)})
    state.journals.replace({"t5-bs": JournalSlot((1577, 534), 0.98, 2)})
    state.screen_signature = "2:0,0,1920x1080"
    path = tmp_path / "state.json"
    state.mark_dirty()
    state.save(path)

    loaded = AppState.load(path)
    check(loaded.anchors.origin == (10, 20), "anchors survive a round trip")
    check(loaded.anchors.offsets.get("accept") == (20, 40), "offsets survive a round trip")
    check(loaded.journals.position("t5-bs") == (1577, 534), "journal slots survive a round trip")
    check(loaded.screen_signature == "2:0,0,1920x1080", "the screen signature survives")

    invalidated = loaded.apply_signature("1:0,0,2560x1440")
    check(invalidated, "a geometry change reports invalidation")
    check(loaded.anchors.origin is None and not loaded.journals, "a geometry change clears cached pixels")

    check(not AppState.load(tmp_path / "missing.json").journals, "a missing state file yields empty state")
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    check(not AppState.load(tmp_path / "broken.json").journals, "a corrupt state file yields empty state")


def test_pyramid(library, config) -> None:
    print("\n[coarse-to-fine search]")
    template = library.menu["ready"]
    frame = synthetic_frame(template, (700, 500), size=(1920, 1080), origin=(0, 0))

    exact = locate_in(frame, template, None, config.match.menu_score)
    check(exact is not None and exact.top_left == (700, 500), f"full search finds the paste point (got {exact})")

    coarse = pyramid_locate(
        frame, template,
        threshold=config.match.menu_score,
        scale=config.match.coarse_scale,
        slack=config.match.coarse_score_slack,
        refine_padding=config.match.refine_padding,
    )
    check(
        coarse is not None and coarse.top_left == (700, 500),
        f"coarse-to-fine agrees with the full search (got {coarse})",
    )

    roi = Rect(660, 460, 220, 220)
    windowed = locate_in(frame, template, roi, config.match.menu_score)
    check(
        windowed is not None and windowed.top_left == (700, 500),
        f"an anchored ROI finds the same point (got {windowed})",
    )
    check(
        locate_in(frame, template, Rect(0, 0, 200, 200), config.match.menu_score) is None,
        "an ROI pointed at the wrong place reports a miss instead of a false hit",
    )


# ----------------------------------------------------------------------
# input injection
#
# No real window and no real cursor: every OS call these tests care about is
# swapped out, so what is under test is the *ordering and gating* of the
# injected events - which is where the intermittent shift-click failures were.
# ----------------------------------------------------------------------


@contextmanager
def patched(module, **attributes):
    saved = {name: getattr(module, name) for name in attributes}
    for name, value in attributes.items():
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(module, name, value)


class FakeUser32:
    """Counts attaches, because an attach is what resets keyboard state.

    Only the input-queue calls are faked; everything else (screen metrics, key
    mapping) falls through to the real user32 so the injected structures are
    built exactly as they are at runtime.
    """

    def __init__(self, key_down: bool = True) -> None:
        self.attach_calls = 0
        self.key_down = key_down
        self._real = winput._user32

    def __getattr__(self, name):
        try:
            real = self.__dict__["_real"]
        except KeyError:  # during __init__, before _real exists
            raise AttributeError(name) from None
        return getattr(real, name)

    def AttachThreadInput(self, _self_thread, _target, attach):  # noqa: N802
        if attach:
            self.attach_calls += 1
        return 1

    def GetKeyState(self, _vk_code):  # noqa: N802
        return -32768 if self.key_down else 0

    def GetWindowThreadProcessId(self, _hwnd, _pid):  # noqa: N802
        return 4242


class FakeWin32Gui:
    error = RuntimeError

    def IsWindow(self, hwnd):  # noqa: N802
        return bool(hwnd)


class FakeCursor:
    """Stands in for win32api, recording pointer placement into a shared log."""

    error = RuntimeError

    def __init__(self, events: List[tuple], at: Tuple[int, int] = (0, 0), stuck: bool = False) -> None:
        self.events = events
        self.pos = at
        self.stuck = stuck

    def SetCursorPos(self, position):  # noqa: N802
        self.events.append(("place", int(position[0]), int(position[1])))
        if not self.stuck:
            self.pos = (int(position[0]), int(position[1]))

    def GetCursorPos(self):  # noqa: N802
        return self.pos

    def keybd_event(self, *_args):
        self.events.append(("keybd_event",))


def _mouse_recorder(events: List[tuple]):
    def record(*inputs):
        for item in inputs:
            if item.type == winput.INPUT_MOUSE:
                events.append(("move", item.union.mi.dx, item.union.mi.dy))
    return record


def _focused_driver(config, hwnd: int = 99):
    driver = winput.InputDriver(config.input, target_titles=["Albion"])
    driver.target_hwnd = hwnd
    return driver


def test_the_shift_probe_cannot_reset_the_modifier_it_measures() -> None:
    """AttachThreadInput resets key state, so it must not run while shift is held.

    This is the defect behind intermittent shift-clicks: verifying the modifier
    against the game's input queue was capable of clearing the modifier.
    """
    fake = FakeUser32()
    with patched(winput, _user32=fake, window_thread_id=lambda _hwnd: 4242):
        check(
            winput.key_state_in_window(1, winput.VK_SHIFT) is True,
            "the game's input queue can be read when no modifier is held",
        )
        check(fake.attach_calls == 1, "reading it costs exactly one attach")

        winput._shift_held = True
        try:
            state = winput.key_state_in_window(1, winput.VK_SHIFT)
        finally:
            winput._shift_held = False

    check(state is None, "the queue read is refused while an injected shift is held")
    check(fake.attach_calls == 1, "no attach happens while shift is held - it would reset the key state")


def test_the_per_click_gate_is_focus_and_never_attaches(config) -> None:
    driver = _focused_driver(config)
    fake = FakeUser32()
    common = dict(_user32=fake, win32gui=FakeWin32Gui(), is_pressed=lambda _vk: True)

    with patched(winput, foreground_hwnd=lambda: 99, **common):
        check(driver.shift_reaches_target() is True, "shift counts as reaching a focused game")
    with patched(winput, foreground_hwnd=lambda: 12345, **common):
        check(
            driver.shift_reaches_target() is False,
            "and as not reaching it when another window has focus - the click would arrive unmodified",
        )
    with patched(winput, foreground_hwnd=lambda: 99, _user32=fake, win32gui=FakeWin32Gui(),
                 is_pressed=lambda _vk: False):
        check(driver.shift_reaches_target() is False, "and as not reaching it when the OS never took the key")

    check(fake.attach_calls == 0, "the per-click gate never touches the game's input queue")


def test_the_move_is_injected_before_the_pointer_is_placed(config) -> None:
    """SetCursorPos first would leave the injected move with nothing to move."""
    events: List[tuple] = []
    driver = winput.InputDriver(config.input)
    cursor = FakeCursor(events, at=(10, 10))

    with patched(winput, send_input=_mouse_recorder(events), win32api=cursor):
        check(driver.move_cursor((400, 300)), "the cursor reaches the target")

    kinds = [event[0] for event in events]
    check(
        kinds[:2] == ["move", "place"],
        f"the movement event is injected before the pointer is placed (got {kinds})",
    )


def test_a_cursor_already_on_the_slot_still_generates_a_movement(config) -> None:
    """Otherwise the game never sees the pointer enter, and clicks its old hover target."""
    events: List[tuple] = []
    driver = winput.InputDriver(config.input)
    cursor = FakeCursor(events, at=(400, 300))

    with patched(winput, send_input=_mouse_recorder(events), win32api=cursor):
        check(driver.move_cursor((400, 300)), "the cursor is already there and stays there")

    target = winput._absolute_move_input(400, 300).union.mi
    moves = [event[1:] for event in events if event[0] == "move"]
    check(len(moves) >= 2, f"a step off the slot precedes the move onto it (got {len(moves)} move(s))")
    check(
        bool(moves) and moves[0] != (target.dx, target.dy) and moves[-1] == (target.dx, target.dy),
        "the last injected move lands on the slot, the first one does not",
    )


def test_no_click_is_fired_at_a_position_the_cursor_never_reached(config) -> None:
    events: List[tuple] = []
    driver = _focused_driver(config)
    cursor = FakeCursor(events, at=(10, 10), stuck=True)
    patches = dict(
        send_input=_mouse_recorder(events), win32api=cursor, win32gui=FakeWin32Gui(),
        _user32=FakeUser32(), foreground_hwnd=lambda: 99, is_pressed=lambda _vk: True,
    )

    with patched(winput, **patches):
        check(driver.click((400, 300)) is False, "a plain click is refused when the cursor could not be placed")
        check(
            driver.shift_click((400, 300)) is False,
            "and so is a shift-click - it would otherwise land on whatever is under the pointer",
        )
    check(winput._shift_held is False, "nothing left the modifier held")


def test_a_rejected_injection_still_releases_the_modifier(config) -> None:
    """The release has to run even when the press raised, or shift stays down."""
    events: List[tuple] = []
    driver = _focused_driver(config)
    cursor = FakeCursor(events, at=(10, 10))

    def reject(*_inputs):
        raise winput.InputError("blocked by UIPI")

    with patched(winput, send_input=reject, win32api=cursor, win32gui=FakeWin32Gui(),
                 _user32=FakeUser32(), foreground_hwnd=lambda: 99, is_pressed=lambda _vk: False):
        fired = driver.shift_click((400, 300))

    check(fired is False, "a shift-click whose modifier never registered reports failure")
    check(winput._shift_held is False, "the attach guard is cleared even when every injection was rejected")
    check(driver._shift_stuck is False, "and the release is not recorded as stuck when the key is up")


def main() -> int:
    import tempfile

    config = Config()
    library = build_library(config, MAPPING_PATH)

    print("library warnings:")
    for warning in library.warnings:
        print(f"  ! {warning}")

    tests: List[Callable[[], None]] = [
        test_geometry,
        lambda: test_template_loading(library),
        lambda: test_stop_laborers(library, config),
        lambda: test_candidate_reduction(library),
        lambda: test_classification(library, config),
        lambda: test_tier_margins(library, config),
        lambda: test_unknown_tier_is_rejected(library, config),
        lambda: test_missing_journal_is_not_a_misread(library, config),
        test_anchors,
        test_coincident_anchors_are_kept,
        lambda: test_advance_tier_not_confused_with_take_all(library, config),
        lambda: test_a_new_high_tier_laborer_keeps_its_journal(library, config),
        lambda: test_assets_load_whatever_the_extension_case(library),
        lambda: test_beeps_only_on_attention(library),
        lambda: test_pickup_assets(library),
        lambda: test_dialog_buttons_rank_against_each_other(library, config),
        lambda: test_journal_scan_finds_every_stack(library, config),
        lambda: test_journal_verify_catches_a_reflow(library, config),
        test_journal_collisions,
        lambda: test_pyramid(library, config),
    ]

    if winput is not None:
        tests.extend([
            lambda: test_the_game_window_is_picked_over_windows_that_merely_mention_it(config),
            test_the_shift_probe_cannot_reset_the_modifier_it_measures,
            lambda: test_the_per_click_gate_is_focus_and_never_attaches(config),
            lambda: test_the_move_is_injected_before_the_pointer_is_placed(config),
            lambda: test_a_cursor_already_on_the_slot_still_generates_a_movement(config),
            lambda: test_no_click_is_fired_at_a_position_the_cursor_never_reached(config),
            lambda: test_a_rejected_injection_still_releases_the_modifier(config),
        ])

    with tempfile.TemporaryDirectory() as directory:
        tests.append(lambda: test_state_roundtrip(Path(directory)))
        tests.append(lambda: test_config_migration_reaches_an_existing_file(Path(directory)))
        for test in tests:
            try:
                test()
            except Exception:
                FAILURES.append("exception")
                traceback.print_exc()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S)")
        return 1
    print("all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
