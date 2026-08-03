"""The laborer cycle: an explicit sequence with verified transitions.

Shape of one cycle:

    COLLECT   click take-all, and advance-tier if it is offered
    IDENTIFY  confirm 'ready', then classify the laborer portrait
    DELIVER   shift-click the journal until the accept prompt appears
    ACCEPT    click accept and confirm it disappeared

Two structural differences from v3:

*   Every step is anchored. Once the dialog's internal offsets are learned, a
    lookup is a ~1 ms match inside a small box; a miss escalates to a full
    coarse-to-fine search that re-learns the anchor, so the UI moving heals
    itself instead of breaking the run.

*   The accept prompt is the success signal for handing over a journal, not
    "the inventory slot emptied". v3 retried shift-clicks until the slot looked
    empty, which conflates "the laborer took it" with "the stack moved" and can
    multiply clicks. Here the shift-click retry loop exits the moment accept
    appears, so a successful hand-over can never be clicked twice.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

from .assets import AssetLibrary, describe_candidates
from .config import CONFUSABLE_ELEMENTS, Config, LABORER_ELEMENT, STATE_PATH
from .feedback import Feedback, Outcome, Timer
from .identify import LaborerVerdict, Rejection, classify
from .inventory import ScanReport, check_journal_at, inventory_rect, scan_journals
from .state import AppState
from .vision import (
    Grabber,
    Match,
    Rect,
    Template,
    coarse_locate,
    locate_in,
    pyramid_locate,
    scaled_source,
)
from .winput import GameWindow, InputDriver, PanicAbort, find_game_window, foreground_title

log = logging.getLogger("laborer.engine")


class CycleTimeout(Exception):
    """The cycle exceeded its wall-clock budget; the UI is not where we think."""


@dataclass
class Geometry:
    monitor_index: int
    monitor: Rect
    search: Rect
    window_title: str

    @property
    def signature(self) -> str:
        return f"{self.monitor_index}:{self.monitor.left},{self.monitor.top},{self.monitor.width}x{self.monitor.height}"


class LaborerEngine:
    def __init__(
        self,
        config: Config,
        library: AssetLibrary,
        state: AppState,
        grabber: Grabber,
        driver: InputDriver,
        feedback: Feedback,
        state_path=STATE_PATH,
    ) -> None:
        self.cfg = config
        self.library = library
        self.state = state
        self.grabber = grabber
        self.driver = driver
        self.feedback = feedback
        self.state_path = state_path

        self.geometry: Optional[Geometry] = None
        self.active_templates: Dict[str, Template] = {}
        self._candidate_signature: Tuple[str, ...] = ()
        self._observations: Dict[str, Tuple[int, int]] = {}
        self._deadline = 0.0
        self._announced = False
        self.consecutive_failures = 0

    # ------------------------------------------------------------------
    # geometry and candidates
    # ------------------------------------------------------------------

    def refresh_geometry(self) -> Optional[Geometry]:
        monitors = self.grabber.monitors()
        window: Optional[GameWindow] = None
        if self.cfg.window.title_contains or self.cfg.window.process_contains:
            window = find_game_window(
                self.cfg.window.title_contains,
                self.cfg.window.process_contains,
            )

        monitor_index = self.cfg.window.monitor_index
        if monitor_index is None and window is not None:
            left, top, width, height = window.bounds
            monitor_index = self.grabber.monitor_at((left + width // 2, top + height // 2))
        if monitor_index is None:
            monitor_index = 1 if len(monitors) > 1 else 0
            # A guess, and on a multi-monitor desk a wrong one half the time -
            # which looks exactly like the bug this replaced: every search runs
            # against a screen the game is not on and simply finds nothing. Say
            # so, rather than let it read as a normal start-up.
            if len(monitors) > 2:
                log.warning(
                    "no game window matched %s / %s - falling back to monitor %d of %d. "
                    "If nothing is ever found, the game is probably on another screen: "
                    "set window.monitor_index in the config.",
                    self.cfg.window.process_contains or "(no process filter)",
                    self.cfg.window.title_contains or "(no title filter)",
                    monitor_index, len(monitors) - 1,
                )
        if monitor_index <= 0 or monitor_index >= len(monitors):
            log.error("monitor index %s is out of range (%d monitors)", monitor_index, len(monitors) - 1)
            return None

        monitor = monitors[monitor_index]
        search = monitor
        if window is not None and self.cfg.window.restrict_to_window:
            left, top, width, height = window.bounds
            clamped = Rect(left, top, width, height).clamp_to(monitor)
            if clamped is not None and clamped.area > monitor.area // 8:
                search = clamped

        # The driver needs the handle, not just the title: it focuses this
        # window before injecting a modifier and reads shift back out of its
        # input queue afterwards.
        self.driver.target_hwnd = window.hwnd if window else None

        geometry = Geometry(
            monitor_index=monitor_index,
            monitor=monitor,
            search=search,
            window_title=window.title if window else "",
        )
        if self.geometry is None or geometry.signature != self.geometry.signature:
            log.info(
                "monitor %d %dx%d at (%d,%d); search region %dx%d at (%d,%d)%s",
                monitor_index, monitor.width, monitor.height, monitor.left, monitor.top,
                search.width, search.height, search.left, search.top,
                f"; window '{geometry.window_title}'" if geometry.window_title else "",
            )
        self.geometry = geometry
        if self.state.apply_signature(geometry.signature):
            self.refresh_candidates(force=True)
        return geometry

    @property
    def search_rect(self) -> Rect:
        if self.geometry is None:
            raise RuntimeError("geometry not resolved; call refresh_geometry() first")
        return self.geometry.search

    def refresh_candidates(self, force: bool = False) -> None:
        """Narrow the laborer templates to tiers we hold a journal for."""
        active, excluded = self.library.active_laborers(self.state.journals.names())
        signature = tuple(sorted(active))
        if not force and signature == self._candidate_signature:
            return
        self._candidate_signature = signature
        self.active_templates = active
        for line in describe_candidates(active, excluded, self.library.stop_laborers):
            log.info(line)

    # ------------------------------------------------------------------
    # inventory
    # ------------------------------------------------------------------

    def scan_inventory(self) -> ScanReport:
        if self.geometry is None and self.refresh_geometry() is None:
            raise RuntimeError("cannot scan without a resolved monitor")

        region = inventory_rect(self.search_rect, self.cfg.roi.inventory_left_ratio)
        clamped = region.clamp_to(self.search_rect) or region
        report = scan_journals(
            self.grabber,
            clamped,
            self.library.scannable_journals(),
            self.cfg,
            self.geometry.monitor_index,
        )
        for line in report.summary():
            log.info(line)
        for collision in report.collisions:
            log.warning(collision)
        if not report.slots:
            # "not in inventory" per journal reads as a fact about the bag. When
            # *every* journal is missing it is far more often a fact about the
            # region: the wrong window was picked, so the wrong screen was
            # searched. Point at the region, which is the thing to check.
            log.warning(
                "no journal at all in %dx%d at (%d,%d) on monitor %d%s - "
                "if the inventory is open, that region is not where it is drawn",
                clamped.width, clamped.height, clamped.left, clamped.top,
                self.geometry.monitor_index,
                f" ('{self.geometry.window_title}')" if self.geometry.window_title else "",
            )

        self.state.journals.replace(report.slots)
        self.state.mark_dirty()
        self.refresh_candidates(force=True)
        self.state.flush(self.state_path)
        return report

    # ------------------------------------------------------------------
    # element location
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        self.driver.check_panic()
        if self._deadline and time.monotonic() > self._deadline:
            raise CycleTimeout(f"cycle exceeded {self.cfg.timing.cycle_timeout:.1f}s")

    def _anchored_rect(self, element: str, size: Tuple[int, int], padding: int) -> Optional[Rect]:
        rect = self.state.anchors.element_rect(element, size, padding)
        if rect is None:
            return None
        clamped = rect.clamp_to(self.search_rect)
        if clamped is None or clamped.width < size[0] or clamped.height < size[1]:
            return None
        return clamped

    def _threshold_for(self, element: str) -> float:
        """Per-element match floor.

        advance-tier needs its own: its template scores ~0.79 against a clean
        take-all button and higher on a live capture, so at the shared 0.80
        floor v4 would "find" advance-tier on the button it had just clicked.
        """
        if element == "advance_tier":
            return self.cfg.match.advance_tier_score
        return self.cfg.match.menu_score

    def _is_confused(self, match: Match) -> bool:
        """True if some other dialog button fits these pixels better than this one.

        take-all, advance-tier and accept are the same button art with different
        text, drawn in the same place at different points in the dialog's life,
        and they score 0.89-0.96 against each other. A score floor cannot
        separate them - only ranking them against the same pixels can, which is
        the identical argument the inventory scan makes about journal icons.
        """
        rivals = CONFUSABLE_ELEMENTS.get(match.name)
        if not rivals:
            return False

        rect = match.rect.inflate(self.cfg.match.confusion_padding).clamp_to(self.search_rect)
        if rect is None:
            return False
        frame = self.grabber.grab(rect)

        for rival_name in rivals:
            rival = self.library.menu.get(rival_name)
            if rival is None or rect.width < rival.width or rect.height < rival.height:
                continue
            contender = locate_in(frame, rival, None, 0.0)
            if contender is not None and contender.score > match.score:
                log.info(
                    "ignoring '%s' at %s (%.3f): '%s' fits the same spot better (%.3f)",
                    match.name, match.top_left, match.score, rival_name, contender.score,
                )
                return True
        return False

    def _locate_anchored(self, element: str) -> Optional[Match]:
        template = self.library.menu[element]
        rect = self._anchored_rect(element, template.size, self.cfg.roi.anchor_padding)
        if rect is None:
            return None
        frame = self.grabber.grab(rect)
        return locate_in(frame, template, None, self._threshold_for(element))

    def _dialog_region(self) -> Optional[Rect]:
        """The learned dialog box, generously padded - or None if not learned yet."""
        rect = self.state.anchors.dialog_rect(self.library.menu_sizes(), self.cfg.roi.dialog_padding)
        return rect.clamp_to(self.search_rect) if rect is not None else None

    def _search(self, region: Rect, element: str) -> Optional[Match]:
        template = self.library.menu[element]
        if region.width < template.width or region.height < template.height:
            return None
        frame = self.grabber.grab(region)
        return pyramid_locate(
            frame,
            template,
            threshold=self._threshold_for(element),
            scale=self.cfg.match.coarse_scale,
            slack=self.cfg.match.coarse_score_slack,
            refine_padding=self.cfg.match.refine_padding,
        )

    def _locate_full(self, element: str) -> Optional[Match]:
        """Search the dialog first, the whole window only as a last resort.

        A dialog element cannot be outside the dialog, and searching the entire
        window for one invites a distant false positive - which is how an accept
        prompt got "found" at 0.804 in a screen corner and clicked. Falling back
        to the full window keeps the self-healing behaviour for a UI that has
        genuinely moved.
        """
        region = self._dialog_region()
        if region is not None:
            match = self._search(region, element)
            if match is not None:
                return match
        return self._search(self.search_rect, element)

    def locate_element(self, element: str) -> Optional[Match]:
        """Anchored box first, full coarse-to-fine search as a self-healing fallback."""
        self._tick()
        match = self._locate_anchored(element)
        if match is not None and not self._is_confused(match):
            self._observations[element] = match.top_left
            log.debug("%s at %s via anchor (%.3f)", element, match.top_left, match.score)
            return match
        match = self._locate_full(element)
        if match is not None and self._is_confused(match):
            return None
        if match is not None:
            self._observations[element] = match.top_left
            log.debug("%s at %s via full search (%.3f)", element, match.top_left, match.score)
        return match

    def wait_for_element(self, element: str, timeout: float) -> Optional[Match]:
        template = self.library.menu[element]
        anchored = self._anchored_rect(element, template.size, self.cfg.roi.anchor_padding) is not None
        deadline = time.monotonic() + timeout

        while True:
            self._tick()
            match = self._locate_anchored(element) if anchored else self._locate_full(element)
            if match is not None and not self._is_confused(match):
                self._observations[element] = match.top_left
                return match
            if time.monotonic() >= deadline:
                break
            time.sleep(self.cfg.timing.poll_seconds)

        if anchored:
            # The anchor may simply be stale; one full search both answers the
            # question and re-learns the offset.
            match = self._locate_full(element)
            if match is not None and self._is_confused(match):
                return None
            if match is not None:
                self._observations[element] = match.top_left
                log.debug("%s found by escalation; anchor refreshed", element)
            return match
        return None

    def wait_until_gone(self, element: str, timeout: float) -> bool:
        template = self.library.menu[element]
        anchored = self._anchored_rect(element, template.size, self.cfg.roi.anchor_padding) is not None
        deadline = time.monotonic() + timeout
        while True:
            self._tick()
            # Without an anchor the anchored lookup returns None unconditionally,
            # which used to read as "gone" on the very first poll.
            match = self._locate_anchored(element) if anchored else self._locate_full(element)
            if match is None or self._is_confused(match):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(self.cfg.timing.poll_seconds)

    # ------------------------------------------------------------------
    # laborer identification
    # ------------------------------------------------------------------

    def _coarse_laborer_rect(self) -> Optional[Rect]:
        """Find *where* the portrait is without deciding *which* it is.

        One downscale of the search region shared by every candidate: ~9 ms per
        template at half scale instead of ~41 ms, so even this cold path costs
        a few hundred milliseconds rather than v3's multiple seconds.
        """
        # Any laborer icon locates the portrait at half scale - they share the
        # frame - so the reduced set is enough here and costs less.
        templates = self.active_templates or self.library.laborers
        if not templates:
            return None
        frame = self.grabber.grab(self.search_rect)
        small = scaled_source(frame, self.cfg.match.coarse_scale)

        best: Optional[Tuple[float, Tuple[int, int]]] = None
        for template in templates.values():
            coarse = coarse_locate(frame, template, self.cfg.match.coarse_scale, small)
            if coarse is None:
                continue
            if best is None or coarse[0] > best[0]:
                best = coarse

        if best is None:
            return None
        score, top_left = best
        floor = self.cfg.match.laborer_score - self.cfg.match.coarse_score_slack
        if score < floor:
            log.debug("coarse laborer search best %.3f < %.3f", score, floor)
            return None

        width, height = self.library.max_laborer_size()
        padding = self.cfg.roi.laborer_padding + self.cfg.match.refine_padding
        rect = Rect(top_left[0] - padding, top_left[1] - padding, width + 2 * padding, height + 2 * padding)
        return rect.clamp_to(self.search_rect)

    def identify_laborer(self) -> Union[LaborerVerdict, Rejection]:
        """Locate the portrait, then classify it against *every* loaded tier.

        Candidate reduction narrows the *locate* step to tiers we hold a
        journal for, because that is where the cost is. The *classify* step
        deliberately uses the full template set: deciding what a laborer is
        must not depend on what is in the bag, or a ready t3-bs gets scored
        only against t5-bs, matches at ~0.98 and receives the wrong journal.
        """
        self._tick()
        classification_templates = self.library.laborers
        max_size = self.library.max_laborer_size()
        rect = self._anchored_rect(LABORER_ELEMENT, max_size, self.cfg.roi.laborer_padding)
        first: Optional[Rejection] = None

        if rect is not None:
            frame = self.grabber.grab(rect, color=True)
            result = classify(
                frame, None, classification_templates, self.library.tier_weights, self.cfg
            )
            if isinstance(result, LaborerVerdict):
                self._remember_laborer(result)
                return result
            first = result
            log.debug("anchored portrait rejected (%s); widening search", result.describe())

        rect = self._coarse_laborer_rect()
        if rect is None:
            return first or Rejection("not_found", "no laborer portrait in the search region")

        frame = self.grabber.grab(rect, color=True)
        result = classify(frame, None, classification_templates, self.library.tier_weights, self.cfg)
        if isinstance(result, LaborerVerdict):
            self._remember_laborer(result)
        return result

    def _remember_laborer(self, verdict: LaborerVerdict) -> None:
        # The verdict already carries the crop-offset-corrected origin, so the
        # anchor does not shift when a different tier is recognised next cycle.
        self._observations[LABORER_ELEMENT] = verdict.icon_origin

    # ------------------------------------------------------------------
    # journal hand-over
    # ------------------------------------------------------------------

    def deliver_journal(self, laborer_name: str) -> Tuple[Optional[Match], Outcome]:
        journal_name = self.library.journal_for(laborer_name)
        if journal_name is None:
            return None, Outcome.NO_JOURNAL

        journals = self.library.scannable_journals()
        if journal_name not in journals:
            return None, Outcome.NO_JOURNAL

        position = self.state.journals.position(journal_name)
        if position is None:
            log.warning("%s needs '%s' but it is not in the inventory cache", laborer_name, journal_name)
            return None, Outcome.NO_JOURNAL

        def still_there(where: Tuple[int, int]) -> bool:
            """Confirm the cached slot, tolerating a panel that is still drawing.

            A single check is a coin flip on timing: advancing a tier redraws
            over the inventory, and a check landing in those few frames sees
            nothing journal-like and declares a reflow that never happened.
            Poll instead, and keep the cached position when it comes back.
            """
            deadline = time.monotonic() + self.cfg.timing.journal_verify_timeout
            waited = False
            reason = ""
            while True:
                self._tick()
                ok, reason = check_journal_at(
                    self.grabber, journal_name, journals, where, self.cfg, self.search_rect
                )
                if ok:
                    if waited:
                        log.debug("slot for '%s' settled at the cached position %s", journal_name, where)
                    return True
                # A different journal is genuinely there; waiting cannot change
                # that, so stop rather than burn the whole budget on it.
                if reason.startswith("different") or time.monotonic() >= deadline:
                    break
                waited = True
                time.sleep(self.cfg.timing.poll_seconds)

            log.info("slot for '%s' at %s - %s", journal_name, where, reason)
            return False

        if not still_there(position):
            log.info("cached slot for '%s' is stale", journal_name)
            if not self.cfg.safety.rescan_inventory_on_stale:
                return None, Outcome.JOURNAL_FAILED
            # A reflowed inventory invalidates every slot, not just this one.
            self.scan_inventory()
            position = self.state.journals.position(journal_name)
            if position is None:
                return None, Outcome.NO_JOURNAL
            if not still_there(position):
                return None, Outcome.JOURNAL_FAILED

        attempts = max(1, self.cfg.input.shift_click_attempts)
        for attempt in range(attempts):
            self._tick()
            if attempt > 0 and not still_there(position):
                # The slot emptied but no accept prompt appeared. Clicking again
                # would hit whatever the inventory reflowed into, i.e. the wrong
                # item, so stop here and report rather than guess.
                log.warning("'%s' left the slot without an accept prompt; not retrying blind", journal_name)
                return None, Outcome.JOURNAL_FAILED

            if not self.driver.shift_click(position):
                # No click was sent, because the game could not be confirmed to
                # have shift. Retrying cannot change that within this cycle, and
                # every retry is another chance for a plain click to pick the
                # journal up onto the cursor. Stop and report.
                log.error("could not deliver a shift-click to '%s'; nothing was clicked", journal_name)
                return None, Outcome.JOURNAL_FAILED
            log.info("shift-clicked '%s' at %s (attempt %d/%d)", journal_name, position, attempt + 1, attempts)

            accept = self.wait_for_element("accept", self.cfg.timing.accept_timeout)
            if accept is not None:
                return accept, Outcome.SUCCESS

            log.info("no accept prompt yet after shift-click %d/%d", attempt + 1, attempts)

        return None, Outcome.JOURNAL_FAILED

    def confirm_accept(self, accept: Match) -> Outcome:
        """Let the accept prompt settle, then click where it actually ended up.

        ``wait_for_element`` returns on the first frame the template crosses the
        threshold, which during a fade- or slide-in is several frames before the
        prompt is stable. Clicking there hits a button that is still moving, or
        one the game has not made interactive yet, and the click is swallowed.

        Re-locating after the pause matters as much as the pause itself: if the
        prompt slid in, the position captured on the first matching frame is
        already stale by the time we would use it.
        """
        time.sleep(self.cfg.timing.accept_settle)

        settled = self.locate_element("accept")
        if settled is None:
            # It matched a moment ago and is gone now. Something else consumed
            # it, so clicking the remembered spot would land on whatever took
            # its place - report instead of guessing.
            log.warning("accept prompt vanished while settling; not clicking a stale position")
            return Outcome.ACCEPT_FAILED

        drift = max(
            abs(settled.center[0] - accept.center[0]),
            abs(settled.center[1] - accept.center[1]),
        )
        if drift:
            log.debug("accept moved %d px while settling; using the settled position", drift)

        if not self.driver.click(settled.center):
            return Outcome.ACCEPT_FAILED
        log.info("clicked accept at %s (%.3f)", settled.center, settled.score)

        if not self.wait_until_gone("accept", self.cfg.timing.accept_timeout):
            log.warning("accept prompt still visible after clicking it")
            return Outcome.ACCEPT_FAILED
        return Outcome.SUCCESS

    # ------------------------------------------------------------------
    # stop laborers
    # ------------------------------------------------------------------

    def pick_up_laborer(self) -> Outcome:
        """Open the dialog's settings menu and take the laborer out of the house.

        Only ever called on a positive t6 template match. Each step is verified
        before the next click goes out: a blind click into a menu that has not
        opened lands on whatever is behind it.
        """
        settings = self.wait_for_element("settings", self.cfg.timing.settings_timeout)
        if settings is None:
            log.error("settings button not found; cannot pick the laborer up")
            return Outcome.PICKUP_FAILED

        if not self.driver.click(settings.center):
            return Outcome.PICKUP_FAILED
        log.info("clicked settings at %s (%.3f)", settings.center, settings.score)
        time.sleep(self.cfg.timing.pickup_settle)

        pick_up = self.wait_for_element("pick_up", self.cfg.timing.pickup_timeout)
        if pick_up is None:
            log.error("'pick up' did not appear after clicking settings; leaving the menu open")
            return Outcome.PICKUP_FAILED

        if not self.driver.click(pick_up.center):
            return Outcome.PICKUP_FAILED
        log.info("clicked pick-up at %s (%.3f)", pick_up.center, pick_up.score)
        time.sleep(self.cfg.timing.confirm_settle)

        confirm = self.wait_for_element("yes", self.cfg.timing.confirm_timeout)
        if confirm is None:
            # The laborer has not been picked up yet - the confirmation is still
            # sitting there waiting, or it never opened. Either way say so
            # rather than report success for something that did not happen.
            log.error("the confirmation did not appear after 'pick up'; nothing was confirmed")
            return Outcome.PICKUP_FAILED

        # Nothing verifies this last click, so an unsent one would be reported
        # as a completed pick-up.
        if not self.driver.click(confirm.center):
            return Outcome.PICKUP_FAILED
        log.info("clicked yes at %s (%.3f)", confirm.center, confirm.score)
        return Outcome.PICKED_UP

    # ------------------------------------------------------------------
    # the cycle
    # ------------------------------------------------------------------

    def input_blocked_reason(self) -> Optional[str]:
        """Is the game the window keyboard input is going to?

        Compares window *handles*, not titles. A title needle strict enough to
        exclude the market tool and a browser tab is also strict enough to
        exclude the client itself if its title ever changes, and the failure
        would be a permanent silent block. The handle came from the process
        filter, so it is exact.
        """
        if not self.cfg.window.require_foreground:
            return None
        routed = self.driver.routing_ok()
        if routed is None or routed:
            return None
        title = foreground_title() or "(none)"
        return f"foreground window is '{title}', not the game"

    def _announce(self, outcome: Outcome) -> None:
        """Beep now rather than at the end of the cycle.

        The t6 alert has to sound *before* the pick-up clicks go out, not after
        them, so the operator can hit the panic key if it looks wrong.
        """
        self._announced = True
        self.feedback.play(outcome)

    def run_cycle(self) -> Outcome:
        timer = Timer()
        self._announced = False
        outcome = self._guarded_cycle(timer)
        log.info("cycle %s (%s)", outcome.value, timer.summary())
        if not self._announced:
            self.feedback.play(outcome)
        if outcome.is_failure:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0
        return outcome

    def _guarded_cycle(self, timer: Timer) -> Outcome:
        self._observations = {}
        self._deadline = 0.0

        blocked = self.input_blocked_reason()
        if blocked is not None:
            log.warning("ignored: %s", blocked)
            return Outcome.BLOCKED

        if self.refresh_geometry() is None:
            return Outcome.ERROR
        if not self.state.journals:
            log.warning("no journal positions cached; press the scan key first")
            return Outcome.NO_JOURNAL

        self.refresh_candidates()
        self._deadline = time.monotonic() + self.cfg.timing.cycle_timeout
        original_cursor = self.driver.cursor_position()

        try:
            return self._run_steps(original_cursor, timer)
        except PanicAbort:
            log.warning("panic key pressed - cycle aborted")
            return Outcome.ABORTED
        except CycleTimeout as error:
            log.error("%s", error)
            return Outcome.ERROR
        except Exception:
            # One bad frame, a monitor unplug or a rejected SendInput must not
            # take the whole assistant down; v3 had no except at all.
            log.exception("unhandled error during cycle")
            return Outcome.ERROR
        finally:
            self._deadline = 0.0
            time.sleep(self.cfg.timing.restore_mouse_delay)
            self.driver.restore_cursor(original_cursor)
            if self._observations:
                self.state.anchors.observe(self._observations)
                self.state.anchors.monitor_index = self.geometry.monitor_index if self.geometry else None
                self.state.mark_dirty()
            self.state.flush(self.state_path)

    def _run_steps(self, original_cursor: Tuple[int, int], timer: Timer) -> Outcome:
        # --- COLLECT ---------------------------------------------------
        take_all = self.locate_element("take_all")
        tier_advanced = False
        if take_all is None:
            log.info("take-all not visible; continuing to the ready check")
        elif not self.driver.click(take_all.center):
            log.warning("take-all was found but not clicked; continuing to the ready check")
        else:
            log.info("clicked take-all at %s (%.3f)", take_all.center, take_all.score)

            # The dialog redraws in the frame we just clicked in, and advance-tier
            # occupies the same spot take-all did. Settle first, then confirm the
            # interface actually changed - anything found before that is the old
            # button, which is how v4 ended up clicking "advance tier" on take-all.
            time.sleep(self.cfg.timing.take_all_settle)
            if self.wait_until_gone("take_all", self.cfg.timing.interface_change_timeout):
                log.debug("take-all is gone; the dialog has redrawn")
            else:
                log.info("take-all is still on screen after %.2fs; advance-tier check is score-gated",
                         self.cfg.timing.take_all_settle + self.cfg.timing.interface_change_timeout)

            advance = self.wait_for_element("advance_tier", self.cfg.timing.advance_tier_timeout)
            if advance is None:
                log.info("no advance-tier offered")
            elif self.driver.click(advance.center):
                log.info("clicked advance-tier at %s (%.3f)", advance.center, advance.score)
                tier_advanced = True
                time.sleep(self.cfg.timing.advance_tier_settle)
                self.driver.restore_cursor(original_cursor)
                if self.cfg.input.click_original_after_advance:
                    self.driver.click(original_cursor)
                    log.debug("re-clicked the original cursor position to reopen the dialog")
        timer.step("collect")

        # --- IDENTIFY --------------------------------------------------
        ready = self.wait_for_element("ready", self.cfg.timing.ready_timeout)
        if ready is None:
            log.info("laborer is not ready")
            timer.step("ready")
            return Outcome.NOT_READY
        timer.step("ready")

        if tier_advanced:
            log.debug("tier advanced; the portrait changed, classifying fresh")

        result = self.identify_laborer()
        timer.step("identify")
        if isinstance(result, Rejection):
            if result.reason == "unknown_tier":
                # Closest template is not close enough: almost certainly a tier
                # above anything captured. Refuse, but do NOT pick it up - a
                # pick-up on a guess cannot be undone, and this branch is by
                # definition the one where we do not know what we are looking at.
                log.warning("refusing to act - %s (no pick-up on an unidentified laborer)", result.describe())
                self._announce(Outcome.STOP_LABORER)
                return Outcome.STOP_LABORER
            log.warning("laborer not identified - %s", result.describe())
            return Outcome.UNKNOWN_LABORER

        log.info("laborer: %s", result.describe())
        if self.library.is_stop(result.name):
            log.warning("stop laborer %s - no journal will be given", result.name)
            # Beep first: the operator hears the t6 before any click goes out
            # and can hit the panic key.
            self._announce(Outcome.STOP_LABORER)
            if not self.cfg.safety.pick_up_stop_laborers:
                return Outcome.STOP_LABORER
            outcome = self.pick_up_laborer()
            timer.step("pick-up")
            return outcome

        journal_name = self.library.journal_for(result.name)
        if journal_name is None or self.state.journals.position(journal_name) is None:
            # Identified correctly, but the bag has nothing for it. This is a
            # supply problem, not a detection failure, and says so out loud.
            log.warning(
                "%s is ready but no '%s' journal is in the inventory - nothing to give it",
                result.name, journal_name or "(unmapped)",
            )
            return Outcome.NO_JOURNAL

        # --- DELIVER ---------------------------------------------------
        accept, outcome = self.deliver_journal(result.name)
        timer.step("deliver")
        if accept is None:
            return outcome

        # --- ACCEPT ----------------------------------------------------
        outcome = self.confirm_accept(accept)
        timer.step("accept")
        return outcome

    # ------------------------------------------------------------------

    def recover(self) -> None:
        """Drop every learned position and start from a clean slate."""
        log.warning("recovering: clearing anchors and rescanning the inventory")
        self.state.anchors.clear()
        self.state.mark_dirty()
        try:
            self.scan_inventory()
        except Exception:
            log.exception("inventory rescan failed during recovery")
        self.consecutive_failures = 0
