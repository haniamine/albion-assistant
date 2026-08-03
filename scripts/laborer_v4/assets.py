"""Asset registry: mapping, templates, tier weights, diagnostics.

This module owns two things the rest of the app depends on.

*Candidate reduction* narrows the laborer templates to exactly those tiers
whose journal was found in the inventory. That set drives the *locate* step,
where the cost is - a coarse half-scale sweep pays per template.

*Tier weights* are built from every loaded tier of a profession, inventory or
not. Deciding what a laborer **is** must never depend on what is in the bag: if
only t2 journals are held, a ready t5-bs still matches t2-bs at 0.996, and a
comparison set narrowed to the inventory has no way to notice. Identify against
everything, then let the caller refuse for want of a journal.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .config import (
    JOURNAL_ASSETS_DIR,
    LABORER_ASSETS_DIR,
    MENU_ASSET_NAMES,
    MENU_ASSETS_DIR,
    Config,
)
from .vision import Template, align_templates, build_tier_weights, load_template

log = logging.getLogger("laborer.assets")

TIER_PATTERN = re.compile(r"^t(\d+)-")


def laborer_tier(name: str) -> Optional[int]:
    match = TIER_PATTERN.match(name)
    return int(match.group(1)) if match else None


def laborer_profession(name: str) -> str:
    _tier, _sep, profession = name.partition("-")
    return profession or name


@dataclass
class AssetLibrary:
    menu: Dict[str, Template]
    laborers: Dict[str, Template]
    journals_by_name: Dict[str, Template]
    # laborer name -> journal asset name; stop laborers are absent.
    laborer_to_journal: Dict[str, str]
    # Per-laborer pixel weights isolating what differs between sibling tiers.
    tier_weights: Dict[str, "np.ndarray"]
    stop_laborers: Set[str]
    warnings: List[str] = field(default_factory=list)

    def is_stop(self, laborer_name: str) -> bool:
        return laborer_name in self.stop_laborers

    def journal_for(self, laborer_name: str) -> Optional[str]:
        return self.laborer_to_journal.get(laborer_name)

    def siblings_of(self, laborer_name: str) -> List[str]:
        """Every loaded tier of the same profession, actionable or not."""
        profession = laborer_profession(laborer_name)
        return sorted(name for name in self.laborers if laborer_profession(name) == profession)

    def max_laborer_size(self) -> Tuple[int, int]:
        if not self.laborers:
            return (0, 0)
        return (
            max(template.width for template in self.laborers.values()),
            max(template.height for template in self.laborers.values()),
        )

    def menu_sizes(self) -> Dict[str, Tuple[int, int]]:
        return {name: template.size for name, template in self.menu.items()}

    def scannable_journals(self) -> Dict[str, Template]:
        """Journals worth scanning for: those an actionable laborer needs.

        mapping.json points stop laborers at a placeholder journal (currently
        'settings') purely so v3's loader would not reject them. Scanning for
        that placeholder would waste time and could cache a bogus slot.
        """
        wanted = set(self.laborer_to_journal.values())
        return {name: template for name, template in self.journals_by_name.items() if name in wanted}

    def active_laborers(self, available_journals: Iterable[str]) -> Tuple[Dict[str, Template], Dict[str, str]]:
        """Reduce laborer templates to those we hold a journal for.

        Stop laborers stay in unconditionally - the whole point of recognising
        them is to refuse to act, which needs no journal.
        Returns (active templates, {excluded laborer: reason}).
        """
        available = set(available_journals)
        active: Dict[str, Template] = {}
        excluded: Dict[str, str] = {}

        for name, template in self.laborers.items():
            if self.is_stop(name):
                active[name] = template
                continue
            journal = self.laborer_to_journal.get(name)
            if journal is None:
                excluded[name] = "no journal mapping"
                continue
            if journal not in available:
                excluded[name] = f"no '{journal}' in inventory"
                continue
            active[name] = template

        return active, excluded


def asset_paths(directory: Path) -> List[Path]:
    """Every PNG in ``directory``, whatever the case of the extension.

    ``glob("*.png")`` is case-insensitive on Windows and case-*sensitive*
    everywhere else, so a hand-captured ``t8-ft.PNG`` is picked up here and
    silently skipped on a Linux checkout. Matching on the suffix explicitly
    makes the two behave the same.
    """
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )


def asset_path(directory: Path, asset_name: str) -> Optional[Path]:
    """Resolve one named asset, tolerating the extension's case.

    Assets get captured by hand and land as .png or .PNG depending on the tool
    used; NTFS does not care but a Linux checkout would, and an explicit lookup
    beats a FileNotFoundError over a file that is plainly sitting there.
    """
    direct = directory / f"{asset_name}.png"
    if direct.exists():
        return direct
    wanted = asset_name.lower()
    for candidate in asset_paths(directory):
        if candidate.stem.lower() == wanted:
            return candidate
    return None


def _menu_asset_path(asset_name: str) -> Optional[Path]:
    return asset_path(MENU_ASSETS_DIR, asset_name)


def is_stop_laborer(
    name: str,
    has_journal: bool,
    stop_prefixes: Sequence[str],
    max_action_tier: int,
) -> bool:
    """Is this laborer one we refuse to work and pick up instead?

    ``stop_prefixes`` is the deliberate answer and always wins.

    ``max_action_tier`` is not a second opinion on the same question - it is a
    guard against a tier we have *no template for* being read as its lower
    sibling, so an unrecognised icon is refused rather than handed the wrong
    journal. A laborer that has its own template *and* a mapping to a journal
    that exists on disk is the opposite of unrecognised, and the ceiling has
    nothing to say about it.

    Letting the ceiling win there is what made adding t8-ft/t8-imb do nothing:
    they parse as tier 8 > 5, so they were filed as stop laborers, and a stop
    laborer's mapping is discarded as a placeholder - which threw the
    t6-ft/t6-imb journals away with it. Nothing scanned for them, and no
    message said why.
    """
    if name.startswith(tuple(stop_prefixes)):
        return True
    tier = laborer_tier(name)
    if tier is None or tier <= max_action_tier:
        return False
    return not has_journal


def load_mapping(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as mapping_file:
        raw = json.load(mapping_file)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a JSON object of laborer -> journal")
    return {str(key): str(value) for key, value in raw.items() if value is not None}


def build_library(config: Config, mapping_path: Path) -> AssetLibrary:
    warnings: List[str] = []
    raw_mapping = load_mapping(mapping_path)

    ceiling = config.laborer.max_action_tier
    stop_prefixes = tuple(config.laborer.stop_prefixes)

    def is_stop_name(name: str, has_journal: bool) -> bool:
        return is_stop_laborer(name, has_journal, stop_prefixes, ceiling)

    # --- menu -----------------------------------------------------------
    menu: Dict[str, Template] = {}
    for element, asset_name in MENU_ASSET_NAMES.items():
        path = _menu_asset_path(asset_name)
        if path is None:
            raise FileNotFoundError(f"menu asset missing: {MENU_ASSETS_DIR / (asset_name + '.png')}")
        menu[element] = load_template(path, name=element)

    # --- laborers -------------------------------------------------------
    laborers: Dict[str, Template] = {}
    for path in asset_paths(LABORER_ASSETS_DIR):
        name = path.stem
        laborers[name] = load_template(path, name=name)

    if not laborers:
        raise FileNotFoundError(f"no laborer templates found in {LABORER_ASSETS_DIR}")

    # Assets in this repo were cropped with different bounds (83x83 to 95x102),
    # so the same on-screen icon matches each at a slightly different top-left.
    # Measure that so sibling match locations can be compared in one space.
    offsets = align_templates(list(laborers.values()))
    laborers = {name: template.with_alignment(offsets.get(name, (0, 0))) for name, template in laborers.items()}
    inconsistent = [name for name, offset in offsets.items() if max(abs(offset[0]), abs(offset[1])) > 6]
    if inconsistent:
        warnings.append(
            "laborer assets cropped inconsistently (auto-corrected): " + ", ".join(sorted(inconsistent))
        )

    # --- journals -------------------------------------------------------
    # Resolve every mapping entry *before* deciding what is a stop laborer:
    # whether a laborer has a real journal on disk is an input to that decision,
    # so it cannot be computed after it.
    resolved: Dict[str, Tuple[str, Path]] = {}
    missing_journal: Dict[str, str] = {}
    for laborer_name, journal_name in raw_mapping.items():
        if laborer_name not in laborers:
            warnings.append(f"mapping.json lists '{laborer_name}' but {laborer_name}.png does not exist")
            continue
        journal_path = asset_path(JOURNAL_ASSETS_DIR, journal_name)
        if journal_path is None:
            missing_journal[laborer_name] = journal_name
            continue
        resolved[laborer_name] = (journal_name, journal_path)

    # Stop *by explicit prefix* is the deliberate kind, and only that kind has a
    # placeholder journal that is meant to be absent. Anything that became a
    # stop laborer via the tier ceiling did so because we could not find its
    # journal, which is a mistake to report rather than a decision to respect.
    stop_by_prefix: Set[str] = {name for name in laborers if name.startswith(stop_prefixes)}
    stop_laborers: Set[str] = {
        name for name in laborers if is_stop_name(name, name in resolved)
    }

    laborer_to_journal: Dict[str, str] = {}
    journals_by_name: Dict[str, Template] = {}
    for laborer_name, (journal_name, journal_path) in resolved.items():
        if laborer_name in stop_laborers:
            # Its mapping is a placeholder; a stop laborer never receives one.
            continue
        laborer_to_journal[laborer_name] = journal_name
        if journal_name not in journals_by_name:
            journals_by_name[journal_name] = load_template(journal_path, name=journal_name)

    for laborer_name, journal_name in sorted(missing_journal.items()):
        if laborer_name in stop_by_prefix:
            continue  # a placeholder mapping; its journal is meant to be absent
        where = JOURNAL_ASSETS_DIR / f"{journal_name}.png"
        if laborer_name in stop_laborers:
            # Demoted to a stop laborer purely because the journal is missing.
            # Left unsaid, this is the same silent disappearance as before, only
            # one step earlier - so name the consequence, not just the file.
            warnings.append(
                f"'{laborer_name}' is above tier {ceiling} and its journal '{journal_name}' "
                f"is not on disk ({where}), so it will be picked up as a stop laborer "
                f"instead of worked - capture that journal to have it worked"
            )
        else:
            warnings.append(f"journal asset missing for '{laborer_name}': {where}")

    # Adding a template and a mapping line is the whole procedure for teaching
    # v4 a new laborer, so say plainly when the ceiling has been stepped over -
    # silence here is what made the previous behaviour so hard to diagnose.
    promoted = sorted(
        name for name in laborer_to_journal
        if (laborer_tier(name) or 0) > ceiling
    )
    if promoted:
        warnings.append(
            f"above laborer.max_action_tier ({ceiling}) but actioned anyway, because "
            f"mapping.json gives each one a journal that exists: " + ", ".join(promoted)
        )

    unmapped = sorted(
        name for name in laborers
        if name not in laborer_to_journal and name not in stop_laborers
    )
    if unmapped:
        warnings.append(
            "laborer templates with no journal mapping (will never be actioned): " + ", ".join(unmapped)
        )

    # The other half of "I added an asset and nothing happened": a journal is
    # only ever scanned for because some laborer asks for it, so one that no
    # mapping points at is invisible no matter how correct the capture is.
    orphan_journals = [
        path.stem for path in asset_paths(JOURNAL_ASSETS_DIR)
        if path.stem not in journals_by_name
    ]
    if orphan_journals:
        warnings.append(
            "journal assets no laborer maps to, so never scanned for: "
            + ", ".join(orphan_journals)
            + " - add a \"laborer\": \"journal\" line to mapping.json"
        )

    # --- tier discrimination weights ------------------------------------
    # Built from every loaded tier of a profession, including stop tiers, so a
    # t6 icon is separable from t5 even though we never act on it. Deliberately
    # independent of the inventory: what a laborer *is* must not depend on what
    # journals happen to be held.
    by_profession: Dict[str, List[Template]] = {}
    for name, template in laborers.items():
        by_profession.setdefault(laborer_profession(name), []).append(template)

    tier_weights: Dict[str, "np.ndarray"] = {}
    for profession, siblings in by_profession.items():
        if len(siblings) < 2:
            warnings.append(
                f"profession '{profession}' has a single tier template; "
                "tier discrimination there falls back to the whole icon"
            )
        tier_weights.update(build_tier_weights(siblings))

    degenerate = [name for name, weight in tier_weights.items() if float(weight.sum()) <= 0.0]
    if degenerate:
        fallback = _fallback_tier_weight(laborers, tier_weights)
        for name in degenerate:
            template = laborers[name]
            tier_weights[name] = _fit_weight(fallback, template.height, template.width)

    warnings.extend(_missing_stop_tier_warnings(laborers, stop_laborers, config))

    return AssetLibrary(
        menu=menu,
        laborers=laborers,
        journals_by_name=journals_by_name,
        laborer_to_journal=laborer_to_journal,
        tier_weights=tier_weights,
        stop_laborers=stop_laborers,
        warnings=warnings,
    )


def _fallback_tier_weight(
    laborers: Dict[str, Template],
    tier_weights: Dict[str, "np.ndarray"],
) -> "np.ndarray":
    """Average badge weighting borrowed from professions that do have siblings.

    The tier badge sits at the same place on every laborer icon, so a
    profession with only one template can still be checked against the shape
    the others revealed.
    """
    usable = [weight for weight in tier_weights.values() if float(weight.sum()) > 0.0]
    if not usable:
        height = max(template.height for template in laborers.values())
        width = max(template.width for template in laborers.values())
        return np.ones((height, width), np.float32)
    height = max(weight.shape[0] for weight in usable)
    width = max(weight.shape[1] for weight in usable)
    accumulator = np.zeros((height, width), np.float32)
    for weight in usable:
        accumulator[: weight.shape[0], : weight.shape[1]] += weight
    return accumulator / len(usable)


def _fit_weight(weight: "np.ndarray", height: int, width: int) -> "np.ndarray":
    fitted = np.zeros((height, width), np.float32)
    rows = min(height, weight.shape[0])
    columns = min(width, weight.shape[1])
    fitted[:rows, :columns] = weight[:rows, :columns]
    return fitted


def _missing_stop_tier_warnings(
    laborers: Dict[str, Template],
    stop_laborers: Set[str],
    config: Config,
) -> List[str]:
    """Flag professions with no template at all above the actionable tier.

    A profession with nothing above the ceiling has no correct template for a
    high-tier icon. It will be caught by the unknown-tier gate (measured badge
    distance 83-93 against every t5/t4/t3/t2 sibling, well over the 55 ceiling)
    and refused - but the operator should know the asset is missing, because
    "refused" is a worse outcome than "recognised".

    Coverage is counted over *every* template above the ceiling, not just the
    stop ones. Once t8-imb was added and correctly became actionable it stopped
    being a stop laborer, and a stop-only count then claimed imb had nothing up
    there while its template sat in the folder - telling the operator to go and
    capture an asset they had just captured.
    """
    ceiling = config.laborer.max_action_tier
    professions = {laborer_profession(name) for name in laborers}
    covered = {
        laborer_profession(name) for name in laborers
        if (laborer_tier(name) or 0) > ceiling
    }
    missing = sorted(professions - covered)
    if not missing:
        return []
    return [
        f"no template above tier {ceiling} for profession(s) "
        + ", ".join(missing)
        + f" - a tier >{ceiling} laborer there will be refused as an unknown tier "
        "rather than recognised; capture the missing asset(s)"
    ]


def describe_candidates(
    active: Dict[str, Template],
    excluded: Dict[str, str],
    stop_laborers: Set[str],
) -> List[str]:
    """Human-readable summary of candidate reduction, logged on every change."""
    actionable = sorted(name for name in active if name not in stop_laborers)
    stops = sorted(name for name in active if name in stop_laborers)
    lines = [
        f"active laborer candidates: {len(active)} "
        f"({len(actionable)} actionable + {len(stops)} stop)",
        "  actionable : " + (", ".join(actionable) if actionable else "(none)"),
        "  stop       : " + (", ".join(stops) if stops else "(none)"),
    ]
    if excluded:
        grouped: Dict[str, List[str]] = {}
        for name, reason in sorted(excluded.items()):
            grouped.setdefault(reason, []).append(name)
        for reason, names in sorted(grouped.items()):
            lines.append(f"  excluded   : {', '.join(names)}  <- {reason}")
    return lines
