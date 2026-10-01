"""Ground emissions settlement per flight.

Airlines reconcile ground emissions against three independent sheets: the planned
consumption, the equipment meter export and the late supplementary meter reading.
Each sheet usually tells a different story and nobody can explain the gap.

This ledger keeps all three figures for every energy stream (fuel, ground power
and preconditioned air), picks an authority by an explicit precedence order, and
explains the variance against the plan.  The conversion rule set (densities,
calorific values, emission factors) is versioned and *pinned* on every
settlement, so changing a factor later never rewrites history.

Readings may still correct a flight whose books are open.  A sealed flight is
immutable: amendments must be booked as a new settlement version that
supersedes the sealed one, while the sealed version (and its digest) stays on
record.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from hashlib import sha256
from json import dumps
from threading import RLock
from typing import Iterable, Mapping, Sequence

from .timeutil import utc


Q = Decimal("0.001")
PCT_Q = Decimal("0.0001")

RULE_FUEL_DENSITY = "fuel.density_kg_l"
RULE_FUEL_KWH_PER_KG = "fuel.kwh_per_kg"
RULE_FUEL_CO2_PER_KG = "fuel.co2e_kg_per_kg"
RULE_POWER_CO2_PER_KWH = "power.co2e_kg_per_kwh"
RULE_AC_CO2_PER_KWH = "ac.co2e_kg_per_kwh"
REQUIRED_RULE_KEYS = frozenset(
    {
        RULE_FUEL_DENSITY,
        RULE_FUEL_KWH_PER_KG,
        RULE_FUEL_CO2_PER_KG,
        RULE_POWER_CO2_PER_KWH,
        RULE_AC_CO2_PER_KWH,
    }
)


class EnergyType(str, Enum):
    FUEL = "fuel"
    POWER = "power"
    AC = "ac"


class ReadingSource(str, Enum):
    PLANNED = "planned"
    METER = "meter"
    SUPPLEMENT = "supplement"


class SettlementStatus(str, Enum):
    OPEN = "open"
    SEALED = "sealed"


ALLOWED_UNITS: Mapping[EnergyType, frozenset[str]] = {
    EnergyType.FUEL: frozenset({"L", "kg"}),
    EnergyType.POWER: frozenset({"kWh"}),
    EnergyType.AC: frozenset({"kWh"}),
}

DEFAULT_AUTHORITY: tuple[ReadingSource, ...] = (
    ReadingSource.SUPPLEMENT,
    ReadingSource.METER,
    ReadingSource.PLANNED,
)


class EmissionsAccountingError(RuntimeError):
    pass


class SettlementSealedError(EmissionsAccountingError):
    def __init__(self, flight_id: str, sealed_version: int):
        self.flight_id = flight_id
        self.sealed_version = sealed_version
        super().__init__(
            f"flight {flight_id} settlement v{sealed_version} is sealed; "
            "book a revision instead of amending it"
        )


def quant(value: Decimal, unit: Decimal = Q) -> Decimal:
    return value.quantize(unit, rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class RuleVersion:
    """One immutable, audit-ready set of conversion rules."""

    version: int
    effective_at: datetime
    factors: Mapping[str, Decimal]
    note: str = ""


@dataclass(frozen=True)
class EnergyReading:
    reading_id: str
    flight_id: str
    energy: EnergyType
    source: ReadingSource
    amount: Decimal
    unit: str
    read_at: datetime
    recorded_at: datetime
    sequence: int = 0


@dataclass(frozen=True)
class SourceFigure:
    source: ReadingSource
    reading_id: str
    amount: Decimal
    unit: str
    read_at: datetime


@dataclass(frozen=True)
class EnergyLine:
    energy: EnergyType
    authority: ReadingSource
    figure: SourceFigure
    sources: tuple[SourceFigure, ...]
    mass_kg: Decimal | None
    energy_kwh: Decimal
    co2e_kg: Decimal
    variance_vs_planned_kwh: Decimal
    variance_vs_planned_pct: Decimal | None
    selection_note: str


@dataclass(frozen=True)
class FlightSettlement:
    flight_id: str
    version: int
    status: SettlementStatus
    rule_version: int
    rules_snapshot: Mapping[str, Decimal]
    lines: tuple[EnergyLine, ...]
    total_energy_kwh: Decimal
    total_co2e_kg: Decimal
    discrepancies_kwh: Mapping[str, Decimal]
    created_at: datetime
    updated_at: datetime
    sealed_at: datetime | None = None
    supersedes: int | None = None
    amended_by_readings: tuple[str, ...] = ()
    digest: str = ""


def _canonical(payload: object) -> bytes:
    return dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


class GroundEmissionsLedger:
    """Settles fuel, ground power and air-conditioning energy per flight."""

    def __init__(self, authority_order: Sequence[ReadingSource] = DEFAULT_AUTHORITY):
        order = tuple(authority_order)
        if set(order) != set(ReadingSource):
            raise ValueError("authority order must cover every reading source")
        self.authority_order = order
        self._rules: dict[int, RuleVersion] = {}
        self._readings: dict[str, EnergyReading] = {}
        self._by_flight: dict[str, list[EnergyReading]] = {}
        self._settlements: dict[str, list[FlightSettlement]] = {}
        self._sequence = 0
        self._lock = RLock()

    # ----- conversion rules -------------------------------------------------

    def publish_rules(
        self,
        factors: Mapping[str, Decimal],
        effective_at: datetime,
        note: str = "",
    ) -> RuleVersion:
        factors = {key: Decimal(value) for key, value in factors.items()}
        missing = REQUIRED_RULE_KEYS - factors.keys()
        if missing:
            raise EmissionsAccountingError(
                f"rule set missing factors: {sorted(missing)}"
            )
        if any(value <= 0 for value in factors.values()):
            raise EmissionsAccountingError("conversion factors must be positive")
        effective_at = utc(effective_at)
        with self._lock:
            version = len(self._rules) + 1
            rule = RuleVersion(version, effective_at, dict(sorted(factors.items())), note)
            self._rules[version] = rule
            return rule

    @property
    def rule_version(self) -> int:
        with self._lock:
            return len(self._rules)

    def rules(self, version: int) -> RuleVersion:
        with self._lock:
            try:
                return self._rules[version]
            except KeyError:
                raise EmissionsAccountingError(f"unknown rule version {version}") from None

    def _require_rules(self, version: int) -> RuleVersion:
        return self.rules(version)

    # ----- readings ----------------------------------------------------------

    def add_reading(
        self,
        reading_id: str,
        flight_id: str,
        energy: EnergyType,
        source: ReadingSource,
        amount: Decimal,
        unit: str,
        read_at: datetime,
        recorded_at: datetime,
    ) -> EnergyReading:
        """Register a reading.

        Open flights are re-settled immediately.  Readings for a sealed flight
        are rejected: callers must open a revision with ``revise``.
        """
        amount = Decimal(amount)
        if amount <= 0:
            raise EmissionsAccountingError("reading amount must be positive")
        if unit not in ALLOWED_UNITS[energy]:
            raise EmissionsAccountingError(
                f"{energy.value} cannot be measured in {unit}"
            )
        with self._lock:
            existing = self._readings.get(reading_id)
            if existing is not None:
                return existing
            history = self._settlements.get(flight_id)
            current = history[-1] if history else None
            if current is not None and current.status is SettlementStatus.SEALED:
                raise SettlementSealedError(flight_id, current.version)
            self._sequence += 1
            reading = EnergyReading(
                reading_id,
                flight_id,
                energy,
                source,
                amount,
                unit,
                utc(read_at),
                utc(recorded_at),
                self._sequence,
            )
            self._readings[reading_id] = reading
            self._by_flight.setdefault(flight_id, []).append(reading)
            if current is not None and current.status is SettlementStatus.OPEN:
                self._rebuild_draft(flight_id, reading.recorded_at)
            elif current is None and self._rules:
                draft = self._build_settlement(
                    flight_id,
                    version=1,
                    rule_version=len(self._rules),
                    at=reading.recorded_at,
                    created_at=reading.recorded_at,
                )
                self._settlements.setdefault(flight_id, []).append(draft)
            return reading

    def readings(self, flight_id: str) -> tuple[EnergyReading, ...]:
        with self._lock:
            return tuple(self._by_flight.get(flight_id, ()))

    # ----- settlement --------------------------------------------------------

    def settle(
        self,
        flight_id: str,
        at: datetime,
        rule_version: int | None = None,
    ) -> FlightSettlement:
        """Create or refresh the open settlement for a flight."""
        at = utc(at)
        with self._lock:
            history = self._settlements.get(flight_id)
            current = history[-1] if history else None
            if current is not None and current.status is SettlementStatus.SEALED:
                raise SettlementSealedError(flight_id, current.version)
            if not self._rules:
                raise EmissionsAccountingError("publish conversion rules before settling")
            if not self._by_flight.get(flight_id):
                raise EmissionsAccountingError(f"no readings for flight {flight_id}")
            if current is None:
                pinned = rule_version or len(self._rules)
            elif rule_version is not None:
                # An open draft may still adopt a newly published rule book;
                # once sealed it can never move.
                pinned = rule_version
            else:
                pinned = current.rule_version
            draft = self._build_settlement(
                flight_id,
                version=current.version if current is not None else self._next_version(history),
                rule_version=pinned,
                at=at,
                created_at=current.created_at if current is not None else at,
                supersedes=current.supersedes if current is not None else None,
                amended=current.amended_by_readings if current is not None else (),
            )
            self._store_draft(flight_id, draft)
            return draft

    def seal(self, flight_id: str, at: datetime) -> FlightSettlement:
        """Close the books: the settlement becomes immutable."""
        at = utc(at)
        with self._lock:
            current = self._require_current(flight_id)
            draft = self._build_settlement(
                flight_id,
                version=current.version,
                rule_version=current.rule_version,
                at=at,
                created_at=current.created_at,
                sealed_at=at,
                supersedes=current.supersedes,
                amended=current.amended_by_readings,
            )
            sealed = self._with_digest(
                FlightSettlement(
                    draft.flight_id,
                    draft.version,
                    SettlementStatus.SEALED,
                    draft.rule_version,
                    draft.rules_snapshot,
                    draft.lines,
                    draft.total_energy_kwh,
                    draft.total_co2e_kg,
                    draft.discrepancies_kwh,
                    draft.created_at,
                    at,
                    at,
                    draft.supersedes,
                    draft.amended_by_readings,
                )
            )
            self._settlements[flight_id][-1] = sealed
            return sealed

    def revise(
        self,
        flight_id: str,
        new_readings: Iterable[EnergyReading],
        at: datetime,
        rule_version: int | None = None,
    ) -> FlightSettlement:
        """Open a new-version draft against a sealed flight.

        The sealed version remains untouched and retrievable; the returned draft
        becomes the next version once sealed.
        """
        at = utc(at)
        readings = tuple(new_readings)
        with self._lock:
            current = self._require_current(flight_id)
            if current.status is not SettlementStatus.SEALED:
                raise EmissionsAccountingError(
                    f"flight {flight_id} is still open; add readings directly"
                )
            if not readings:
                raise EmissionsAccountingError("revision requires new readings")
            added: list[str] = []
            for reading in readings:
                if reading.reading_id in self._readings:
                    raise EmissionsAccountingError(
                        f"reading {reading.reading_id} already registered"
                    )
                if reading.flight_id != flight_id:
                    raise EmissionsAccountingError("reading belongs to another flight")
                if Decimal(reading.amount) <= 0:
                    raise EmissionsAccountingError("reading amount must be positive")
                if reading.unit not in ALLOWED_UNITS[reading.energy]:
                    raise EmissionsAccountingError(
                        f"{reading.energy.value} cannot be measured in {reading.unit}"
                    )
            for reading in readings:
                self._sequence += 1
                stored = EnergyReading(
                    reading.reading_id,
                    reading.flight_id,
                    reading.energy,
                    reading.source,
                    Decimal(reading.amount),
                    reading.unit,
                    utc(reading.read_at),
                    utc(reading.recorded_at),
                    self._sequence,
                )
                self._readings[stored.reading_id] = stored
                self._by_flight.setdefault(flight_id, []).append(stored)
                added.append(stored.reading_id)
            draft = self._build_settlement(
                flight_id,
                version=current.version + 1,
                rule_version=rule_version or len(self._rules),
                at=at,
                created_at=at,
                supersedes=current.version,
                amended=tuple(added),
            )
            self._settlements.setdefault(flight_id, []).append(draft)
            return draft

    def current(self, flight_id: str) -> FlightSettlement:
        with self._lock:
            return self._require_current(flight_id)

    def history(self, flight_id: str) -> tuple[FlightSettlement, ...]:
        with self._lock:
            return tuple(self._settlements.get(flight_id, ()))

    def sealed_version(self, flight_id: str, version: int) -> FlightSettlement:
        with self._lock:
            for settlement in self._settlements.get(flight_id, ()):
                if (
                    settlement.version == version
                    and settlement.status is SettlementStatus.SEALED
                ):
                    return settlement
            raise KeyError((flight_id, version))

    def rules_applied(self, flight_id: str) -> Mapping[str, Decimal]:
        """Conversion factors pinned on the current settlement."""
        return dict(self.current(flight_id).rules_snapshot)

    def verify(self, flight_id: str) -> bool:
        """Recompute the digest of every sealed version of the flight."""
        with self._lock:
            for settlement in self._settlements.get(flight_id, ()):
                if settlement.status is SettlementStatus.SEALED:
                    if self._digest(settlement) != settlement.digest:
                        return False
            return True

    # ----- internals ----------------------------------------------------------

    def _next_version(self, history: list[FlightSettlement] | None) -> int:
        return 1 if not history else max(s.version for s in history) + 1

    def _store_draft(self, flight_id: str, draft: FlightSettlement) -> None:
        history = self._settlements.setdefault(flight_id, [])
        if history and history[-1].status is SettlementStatus.OPEN:
            history[-1] = draft
        else:
            history.append(draft)

    def _rebuild_draft(self, flight_id: str, at: datetime) -> FlightSettlement:
        current = self._settlements[flight_id][-1]
        draft = self._build_settlement(
            flight_id,
            version=current.version,
            rule_version=current.rule_version,
            at=at,
            created_at=current.created_at,
            supersedes=current.supersedes,
            amended=current.amended_by_readings,
        )
        self._settlements[flight_id][-1] = draft
        return draft

    def _require_current(self, flight_id: str) -> FlightSettlement:
        history = self._settlements.get(flight_id)
        if not history:
            raise EmissionsAccountingError(f"no settlement for flight {flight_id}")
        return history[-1]

    def _pick_figure(
        self, readings: Sequence[EnergyReading]
    ) -> tuple[SourceFigure, tuple[SourceFigure, ...]]:
        # Latest reading wins within a source (read_at, then intake sequence).
        chosen_by_source: dict[ReadingSource, EnergyReading] = {}
        for reading in readings:
            incumbent = chosen_by_source.get(reading.source)
            if incumbent is None or (reading.read_at, reading.sequence) > (
                incumbent.read_at,
                incumbent.sequence,
            ):
                chosen_by_source[reading.source] = reading
        figures = tuple(
            SourceFigure(
                r.source,
                r.reading_id,
                Decimal(r.amount),
                r.unit,
                r.read_at,
            )
            for r in sorted(
                chosen_by_source.values(),
                key=lambda r: (r.source.value, r.read_at, r.sequence),
            )
        )
        winning: EnergyReading | None = None
        for source in self.authority_order:
            candidate = chosen_by_source.get(source)
            if candidate is not None:
                winning = candidate
                break
        assert winning is not None
        return (
            SourceFigure(
                winning.source,
                winning.reading_id,
                Decimal(winning.amount),
                winning.unit,
                winning.read_at,
            ),
            figures,
        )

    def _convert(
        self,
        energy: EnergyType,
        figure: SourceFigure,
        factors: Mapping[str, Decimal],
    ) -> tuple[Decimal | None, Decimal, Decimal]:
        amount = figure.amount
        if energy is EnergyType.FUEL:
            mass = quant(
                amount * factors[RULE_FUEL_DENSITY]
                if figure.unit == "L"
                else amount
            )
            energy_kwh = quant(mass * factors[RULE_FUEL_KWH_PER_KG])
            co2 = quant(mass * factors[RULE_FUEL_CO2_PER_KG])
            return mass, energy_kwh, co2
        factor_key = (
            RULE_POWER_CO2_PER_KWH if energy is EnergyType.POWER else RULE_AC_CO2_PER_KWH
        )
        return None, quant(amount), quant(amount * factors[factor_key])

    def _build_settlement(
        self,
        flight_id: str,
        version: int,
        rule_version: int,
        at: datetime,
        created_at: datetime,
        sealed_at: datetime | None = None,
        supersedes: int | None = None,
        amended: tuple[str, ...] = (),
    ) -> FlightSettlement:
        rules = self._require_rules(rule_version)
        factors = rules.factors
        by_energy: dict[EnergyType, list[EnergyReading]] = {}
        for reading in self._by_flight.get(flight_id, ()):
            by_energy.setdefault(reading.energy, []).append(reading)

        lines: list[EnergyLine] = []
        discrepancies: dict[str, Decimal] = {}
        total_kwh = Decimal("0.000")
        total_co2 = Decimal("0.000")
        for energy in sorted(by_energy, key=lambda e: e.value):
            figure, sources = self._pick_figure(by_energy[energy])
            mass, energy_kwh, co2 = self._convert(energy, figure, factors)
            planned = next((s for s in sources if s.source is ReadingSource.PLANNED), None)
            if planned is not None:
                _, planned_kwh, _ = self._convert(energy, planned, factors)
                delta = quant(energy_kwh - planned_kwh)
                pct = (
                    quant(delta * Decimal("100") / planned_kwh, PCT_Q)
                    if planned_kwh != 0
                    else None
                )
            else:
                delta = Decimal("0.000")
                pct = None
            discrepancies[energy.value] = delta
            note = self._selection_note(figure.source, sources)
            lines.append(
                EnergyLine(
                    energy,
                    figure.source,
                    figure,
                    sources,
                    mass,
                    energy_kwh,
                    co2,
                    delta,
                    pct,
                    note,
                )
            )
            total_kwh += energy_kwh
            total_co2 += co2

        return FlightSettlement(
            flight_id,
            version,
            SettlementStatus.SEALED if sealed_at else SettlementStatus.OPEN,
            rule_version,
            dict(factors),
            tuple(lines),
            quant(total_kwh),
            quant(total_co2),
            dict(sorted(discrepancies.items())),
            created_at,
            at,
            sealed_at,
            supersedes,
            tuple(amended),
        )

    def _selection_note(
        self, authority: ReadingSource, sources: Sequence[SourceFigure]
    ) -> str:
        present = {s.source for s in sources}
        if authority is ReadingSource.PLANNED:
            return "planned figure used: no actual reading available"
        rank = self.authority_order.index
        lower = [
            s.value
            for s in self.authority_order
            if s in present and rank(s) > rank(authority)
        ]
        if not lower:
            return f"{authority.value} figure selected by authority order"
        return (
            f"{authority.value} figure selected by authority order; "
            f"overrides {', '.join(lower)}"
        )

    def _digest_body(self, settlement: FlightSettlement) -> dict[str, object]:
        return {
            "flight_id": settlement.flight_id,
            "version": settlement.version,
            "status": settlement.status.value,
            "rule_version": settlement.rule_version,
            "sealed_at": settlement.sealed_at.isoformat() if settlement.sealed_at else None,
            "supersedes": settlement.supersedes,
            "amended_by_readings": list(settlement.amended_by_readings),
            "rules_snapshot": {k: str(v) for k, v in sorted(settlement.rules_snapshot.items())},
            "lines": [
                {
                    "energy": line.energy.value,
                    "authority": line.authority.value,
                    "figure": {
                        "source": line.figure.source.value,
                        "reading_id": line.figure.reading_id,
                        "amount": str(line.figure.amount),
                        "unit": line.figure.unit,
                    },
                    "sources": [
                        {
                            "source": s.source.value,
                            "reading_id": s.reading_id,
                            "amount": str(s.amount),
                            "unit": s.unit,
                        }
                        for s in line.sources
                    ],
                    "mass_kg": str(line.mass_kg) if line.mass_kg is not None else None,
                    "energy_kwh": str(line.energy_kwh),
                    "co2e_kg": str(line.co2e_kg),
                    "variance_vs_planned_kwh": str(line.variance_vs_planned_kwh),
                    "variance_vs_planned_pct": (
                        str(line.variance_vs_planned_pct)
                        if line.variance_vs_planned_pct is not None
                        else None
                    ),
                }
                for line in sorted(settlement.lines, key=lambda l: l.energy.value)
            ],
            "total_energy_kwh": str(settlement.total_energy_kwh),
            "total_co2e_kg": str(settlement.total_co2e_kg),
        }

    def _digest(self, settlement: FlightSettlement) -> str:
        return sha256(_canonical(self._digest_body(settlement))).hexdigest()

    def _with_digest(self, settlement: FlightSettlement) -> FlightSettlement:
        return FlightSettlement(
            settlement.flight_id,
            settlement.version,
            settlement.status,
            settlement.rule_version,
            settlement.rules_snapshot,
            settlement.lines,
            settlement.total_energy_kwh,
            settlement.total_co2e_kg,
            settlement.discrepancies_kwh,
            settlement.created_at,
            settlement.updated_at,
            settlement.sealed_at,
            settlement.supersedes,
            settlement.amended_by_readings,
            self._digest(settlement),
        )
