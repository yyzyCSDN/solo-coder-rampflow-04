"""Per-flight ground emissions reconciliation.

Airlines reconcile ramp energy from three independent evidence trails:

* planned consumption from the turnaround/energy plan,
* equipment meter readings (fuel dispenser, GPU kWh meter, PCA kWh meter),
* late manual readings that arrive after the fact.

They land in different tables and the variances are otherwise unexplainable.
This ledger settles fuel, ground power (GPU) and preconditioned air (PCA) into
one statement per flight, keeping every source value on the line so each
variance is traceable to a reading.

Conversion rules (fuel density/temperature correction and CO2 emission factors)
are published as numbered, append-only editions.  Every statement keeps the
exact edition it was rated under, so a later factor publication never silently
rewrites history.  Open statements may be revised by new readings; a closed
statement is immutable and a late reading can only produce a new, superseding
amendment version.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Iterable, Mapping

from .timeutil import utc

Q = Decimal("0.001")


def quant(value: Decimal) -> Decimal:
    return Decimal(value).quantize(Q, rounding=ROUND_HALF_UP)


class EnergyKind(str, Enum):
    FUEL = "fuel"   # turbine fuel, metered in litres or kg
    GPU = "gpu"     # ground power unit electricity
    PCA = "pca"     # preconditioned air electricity


class ReadingSource(str, Enum):
    PLANNED = "planned"     # energy plan
    METERED = "metered"     # equipment meter
    LATE = "late"           # backfilled manual reading


class StatementStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


class GroundEmissionsError(RuntimeError):
    pass


class StatementClosedError(GroundEmissionsError):
    """Raised when new evidence targets a sealed statement.

    The reading is refused rather than partially applied; callers must create an
    amendment, which leaves the sealed version untouched.
    """

    def __init__(self, flight_id: str, revision: int):
        super().__init__(
            f"emissions statement for flight {flight_id} revision {revision} "
            "is closed; record the reading through an amendment"
        )
        self.flight_id = flight_id
        self.revision = revision


@dataclass(frozen=True)
class EmissionFactors:
    """One numbered, immutable edition of the conversion rules."""

    edition: int
    effective_from: datetime
    fuel_density_kg_l: Decimal
    fuel_temp_coefficient: Decimal
    reference_temp_c: Decimal
    fuel_co2_kg_per_kg: Decimal
    electricity_co2_kg_per_kwh: Decimal
    note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "effective_from", utc(self.effective_from))
        for name in (
            "fuel_density_kg_l",
            "fuel_temp_coefficient",
            "reference_temp_c",
            "fuel_co2_kg_per_kg",
            "electricity_co2_kg_per_kwh",
        ):
            object.__setattr__(self, name, Decimal(getattr(self, name)))
        if self.edition <= 0:
            raise GroundEmissionsError("edition must be positive")
        if self.fuel_density_kg_l <= 0 or self.fuel_co2_kg_per_kg <= 0:
            raise GroundEmissionsError("fuel factors must be positive")
        if self.electricity_co2_kg_per_kwh < 0:
            raise GroundEmissionsError("electricity factor cannot be negative")

    def fuel_liters_to_kg(
        self, amount_l: Decimal, temperature_c: Decimal | None = None
    ) -> Decimal:
        """Temperature-correct metered litres to mass at the reference temp."""
        temp = self.reference_temp_c if temperature_c is None else Decimal(temperature_c)
        expansion = Decimal("1") + self.fuel_temp_coefficient * (
            temp - self.reference_temp_c
        )
        if expansion <= 0:
            raise GroundEmissionsError("invalid fuel temperature correction")
        return quant(Decimal(amount_l) / expansion * self.fuel_density_kg_l)

    def fuel_co2_kg(self, fuel_kg: Decimal) -> Decimal:
        return quant(Decimal(fuel_kg) * self.fuel_co2_kg_per_kg)

    def electricity_co2_kg(self, energy_kwh: Decimal) -> Decimal:
        return quant(Decimal(energy_kwh) * self.electricity_co2_kg_per_kwh)


class EmissionFactorRegistry:
    """Append-only registry of conversion-rule editions."""

    def __init__(self) -> None:
        self._editions: dict[int, EmissionFactors] = {}

    def __len__(self) -> int:
        return len(self._editions)

    @property
    def editions(self) -> tuple[EmissionFactors, ...]:
        return tuple(self._editions[i] for i in sorted(self._editions))

    def publish(
        self,
        effective_from: datetime,
        fuel_density_kg_l: Decimal,
        fuel_temp_coefficient: Decimal,
        reference_temp_c: Decimal,
        fuel_co2_kg_per_kg: Decimal,
        electricity_co2_kg_per_kwh: Decimal,
        note: str = "",
    ) -> EmissionFactors:
        effective = utc(effective_from)
        if self._editions:
            latest = self._editions[max(self._editions)]
            if effective < latest.effective_from:
                raise GroundEmissionsError(
                    "factor editions must be published in effective-date order"
                )
        edition = len(self._editions) + 1
        factors = EmissionFactors(
            edition,
            effective,
            fuel_density_kg_l,
            fuel_temp_coefficient,
            reference_temp_c,
            fuel_co2_kg_per_kg,
            electricity_co2_kg_per_kwh,
            note,
        )
        self._editions[edition] = factors
        return factors

    def publish_standard(
        self,
        effective_from: datetime,
        *,
        fuel_density_kg_l: Decimal = Decimal("0.8"),
        fuel_co2_kg_per_kg: Decimal = Decimal("3.16"),
        electricity_co2_kg_per_kwh: Decimal = Decimal("0.42"),
        note: str = "Jet A-1 at 15C with grid electricity",
    ) -> EmissionFactors:
        return self.publish(
            effective_from,
            fuel_density_kg_l,
            Decimal("0.00095"),
            Decimal("15"),
            fuel_co2_kg_per_kg,
            electricity_co2_kg_per_kwh,
            note,
        )

    def edition(self, number: int) -> EmissionFactors:
        try:
            return self._editions[number]
        except KeyError:
            raise GroundEmissionsError(f"unknown factor edition {number}") from None

    def edition_in_effect(self, at: datetime) -> EmissionFactors:
        moment = utc(at)
        candidates = [f for f in self._editions.values() if f.effective_from <= moment]
        if not candidates:
            raise GroundEmissionsError("no factor edition is effective yet")
        return max(candidates, key=lambda f: f.edition)


@dataclass(frozen=True)
class ConsumptionReading:
    reading_id: str
    flight_id: str
    kind: EnergyKind
    source: ReadingSource
    amount: Decimal
    unit: str
    recorded_at: datetime
    meter_id: str | None = None
    temperature_c: Decimal | None = None


@dataclass(frozen=True)
class EnergyLine:
    kind: EnergyKind
    unit: str
    planned: Decimal
    metered: Decimal
    late: Decimal
    adopted: Decimal
    variance: Decimal
    basis: ReadingSource
    reading_ids: tuple[str, ...]


@dataclass(frozen=True)
class EmissionsStatement:
    statement_id: str
    flight_id: str
    revision: int
    status: StatementStatus
    factors: EmissionFactors
    lines: Mapping[EnergyKind, EnergyLine]
    fuel_kg: Decimal
    gpu_kwh: Decimal
    pca_kwh: Decimal
    fuel_co2_kg: Decimal
    electricity_co2_kg: Decimal
    total_co2_kg: Decimal
    created_at: datetime
    supersedes: int | None
    closed_at: datetime | None
    reason: str

    def reading_ids(self) -> frozenset[str]:
        ids: set[str] = set()
        for line in self.lines.values():
            ids.update(line.reading_ids)
        return frozenset(ids)


def variance_breakdown(statement: EmissionsStatement) -> Mapping[str, Mapping[str, object]]:
    """Explain, per energy type, why adopted differs from planned."""
    out: dict[str, Mapping[str, object]] = {}
    for kind in (EnergyKind.FUEL, EnergyKind.GPU, EnergyKind.PCA):
        line = statement.lines.get(kind)
        if line is None:
            continue
        out[kind.value] = {
            "planned": line.planned,
            "adopted": line.adopted,
            "variance": line.variance,
            "basis": line.basis.value,
            "reading_ids": line.reading_ids,
            "unit": line.unit,
        }
    return out


class GroundEmissionsLedger:
    """Settles ramp energy evidence into versioned per-flight statements."""

    def __init__(self, registry: EmissionFactorRegistry | None = None) -> None:
        self.factors = registry or EmissionFactorRegistry()
        self.readings: dict[str, ConsumptionReading] = {}
        self._flight_readings: dict[str, list[str]] = defaultdict(list)
        self._statements: dict[str, list[EmissionsStatement]] = {}

    # ----- readings -----------------------------------------------------

    def record_planned(
        self,
        reading_id: str,
        flight_id: str,
        kind: EnergyKind,
        amount: Decimal,
        unit: str,
        at: datetime,
        temperature_c: Decimal | None = None,
    ) -> ConsumptionReading:
        return self.record_reading(
            reading_id, flight_id, kind, ReadingSource.PLANNED, amount, unit, at,
            temperature_c=temperature_c,
        )

    def record_meter_reading(
        self,
        reading_id: str,
        flight_id: str,
        kind: EnergyKind,
        amount: Decimal,
        unit: str,
        at: datetime,
        meter_id: str,
        temperature_c: Decimal | None = None,
    ) -> ConsumptionReading:
        return self.record_reading(
            reading_id, flight_id, kind, ReadingSource.METERED, amount, unit, at,
            meter_id=meter_id, temperature_c=temperature_c,
        )

    def record_late_reading(
        self,
        reading_id: str,
        flight_id: str,
        kind: EnergyKind,
        amount: Decimal,
        unit: str,
        at: datetime,
        meter_id: str | None = None,
        temperature_c: Decimal | None = None,
    ) -> ConsumptionReading:
        return self.record_reading(
            reading_id, flight_id, kind, ReadingSource.LATE, amount, unit, at,
            meter_id=meter_id, temperature_c=temperature_c,
        )

    def record_reading(
        self,
        reading_id: str,
        flight_id: str,
        kind: EnergyKind,
        source: ReadingSource,
        amount: Decimal,
        unit: str,
        at: datetime,
        meter_id: str | None = None,
        temperature_c: Decimal | None = None,
    ) -> ConsumptionReading:
        existing = self.readings.get(reading_id)
        if existing is not None:
            return existing  # idempotent replay, mirroring the fuel ledger
        reading = self._normalize_reading(
            reading_id, flight_id, kind, source, amount, unit, at,
            meter_id, temperature_c,
        )
        latest = self.latest_statement(flight_id)
        if latest is not None and latest.status is StatementStatus.CLOSED:
            raise StatementClosedError(flight_id, latest.revision)
        self._store(reading)
        if latest is not None and latest.status is StatementStatus.OPEN:
            # New evidence silently corrects the open statement as a revision.
            self._build_statement(
                flight_id,
                latest, latest.factors, utc(at), latest.revision,
                status=StatementStatus.OPEN, reason=latest.reason,
            )
        return reading

    # ----- statements ---------------------------------------------------

    def settle(
        self,
        flight_id: str,
        at: datetime,
        edition: int | None = None,
        reason: str = "",
    ) -> EmissionsStatement:
        moment = utc(at)
        latest = self.latest_statement(flight_id)
        if latest is not None and latest.status is StatementStatus.CLOSED:
            raise GroundEmissionsError(
                f"flight {flight_id} is closed; use amend() for new evidence"
            )
        if edition is not None:
            factors = self.factors.edition(edition)
        elif latest is not None:
            factors = latest.factors
        else:
            factors = self.factors.edition_in_effect(moment)
        if not self._flight_readings[flight_id]:
            raise GroundEmissionsError(f"no energy evidence for flight {flight_id}")
        if (
            latest is not None
            and latest.status is StatementStatus.OPEN
            and latest.factors.edition == factors.edition
            and latest.reading_ids() == frozenset(self._flight_readings[flight_id])
        ):
            return latest
        revision = latest.revision + 1 if latest is not None else 1
        return self._build_statement(
            flight_id,
            latest, factors, moment, revision,
            status=StatementStatus.OPEN, reason=reason,
        )

    def close(self, flight_id: str, at: datetime, reason: str = "") -> EmissionsStatement:
        latest = self.latest_statement(flight_id)
        if latest is None:
            raise GroundEmissionsError(f"no statement for flight {flight_id}")
        if latest.status is StatementStatus.CLOSED:
            return latest
        sealed = replace(
            latest,
            status=StatementStatus.CLOSED,
            closed_at=utc(at),
            reason=reason or latest.reason,
        )
        self._statements[flight_id][-1] = sealed
        return sealed

    def amend(
        self,
        flight_id: str,
        reason: str,
        at: datetime,
        readings: Iterable[ConsumptionReading] = (),
        edition: int | None = None,
    ) -> EmissionsStatement:
        """Attach late evidence to a sealed flight as a new version.

        The amendment is re-rated under the edition the sealed statement used,
        honouring the rules in force at the time; pass ``edition`` explicitly to
        produce a deliberate re-rated version.  In all cases prior revisions are
        retained.
        """
        latest = self.latest_statement(flight_id)
        if latest is None:
            raise GroundEmissionsError(f"no statement for flight {flight_id}")
        if latest.status is StatementStatus.OPEN:
            raise GroundEmissionsError(
                f"flight {flight_id} is still open; record readings directly"
            )
        if not reason:
            raise GroundEmissionsError("an amendment needs a reason")
        # Normalize and validate everything before committing, so a rejected
        # amendment cannot leave half of its evidence behind.
        to_store: list[ConsumptionReading] = []
        for reading in readings:
            if reading.flight_id != flight_id:
                raise GroundEmissionsError("reading belongs to another flight")
            existing = self.readings.get(reading.reading_id)
            if existing is not None:
                if existing.flight_id != flight_id:
                    raise GroundEmissionsError(
                        f"reading {reading.reading_id} already filed for another flight"
                    )
                continue
            to_store.append(
                self._normalize_reading(
                    reading.reading_id,
                    reading.flight_id,
                    reading.kind,
                    reading.source,
                    reading.amount,
                    reading.unit,
                    reading.recorded_at,
                    reading.meter_id,
                    reading.temperature_c,
                )
            )
        factors = self.factors.edition(edition) if edition is not None else latest.factors
        for normalized in to_store:
            self._store(normalized)
        if not to_store and factors.edition == latest.factors.edition:
            return latest
        moment = utc(at)
        return self._build_statement(
            flight_id,
            latest, factors, moment, latest.revision + 1,
            status=StatementStatus.CLOSED, reason=reason, closed_at=moment,
        )

    def latest_statement(self, flight_id: str) -> EmissionsStatement | None:
        revisions = self._statements.get(flight_id)
        return revisions[-1] if revisions else None

    def history(self, flight_id: str) -> tuple[EmissionsStatement, ...]:
        return tuple(self._statements.get(flight_id, ()))

    def flight_readings(self, flight_id: str) -> tuple[ConsumptionReading, ...]:
        return tuple(
            self.readings[rid] for rid in self._flight_readings.get(flight_id, ())
        )

    # ----- internals ----------------------------------------------------

    @staticmethod
    def _normalize_reading(
        reading_id: str,
        flight_id: str,
        kind: EnergyKind,
        source: ReadingSource,
        amount: Decimal,
        unit: str,
        at: datetime,
        meter_id: str | None,
        temperature_c: Decimal | None,
    ) -> ConsumptionReading:
        if not reading_id or not flight_id:
            raise GroundEmissionsError("reading and flight ids are required")
        kind = kind if isinstance(kind, EnergyKind) else EnergyKind(kind)
        source = source if isinstance(source, ReadingSource) else ReadingSource(source)
        value = Decimal(amount)
        if value <= 0:
            raise GroundEmissionsError("reading amount must advance (be positive)")
        normalized_unit = unit.upper()
        allowed = ("L", "KG") if kind is EnergyKind.FUEL else ("KWH",)
        if normalized_unit not in allowed:
            raise GroundEmissionsError(
                f"{kind.value} accepts {allowed}, got {normalized_unit!r}"
            )
        temp = Decimal(temperature_c) if temperature_c is not None else None
        if temp is not None and normalized_unit != "L":
            raise GroundEmissionsError("temperature only applies to metered litres")
        if source is ReadingSource.METERED and not meter_id:
            raise GroundEmissionsError("metered reading requires a meter id")
        return ConsumptionReading(
            reading_id,
            flight_id,
            kind,
            source,
            quant(value),
            normalized_unit,
            utc(at),
            meter_id,
            temp,
        )

    def _store(self, reading: ConsumptionReading) -> None:
        self.readings[reading.reading_id] = reading
        self._flight_readings[reading.flight_id].append(reading.reading_id)

    def _to_statement_unit(
        self, reading: ConsumptionReading, factors: EmissionFactors
    ) -> Decimal:
        if reading.kind is EnergyKind.FUEL:
            if reading.unit == "L":
                return factors.fuel_liters_to_kg(reading.amount, reading.temperature_c)
            return quant(reading.amount)
        return quant(reading.amount)

    def _lines(
        self, flight_id: str, factors: EmissionFactors
    ) -> dict[EnergyKind, EnergyLine]:
        grouped: dict[EnergyKind, list[ConsumptionReading]] = defaultdict(list)
        for rid in self._flight_readings[flight_id]:
            grouped[self.readings[rid].kind].append(self.readings[rid])
        lines: dict[EnergyKind, EnergyLine] = {}
        for kind in (EnergyKind.FUEL, EnergyKind.GPU, EnergyKind.PCA):
            items = grouped.get(kind)
            if not items:
                continue
            planned = Decimal("0")
            metered = Decimal("0")
            late: list[ConsumptionReading] = []
            for reading in items:
                value = self._to_statement_unit(reading, factors)
                if reading.source is ReadingSource.PLANNED:
                    planned += value
                elif reading.source is ReadingSource.METERED:
                    metered += value
                else:
                    late.append(reading)
            if late:
                winner = max(late, key=lambda r: (r.recorded_at, r.reading_id))
                late_value = self._to_statement_unit(winner, factors)
                adopted, basis = late_value, ReadingSource.LATE
            elif metered > 0:
                adopted, basis = metered, ReadingSource.METERED
            else:
                adopted, basis = planned, ReadingSource.PLANNED
            lines[kind] = EnergyLine(
                kind,
                "KG" if kind is EnergyKind.FUEL else "KWH",
                quant(planned),
                quant(metered),
                quant(late_value) if late else quant(0),
                quant(adopted),
                quant(adopted - planned),
                basis,
                tuple(sorted(r.reading_id for r in items)),
            )
        return lines

    def _build_statement(
        self,
        flight_id: str,
        previous: EmissionsStatement | None,
        factors: EmissionFactors,
        created_at: datetime,
        revision: int,
        *,
        status: StatementStatus,
        reason: str,
        closed_at: datetime | None = None,
    ) -> EmissionsStatement:
        lines = self._lines(flight_id, factors)
        fuel_line = lines.get(EnergyKind.FUEL)
        gpu_line = lines.get(EnergyKind.GPU)
        pca_line = lines.get(EnergyKind.PCA)
        fuel_kg = quant(fuel_line.adopted) if fuel_line else Decimal("0.000")
        gpu_kwh = quant(gpu_line.adopted) if gpu_line else Decimal("0.000")
        pca_kwh = quant(pca_line.adopted) if pca_line else Decimal("0.000")
        fuel_co2 = factors.fuel_co2_kg(fuel_kg)
        power_co2 = factors.electricity_co2_kg(quant(gpu_kwh + pca_kwh))
        statement = EmissionsStatement(
            f"GE-{flight_id}-R{revision}",
            flight_id,
            revision,
            status,
            factors,
            lines,
            fuel_kg,
            gpu_kwh,
            pca_kwh,
            fuel_co2,
            power_co2,
            quant(fuel_co2 + power_co2),
            utc(created_at),
            previous.revision if previous is not None else None,
            utc(closed_at) if closed_at else None,
            reason,
        )
        history = self._statements.setdefault(flight_id, [])
        if history and revision < history[-1].revision:
            raise GroundEmissionsError("cannot overwrite an existing revision")
        if revision <= len(history):
            history[revision - 1] = statement
        else:
            history.append(statement)
        return statement
