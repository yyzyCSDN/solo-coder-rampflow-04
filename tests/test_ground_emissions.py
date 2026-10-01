import unittest
from datetime import datetime, timezone
from decimal import Decimal

from rampflow.ground_emissions import (
    EnergyKind,
    ReadingSource,
    StatementStatus,
    EmissionFactorRegistry,
    GroundEmissionsLedger,
    GroundEmissionsError,
    StatementClosedError,
    ConsumptionReading,
    variance_breakdown,
)
from rampflow.workflows.control import RampWorkflowControl

UTC = timezone.utc
D = lambda h, m=0: datetime(2026, 6, 1, h, m, tzinfo=UTC)

FUEL = EnergyKind.FUEL
GPU = EnergyKind.GPU
PCA = EnergyKind.PCA


def ledger() -> GroundEmissionsLedger:
    registry = EmissionFactorRegistry()
    registry.publish_standard(D(0))
    return GroundEmissionsLedger(registry)


class GroundEmissionsTests(unittest.TestCase):
    def test_three_sources_settled_per_flight_with_variance_explained(self):
        book = ledger()
        # Plan says 1000 L fuel, 50 kWh GPU, 30 kWh PCA.
        book.record_planned("P-F", "F1", FUEL, Decimal("1000"), "L", D(6))
        book.record_planned("P-G", "F1", GPU, Decimal("50"), "kWh", D(6))
        book.record_planned("P-A", "F1", PCA, Decimal("30"), "kWh", D(6))
        # Equipment meters disagree with the plan.
        book.record_meter_reading("M-F", "F1", FUEL, Decimal("1100"), "L", D(7),
                                  meter_id="pump-1", temperature_c=Decimal("15"))
        book.record_meter_reading("M-G", "F1", GPU, Decimal("46"), "kWh", D(7),
                                  meter_id="gpu-9")
        book.record_meter_reading("M-A", "F1", PCA, Decimal("33"), "kWh", D(7),
                                  meter_id="pca-2")
        statement = book.settle("F1", D(8))
        self.assertEqual(statement.status, StatementStatus.OPEN)
        fuel = statement.lines[FUEL]
        self.assertEqual((fuel.planned, fuel.metered, fuel.adopted),
                         (Decimal("800.000"), Decimal("880.000"), Decimal("880.000")))
        self.assertEqual(fuel.variance, Decimal("80.000"))
        self.assertIs(fuel.basis, ReadingSource.METERED)
        self.assertEqual(statement.gpu_kwh, Decimal("46.000"))
        self.assertEqual(statement.pca_kwh, Decimal("33.000"))
        self.assertEqual(statement.fuel_kg, Decimal("880.000"))
        # 880 kg fuel * 3.16 + (46+33) kWh * 0.42
        self.assertEqual(statement.fuel_co2_kg, Decimal("2780.800"))
        self.assertEqual(statement.electricity_co2_kg, Decimal("33.180"))
        self.assertEqual(statement.total_co2_kg, Decimal("2813.980"))
        # The difference is attributable to concrete evidence, not a mystery.
        explain = variance_breakdown(statement)
        self.assertEqual(explain["gpu"]["basis"], "metered")
        self.assertEqual(explain["gpu"]["variance"], Decimal("-4.000"))
        self.assertEqual(explain["fuel"]["reading_ids"], ("M-F", "P-F"))

    def test_new_reading_revises_open_statement_in_place(self):
        book = ledger()
        book.record_planned("P1", "F2", GPU, Decimal("50"), "kWh", D(6))
        first = book.settle("F2", D(7))
        self.assertEqual(first.revision, 1)
        # A meter arriving after the first settlement corrects the same revision...
        book.record_meter_reading("M1", "F2", GPU, Decimal("48"), "kWh", D(8),
                                  meter_id="gpu-9")
        self.assertEqual(book.latest_statement("F2").revision, 1)
        self.assertEqual(book.latest_statement("F2").gpu_kwh, Decimal("48.000"))
        # ...but a deliberate re-settle under another edition opens revision 2.
        book.factors.publish_standard(
            D(9), electricity_co2_kg_per_kwh=Decimal("0.50"), note="grid update"
        )
        revised = book.settle("F2", D(10), edition=2)
        self.assertEqual(revised.revision, 2)
        self.assertEqual(revised.supersedes, 1)
        self.assertEqual(revised.electricity_co2_kg, Decimal("24.000"))
        self.assertEqual(len(book.history("F2")), 2)

    def test_closed_statement_rejects_late_reading_but_accepts_amendment(self):
        book = ledger()
        book.record_planned("P1", "F3", GPU, Decimal("50"), "kWh", D(6))
        sealed = book.settle("F3", D(7))
        sealed = book.close("F3", D(9), "month-end books closed")
        with self.assertRaises(StatementClosedError):
            book.record_late_reading("L1", "F3", GPU, Decimal("47"), "kWh", D(12))
        # Nothing leaked from the refused reading.
        self.assertNotIn("L1", book.readings)
        self.assertEqual(book.latest_statement("F3"), sealed)
        amendment = book.amend(
            "F3", "backfilled GPU meter from maintenance log", D(13),
            readings=[
                ConsumptionReading("L1", "F3", GPU, ReadingSource.LATE,
                                   Decimal("47"), "kWh", D(12), meter_id="gpu-9"),
            ],
        )
        self.assertEqual(amendment.revision, 2)
        self.assertIs(amendment.status, StatementStatus.CLOSED)
        self.assertEqual(amendment.gpu_kwh, Decimal("47.000"))
        self.assertIs(amendment.lines[GPU].basis, ReadingSource.LATE)
        # The sealed original is untouched and still queryable.
        original = book.history("F3")[0]
        self.assertEqual(original.gpu_kwh, Decimal("50.000"))
        self.assertIs(original.status, StatementStatus.CLOSED)
        self.assertIsNotNone(original.closed_at)

    def test_closed_statement_remembers_its_conversion_rules(self):
        registry = EmissionFactorRegistry()
        registry.publish_standard(D(0))
        book = GroundEmissionsLedger(registry)
        book.record_planned("P1", "F4", FUEL, Decimal("1000"), "L", D(6))
        statement = book.settle("F4", D(6))
        book.close("F4", D(7))
        old_total = statement.total_co2_kg
        # A new factor edition is published later; the sealed statement is
        # immune, but an amendment can be deliberately re-rated.
        registry.publish_standard(
            D(20), fuel_co2_kg_per_kg=Decimal("3.10"),
            electricity_co2_kg_per_kwh=Decimal("0.10"), note="new grid",
        )
        self.assertEqual(book.latest_statement("F4").total_co2_kg, old_total)
        amendment = book.amend("F4", "re-rate under published 2026 grid", D(21),
                               edition=2)
        self.assertEqual(amendment.factors.edition, 2)
        self.assertEqual(amendment.fuel_co2_kg, Decimal("2480.000"))
        self.assertEqual(book.history("F4")[0].factors.edition, 1)

    def test_fuel_temperature_and_unit_conversion(self):
        registry = EmissionFactorRegistry()
        factors = registry.publish_standard(D(0))
        # 1000 L dispensed at 30C expands by 0.00095 * 15 => fewer standard litres.
        mass = factors.fuel_liters_to_kg(Decimal("1000"), Decimal("30"))
        self.assertLess(mass, Decimal("800.000"))
        self.assertEqual(
            mass,
            Decimal(Decimal("1000")
                    / (Decimal("1") + Decimal("0.00095") * 15)
                    * Decimal("0.8")).quantize(Decimal("0.001")),
        )
        # Fuel already weighed in kg is never temperature-corrected.
        book = GroundEmissionsLedger(registry)
        book.record_planned("P1", "F5", FUEL, Decimal("800"), "kg", D(6))
        statement = book.settle("F5", D(6))
        self.assertEqual(statement.fuel_kg, Decimal("800.000"))

    def test_late_reading_overrides_meter_plan_chain(self):
        book = ledger()
        book.record_planned("P1", "F6", PCA, Decimal("30"), "kWh", D(6))
        book.record_meter_reading("M1", "F6", PCA, Decimal("34"), "kWh", D(7),
                                  meter_id="pca-2")
        # Manual correction after the meter download wins.
        book.record_late_reading("L0", "F6", PCA, Decimal("31"), "kWh", D(10),
                                 meter_id="pca-2")
        statement = book.settle("F6", D(11))
        self.assertIs(statement.lines[PCA].basis, ReadingSource.LATE)
        self.assertEqual(statement.lines[PCA].late, Decimal("31.000"))
        self.assertEqual(statement.pca_kwh, Decimal("31.000"))

    def test_reading_idempotency_and_validation(self):
        book = ledger()
        first = book.record_planned("P1", "F7", GPU, Decimal("50"), "kWh", D(6))
        again = book.record_planned("P1", "F7", GPU, Decimal("99"), "kWh", D(6))
        self.assertIs(first, again)
        self.assertEqual(len(book.flight_readings("F7")), 1)
        with self.assertRaises(GroundEmissionsError):
            book.record_planned("P2", "F7", GPU, Decimal("-5"), "kWh", D(6))
        with self.assertRaises(GroundEmissionsError):
            book.record_planned("P3", "F7", GPU, Decimal("50"), "MJ", D(6))
        with self.assertRaises(GroundEmissionsError):
            book.record_planned("P4", "F7", FUEL, Decimal("800"), "kg", D(6),
                                temperature_c=Decimal("20"))
        # No statement can be settled without any evidence.
        with self.assertRaises(GroundEmissionsError):
            book.settle("GHOST", D(8))

    def test_settle_requires_an_effective_edition(self):
        book = GroundEmissionsLedger(EmissionFactorRegistry())
        book.record_planned("P1", "F8", GPU, Decimal("50"), "kWh", D(6))
        with self.assertRaises(GroundEmissionsError):
            book.settle("F8", D(6))
        book.factors.publish_standard(D(0))
        self.assertEqual(book.settle("F8", D(6)).total_co2_kg, Decimal("21.000"))

    def test_amendment_rejects_foreign_or_partial_evidence(self):
        book = ledger()
        book.record_planned("P1", "F9", GPU, Decimal("50"), "kWh", D(6))
        book.settle("F9", D(7))
        book.close("F9", D(8))
        # A reading belonging to another flight cannot be smuggled in.
        with self.assertRaises(GroundEmissionsError):
            book.amend("F9", "wrong flight", D(9), readings=[
                ConsumptionReading("X1", "OTHER", GPU, ReadingSource.LATE,
                                   Decimal("10"), "kWh", D(9)),
            ])
        self.assertNotIn("X1", book.readings)
        with self.assertRaises(GroundEmissionsError):
            book.amend("F9", "", D(9))
        # Re-running the same amendment is a no-op, not another revision.
        reading = ConsumptionReading("L1", "F9", GPU, ReadingSource.LATE,
                                     Decimal("49"), "kWh", D(9))
        once = book.amend("F9", "manual meter", D(10), readings=[reading])
        twice = book.amend("F9", "manual meter", D(11), readings=[reading])
        self.assertEqual(once, twice)
        self.assertEqual(len(book.history("F9")), 2)

    def test_open_flight_cannot_be_amended(self):
        book = ledger()
        book.record_planned("P1", "F10", GPU, Decimal("50"), "kWh", D(6))
        book.settle("F10", D(7))
        with self.assertRaises(GroundEmissionsError):
            book.amend("F10", "too early", D(8))


class GroundEmissionsWorkflowTests(unittest.TestCase):
    def test_settlement_event_only_after_domain_succeeds(self):
        control = RampWorkflowControl()
        book = ledger()
        book.record_planned("P1", "F11", GPU, Decimal("50"), "kWh", D(6))

        def settle():
            statement = book.settle("F11", D(7))
            return book.close("F11", D(8)) and statement

        receipt = control.settle_ground_emissions("op-ge-1", "F11", D(8), settle)
        self.assertEqual(receipt.event.topic, "ground.emissions.statement.settled")
        self.assertEqual(receipt.event.subject, "F11")
        self.assertEqual(receipt.result.total_co2_kg, Decimal("21.000"))
        # Replay returns the same receipt without a second event.
        again = control.settle_ground_emissions("op-ge-1", "F11", D(8), settle)
        self.assertIs(again, receipt)
        self.assertEqual(control.stream.head, 1)


if __name__ == "__main__":
    unittest.main()
