import unittest
from datetime import datetime, timezone
from decimal import Decimal

from rampflow.ground_emissions import (
    EnergyReading,
    EnergyType,
    GroundEmissionsLedger,
    ReadingSource,
    RULE_FUEL_DENSITY,
    RULE_FUEL_KWH_PER_KG,
    RULE_FUEL_CO2_PER_KG,
    RULE_POWER_CO2_PER_KWH,
    RULE_AC_CO2_PER_KWH,
    SettlementSealedError,
    SettlementStatus,
)
from rampflow.workflows.control import RampWorkflowControl


UTC = timezone.utc
D = lambda h, m=0: datetime(2026, 1, 1, h, m, tzinfo=UTC)

FACTORS = {
    RULE_FUEL_DENSITY: "0.80",
    RULE_FUEL_KWH_PER_KG: "12.0",
    RULE_FUEL_CO2_PER_KG: "3.15",
    RULE_POWER_CO2_PER_KWH: "0.45",
    RULE_AC_CO2_PER_KWH: "0.55",
}


def _line(settlement, energy: EnergyType):
    return next(line for line in settlement.lines if line.energy is energy)


class GroundEmissionsTests(unittest.TestCase):
    def setUp(self):
        self.ledger = GroundEmissionsLedger()
        self.ledger.publish_rules(FACTORS, D(9), "winter factors")

    def _feed_three_sheets(self, flight="F1", fuel="1000"):
        add = self.ledger.add_reading
        add("p-fuel", flight, EnergyType.FUEL, ReadingSource.PLANNED, fuel, "L", D(8), D(8))
        add("m-fuel", flight, EnergyType.FUEL, ReadingSource.METER, "1100", "L", D(10, 10), D(10, 20))
        add("p-gpu", flight, EnergyType.POWER, ReadingSource.PLANNED, "90", "kWh", D(8), D(8))
        add("m-gpu", flight, EnergyType.POWER, ReadingSource.METER, "84", "kWh", D(10, 10), D(10, 20))
        add("p-ac", flight, EnergyType.AC, ReadingSource.PLANNED, "60", "kWh", D(8), D(8))
        add("m-ac", flight, EnergyType.AC, ReadingSource.METER, "58", "kWh", D(10, 10), D(10, 20))

    def test_three_sheets_settled_per_flight(self):
        self._feed_three_sheets()
        settlement = self.ledger.settle("F1", D(11))
        self.assertEqual(settlement.status, SettlementStatus.OPEN)
        self.assertEqual(len(settlement.lines), 3)

        fuel = _line(settlement, EnergyType.FUEL)
        # Meter reading is authoritative while the plan stays on record.
        self.assertEqual(fuel.authority, ReadingSource.METER)
        self.assertEqual(
            {s.source for s in fuel.sources},
            {ReadingSource.PLANNED, ReadingSource.METER},
        )
        self.assertEqual(fuel.mass_kg, Decimal("880.000"))
        self.assertEqual(fuel.energy_kwh, Decimal("10560.000"))
        self.assertEqual(fuel.co2e_kg, Decimal("2772.000"))

        gpu = _line(settlement, EnergyType.POWER)
        self.assertEqual(gpu.co2e_kg, Decimal("37.800"))
        self.assertEqual(gpu.variance_vs_planned_kwh, Decimal("-6.000"))

        total_co2 = fuel.co2e_kg + gpu.co2e_kg + _line(settlement, EnergyType.AC).co2e_kg
        self.assertEqual(settlement.total_co2e_kg, total_co2)

    def test_supplementary_reading_corrects_open_flight(self):
        self._feed_three_sheets()
        before = self.ledger.settle("F1", D(11))
        self.ledger.add_reading(
            "s-fuel", "F1", EnergyType.FUEL, ReadingSource.SUPPLEMENT,
            "1080", "L", D(10, 40), D(11, 5),
        )
        after = self.ledger.current("F1")
        fuel = _line(after, EnergyType.FUEL)
        self.assertEqual(fuel.authority, ReadingSource.SUPPLEMENT)
        self.assertEqual(fuel.figure.reading_id, "s-fuel")
        # The draft stays a single version until sealed.
        self.assertEqual(after.version, before.version)
        self.assertIn("meter", fuel.selection_note)
        # Plan, meter and supplement are all retained.
        self.assertEqual(len(fuel.sources), 3)

    def test_factors_are_pinned_and_new_rules_do_not_rewrite_seals(self):
        self._feed_three_sheets()
        sealed = self.ledger.seal("F1", D(12))
        self.assertEqual(sealed.digest, self.ledger.current("F1").digest)
        old_co2 = sealed.total_co2e_kg
        self.assertTrue(sealed.digest)

        self.ledger.publish_rules(
            {**FACTORS, RULE_FUEL_CO2_PER_KG: "3.50"}, D(13), "new factor"
        )
        # Sealed version is untouched.
        self.assertEqual(self.ledger.current("F1").total_co2e_kg, old_co2)
        self.assertTrue(self.ledger.verify("F1"))

    def test_reading_after_seal_is_rejected_but_revision_remains(self):
        self._feed_three_sheets()
        sealed = self.ledger.seal("F1", D(12))
        with self.assertRaises(SettlementSealedError):
            self.ledger.add_reading(
                "late", "F1", EnergyType.FUEL, ReadingSource.SUPPLEMENT,
                "1090", "L", D(13), D(13, 5),
            )

        revision_reading = EnergyReading(
            "late", "F1", EnergyType.FUEL, ReadingSource.SUPPLEMENT,
            Decimal("1090"), "L", D(13), D(13, 5),
        )
        draft = self.ledger.revise("F1", [revision_reading], D(13, 10))
        self.assertEqual(draft.status, SettlementStatus.OPEN)
        self.assertEqual(draft.version, sealed.version + 1)
        self.assertEqual(draft.supersedes, sealed.version)
        self.assertEqual(draft.amended_by_readings, ("late",))
        resealed = self.ledger.seal("F1", D(13, 20))
        self.assertEqual(resealed.version, 2)

        history = self.ledger.history("F1")
        self.assertEqual([s.version for s in history], [1, 2])
        first = self.ledger.sealed_version("F1", 1)
        self.assertEqual(first.digest, sealed.digest)
        self.assertEqual(first.status, SettlementStatus.SEALED)
        self.assertNotEqual(first.total_co2e_kg, resealed.total_co2e_kg)
        self.assertTrue(self.ledger.verify("F1"))

    def test_revision_can_pin_current_rule_version(self):
        self._feed_three_sheets()
        self.ledger.seal("F1", D(12))
        self.ledger.publish_rules(
            {**FACTORS, RULE_AC_CO2_PER_KWH: "0.70"}, D(13), "grid mix update"
        )
        reading = EnergyReading(
            "late", "F1", EnergyType.AC, ReadingSource.SUPPLEMENT,
            Decimal("61"), "kWh", D(13), D(13, 5),
        )
        draft = self.ledger.revise("F1", [reading], D(13, 10))
        self.assertEqual(draft.rule_version, 2)
        self.assertEqual(
            draft.rules_snapshot[RULE_AC_CO2_PER_KWH], Decimal("0.70")
        )

    def test_revision_requires_sealed_flight_and_new_readings(self):
        self._feed_three_sheets()
        self.ledger.settle("F1", D(11))
        with self.assertRaises(Exception):
            self.ledger.revise("F1", [], D(12))
        self.ledger.seal("F1", D(12))
        with self.assertRaises(Exception):
            self.ledger.revise("F1", [], D(13))

    def test_plan_only_flight_is_flagged_in_note(self):
        self.ledger.add_reading(
            "p1", "F2", EnergyType.POWER, ReadingSource.PLANNED,
            "10", "kWh", D(8), D(8),
        )
        settlement = self.ledger.settle("F2", D(11))
        line = _line(settlement, EnergyType.POWER)
        self.assertEqual(line.authority, ReadingSource.PLANNED)
        self.assertEqual(line.variance_vs_planned_kwh, Decimal("0.000"))
        self.assertIn("no actual reading", line.selection_note)

    def test_invalid_factor_set_and_units_rejected(self):
        ledger = GroundEmissionsLedger()
        with self.assertRaises(Exception):
            ledger.publish_rules({RULE_FUEL_DENSITY: "0.8"}, D(9))
        ledger.publish_rules(FACTORS, D(9))
        with self.assertRaises(Exception):
            ledger.add_reading(
                "bad", "F9", EnergyType.POWER, ReadingSource.METER,
                "5", "L", D(10), D(10),
            )
        with self.assertRaises(Exception):
            ledger.settle("F9", D(10))

    def test_workflow_event_only_after_seal_succeeds(self):
        self._feed_three_sheets()
        control = RampWorkflowControl()
        receipt = control.seal_ground_emissions(
            "op-seal-1", "F1", D(12),
            lambda: self.ledger.seal("F1", D(12)),
        )
        self.assertEqual(control.stream.head, 1)
        self.assertEqual(receipt.event.topic, "ground.emissions.settlement.sealed")
        # Idempotent replay returns the same receipt.
        again = control.seal_ground_emissions(
            "op-seal-1", "F1", D(12),
            lambda: self.ledger.seal("F1", D(12)),
        )
        self.assertIs(again, receipt)


if __name__ == "__main__":
    unittest.main()
