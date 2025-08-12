import logging

import pandas as pd
from dateutil import parser
from datetime import date
import numpy as np

class InvalidCSVError(Exception):
    def __init__(self, message: str, original_exception: Exception | None = None):
        super().__init__(message)
        self.original_exception = original_exception


class InvalidDataError(Exception):
    def __init__(self, message: str, original_exception: Exception | None = None):
        super().__init__(message)
        self.original_exception = original_exception


class InvalidInputError(Exception):
    def __init__(self, message: str, original_exception: Exception | None = None):
        super().__init__(message)
        self.original_exception = original_exception


class IRRComputationError(Exception):
    def __init__(self, message: str, original_exception: Exception | None = None):
        super().__init__(message)
        self.original_exception = original_exception


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RedIQSimulator:
    def __init__(self):
        self.rentRoll = []
        self.operatingStatement = []
        self.standardizedRentRoll = []
        self.standardizedOperatingStatement = []
        self.comps = []
        self.anomalies = {}

    def load_data(self, rent_roll_path: str | None = None, operating_statement_path: str | None = None):
        """Load CSV data from the provided paths.

        Raises:
            InvalidCSVError: if a file cannot be read or required columns are missing.
        """
        if rent_roll_path:
            try:
                df_rr = pd.read_csv(rent_roll_path)
            except FileNotFoundError as e:
                logger.error("Rent roll file not found: %s", rent_roll_path)
                raise InvalidCSVError(f"Rent roll file not found: {rent_roll_path}", e) from e
            except pd.errors.ParserError as e:
                logger.error("Error parsing rent roll CSV: %s", e)
                raise InvalidCSVError("Error parsing rent roll CSV", e) from e
            required_rr = {"Unit", "Rent"}
            if not required_rr.issubset(df_rr.columns):
                missing = required_rr - set(df_rr.columns)
                raise InvalidCSVError(f"Rent roll missing required columns: {missing}")
            if df_rr.empty:
                raise InvalidCSVError("Rent roll CSV is empty")
            self.rentRoll = df_rr.to_dict(orient="records")

        if operating_statement_path:
            try:
                df_op = pd.read_csv(operating_statement_path)
            except FileNotFoundError as e:
                logger.error("Operating statement file not found: %s", operating_statement_path)
                raise InvalidCSVError(f"Operating statement file not found: {operating_statement_path}", e) from e
            except pd.errors.ParserError as e:
                logger.error("Error parsing operating statement CSV: %s", e)
                raise InvalidCSVError("Error parsing operating statement CSV", e) from e
            required_op = {"Category", "Amount", "Date"}
            if not required_op.issubset(df_op.columns):
                missing = required_op - set(df_op.columns)
                raise InvalidCSVError(f"Operating statement missing required columns: {missing}")
            if df_op.empty:
                raise InvalidCSVError("Operating statement CSV is empty")
            self.operatingStatement = df_op.to_dict(orient="records")

        if rent_roll_path or operating_statement_path:
            logger.info("Data loaded successfully.")

    def standardize_data(self):
        """Map raw column names to standard names and coerce types."""
        column_mapping = {
            "Unit": "unit_number",
            "Rent": "current_rent",
            "Lease Start": "lease_start",
            "Lease End": "lease_end",
            "Tenant": "tenant_name",
        }
        standardized_rent_roll: list[dict] = []
        for i, row in enumerate(self.rentRoll, start=1):
            new_row: dict = {}
            for key, value in row.items():
                new_key = column_mapping.get(key, key.lower().replace(" ", "_"))
                if new_key == "current_rent":
                    num = pd.to_numeric(value, errors="coerce")
                    if pd.isna(num) or num < 0:
                        raise InvalidDataError(
                            f"Invalid rent value '{value}' in row {i}"
                        )
                    new_row[new_key] = float(num)
                elif new_key in ["lease_start", "lease_end"]:
                    try:
                        if isinstance(value, str):
                            value = value.strip()
                        new_row[new_key] = parser.parse(value).date()
                    except (ValueError, parser.ParserError) as e:
                        raise InvalidDataError(
                            f"Invalid date '{value}' in row {i}", e
                        ) from e
                else:
                    new_row[new_key] = value
            standardized_rent_roll.append(new_row)

        op_column_mapping = {"Category": "category", "Amount": "amount", "Date": "date"}
        standardized_op_statement: list[dict] = []
        for i, row in enumerate(self.operatingStatement, start=1):
            new_row = {}
            for key, value in row.items():
                new_key = op_column_mapping.get(key, key.lower().replace(" ", "_"))
                if new_key == "amount":
                    num = pd.to_numeric(value, errors="coerce")
                    if pd.isna(num) or num < 0:
                        raise InvalidDataError(
                            f"Invalid amount value '{value}' in row {i}"
                        )
                    new_row[new_key] = float(num)
                elif new_key == "date":
                    try:
                        if isinstance(value, str):
                            value = value.strip()
                        new_row[new_key] = parser.parse(value).date()
                    except (ValueError, parser.ParserError) as e:
                        raise InvalidDataError(
                            f"Invalid date '{value}' in row {i}", e
                        ) from e
                else:
                    new_row[new_key] = value
            standardized_op_statement.append(new_row)

        self.standardizedRentRoll = standardized_rent_roll
        self.standardizedOperatingStatement = standardized_op_statement
        logger.info("Data standardized.")

    def detect_anomalies(self, threshold: float = 3.0):
        """Detect anomalies such as outlier rents and future-dated leases."""
        rents = [row.get("current_rent") for row in self.standardizedRentRoll if "current_rent" in row]
        if len(rents) >= 2:
            mean = np.mean(rents)
            std = np.std(rents)
            if std > 0:
                rent_anomalies = [
                    row
                    for row in self.standardizedRentRoll
                    if abs((row.get("current_rent") - mean) / std) > threshold
                ]
                self.anomalies["rent"] = rent_anomalies
                logger.info("Detected %d rent anomalies.", len(rent_anomalies))
            else:
                self.anomalies["rent"] = []
                logger.info("Standard deviation of rent is zero; no anomalies detected.")
        else:
            self.anomalies["rent"] = []
            logger.info("Not enough rent data to detect anomalies.")

        today = date.today()
        lease_anomalies = []
        skipped = []
        for row in self.standardizedRentRoll:
            start = row.get("lease_start")
            if isinstance(start, date):
                if start > today:
                    lease_anomalies.append(row)
            else:
                skipped.append(row)
        self.anomalies["lease_start"] = lease_anomalies
        if skipped:
            logger.warning("Skipped %d rows with missing lease start dates.", len(skipped))
            self.anomalies["lease_start_skipped"] = skipped
        logger.info("Detected %d future lease start anomalies.", len(lease_anomalies))

    def visualize_cash_flows(self):
        """Aggregate operating statement by year-month and log results."""
        monthly_cash: dict[tuple[int, int], float] = {}
        for row in self.standardizedOperatingStatement:
            d, amount = row.get("date"), row.get("amount", 0.0)
            if isinstance(d, date):
                key = (d.year, d.month)
                monthly_cash[key] = monthly_cash.get(key, 0.0) + amount
        for (y, m), val in sorted(monthly_cash.items()):
            logger.info("%d-%02d: %s", y, m, val)

    def build_comps(self, comp_data: dict):
        self.comps.append(comp_data)
        logger.info("Comp added.")

    def compare_expenses(self, current_expenses: float):
        if not self.comps:
            logger.warning("No comps available.")
            return {}
        avg = sum(c.get("expenses", 0.0) for c in self.comps) / len(self.comps)
        return {
            "current": current_expenses,
            "comp_avg": avg,
            "difference": current_expenses - avg,
        }

    def basic_valuation_model(
        self, cap_rate: float, noi: float, hold_period: int, growth_rate: float
    ) -> dict:
        """Basic valuation using a simplified cash-flow model.

        Args:
            cap_rate: Capitalization rate as a decimal fraction (>0).
            noi: Net operating income (>0).
            hold_period: Holding period in years (positive integer).
            growth_rate: Annual growth rate (> -1).
        """
        if cap_rate <= 0 or noi <= 0:
            raise InvalidInputError("cap_rate and noi must be positive")
        if not isinstance(hold_period, int) or hold_period <= 0:
            raise InvalidInputError("hold_period must be a positive integer")
        if growth_rate <= -1:
            raise InvalidInputError("growth_rate must be greater than -1")

        cash_flows: list[float] = []
        for t in range(1, hold_period + 1):
            cf = noi * (1 + growth_rate) ** t
            if cf < 0:
                raise InvalidInputError("Calculated cash flow is negative")
            cash_flows.append(round(cf, 2))
        terminal_value = round(cash_flows[-1] / cap_rate, 2)
        total_cf = cash_flows[:-1] + [cash_flows[-1] + terminal_value]
        all_cf = [-noi] + total_cf
        try:
            irr = self._calculate_irr(all_cf)
        except IRRComputationError as e:
            logger.error("IRR computation failed: %s", e)
            irr = None
        return {
            "projected_cash_flows": total_cf,
            "irr": irr,
            "terminal_value": terminal_value,
        }

    def _calculate_irr(self, cash_flows: list[float]) -> float:
        """Calculate IRR using a simple bisection method."""
        low, high, eps = -0.999999, 10.0, 1e-5
        try:
            npv_low = self._calculate_npv(low, cash_flows)
            npv_high = self._calculate_npv(high, cash_flows)
        except ValueError as e:
            raise IRRComputationError(str(e), e) from e
        if npv_low * npv_high > 0:
            raise IRRComputationError("IRR not bracketed in search interval")
        while high - low > eps:
            mid = (low + high) / 2
            npv = self._calculate_npv(mid, cash_flows)
            if npv > 0:
                low = mid
            else:
                high = mid
        return (low + high) / 2

    def _calculate_npv(self, rate: float, cash_flows: list[float]) -> float:
        if rate <= -1:
            raise ValueError("Rate must be greater than -1")
        return sum(cf / (1 + rate) ** t for t, cf in enumerate(cash_flows))

    def sensitivity_analysis(self, noi, cap_rates, growth_rates):
        results = []
        for cr in cap_rates:
            for gr in growth_rates:
                val = self.basic_valuation_model(cr, noi, 5, gr)
                results.append({
                    'cap_rate': cr,
                    'growth_rate': gr,
                    'value': val['terminal_value']
                })
        return results
