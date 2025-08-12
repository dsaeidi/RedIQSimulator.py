import pandas as pd
from datetime import datetime, date
import numpy as np

class RedIQSimulator:
    def __init__(self):
        self.rentRoll = []
        self.operatingStatement = []
        self.standardizedRentRoll = []
        self.standardizedOperatingStatement = []
        self.comps = []
        self.anomalies = {}

    def load_data(self, rent_roll_path=None, operating_statement_path=None):
        if rent_roll_path:
            self.rentRoll = pd.read_csv(rent_roll_path).to_dict(orient='records')
        if operating_statement_path:
            self.operatingStatement = pd.read_csv(operating_statement_path).to_dict(orient='records')
        print("Data loaded successfully.")

    def standardize_data(self):
        # Map raw column names to standard names and coerce types
        column_mapping = {
            'Unit': 'unit_number',
            'Rent': 'current_rent',
            'Lease Start': 'lease_start',
            'Lease End': 'lease_end',
            'Tenant': 'tenant_name'
        }
        standardized_rent_roll = []
        for row in self.rentRoll:
            new_row = {}
            for key, value in row.items():
                new_key = column_mapping.get(key, key.lower().replace(' ', '_'))
                if new_key == 'current_rent':
                    try:
                        new_row[new_key] = float(value)
                    except (ValueError, TypeError):
                        new_row[new_key] = 0.0
                elif new_key in ['lease_start', 'lease_end']:
                    try:
                        new_row[new_key] = datetime.strptime(value, '%Y-%m-%d').date()
                    except (ValueError, TypeError):
                        new_row[new_key] = None
                else:
                    new_row[new_key] = value
            standardized_rent_roll.append(new_row)

        op_column_mapping = {
            'Category': 'category',
            'Amount': 'amount',
            'Date': 'date'
        }
        standardized_op_statement = []
        for row in self.operatingStatement:
            new_row = {}
            for key, value in row.items():
                new_key = op_column_mapping.get(key, key.lower().replace(' ', '_'))
                if new_key == 'amount':
                    try:
                        new_row[new_key] = float(value)
                    except (ValueError, TypeError):
                        new_row[new_key] = 0.0
                elif new_key == 'date':
                    try:
                        new_row[new_key] = datetime.strptime(value, '%Y-%m-%d').date()
                    except (ValueError, TypeError):
                        new_row[new_key] = None
                else:
                    new_row[new_key] = value
            standardized_op_statement.append(new_row)

        self.standardizedRentRoll = standardized_rent_roll
        self.standardizedOperatingStatement = standardized_op_statement
        print("Data standardized.")

    def detect_anomalies(self, threshold=3.0):
        # Rent anomalies via z‑score
        rents = [row.get('current_rent', 0.0) for row in self.standardizedRentRoll]
        if rents:
            mean = np.mean(rents)
            std = np.std(rents)
            rent_anomalies = [
                row for row in self.standardizedRentRoll
                if std > 0 and abs((row.get('current_rent', 0.0) - mean) / std) > threshold
            ]
            self.anomalies['rent'] = rent_anomalies
            print(f"Detected {len(rent_anomalies)} rent anomalies.")
        # Future lease start anomalies
        today = date.today()
        lease_anomalies = [
            row for row in self.standardizedRentRoll
            if isinstance(row.get('lease_start'), date) and row['lease_start'] > today
        ]
        self.anomalies['lease_start'] = lease_anomalies
        print(f"Detected {len(lease_anomalies)} future lease start anomalies.")

    def visualize_cash_flows(self):
        # Aggregate operating statement by year-month and print
        monthly_cash = {}
        for row in self.standardizedOperatingStatement:
            d, amount = row.get('date'), row.get('amount', 0.0)
            if isinstance(d, date):
                key = (d.year, d.month)
                monthly_cash[key] = monthly_cash.get(key, 0.0) + amount
        print("Monthly Cash Flows:")
        for (y, m), val in sorted(monthly_cash.items()):
            print(f"{y}-{m:02d}: {val}")

    def build_comps(self, comp_data):
        self.comps.append(comp_data)
        print("Comp added.")

    def compare_expenses(self, current_expenses):
        if not self.comps:
            print("No comps available.")
            return {}
        avg = sum(c.get('expenses', 0.0) for c in self.comps) / len(self.comps)
        return {
            'current': current_expenses,
            'comp_avg': avg,
            'difference': current_expenses - avg
        }

    def basic_valuation_model(self, cap_rate, noi, hold_period, growth_rate):
        cash_flows = [noi * (1 + growth_rate) ** t for t in range(1, hold_period + 1)]
        terminal_value = cash_flows[-1] / cap_rate
        total_cf = cash_flows[:-1] + [cash_flows[-1] + terminal_value]
        all_cf = [-noi] + total_cf
        irr = self._calculate_irr(all_cf)
        return {
            'projected_cash_flows': total_cf,
            'irr': irr,
            'terminal_value': terminal_value
        }

    def _calculate_irr(self, cash_flows):
        # Simple bisection method
        low, high, eps = -1.0, 10.0, 1e-5
        while high - low > eps:
            mid = (low + high) / 2
            npv = self._calculate_npv(mid, cash_flows)
            if npv > 0:
                low = mid
            else:
                high = mid
        return (low + high) / 2

    def _calculate_npv(self, rate, cash_flows):
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
