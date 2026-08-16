"""Strategy package: pluggable per-account trading strategies.

Each strategy implements pure decision logic (no broker calls) so it is
unit-testable; the runners (decide.py, daytrade_runner.py) handle execution
and account routing via accounts.py.
"""
