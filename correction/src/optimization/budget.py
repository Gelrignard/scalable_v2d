"""Cooperative wall-clock budget shared by bounded preprocessing stages."""
from time import perf_counter


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, seconds):
        if seconds <= 0:
            raise ValueError('budget must be positive')
        self.started = perf_counter()
        self.seconds = seconds

    @property
    def remaining(self):
        return max(0., self.seconds-(perf_counter()-self.started))

    def check(self):
        if self.remaining <= 0:
            raise BudgetExceeded('CPU correction budget exhausted')

    @property
    def elapsed(self):
        return perf_counter()-self.started
