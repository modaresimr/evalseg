
from . import AggregatorABS


class MultiCaseAggregator(AggregatorABS):

    def __init__(self, base_aggregator):
        self.base_aggregator = base_aggregator

    def add(self, metric_out):
        self.base_aggregator.add(metric_out)

    def get(self):
        return self.base_aggregator.get()
