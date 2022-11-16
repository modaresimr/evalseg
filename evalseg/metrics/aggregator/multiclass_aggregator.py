
from . import AggregatorABS


class MultiClassAggregator(AggregatorABS):

    def __init__(self, base_aggregator, num_class):
        self.base_aggregator = base_aggregator
        self.num_class = num_class

    def add(self, metric_out):
        for c in range(self.num_class):
            self.base_aggregator.add(metric_out[c])

    def get(self):
        return self.base_aggregator.get()
