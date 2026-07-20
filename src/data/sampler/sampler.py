import random

class DatasetSampler:
    """
    Handles reproducible dataset subsampling.
    """

    def __init__(self, config: dict):
        self.sample_size = config['data']['max_size']
        self.seed = config['seed']

    def sample(self, dataset):
        sample_size = min(self.sample_size, len(dataset))
        indices = random.Random(self.seed).sample(range(len(dataset)), sample_size)

        if hasattr(dataset, "select"):
            return dataset.select(indices)

        return [dataset[index] for index in indices]
