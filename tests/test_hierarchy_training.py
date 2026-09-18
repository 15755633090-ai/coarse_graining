"""Training-orchestration contracts without starting a formal experiment."""
import unittest

import torch

from scripts.experiments.run_hierarchy_lipo import GeneratorBucketBatchSampler


class HierarchyTrainingTests(unittest.TestCase):
    def test_bucket_sampler_is_complete_and_resume_reproducible(self):
        counts = [30, 2, 18, 7, 42, 10, 25, 4, 33, 12, 20]
        generator = torch.Generator().manual_seed(91)
        state = generator.get_state().clone()
        first = list(GeneratorBucketBatchSampler(counts, 3, generator, True, multiplier=2))
        self.assertEqual(
            sorted(index for batch in first for index in batch), list(range(len(counts))),
        )
        generator.set_state(state)
        resumed = list(GeneratorBucketBatchSampler(counts, 3, generator, True, multiplier=2))
        self.assertEqual(first, resumed)
        ordered = list(GeneratorBucketBatchSampler(
            counts, 3, torch.Generator().manual_seed(0), False, multiplier=20,
        ))
        flattened = [index for batch in ordered for index in batch]
        self.assertEqual(flattened, sorted(range(len(counts)), key=counts.__getitem__))


if __name__ == "__main__":
    unittest.main()
