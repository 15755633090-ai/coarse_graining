from __future__ import annotations

import unittest
import random

import torch

from multiscale_tokenizer.partition import (
    derive_partition_seed,
    partition_graph,
    partition_molecule,
)
from multiscale_tokenizer.training import _partition_seeds


def path_graph(num_nodes: int) -> list[list[int]]:
    adjacency = [[] for _ in range(num_nodes)]
    for node in range(num_nodes - 1):
        adjacency[node].append(node + 1)
        adjacency[node + 1].append(node)
    return adjacency


class PartitionTests(unittest.TestCase):
    def test_partition_is_exclusive_and_complete(self) -> None:
        adjacency = path_graph(30)
        partition = partition_graph(adjacency, seed=7)
        partition.validate(30)
        self.assertTrue(partition.q_mask.any())
        self.assertTrue((partition.levels == 4).any())

    def test_level4_iterates_until_a_long_graph_is_closed(self) -> None:
        partition = partition_graph(path_graph(60), seed=11)
        self.assertGreaterEqual(int(partition.stats["level4_rounds"]), 3)
        self.assertEqual(int(partition.stats["level4_nodes"]), 60 - int(partition.stats["q_size"]) - int(partition.stats["level2_nodes"]) - int(partition.stats["level3_nodes"]))

    def test_local_residual_does_not_merge_two_ends(self) -> None:
        partition = partition_graph(path_graph(10), seed=3)
        residuals = [
            partition.members[index].tolist()
            for index in range(len(partition.members))
            if bool(partition.residual[index])
        ]
        self.assertTrue(residuals)
        self.assertTrue(all(len(members) < 10 for members in residuals))

    def test_same_seed_is_reproducible_and_other_seeds_can_differ(self) -> None:
        adjacency = path_graph(20)
        first = partition_graph(adjacency, seed=13)
        second = partition_graph(adjacency, seed=13)
        third = partition_graph(adjacency, seed=14)
        self.assertTrue(first.q_mask.equal(second.q_mask))
        self.assertTrue(first.levels.equal(second.levels))
        signatures = [
            tuple(members.tolist()) for members in first.members
        ], [
            tuple(members.tolist()) for members in third.members
        ]
        self.assertNotEqual(signatures[0], signatures[1])

    def test_disconnected_components_are_tokenized_locally(self) -> None:
        adjacency = [
            [1], [0], [3], [2], [5], [4],
        ]
        partition = partition_graph(adjacency, seed=5)
        partition.validate(6)
        self.assertEqual(int(partition.stats["num_components"]), 3)
        self.assertEqual(int(partition.stats["q_size"]), 6)

    def test_partition_seed_stream_is_explicit(self) -> None:
        self.assertNotEqual(
            derive_partition_seed(11, 3, 0),
            derive_partition_seed(12, 3, 0),
        )
        self.assertNotEqual(
            derive_partition_seed(11, 3, 0),
            derive_partition_seed(11, 3, 1),
        )

    def test_train_redraws_by_epoch_while_eval_stays_fixed(self) -> None:
        sample_ids = torch.tensor([3, 4], dtype=torch.long)
        train_epoch_zero = _partition_seeds(
            sample_ids, base_seed=17, epoch=0, training=True,
        )
        train_epoch_one = _partition_seeds(
            sample_ids, base_seed=17, epoch=1, training=True,
        )
        validation_epoch_five = _partition_seeds(
            sample_ids, base_seed=17, epoch=5, training=False,
        )
        self.assertNotEqual(train_epoch_zero, train_epoch_one)
        self.assertEqual(train_epoch_zero, validation_epoch_five)

    def test_molecule_mapping_preserves_residual_center_marker(self) -> None:
        nodes = 16
        bonds = torch.zeros((nodes, nodes), dtype=torch.long)
        for left in range(1, 14):
            right = left + 1
            bonds[left, right] = bonds[right, left] = 1
        mask = torch.zeros(nodes, dtype=torch.bool)
        mask[1:15] = True
        partition = partition_molecule(bonds, node_mask=mask, seed=9)
        residual = partition.residual & partition.centers.eq(-1)
        standard = ~partition.residual
        self.assertTrue(residual.any())
        self.assertTrue(partition.centers[standard].ge(0).all())
        self.assertTrue(partition.centers[standard].lt(nodes).all())

    def test_molecule_partition_excludes_padding_nodes(self) -> None:
        nodes = 40
        bonds = torch.zeros((nodes, nodes), dtype=torch.long)
        for left in range(2, 14):
            right = left + 1
            bonds[left, right] = bonds[right, left] = 1
        mask = torch.zeros(nodes, dtype=torch.bool)
        mask[2:15] = True
        partition = partition_molecule(
            bonds,
            node_mask=mask,
            seed=4,
            include_stats=False,
        )
        self.assertFalse(partition.q_mask[~mask].any())
        self.assertTrue(partition.owner[~mask].eq(-1).all())
        self.assertFalse(any(
            (~mask[members]).any() for members in partition.members
        ))
        self.assertEqual(int(partition.q_mask.sum()), 7)

    def test_fixed_seed_partition_is_equivariant_under_100_relabelings(self) -> None:
        adjacency = [[] for _ in range(13)]
        edges = (
            (0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (5, 6),
            (2, 7), (7, 8), (8, 9), (9, 10), (10, 11), (8, 12),
        )
        for left, right in edges:
            adjacency[left].append(right)
            adjacency[right].append(left)

        def signature(partition, inverse: list[int]):
            q = frozenset(
                inverse[index]
                for index in torch.where(partition.q_mask)[0].tolist()
            )
            tokens = sorted(
                (
                    int(partition.levels[index]),
                    bool(partition.residual[index]),
                    int(partition.rounds[index]),
                    tuple(sorted(
                        inverse[node]
                        for node in partition.members[index].tolist()
                    )),
                )
                for index in range(len(partition.members))
            )
            return q, tokens

        expected = signature(partition_graph(adjacency, seed=29), list(range(13)))
        rng = random.Random(5)
        for _ in range(100):
            permutation = list(range(13))
            rng.shuffle(permutation)
            inverse = [0] * 13
            for old, new in enumerate(permutation):
                inverse[new] = old
            relabeled = [[] for _ in range(13)]
            for left, right in edges:
                new_left, new_right = permutation[left], permutation[right]
                relabeled[new_left].append(new_right)
                relabeled[new_right].append(new_left)
            actual = signature(
                partition_graph(relabeled, seed=29),
                inverse,
            )
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
