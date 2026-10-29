import unittest

import torch

from jiuge_training.weights import packed_qkv_slice


class Reader:
    def __init__(self, tensors):
        self.tensors = tensors

    def tensor(self, name, index=None):
        return (
            self.tensors[name] if index is None else self.tensors[name][index]
        )


class PackingTest(unittest.TestCase):
    def test_qkv_rank_slices_including_tp_greater_than_kv(self):
        heads, groups, dim = 8, 2, 4
        q = torch.arange(heads * dim * 3).reshape(heads * dim, 3)
        k = torch.arange(groups * dim * 3).reshape(groups * dim, 3) + 1000
        v = k + 1000
        reader = Reader(
            {'a.q_proj.weight': q, 'a.k_proj.weight': k, 'a.v_proj.weight': v}
        )
        expected = torch.cat([q[:16], k[:4], v[:4], q[16:], k[4:], v[4:]])
        for world in (1, 2, 4, 8):
            actual = torch.cat(
                [
                    packed_qkv_slice(
                        reader, 'a', 'weight', heads, groups, dim, rank, world
                    )
                    for rank in range(world)
                ]
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
