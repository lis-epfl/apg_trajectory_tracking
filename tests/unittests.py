import torch
import numpy as np
import unittest


def test_project():

    # batch size 4, dim 3
    a_on_line = torch.randn(4, 3)
    b_on_line = torch.randn(4, 3)
    points = torch.randn(4, 3)
    x = project_to_line(a_on_line, b_on_line, points)

    for i in range(4):
        a = a_on_line[i].numpy()
        p = points[i].numpy()
        res = x[i].numpy()
        # check phytagoras
        assert np.isclose(
            np.sum((p - a)**2),
            np.sum((res - a)**2) + np.sum((res - p)**2)
        )
