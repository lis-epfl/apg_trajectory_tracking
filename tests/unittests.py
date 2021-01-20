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


def test_circle():
    mid_point = np.array([1, 2, 3])
    radius = 2

    circle = Circle(mid_point, radius, plane=[0, 2])
    point = np.array([-1, 2, 3])  # np.array([2.5,2,2])
    projected = circle.to_3D(circle.project_point(point))

    target = circle.next_target(point, 1)
    print(target)

    assert np.isclose(np.linalg.norm(target - point))

    circle.plot_circle()
    mid_x, mid_y = circle.mid_point[circle.plane]
    # plt.scatter(points[:,0], points[:,1])
    plt.scatter(point[0], point[2], c="red")
    plt.scatter(projected[0], projected[2], c="green")
    plt.scatter(target[0], target[2], c="yellow")
    plt.xlim(-1.5, 3.5)
    plt.ylim(0.5, 5.5)
    plt.show()