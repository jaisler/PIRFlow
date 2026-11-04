# SPDX-License-Identifier: MIT
import numpy as np
import pyvista as pv

from pirflow.sampling.sampling import SamplingData


def test_vector_velocity_uses_configured_component():
    sampled = pv.PolyData(np.array([[0., 0., 0.], [1., 0., 0.]]))
    sampled["CFD_velocity"] = np.array([[10., 20.], [30., 40.]])
    data = SamplingData({
        "dimension": 2,
        "run": {"routines": {"sampling": False}},
        "cfd": {"fields": ["CFD_velocity"], "components": [1]},
    })

    values = data._sample_cfd_field(sampled, 0)

    np.testing.assert_array_equal(values, [20., 40.])


def test_scalar_velocity_uses_component_zero():
    sampled = pv.PolyData(np.array([[0., 0., 0.], [1., 0., 0.]]))
    sampled["u-velocity"] = np.array([10., 30.])
    data = SamplingData({
        "dimension": 2,
        "run": {"routines": {"sampling": False}},
        "cfd": {"fields": ["u-velocity"], "components": [0]},
    })

    values = data._sample_cfd_field(sampled, 0)

    np.testing.assert_array_equal(values, [10., 30.])
