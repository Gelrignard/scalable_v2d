"""Manufactured observations for causal tests with known clean ground truth."""
import numpy as np

from .horizon_reference import ContactPatch, HorizonReference, stationary_base
from correction.src.dynamics.symbolic_mechanics import CartesianPads


def controlled_carry(*, seed=7, noise=.004, outlier=.025, knots=21, contact=True,
                     observation_rate=30, top_only=False):
    robot = CartesianPads(count=2)
    times = np.linspace(0, 2., knots)
    observations = np.linspace(0, 2., 2*observation_rate+1)
    def truth(t):
        u = t/2
        s = 10*u**3-15*u**4+6*u**5
        return np.array([.04, .0, .12])[None]*s[:, None]+[0., 0., .4]
    clean = truth(observations)
    generator = np.random.default_rng(seed)
    observed = clean+generator.normal(0, noise, clean.shape)
    observed[len(observed)//2, 2] += outlier
    observed[0], observed[-1] = clean[0], clean[-1]
    contacts = []
    centers = []
    half = np.array([.08, .03, .1])
    for i, sign in enumerate((-1, 1)):
        center = np.array([0., sign*half[1], 0.])
        inward = np.array([0., -sign, 0.])
        tangent = np.array([[1., 0.], [0., 0.], [0., 1.]])
        if top_only:
            center = np.array([sign*.04, 0., half[2]])
            inward = np.array([0., 0., -1.])
            tangent = np.array([[1., 0.], [0., 1.], [0., 0.]])
        contacts.append(ContactPatch(f'pad{i}', center, inward, tangent,
                                     np.full(knots, contact), robot_index=i, radius=0.))
        centers.append(center-robot.radius*inward)
    center_array = truth(times)[:, None, :]+np.asarray(centers)[None]
    original = center_array.copy()
    # A demonstrably poor retargeting: opens the grasp by 6 mm and oscillates
    # in height, while preserving exactly the common initial state.
    blend = np.sin(np.pi*times/times[-1]/2)**2
    for i, sign in enumerate((-1, 1)):
        original[:, i, 1] += sign*.006*blend
    original[:, :, 2] += .006*np.sin(8*np.pi*times)[:, None]*blend[:, None]
    identity = np.tile([1., 0., 0., 0.], (len(observations), 1))
    mass = .5
    size = 2*half
    ref = HorizonReference(times, observations, observed, identity, original.reshape(knots, -1),
        stationary_base(knots), center_array[0].ravel(), np.zeros(robot.n), clean[0], identity[0],
        np.zeros(3), np.zeros(3), clean[-1], identity[-1], contacts, mass,
        np.diag(mass*(sum(size**2)-size**2)/12), half, goal_rest=True,
        position_sigma=.008, rotation_sigma=np.deg2rad(3.),
        goal_position_tolerance=.005, goal_rotation_tolerance=np.deg2rad(2.),
        metadata=dict(source='manufactured two-pad carry; not dataset or G1', noise_seed=seed,
                      position_noise_sigma_m=noise, isolated_outlier_m=outlier, initially_grasped=True))
    return ref, robot, dict(times=observations, positions=clean, wxyz=identity)
