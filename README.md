# data-fusion

A collection of implementations of algorithms for data fusion in decentralized sensor networks and simulations to test 
and assess them.

Currently implemented algorithms:

- [Covariance Intersection (CI)](algorithms/covariance_intersection.py)
- [Ellipsoidal Intersection (EI)](algorithms/ellipsoidal_intersection.py)
- [Inverse Covariance Intersection (ICI)](algorithms/inverse_covariance_intersection.py)

For testing and comparison:
- [Naive fusion](algorithms/naive.py)

Simulations:
- [SimpleSensorNetwork](simulation/simple_sensor_network.py), a network of two sensor nodes running a local Kalman filter
to estimate the global state of a constant acceleration model.