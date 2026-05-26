# CR_Extended

Extending the confidence-region project from 2D fields to general-dimensional fields, with particular focus on the 3D case.

## Data convention

Throughout the code, spatial fields are stored as NumPy arrays using the following convention:

```python
field.shape == (N1, N2, ..., Nd)
```

where `d` is the spatial dimension.

A collection of subject-level observations is stored as:

```python
instances.shape == (n_subject, N1, N2, ..., Nd)
```

The first axis indexes subjects, while the remaining axes correspond to spatial coordinates.

Boundary, contour, or surface points are represented as point clouds:

```python
boundary_points.shape == (n_points, d)
```

The experiment-running code should live in `run_experiment.py`, while reusable mathematical and simulation functions should live in the other files.
