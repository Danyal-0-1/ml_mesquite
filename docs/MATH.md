# The mathematics

*Status: M1. Covers conventions, the target and the correction. The measurement model, error terms, Step 0 and
the metrics are added in M3 and M4.*

## 1. Picture first

Every pod reports an orientation that is right about tilt and wrong about heading. The heading error is one
rotation about the vertical, Y(−ψ_i), placed on the **left** of the true orientation, and each pod has its own.
For a joint, only the difference of the two pods' ψ matters: δ = ψ_child − ψ_parent. Rotate the child's report
by Y(δ) and it shares its parent's north. Do this outward from the pelvis and every pod shares the pelvis pod's
north, which itself stays unknown.

## 2. Conventions (owned by [mesqsim/conventions.py](../mesqsim/conventions.py))

| quantity | convention | pinned by |
|---|---|---|
| quaternion storage | `(x, y, z, w)`, scalar last, float64, as in `scipy.spatial.transform.Rotation` | `test_quaternion_order_is_xyzw_like_scipy` |
| product | Hamilton: i·j = k. `quat_mul(a, b)` applies `b` first, then `a` | `test_hamilton_product`, `test_composition_order_is_right_to_left` |
| sign | `q` and `−q` are the same rotation; canonical form has `w ≥ 0` (the wire's form) | `test_q_and_minus_q_are_the_same_rotation` |
| orientation | maps body to world: `v_world = R · v_body` | `test_orientation_maps_body_to_world` |
| world frame | gravity-aligned, **+Z up**, heading arbitrary but fixed | dataset loaders (M2) |
| Y(α) | rotation by α about world +Z, right-handed; Y(90°) maps +x to +y | `test_y90_maps_x_to_y` |
| heading | first angle of the intrinsic Z-Y-X decomposition, `atan2(R[1,0], R[0,0])`; so heading(Y(α)·R) = wrap(heading(R) + α) | `test_heading_shifts_by_alpha` |
| wrap | into (−180°, 180°]: wrap(190°) = −170°, wrap(−180°) = 180° | `test_wrap_values` |
| Exp / Log | rotation vector (axis × angle, radians) ↔ quaternion | `test_exp_log_roundtrip_and_geodesic_angle` |
| geodesic angle | 2·atan2(‖v‖, \|w\|), in [0°, 180°] | same |
| units | radians in code, degrees in reports; the unit is in the name (`_rad`, `_deg`, `_s`, `_ms`) | review |
| statistics of angles | circular mean `atan2(mean sin, mean cos)`, circular std `sqrt(−2 ln R)` | `test_circular_statistics` |

Heading is ill-conditioned when the body x axis is vertical (pitch ±90°). The relative-heading target δ does not
use the heading decomposition at all; it is defined from the ψ_i of the measurement model.

## 3. Target and correction (owned by [mesqsim/correction.py](../mesqsim/correction.py))

```text
target for joint (parent p, child c):   δ_pc(t) = wrap(ψ_c(t) − ψ_p(t))
correction, outward from the pelvis:    ĉ_root = 0,   ĉ_c = ĉ_p + δ̂_pc,   q̃_c = Y(ĉ_c) · q_c
```

Why it works: with q_c = Y(−ψ_c)·s_c·m_c and exact estimates, ĉ_c = ψ_c − ψ_root, so
q̃_c = Y(ψ_c − ψ_root)·Y(−ψ_c)·s_c·m_c = Y(−ψ_root)·s_c·m_c. Every pod ends up with the same unknown heading.

### Worked example (a unit test and validation check 3.1)

Parent at the root with ψ_p = 12°, child with ψ_c = 5°.

1. δ_pc = wrap(5° − 12°) = −7°.
2. ĉ_c = 0 + (−7°), so q̃_c = Y(−7°)·Y(−5°)·s_c·m_c = Y(−12°)·s_c·m_c: the child shares the parent's north.
3. With the wrong sign, Y(+7°)·Y(−5°) = Y(+2°), which differs from Y(−12°) by **14°**.

## 4. What cannot be known, as tests

| statement | test |
|---|---|
| Rotating every report by one Y(β) changes no target δ (exactly) | `test_common_yaw_leaves_targets_unchanged_exactly` |
| Rotating one pod's frame by Y(γ) changes only the targets of its joints, by exactly ±γ | `test_single_pod_offset_moves_only_its_joints_by_gamma` |
| A correction acts about the vertical only, so tilt never changes | `test_correction_never_changes_tilt`, check 4.1 |
