# Mesquite motion and simulation: the repository map and rotation mathematics

Prepared on 2026-11-22 against the local `main` checkout at commit `61862eb`.

This file brings together the complete Module 0 and Module 1 explanation from the tutoring conversation,
including the reading plan, references, mathematical derivations, code traces, numerical examples, runnable
snippets, and self-check answers. Mathematical steps are expanded so the document can be studied independently.
Modules 2–8 are the planned continuation; their full lessons have not yet been written in this document.

The explanation follows this repository's conventions. Implementation claims have visible `path:line` citations.
A mathematical derivation explains why a formula works; it does not imply the code implements every intermediate
algebraic step. All numerical worked examples are constructed toy examples, not measurements from AMASS, the
recordings, or `twin_v1`.

Links are relative to this file. Code links include GitHub-style line anchors. Citations to Markdown documents
give the source line in the label and open the document itself. Line numbers describe the checkout above.

## How to study this file

Start with the picture, then read the questions, sources, mathematics, code trace, examples, and self-checks.
Work an example on paper before running its snippet. For a matrix product, read right to left and name each
input and output coordinate frame.

Run shell snippets from the repository root with `.venv/bin/python`. These modules need no licensed files.
Later data-dependent work uses `MESQ_DATA_DIR` and `MESQ_OUTPUT_DIR` through the repository's path helpers.
[mesqsim/paths.py:18](../../mesqsim/paths.py#L18),
[mesqsim/paths.py:23](../../mesqsim/paths.py#L23)

## Module 0: the map

Picture two virtual suits watching the same human motion:

```text
                    AMASS motion + SMPL+H model
                               |
                     body motion and orientation
                               |
               +---------------+-----------------+
               |                                 |
       STEP 1: statistical                STEP 3: physical twin
       directly add errors                move sites on the skin
       to orientations                    |
               |                          ideal gyro + accelerometer
               |                          | sensor errors
               |                          six-axis fusion
               |                                 |
               +------ wire, clocks, radio ------+
                               |
                     received observations
                               |
                     STEP 0: decode them
                               |
              +----------------+-----------------+
              |                                  |
       baselines / STEP 2 GEO              features + target arrays
              |                                  |
            score                         twin_v1 stored sessions
                                                 |
                                          windows -> your model
                                                 |
                                              score-model
```

Simulator truth accompanies these paths so training and scoring know the answer. A deployable estimator receives
observations, not that truth.
[docs/MUST_KNOW.md:31](../MUST_KNOW.md),
[mesqsim/learn/store.py:143](../../mesqsim/learn/store.py#L143)

### 0.1 Questions this module answers

1. What enters the project?
2. Why are there two simulators?
3. Where does decoding happen?
4. What becomes a training example?
5. What exists already, and what remains for Danyal to implement?

### 0.2 Files read

- [README.md:15](../../README.md): the suit, data path, and Steps 1–2.
- [REPOSITORY_GUIDE.md:6](../../REPOSITORY_GUIDE.md): releases and component map.
- [docs/MUST_KNOW.md:24](../MUST_KNOW.md): relative headings, truth, and leakage rules.
- [docs/explain/MENTAL_PICTURE.md:11](../explain/MENTAL_PICTURE.md): the fifteen-watches picture.

The dataset path was also checked against the implementation cited below.

One scope correction matters: `MUST_KNOW.md` explicitly describes Steps 1–2. Its historical “0 measured” parameter
count does not describe the twin. The repository guide identifies the older documents whose Step 3 updates remain
unwritten; use the Step 3 guides for the newer system.
[docs/MUST_KNOW.md:3](../MUST_KNOW.md),
[docs/MUST_KNOW.md:13](../MUST_KNOW.md),
[REPOSITORY_GUIDE.md:145](../../REPOSITORY_GUIDE.md),
[docs/step_3/DATASET_CARD.md:77](../step_3/DATASET_CARD.md)

### 0.3 The mathematics

Let $s_i$ map segment coordinates into world coordinates, and let mounting $m_i$ map pod coordinates into segment
coordinates. For a vector:

$$
v_{\rm segment}=m_i v_{\rm pod},\qquad
v_{\rm world}=s_i v_{\rm segment}=s_i m_i v_{\rm pod}.
$$

Therefore the true pod orientation is $P_i=s_i m_i$. The simulator also applies the session's common facing
rotation when constructing the world trajectory.
[mesqsim/sim/truth.py:24](../../mesqsim/sim/truth.py#L24)

Step 1 directly constructs the reported orientation:

$$
q_i=Y(-\psi_i)P_i\operatorname{Exp}(\eta_i).
$$

Here $q_i$ is the report, $\psi_i$ is the private heading reference, and $\eta_i$ is a small rotation vector
modeling fusion error. $Y$ is a world-Z yaw rotation and $\operatorname{Exp}$ converts a rotation vector into a
quaternion. Products compose rotations right to left.
[mesqsim/sim/simulate.py:150](../../mesqsim/sim/simulate.py#L150)

Step 3's physical modes instead generate sensor measurements, corrupt them, and run fusion. Their orientation
errors emerge from that chain. The twin also retains a selectable `analytic` mode reproducing Step 1's style of
error model, so the chosen fusion mode matters.
[mesqsim/twin/session.py:414](../../mesqsim/twin/session.py#L414),
[mesqsim/twin/session.py:421](../../mesqsim/twin/session.py#L421),
[mesqsim/twin/fusion.py:3](../../mesqsim/twin/fusion.py#L3)

For parent pod $p$, child pod $c$, and phone reference $\psi_{\rm ph}$, the targets are

$$
\delta_{pc}=\operatorname{wrap}(\psi_c-\psi_p),\qquad
\epsilon_i=\operatorname{wrap}(\psi_i-\psi_{\rm ph}).
$$

Subtract the phone-relative references before wrapping:

$$
(\psi_c-\psi_{\rm ph})-(\psi_p-\psi_{\rm ph})
=\psi_c-\psi_{\rm ph}-\psi_p+\psi_{\rm ph}
=\psi_c-\psi_p.
$$

Thus

$$
\delta_{pc}=\operatorname{wrap}(\epsilon_c-\epsilon_p),\qquad
\epsilon_c=\operatorname{wrap}(\epsilon_p+\delta_{pc}).
$$

These are circular angles: adding a full turn changes the number but not the direction. Wrapping chooses a
representative in $(-\pi,\pi]$ radians, or $(-180^\circ,180^\circ]$ in reports.
[mesqsim/learn/targets.py:23](../../mesqsim/learn/targets.py#L23),
[mesqsim/learn/scorer.py:159](../../mesqsim/learn/scorer.py#L159),
[mesqsim/conventions.py:189](../../mesqsim/conventions.py#L189)

### 0.4 The code, traced

| Stage | What it adds and where it lives |
|---|---|
| Motion loading | Load AMASS poses, compute global joint rotations, and select suit segment orientations. [mesqsim/data/amass.py:96](../../mesqsim/data/amass.py#L96), [mesqsim/data/amass.py:125](../../mesqsim/data/amass.py#L125), [mesqsim/data/amass.py:134](../../mesqsim/data/amass.py#L134) |
| Statistical suit | Heading and fusion errors; sampling, timing, quantization, and reception. [mesqsim/sim/simulate.py:96](../../mesqsim/sim/simulate.py#L96) |
| Decoding and geometry | Step 0 supplies decoded series; Step 2 estimates relative heading from body constraints. [REPOSITORY_GUIDE.md:61](../../REPOSITORY_GUIDE.md), [REPOSITORY_GUIDE.md:90](../../REPOSITORY_GUIDE.md) |
| Physical suit | Skin motion → sensor physics → errors → fusion → communications and phone. [REPOSITORY_GUIDE.md:114](../../REPOSITORY_GUIDE.md), [mesqsim/twin/session.py:421](../../mesqsim/twin/session.py#L421) |
| Calibration | Fit the twin to recordings using rejection approximate Bayesian computation and inspect predictive comparisons. [mesqsim/twin/calibration.py:14](../../mesqsim/twin/calibration.py#L14) |
| Dataset generation | `build_session` makes a motion program, simulates it, computes features from decoded observations, and independently builds targets. [mesqsim/learn/store.py:125](../../mesqsim/learn/store.py#L125) |
| Storage | Input, target, and diagnostic arrays go into an HDF5 session group. Select windows from stored sessions later. [mesqsim/learn/store.py:152](../../mesqsim/learn/store.py#L152), [mesqsim/learn/store.py:170](../../mesqsim/learn/store.py#L170), [mesqsim/learn/store.py:395](../../mesqsim/learn/store.py#L395) |
| Evaluation | Call `model.predict(inputs)` and compare predictions with masked truth using wrapped angular errors. [mesqsim/learn/scorer.py:128](../../mesqsim/learn/scorer.py#L128) |

The key construction lines are:

```python
f = features(decode(s.frames, s.session), s.phone, t + s.meta["plan"]["host_epoch_s"],
             cfg1["geometry.clock_window_s"], cfg1["geometry.clock_block_s"])
y = session_targets(s.labels_joints, st)
```

The first operation decodes pod frames and builds features with received phone data on the analysis grid.
The second constructs supervision from simulator labels. Stored together, these still have distinct roles:
inputs go to prediction; targets go to the training loss and scorer.
[mesqsim/learn/store.py:143](../../mesqsim/learn/store.py#L143)

For $T$ stored steps, `y_delta` has shape $(T,14)$ and `y_eps` has shape $(T,15)$, both `float32` in radians.
Their matching validity arrays are Boolean. Module 7 will explain all features and masks.
[mesqsim/learn/targets.py:6](../../mesqsim/learn/targets.py#L6),
[mesqsim/learn/targets.py:32](../../mesqsim/learn/targets.py#L32)

The full twin can run the page, but dataset construction defaults to `page=False` and stores no BVH. Page rendering
is not a compulsory step between fusion and stored training features.
[mesqsim/learn/store.py:125](../../mesqsim/learn/store.py#L125)

The training loader, networks, losses, and training loop are deliberately left for Danyal to write. The repository
supplies specifications, tests, and the scorer; this tutorial does not imply a trained TCN already exists here.
[docs/step_3/LEARN_THE_MODEL.md:3](../step_3/LEARN_THE_MODEL.md)

### 0.5 Worked numerical example

Use the documented sign example $\psi_p=12^\circ$, $\psi_c=5^\circ$:

$$
\delta_{pc}=\operatorname{wrap}(5^\circ-12^\circ)=-7^\circ.
$$

Choose an illustrative phone reference $\psi_{\rm ph}=2^\circ$:

$$
\epsilon_p=12^\circ-2^\circ=10^\circ,\qquad
\epsilon_c=5^\circ-2^\circ=3^\circ.
$$

Then $\epsilon_c-\epsilon_p=3^\circ-10^\circ=-7^\circ=\delta_{pc}$. A prediction of $-5^\circ$ has error

$$
\left|\operatorname{wrap}(-5^\circ-(-7^\circ))\right|
=|\operatorname{wrap}(2^\circ)|=2^\circ.
$$

The sign example is in [docs/MUST_KNOW.md:46](../MUST_KNOW.md); the error follows
[mesqsim/learn/scorer.py:145](../../mesqsim/learn/scorer.py#L145).

### 0.6 Self-check questions

1. Which simulator directly assigns orientation errors?
2. What lies between body motion and reported orientation in Step 3's physical modes?
3. Is Step 0 an estimator of heading drift?
4. May a deployed network read `y_eps`?
5. Does the repository already contain Danyal's finished learned model?

---

**Answers — try the questions before reading further.**

1. Step 1.
2. Sensor physics, sensor errors, and fusion, followed by communications.
3. No. It decodes and organizes observations and performs registrations.
4. No. It is a target for loss and scoring, not a prediction input.
5. No. It contains the model kit and exercises.

## The nine-module reading plan

These are estimated study times including examples, not measured runtimes. Modules 0 and 1 are here; Modules 2–8
describe the planned continuation.

| Module | Subject | Estimated study time |
|---|---|---:|
| 0 | Repository map | 5–10 minutes |
| 1 | Rotations and every convention helper | 75–100 minutes |
| 2 | AMASS and SMPL+H: files, body tree, forward kinematics, shape, and skinning | 45–60 minutes |
| 3 | Mapping the body to the suit: segment choices, mountings, facing, and joint rotations | 30–45 minutes |
| 4 | Statistical simulator: heading, session, clocks, wire, labels, Step 0, and baselines | 75–100 minutes |
| 5 | Physical twin: skin sites, IMU physics, errors, fusion, phone, procedure, and page | 90–120 minutes |
| 6 | Calibration: recording statistics, rejection ABC, and predictive checks | 60–75 minutes |
| 7 | Causal features, targets, masks, shards, splits, and windows in `twin_v1` | 75–100 minutes |
| 8 | One instant followed from an AMASS frame to stored feature and target rows | 45–60 minutes |

## Module 1: rotations as this repository uses them

Picture a pod with three arrows painted on it:

```text
           pod/body coordinates                 world coordinates

           x_B, y_B, z_B                         x_W, y_W, z_W
                 |                                     ^
                 +---------- orientation R ------------+

                  v_W = R v_B
                  v_B = R^T v_W

       R says where the pod's three painted arrows point in the world.
```

A vector and its coordinates are different things. A physical arrow can stay unchanged while its three
coordinate numbers change because we express it using another set of axes.

### 1.1 Questions this module answers

- What exactly does a matrix or quaternion describe here?
- Why does multiplication order matter?
- What do axis-angle, `Exp`, `Log`, heading, and tilt mean?
- How do orientations produce relative orientations and angular rates?
- How should rotations and angles be interpolated and averaged?
- What does every function in `conventions.py` do, and which tests check it?

### 1.2 Files read

- [mesqsim/conventions.py:1](../../mesqsim/conventions.py#L1): implementation and conventions.
- [tests/test_conventions.py:1](../../tests/test_conventions.py#L1): original rotation tests.
- [tests/test_conventions_step2.py:1](../../tests/test_conventions_step2.py#L1): Cardan and von Mises tests.
- [docs/explain/MATH_EXPLAINED.md:14](../explain/MATH_EXPLAINED.md): existing rotation explanations.
- [docs/MATH.md:241](../MATH.md): existing circular-uncertainty toolkit explanation.
- [mesqsim/estimators/clock.py:159](../../mesqsim/estimators/clock.py#L159): finite-interval body rates.
- [mesqsim/twin/physics.py:56](../../mesqsim/twin/physics.py#L56): spline-based physical truth.
- [mesqsim/twin/session.py:421](../../mesqsim/twin/session.py#L421): how the session supplies fusion inputs.

### 1.3 The mathematics, derived

#### A. A small mathematical vocabulary

An element of $\mathbb R^3$ is a column of three real numbers. Its NumPy shape is `(3,)`, although the algebra
writes it as a column. Its transpose $v^\top$ is the corresponding row in matrix algebra.

For $a=(a_x,a_y,a_z)^\top$ and $b=(b_x,b_y,b_z)^\top$, the dot product and norm are

$$
a^\top b=a_xb_x+a_yb_y+a_zb_z,\qquad
\|a\|=\sqrt{a^\top a}.
$$

A unit vector has norm one. Perpendicular vectors have zero dot product. The cross product is

$$
a\times b=
\begin{bmatrix}
a_yb_z-a_zb_y\\
a_zb_x-a_xb_z\\
a_xb_y-a_yb_x
\end{bmatrix}.
$$

It is perpendicular to both inputs, with sign fixed by the right-hand rule:
$e_x\times e_y=e_z$ but $e_y\times e_x=-e_z$.

For a matrix $A$, each output component of $Av$ is a row dotted with $v$:

$$
(Av)_i=\sum_j A_{ij}v_j.
$$

The identity $I$ leaves vectors unchanged. Transposing a product reverses order:
$(AB)^\top=B^\top A^\top$.

Radians measure arc length divided by radius. A full turn is $2\pi$ radians, so

$$
\alpha_{\rm rad}=\alpha_{\rm deg}\frac{\pi}{180},\qquad
\alpha_{\rm deg}=\alpha_{\rm rad}\frac{180}{\pi}.
$$

The repository uses radians in code and degrees in reports.
[mesqsim/conventions.py:19](../../mesqsim/conventions.py#L19)

#### B. Rotation matrices preserve lengths, angles, and handedness

An orientation here means $v_W=Rv_B$. Multiplying by $e_x=(1,0,0)^\top$ selects the first matrix column.
Thus the columns are body x, y, and z expressed in world coordinates.

Length preservation requires

$$
\|Rv\|^2=(Rv)^\top(Rv)=v^\top R^\top Rv=v^\top v
$$

for every $v$. Hence $R^\top R=I$. To justify that conclusion, let $M=R^\top R-I$. It is symmetric, and
$v^\top Mv=0$ for every $v$. Choosing basis vectors makes its diagonal zero; choosing $e_i+e_j$ then makes
its off-diagonal entries zero. Therefore $M=0$.

The columns are mutually perpendicular unit vectors: orthonormal columns. Dot products are preserved too:

$$
(Ra)^\top(Rb)=a^\top R^\top Rb=a^\top b.
$$

Since dot products determine angles, angles are preserved. Also $R^{-1}=R^\top$.
Taking determinants gives $(\det R)^2=1$, so an orthogonal matrix has determinant $+1$ or $-1$.
A proper rotation has $\det R=+1$; the negative case includes reflections.
For example, $\operatorname{diag}(-1,1,1)$ reflects one axis. The set of proper three-dimensional rotations is
called $SO(3)$.

The repository's body-to-world meaning is explicit in
[mesqsim/conventions.py:12](../../mesqsim/conventions.py#L12) and implemented by
[mesqsim/conventions.py:142](../../mesqsim/conventions.py#L142).

Name the frames: if $R_{SP}$ maps pod → segment and $R_{WS}$ maps segment → world, then

$$
v_S=R_{SP}v_P,\qquad
v_W=R_{WS}v_S=(R_{WS}R_{SP})v_P.
$$

Therefore $R_{WP}=R_{WS}R_{SP}$. The rightmost transformation acts first. This explains $P_i=s_i m_i$ and
the matching quaternion composition order.
[mesqsim/conventions.py:85](../../mesqsim/conventions.py#L85)

#### C. Axis-angle and the matrix exponential

Let $n$ be a unit axis and $\theta$ a right-handed angle in radians. The rotation vector $\phi=\theta n$
stores angle in its length and axis in its direction. At zero angle the axis is irrelevant.

Split a vector:

$$
v_\parallel=n(n^\top v),\qquad v_\perp=v-v_\parallel.
$$

The parallel component stays fixed; the perpendicular plane rotates:

$$
Rv=v_\parallel+\cos\theta\,v_\perp+\sin\theta(n\times v).
$$

Substitute the split:

$$
Rv=\cos\theta\,v+(1-\cos\theta)n(n^\top v)+\sin\theta(n\times v).
$$

Define the cross-product matrix

$$
[n]_\times=
\begin{bmatrix}
0&-n_z&n_y\\n_z&0&-n_x\\-n_y&n_x&0
\end{bmatrix},
\qquad [n]_\times v=n\times v.
$$

This gives Rodrigues' formula:

$$
R=\cos\theta I+(1-\cos\theta)nn^\top+\sin\theta[n]_\times.
$$

To derive the exponential connection, put $K=[n]_\times$. The triple-product identity gives

$$
K^2v=n\times(n\times v)=n(n^\top v)-v,
$$

so $K^2=nn^\top-I$. Since $Kn=0$, $K^3=K(nn^\top-I)=-K$.
The matrix exponential is a power series, not entrywise `np.exp`:

$$
e^{\theta K}=I+\theta K+\frac{\theta^2K^2}{2!}+\frac{\theta^3K^3}{3!}+\cdots.
$$

Collect odd and even powers using $K^3=-K$ and $K^4=-K^2$:

$$
e^{\theta K}
=I+\left(\theta-\frac{\theta^3}{3!}+\cdots\right)K
+\left(\frac{\theta^2}{2!}-\frac{\theta^4}{4!}+\cdots\right)K^2
=I+\sin\theta K+(1-\cos\theta)K^2.
$$

Substituting $K^2=nn^\top-I$ recovers Rodrigues. In this repository `exp_map(phi)` returns the equivalent
quaternion, rather than the matrix $e^{[\phi]_\times}$.
[mesqsim/conventions.py:206](../../mesqsim/conventions.py#L206)

#### D. Unit quaternions and the Hamilton product

Write a scalar-last quaternion as

$$
q=(u,w)=(x,y,z,w),\qquad \|u\|^2+w^2=1.
$$

The vector part $u=(x,y,z)^\top$ and scalar $w$ are dimensionless. Quaternion components are not Euler angles.
The Hamilton product is

$$
(u_a,w_a)\otimes(u_b,w_b)
=\left(w_au_b+w_bu_a+u_a\times u_b,\;
w_aw_b-u_a^\top u_b\right).
$$

Expanding the cross product gives the implementation. For example, its first vector component is
$w_ab_x+w_ba_x+a_yb_z-a_zb_y$. Swapping factors reverses the cross-product term, so multiplication generally
does not commute.
[mesqsim/conventions.py:85](../../mesqsim/conventions.py#L85)

The conjugate is $q^*=(-u,w)$. Its product with $q$ has vector part $-wu+wu+u\times(-u)=0$ and scalar part
$w^2-u^\top(-u)=w^2+\|u\|^2$. Thus

$$
q\otimes q^*=(0,\|q\|^2),\qquad q^{-1}=\frac{q^*}{\|q\|^2}.
$$

For unit $q$, inverse equals conjugate. Otherwise conjugation alone is not the inverse.
The identity is $(0,0,0,1)$.
[mesqsim/conventions.py:78](../../mesqsim/conventions.py#L78),
[mesqsim/conventions.py:105](../../mesqsim/conventions.py#L105)

#### E. Deriving the quaternion action on a vector

Represent $v$ as the pure quaternion $(v,0)$. For unit $q$:

$$
(v',0)=q\otimes(v,0)\otimes q^*.
$$

For the first product set $a=wv+u\times v$ and $b=-u^\top v$. Then $q\otimes(v,0)=(a,b)$.
Multiplying by $q^*=(-u,w)$ gives vector part

$$
-bu+wa-a\times u
=(u^\top v)u+w^2v+w(u\times v)
-(wv)\times u-(u\times v)\times u.
$$

Use $v\times u=-u\times v$ and
$(u\times v)\times u=\|u\|^2v-u(u^\top v)$:

$$
v'=(w^2-\|u\|^2)v+2u(u^\top v)+2w(u\times v).
$$

The scalar part is
$bw+a^\top u=-w(u^\top v)+w(v^\top u)+(u\times v)^\top u=0$.
For unit $q$, $w^2+\|u\|^2=1$, so the vector expression also becomes

$$
v'=v+2w(u\times v)+2u\times(u\times v).
$$

Define $t=2u\times v$. Then $v'=v+wt+u\times t$, exactly:

```python
t = 2.0 * np.cross(qv, v)
return v + w * t + np.cross(qv, t)
```

These lines avoid explicitly building a matrix or pure quaternions.
[mesqsim/conventions.py:138](../../mesqsim/conventions.py#L138)

Composition follows by associativity:

$$
q_a\otimes\bigl(q_b\otimes(v,0)\otimes q_b^{-1}\bigr)\otimes q_a^{-1}
=(q_a\otimes q_b)\otimes(v,0)\otimes(q_a\otimes q_b)^{-1}.
$$

The inverse of a product reverses factors. Thus $q_a\otimes q_b$ applies $q_b$ first, then $q_a$.
[tests/test_conventions.py:65](../../tests/test_conventions.py#L65)

#### F. The half-angle, matrix conversion, and quaternion signs

The axis-angle quaternion is

$$
q=\left(n\sin\frac{\theta}{2},\cos\frac{\theta}{2}\right).
$$

The action derived above has matrix

$$
R(q)=(w^2-\|u\|^2)I+2uu^\top+2w[u]_\times.
$$

Substitute $u=n\sin(\theta/2)$ and $w=\cos(\theta/2)$. Its coefficients become

$$
w^2-\|u\|^2=\cos\theta,\quad
2\sin^2(\theta/2)=1-\cos\theta,\quad
2\cos(\theta/2)\sin(\theta/2)=\sin\theta.
$$

This is Rodrigues again: the quaternion half-angle produces the full angle in its vector action.
Expanding gives the code's matrix:

$$
R(q)=
\begin{bmatrix}
1-2(y^2+z^2)&2(xy-wz)&2(xz+wy)\\
2(xy+wz)&1-2(x^2+z^2)&2(yz-wx)\\
2(xz-wy)&2(yz+wx)&1-2(x^2+y^2)
\end{bmatrix}.
$$

The implementation normalizes before computing these entries.
[mesqsim/conventions.py:142](../../mesqsim/conventions.py#L142)

All entries are quadratic, so reversing all four component signs leaves every entry unchanged:
$R(-q)=R(q)$. In the sandwich action, the two minus signs cancel too. This is $q\equiv -q$.

For the reverse conversion, the diagonal sum gives

$$
\operatorname{tr}(R)=3-4(x^2+y^2+z^2)=4w^2-1.
$$

For $w>0$:

$$
w=\frac12\sqrt{1+\operatorname{tr}(R)}.
$$

Subtracting off-diagonal entries gives $R_{21}-R_{12}=4wx$, $R_{02}-R_{20}=4wy$, and $R_{10}-R_{01}=4wz$.
Therefore

$$
x=\frac{R_{21}-R_{12}}{4w},\quad
y=\frac{R_{02}-R_{20}}{4w},\quad
z=\frac{R_{10}-R_{01}}{4w}.
$$

These divisions become unsuitable near $w=0$. The repository delegates conversion to SciPy and canonicalizes;
it does not implement those divisions itself.
[mesqsim/conventions.py:159](../../mesqsim/conventions.py#L159)

Canonicalization chooses $w\ge0$. At exactly $w=0$ it leaves the sign unchanged. It is explicit:
`quat_mul`, `yaw_quat`, `exp_map`, and `quat_slerp` need not return canonical signs at every intermediate step.
[mesqsim/conventions.py:123](../../mesqsim/conventions.py#L123),
[mesqsim/conventions.py:168](../../mesqsim/conventions.py#L168),
[mesqsim/conventions.py:206](../../mesqsim/conventions.py#L206),
[mesqsim/conventions.py:232](../../mesqsim/conventions.py#L232)

#### G. Exp, Log, and rotation distance

For $\theta=\|\phi\|$, substitute $n=\phi/\theta$ into the quaternion:

$$
\operatorname{Exp}(\phi)
=\left(\frac{\sin(\theta/2)}{\theta}\phi,\cos(\theta/2)\right).
$$

The sine series gives $\sin(\theta/2)/\theta=1/2-\theta^2/48+\cdots$, so its zero-angle limit is $1/2$.
NumPy defines $\operatorname{sinc}(a)=\sin(\pi a)/(\pi a)$, hence

$$
\frac{\sin(\theta/2)}{\theta}
=\frac12\operatorname{sinc}\left(\frac{\theta}{2\pi}\right).
$$

This explains:

```python
xyz = v * 0.5 * np.sinc(angle / (2.0 * np.pi))
```

Input shape is `(...,3)` in radians. The norm retains a singleton final dimension for broadcasting, and
concatenating the cosine gives `(...,4)`.
[mesqsim/conventions.py:208](../../mesqsim/conventions.py#L208)

For Log, canonicalize $q=(u,w)$ and put $s=\|u\|$. Since $s=\sin(\theta/2)$ and $w=\cos(\theta/2)$:

$$
\theta=2\operatorname{atan2}(s,w),\qquad
\operatorname{Log}(q)=\frac{\theta}{s}u.
$$

Near $s=0$, $\operatorname{atan2}(s,w)\approx s/w$, so $\theta/s\to2/w$. This is the small-angle branch.
[mesqsim/conventions.py:215](../../mesqsim/conventions.py#L215)

Log chooses a principal angle in $[0,\pi]$. Thus $\operatorname{Log}(\operatorname{Exp}(\phi))=\phi$ holds for
$\|\phi\|<\pi$, not globally. Extra turns disappear; at $\pi$, opposite axis signs are ambiguous.
The roundtrip test stays below $\pi$.
[tests/test_conventions.py:122](../../tests/test_conventions.py#L122)

The shortest rotation amount, or geodesic angle, is

$$
d(q)=2\operatorname{atan2}(\|u\|,|w|).
$$

The absolute value makes both quaternion signs agree. To compare $q_a$ with $q_b$, evaluate this on
$q_a^{-1}\otimes q_b$.
[mesqsim/conventions.py:226](../../mesqsim/conventions.py#L226)

#### H. Yaw, intrinsic Z-Y-X, and the heading proof

Here $Y(\alpha)=R_z(\alpha)$:

$$
Y(\alpha)=
\begin{bmatrix}
\cos\alpha&-\sin\alpha&0\\
\sin\alpha&\cos\alpha&0\\
0&0&1
\end{bmatrix},
\qquad
q_Y(\alpha)=\left(0,0,\sin\frac\alpha2,\cos\frac\alpha2\right).
$$

Despite its letter, $Y$ is a Z-axis rotation.
[mesqsim/conventions.py:168](../../mesqsim/conventions.py#L168)

Heading is extracted from a general orientation. The intrinsic Z-Y-X decomposition is

$$
R=R_z(h)R_y(p)R_x(r),
$$

where $h,p,r$ are heading, pitch, roll. The other elementary matrices are

$$
R_x(r)=
\begin{bmatrix}1&0&0\\0&\cos r&-\sin r\\0&\sin r&\cos r\end{bmatrix},
\qquad
R_y(p)=
\begin{bmatrix}\cos p&0&\sin p\\0&1&0\\-\sin p&0&\cos p\end{bmatrix}.
$$

Intrinsic means rotations about successively moving axes: current Z, then current Y, then current X.
The matrix product still acts right to left on a coordinate vector. Multiplying gives first column

$$
Re_x=(\cos h\cos p,\ \sin h\cos p,\ -\sin p)^\top.
$$

On $|p|<\pi/2$, $\cos p>0$, so $h=\operatorname{atan2}(R_{10},R_{00})$: the direction of body +x projected
onto the horizontal plane. That is `heading_rad`.
[mesqsim/conventions.py:175](../../mesqsim/conventions.py#L175)

For the heading-shift proof, write

$$
(R_{00},R_{10})=\rho(\cos h,\sin h),\qquad
\rho=\sqrt{R_{00}^2+R_{10}^2}>0.
$$

After left yaw $R'=Y(\alpha)R$:

$$
R'_{00}=\cos\alpha R_{00}-\sin\alpha R_{10}=\rho\cos(h+\alpha),
$$

$$
R'_{10}=\sin\alpha R_{00}+\cos\alpha R_{10}=\rho\sin(h+\alpha).
$$

Thus, as a circular angle,

$$
\operatorname{heading}(Y(\alpha)R)=\operatorname{wrap}(h+\alpha).
$$

If $\rho=0$, body +x is vertical and heading is undefined. The property test avoids that singularity.
Also `heading_rad` returns raw `atan2`, which can return $-\pi$. Only `wrap_rad` explicitly enforces
$(-\pi,\pi]$. The endpoint-safe statement wraps both sides:

$$
\operatorname{wrap}(\operatorname{heading}(Y(\alpha)R))
=\operatorname{wrap}(\operatorname{heading}(R)+\alpha).
$$

[mesqsim/conventions.py:178](../../mesqsim/conventions.py#L178),
[tests/test_conventions.py:91](../../tests/test_conventions.py#L91)

#### I. Tilt and what gravity cannot reveal

Let world up be $e_z=(0,0,1)^\top$. Since $R$ maps body → world:

$$
g_B=R^\top e_z.
$$

This is `gravity_in_body`: a dimensionless unit up vector, despite its name.
[mesqsim/conventions.py:263](../../mesqsim/conventions.py#L263)

For $R'=Y(\alpha)R$:

$$
g'_B=(Y(\alpha)R)^\top e_z
=R^\top Y(\alpha)^\top e_z
=R^\top e_z=g_B.
$$

The third equality uses $Y(\alpha)^\top e_z=e_z$. All these headings have the same tilt observation.
A unit direction supplies two independent constraints because its components obey $g_B^\top g_B=1$.
A full orientation has three degrees of freedom, leaving rotation about vertical free.

This is why ideal gravity constrains tilt but cannot identify heading. Real accelerometers also feel motion;
their output is not always gravity alone, and fusion is imperfect.

Physical gravity in the twin points downward:

$$
g_W=(0,0,-9.81)\ {\rm m/s^2}.
$$

Its accelerometer calculation is $R^\top(a_W-g_W)$, with site acceleration $a_W$ in world coordinates.
At rest and identity this gives upward specific force $(0,0,9.81)$ in sensor coordinates.
Module 5 will derive this in context.
[mesqsim/twin/physics.py:26](../../mesqsim/twin/physics.py#L26),
[mesqsim/twin/physics.py:68](../../mesqsim/twin/physics.py#L68)

#### J. Relative orientation and cancellation of a common world rotation

Let $R_p,R_c$ map parent and child coordinates into world coordinates. Starting with child coordinates:

$$
v_W=R_cv_c,\qquad
v_p=R_p^\top v_W=R_p^\top R_cv_c.
$$

Therefore

$$
R_{pc}=R_p^\top R_c
$$

maps child → parent. Reversing the product gives its inverse mapping.
If both orientations receive a common world rotation $A$, then

$$
R'_{pc}=(AR_p)^\top(AR_c)
=R_p^\top A^\top A R_c
=R_p^\top R_c.
$$

The common world rotation cancels. This identity holds for any common proper rotation. The suit's unobservable
physical freedom is specifically yaw because gravity establishes vertical.

In quaternion form, $q_{pc}=q_p^{-1}\otimes q_c$. For unit inputs, conjugation supplies the inverse.
The same relative-product pattern appears in the rate code.
[mesqsim/estimators/clock.py:168](../../mesqsim/estimators/clock.py#L168)

#### K. Angular velocity, finite increments, and the frame of the result

Let $R_-=R(t_-)$, $R_+=R(t_+)$, and $\Delta t=t_+-t_->0$.
The body-coordinate increment is

$$
\Delta R_B=R_-^\top R_+.
$$

It satisfies $R_+=R_-\Delta R_B$: it is multiplied on the right.
As a coordinate transformation it maps the later body frame into the earlier body frame. Its principal rotation
vector is an increment expressed using the earlier body's axes:

$$
\omega_B\approx\frac{\operatorname{Log}(\Delta R_B)}{\Delta t}.
$$

Here matrix $\operatorname{Log}$ denotes the rotation vector matching quaternion `log_map`, not an entrywise
logarithm. More formally, the matrix logarithm is $[\operatorname{Log}(\Delta R_B)]_\times$.

Derive the instantaneous limit using Taylor expansion:

$$
R(t+\Delta t)=R(t)+\dot R(t)\Delta t+O(\Delta t^2).
$$

The dot denotes time differentiation. $O(\Delta t^2)$ groups terms shrinking at least quadratically with the
interval. Multiplying by $R(t)^\top$ gives

$$
R(t)^\top R(t+\Delta t)
=I+R^\top\dot R\,\Delta t+O(\Delta t^2).
$$

Differentiate $R^\top R=I$:

$$
\dot R^\top R+R^\top\dot R=0.
$$

Thus $(R^\top\dot R)^\top=-R^\top\dot R$: it is skew-symmetric and has the form $[\omega_B]_\times$.
If $M^\vee$ converts a cross-product matrix $M$ back to its three-vector, then

$$
\omega_B=(R^\top\dot R)^\vee,\qquad
\dot R=R[\omega_B]_\times.
$$

In world coordinates, the increment appears on the left:

$$
\Delta R_W=R_+R_-^\top,\qquad R_+=\Delta R_W R_-.
$$

Insert $R_+=R_-\Delta R_B$:

$$
\Delta R_W=R_-\Delta R_B R_-^\top.
$$

Conjugation rotates the increment's axis. Away from principal-angle ambiguity:

$$
\operatorname{Log}(\Delta R_W)
=R_-\operatorname{Log}(\Delta R_B).
$$

Body and world rates describe the same turning using different axes. Infinitesimally,
$\omega_W=R\omega_B$ and $[\omega_W]_\times=\dot R R^\top$.

The finite-log estimate is exact for constant angular-velocity exponential motion when the principal increment
is unambiguous. For changing angular velocity it summarizes an interval; it need not equal an instantaneous
endpoint or midpoint rate. Extra complete turns between samples cannot be recovered.

The repository computes:

```python
rel = quat_mul(quat_conj(q[:-1][ok]), q[1:][ok])
return 0.5 * (tau[:-1][ok] + tau[1:][ok]), log_map(rel) / dt[ok][:, None]
```

For $M$ accepted intervals, it returns midpoint times $(M,)$ in seconds and rates $(M,3)$ in rad/s.
The Boolean filter accepts positive gaps within the configured maximum.
[mesqsim/estimators/clock.py:159](../../mesqsim/estimators/clock.py#L159)

A constant private heading cancels. If $\hat R(t)=Y(-\psi)P(t)$ with constant $\psi$, then

$$
\hat R_-^\top\hat R_+
=P_-^\top Y(-\psi)^\top Y(-\psi)P_+
=P_-^\top P_+.
$$

A time-varying heading error does not cancel this way.
[docs/MATH.md:232](../MATH.md)

The physical helper `imu_truth` exposes a continuous body rate from a `RotationSpline` derivative.
There is a further implementation detail: the twin session feeds `ideal_gyro_rad_s(imu.q, dt)` into its
sensor-error model. That helper uses discrete increments so ideal exponential gyro propagation reproduces each
sampled relative rotation. Sensor errors and accelerometer corrections can still change the fused output.
Distinguish the continuous spline rate from the discrete gyro supplied to fusion.
[mesqsim/twin/physics.py:61](../../mesqsim/twin/physics.py#L61),
[mesqsim/twin/sensors.py:96](../../mesqsim/twin/sensors.py#L96),
[mesqsim/twin/session.py:427](../../mesqsim/twin/session.py#L427)

#### L. Slerp and time interpolation

Find the relative rotation and its principal vector:

$$
d=q_0^{-1}\otimes q_1,\qquad \phi=\operatorname{Log}(d).
$$

Scaling $\phi$ keeps its axis and scales its angle. Starting at $q_0$ gives

$$
q(u)=q_0\otimes\operatorname{Exp}(u\phi).
$$

At $u=0$ the increment is identity; at $u=1$ the orientation equals $q_1$. The angle is proportional to $u$,
so speed is constant along the path. This is spherical linear interpolation, or slerp.
The principal Log chooses a shortest path; at relative angle $\pi$ a shortest path is not unique.
[mesqsim/conventions.py:232](../../mesqsim/conventions.py#L232)

At $u=1$ the components may equal $-q_1$ rather than $q_1$. The orientation agrees either way, so the endpoint
test compares canonicalized quaternions.
[tests/test_conventions.py:133](../../tests/test_conventions.py#L133)

For query time $t$ bracketed by $t_0,t_1$:

$$
u=\frac{t-t_0}{t_1-t_0}.
$$

`interp_quat` finds the bracket, clamps $u$ into $[0,1]$, and holds endpoint orientations outside the source
range. Increasing source times are a precondition rather than a fully checked property. A singleton source is
broadcast as a copy.
[mesqsim/conventions.py:244](../../mesqsim/conventions.py#L244)

Interpolation alone does not guarantee causality: a bracketing sample can be later than the query time.
An inference caller must restrict which samples have arrived by its analysis time.

#### M. Wrapping and the topology of an angle

Directions repeat every full turn. For half-period $H$, either $\pi$ radians or $180^\circ$, first compute

$$
y=((x+H)\bmod 2H)-H.
$$

Modulo puts $x+H$ in $[0,2H)$, then subtraction gives $[-H,H)$. The code moves $-H$ to $+H$, yielding
$(-H,H]$.
[mesqsim/conventions.py:189](../../mesqsim/conventions.py#L189)

For example,

$$
\operatorname{wrap}(190^\circ)=-170^\circ,\quad
\operatorname{wrap}(-180^\circ)=180^\circ,\quad
\operatorname{wrap}(540^\circ)=180^\circ.
$$

Compute angular error after wrapping the difference: $179^\circ$ and $-179^\circ$ are two degrees apart,
not 358 degrees apart.
[tests/test_conventions.py:102](../../tests/test_conventions.py#L102),
[mesqsim/learn/scorer.py:145](../../mesqsim/learn/scorer.py#L145)

#### N. Circular means, resultant length, and circular standard deviation

The complex number $i$ satisfies $i^2=-1$. Euler's identity packages a unit-circle vector:

$$
e^{i\theta}=\cos\theta+i\sin\theta.
$$

Represent each angle as $z_j=e^{i\theta_j}$, then average:

$$
\bar z=\frac1N\sum_jz_j=C+iS,
$$

$$
C=\frac1N\sum_j\cos\theta_j,\qquad
S=\frac1N\sum_j\sin\theta_j.
$$

The mean direction and mean resultant length are

$$
\bar\theta=\operatorname{atan2}(S,C),\qquad
r=|\bar z|=\sqrt{C^2+S^2}.
$$

The triangle inequality gives $0\le r\le1$. Aligned samples give one; cancellation reduces it.
Why choose this direction? For a proposed mean $\mu$:

$$
|z_j-e^{i\mu}|^2
=2-2\operatorname{Re}(z_je^{-i\mu})
=2-2\cos(\theta_j-\mu).
$$

Summing and expanding the cosine:

$$
\sum_j|z_j-e^{i\mu}|^2=2N-2N(C\cos\mu+S\sin\mu).
$$

Minimizing that sum means maximizing the dot product of $(C,S)$ with the unit vector
$(\cos\mu,\sin\mu)$. It must point along $(C,S)$.

If $r=0$ there is no preferred direction. The code returns an `atan2` value, but that number does not identify a
mathematically meaningful mean.
[mesqsim/conventions.py:282](../../mesqsim/conventions.py#L282)

The implemented circular standard deviation is

$$
\sigma_c=\sqrt{-2\log r}.
$$

For a wrapped normal, the first circular moment has length $r=e^{-\sigma^2/2}$. Taking logs and solving gives
the expression above. Section Q derives this Gaussian moment. Wrapping does not change $e^{i\theta}$, since
$e^{i(\theta+2k\pi)}=e^{i\theta}$.

At $r=1$, the spread is zero; as $r\to0$, mathematical spread diverges. The code clips $r$ to
$[10^{-300},1]$ before the logarithm so evaluation remains finite.
[mesqsim/conventions.py:288](../../mesqsim/conventions.py#L288)

#### O. Averaging nearby quaternions

A component-wise mean of $q$ and $-q$ is zero even though both represent one orientation. The repository first
aligns signs: compare each sample's four-dimensional dot product with the first quaternion; flip the sample
where that dot product is negative.

For nearby unit quaternions, this selects representatives on the same side of the four-dimensional unit sphere.
Then

$$
\bar q_{\rm raw}=\frac1N\sum_j\tilde q_j,\qquad
\bar q=\frac{\bar q_{\rm raw}}{\|\bar q_{\rm raw}\|},
$$

followed by canonicalization. `quat_mean` assumes nearby unit inputs, such as a box or T-pose window.
It does not normalize every input separately and is not a general intrinsic/geodesic mean for widely separated
rotations.
[mesqsim/conventions.py:295](../../mesqsim/conventions.py#L295)

#### P. Cardan angles and generic extraction

A Cardan, or Tait–Bryan, sequence uses three distinct axes. For example,
$R=R_X(a)R_Z(b)R_Y(c)$ is intrinsic `"XZY"` here. All six distinct-axis sequences are supported.
[mesqsim/conventions.py:311](../../mesqsim/conventions.py#L311),
[tests/test_conventions_step2.py:26](../../tests/test_conventions_step2.py#L26)

First multiply the XYZ case:

$$
R_X(a)R_Y(b)R_Z(c)=
\begin{bmatrix}
\cos b\cos c&-\cos b\sin c&\sin b\\
\cos a\sin c+\sin a\sin b\cos c&
\cos a\cos c-\sin a\sin b\sin c&-\sin a\cos b\\
\sin a\sin c-\cos a\sin b\cos c&
\sin a\cos c+\cos a\sin b\sin c&\cos a\cos b
\end{bmatrix}.
$$

For $|b|<\pi/2$, the last column and first row isolate

$$
b=\arcsin R_{02},\quad
a=\operatorname{atan2}(-R_{12},R_{22}),\quad
c=\operatorname{atan2}(-R_{01},R_{00}).
$$

For general axis indices $(i,j,k)$, define parity $e=+1$ for cyclic XYZ, YZX, ZXY and $e=-1$ for the other
orders. Swapping two axes reverses the handedness of the ordered basis and supplies the parity sign.
The relevant entries become

$$
R_{ik}=e\sin b,
$$

$$
R_{jk}=-e\sin a\cos b,\qquad R_{kk}=\cos a\cos b,
$$

$$
R_{ij}=-e\cos b\sin c,\qquad R_{ii}=\cos b\cos c.
$$

The common positive factor $\cos b$ cancels in each `atan2`:

$$
b=\arcsin(eR_{ik}),\qquad
a=\operatorname{atan2}(-eR_{jk},R_{kk}),\qquad
c=\operatorname{atan2}(-eR_{ij},R_{ii}).
$$

These are the code's extraction formulas.
[mesqsim/conventions.py:320](../../mesqsim/conventions.py#L320)

At $b=\pm\pi/2$, $\cos b=0$ and the decomposition loses uniqueness: gimbal lock. The implementation applies
the formulas without a special ambiguity-resolving branch.
It uppercases `seq`, so lowercase input is also intrinsic here; do not transfer SciPy's lowercase/extrinsic
distinction into this wrapper. Repeated-axis proper Euler sequences such as `"XYX"` are rejected.
[mesqsim/conventions.py:320](../../mesqsim/conventions.py#L320),
[tests/test_conventions_step2.py:39](../../tests/test_conventions_step2.py#L39)

#### Q. Circular uncertainty and every von Mises helper

These functions complete the current conventions file. Their application in an estimator comes later.

**Probability vocabulary.** A density $p(x)$ integrates to one. Probability over an interval is its integral over
that interval. Expectation means a weighted integral: $E[f(X)]=\int f(x)p(x)\,dx$.

The von Mises distribution is a density on a circle:

$$
p(x\mid\mu,\kappa)
=\frac{\exp(\kappa\cos(x-\mu))}{2\pi I_0(\kappa)},\qquad \kappa\ge0.
$$

Here $\mu$ is mean direction, $\kappa$ is dimensionless concentration, and the modified Bessel function $I_0$
supplies normalization. At $\kappa=0$, it is uniform with density $1/(2\pi)$. For large $\kappa$, most mass lies
near $\mu$. Put $e=x-\mu$ and use $\cos e\approx1-e^2/2$:

$$
e^{\kappa\cos e}\approx e^\kappa e^{-\kappa e^2/2}.
$$

Normalization absorbs $e^\kappa$; the remaining shape is Gaussian with variance $1/\kappa$.
[mesqsim/conventions.py:337](../../mesqsim/conventions.py#L337)

**Resultant function.** For a zero-centered density define

$$
Z(\kappa)=\int_{-\pi}^{\pi}e^{\kappa\cos x}\,dx=2\pi I_0(\kappa).
$$

Differentiating under the integral:

$$
Z'(\kappa)=\int_{-\pi}^{\pi}\cos x\,e^{\kappa\cos x}\,dx.
$$

Dividing by $Z$ gives

$$
A(\kappa)=E[\cos X]=\frac{Z'}Z=\frac{I_1(\kappa)}{I_0(\kappa)}.
$$

Symmetry makes $E[\sin X]=0$. Thus $A$ is mean resultant length, and a shifted distribution has
$E[e^{iX}]=e^{i\mu}A(\kappa)$.

`vm_mean_resultant` evaluates $A$ through `vm_one_minus_resultant`, preserving accuracy when $A$ is close
to one. Below $\kappa=1000$, scaled Bessel functions avoid overflow; their common scaling cancels from the ratio.
At and above 1000 the source uses the asymptotic approximation

$$
1-A(\kappa)\approx
\frac1{2\kappa}+\frac1{8\kappa^2}+\frac1{8\kappa^3}+\frac{25}{128\kappa^4}.
$$

The coefficients belong to the numerical approximation documented in the source, not a new exact identity.
[mesqsim/conventions.py:349](../../mesqsim/conventions.py#L349),
[tests/test_conventions_step2.py:51](../../tests/test_conventions_step2.py#L51),
[tests/test_conventions_step2.py:60](../../tests/test_conventions_step2.py#L60)

**Derivative and inverse.** Differentiate $A=Z'/Z$:

$$
A'=\frac{Z''}Z-\left(\frac{Z'}Z\right)^2
=E[\cos^2X]-E[\cos X]^2.
$$

Integrate a total derivative over one period:

$$
0=\int_{-\pi}^{\pi}\frac{d}{dx}
\left(\sin x\,e^{\kappa\cos x}\right)dx
=\int_{-\pi}^{\pi}(\cos x-\kappa\sin^2x)e^{\kappa\cos x}\,dx.
$$

After dividing by $Z$, $E[\sin^2X]=A/\kappa$. Since $\cos^2X=1-\sin^2X$:

$$
A'=1-\frac A\kappa-A^2.
$$

At zero concentration, the uniform average of $\cos^2X$ is $1/2$, giving $A'(0)=1/2$.
`_vm_dA` uses this identity and a separate large-concentration approximation.
[mesqsim/conventions.py:368](../../mesqsim/conventions.py#L368)

Let $d=1-A(\kappa)$. The inverse helper calls it `eps`; it is unrelated to phone-relative target $\epsilon_i$.
To solve for concentration, set

$$
g(\kappa)=1-A(\kappa)-d,\qquad g'(\kappa)=-A'(\kappa).
$$

Newton's method solves the local tangent-line approximation
$0\approx g(\kappa)+g'(\kappa)(\kappa_{\rm new}-\kappa)$:

$$
\kappa_{\rm new}=\kappa-\frac{g}{g'}
=\kappa+\frac{1-A(\kappa)-d}{A'(\kappa)}.
$$

The implementation starts from a piecewise approximation, takes six Newton steps, and clips to
`KAPPA_MAX = 1e12`. For $d\ge1$ it returns zero; for $d\le0$ it returns the cap.
`vm_kappa_from_resultant(r)` delegates with $d=1-r$.
[mesqsim/conventions.py:343](../../mesqsim/conventions.py#L343),
[mesqsim/conventions.py:378](../../mesqsim/conventions.py#L378),
[mesqsim/conventions.py:401](../../mesqsim/conventions.py#L401)

**Product of beliefs.** Multiplying densities adds log densities. Ignoring constants independent of $x$:

$$
\kappa_1\cos(x-\mu_1)+\kappa_2\cos(x-\mu_2)=C\cos x+S\sin x,
$$

$$
C=\kappa_1\cos\mu_1+\kappa_2\cos\mu_2,\qquad
S=\kappa_1\sin\mu_1+\kappa_2\sin\mu_2.
$$

Set $\kappa=\sqrt{C^2+S^2}$ and $\mu=\operatorname{atan2}(S,C)$. Then
$C=\kappa\cos\mu$, $S=\kappa\sin\mu$, so the exponent is $\kappa\cos(x-\mu)$: another von Mises density.
Equivalently,

$$
\kappa e^{i\mu}=\kappa_1e^{i\mu_1}+\kappa_2e^{i\mu_2}.
$$

`vm_product` performs this complex-vector addition and caps concentration. A factor with $\kappa=0$ adds
no information; equal opposite vectors cancel. At resulting concentration zero, the mean carries no information.
[mesqsim/conventions.py:406](../../mesqsim/conventions.py#L406)

**Gaussian circular moment.** A zero-mean Gaussian of variance $v>0$ has density

$$
f(w)=\frac1{\sqrt{2\pi v}}e^{-w^2/(2v)},\qquad f'(w)=-wf(w)/v.
$$

Define the characteristic function $\varphi(t)=E[e^{itW}]$. Differentiate and substitute $wf(w)=-vf'(w)$:

$$
\varphi'(t)
=i\int w e^{itw}f(w)\,dw
=-iv\int e^{itw}f'(w)\,dw.
$$

Integration by parts has vanishing boundary terms because the Gaussian decays at infinity:

$$
\int e^{itw}f'(w)\,dw=-it\int e^{itw}f(w)\,dw=-it\varphi(t).
$$

Hence $\varphi'(t)=-vt\varphi(t)$. With $\varphi(0)=1$, solving gives

$$
\varphi(t)=e^{-vt^2/2}.
$$

At $t=1$, $E[e^{iW}]=e^{-v/2}$. Shifting mean to $\mu$ adds factor $e^{i\mu}$.
This also derives the wrapped-normal relation used in circular standard deviation.

**Prediction under diffusion.** Suppose independent heading increment $W$ is Gaussian with variance $D\Delta t$.
Here $D$ is a diffusion variance rate in rad²/s, named `q_rad2_per_s` in code, not a quaternion.
Independence gives

$$
E[e^{i(X+W)}]=E[e^{iX}]E[e^{iW}]
=e^{i\mu}A(\kappa)e^{-D\Delta t/2}.
$$

Match that first circular moment to a new von Mises belief:

$$
A(\kappa')=A(\kappa)e^{-D\Delta t/2}.
$$

This is moment matching: the complete diffused density is generally not exactly von Mises.
For accuracy, set $d=1-A(\kappa)$ and $x=D\Delta t/2$:

$$
d'=1-(1-d)e^{-x}=de^{-x}-(e^{-x}-1).
$$

The final difference is computed using `expm1(-x)` to avoid subtraction of nearly equal numbers.
[mesqsim/conventions.py:417](../../mesqsim/conventions.py#L417)

For concentrated beliefs and small diffusion:

$$
1-\frac1{2\kappa'}
\approx\left(1-\frac1{2\kappa}\right)\left(1-\frac{D\Delta t}{2}\right).
$$

Discard products of small corrections:

$$
\frac1{\kappa'}\approx\frac1\kappa+D\Delta t.
$$

Approximate angular variances add. For valid nonnegative diffusion, elapsed time cannot increase concentration.
Nonpositive elapsed time leaves it unchanged.
[mesqsim/conventions.py:425](../../mesqsim/conventions.py#L425)

**Central intervals.** Let $F$ be the cumulative probability for a symmetric zero-centered von Mises density.
A central interval with half-width $h$ and probability $\ell$ satisfies

$$
P(-h\le X\le h)=F(h)-F(-h)=2F(h)-1=\ell.
$$

Thus $F(h)=(1+\ell)/2$: $h$ is that quantile. For uniform density,

$$
\ell=\frac{2h}{2\pi}\quad\Longrightarrow\quad h=\pi\ell.
$$

`_halfwidth_table` caches 1101 quantiles over $\log_{10}\kappa\in[-4,7]$. `vm_halfwidth_rad` interpolates
in log concentration, uses the uniform approximation below $10^{-4}$, and above $10^7$ uses

$$
h\approx z_{(1+\ell)/2}/\sqrt\kappa,
$$

where $z_a$ is the standard-normal quantile at cumulative probability $a$.
[mesqsim/conventions.py:436](../../mesqsim/conventions.py#L436),
[mesqsim/conventions.py:447](../../mesqsim/conventions.py#L447)

**Log density without overflow.** Take the logarithm:

$$
\log p(x\mid0,\kappa)
=\kappa\cos x-\log(2\pi)-\log I_0(\kappa).
$$

For $\kappa\ge0$, $I_0(\kappa)=e^\kappa i0e(\kappa)$. Substitute and collect:

$$
\log p
=\kappa\cos x-\kappa-\log(2\pi)-\log i0e(\kappa)
=\kappa(\cos x-1)-\log(2\pi\,i0e(\kappa)).
$$

This is `vm_logpdf`. For a nonzero mean, supply angular error $x-\mu$.
[mesqsim/conventions.py:463](../../mesqsim/conventions.py#L463)

### 1.4 The code, traced function by function

For the tables below:

- $B$ is a tuple of leading batch dimensions. $B+(4,)$ appends a component dimension of 4; for example
  $B=(T,15)$ gives quaternion shape $(T,15,4)$. A single quaternion has shape $(4,)$.
- $Q$ is the shape of a query-time array.
- Rotation helpers use `float64` arrays. Quaternions and matrices are dimensionless. Rotation vectors and angles
  use radians unless a function name says degrees. A rotated physical vector keeps its original units.
- These helpers are available to simulation and deployment. Whether an input is hidden truth depends on the caller.

There are 38 functions, including private helpers `_wrap`, `_vm_dA`, and `_halfwidth_table`. The file also defines
`QUAT_ORDER` and the numerical cap `KAPPA_MAX`.
[mesqsim/conventions.py:66](../../mesqsim/conventions.py#L66),
[mesqsim/conventions.py:343](../../mesqsim/conventions.py#L343)

#### Quaternion construction and algebra

| Function and implementation | Shape and important behavior | Existing test |
|---|---|---|
| `as_quat`, [mesqsim/conventions.py:70](../../mesqsim/conventions.py#L70) | Convert to `float64` and check last dimension is 4. Neither normalize nor canonicalize. | Indirect coverage; no standalone invocation. |
| `quat_identity`, [mesqsim/conventions.py:78](../../mesqsim/conventions.py#L78) | Leading shape $B$ → $B+(4,)$, filled with $(0,0,0,1)$. | No direct test. |
| `quat_mul`, [mesqsim/conventions.py:85](../../mesqsim/conventions.py#L85) | Broadcast inputs → $B+(4,)$. Hamilton composition; no automatic normalization or canonicalization. | [tests/test_conventions.py:56](../../tests/test_conventions.py#L56), [tests/test_conventions.py:65](../../tests/test_conventions.py#L65) |
| `quat_conj`, [mesqsim/conventions.py:105](../../mesqsim/conventions.py#L105) | Retain $B+(4,)$; negate xyz. Equals inverse for unit input. | Used in [tests/test_conventions.py:138](../../tests/test_conventions.py#L138); no standalone inverse test. |
| `quat_inv`, [mesqsim/conventions.py:111](../../mesqsim/conventions.py#L111) | Conjugate divided by squared norm; shape preserved. | No direct test. |
| `quat_normalize`, [mesqsim/conventions.py:117](../../mesqsim/conventions.py#L117) | Divide by final-axis norm; shape preserved. | Used throughout, including [tests/test_conventions.py:38](../../tests/test_conventions.py#L38). |
| `quat_canonical`, [mesqsim/conventions.py:123](../../mesqsim/conventions.py#L123) | Flip all components where $w<0$; retain shape; exact $w=0$ unchanged. | [tests/test_conventions.py:84](../../tests/test_conventions.py#L84) |
| `quat_rotate`, [mesqsim/conventions.py:132](../../mesqsim/conventions.py#L132) | Unit $q$ ending in 4 and vectors ending in 3 → broadcast vectors ending in 3. Does not normalize. | [tests/test_conventions.py:42](../../tests/test_conventions.py#L42), [tests/test_conventions.py:75](../../tests/test_conventions.py#L75) |
| `quat_to_matrix`, [mesqsim/conventions.py:142](../../mesqsim/conventions.py#L142) | $B+(4,)$ → $B+(3,3)$; normalize first. | SciPy comparison/roundtrip, [tests/test_conventions.py:48](../../tests/test_conventions.py#L48). |
| `quat_from_matrix`, [mesqsim/conventions.py:159](../../mesqsim/conventions.py#L159) | $B+(3,3)$ → $B+(4,)$; flatten batch, call SciPy, restore batch, canonicalize. | [tests/test_conventions.py:51](../../tests/test_conventions.py#L51) |

Read the start of `quat_mul` literally:

```python
a = as_quat(a)
b = as_quat(b)
ax, ay, az, aw = np.moveaxis(a, -1, 0)
bx, by, bz, bw = np.moveaxis(b, -1, 0)
```

Conversion fixes dtype and component count. Moving the last axis to the front exposes each component while
preserving batch dimensions. The following output expressions are the Hamilton product:

```python
aw * bx + ax * bw + ay * bz - az * by,
aw * by - ax * bz + ay * bw + az * bx,
aw * bz + ax * by - ay * bx + az * bw,
aw * bw - ax * bx - ay * by - az * bz,
```

The first three are $w_au_b+w_bu_a+u_a\times u_b$; the fourth is $w_aw_b-u_a^\top u_b$.
`np.stack(..., axis=-1)` restores xyzw order.
[mesqsim/conventions.py:90](../../mesqsim/conventions.py#L90)

#### Heading, maps, interpolation, and means

| Function and implementation | Shape, units, and code trace | Existing test |
|---|---|---|
| `yaw_quat`, [mesqsim/conventions.py:168](../../mesqsim/conventions.py#L168) | Angles of shape $B$ rad → $B+(4,)$; stack zeros and half-angle sine/cosine. | [tests/test_conventions.py:42](../../tests/test_conventions.py#L42) |
| `heading_rad`, [mesqsim/conventions.py:175](../../mesqsim/conventions.py#L175) | $B+(4,)$ → shape $B$ rad. Compute `r10`, `r00`, then `atan2`. Works for nonzero scaled quaternions. | [tests/test_conventions.py:93](../../tests/test_conventions.py#L93) |
| `_wrap`, [mesqsim/conventions.py:189](../../mesqsim/conventions.py#L189) | Preserve shape; modulo followed by endpoint replacement. | Through wrappers, [tests/test_conventions.py:102](../../tests/test_conventions.py#L102). |
| `wrap_rad`, [mesqsim/conventions.py:196](../../mesqsim/conventions.py#L196) | Shape preserved; radians into $(-\pi,\pi]$. | [tests/test_conventions.py:109](../../tests/test_conventions.py#L109) |
| `wrap_deg`, [mesqsim/conventions.py:201](../../mesqsim/conventions.py#L201) | Shape preserved; degrees into $(-180,180]$. | [tests/test_conventions.py:102](../../tests/test_conventions.py#L102), [tests/test_conventions.py:115](../../tests/test_conventions.py#L115) |
| `exp_map`, [mesqsim/conventions.py:206](../../mesqsim/conventions.py#L206) | $B+(3,)$ rad → $B+(4,)$; norm, sinc-scaled xyz, cosine scalar. | [tests/test_conventions.py:122](../../tests/test_conventions.py#L122) |
| `log_map`, [mesqsim/conventions.py:215](../../mesqsim/conventions.py#L215) | $B+(4,)$ → $B+(3,)$ rad; canonicalize, principal angle, scaled vector part. | [tests/test_conventions.py:122](../../tests/test_conventions.py#L122) |
| `geodesic_angle_rad`, [mesqsim/conventions.py:226](../../mesqsim/conventions.py#L226) | $B+(4,)$ → shape $B$ rad in $[0,\pi]$. | [tests/test_conventions.py:126](../../tests/test_conventions.py#L126) |
| `quat_slerp`, [mesqsim/conventions.py:232](../../mesqsim/conventions.py#L232) | Endpoints and dimensionless fractions → broadcast shape ending in 4. Normalize, relative product, Log, scale, Exp, compose. Does not clamp $u$. | [tests/test_conventions.py:133](../../tests/test_conventions.py#L133) |
| `interp_quat`, [mesqsim/conventions.py:244](../../mesqsim/conventions.py#L244) | Source $(N,)$ seconds, $(N,4)$ quaternions, query shape $Q$ → $Q+(4,)$. Find bracket and slerp; singleton source copied. | [tests/test_conventions.py:141](../../tests/test_conventions.py#L141) |
| `gravity_in_body`, [mesqsim/conventions.py:263](../../mesqsim/conventions.py#L263) | $B+(4,)$ → dimensionless up vector $B+(3,)$; normalize, conjugate, rotate world up. | [tests/test_conventions.py:146](../../tests/test_conventions.py#L146) |
| `wxyz_to_xyzw`, [mesqsim/conventions.py:270](../../mesqsim/conventions.py#L270) | $B+(4,)$ → $B+(4,)$; move scalar first to scalar last. | [tests/test_conventions.py:52](../../tests/test_conventions.py#L52) |
| `xyzw_to_wxyz`, [mesqsim/conventions.py:276](../../mesqsim/conventions.py#L276) | $B+(4,)$ → $B+(4,)$; move scalar last to first. Neither reordering changes coordinate frames. | [tests/test_conventions.py:53](../../tests/test_conventions.py#L53) |
| `circ_mean_rad`, [mesqsim/conventions.py:282](../../mesqsim/conventions.py#L282) | Reduce selected axis; radians in/out. Mean sine/cosine, `atan2`, wrap. | [tests/test_conventions.py:153](../../tests/test_conventions.py#L153) |
| `circ_std_rad`, [mesqsim/conventions.py:288](../../mesqsim/conventions.py#L288) | Reduce selected axis; radians in/out. Resultant length, clipping, square root of $-2\log r$. | [tests/test_conventions.py:153](../../tests/test_conventions.py#L153) |
| `quat_mean`, [mesqsim/conventions.py:295](../../mesqsim/conventions.py#L295) | Reduce sample axis, retain last dimension 4; align signs, average, normalize, canonicalize. Nearby unit inputs assumed. | [tests/test_conventions.py:161](../../tests/test_conventions.py#L161) |
| `matrix_to_cardan_rad`, [mesqsim/conventions.py:311](../../mesqsim/conventions.py#L311) | $B+(3,3)$ → $B+(3,)$ rad. Validate distinct axes, extract angles. | [tests/test_conventions_step2.py:29](../../tests/test_conventions_step2.py#L29), [tests/test_conventions_step2.py:39](../../tests/test_conventions_step2.py#L39) |
| `quat_to_cardan_rad`, [mesqsim/conventions.py:332](../../mesqsim/conventions.py#L332) | $B+(4,)$ → $B+(3,)$ rad; convert to matrix and delegate. | [tests/test_conventions_step2.py:29](../../tests/test_conventions_step2.py#L29), [tests/test_conventions_step2.py:108](../../tests/test_conventions_step2.py#L108) |

Heading uses these exact lines:

```python
r10 = 2.0 * (x * y + w * z)
r00 = w * w + x * x - y * y - z * z
return np.arctan2(r10, r00)
```

For unit $q$, `r00` equals $1-2(y^2+z^2)$. Scaling a nonzero quaternion by a real factor multiplies both
arguments by that factor squared, so `atan2` gives the same heading.
[mesqsim/conventions.py:184](../../mesqsim/conventions.py#L184)

#### Circular uncertainty

| Function and implementation | Inputs → outputs and implementation | Existing test |
|---|---|---|
| `vm_one_minus_resultant`, [mesqsim/conventions.py:349](../../mesqsim/conventions.py#L349) | Concentration → dimensionless $1-A$; shape preserved. Scaled Bessel ratio/asymptotic series. | [tests/test_conventions_step2.py:51](../../tests/test_conventions_step2.py#L51), [tests/test_conventions_step2.py:60](../../tests/test_conventions_step2.py#L60) |
| `vm_mean_resultant`, [mesqsim/conventions.py:363](../../mesqsim/conventions.py#L363) | Concentration → dimensionless $A$; one minus preceding helper. | [tests/test_conventions_step2.py:51](../../tests/test_conventions_step2.py#L51) |
| `_vm_dA`, [mesqsim/conventions.py:368](../../mesqsim/conventions.py#L368) | Concentration and $1-A$ → derivative $A'$. | Indirectly through inverse roundtrips. |
| `vm_kappa_from_one_minus_resultant`, [mesqsim/conventions.py:378](../../mesqsim/conventions.py#L378) | Dimensionless $1-A$ → same-shaped concentration. Initial approximation, Newton, clipping. | Indirectly through inverse/prediction tests. |
| `vm_kappa_from_resultant`, [mesqsim/conventions.py:401](../../mesqsim/conventions.py#L401) | Dimensionless $A$ → concentration; delegate with $1-A$. | [tests/test_conventions_step2.py:51](../../tests/test_conventions_step2.py#L51) |
| `vm_product`, [mesqsim/conventions.py:406](../../mesqsim/conventions.py#L406) | Two angles in rad and concentrations → wrapped mean in rad and capped concentration; broadcast compatible shapes. | [tests/test_conventions_step2.py:65](../../tests/test_conventions_step2.py#L65) |
| `vm_predict_kappa`, [mesqsim/conventions.py:417](../../mesqsim/conventions.py#L417) | Concentration, diffusion rad²/s, elapsed seconds → concentration. Use `expm1` for accuracy; nonpositive time returns input. | [tests/test_conventions_step2.py:79](../../tests/test_conventions_step2.py#L79) |
| `_halfwidth_table`, [mesqsim/conventions.py:436](../../mesqsim/conventions.py#L436) | Scalar level → cached `float64` arrays of log concentrations and half-widths in rad, each $(1101,)$. | Indirectly through intervals. |
| `vm_halfwidth_rad`, [mesqsim/conventions.py:447](../../mesqsim/conventions.py#L447) | Concentration array and scalar level → same-shaped half-widths in rad. Uniform/table/normal branches. | [tests/test_conventions_step2.py:90](../../tests/test_conventions_step2.py#L90) |
| `vm_logpdf`, [mesqsim/conventions.py:463](../../mesqsim/conventions.py#L463) | Errors in rad and concentration → broadcast log-density, centered at zero. | [tests/test_conventions_step2.py:102](../../tests/test_conventions_step2.py#L102) |

These generally coerce inputs to `float64`. `vm_product` does not explicitly coerce every input, so its
concentration output can follow input dtype.
[mesqsim/conventions.py:413](../../mesqsim/conventions.py#L413)

The helpers assume meaningful inputs. Normalization and inversion do not guard against zero quaternions.
The circular tools assume suitable nonnegative concentration/diffusion and a valid probability level; do not infer
validation absent from the source.
[mesqsim/conventions.py:111](../../mesqsim/conventions.py#L111),
[mesqsim/conventions.py:425](../../mesqsim/conventions.py#L425),
[mesqsim/conventions.py:455](../../mesqsim/conventions.py#L455)

### 1.5 Worked numerical examples, by hand and checked against the code

These are synthetic constructions, not recording measurements.

#### Example A: Y(90 degrees)

Insert $\cos90^\circ=0$ and $\sin90^\circ=1$:

$$
Y(90^\circ)=
\begin{bmatrix}0&-1&0\\1&0&0\\0&0&1\end{bmatrix}.
$$

Therefore

$$
Y(90^\circ)\begin{bmatrix}1\\0\\0\end{bmatrix}
=\begin{bmatrix}0\\1\\0\end{bmatrix}.
$$

Its scalar-last quaternion is

$$
q_Y(90^\circ)=(0,0,\sin45^\circ,\cos45^\circ)
\approx(0,0,0.707107,0.707107).
$$

It maps +x to +y, +y to −x, and leaves +z fixed, matching the sign test.
[tests/test_conventions.py:42](../../tests/test_conventions.py#L42)

#### Example B: composing two yaws

Yaw vector parts are parallel, so their cross product vanishes. The Hamilton product gives

$$
z=\cos\frac a2\sin\frac b2+\sin\frac a2\cos\frac b2=\sin\frac{a+b}{2},
$$

$$
w=\cos\frac a2\cos\frac b2-\sin\frac a2\sin\frac b2=\cos\frac{a+b}{2}.
$$

The x and y components remain zero. Hence $q_Y(a)\otimes q_Y(b)=q_Y(a+b)$ and

$$
Y(60^\circ)Y(30^\circ)=Y(90^\circ).
$$

Checking with `quat_mul` and `heading_rad` gives approximately $90^\circ$; floating-point output can be
`89.99999999999999`.

Yaws commute because they share an axis. General rotations do not. Apply $R_z(90^\circ)R_x(90^\circ)$
to $e_y$: the x rotation sends it to $e_z$, then z leaves it unchanged. Reverse the order: z sends it to $-e_x$,
then x leaves that unchanged. The outputs differ, as the repository checks.
[tests/test_conventions.py:65](../../tests/test_conventions.py#L65)

#### Example C: pitch 30 degrees, then yaw 60 degrees

Start at identity, pitch about fixed world +Y, then yaw about fixed world +Z:

$$
R=Y(60^\circ)R_y(30^\circ)
=
\begin{bmatrix}
1/2&-\sqrt3/2&0\\\sqrt3/2&1/2&0\\0&0&1
\end{bmatrix}
\begin{bmatrix}
\sqrt3/2&0&1/2\\0&1&0\\-1/2&0&\sqrt3/2
\end{bmatrix}.
$$

Multiplication gives

$$
R=
\begin{bmatrix}
\sqrt3/4&-\sqrt3/2&1/4\\
3/4&1/2&\sqrt3/4\\
-1/2&0&\sqrt3/2
\end{bmatrix}
\approx
\begin{bmatrix}
0.433013&-0.866025&0.250000\\
0.750000&0.500000&0.433013\\
-0.500000&0&0.866025
\end{bmatrix}.
$$

For example, the top-left entry is $(1/2)(\sqrt3/2)+(-\sqrt3/2)(0)=\sqrt3/4$.
Thus

$$
h=\operatorname{atan2}(0.75,0.433013)=60^\circ,\qquad
R^\top e_z=(-1/2,0,\sqrt3/2)^\top.
$$

Quaternion multiplication gives

$$
q\approx(-0.129410,\ 0.224144,\ 0.482963,\ 0.836516).
$$

Its negative x component comes from $e_z\times e_y=-e_x$ in the product of yaw and pitch vector parts.
It does not mean negative roll. Quaternion components are not Euler angles.

Apply another world yaw of $50^\circ$:
$R'=Y(50^\circ)R=Y(110^\circ)R_y(30^\circ)$.
Heading becomes $110^\circ$; the body-coordinate up vector stays unchanged.

Run from the repository root:

```bash
.venv/bin/python - <<'PY'
import numpy as np
from mesqsim import conventions as c
d = np.deg2rad
q90 = c.yaw_quat(d(90))
q = c.quat_mul(c.yaw_quat(d(60)), c.exp_map([0, d(30), 0]))
print(np.round(q90, 6))
print(np.round(c.quat_to_matrix(q90), 6))
print(np.round(q, 6))
print(np.rad2deg(c.heading_rad(q)))
print(np.round(c.gravity_in_body(q), 6))
print(np.rad2deg(c.heading_rad(c.quat_mul(c.yaw_quat(d(50)), q))))
PY
```

Expected values, allowing tiny floating-point differences and signed zeros:

```text
[0.       0.       0.707107 0.707107]
[[ 0. -1.  0.]
 [ 1.  0.  0.]
 [ 0.  0.  1.]]
[-0.12941   0.224144  0.482963  0.836516]
59.99999999999999
[-0.5       0.        0.866025]
110.0
```

#### Example D: the same turning in body and world coordinates

Choose

$$
R_0=R_y(30^\circ),\qquad R_1=Y(10^\circ)R_0,\qquad \Delta t=0.1\ {\rm s}.
$$

Then

$$
R_1R_0^\top=Y(10^\circ)R_0R_0^\top=Y(10^\circ),
$$

so

$$
\omega_W=(0,0,100^\circ/{\rm s})^\top
=(0,0,1.745329)^\top\ {\rm rad/s}.
$$

Express its axis in the earlier body's coordinates:

$$
\omega_B=R_0^\top\omega_W
=\begin{bmatrix}
-(1/2)(1.745329)\\0\\(\sqrt3/2)(1.745329)
\end{bmatrix}
\approx
\begin{bmatrix}-0.872665\\0\\1.511499\end{bmatrix}\ {\rm rad/s}.
$$

The vectors describe the same motion. A tilted pod resolves world-vertical turning into a mixture of its own
x and z components.

```bash
.venv/bin/python - <<'PY'
import numpy as np
from mesqsim import conventions as c
q0 = c.exp_map([0, np.deg2rad(30), 0])
q1 = c.quat_mul(c.yaw_quat(np.deg2rad(10)), q0)
wb = c.log_map(c.quat_mul(c.quat_conj(q0), q1)) / 0.1
ww = c.log_map(c.quat_mul(q1, c.quat_conj(q0))) / 0.1
print("body rad/s:", np.round(wb, 6))
print("world rad/s:", np.round(ww, 6))
print("body rotated to world:", np.round(c.quat_rotate(q0, wb), 6))
PY
```

Expect `[-0.872665, 0, 1.511499]`, then `[0, 0, 1.745329]` for both world-coordinate lines.

#### Example E: interpolation and circular averaging

Halfway from identity to $Y(90^\circ)$:

$$
q(1/2)=\operatorname{Exp}\left(\tfrac12(0,0,\pi/2)\right)=q_Y(45^\circ).
$$

For $179^\circ$ and $-179^\circ$:

$$
S=\frac{\sin179^\circ+\sin(-179^\circ)}2=0,\qquad
C=\frac{\cos179^\circ+\cos(-179^\circ)}2=-\cos1^\circ<0.
$$

The mean vector points along negative x: the circular mean is $180^\circ$. Arithmetic mean $0^\circ$
points the opposite way. Resultant length is $\cos1^\circ$, so

$$
\sigma_c=\sqrt{-2\log(\cos1^\circ)}\approx1.000025^\circ.
$$

Cosine and standard deviation use radians internally; the final result is converted to degrees. The small
difference from $1^\circ$ follows from the circular standard-deviation definition.

#### Example F: combining circular beliefs

Take concentration 2 at $0^\circ$ and concentration 2 at $90^\circ$:

$$
z=2e^{i0}+2e^{i\pi/2}=2+2i.
$$

Hence

$$
\mu=\operatorname{atan2}(2,2)=45^\circ,\qquad
\kappa=\sqrt{2^2+2^2}=2\sqrt2\approx2.828427.
$$

For a uniform belief, a central 95% interval has half-width $h=\pi(0.95)=171^\circ$.
Check the circular examples without licensed data:

```bash
.venv/bin/python - <<'PY'
import numpy as np
from mesqsim import conventions as c
a = np.deg2rad([179, -179])
print("mean deg:", np.rad2deg(c.circ_mean_rad(a)))
print("std deg:", np.rad2deg(c.circ_std_rad(a)))
q = c.quat_slerp(c.quat_identity(), c.yaw_quat(np.pi/2), 0.5)
print("midpoint deg:", np.rad2deg(c.heading_rad(q)))
mu, k = c.vm_product(0.0, 2.0, np.pi/2, 2.0)
print("product:", np.rad2deg(mu), k)
print("uniform halfwidth deg:", np.rad2deg(c.vm_halfwidth_rad(0, .95)))
PY
```

Expected values: approximately `180`, `1.000025`, `45`, then `45 2.828427`, and `171`.

### 1.6 Self-check questions

1. A mounting $m$ maps pod → segment and $s$ maps segment → world. Is the pod orientation $sm$ or $ms$?
   Derive it by transforming a vector.
2. Why can $q$ and $-q$ represent the same rotation? Does `quat_mul` automatically choose the canonical sign?
3. Why does $Y(\alpha)R$ change heading but leave $R^\top e_z$ unchanged? When is heading undefined?
4. Which maps child → parent: $R_p^\top R_c$ or $R_c^\top R_p$? Which finite increment gives a body rate?
5. What is the circular mean of $179^\circ$ and $-179^\circ$? What does $\kappa=0$ mean? Is diffusion
   prediction exact as a complete von Mises distribution?

---

**Answers — try the questions before reading further.**

1. **$sm$.** First $v_S=mv_P$, then $v_W=sv_S=smv_P$. The frame chain determines the order.
2. Both signs give the same quadratic matrix and the same sandwich action. **No:** canonicalization is separate.
3. Yaw rotates the horizontal projection of body +x, adding $\alpha$. But $Y(\alpha)^\top e_z=e_z$, so the
   body-coordinate up vector stays fixed. Heading is undefined when body +x is vertical.
4. **$R_p^\top R_c$** maps child → world → parent. Body rate uses
   $\operatorname{Log}(R_-^\top R_+)/\Delta t$.
5. **$180^\circ$.** Zero concentration is uniform belief, whose mean carries no information. Diffusion prediction
   matches the first circular moment; the complete density is generally not exactly von Mises.

## Verification and reference notes

The original lesson was checked using:

```bash
.venv/bin/python -m pytest -q tests/test_conventions.py tests/test_conventions_step2.py
```

The result was **30 passed**. The function tables distinguish direct tests, indirect coverage, and missing
standalone tests; they do not claim every helper has its own test.

When consolidating this file, the two convention suites and the existing repository-guide and explanation-link
checks were run together: **62 passed**. All three runnable example snippets above were executed successfully,
and local reference targets, source-line anchors, function coverage, and Markdown/math delimiters were checked.

The two original runnable snippets are preserved, and an additional body/world-rate snippet makes the frame
conversion reproducible. Tiny floating-point changes do not indicate different rotations. Use tolerances and
account for the quaternion sign ambiguity when comparing components.

This file reports no new numbers from AMASS, body-model files, recordings, or `twin_v1`. Later data-specific
numbers should be obtained with a short repository-root snippet and its recorded output.

References are separated by purpose:

- **Implementation:** [mesqsim/conventions.py](../../mesqsim/conventions.py).
- **Verification:** [tests/test_conventions.py](../../tests/test_conventions.py) and
  [tests/test_conventions_step2.py](../../tests/test_conventions_step2.py).
- **Existing explanations:** [MENTAL_PICTURE](../explain/MENTAL_PICTURE.md),
  [MATH_EXPLAINED](../explain/MATH_EXPLAINED.md), and [docs/MATH.md](../MATH.md).
- **Navigation:** [README](../../README.md), [REPOSITORY_GUIDE](../../REPOSITORY_GUIDE.md), and
  [MUST_KNOW](../MUST_KNOW.md), with the historical-document qualification in Module 0.
- **Step 3 data and learning:** [DATASET_CARD](../step_3/DATASET_CARD.md) and
  [LEARN_THE_MODEL](../step_3/LEARN_THE_MODEL.md).
- **Specifications:** [STEP1_BRIEF](../brief/STEP1_BRIEF.md), [STEP2_BRIEF](../brief/STEP2_BRIEF.md), and
  [STEP3_BRIEF](../brief/STEP3_BRIEF.md).

One refinement added while saving the lesson is worth retaining: `imu_truth` exposes a spline derivative, but
the session uses the discrete ideal-gyro helper before sensor errors and fusion. Section K cites both operations.
This refines the earlier verbal description without changing the rotation-rate derivation.

The next lesson is Module 2, AMASS and SMPL+H. Pause after the self-checks and resolve unclear rotation or frame
operations before continuing.
