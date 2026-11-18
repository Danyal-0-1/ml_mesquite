# The mental picture: what happens, with your eyes closed

*Explanation pipeline, file 1 of 5 · about 20 minutes · start here · [index](README.md) · next:
[PIPELINE_EXPLAINED.md](PIPELINE_EXPLAINED.md)*

This file has no equations. It builds one picture of the whole project — Step 1 and Step 2 — that you can replay
from memory. Every later file makes a piece of this picture precise.

---

## 1. The suit: fifteen watches that agree about "down" but not about "north"

Picture a person wearing **fifteen watches** ("pods"): on the pelvis, the chest, the head, both upper arms, both
forearms, both hands, both thighs, both shins and both feet. Inside every watch a small motion chip combines a
gyroscope (how fast it is turning) and an accelerometer (which way gravity pulls) into one answer: *how am I
oriented?*

- **Gravity is the same for everyone.** Every watch feels the same "down", so every watch knows its **tilt**
  correctly: how far it leans forward, backward or sideways.
- **There is no compass.** When a watch switches on it simply *invents* a "north" of its own: a private direction it
  calls zero heading. Fifteen watches, fifteen private norths.
- **The private north slides.** A little every minute (the gyroscope's leftover bias), a little more every time the
  watch spins round (its rate is slightly mis-scaled), and a little at random (noise). After half an hour two
  watches that agreed at the start can disagree by tens of degrees.
- **A watch can restart.** Unplugging or plugging one in restarts it; it then invents a brand-new north (the first
  measurement of the real suit saw a pod 78° wrong for about 16 s after each restart).

Thirty-two times a second, each watch sends a **postcard**: 16 bytes holding its orientation, a postcard number and
the time on its own little clock when the picture was taken. A hub collects the postcards; a laptop draws an
**avatar**, one bone per watch.

**The visible problem.** Because the norths disagree, the avatar's joints look **twisted about the vertical**: the
shin seems turned against the thigh although the knee only bent. For every one of the **14 joints** we want the
*relative heading* — how far the child watch's north has slid away from its parent watch's north. We call it **δ**
("delta"). Fix δ at every joint, outward from the pelvis, and every watch shares the pelvis watch's north.

## 2. The room: north can never be known

Now picture the whole room turning slowly about the vertical, with the wearer standing still inside it. What do
the watches send? **Exactly the same postcards.** Nothing in them notices that the room turned.

So no method, however clever, can ever tell which way the body faces in the room, or where true north is. Only
headings **relative to one another** can be recovered: one δ per joint. Every result in this project is about δ.

## 3. The simulator: a flight simulator for the suit

To learn whether a method recovers δ we need the right answer. The real suit cannot give it. So Step 1 built a
**flight simulator for the suit**:

1. Take real human motion from **AMASS**: thousands of recordings of real people moving in optical
   motion-capture studios, already turned into the angles of a body model (SMPL+H).
2. Strap fifteen virtual watches onto that body, each a little crooked, as a real person would strap them.
3. Live a whole session as the suit would: the watches switched on together **in a box** with their faces aligned,
   taken out and **strapped on**, the wearer standing in a **T-pose** (arms out), an idle wait, then the **motion**.
4. Give every watch its private north, its sliding, its noisy tilt, its own clock and its own radio losses, using
   numbers we chose (most are guesses until the bench measures the real pods).
5. Write the postcards the real suit would have sent — and, **in a separate sealed envelope**, the truth: every
   watch's true north at every moment, and every joint's true δ.

The envelope is opened only by the **scorer**. Any method that peeks at it could never run on the real suit.

## 4. Step 0: the post office

The laptop's first job is a **post office** that sorts the postcards. It is the same code for simulated and real
recordings:

- It **puts the postcards in order** for each watch and reads the watch clocks, which roll over every 65.5 seconds.
- It labels each card: **fresh** (a new picture), **held** (the watch sent the same picture again because its sampling
  task stalled), or **lost** (a number is missing: the radio dropped it).
- It notices **restarts** (the numbers jump back to zero) and starts a new chapter, an *epoch*, for that watch.
- It takes the **box photo**: while the watches lay aligned in the box, each one's heading tells how its private
  north was turned. That is **box registration**.
- It takes the **T-pose photo**: how each watch sat on the body while the wearer stood in the known pose.

## 5. The baselines: the simplest answers

Step 1 scored four simple answers against the envelope:

- **B0: do nothing.** δ = 0. Its error is the whole problem: on average about 90° with arbitrary norths.
- **B1: trust the box.** The box photo lines the norths up at the start, but they keep sliding afterwards, and a
  restart makes the box photo stale. This is the honest "no learning" answer.
- **O1: an oracle that knows the norths at switch-on.** Its error is the sliding alone.
- **O2: an oracle that knows δ.** Its error must be exactly zero; it checks that the scorer is wired correctly.

## 6. Step 2: the geometry

Step 2 asks: *can the body itself tell us δ?* Joints are not free: a knee folds about a pin, an elbow has two pins,
a hip has walls it cannot pass. These constraints involve both watches of a joint — and they hold whatever the
private norths are.

### 6.1 The knee is a door hinge

Picture the knee as a **door hinge**. The **hinge pin** is one physical line through the knee. The thigh watch sees
that pin in its own frame; the shin watch sees the same pin in its frame.

- Because **tilt is shared**, both watches agree on how *steeply* the pin points.
- Only their **norths** differ. So lay both views of the pin flat on the floor: **two shadows**. The angle between
  the two shadows is exactly how far the shin watch's north has slid from the thigh watch's: **δ**.
- One glance is enough when we know where the pin is. Many glances, while the leg moves, make it sharper.

### 6.2 The blind spot

When the pin stands **straight up**, its shadow is a dot. A dot has no direction, so it says nothing about δ.
(A knee pin is nearly horizontal in walking, so the knee rarely hits this blind spot; other hinges can.)

### 6.3 Finding the pin without being told (GEO-ID)

In the simulator we could cheat and read where each watch sits on its bone: that is **GEO-K**, the "known axes"
ceiling, never usable on the real suit. The deployable **GEO-ID** must find the pin itself:

- **Watch the spinning.** For a true hinge, the part of each bone's spin that is *not* about the pin is the same
  for thigh and shin. Only the true pin makes that true at every moment. Spin rates do not depend on north at all.
- **Do not guess when the leg turns as one piece.** If thigh and shin turn together (turning on the spot,
  swaying), every pin fits. GEO-ID then admits it cannot see the pin and keeps the box answer.
- **Use the T-pose photo.** In the T-pose the knee is straight: its pin is **level**, and it is **one line through
  both bones**. That leaves only one thing to find — which way the level pin points — which makes the search robust.
- **Which end of the pin is which.** A pin looks the same flipped end to end. Flip only one bone's pin and the answer
  is wrong by half a turn. The T-pose photo settles it: the wrong choice would mean the shin had been turned half a
  turn since the T-pose. And **the knee folds only one way**, which tells which end of the pin points to the left.

### 6.4 Elbows, wrists and ankles: two pins

These joints are more like a **universal joint**: two pins at a right angle, one fixed in each bone. That gives
**one equation per moment** instead of a direct answer. Usually many moments pin δ down; sometimes the equation has a
**mirror answer** half a turn away that fits equally well, and then the joint says nothing rather than pick one. Real
elbows, wrists and ankles also twist about a third axis (in AMASS by 9° to 20°), so the two-pin picture is rough.

### 6.5 Ball joints: walls instead of pins

Hips, shoulders, the lower back and the neck have **no pin**. The only clue is that they **cannot bend past their
limits**, learned from the training subjects' real motion. Picture the joint inside a **room with walls**: while
the motion stays in the middle of the room, every δ is possible and the joint says nothing. Only when the motion
**touches a wall** do some δ become impossible. So at ball joints the geometry usually says almost nothing — the
expected result, not a bug.

To use the walls GEO-ID needs each bone's frame, which the T-pose photo gives once we know which way the body
faced in the T-pose. **The knees tell us**: their pins point left–right.

## 7. Confidence and memory

- **Confidence (κ, "kappa")** is how sharp the shadow is: how strongly a stretch of motion points to one δ. Zero means
  "no information"; large means "I am sure to within a degree or two". It is fitted on held-out validation people,
  so that "68 % sure" really is right about 68 % of the time.
- **Memory (the filter).** Each joint keeps a belief. It **starts from the box photo**, **sharpens** whenever a
  stretch of motion says something, and **slowly fades** while nothing new arrives, because the norths keep sliding.
  When a watch **restarts**, the joints touching it **forget** at once: the new north is unknown.
- **"Uncertain"** is a flag raised while the confidence is below a threshold learned on validation. The estimate is
  still shown; the flag says not to trust it too much.

## 8. The whole picture in one diagram

```text
 real motion (AMASS people)                         the sealed envelope (truth: every north, every δ)
        │                                                        │  opened only by the scorer
        ▼                                                        ▼
 SIMULATOR: box → strap on → T-pose → idle → motion        ┌──────────────┐
   private norths slide · tilt noise · clocks · radio      │   SCORER     │◄── errors, coverage, strata
        │  postcards (16 bytes, 32 per second)             └──────▲───────┘
        ▼                                                        │
 STEP 0, the post office: order · fresh/held/lost · restarts · box photo · T-pose photo
        │                                                        │
        ├──► B0 nothing · B1 box · O1, O2 oracles ───────────────┤
        │                                                        │
        └──► GEO, every second over the last 8 seconds, per joint:│
               knee: hinge pin and its shadow (pin found from spin, tied to the T-pose)
               elbow · wrist · ankle: two pins (GEO-K) or walls (GEO-ID)
               hips · shoulders · back · neck: walls of the range of motion
             ──► memory per joint: start from the box · sharpen · fade · forget at a restart
             ──► δ̂ with its confidence κ and an "uncertain" flag, 32 times a second ──────┘

 DIODEM (real IMUs on mechanical arms and legs, cameras for truth) ──► the same GEO ──► against the cameras
```

## 9. What goes in, what comes out, what never can

| | what | notes |
|---|---|---|
| **in** | the postcards (`frames.parquet`) and the phase times (`session.json`) | exactly what the real logger writes |
| **in (GEO-K only)** | where each watch truly sits on its bone | simulator truth; a ceiling, never deployable |
| **in (beliefs)** | the estimator's own settings (`geometry:` in the config) | some guessed, some fitted on train/validation people |
| **out** | for every joint and 32 Hz step: δ̂, its confidence κ, an "uncertain" flag | the scorer adds the errors |
| **out (scorer)** | error tables, coverage, figures, the test of hypothesis H2 | `docs/RESULTS_STEP2.md` |
| **never** | true north, the body's facing in the room | turn the room and nothing changes |

## 10. A reality check with real sensors

**DIODEM** is a public dataset of real IMUs on two mechanical five-segment chains (an "arm" and a "gait" chain),
whose motion cameras recorded. Running the watches' fusion without the compass, their relative heading drifted by about 80° in six
minutes on one hinge. The unchanged GEO-ID followed the cameras' answer instead: a reminder that the hinge idea works
on real hardware — and a warning that settings learned from people transfer imperfectly to machines.

## 11. Self-check questions

1. **Why can no method recover north or the body's facing?** Turning the whole room, wearer included, changes no
   postcard; every north gives the same data. Only relative headings δ are knowable.
2. **What does a watch get right, and what wrong?** Right: tilt, because gravity is common. Wrong: heading, because
   each watch invents and slowly loses its own north.
3. **Why is B1 not perfect?** The box lines the norths up at switch-on, but they keep sliding, and a restart makes
   the box photo stale.
4. **Why can the knee tell us δ?** Both watches see the same hinge pin; tilt is shared, so the pins' shadows on the
   floor differ only by the norths' difference, δ.
5. **When does the knee say nothing?** When its pin stands straight up (the shadow is a dot), and — for GEO-ID —
   while thigh and shin turn together, so no pin can be told apart from the others.
6. **What goes wrong if one end of the pin is mistaken for the other?** The answer is off by half a turn (180°).
   The T-pose photo and the knee's one-way folding prevent it.
7. **Why do hips and shoulders usually say almost nothing?** They have no pin, only walls; the motion seldom
   touches a wall, and until it does every δ is possible.
8. **What is κ, and what does "uncertain" mean?** κ is the confidence (0 = nothing known); "uncertain" means κ is
   below a threshold learned on validation, while the estimate is still shown.
9. **What happens to a joint's memory when a watch restarts?** It forgets at once (κ = 0): the new north is unknown.
10. **What is GEO-K for, if it cannot run on the real suit?** It is the ceiling: what geometry achieves with the
    true pins and straps. The gap between GEO-K and GEO-ID is the price of finding the pins ourselves.

---

*Next: [PIPELINE_EXPLAINED.md](PIPELINE_EXPLAINED.md) follows one session through every stage, with real numbers.*
