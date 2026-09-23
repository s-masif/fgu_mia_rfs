# CAC-PPO — Forensic Leakage Audit (Python side)

**Scope:** `rl_s_CAC.py`, `rl_s_m_CAC.py`, `CAC_PPO.py`, `CAC_m_PPO.py`, `contention_critic.py`,
`contention_m_critic.py`, `environment.py`, `environment_m.py`, `attention_encoder.py`,
`processing_model.py`, `reward_function.py`.
OMNeT++/Simu5G assumed correct and not audited.

**Question:** does any code path let the policy, critic, reward calculation or reported metrics use
information that should not be available at decision time — in particular, anything that could inflate the
~90% completion ratio?

---

## LEAKAGE VERDICT

🟠 **POSSIBLE LEAKAGE / IMPORTANT ISSUE**

To be precise: **no leakage was found.** No code path lets post-action information reach action selection,
the reward calculation or the completion metric. The 🟠 is for two confirmed **non-leakage** issues:

1. the masked server runs the **unmasked** agent (`CAC_m_PPO.py` and `contention_m_critic.py` are never
   imported by anything);
2. advantages are computed with a critic that was just trained on the same rollout (in-sample advantage
   estimation).

Neither can inflate completion. Details in [§7](#7-masked-cac-leakage-audit--the-one-important-finding)
and [§11](#11-replayppo-leakage-audit).

---

## Summary table

| Leakage category | File/function | Exact code behavior | Leakage? | Could inflate completion? | Severity | Explanation |
|---|---|---|---|---|---|---|
| Future info in state | `CAC_PPO.build_state_tensor` | 10 features from the current JSON message only | No | No | — | cpu/data/deadline/priority/x/y/isActive/dist0-2; distances recomputed from current x,y |
| Channel-rate timing | `rl_s_*.run` → `environment*.step` | rates enter only as `env.step(channel_rates=...)`, paired with `prev_actions` | No | No | — | never reach `build_state_tensor`; the actor never sees any rate |
| Reward → state | all servers | `results` / `user_rewards` used only for `store()` after the action | No | No | — | no reward, delay or `met_deadline` in the observation |
| Critic target | `_update_critic`, `train_step` | targets are observed `r_obs/10`; critic output never enters actor input | No | No | — | normal RL |
| Counterfactual | `contention_critic.counterfactual_advantages` | `@torch.no_grad`, only `critic(...)` calls, no `env.step` | No | No | — | no ground truth for alternative actions anywhere |
| Old vs new policy | `select_action` → `store` → PPO | `probs` / `log_probs` sampled under `no_grad`, detached, never recomputed | No | No | — | `ratio = exp(new_lp − LP[mb])` uses frozen `LP` |
| User alignment | servers + `environment*.step` | sort by userId → state rows → actions → `actions_by_id` → results | No | No | — | verified numerically (§3) |
| Mask handling (masked run) | `rl_s_m_CAC.build_agent` | imports `CAC_PPO` (unmasked); masked modules never imported | No | No | **High** | the masked experiment is not running masked code |
| CPU contention | `ProcessingModel.compute_allocated_freq` | `20000/k`, k = active offloaders including self | No | No | — | 4 users → 5000 MHz each, verified |
| Delay cap | `environment*.process_user` | `display = min(tx,0.3)+proc`, reward uses `tx+proc` | No | **No** | Medium (reporting) | 1.56 M configurations brute-forced: zero true-miss-reported-as-success |
| Missing rate | same | `tx=999` → `display = 0.3+proc > deadline` | No | No | — | always a failure; verified |
| Units | env + processing model | Mbit/Mbps, Mcycles/MHz, seconds | No | No | — | all consistent |
| Reporting | `on_episode_end` | completion = mean of per-slot ratios; delay over **all** active users | No | No | Low | empty-priority slots logged as 0.0 (deflates) |
| Train/eval | servers | metrics come from the training rollout, stochastic policy | No | No | Low | must be stated in the paper |
| Replay | `deque(maxlen=...)`, `train_step` | rollout → rewards → replay → critic → advantages → PPO | No | No | Medium | advantages use the critic already fit on this rollout |
| PPO reuse | `train_step` | `S, Aa, LP, P, R, adv` all frozen | No | No | — | correct |
| Cross-variant | file paths / ports | `CAC_metrics_30_users_1500.csv` vs `CAC_seed42_metrics.csv`; ports 5566 vs 5565 | No | No | Low | no shared state; header comment says 5567, code default is 5566 |

---

## 1. Decision-time information audit

At slot *t* the actor sees exactly `build_state_tensor(users)` built from message *t*, which OMNeT++ sends
**before** any action exists for that slot. `select_action` is `@torch.no_grad`, takes only `users`, and
returns `(state, actions, log_probs, probs)`. Nothing from `env.step`, `results`, `channel_rates` or the
critic is in that path.

> **Explicit answer: no. The actor receives no information derived after its action is chosen.**

## 2. State leakage audit

The ten features are task attributes and geometry from the current slot. The three distances are recomputed
from the same `locationX/Y` in that message using hardcoded UAV coordinates that match `omnetpp.ini`.
Inactive users are zeroed except `isActive`. No reward, delay, completion, `tx_delay`, `proc_delay`,
`met_deadline`, energy or channel rate appears in the vector.

## 3. Action/reward alignment audit

Chain:

```
sorted(msg["users"], key=userId)
  -> build_state_tensor(users)                 row i  = sorted position i
  -> actions                                   same row order
  -> actions_by_id[u["userId"]] = actions_list[idx]      (sent to OMNeT++)
  -> prev_actions = actions_list               sorted order, matches prev_users
  -> env.step iterates prev_users in that order
  -> results[i] corresponds to row i
  -> user_rewards built from results in order  -> critic targets, PPO log-probs
```

Verified with the requested example:

```
userIds [5,2,9] -> sorted [2,5,9]
state rows: row0=user2, row1=user5, row2=user9
actions (row order) = [3,1,2]
actions_by_id[2]=3, actions_by_id[5]=1, actions_by_id[9]=2   OK
```

One detail that makes this robust: `env.step` looks up rates as `channel_rates.get(user_id)` by **userId**,
not by index. No misalignment anywhere.

## 4. Channel-rate timing audit

Rates arrive in message *t* and were produced by slot *t−1* transmissions (`StateBuffer` clears
`collectedChannelRates` at each dispatch). Python uses them in
`env.step(prev_users, prev_actions, channel_rates)` — the reward for slot *t−1*, matching the actions that
generated them. The state for slot *t* is built separately and contains no rate.

Rates are therefore used **only after** action selection, for reward and delay. There is no path inserting a
selected-UAV rate into the state. The actor in fact has **less** information than it could legitimately have.

## 5. Reward/critic leakage audit

`r_hat` is used in `counterfactual_advantages` only, after the rollout is complete. Critic parameters never
touch the actor's input. The residual `eta*(r_obs − r_hat)/N` is applied at training time only. The actor's
input is `state` alone.

## 6. Counterfactual leakage audit

`counterfactual_advantages` is decorated `@torch.no_grad()` and contains only
`critic(x[t_c], acts, active[t_c])` calls. `acts = actions[t_c].clone()` with a single position overwritten,
state untouched. `G_cf` is initialised to `G_fact` and only non-factual entries are recomputed — correct,
since the factual alternative equals the factual value by definition. The observed reward of the factual
action is never reused as the value of an alternative; it enters only through the separate residual term.

## 7. Masked CAC leakage audit — the one important finding

`rl_s_m_CAC.py` line 107:

```python
if self.agent_kind == "cac":
    from CAC_PPO import CACPPOAgent          # <-- the UNMASKED agent
```

A grep confirms **`CAC_m_PPO.py` and `contention_m_critic.py` are never imported by anything.** So the
"masked" run is:

- **unmasked actor** — inactive users are *not* forced to action 0;
- **unmasked critic** — team objective `sum(r_hat)/30`, MSE over all 30 users, while `environment_m` returns
  the mean over active users;
- **inactive rewards zeroed by the server** (`0.0 if r["inactive"]`), so inactive users are trained toward a
  constant 0 target and receive no +0.5/−2 signal at all. They act near-randomly; since OMNeT++ skips
  offloads from users with no task, that costs nothing physically.

This is **not leakage** and cannot inflate completion — completion counts active users only, and inactive
actions consume no radio or CPU resources in either OMNeT++ (`TaskGeneratorApp` returns before
handover/probe) or `environment_m` (`if prev_users[i]["taskId"] != -1`). But the masked-vs-unmasked
comparison is currently meaningless: both runs use the same agent, differing only in the environment module
and the zeroing of inactive rewards.

**Minimal fix** in `rl_s_m_CAC.build_agent`:

```python
from CAC_m_PPO import CACPPOAgent
```

Classification: affects **training** only, not the completion metric.

## 8. Unmasked CAC leakage audit

`CAC_PPO.select_action` has no forced action and no mask. `contention_critic.ContentionCritic.forward`
returns `r_hat` for all users without multiplying by `active`. `counterfactual_advantages` enumerates all
users. The PPO loss, entropy and normalisation run over all entries. The `active` argument is used **only**
as the contention mask (same-UAV groups and attention keys), which is legitimate: contention is physically
determined by active offloaders only, and `isActive` is already an observed feature. No hidden mask exists
downstream.

## 9. CPU contention audit

Verified numerically with your own code:

- `compute_allocated_freq(4)` → **5000.0 MHz** each (your manual test passes);
- k = 1..10 → 20000, 10000, 6666.7, 5000, 4000, 3333.3, 2857.1, 2500, 2222.2, 2000;
- k counts active offloaders per UAV **including the user itself**, excludes inactive users
  (`taskId != -1` guard) and excludes other UAVs;
- the critic's `n_same` uses the same rule (group size including self, from `same_server_matrix`), with self
  excluded only from the attention bias.

No offloader is skipped and no user receives the full 20 GHz while others share the same UAV.

## 10. Delay/deadline metric inflation audit — highest-priority check

Exact implemented formulas:

```
offload: tx   = dataSize / (rate/1e6) + 0.006      (rate present)
         tx   = 999.0                               (rate missing)
         proc = cpuCycles / (20000/k)
local:   tx   = 0 ;  proc = cpuCycles / 1500

total_delay_reward  = tx + proc                -> reward
total_delay_display = min(tx, 0.300) + proc    -> met_deadline AND avg_delay_ms
completion          = (total_delay_display <= deadline)
```

**The cap cannot turn a failure into a success.** If `tx ≥ 0.3` then `display ≥ 0.3 + proc > 0.30 ≥ deadline`,
because `proc > 0` and `deadlineMax = 0.30`. Brute force over 1,562,715 combinations
(cpu ∈ [300,500], data ∈ [1,3], deadline ∈ [0.25,0.30], k ∈ [1,15], rate ∈ [1,100] Mbps ∪ {None}):
**zero cases reported successful while truly late.** When `tx < 0.3` the two delays are identical, so reward
and completion agree exactly.

**Consequence (reporting only):** a lost probe contributes **380 ms** to `avg_delay_ms`
(`min(999,0.3) + 0.08`) instead of 999.08 s. `avg_delay_ms` therefore systematically *understates* delay for
late offloads while still counting them as failures. Report it as a censored statistic, or alongside
`lost_probe_count` (already logged by the masked server).

## 11. Missing channel rate handling

`channel_rates.get(user_id, None)` → `None` → `tx_delay = 999.0`, `tx_delay_initial = 0.30`.
Numeric example (k=4, cpu=400, deadline=0.30):

```
tx(reward) = 999.0 s,  proc = 0.080 s
display    = min(999,0.3) + 0.08 = 0.380 s  >  0.30
-> met_deadline = False,  reward = -10.0
```

Missing, late and invalid rates all count as failures. A rate of exactly 0 would divide by zero
(`data/0` → `inf` → still a failure); `TaskSinkApp` never emits 0, so this is theoretical. Users with missing
rates stay in the denominator of every metric — they do not disappear.

## 12. Unit audit

All consistent: `rate/1e6` bps→Mbps; Mbit/Mbps = s; Mcycles/MHz = s; deadlines in seconds; only the CSV
multiplies by 1000 for ms. Your examples check out: 2 Mbit / 20 Mbps = 0.1 s; 400 Mcycles / 5000 MHz = 0.08 s.

## 13. CSV/reporting bias audit

- `completion_ratio` = **mean of per-slot ratios**, not total successes / total active tasks. Slots with zero
  active users are skipped (`if not act: return` / `if active_results:`), so there is no zero-slot bias.
- `avg_delay_ms` averages over **all** active users including failures — no survivorship bias. The separate
  `completed_task_delay_ms` (masked server) averages successes only; do not mix the two.
- `high_priority_completion` / `low_priority_completion` append **0.0** when a slot contains no user of that
  priority. Rare with 22.5 active users and p=0.4, and it biases *downward*.
- `a_reward` is the sum over slots of the per-slot mean. In the unmasked server `reward` uses
  `environment.py`'s mean over all 30 users (inactive included); in the masked server `a_reward` uses
  `environment_m`'s mean over active users. **These columns are not comparable across variants.**
- `offload_ratio`, `local_ratio`, `uav*_ratio` are over active users only; `inactive_correct_ratio` over
  inactive users only.

No metric excludes failed tasks in a way that would raise completion.

## 14. Training/evaluation audit

Metrics are computed from the **training rollout**, under the **stochastic** policy, from actual environment
results (never from critic predictions or training targets). Rewards are collected before the update and the
CSV row is written after it, which is fine but should be stated explicitly in the paper. There is no separate
deterministic evaluation rollout.

## 15. Replay/PPO leakage audit

Chronology per rollout: collect 500 slots with a fixed policy → rewards observed → `replay.append(...)`
(detached) → critic trained → advantages → PPO update → `clear_buffer()`. `train_step` is called only at the
rollout boundary, so **no later slot can influence an earlier action within the same rollout**. The replay is
a `deque(maxlen=replay_rollouts)` of detached tensors; PPO uses only frozen rollout data and never recomputes
`LP` or `P`.

**Issue:** advantages are computed **after** the critic has been trained on this same rollout, including these
slots' rewards. The critic partially memorises the realised outcomes, which shrinks the residual term and
biases the counterfactual baseline toward the taken actions. This is in-sample advantage estimation, not
leakage into action selection, and affects **training** only. The honest out-of-sample number is
`critic_pre_r2`.

*Minimal fix:* compute advantages with a copy of the critic taken **before** `_update_critic`, or hold the
newest rollout out of critic training for one episode.

## 16. Cross-run contamination and seeds

Separate processes, separate model objects, separate replay buffers.

| | Unmasked (`rl_s_CAC.py`) | Masked (`rl_s_m_CAC.py`) |
|---|---|---|
| CSV | `CSV_results/CAC_metrics_30_users_1500.csv` | `CSV_results/CAC_seed42_metrics.csv` |
| Checkpoints | `checkpoints/CAC_ep*.pt` | `checkpoints/CAC_seed42_*` |
| Port (default) | 5566 (header comment says 5567) | 5565 |
| Environment | `environment.py` | `environment_m.py` |

No collisions, no shared globals, each process has its own RNG state. Fix the port comment or the default so
an OMNeT++ config cannot point at a dead port.

Both use `set_seed(42)`, which gives reproducibility. Because both servers currently instantiate the **same**
agent class, identical episode-1 physical metrics between the two runs are expected — and would no longer be
expected once the masked variant is wired correctly, since forced inactive actions change the sampled action
vector immediately.

## 17. Why ~90% completion may be legitimate

Your framing is right: after 20 episodes the agent has seen ≈225,000 user-level decisions with 4 actions,
γ = 0, deterministic CPU sharing and a per-user advantage. Fast convergence is expected.

The numbers also show 90% is reachable **within this environment model**:

- local-only feasibility is **56.4%** (`cpu/1500 ≤ deadline` over the ini ranges);
- for the rest, offloading succeeds once the measured rate clears a modest bar (2 Mbit, 400 Mcycles,
  deadline 0.275 s):

| users on the UAV (k) | processing delay | tx budget | required rate |
|---|---|---|---|
| 1 | 20.0 ms | 249.0 ms | 8.0 Mbps |
| 2 | 40.0 ms | 229.0 ms | 8.7 Mbps |
| 4 | 80.0 ms | 189.0 ms | 10.6 Mbps |
| 6 | 120.0 ms | 149.0 ms | 13.4 Mbps |
| 8 | 160.0 ms | 109.0 ms | 18.3 Mbps |
| 10 | 200.0 ms | 69.0 ms | 29.0 Mbps |
| 12 | 240.0 ms | 29.0 ms | 69.0 Mbps |

- `TaskSinkApp` computes `rate = 524056 bits / (raw − 0.006)`, so a raw delay of 15 ms yields ~58 Mbps and
  10 ms yields ~131 Mbps. Rates above 20 Mbps are the norm here.

So a policy that offloads most tasks and keeps per-UAV load balanced (k ≈ 7–8) should clear 85–90%. The
earlier APPO run sat at 71% with an offload ratio of only ~52% and no explicit load balancing, which is
consistent.

**Check this directly in the CSV** — the masked server already logs everything needed:

```
offload_ratio          should be well above APPO's ~0.52
jain_index_uav_load    should be close to 1.0
mean_uav0/1/2_users    should be roughly equal
lost_probe_count       should be low
local_ratio            should track the share of locally feasible tasks
```

If all of that holds, the result is behaviourally coherent rather than a metric artifact.

**Caveat about absolute realism (not leakage):** the simulator transmits a fixed 65,507-byte packet and
Python extrapolates that rate to the full 1–3 Mbit task; the 6 ms subtraction inflates measured rates when
raw delays are small; the uplink scheduler is Max C/I. These make offloading cheaper than a real 5G uplink.
They apply equally to APPO, PPO and CAC, so relative comparisons survive, but the absolute 90% should not be
presented as a 5G-realistic completion rate.

---

## Concrete issues found

| # | Issue | File / function | Affects | Severity | Minimal fix |
|---|---|---|---|---|---|
| 1 | Masked server runs the unmasked agent | `rl_s_m_CAC.build_agent` (line 107) | training | High | `from CAC_m_PPO import CACPPOAgent` |
| 2 | Advantages use a critic already fit on the same rollout | `CAC_PPO.train_step` | training | Medium | snapshot the critic before `_update_critic`, use it for the counterfactuals |
| 3 | `avg_delay_ms` uses the capped delay (380 ms for a lost probe) | `environment*.process_user` + servers | reporting | Medium | report as censored, or add an uncapped delay column |
| 4 | Empty-priority slots logged as 0.0 | `log_slot` / `on_episode_end` | reporting | Low | skip those slots instead of appending 0.0 |
| 5 | Port comment vs default mismatch (5567 vs 5566) | `rl_s_CAC.py` header / `__init__` | ops | Low | make them agree |
| 6 | `reward` column means different things across variants | both servers | reporting | Low | do not plot them on the same axes |

---

## Final answer: can you trust the ~90% completion result?

**As a measurement of what your Python environment model reports, yes.** Every path the audit asked about was
traced and no leakage was found: the actor sees only current-slot task and geometry data, channel rates never
reach it, rewards never enter the state, counterfactuals use the critic alone, old-policy probabilities are
frozen and detached, user identities stay aligned end to end, contention is counted correctly, the delay cap
provably cannot flip a failure into a success, and missing rates always count as failures.

**Two qualifications.**

1. The number comes from the **training rollout under a stochastic policy** — a training metric, not held-out
   evaluation.
2. Whatever run produced ~90% is labelled "masked" but was executed with **unmasked agent code**. Fix that
   import and re-run before attributing the result to the masked variant, and compare `offload_ratio` and
   `jain_index_uav_load` against APPO to confirm the gain is a real behavioural change.
