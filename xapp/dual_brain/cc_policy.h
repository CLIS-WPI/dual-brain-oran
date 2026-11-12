/* Trigger decision of the congestion xApp, one call per KPM sample (R1, dualbrain8).
 *
 *  threshold policy = persistence baseline: "hot" once cell PRB >= threshold has held for
 *      min_duration_samples consecutive samples (the dataset label's sustained rule, evaluated at t);
 *      it acts on the first hot sample (no extra debounce — the sustain already is one).
 *  ml policy: acts after trigger_consecutive consecutive positive predictions (debounce; the template
 *      slot control.debounce_n, default ceil(1000 / T_s)), so a 1-2 s post-idle MCS spike cannot trigger it.
 */
#ifndef HARMONY_CC_POLICY_H
#define HARMONY_CC_POLICY_H

static inline int cc_threshold_hot(int* above_streak, double cell, double threshold, int min_duration_samples)
{
  *above_streak = cell >= threshold ? *above_streak + 1 : 0;
  return *above_streak >= (min_duration_samples > 0 ? min_duration_samples : 1);
}

static inline int cc_should_trigger(int threshold_policy, int hot_streak, int trigger_consecutive)
{
  return hot_streak >= (threshold_policy ? 1 : trigger_consecutive);
}

/* Lease release (dualbrain10, R1): under the cap, cell PRB and the model no longer separate congestion from its
 * aftermath (the capped UEs drain their backlog at <= 100 - reserve_pct %), so release is time-based. The lease is a
 * minimum hold: the reservation is kept for lease_samples after it starts; after that it stays only while the policy
 * is hot and lifts at the first non-hot sample (2026-11-04: full-lease renewal held the cap through the next ON when
 * the lease expired during the backlog drain). Same rule for every policy.
 * Returns 0 keep (within the lease), 1 extended (past the lease, hot), 2 release. *lease_start is not changed. */
#define CC_LEASE_KEEP 0
#define CC_LEASE_RENEW 1
#define CC_LEASE_RELEASE 2
static inline int cc_lease_step(unsigned long long samples, unsigned long long* lease_start, int lease_samples, int hot)
{
  if (samples - *lease_start < (unsigned long long)lease_samples)
    return CC_LEASE_KEEP;
  return hot ? CC_LEASE_RENEW : CC_LEASE_RELEASE;
}

/* Re-arm after release (dualbrain11, R1, 2026-11-05): when the cap lifts, the capped UEs drain their backlog and the
 * cell jumps back up, which made the policy re-trigger 2.5 s after every release (Exp-04 dry run 6). After a release
 * the trigger is disarmed; it re-arms only once cell PRB has stayed below the label threshold for the label's
 * min_duration_samples (the spec's own congestion rule, no new constant). The debounce is unchanged. Same rule for
 * every policy. Call once per sample while disarmed; returns 1 on the sample that re-arms. */
static inline int cc_rearm_step(int* armed, int* below_streak, double cell, double threshold, int min_duration_samples)
{
  if (*armed)
    return 0;
  *below_streak = cell < threshold ? *below_streak + 1 : 0;
  if (*below_streak >= (min_duration_samples > 0 ? min_duration_samples : 1)) {
    *armed = 1;
    *below_streak = 0;
    return 1;
  }
  return 0;
}

#endif
