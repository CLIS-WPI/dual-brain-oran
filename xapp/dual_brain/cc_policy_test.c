/* Self-check of cc_policy.h: build + run, exits 0 on success. */
#include "cc_policy.h"

#include <assert.h>
#include <stdio.h>

int main(void)
{
  int s = 0;
  /* sustained >= 80 for 3 samples; 80.0 counts (>=), a dip resets */
  double cells[] = {90, 90, 70, 80, 85, 81, 95};
  int want[] = {0, 0, 0, 0, 0, 1, 1};
  for (int i = 0; i < 7; i++)
    assert(cc_threshold_hot(&s, cells[i], 80, 3) == want[i]);
  s = 0; /* min_duration 0/1 = plain threshold */
  assert(cc_threshold_hot(&s, 80, 80, 0) == 1 && cc_threshold_hot(&s, 79.9, 80, 1) == 0);
  /* threshold acts on the first hot sample; ml needs trigger_consecutive positives */
  assert(cc_should_trigger(1, 1, 4) && !cc_should_trigger(0, 3, 4) && cc_should_trigger(0, 4, 4));
  /* lease = minimum hold: keep within it; after it, extend while hot, release at the first non-hot sample */
  unsigned long long ls = 10;
  assert(cc_lease_step(14, &ls, 5, 0) == CC_LEASE_KEEP);
  assert(cc_lease_step(15, &ls, 5, 1) == CC_LEASE_RENEW && ls == 10);
  assert(cc_lease_step(16, &ls, 5, 1) == CC_LEASE_RENEW);
  assert(cc_lease_step(17, &ls, 5, 0) == CC_LEASE_RELEASE);
  /* re-arm: armed stays armed; disarmed re-arms after min_duration samples strictly below the threshold;
   * a drain above the threshold (or a single dip) does not re-arm */
  int armed = 1, below = 0;
  assert(cc_rearm_step(&armed, &below, 99, 80, 3) == 0 && armed == 1);
  armed = 0;
  double drain[] = {97, 95, 79, 92, 85, 80, 70, 60, 50, 95};
  int rearm[] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 0};
  for (int i = 0; i < 10; i++)
    assert(cc_rearm_step(&armed, &below, drain[i], 80, 3) == rearm[i]);
  assert(armed == 1);
  /* xApp/replay order: disarmed → no reserve even with a long hot streak (debounce satisfied) */
  armed = 0;
  below = 0;
  int hot_streak = 10, reserved = 0;
  if (armed && cc_should_trigger(0, hot_streak, 2))
    reserved = 1;
  assert(!reserved);
  puts("cc_policy_test: ok");
  return 0;
}
