/* Self-check of cc_capped(): build + run, exits 0 on success. */
#include "cc_assoc.h"

#include <assert.h>
#include <stdio.h>

int main(void)
{
  /* R1: UE refs 1, 2 (center), 3 (protected); congestor = 1 */
  assert(cc_capped(1, 1, 3) && cc_capped(2, 1, 3) && !cc_capped(3, 1, 3)); /* both center UEs capped */
  assert(cc_capped(3, 3, 3) == 0); /* the protected UE is never capped, even if it is the congestor */
  /* legacy (MILCOM): only the congestor */
  assert(cc_capped(1, 1, -1) && !cc_capped(2, 1, -1) && !cc_capped(3, 1, -1));
  /* RAN UE ID 0 is a valid protected id */
  assert(!cc_capped(0, 1, 0) && cc_capped(1, 1, 0));
  puts("cc_assoc_test: ok");
  return 0;
}
