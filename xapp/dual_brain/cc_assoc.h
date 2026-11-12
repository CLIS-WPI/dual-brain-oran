/* Which UEs the congestion xApp puts into the capped (best-effort) slice on a reservation.
 *
 *  protected_ref >= 0  (R1)     every UE except the protected one (KPM RAN UE ID) — capped
 *                               *together*: the slice quota is shared by its members, so two heavy
 *                               UEs cannot absorb the PRBs freed for the protected UE.
 *  protected_ref <  0  (legacy) only the congestor (the UE with the highest recent PRB share).
 */
#ifndef HARMONY_CC_ASSOC_H
#define HARMONY_CC_ASSOC_H

#include <stdint.h>

static inline int cc_capped(uint64_t ue_ref, uint64_t congestor_ref, long long protected_ref)
{
  return protected_ref >= 0 ? ue_ref != (uint64_t)protected_ref : ue_ref == congestor_ref;
}

#endif
