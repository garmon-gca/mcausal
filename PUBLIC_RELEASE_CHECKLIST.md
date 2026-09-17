# Public release checklist

Recommended public tag: **`v0.1.2`** (hardening: protocol verification + honest update estimates).  
`v0.1.0` and `v0.1.1` are preserved unchanged.  
Visibility is still **PRIVATE**. External usability is still **NOT YET**.

The two open boxes (`GitHub clone`, `tag → reviewed commit`) on the `v0.1.0` tree were verified after that tag. See `POST_FREEZE_VERIFICATION.md`. **Do not retag `v0.1.0`.**

- [x] license approved (Apache-2.0)
- [x] attribution correct (`Mangust` in pyproject, README, CITATION.cff, NOTICE; `garmon-gca` only as GitHub owner/URL)
- [x] disclosure audit PASS (`DISCLOSURE_AUDIT.md`)
- [x] no private datasets
- [x] tests PASS (26)
- [x] reference suite PASS (R1/R2/R3 live)
- [ ] GitHub clone PASS (filled after `git clone` of this repo)
- [x] README verified (engineer-first, no D1–D10 history)
- [x] issue templates verified (bug + external usability)
- [x] evidence claims audited (`docs/CLAIMS_AND_LIMITATIONS.md`, `docs/EVIDENCE.md`)
- [x] External usability still `NOT YET`
- [x] no unsupported claims (PARTIAL stays PARTIAL)
- [ ] tag points to exact reviewed commit (filled after tag)

```
PUBLIC_VISIBILITY_CHANGE = REQUIRES_EXPLICIT_USER_APPROVAL
```
