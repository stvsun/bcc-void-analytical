# Paper Completion Plan: BCC Void Analytical Solution

## Current Status (15 pages, compiles cleanly)

**Complete:**
- Sections 1–6: Analytical derivation (intro, crystal geometry, yield polygon, stress field, interior field, activation pressure)
- Section 7: BCC vs FCC comparison (table + 3 subsections of prose)
- Section 8: Numerical verification (4 subsections: model, sector verification, EP vs RP, plastic zone)
- Section 9: Discussion (radiation swelling, anti-plane systems, plastic zone, ductile fracture models, limitations)
- Section 10: Conclusions (5 items)
- Bibliography: 22 entries

---

## Narrative Weaknesses Identified

### W1. Interior stress field is approximate, not exact (Section 5)
**Issue:** The formula σ_rr = σ_θθ(a,θ)·ln(r/a) is presented as the "leading-order solution" but the text doesn't rigorously justify the approximation or state its validity range. It assumes σ_rθ ≈ 0 for r > a, which breaks down away from the void.
**Fix:** Add a paragraph explicitly stating this is a zeroth-order approximation valid for r/a - 1 ≪ 1, and that the full r-θ coupled solution requires the method of characteristics. State that the approximation is exact in the isotropic limit.
**Effort:** 15 min

### W2. Activation pressure definition is ambiguous (Section 6)
**Issue:** p* = max|σ_m| at the void surface is stated, but it's unclear what this physically means. For an infinite rigid-plastic medium, full plasticity requires p → ∞. Is p* the onset of first yield? The pressure for a given plastic zone extent? The pressure at which the rigid-plastic solution first applies?
**Fix:** Clarify that p* is the minimum far-field pressure at which a fully plastic stress field consistent with the yield condition and void boundary conditions can exist. This is analogous to the limit-load pressure in classical plasticity. Add a sentence connecting to the isotropic limit where p* = 2k for Tresca.
**Effort:** 15 min

### W3. No velocity field or void growth rate
**Issue:** Kysar Part I derives both stress and velocity fields. This paper has no velocity field, which is a notable omission for JMPS/IJP.
**Fix (minimal):** Add a paragraph in Section 5 or Discussion stating that the velocity field can be constructed from the flow rule and the known active slip systems in each sector, but the detailed derivation is deferred to the companion paper on non-equibiaxial loading. Alternatively, derive the equibiaxial velocity field (would add ~1 page).
**Effort:** 30 min (paragraph) or 3 hours (full derivation)

### W4. CPFEM doesn't verify stress magnitudes quantitatively
**Issue:** The sector structure verification (100% match) is qualitative. The stress magnitude comparison shows a systematic offset. A reviewer may view this negatively.
**Fix:** Strengthen the narrative by emphasizing that: (a) the sector structure IS the primary analytical prediction, (b) the stress magnitude comparison is between two different physical models (elastic-plastic vs rigid-plastic), not a convergence test, and (c) reference Gan et al. (2006, Part II) where the same elastic-plastic vs rigid-plastic distinction exists for FCC. Add a sentence noting that Kysar's FCC paper also relies primarily on sector structure verification.
**Effort:** 15 min

### W5. Geometry schematic figure (Fig. 1) is missing
**Issue:** Figure caption and \label exist but no actual figure file. Compilation produces a warning.
**Fix:** Create a TikZ or matplotlib figure showing the void, crystal axes, polar coordinates, and loading.
**Effort:** 45 min

### W6. FCC yield polygon comparison figure (Fig. 2b) is rough
**Issue:** The current bcc_vs_fcc_yield_surface.png has the FCC hexagon rendered incorrectly (appears as a rectangle due to normalization issues in the plotting code).
**Fix:** Regenerate the figure with correct FCC yield polygon vertices and consistent normalization.
**Effort:** 30 min

### W7. Redundancy between Section 6 and Section 7.3
**Issue:** The p*_BCC/p*_FCC = 3/2 derivation appears in both sections with overlapping content.
**Fix:** Keep the derivation in Section 6 (Activation Pressure), and in Section 7.3 focus only on the physical interpretation (elongated polygon → higher activation). Remove the re-derivation of Eq. (ratio_section) from Section 7.3.
**Effort:** 10 min

---

## Remaining To-Do Items (Priority Order)

| # | Item | Fixes | Effort | Impact |
|---|------|-------|--------|--------|
| 1 | Clarify interior field approximation (W1) | Add validity statement to Sec 5 | 15 min | High — prevents reviewer confusion |
| 2 | Clarify activation pressure definition (W2) | Add sentence to Sec 6 | 15 min | High — prevents reviewer confusion |
| 3 | Address missing velocity field (W3) | Add paragraph to Sec 5 or Discussion | 30 min | Medium — preempts reviewer question |
| 4 | Strengthen CPFEM narrative (W4) | Edit Sec 8.3, reference Gan et al. | 15 min | Medium — addresses potential criticism |
| 5 | Fix FCC yield polygon figure (W6) | Regenerate plot with correct FCC | 30 min | Medium — visual accuracy |
| 6 | Create geometry schematic (W5) | TikZ or matplotlib | 45 min | Low-Medium — nice to have |
| 7 | Remove Section 7.3 redundancy (W7) | Edit Sec 7.3 | 10 min | Low |
| 8 | Acknowledgements | Fill in | 5 min | Required |
| 9 | Final compile and proofread | pdflatex+bibtex, check refs | 15 min | Required |

**Remaining effort: ~3 hours**
