# In-the-wild examples

[Back to README](../README.md#demo)

The demo inputs are in [assets/](../assets). Selected screenshot borders and overlays have been
removed by rectangular cropping; retained pixels have not been retouched. Every image below
is an individual file. Purifier images were generated using the released checkpoint pair.
The displayed routing labels include the author's corrections: **demo_06, demo_08, and demo_09
are KEEP**, and their inputs are passed through unchanged without invoking the purifier.
These corrected labels are qualitative presentation labels, not newly measured router scores.
At runtime, the frozen threshold is **0.006440580124035478**; purification is triggered when
`p >= threshold`. The demo JSON records the actual model decision and probability.
These are qualitative examples without clean ground truth.
The examples illustrate both PURIFY and KEEP. Runtime decisions must be recomputed if an input
is cropped or otherwise changed.

| Case | Input | Router | Purified image / action |
|---|---|---|---|
| demo_01 | <img src="../assets/demo_01.png" width="200" alt="Input demo_01"> | **PURIFY** | <img src="images/purified/demo_01.png" width="200" alt="Purified demo_01"> |
| demo_02 | <img src="../assets/demo_02.png" width="200" alt="Input demo_02"> | **PURIFY** | <img src="images/purified/demo_02.png" width="200" alt="Purified demo_02"> |
| demo_03 | <img src="../assets/demo_03.png" width="200" alt="Input demo_03"> | **PURIFY** | <img src="images/purified/demo_03.png" width="200" alt="Purified demo_03"> |
| demo_04 | <img src="../assets/demo_04.png" width="200" alt="Input demo_04"> | **PURIFY** | <img src="images/purified/demo_04.png" width="200" alt="Purified demo_04"> |
| demo_05 | <img src="../assets/demo_05.png" width="200" alt="Input demo_05"> | **PURIFY** | <img src="images/purified/demo_05.png" width="200" alt="Purified demo_05"> |
| demo_06 | <img src="../assets/demo_06.png" width="200" alt="Input demo_06"> | **KEEP** | Input unchanged; purifier not invoked. |
| demo_07 | <img src="../assets/demo_07.png" width="200" alt="Input demo_07"> | **PURIFY** | <img src="images/purified/demo_07.png" width="200" alt="Purified demo_07"> |
| demo_08 | <img src="../assets/demo_08.png" width="200" alt="Input demo_08"> | **KEEP** | Input unchanged; purifier not invoked. |
| demo_09 | <img src="../assets/demo_09.png" width="200" alt="Input demo_09"> | **KEEP** | Input unchanged; purifier not invoked. |
