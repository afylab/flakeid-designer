# Archived tools

One-off analyses, comparisons and labelling-set makers from the development of FlakeIdentifier (September-October
2026). Nothing in the package or the other tools uses them; they are kept for reference. They were written to run from
`tools/` (they find the repository as their parent's parent and import sibling tools from `tools/`), so move one back
to `tools/` before running it; some also expect the data folders of that time.

| file | what it was for |
|---|---|
| `afm_calibration_report.py` | Render tools/calibrate_afm.py's report.json (plus an optional k-fold run) as a single HTML page. |
| `analyze_100x_scores.py` | What separates the 100x images scored useful from the rest, and what the 10x analysis could have known. |
| `artifact_compare.py` | Before / after images of the 100x optics-artifact removal, to check what it removes and whether it adds anything. |
| `compare_scans.py` | Compare two scans of the same chip (e.g. before / after changing the acquisition). |
| `contrast_options.py` | Contrast-boosted renderings of 100x images, to compare ways of showing debris, residue and terraces on a flake. |
| `make_demo_catalog.py` | Designer catalog from high-magnification images of individual flakes. |
| `make_label_set_mm.py` | Flake-labelling set (tools/label_page.html) from a Micro-Manager scan analysed by find_flakes. |
| `make_residue_label_set.py` | Flake vs tape-residue labelling set (tools/residue_page.html) from live-analysed chips. |
| `make_thickness_label_set.py` | hBN thin / thick labelling set (tools/thickness_page.html) from a live-analysed Micro-Manager scan. |
| `make_wrinkle_label_set.py` | Graphite wrinkle labelling set (tools/wrinkle_page.html) from 100x frames. |
| `residue_page.html` | labelling page for make_residue_label_set.py |
| `score_page.html` | scoring page of an early 100x label set |
| `terrace_tuning_data.py` | Data for the terrace-threshold judging page: the step-line terraces (tools/terraces100.py) of a set of hBN flakes for every combination of the thresholds, with the flakes' pictures. |
| `thickness_page.html` | labelling page for make_thickness_label_set.py |
| `wrinkle_page.html` | labelling page for make_wrinkle_label_set.py |
