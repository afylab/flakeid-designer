# vdW Stack Designer: one file per stack, four views (plan of record, 2026-10-14)

Decided with Jordan on 2026-10-14. The designer follows the process of building a heterostructure: find flakes,
plan the stack, assemble it at the stacker, finalize it for lithography. Flakes shift, turn, fold and tear during
assembly; some changes are chosen at the stacker, some come from the apparatus and its limited alignment precision.
The stack file keeps all of it.

## Views (selector at the top)

| view | what | where today's features go |
|---|---|---|
| **Library** | runs, flakes, screening (keep / discard), screened badges; **manual flake import**: load microscope pictures, draw outlines by hand; each picture can have a **context picture** (e.g. low magnification) used instead of the scan map to find the flake on the chip at the stacker | runs list, review, drawn terraces' data |
| **Plan** | today's design interface | stack list, layer panel, Design with Claude (only here), drawn terraces |
| **Stack** | the assembly, step by step; stacker version: live camera; office version: every step viewable **and editable** afterwards | pick-up marker, live view, Take picture, Edit layer, Move/turn stack, Outlines, Mirror, Calibrate from marks |
| **Fabrication** (was Finalize; internal key `finalize`) | the fabrication steps: each a set of the scanner's Device photos (100x / 50x / 10x / 5x + extras) with what was done, viewed with ← → (step) and ↑ ↓ (magnification); the finished stack on pictures from the better microscope (always the source for KLayout), final positions and **editable shapes**, Export to KLayout | stack picture bar (Align, Move picture, Move/turn stack), KLayout export |

- Overlap: every view shows the overlap of the current configuration only (the real one there).
- KLayout settings (template GDS, layer properties, last session): per person (as now, ~/.flakeid_designer.json).
- Stack view: after a pick-up, the picture is taken, then the layers are lined up with it; a **Confirm positions**
  button records the step and moves on to the next layer.
- Each step: notes (process details) and **attachments** (AFM data, optical pictures from other microscopes).
- A lock / "being edited by" note per stack: the usual case is one person with the stack open on their own PC and
  on the stacker PC.
- The claude.ai designer page is retired (the portable designer and the apps only).

## Stack file: `_designer_data/stacks/<id>.json` (format `vdw-stack/1`)

```
{format, id, name, created, updated, by, status: planning | stacking | finalized,
 plan:     {layers: [...as today's design layers...], overlap_um2, overlap_d_um,
            revisions: [{at, by, note, layers}]},          # plan changes after stacking started
 steps:    [{id, kind: pickup | release | event, layer_uid, at, by, station, objective, um_per_px,
             photos: [{url, w, h, f, s, rot, mirror, tx, ty, at}],     # the picture and its placement
             attachments: [{url, name, kind: afm | optical | other, note}],
             poses: {layer_uid: {x, y, rot, mirror}},                  # where each layer really is at this step
             flags: {layer_uid: [{what, cause: chosen | apparatus | drift | unknown, note}]},
             notes}],
 final:    {images: [{url, ..., placement, objective, name}], poses: {...}, shapes: {layer_uid: outline}, notes},
 exports:  [{at, by, files, template, added_to}],
 fab:      {steps: [{id, title, description, at, by, taken, set, device, source, oxide_nm,
                     photos: [{objective, label, um_per_px, w, h, url, thumb, terrace_map, terraces: [{n, area_um2, contrast, hc}]}]}]},
                                                     # photos copied to stack_files/<id>/fab/<set>/
 legacy:   {sources: {design_id: <the original design file, verbatim>}}}
```

Positions are in the stack's own design coordinates; each picture also keeps its placement on them (so a picture
can be shown again where it was, in its camera's frame).

## Legacy data

1. **Backup first**: the whole `_designer_data` copied to `_designer_data_backup_<date>` next to it, every file
   checked (size and checksum) against the original; the migration refuses to run without a checked backup.
2. Migrate everything: each design plus its as-built copies -> one stack file (as-built positions -> final, the plan
   from the layers' planned positions, picked-up layers -> pick-up steps with their pictures). The original design
   files are embedded verbatim (`legacy.sources`) and left in `designs/` untouched.

## Order of work

1. Stack file format, backup, migration (tested on a copy of the real data first).
2. Server storage for stacks; the view selector; today's features moved into the views unchanged.
3. Stack view: step timeline, Confirm positions, notes and attachments, flags with causes; office review / editing.
4. Fabrication view (was Finalize): better-microscope pictures, final positions, shape editing, recorded exports;
   fabrication steps from the Device photos (2026-10-19).
5. Library view: manual flake import with drawn outlines and context pictures.

## Stack folders (2026-10-19, Jordan): one folder per device, readable without the designer

From 2026-10-20 on, a stack lives in a folder the user chooses (e.g. <share>/<your name>/Devices/<device>/); the
designer opens it with **Open...** (the folder, or the .stack file in it), **Open recent**, **New...** (a folder + a
name), and saves in place.  The stacks of _designer_data/stacks stay readable (Open > Older stacks); new ones are folders.

```
<device folder>/
  <name>.stack                  the stack file (JSON, format vdw-stack/1 as above); file references inside it are
                                paths relative to the folder (the folder can be moved / copied whole)
  01_stacking/                  pick-up pictures and attachments of the stacking steps (designer)
  NN_<step>/                    a fabrication step (NN = 02, 03, ... in order; <step> = its name, e.g. contacts)
      step.json                 {"format": "vdw-fabstep/1", "name", "description", "at" (ms), "by", "device", "oxide_nm",
                                 "source": "scanner device photos" | "designer import" | "designer: new step from
                                 pictures", "photos": [{"objective", "label", "file", "um_per_px", "w", "h",
                                 "terrace_map", "terraces": [{"n", "area_um2", "contrast", "hc"}], "align"}]}
                                 (file / terrace_map / hc: names in this folder; align: the photo placed on the
                                 substrate's layout, {"m": 3 x 3 pixel (x right, y down) -> layout um, "how": "marks" |
                                 "matched", "um_per_px", "angle_deg", "marks", "rms_um", "gds", "at"})
      <device>_<step>_<mag>.jpg main photos (100x, 50x, 10x, 5x); extra ones <device>_<step>_<mag>_2.jpg, ...
      <device>_<step>_100x_terraces.jpg, <device>_<step>_100x_T1.jpg, ...   (terrace map, colour +-8 % frames)
      <device>_<step>.html      the step's page (all photos, scale bars, terraces)
  <substrate>.gds               the chip's layout with its alignment marks (Fabrication › Import GDS…; the stack's
                                "substrate": the step photos are aligned on it, the KLayout export uses it)
  final/                        better-microscope pictures (originals, for KLayout)
  klayout/                      KLayout exports
```

- Device name: the name of the folder's .stack file (without .stack), else the folder's name.
- Next step number: one more than the highest NN_ folder (01_stacking counts as 01).
- The scanner's Device photos write a step folder directly (raw TIFFs deleted once the JPEGs, step.json and the page
  are written); the designer finds steps by their step.json (Fabrication view), and a step's description is edited
  there (written back to step.json).
- Stack folders are found by the user (Open / Open recent, per user in ~/.flakeid_designer.json), not by scanning.
- Steps from pictures taken any way (another microscope, SEM, a phone): Fabrication › New step from pictures…
- Aligned photos: Fabrication › Align pictures places every step photo on the substrate layout's alignment marks
  (marks_to_klayout.py --fits-json; pictures without enough marks are matched to placed ones); the Fabrication steps
  view then keeps the same place of the chip from step to step (Aligned, A), zoom / drag, and fades in the previous
  step's picture (slider, or hold B).

Implemented 2026-10-19 (designer 2026-10-19h, flakeid_designer/stackdir.py; scanner py-2026-10-19k):
- The page knows a folder stack by "d~" + a hash of its folder's path; its files are served at /stackfile/<id>/<path>.
  Inside the .stack file every such reference is written "stack:<path in the folder>".
- Several .stack files in one folder: the newest is used (designer and scanner alike).
- A step folder without step.json is an unfinished write (the scanner writes step.json last): not shown, but it takes
  its number.  A step taken off the designer's list stays on disk (listed in the stack's fab.removed).
- The lock note is <folder>/.designer_lock.json (renewed every minute while the stack is open).
