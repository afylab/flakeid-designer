vdW Stack Designer, portable version (FlakeIdentifier, Young Lab)

Install once per computer
  Windows: double-click "Install shortcut (Windows).bat"   (needs Python 3.8+ from python.org)
  Mac:     double-click "Install shortcut (Mac).command"   (uses python3; macOS may ask to allow it the first time)
  This puts a "FlakeIdentifier Designer" icon on the desktop (Mac: also in Applications; drag it to the Dock).

Use
  Four views (buttons at the top), in the order of the work:
    Library   scan runs, the flakes found, screening (keep / discard)
    Plan      design the stack (Design with Claude is here)
    Stack     assemble it at the stacker: pick-up order, live camera, pictures, where each layer really went
    Finalize  the finished stack on a picture from the better microscope; Export to KLayout
  Directory (next to the view buttons): whose scan results, stacks and decisions; switch to another person's.
  Right-hand panel: Flake info (the selected layer), Visibility (overlap, contacts, layer images, outlines, planned
  positions, hBN, grid), plus Design with Claude in Plan and Steps in Stack.  In Stack the pick-up marker is always
  on, at the next layer to pick up.
  Library > Imported flakes > Import flake: a flake from any microscope picture.  Choose the picture and its objective
  (scale), click its corners to draw the outline (click the first corner to close), enter material and oxide, then
  type the thickness or "Estimate thickness from colour" (the flake's colour against the substrate around it, with the
  scan's optical model: rough, tens of %; hBN may give two candidates; confirm with AFM when it matters).  A context
  picture (e.g. low magnification) with the flake marked is shown at the stacker instead of the scan map ("Context
  picture" in Flake info).  Imported flakes are in the flake browser (Scan: manual) and go into stacks like any other.
  One file per stack (plan, stacking, final state), in <library>/_designer_data/stacks/.  Save always updates the
  open stack (it is remembered when the page is reopened); New starts another.  Each view keeps its own layer
  positions: Stack starts from the plan, Finalize from where stacking left the layers; the plan shows dashed.
  Stack view, Steps tab: each pick-up is a step.  Mark a layer picked up (pick-up marker), take a picture, line the
  picked-up layers up with it (Edit layer), then Confirm positions: the step keeps the picture and where the layers
  are, and the next layer is up.  "+ Step" records the release onto the target or any other event.  Each step has
  notes, attachments (AFM data, pictures from other microscopes; any file) and per-layer notes (shifted, folded,
  torn, bubble ... and why: chosen at the stacker, apparatus / alignment limit, unexpected).  Click a step to see its
  picture with the layers where they were then; changes there edit that step (saved by themselves).  Any computer
  can open and edit the steps afterwards.
  Two computers on one stack: a note says where else it is open; a save never overwrites a version saved elsewhere
  after you opened the stack: it asks (open that version, or save yours over it).
  Old design files (before stacks): a banner offers "Convert to stacks"; it first makes a backup of the whole
  _designer_data folder next to it (_designer_data_backup_<date>, every file checked) and leaves the old files as
  they were.
  Click the icon: the designer opens in your browser.  In "Scan results", add YOUR OWN scan results folder (for example
  <share>/<your name>/scanresults) and tick the runs to load.  Runs are the "designer" folders the scan
  program writes into each run's results.  Flakes shared with others: put those runs in a shared folder and add it as
  a second folder.
  Your designs and keep/discard decisions are kept in <your first folder>/_designer_data (yours only).
  "Import new scan results..." adds another folder of runs to your library.
  A shared computer: "Change directory..." (in Scan results) switches to another person's library: their scan results
  folder, with their designs, decisions and runs; libraries used before on the computer are listed for one click.
  Runs offered for screening are remembered in the library's data, so they are offered only once, on any computer.
  The runs list (Select directory -> Runs in this directory...) shows for each run whether it was screened: "not
  screened", "partly screened" with how many of its flakes have a keep / discard decision, or "screened" (all of them),
  counted from everyone's decisions in the library.
  A new run's designer files are built by the designer itself, on your computer, not on the microscope PC: when the
  scan program has finished copying a run to the share (it writes copy_done.json last), the designer finds the run,
  builds its designer/ folder in the background (a banner says "Building designer files ..."; a few minutes per run)
  and the run then appears.  A re-imaged run is rebuilt the same way.  Whoever opens the designer first builds it, for
  everyone (a lock file designer/.building keeps two computers from building the same run).  Older runs without a
  designer folder are listed as "not built" with a Build button.  A build that fails leaves designer/build_failed.txt in the run's
  folder on the share (who, which computer, the error): send it if a run does not appear.  The browser version needs numpy, scipy,
  opencv-python and scikit-image (pip install numpy scipy opencv-python scikit-image); the apps include them.
  Kept flakes' pictures: the flakes in the Library's "kept" list (a terrace kept; not picked-up ones) have all their
  100x pictures copied into the run's folder on the share, <run>/kept/, one subfolder per kind, one version of a picture
  per folder: raw, raw_avg (frame average), raw_tiff (if still there), natural, natural_avg, color_8percent (+-8 %
  colour), color_8percent_avg, bw (+-4 % grey), bw_avg, thickness_change, debris, ir, mosaic, mosaic_natural,
  ir_domains (16-bit PNG), ir_domains_preview; files are named <piece>-f<flake number>_<original name>, and
  kept/kept_flakes.csv lists them.  Discarding the flake later removes its copies.  The folder is brought up to date
  when the designer starts, a few seconds after each decision, and with "Update kept folder" (a loaded run in the
  Library: also removes leftover pictures that are not in its list).
  Stars: rate a flake 1-3 stars in the 100x picture / screening (keys 8, 9, 0; again to clear), shared in the library.
  The Library's Show (decisions) and Stars filters come first; sort by "most stars".  AFM list: "groups: kept / stars"
  takes the flakes in any of the ticked groups (e.g. 1 star + 3 stars); "Show the flakes in the list" lists them with
  a small picture (click: the 100x picture).  "Export these N flakes (JSON)" saves the list's flakes (e.g. all the
  2-star ones, or flakes picked by hand with Add manually / ✕) as flakes_<name>_<time>.json in the run folder and in
  the AFM lists: the scan program's "Remeasure flake coordinates" › Custom reads it.  Library › a run › Reset screening clears its keep / discard decisions
  (stars and picked-up flakes stay; asks once more).
  AFM center: the red cross and scan square on a flake's 100x picture (and the red cross on the Library pictures,
  "AFM center" checkbox): the picture's center unless moved; click or drag on the picture to move it (saved with the
  flake for the library; "Reset AFM center" puts it back).  An AFM list scans each flake there; its table's 100x
  button opens the scan-site window to move it too.  The list's positions are the stage positions the flakes' 100x
  pictures were taken at (plus the AFM center's offset in the picture), fitted to the reference flakes' AFM X, Y: no
  10x coordinates (since 2026-10-30c; a list of mounted chips uses the block's corners, marked at 10x).  The 100x window's i icon lists the keyboard shortcuts; Measure
  and Clear rulers are on the second row.
  Where AFM centers are kept: with the library's decisions (<library>/_designer_data/review/<person>.jsonl, id
  "<run>:<piece>-f0012.clean"), and next to the 100x pictures (<piece>/100x/afm_centers.json, by picture file name).
  Both are offsets on the flake's 100x picture (fraction of the picture, and um from its center), never stage
  coordinates: a run made again from the same 100x pictures with new stage positions keeps them (copy
  afm_centers.json with the pictures; the designer takes it for flakes of the new run without a center yet: the
  picture's new position plus the um offset, turned by "rot_deg" if the file gives the chip's turn).
  Orientation: everything (flakes, stacks, the 10x chip picture, the AFM list's map) is shown turned 180 degrees by
  default ("View" rotation at the top of those windows; this computer remembers its choice).  The 10x chip picture can
  also be flipped left-right / top-bottom (Flip buttons in the Chip overview and the AFM list).  View only: positions,
  the design and AFM lists stay in the scan camera's frame.  In the AFM list, the 10x surroundings of each reference
  flake zoom with the mouse wheel (drag to move, double-click: back), as the chip map does.
  100x picture window (and screening): a scale bar in the bottom right; Measure (key R) measures distances on the
  picture: click two points (right-click or Backspace removes the last line, Esc stops).
  Screening and the Library list: the arrow buttons at the top left of the 100x window (or the arrow keys) go to the
  previous / next flake, e.g. to change a keep / discard made by mistake.  Pictures: 1 natural, 2 high contrast, 3 the
  flake in the chip's 10x scan (read from the run's folder on the share; mouse wheel zooms, drag moves, double-click:
  back, O its 10x outline), 4 thickness change, I IR domains.
  Library › a run › Rebuild: builds the run's designer files again on this computer, all of them (e.g. a chip picture
  that looks wrong); decisions, stars, AFM lists and designs stay.
  "Design with Claude" explains how to design stacks with your own Claude account.
  Problems: send ~/.flakeid_designer_log.txt (Windows: C:\Users\<you>\.flakeid_designer_log.txt) and a screenshot.

While stacking
  "View ↻" (canvas toolbar) turns every flake picture and the stack together, e.g. 180° to see them as with the
  Olympus Stream camera; it changes only the view (positions and saved designs stay in the scan camera's frame).
  "Pick-up marker" shows only the layers picked up so far; move the marker in the stack list with its arrows (or the
  , and . keys); the overlap and contacts then describe the stamp at that step.  A flake already used in a design is
  hidden from "Add layer" (tick "show flakes already used" to see them greyed, with the design's name).
  "✓ picked up" (beside the marker) marks the flakes of the layers above it as picked up; ✓ on a layer's row marks or
  unmarks that one.  The design is saved and the mark is kept with the shared keep / discard decisions, so the flake
  stays out of every other stack even if this design is deleted (unmarking puts back its earlier decision).

Live camera at a transfer station (stacker version): the camera's scale
  Each objective (5x, 10x, 20x, 40x, 50x) has its own µm per pixel.  Calibrate each once: put a chip with the pads
  layout's alignment marks under the objective, in focus, with the four two-L marks at the chip's middle in view (at 5x
  and 10x more of the chip is fine), choose the objective in the live bar and press "Calibrate from marks".  A few live
  frames are averaged, the marks are found and matched to the chip's layout (the same GDS as Export to KLayout's
  template; it asks for it the first time), and it shows the µm per pixel, the camera's turn against the chip, the
  marks used (circled on the picture) and the fit's rms.  Save keeps it for this station ("✓ marks" beside the field).
  It refuses a fit on too few marks, and warns when a result does not fit the other calibrated objectives (µm per pixel
  x magnification is the same for all of them).  Set Flip first if the station's camera sees the chip mirrored: the
  marks look the same mirrored, so the picture cannot tell.
  The stacker camera sees the chip mirrored left-right, so the stack is mirrored left-right against the live picture
  by default; "Mirror <->" and "Mirror ^v" change that (kept for the station).  Ticking a layer's "image" box shows the
  layers' images over the live picture (Layer images on).  A grey picture from an Imaging Source camera: the program
  now sets the camera to a colour (Bayer) pixel format (the camera may have been left in Mono8 by IC Capture); the
  log ~/.flakeid_designer_log.txt says which format it found and set.
  Edit layer (live bar): the most recent layer (the last one marked picked up; select another to edit it) is moved and turned to where it
  really went at the stacker: drag moves it, Shift+drag turns it, arrow keys nudge 1 um (Shift 10), [ and ] turn 0.2
  degrees (Shift 2).  Its planned place stays as a dotted outline and the bar shows how far it is from the plan; Ctrl+Z
  undoes.  Edit layer again: dragging moves the whole stack over the picture as before.
  Pick-up marker: it starts at the first layer not marked picked up yet.  After "picked up" a prompt asks for two
  things before the next layer: Take picture (the live picture of the stamp is saved with that layer, placed over the
  design as it was; the layer's panel lists it, click to see it under the stack later) and Adjust layer positions (live
  view with Edit layer on that layer: drag it onto its flake; select other picked-up layers to adjust them too).
  Outlines on / off (top-left corner of the live picture, or the O key): hides every layer outline to check the
  alignment against the bare camera picture.

Drawing your own terraces
  Open a flake's 100x picture (a layer's picture button, or a flake in "Add layer") and press "Draw terrace": click
  the corners, then click the first corner again (or double-click, or Enter) to close it; Backspace removes the last
  corner, Esc cancels.  Edges snap to the flake's crystal directions (every 30 degrees) when within 5 degrees; hold
  Alt to place a corner freely.  Give the thickness (the picked flake's is filled in) and a note, then "Save terrace".
  A drawn terrace (marked ✎) works like a detected one: it is found by "Add layer", listed under the layer's "Other
  terraces", and "Use for layer N" puts it in the open stack in place of the current terrace.  Open it and press
  "Delete this terrace" twice to remove it.  Drawn terraces are shared with everyone using the same library
  (_designer_data/terraces/*.json: the outline in the picture's pixels, who drew it, when, on which picture), and are
  kept as examples for teaching the automatic terrace finder.

Recording where the layers really ended up (as built)
  After assembling a stack: open its design, press "Save as-built copy" (the plan stays as it was), then
  "Stack picture..." to load a microscope image of the finished stack (JPEG or PNG; choose the objective for the scale).
  "Align (2 points)": click a feature on the picture, then the same point on the design, twice (far apart): the picture
  is moved, turned and scaled to match.  Then drag each layer onto its outline in the picture ("Layer images" off shows
  outlines only; dashed white lines show where each layer was planned) and Save.  The contact check and cross-section
  then describe the stack as built.  Pictures are kept in _designer_data/stack_images.
  Move/turn stack (with a stack picture): all layers moved and turned together over the picture, keeping their places
  in the stack: drag moves them, Shift+drag turns them about the stack's centre, arrow keys nudge 1 um (Shift 10), [ ]
  turn 0.2 degrees (Shift 2).  Usually the stack is already close: line it up as a whole first, then fine-tune single
  layers.

Export to KLayout (pictures of a finished device placed on the chip layout by its alignment marks)
  "Export to KLayout..." (top bar): choose the template GDS (the chip layout with the marks), the folder with the
  pictures (untick any to leave out), optionally an existing KLayout session to add them to as a new tab (a backup of
  it is saved first), the layer properties (.lyp) and the output folder (default: <picture folder>/<name>_klayout).
  Paths can also be typed or pasted.  Export writes <name>.lys, <name>.gds and <name>_check.jpg (the fitted marks drawn
  on each picture: check it); open the session in KLayout with File > Restore Session.  The choices are remembered.
  It needs three packages in the designer's Python, once: python -m pip install numpy opencv-python gdstk
  (the dialog shows the exact command when they are missing).
  Adding to an existing session: KLayout does not reload a session file, so a KLayout window that already shows it
  will not show the new tab.  Close that window WITHOUT saving the session (saving would remove the new tab), then use
  "Open in KLayout" in the result (a new window with the session restored) or File > Restore Session.
  "Stack design" (ticked when the open design has a stack picture): the design's direct contacts (where gate / sample
  layers touch with no dielectric between: layer 253, /0 shorts, /1 sample-sample) and each layer's outline (254/i)
  go into the GDS, placed on the chip through the stack picture (align the layers on it first: as-built record).

Working away from the drive (offline stacks)
  Your active stacks (not finalized) and the stacks you opened in the last 30 days are copied to this computer while
  the drive is reachable: the stack, its pictures, and its flakes with their 100x pictures (a few MB per stack; in
  %LOCALAPPDATA%\flakeid-designer\offline on Windows, ~/Library/Application Support/flakeid-designer/offline on a Mac).
  "Offline..." (top bar) lists them with their state; there each stack can be set to Keep offline / Not offline /
  Automatic, and "Keep my active stacks offline automatically" turned off (setting offline_auto in
  ~/.flakeid_designer.json).  When the drive is not reachable the designer says so at the top and works on these
  stacks only: edit the plan, add steps, pictures and notes, keep / discard their flakes; save as usual.  When the
  drive is back the changes are uploaded by themselves ("Upload now" to do it at once).  If someone changed the same
  stack on the drive meanwhile, nothing is overwritten: your version is saved as a stack of its own,
  "<name> (offline edits <date>)".  Copies are made only while the designer is running: open it once on the lab
  network before leaving.

The designer runs on your computer only (nothing is uploaded); it stops by itself 15 minutes after its last tab closes.
Command line: python flakeid_designer --help
  Update… (top, next to Refresh; 2026-10-30): the newest designer from GitHub (afylab/flakeid-designer):
  what is new, then Install: the code is downloaded, checked, backed up (~/.flakeid_designer_update_backups, the last 3) and
  copied over; settings, libraries, designs and decisions stay.  The browser version then restarts and the page reloads;
  the app is closed and opened again.  No GitHub token is needed (a private fork needs a read-only one; the Update window
  says how; on a computer with the GitHub CLI logged in, its login is used).  python -m flakeid_designer.update does it from a terminal.
