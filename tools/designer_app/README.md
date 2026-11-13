# vdW Stack Designer — desktop app

The designer (`flakeid_designer/`) in its own window, opened and closed from the Dock / taskbar. Same page, same
libraries, designs and decisions as the browser version (`flakeid_designer/app.py`: the local server on port 8766
and the page in a native window via pywebview — WebKit on macOS, WebView2 on Windows 10/11). Self-contained: no Python
needed on the computers it is installed on; a new version means installing the new build.

## Versions

- **Office** (`vdW Stack Designer`): for designing on personal computers; no camera.
- **Stacker** (`vdW Stack Designer (Stacker)`): adds the live camera for the transfer stations (header button
  **Live camera**): the station's camera picture under the design, placed with Align (2 points) / Move / Flip, its scale
  and turn kept per station and objective. Cameras: the stations' DFK 33UP1300 through IC Imaging Control 4 (The
  Imaging Source), any camera Windows shows as a normal camera (OpenCV), and a simulated camera (with stamp jog
  buttons) for trying it without hardware. Same designer and design files as the office version.

## Build

- **Mac (Apple silicon):** `sh tools/designer_app/build_mac.sh` (office) or `sh tools/designer_app/build_mac.sh stacker`
  → `dist_app/<name> <version>.dmg`
- **Windows 10/11:** on a Windows 10/11 PC with Python 3.9+ (python.org): `tools\designer_app\build_windows.bat` (office)
  or `tools\designer_app\build_windows.bat stacker` → `dist_app\<name>-<version>-windows.zip` (PyInstaller cannot build
  Windows apps on a Mac).
- **Stacker PCs:** IC Imaging Control 4 needs The Imaging Source's IC4 driver for USB3 Vision cameras on the PC (from
  theimagingsource.com, "IC Imaging Control 4"); without it the camera may still work through the "System camera
  (OpenCV)" choice via the DirectShow driver IC Capture installs.

The version is `VERSION` in `flakeid_designer/__main__.py`.

## Install

- **Mac:** open the .dmg, drag the app to Applications. The app is not signed: the first time, right-click it →
  Open → Open. Keep it in the Dock (right-click its Dock icon → Options → Keep in Dock).
- **Windows:** unzip (to a short folder, e.g. `C:\vdw`: Windows refuses paths over 260 characters), run `Install.bat` (per user, no admin): installs to `%LOCALAPPDATA%\Programs\vdW Stack Designer`
  with a desktop and Start-menu shortcut; pin it to the taskbar from the Start menu. Running it again updates.

## Notes

- Exports (catalog CSV, design JSON) open a native Save dialog; folder pickers are native dialogs.
- The browser version (`Install shortcut` files in `flakeid_designer/`) keeps working alongside.
