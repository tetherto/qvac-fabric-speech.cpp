# Record the Parakeet iPhone demo

This guide is for anyone who needs to install Parakeet Lab on an iPhone and
record a product demo. No model files need to be copied into Xcode manually:
the preparation script downloads the source checkpoint, creates the QVAC model
artifacts, and embeds them in the app build.

## What you need

- An Apple-silicon Mac with at least 10 GB of free disk space.
- The full Xcode application, opened once so its license and components are
  installed.
- CMake and Python 3.11. With Homebrew, install them with
  `brew install cmake python@3.11`.
- A physical iPhone or iPad running iOS 16.4 or newer, a cable, and roughly
  2 GB of free device storage.
- An Apple ID configured in Xcode under **Xcode > Settings > Accounts**.
- A stable internet connection. Initial setup downloads a 2.5 GB checkpoint
  and Python dependencies and can take 20–45 minutes.

The app is device-only. The iOS Simulator cannot provide representative Metal
or Neural Engine performance.

## 1. Prepare the demo

Open Terminal, enter the repository, and run:

```bash
cd /path/to/qvac-fabric-speech.cpp
./examples/parakeet-ios/scripts/prepare-demo.sh
```

The script is safe to rerun. Completed downloads and generated artifacts are
reused. When it finishes, open the project:

```bash
open examples/parakeet-ios/build-ios/ParakeetLab.xcodeproj
```

## 2. Install it on the iPhone

1. Connect and unlock the iPhone. Tap **Trust** if prompted.
2. In Xcode, select the **ParakeetLab** project and then the **ParakeetLab**
   target.
3. Open **Signing & Capabilities** and choose your team. If Xcode reports that
   the bundle identifier is already taken, replace it with a unique value such
   as `com.yourname.ParakeetLab`.
4. Select the connected iPhone in the device selector at the top of Xcode.
5. Press the Run button. The first build is slower because the native engine
   and the large model bundle must be compiled, copied, signed, and installed.
6. If the phone asks for Developer Mode, enable it under
   **Settings > Privacy & Security > Developer Mode**, restart the phone, and
   press Run again.

Keep the phone connected until installation completes. The model files are
inside the app; inference itself does not require a network connection.

## 3. Run the presentation

1. Open **Parakeet Lab** and keep **Core ML** selected.
2. Leave the bundled sample selected, or tap **Import file** to choose another
   audio file.
3. Tap **Transcribe**. The transcript appears while the benchmark runs.
4. Wait for **Transcription complete**. The result card shows inference
   time in seconds, real-time speed, and model-load time.
5. Switch to **Metal** and tap **Transcribe** again to show the on-device
   backend comparison.

The first run after installation includes model initialization and may be less
visually smooth. Do one rehearsal before recording. For a fair comparison, use
the same audio for each mode and allow each transcription to finish.

## 4. Record a clean demo

Before recording:

- Enable Do Not Disturb and hide notification previews.
- Use portrait orientation and set display brightness high enough to film.
- Keep the device connected to power.
- Rehearse the sequence once so both models are initialized.

To capture directly on the phone, add **Screen Recording** to Control Center,
start recording, perform the demo, then stop the recording. The video is saved
to Photos.

For a cursor-free capture on the Mac, connect the iPhone, open QuickTime
Player, choose **File > New Movie Recording**, select the iPhone from the arrow
beside the record button, and record the iPhone screen.

## Troubleshooting

**The preparation command stops during download**

Run the same command again. The checkpoint download resumes when possible.

**Xcode says signing requires a development team**

Choose your Apple ID team under **Signing & Capabilities**. A free personal
team is sufficient for a short-lived local installation.

**The app says the Core ML encoder did not load**

Do not add models to Xcode by hand. Rerun `prepare-demo.sh`, then use
**Product > Clean Build Folder** in Xcode and Run again.

**The build is unexpectedly small**

The finished app should contain both the Q8 GGUF and the compiled Core ML
encoder. Confirm that `prepare-demo.sh` ended with its final green check marks
before opening Xcode.

Technical implementation details are in
[`parakeet-ios/README.md`](parakeet-ios/README.md).
