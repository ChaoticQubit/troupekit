# iOS demo app

A SwiftUI support agent built on troupekit: it answers from bundled support articles, streams its answer, and resumes a run after the app is killed.

## Run it

1. Build the XCFramework once: `sh scripts/build-xcframework.sh` from the repository root.
2. Open `examples/ios/TroupekitDemo.xcodeproj` in Xcode 16 or newer.
3. Pick an iOS 16 or newer simulator or device and run the `TroupekitDemo` scheme. On a device, set your own team under Signing.

UI tests: `xcodebuild test -project examples/ios/TroupekitDemo.xcodeproj -scheme TroupekitDemo -destination 'platform=iOS Simulator,name=iPhone 16'`. They use a scripted model, so they need no Apple Intelligence and no network.

## Which model answers

- **Apple's on-device model** when the device supports Apple Intelligence, it is turned on, and the model has finished downloading (iOS 26 or newer). The answer never leaves the device.
- **A cloud model** otherwise. Tap "Key" and paste your API key. It is stored in the Keychain, on this device only, readable only while the device is unlocked. It is never part of the app bundle.

The on-device model has a 4096-token window shared by instructions, articles and the answer, so the demo keeps the number of articles per question low on that path.

## Kill and resume

The app saves the run id the moment a run starts and a checkpoint after every step. Force-quit it mid-answer and open it again: it resumes the same run from the last checkpoint and shows the answer once. A step that was running when the app died runs again.

## Background window

The app registers a background processing task (`dev.troupekit.demo.resume`) that resumes an unfinished run for up to 25 seconds, then saves and stops. iOS decides when the window opens. Leave one model call of slack inside your own windows: a step in flight always finishes.

## On-device model files

Local GGUF models arrive with troupekit 0.3. Ship them with Background Assets rather than inside the app bundle, so the App Store download stays small.
