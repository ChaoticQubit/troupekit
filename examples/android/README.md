# Android demo app

A support agent built on troupekit with platform Android views: it answers from bundled support articles, streams its answer, and resumes a run after the app is force-stopped. It uses no Compose, no WorkManager and no AndroidX at runtime.

## Run it

1. Build the AAR once: `sh scripts/build-aar.sh` from the repository root.
2. Start an emulator or connect a device, API 26 or newer.
3. In `examples/android`: `./gradlew :app:installDebug`.

End-to-end tests: `./gradlew :e2e:connectedAndroidTest`. They run from a separate test APK that drives the app through the accessibility tree. That way `am force-stop` of the app does not stop the test. They use a scripted model, so they need no model file and no network.

## Which model answers

- **On-device, LiteRT-LM**: when a `.litertlm` model file is present in the app's files folder. The app never bundles a model. Download it at first launch, or push it for testing (`adb push gemma3-1b-it-int4.litertlm /data/local/tmp/`, then import it). The Gemma weights come under the Gemma license.
- **A cloud model** otherwise. Tap "Key" and paste your API key. It is encrypted with an AES-GCM key held in the Android Keystore. Only the ciphertext is stored, and the key is never part of the APK.

## Kill and resume

The app saves the run id the moment a run starts and a checkpoint after every step. Force-stop it mid-answer and open it again: it resumes the same run from the last checkpoint and shows the answer once. A step that was running when the app died runs again.

## Background window

A `JobScheduler` job (id 1) resumes an unfinished run for up to 25 seconds once a network is available, then saves and stops. Android decides when the job runs. Leave one model call of slack inside your own windows: a step in flight always finishes.

## Knowledge embedder

Android ships no on-device text embedder. The demo uses the catalog embedder (cloud) for its articles until an on-device one is approved.
