#!/usr/bin/env bash
# Compiles Glasses3DRelay.swift against the real Meta Wearables DAT SDK.
#
# Builds an isolated SPM package rather than touching the VisionClaw project, so
# a failure here is a problem with the relay, not with your app. Pinned to DAT
# 0.4.0 — the same version CameraAccess resolves.
#
#   ./verify-build.sh          # Swift 5 (matches the CameraAccess target)
#   ./verify-build.sh 6.0      # also check strict concurrency
set -euo pipefail
cd "$(dirname "$0")/Glasses3DRelayKit"

SWIFT_VER="${1:-5.0}"
# Keep the source of truth in ios/Glasses3DRelay.swift; this package just wraps it.
cp ../Glasses3DRelay.swift Sources/Glasses3DRelayKit/

echo "==> building relay against DAT SDK (Swift ${SWIFT_VER})"
xcodebuild -scheme Glasses3DRelayKit \
  -destination 'generic/platform=iOS' \
  -derivedDataPath ".build-${SWIFT_VER}" \
  SWIFT_VERSION="${SWIFT_VER}" build \
  2>&1 | grep -E 'error:|warning:|BUILD (SUCCEEDED|FAILED)' || true
