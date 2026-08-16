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
# Exit with xcodebuild's status, not grep's.
#
# This previously ended in `|| true`, which made the script exit 0 even when the
# build failed — it printed "** BUILD FAILED **" and then reported success, so
# any caller or CI step checking the exit code saw a passing verification. That
# defeats the entire purpose of a script named verify-build.
#
# Simply deleting `|| true` is NOT the fix: grep exits 1 when it matches
# nothing, so a clean build producing no error/warning lines would then report a
# false failure. The status has to come specifically from xcodebuild, which is
# what PIPESTATUS[0] captures — read immediately after the pipeline, before any
# other command can overwrite it.
set +e
xcodebuild -scheme Glasses3DRelayKit \
  -destination 'generic/platform=iOS' \
  -derivedDataPath ".build-${SWIFT_VER}" \
  SWIFT_VERSION="${SWIFT_VER}" build \
  2>&1 | grep -E 'error:|warning:|BUILD (SUCCEEDED|FAILED)'
build_status=${PIPESTATUS[0]}
set -e

if [ "$build_status" -ne 0 ]; then
  echo "==> relay build FAILED (xcodebuild exit ${build_status})" >&2
  exit "$build_status"
fi
echo "==> relay build OK"
