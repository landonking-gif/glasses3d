//
// Glasses3DRelay.swift
//
// Ships Meta Ray-Ban camera frames to the glasses3d reconstruction server.
//
// Drop this into the VisionClaw CameraAccess target (it uses the same MWDATCore /
// MWDATCamera imports and the same StreamSession patterns as
// StreamSessionViewModel.swift), then:
//
//     let relay = Glasses3DRelay(wearables: wearables)
//     relay.serverURL = URL(string: "ws://192.168.1.42:8765")!
//     await relay.start()
//
// Or, if you would rather not run a second StreamSession, delete the session
// management here and call `relay.send(image:)` from the existing
// `videoFramePublisher` listener alongside `webrtcSessionVM?.pushVideoFrame`.
//
// Wire format must match server/protocol.py:
//     [4-byte big-endian header length][UTF-8 JSON header][JPEG payload]
//

import Foundation
import CoreMedia
import CoreVideo
import MWDATCamera
import MWDATCore
import UIKit

// MARK: - Wire format

private struct FrameHeader: Encodable {
  let seq: Int
  let ts_capture: Double
  let ts_send: Double
  let width: Int
  let height: Int
  let fmt: String
  let session: String
}

@MainActor
final class Glasses3DRelay: ObservableObject {

  // MARK: Published state

  @Published private(set) var isConnected = false
  @Published private(set) var isStreaming = false
  @Published private(set) var framesSent = 0
  @Published private(set) var framesDropped = 0
  @Published private(set) var lastError: String?

  // MARK: Configuration

  var serverURL = URL(string: "ws://127.0.0.1:8765")!
  var jpegQuality: CGFloat = 0.7

  /// Full-resolution stills are roughly 9x the pixels of the 720p stream. Firing
  /// one periodically and injecting it as a high-resolution observation recovers
  /// much of the detail the stream ceiling costs the reconstruction.
  var photoEveryNFrames = 0   // 0 disables

  /// Cap on frames in flight. Without this, a slow link builds an unbounded
  /// backlog: memory climbs and every frame the server receives describes where
  /// you were looking seconds ago, which corrupts pose association far more
  /// insidiously than a dropped frame does.
  var maxInFlight = 3

  // MARK: Private

  private let wearables: WearablesInterface
  private let deviceSelector: AutoDeviceSelector
  private var streamSession: StreamSession
  private var videoToken: AnyListenerToken?
  private var stateToken: AnyListenerToken?
  private var errorToken: AnyListenerToken?
  private var photoToken: AnyListenerToken?

  private var socket: URLSessionWebSocketTask?
  private lazy var urlSession = URLSession(configuration: .default)

  private var seq = 0
  private var inFlight = 0
  private let sessionID = UUID().uuidString.prefix(8).lowercased()
  private let encodeQueue = DispatchQueue(label: "glasses3d.encode", qos: .userInitiated)

  init(wearables: WearablesInterface, resolution: StreamingResolution = .high) {
    self.wearables = wearables
    self.deviceSelector = AutoDeviceSelector(wearables: wearables)
    // .high is 720x1280 — portrait. The server's MockSource renders portrait
    // for the same reason: it is easy to forget until geometry comes out rotated.
    let config = StreamSessionConfig(
      videoCodec: VideoCodec.raw,
      resolution: resolution,
      frameRate: 24)
    self.streamSession = StreamSession(streamSessionConfig: config,
                                       deviceSelector: deviceSelector)
    attachListeners()
  }

  // MARK: - Lifecycle

  func start() async {
    do {
      var status = try await wearables.checkPermissionStatus(.camera)
      if status != .granted {
        status = try await wearables.requestPermission(.camera)
      }
      guard status == .granted else {
        lastError = "Camera permission denied"
        return
      }
    } catch {
      lastError = "Permission error: \(error)"
      return
    }
    connect()
    await streamSession.start()
    isStreaming = true
  }

  func stop() async {
    await streamSession.stop()
    isStreaming = false
    socket?.cancel(with: .goingAway, reason: nil)
    socket = nil
    isConnected = false
  }

  // MARK: - Transport

  private func connect() {
    socket?.cancel(with: .goingAway, reason: nil)
    let task = urlSession.webSocketTask(with: serverURL)
    task.resume()
    socket = task
    isConnected = true
    receiveLoop()   // keeps the socket alive; the server may send control text
  }

  private func receiveLoop() {
    socket?.receive { [weak self] result in
      Task { @MainActor [weak self] in
        guard let self else { return }
        switch result {
        case .success:
          self.receiveLoop()
        case .failure(let error):
          self.isConnected = false
          self.lastError = "Socket closed: \(error.localizedDescription)"
        }
      }
    }
  }

  // MARK: - Frame path

  private func attachListeners() {
    stateToken = streamSession.statePublisher.listen { [weak self] state in
      Task { @MainActor [weak self] in
        if case .stopped = state { self?.isStreaming = false }
      }
    }

    errorToken = streamSession.errorPublisher.listen { [weak self] error in
      Task { @MainActor [weak self] in
        // .hingesClosed means the glasses were folded — worth surfacing plainly
        // rather than as a generic streaming failure.
        self?.lastError = String(describing: error)
      }
    }

    videoToken = streamSession.videoFramePublisher.listen { [weak self] videoFrame in
      Task { @MainActor [weak self] in
        guard let self, self.isConnected else { return }
        let captureTime = Date().timeIntervalSince1970

        // makeUIImage() goes through VideoToolbox GPU rendering, which iOS
        // suspends in the background. VisionClaw's StreamSessionViewModel solves
        // this with a VTDecompressionSession fallback; reuse that path if you
        // need reconstruction to continue with the screen locked.
        guard let image = videoFrame.makeUIImage() else { return }
        self.send(image: image, captureTime: captureTime)

        if self.photoEveryNFrames > 0, self.seq % self.photoEveryNFrames == 0 {
          self.streamSession.capturePhoto(format: .jpeg)
        }
      }
    }

    photoToken = streamSession.photoDataPublisher.listen { [weak self] photoData in
      Task { @MainActor [weak self] in
        guard let self, let image = UIImage(data: photoData.data) else { return }
        // Tagged "jpeg-hires" so the server can route it as a high-resolution
        // keyframe observation rather than a normal stream frame.
        self.send(image: image, captureTime: Date().timeIntervalSince1970,
                  format: "jpeg-hires")
      }
    }
  }

  func send(image: UIImage, captureTime: Double, format: String = "jpeg") {
    guard let socket, isConnected else { return }

    // Backpressure. Dropping the newest frame is correct here: the server wants
    // the freshest possible view, and a queued backlog would make every
    // subsequent frame later still.
    guard inFlight < maxInFlight else {
      framesDropped += 1
      return
    }

    let n = seq
    seq += 1
    inFlight += 1

    let quality = jpegQuality
    let session = String(sessionID)

    encodeQueue.async { [weak self] in
      guard let payload = image.jpegData(compressionQuality: quality) else {
        Task { @MainActor [weak self] in self?.inFlight -= 1 }
        return
      }
      let header = FrameHeader(
        seq: n,
        ts_capture: captureTime,
        ts_send: Date().timeIntervalSince1970,
        width: Int(image.size.width * image.scale),
        height: Int(image.size.height * image.scale),
        fmt: format,
        session: session)

      guard let headerData = try? JSONEncoder().encode(header) else {
        Task { @MainActor [weak self] in self?.inFlight -= 1 }
        return
      }

      var message = Data()
      var length = UInt32(headerData.count).bigEndian
      withUnsafeBytes(of: &length) { message.append(contentsOf: $0) }
      message.append(headerData)
      message.append(payload)

      socket.send(.data(message)) { [weak self] error in
        Task { @MainActor [weak self] in
          guard let self else { return }
          self.inFlight -= 1
          if let error {
            self.lastError = "Send failed: \(error.localizedDescription)"
            self.isConnected = false
          } else {
            self.framesSent += 1
          }
        }
      }
    }
  }
}
