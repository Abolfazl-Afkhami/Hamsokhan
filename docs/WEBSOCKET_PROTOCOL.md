# Hamsokhan: Real-Time Full-Duplex WebSocket Protocol & Barge-in Specification

> **Protocol Version:** 2.0 (High-Frequency Binary Framing & Dual-Track Barge-in)  
> **Standard:** WebSockets over TLS (RFC 6455 / WSS)  
> **Audio Encodings:** Raw Linear PCM, 16-bit Signed Integer, Little-Endian (`int16_le`)

---

## 1. Architectural Transport Design

Hamsokhan establishes a persistent, bidirectional, full-duplex WebSocket session between the client (Web Browser, iOS/Android Client, or Telephony SIP Gateway) and the inference backend. To maximize network throughput and minimize serialization latency, the transport layer is strictly partitioned into two parallel streams across the single connection:

1. **High-Frequency Binary Audio Stream:** Raw audio buffers transmitted as uncompressed binary chunks (`ArrayBuffer` / `bytes`). No Base64, JSON wrapper, or container metadata overhead.
2. **Asynchronous JSON Control Channel:** Out-of-band text frames for session configuration, interrupt signaling (barge-in), state synchronization, and performance telemetry.

```mermaid
sequenceDiagram
    autonumber
    actor User as Client / User Device
    participant WS as WebSocket Gateway
    participant VAD as Silero VAD v5 (CPU)
    participant Engine as Hamsokhan V2 Model Engine
    participant Cache as KV-Cache & Session Manager

    Note over User,WS: Full-Duplex Connection Established
    User->>WS: Stream Binary Audio (16 kHz PCM, 80 ms chunks)
    WS->>VAD: Forward Causal Frames
    VAD-->>Engine: Speech Onset Verified (P > 0.6)
    Engine->>Engine: Autoregressive Latent Generation & Flow Synthesis
    Engine->>WS: Emit Synthesized Audio (24 kHz PCM, 80 ms chunks)
    WS->>User: Playback Audio Stream

    Note over User,Engine: User Interrupts Assistant Mid-Sentence (Barge-in)
    User->>WS: User Speaks Suddenly During Playback
    VAD->>WS: Acoustic Fast-Path Signal (< 10 ms)
    User->>WS: JSON Control Frame (event: "interrupt", playback_sample: N)
    WS->>WS: Immediately Drain Outgoing Audio Queue (Flush)
    WS->>Cache: Truncate KV-Cache to Timestep of Playback Sample N
    WS->>User: JSON Acknowledgment (event: "interrupted")
    Note over User,WS: Assistant Silent; Listening for Next Utterance
```

---

## 2. Binary Audio Format Specifications

To enforce zero-copy network handling, audio buffers are formatted with exact sample alignment matching the native rates of the acoustic encoder and neural vocoder:

### 2.1. Client-to-Server Ingress Stream (User Voice)
* **Sampling Rate:** $16,000 \text{ Hz}$ (16 kHz)
* **Sample Format:** 16-bit Signed Integer, Little-Endian (`int16_le`)
* **Channels:** 1 (Mono)
* **Chunk Duration:** $80 \text{ ms}$ ($0.080 \text{ seconds}$)
* **Sample Count per Frame:**
  $$N_{\text{in}} = 16,000 \text{ samples/sec} \times 0.080 \text{ sec} = 1,280 \text{ samples}$$
* **Payload Size:**
  $$\text{Bytes}_{\text{in}} = 1,280 \text{ samples} \times 2 \text{ bytes/sample} = 2,560 \text{ bytes}$$

### 2.2. Server-to-Client Egress Stream (Synthesized Assistant Voice)
* **Sampling Rate:** $24,000 \text{ Hz}$ (24 kHz Vocos native output)
* **Sample Format:** 16-bit Signed Integer, Little-Endian (`int16_le`)
* **Channels:** 1 (Mono)
* **Chunk Duration:** $80 \text{ ms}$ ($0.080 \text{ seconds}$)
* **Sample Count per Frame:**
  $$N_{\text{out}} = 24,000 \text{ samples/sec} \times 0.080 \text{ sec} = 1,920 \text{ samples}$$
* **Payload Size:**
  $$\text{Bytes}_{\text{out}} = 1,920 \text{ samples} \times 2 \text{ bytes/sample} = 3,840 \text{ bytes}$$

---

## 3. JSON Control Frame Protocols

All control frames are transmitted as standard JSON text packets.

### 3.1. Client Barge-in Signal (`interrupt`)
Dispatched by the client immediately upon detecting local user speech during assistant playback:

```json
{
  "type": "control",
  "event": "interrupt",
  "timestamp_ms": 1714567890123,
  "client_playback_sample": 38400
}
```
* **`type`:** Constant `"control"`.
* **`event`:** Constant `"interrupt"`.
* **`timestamp_ms`:** Client Unix epoch timestamp in milliseconds.
* **`client_playback_sample`:** The absolute index of the last 24 kHz audio sample actually rendered by the user's audio driver. This value is critical for conversational context synchronization.

### 3.2. Server Interrupt Acknowledgment (`interrupted`)
Emitted by the server immediately after draining its outgoing queue and halting inference:

```json
{
  "type": "control",
  "event": "interrupted",
  "cleared_samples": 57600,
  "session_state": "listening"
}
```
* **`cleared_samples`:** The number of queued audio samples that were dropped from the network buffer.
* **`session_state`:** Confirmation that the backend has transitioned back to `"listening"`.

### 3.3. Real-Time Telemetry & Performance Metrics (`telemetry`)
Dispatched periodically or at the conclusion of each dialogue turn:

```json
{
  "type": "telemetry",
  "ttfa_ms": 218,
  "turn_taking_prob": 0.92,
  "vram_allocated_mb": 2480,
  "audio_rtf": 0.14
}
```
* **`ttfa_ms`:** Time-to-First-Audio latency in milliseconds for the current turn.
* **`turn_taking_prob`:** Model confidence score indicating whether the user completed their turn.
* **`vram_allocated_mb`:** Active PyTorch CUDA memory footprint on the inference GPU.
* **`audio_rtf`:** Real-Time Factor ($\frac{\text{Synthesis Time}}{\text{Audio Duration}}$). Values $< 1.0$ indicate real-time operation.

---

## 4. Dual-Track Barge-in & State Truncation Architecture

Natural conversation requires immediate cessation of assistant speech when the user begins speaking, without corrupting the historical context of the dialogue.

```
                          Incoming Audio Frame (80 ms)
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
   [Acoustic Fast-Path]                                [Semantic Confirmation]
  • Silero VAD v5 on Host CPU                         • Turn-Taking Head on GPU
  • Decision Window: < 10 ms                          • Decision Window: 80 ms – 120 ms
  • 2 frames with P > 0.6                             • Latent semantic classification
  • Instantly drains egress buffer                    • Differentiates true barge-in from
    and silences audio output                           brief backchannel affirmations ("Uh-huh")
```

### KV-Cache State Truncation Mechanism
In typical naive implementations, an assistant generates a 10-second response in advance. If the user interrupts after only 2 seconds, the LLM's historical context retains the entire unuttered 10-second monologue, leading to incoherent conversational drift in subsequent turns.

In Hamsokhan, conversational alignment is maintained deterministically:
1. When the client sends an `"interrupt"` frame with `client_playback_sample` $S$, the server calculates the absolute playback duration:
   $$T_{\text{played}} = \frac{S}{24,000} \text{ seconds}$$
2. The playback duration is mapped to the exact number of language tokens $K_{\text{heard}}$ generated up to $T_{\text{played}}$:
   $$K_{\text{heard}} = \left\lfloor T_{\text{played}} \times \text{Framerate}_{\text{tokens}} \right\rfloor$$
3. The server slices the autoregressive Key-Value Cache (KV-Cache) across all transformer layers:
   $$\mathbf{K}_{\text{layer}} = \mathbf{K}_{\text{layer}}[:, :, :K_{\text{heard}}, :], \quad \mathbf{V}_{\text{layer}} = \mathbf{V}_{\text{layer}}[:, :, :K_{\text{heard}}, :]$$
4. Unspoken tokens are purged, aligning the assistant’s context with the user's actual auditory experience.

---

## 5. Packet Boundary Continuity (Overlap-Add Synthesis)

Streaming chunked audio synthesis through neural vocoders can introduce subtle phase discontinuities and audible clicking artifacts at the 80 ms packet boundaries. 

Hamsokhan utilizes an Overlap-Add (OLA) crossfade strategy:
* Each 80 ms chunk is synthesized with a 15 ms leading context overlap.
* A raised-cosine Hanning window is applied across the overlap region:
  $$w[n] = 0.5 \left( 1 - \cos\left( \frac{2\pi n}{M - 1} \right) \right)$$
* Consecutive chunks are linearly blended, ensuring smooth phase transitions across all frequencies and eliminating high-frequency boundary pops.
