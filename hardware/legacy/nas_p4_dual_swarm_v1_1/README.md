# JANUS NAS archaeology · ESP32-P4 dual-swarm firmware v1.1

**Source:** owner QNAP `Janus/All Janus/G/JANUS_P4_DUAL_SWARM_CORE_v1_1/` recovered 2026-10-31.

Historical **reference-only**, not flashed to any physical device and not a cloud/Keymaster compute worker. The original firmware `JANUS_P4_DUAL_SWARM_CORE_v1_1.ino` has owner NAS SHA-256:

`b33b54d6e294a27a047c980fa81f5933ae2ae771260411427d9ac6c172dfd636`

GitHub text may have normalized trailing newline; SHA refers to exact NAS bytes.

## Roles and transport

- `P4_A`: hashing, scouting, intention score.
- `P4_B`: peer frame mirror, verification and memory.
- `P4_C`: reserved coordinator/relay.
- Frames: `JP4,1,node,role,mode,seq,nonce,salt,job,z,digest,hps*CSUM` over Serial/UART.
- ESP32-P4 has **no native Wi-Fi or Bluetooth**; companion ESP32-C6/S3 needed for ESP-NOW/Wi-Fi transport. This historic sketch is *not* a replacement for currently-owned S3 devices.
- Optional UART pins and electrical wiring are board-specific: do not assume historical defaults are safe to connect.

## Integration gates

This code is an **old research prototype**, not independently build-tested on a pinned P4 board/toolchain, safety-reviewed for external hosts, or authorized to mine on pooled services. Do not deploy it through GitHub Actions, start continuous hashing on QNAP, use it as a 24/7 server or connect an internet-facing job broker.

Potential safe reuse is extracting a device-independent frame parser, with bounded message size, CRC/integrity checks and a synthetic two-node replay test. No peer identity/authentication or cryptographic attestation is established by the simple historic FNV-1a checksum.

No credentials, private database rows or firmware binaries were imported. The original on NAS is unchanged. Static CI validates structure only and does **not** flash or execute the firmware.
