/*
  JANUS P4 DUAL SWARM CORE v1.1
  Target: ESP32-P4 Arduino / PlatformIO Arduino

  This is the P4 sketch for the original Janus P4 idea:
  - P4_A: HASH / SCOUT / INTENTION compute node
  - P4_B: MIRROR / VERIFY / MEMORY compute node
  - companion ESP32-C6/S3: Wi-Fi / ESP-NOW / pool bridge

  ESP32-P4 does not have native Wi-Fi/Bluetooth.
  This sketch therefore does not include WiFi.h or esp_now.h.
  Network transport is represented by JP4 bridge frames over Serial/UART.

  Flash the same sketch to both P4 boards.
  Change JANUS_DEFAULT_ROLE to "P4_A" or "P4_B" before flashing,
  or send command: role A / role B
*/

#include <Arduino.h>
#include <mbedtls/sha256.h>
#include "esp_chip_info.h"
#include "esp_heap_caps.h"
#include "esp_mac.h"
#include "esp_system.h"
#include "esp_timer.h"

// ============================================================
// USER CONFIG
// ============================================================

#define JANUS_P4_VERSION             "JANUS_P4_DUAL_SWARM_CORE_v1.1"

// Change this before flashing the second P4:
//   "P4_A" = hash/scout/intention
//   "P4_B" = mirror/verify/memory
//   "P4_C" = coordinator/relay
#define JANUS_DEFAULT_ROLE           "P4_A"

// Set to 1 after wiring a companion ESP32-C6/S3/S2 by UART.
// USB Serial always works as command/telemetry bridge.
#define JANUS_USE_UART_BRIDGE        0
#define JANUS_UART_RX_PIN            44
#define JANUS_UART_TX_PIN            43
#define JANUS_UART_BAUD              115200

// Optional heartbeat LED. Set to -1 if your P4 board has no safe LED pin.
#define JANUS_HEARTBEAT_LED_PIN      -1
#define JANUS_HEARTBEAT_ACTIVE       HIGH

// Hash and telemetry tuning.
#define JANUS_HASH_BATCH_FAST        768
#define JANUS_HASH_BATCH_MIRROR      256
#define JANUS_WORKER_STACK           9216
#define JANUS_TELEMETRY_MS           1000
#define JANUS_FRAME_MS               500
#define JANUS_PEER_TIMEOUT_MS        5000
#define JANUS_JOB_TIMEOUT_MS         30000
#define JANUS_PROOF_Z_THRESHOLD      24
#define JANUS_ANCHOR_RING            24
#define JANUS_VERIFY_QUEUE           8

// Keep at 0 for broad Arduino compatibility.
#define JANUS_PIN_WORKER_TO_CORE     0
#define JANUS_WORKER_CORE            1

// ============================================================
// OPTIONAL UART BRIDGE
// ============================================================

#if JANUS_USE_UART_BRIDGE
HardwareSerial JanusBridge(1);
#endif

// ============================================================
// TYPES
// ============================================================

enum JanusRole : uint8_t {
  ROLE_P4_A = 0,
  ROLE_P4_B = 1,
  ROLE_P4_C = 2
};

enum JanusMode : uint8_t {
  MODE_AUTO = 0,
  MODE_HASH = 1,
  MODE_SCOUT = 2,
  MODE_MIRROR = 3,
  MODE_VERIFY = 4,
  MODE_RELAY = 5,
  MODE_IDLE = 6
};

struct JanusHashInput {
  uint32_t magic;
  uint32_t schema;
  uint32_t job_seed;
  uint32_t salt;
  uint32_t seq;
  uint32_t nonce;
  uint8_t  node_mac[6];
  char     role_tag[8];
  uint8_t  reserved[18];
};

struct JanusAnchor {
  bool     valid;
  bool     local;
  bool     verified;
  char     node_id[13];
  char     role[8];
  uint32_t z;
  uint32_t seq;
  uint32_t nonce;
  uint32_t salt;
  uint32_t job_seed;
  uint8_t  digest[32];
  uint32_t seen_ms;
};

struct JanusCandidate {
  bool     valid;
  char     node_id[13];
  char     role[8];
  uint32_t z;
  uint32_t seq;
  uint32_t nonce;
  uint32_t salt;
  uint32_t job_seed;
  uint8_t  digest[32];
  uint32_t seen_ms;
};

struct JanusStats {
  uint64_t total_hashes;
  uint64_t last_total_hashes;
  uint32_t rounds;

  uint32_t best_z;
  uint32_t best_seq;
  uint32_t best_nonce;
  uint32_t best_salt;
  uint32_t best_job_seed;
  uint8_t  best_digest[32];

  uint32_t local_proofs;
  uint32_t peer_frames_ok;
  uint32_t peer_frames_bad_csum;
  uint32_t peer_frames_drop;
  uint32_t verify_ok;
  uint32_t verify_fail;
  uint32_t mirror_hits;
  uint32_t bridge_tx;
  uint32_t cmd_count;
  uint32_t bad_cmd_count;
};

struct JanusPeer {
  bool     seen;
  char     node_id[13];
  char     role[8];
  char     mode[10];
  uint32_t seq;
  uint32_t nonce;
  uint32_t salt;
  uint32_t job_seed;
  uint32_t z;
  float    hps;
  uint8_t  digest[32];
  uint32_t last_ms;
};

struct JanusJob {
  uint32_t seed;
  uint32_t target_z;
  uint32_t started_ms;
  uint32_t ttl_ms;
  char     job_id[17];
};

// ============================================================
// GLOBAL STATE
// ============================================================

static portMUX_TYPE gMux = portMUX_INITIALIZER_UNLOCKED;

static JanusRole gRole = ROLE_P4_A;
static JanusMode gMode = MODE_AUTO;
static JanusStats gStats;
static JanusPeer gPeer;
static JanusJob gJob;

static JanusAnchor gAnchors[JANUS_ANCHOR_RING];
static uint8_t gAnchorHead = 0;

static JanusCandidate gVerifyQueue[JANUS_VERIFY_QUEUE];
static uint8_t gVerifyHead = 0;

static uint8_t gBaseMac[6];
static char gNodeId[13];

static volatile bool gWorkerRunning = false;
static volatile bool gResetBest = false;
static volatile uint32_t gExtraSalt = 0;

static float gLastHps = 0.0f;
static float gIntention = 0.0f;
static float gBlackboard = 0.0f;

static char usbLine[256];
static size_t usbLineLen = 0;

#if JANUS_USE_UART_BRIDGE
static char bridgeLine[256];
static size_t bridgeLineLen = 0;
#endif

// ============================================================
// TEXT HELPERS
// ============================================================

static const char *roleName(JanusRole role) {
  switch (role) {
    case ROLE_P4_A: return "P4_A";
    case ROLE_P4_B: return "P4_B";
    case ROLE_P4_C: return "P4_C";
    default: return "P4_X";
  }
}

static const char *modeName(JanusMode mode) {
  switch (mode) {
    case MODE_AUTO: return "AUTO";
    case MODE_HASH: return "HASH";
    case MODE_SCOUT: return "SCOUT";
    case MODE_MIRROR: return "MIRROR";
    case MODE_VERIFY: return "VERIFY";
    case MODE_RELAY: return "RELAY";
    case MODE_IDLE: return "IDLE";
    default: return "UNK";
  }
}

static JanusRole parseRoleName(const char *s) {
  if (!strcasecmp(s, "A") || !strcasecmp(s, "P4_A")) return ROLE_P4_A;
  if (!strcasecmp(s, "B") || !strcasecmp(s, "P4_B")) return ROLE_P4_B;
  if (!strcasecmp(s, "C") || !strcasecmp(s, "P4_C")) return ROLE_P4_C;
  return gRole;
}

static JanusMode parseModeName(const char *s) {
  if (!strcasecmp(s, "AUTO")) return MODE_AUTO;
  if (!strcasecmp(s, "HASH")) return MODE_HASH;
  if (!strcasecmp(s, "SCOUT")) return MODE_SCOUT;
  if (!strcasecmp(s, "MIRROR")) return MODE_MIRROR;
  if (!strcasecmp(s, "VERIFY")) return MODE_VERIFY;
  if (!strcasecmp(s, "RELAY")) return MODE_RELAY;
  if (!strcasecmp(s, "IDLE")) return MODE_IDLE;
  return gMode;
}

static void printU64(Stream &out, uint64_t v) {
  char b[32];
  snprintf(b, sizeof(b), "%llu", (unsigned long long)v);
  out.print(b);
}

static uint32_t fnv1a32Bytes(const uint8_t *data, size_t len) {
  uint32_t h = 2166136261UL;
  for (size_t i = 0; i < len; i++) {
    h ^= data[i];
    h *= 16777619UL;
  }
  return h;
}

static uint32_t fnv1a32Text(const char *s) {
  return fnv1a32Bytes((const uint8_t *)s, strlen(s));
}

static uint32_t xorshift32(uint32_t x) {
  if (x == 0) x = 0xA5A5F00DUL;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

static bool parseHex32(const char *s, uint32_t *out) {
  while (*s == ' ' || *s == '\t') s++;
  if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) s += 2;

  uint32_t v = 0;
  int n = 0;
  while (*s) {
    char c = *s++;
    uint8_t x;
    if (c >= '0' && c <= '9') x = c - '0';
    else if (c >= 'a' && c <= 'f') x = c - 'a' + 10;
    else if (c >= 'A' && c <= 'F') x = c - 'A' + 10;
    else break;
    v = (v << 4) | x;
    n++;
    if (n >= 8) break;
  }

  if (n == 0) return false;
  *out = v;
  return true;
}

static bool parseU32(const char *s, uint32_t *out) {
  if (!s || !*s) return false;
  char *endp = nullptr;
  unsigned long v = strtoul(s, &endp, 10);
  if (endp == s) return false;
  *out = (uint32_t)v;
  return true;
}

static void digestToHex(const uint8_t digest[32], char out[65]) {
  static const char *hex = "0123456789abcdef";
  for (int i = 0; i < 32; i++) {
    out[i * 2] = hex[(digest[i] >> 4) & 0x0F];
    out[i * 2 + 1] = hex[digest[i] & 0x0F];
  }
  out[64] = '\0';
}

static bool hexToDigest(const char *hex, uint8_t digest[32]) {
  if (!hex || strlen(hex) < 64) return false;
  for (int i = 0; i < 32; i++) {
    char a = hex[i * 2];
    char b = hex[i * 2 + 1];
    uint8_t hi, lo;
    if (a >= '0' && a <= '9') hi = a - '0';
    else if (a >= 'a' && a <= 'f') hi = a - 'a' + 10;
    else if (a >= 'A' && a <= 'F') hi = a - 'A' + 10;
    else return false;

    if (b >= '0' && b <= '9') lo = b - '0';
    else if (b >= 'a' && b <= 'f') lo = b - 'a' + 10;
    else if (b >= 'A' && b <= 'F') lo = b - 'A' + 10;
    else return false;

    digest[i] = (hi << 4) | lo;
  }
  return true;
}

static bool nodeIdToMac(const char *id, uint8_t mac[6]) {
  if (!id || strlen(id) < 12) return false;
  char tmp[3] = {0, 0, 0};
  for (int i = 0; i < 6; i++) {
    tmp[0] = id[i * 2];
    tmp[1] = id[i * 2 + 1];
    uint32_t v = 0;
    if (!parseHex32(tmp, &v)) return false;
    mac[i] = (uint8_t)v;
  }
  return true;
}

static uint8_t leadingZeroBits(const uint8_t digest[32]) {
  uint8_t z = 0;
  for (int i = 0; i < 32; i++) {
    uint8_t b = digest[i];
    if (b == 0) {
      z += 8;
      continue;
    }
    for (int bit = 7; bit >= 0; bit--) {
      if (b & (1U << bit)) return z;
      z++;
    }
  }
  return z;
}

static void sha256Bytes(const void *data, size_t len, uint8_t out[32]) {
  mbedtls_sha256((const unsigned char *)data, len, out, 0);
}

// ============================================================
// HASH MODEL
// ============================================================

static void buildHashInput(
  JanusHashInput *input,
  const uint8_t mac[6],
  const char *role,
  uint32_t jobSeed,
  uint32_t salt,
  uint32_t seq,
  uint32_t nonce
) {
  memset(input, 0, sizeof(*input));
  input->magic = 0x4A503443UL; // JP4C
  input->schema = 0x00010001UL;
  input->job_seed = jobSeed;
  input->salt = salt;
  input->seq = seq;
  input->nonce = nonce;
  memcpy(input->node_mac, mac, 6);
  strncpy(input->role_tag, role, sizeof(input->role_tag) - 1);
}

static uint8_t hashCandidate(
  const uint8_t mac[6],
  const char *role,
  uint32_t jobSeed,
  uint32_t salt,
  uint32_t seq,
  uint32_t nonce,
  uint8_t digest[32]
) {
  JanusHashInput input;
  buildHashInput(&input, mac, role, jobSeed, salt, seq, nonce);
  sha256Bytes(&input, sizeof(input), digest);
  return leadingZeroBits(digest);
}

// ============================================================
// MEMORY / BLACKBOARD
// ============================================================

static void rememberAnchor(
  bool local,
  bool verified,
  const char *nodeId,
  const char *role,
  uint32_t z,
  uint32_t seq,
  uint32_t nonce,
  uint32_t salt,
  uint32_t jobSeed,
  const uint8_t digest[32]
) {
  portENTER_CRITICAL(&gMux);
  JanusAnchor *a = &gAnchors[gAnchorHead];
  gAnchorHead = (gAnchorHead + 1) % JANUS_ANCHOR_RING;
  memset(a, 0, sizeof(*a));
  a->valid = true;
  a->local = local;
  a->verified = verified;
  strncpy(a->node_id, nodeId, sizeof(a->node_id) - 1);
  strncpy(a->role, role, sizeof(a->role) - 1);
  a->z = z;
  a->seq = seq;
  a->nonce = nonce;
  a->salt = salt;
  a->job_seed = jobSeed;
  memcpy(a->digest, digest, 32);
  a->seen_ms = millis();
  gBlackboard = (gBlackboard * 0.85f) + ((float)z * (verified ? 0.20f : 0.12f));
  portEXIT_CRITICAL(&gMux);
}

static void queueVerify(const JanusCandidate *cand) {
  portENTER_CRITICAL(&gMux);
  gVerifyQueue[gVerifyHead] = *cand;
  gVerifyQueue[gVerifyHead].valid = true;
  gVerifyHead = (gVerifyHead + 1) % JANUS_VERIFY_QUEUE;
  portEXIT_CRITICAL(&gMux);
}

static bool popVerify(JanusCandidate *cand) {
  portENTER_CRITICAL(&gMux);
  for (uint8_t i = 0; i < JANUS_VERIFY_QUEUE; i++) {
    if (gVerifyQueue[i].valid) {
      *cand = gVerifyQueue[i];
      gVerifyQueue[i].valid = false;
      portEXIT_CRITICAL(&gMux);
      return true;
    }
  }
  portEXIT_CRITICAL(&gMux);
  return false;
}

static void resetBestOnly() {
  portENTER_CRITICAL(&gMux);
  gStats.best_z = 0;
  gStats.best_seq = 0;
  gStats.best_nonce = 0;
  gStats.best_salt = 0;
  gStats.best_job_seed = gJob.seed;
  memset(gStats.best_digest, 0, sizeof(gStats.best_digest));
  portEXIT_CRITICAL(&gMux);
}

// ============================================================
// JP4 FRAME PROTOCOL
// ============================================================

// Compact peer frame:
// JP4,1,node,role,mode,seq,nonce,salt,job,z,digest,hps*CSUM
// CSUM = FNV1a32 over bytes before '*'.

static void emitJP4Frame(Stream &out) {
  JanusStats s;
  JanusJob job;

  portENTER_CRITICAL(&gMux);
  memcpy(&s, &gStats, sizeof(s));
  memcpy(&job, &gJob, sizeof(job));
  portEXIT_CRITICAL(&gMux);

  char digestHex[65];
  digestToHex(s.best_digest, digestHex);

  char body[256];
  snprintf(
    body,
    sizeof(body),
    "JP4,1,%s,%s,%s,%lu,%lu,%08lx,%08lx,%lu,%s,%.1f",
    gNodeId,
    roleName(gRole),
    modeName(gMode),
    (unsigned long)s.best_seq,
    (unsigned long)s.best_nonce,
    (unsigned long)s.best_salt,
    (unsigned long)job.seed,
    (unsigned long)s.best_z,
    digestHex,
    gLastHps
  );

  uint32_t csum = fnv1a32Text(body);
  out.print(body);
  out.print("*");
  out.println(csum, HEX);

  portENTER_CRITICAL(&gMux);
  gStats.bridge_tx++;
  portEXIT_CRITICAL(&gMux);
}

static bool parseJP4Frame(char *line) {
  char *star = strchr(line, '*');
  if (!star) {
    portENTER_CRITICAL(&gMux);
    gStats.peer_frames_drop++;
    portEXIT_CRITICAL(&gMux);
    return false;
  }

  *star = '\0';
  uint32_t got = 0;
  if (!parseHex32(star + 1, &got)) {
    portENTER_CRITICAL(&gMux);
    gStats.peer_frames_drop++;
    portEXIT_CRITICAL(&gMux);
    return false;
  }

  uint32_t want = fnv1a32Text(line);
  if (got != want) {
    portENTER_CRITICAL(&gMux);
    gStats.peer_frames_bad_csum++;
    portEXIT_CRITICAL(&gMux);
    return false;
  }

  char *fields[12];
  uint8_t count = 0;
  char *tok = strtok(line, ",");
  while (tok && count < 12) {
    fields[count++] = tok;
    tok = strtok(nullptr, ",");
  }

  if (count != 12 || strcmp(fields[0], "JP4") != 0) {
    portENTER_CRITICAL(&gMux);
    gStats.peer_frames_drop++;
    portEXIT_CRITICAL(&gMux);
    return false;
  }

  uint32_t seq = 0, nonce = 0, salt = 0, job = 0, z = 0;
  if (!parseU32(fields[5], &seq)) return false;
  if (!parseU32(fields[6], &nonce)) return false;
  if (!parseHex32(fields[7], &salt)) return false;
  if (!parseHex32(fields[8], &job)) return false;
  if (!parseU32(fields[9], &z)) return false;

  JanusCandidate cand;
  memset(&cand, 0, sizeof(cand));
  cand.valid = true;
  strncpy(cand.node_id, fields[2], sizeof(cand.node_id) - 1);
  strncpy(cand.role, fields[3], sizeof(cand.role) - 1);
  cand.node_id[sizeof(cand.node_id) - 1] = '\0';
  cand.role[sizeof(cand.role) - 1] = '\0';
  cand.seq = seq;
  cand.nonce = nonce;
  cand.salt = salt;
  cand.job_seed = job;
  cand.z = z;
  cand.seen_ms = millis();
  if (!hexToDigest(fields[10], cand.digest)) {
    portENTER_CRITICAL(&gMux);
    gStats.peer_frames_drop++;
    portEXIT_CRITICAL(&gMux);
    return false;
  }

  portENTER_CRITICAL(&gMux);
  gPeer.seen = true;
  strncpy(gPeer.node_id, fields[2], sizeof(gPeer.node_id) - 1);
  strncpy(gPeer.role, fields[3], sizeof(gPeer.role) - 1);
  strncpy(gPeer.mode, fields[4], sizeof(gPeer.mode) - 1);
  gPeer.node_id[sizeof(gPeer.node_id) - 1] = '\0';
  gPeer.role[sizeof(gPeer.role) - 1] = '\0';
  gPeer.mode[sizeof(gPeer.mode) - 1] = '\0';
  gPeer.seq = seq;
  gPeer.nonce = nonce;
  gPeer.salt = salt;
  gPeer.job_seed = job;
  gPeer.z = z;
  gPeer.hps = atof(fields[11]);
  memcpy(gPeer.digest, cand.digest, 32);
  gPeer.last_ms = millis();
  gStats.peer_frames_ok++;
  portEXIT_CRITICAL(&gMux);

  if (z >= 12) {
    queueVerify(&cand);
  }

  return true;
}

// ============================================================
// VERIFIER
// ============================================================

static void verifyOneCandidate() {
  JanusCandidate cand;
  if (!popVerify(&cand)) return;

  uint8_t mac[6];
  uint8_t digest[32];
  if (!nodeIdToMac(cand.node_id, mac)) {
    portENTER_CRITICAL(&gMux);
    gStats.verify_fail++;
    portEXIT_CRITICAL(&gMux);
    return;
  }

  uint8_t z = hashCandidate(
    mac,
    cand.role,
    cand.job_seed,
    cand.salt,
    cand.seq,
    cand.nonce,
    digest
  );

  bool ok = (z == cand.z) && (memcmp(digest, cand.digest, 32) == 0);

  portENTER_CRITICAL(&gMux);
  if (ok) gStats.verify_ok++;
  else gStats.verify_fail++;
  if (ok && z >= 16) gStats.mirror_hits++;
  portEXIT_CRITICAL(&gMux);

  if (ok) {
    rememberAnchor(
      false,
      true,
      cand.node_id,
      cand.role,
      cand.z,
      cand.seq,
      cand.nonce,
      cand.salt,
      cand.job_seed,
      cand.digest
    );
  }
}

// ============================================================
// TELEMETRY
// ============================================================

static void readBaseMac() {
  if (esp_read_mac(gBaseMac, ESP_MAC_BASE) != ESP_OK) {
    uint32_t r1 = esp_random();
    uint32_t r2 = esp_random();
    memcpy(gBaseMac, &r1, 4);
    memcpy(gBaseMac + 4, &r2, 2);
  }

  snprintf(
    gNodeId,
    sizeof(gNodeId),
    "%02X%02X%02X%02X%02X%02X",
    gBaseMac[0], gBaseMac[1], gBaseMac[2],
    gBaseMac[3], gBaseMac[4], gBaseMac[5]
  );
}

static void printBootInfo() {
  esp_chip_info_t chip;
  esp_chip_info(&chip);

  Serial.println();
  Serial.println("============================================================");
  Serial.println(JANUS_P4_VERSION);
  Serial.println("Dual P4 Janus core: HASH/SCOUT + MIRROR/VERIFY/MEMORY");
  Serial.println("============================================================");
  Serial.printf("node_id=%s default_role=%s cores=%u revision=%u\n",
                gNodeId, roleName(gRole), chip.cores, chip.revision);
  Serial.printf("free_heap=%u psram=%u\n",
                (unsigned)ESP.getFreeHeap(),
                (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  Serial.println("native_wifi=false bridge_protocol=JP4");
  Serial.println("commands: help | status | role A/B/C | mode AUTO/HASH/MIRROR/VERIFY/RELAY/IDLE");
  Serial.println("commands: job SEEDHEX [TARGET_Z] [JOBID] | seed HEX | reset");
  Serial.println();
}

static void printHelp(Stream &out) {
  out.println("JANUS P4 Dual Swarm Core commands:");
  out.println("  status                         - print JSON status");
  out.println("  role A | role B | role C        - set P4 role");
  out.println("  mode AUTO|HASH|SCOUT|MIRROR|VERIFY|RELAY|IDLE");
  out.println("  job SEEDHEX [TARGET_Z] [JOBID]  - set external/pool job seed");
  out.println("  seed HEX                        - mix extra local salt");
  out.println("  reset                           - reset local best");
  out.println("  JP4,...*CSUM                    - peer frame from companion");
}

static uint32_t peerAgeMs() {
  portENTER_CRITICAL(&gMux);
  bool seen = gPeer.seen;
  uint32_t last = gPeer.last_ms;
  portEXIT_CRITICAL(&gMux);
  if (!seen) return 0xFFFFFFFFUL;
  return millis() - last;
}

static uint32_t jobAgeMs() {
  portENTER_CRITICAL(&gMux);
  uint32_t started = gJob.started_ms;
  portEXIT_CRITICAL(&gMux);
  return millis() - started;
}

static void updateAdaptiveFields() {
  JanusStats s;
  JanusPeer p;
  portENTER_CRITICAL(&gMux);
  memcpy(&s, &gStats, sizeof(s));
  memcpy(&p, &gPeer, sizeof(p));
  portEXIT_CRITICAL(&gMux);

  float link = (p.seen && (millis() - p.last_ms < JANUS_PEER_TIMEOUT_MS)) ? 1.0f : 0.0f;
  float proof = (float)s.best_z * 0.7f + (float)p.z * 0.3f;
  float verify = (float)s.verify_ok - (float)s.verify_fail * 0.5f;
  gIntention = (gIntention * 0.92f) + ((proof + verify + link * 8.0f) * 0.08f);
}

static void emitStatus(Stream &out, const char *kind) {
  JanusStats s;
  JanusPeer p;
  JanusJob j;

  portENTER_CRITICAL(&gMux);
  memcpy(&s, &gStats, sizeof(s));
  memcpy(&p, &gPeer, sizeof(p));
  memcpy(&j, &gJob, sizeof(j));
  portEXIT_CRITICAL(&gMux);

  char digestHex[65];
  digestToHex(s.best_digest, digestHex);

  out.print("{\"kind\":\"");
  out.print(kind);
  out.print("\",\"ver\":\"");
  out.print(JANUS_P4_VERSION);
  out.print("\",\"node_id\":\"");
  out.print(gNodeId);
  out.print("\",\"role\":\"");
  out.print(roleName(gRole));
  out.print("\",\"mode\":\"");
  out.print(modeName(gMode));
  out.print("\",\"native_wifi\":false");
  out.print(",\"uart_bridge\":");
  out.print(JANUS_USE_UART_BRIDGE ? "true" : "false");
  out.print(",\"uptime_ms\":");
  out.print(millis());
  out.print(",\"worker\":");
  out.print(gWorkerRunning ? "true" : "false");
  out.print(",\"job_id\":\"");
  out.print(j.job_id);
  out.print("\",\"job_seed\":\"0x");
  out.print(j.seed, HEX);
  out.print("\",\"job_age_ms\":");
  out.print(jobAgeMs());
  out.print(",\"target_z\":");
  out.print(j.target_z);
  out.print(",\"hps\":");
  out.print(gLastHps, 1);
  out.print(",\"total_hashes\":");
  printU64(out, s.total_hashes);
  out.print(",\"rounds\":");
  out.print(s.rounds);
  out.print(",\"best_z\":");
  out.print(s.best_z);
  out.print(",\"best_seq\":");
  out.print(s.best_seq);
  out.print(",\"best_nonce\":");
  out.print(s.best_nonce);
  out.print(",\"best_salt\":\"0x");
  out.print(s.best_salt, HEX);
  out.print("\",\"best_digest\":\"");
  out.print(digestHex);
  out.print("\",\"local_proofs\":");
  out.print(s.local_proofs);
  out.print(",\"peer_seen\":");
  out.print(p.seen ? "true" : "false");
  out.print(",\"peer_age_ms\":");
  out.print(peerAgeMs());
  out.print(",\"peer_node\":\"");
  out.print(p.node_id);
  out.print("\",\"peer_role\":\"");
  out.print(p.role);
  out.print("\",\"peer_z\":");
  out.print(p.z);
  out.print(",\"peer_hps\":");
  out.print(p.hps, 1);
  out.print(",\"peer_ok\":");
  out.print(s.peer_frames_ok);
  out.print(",\"peer_bad_csum\":");
  out.print(s.peer_frames_bad_csum);
  out.print(",\"verify_ok\":");
  out.print(s.verify_ok);
  out.print(",\"verify_fail\":");
  out.print(s.verify_fail);
  out.print(",\"mirror_hits\":");
  out.print(s.mirror_hits);
  out.print(",\"bridge_tx\":");
  out.print(s.bridge_tx);
  out.print(",\"intention\":");
  out.print(gIntention, 2);
  out.print(",\"blackboard\":");
  out.print(gBlackboard, 2);
  out.print(",\"free_heap\":");
  out.print((unsigned)ESP.getFreeHeap());
  out.println("}");
}

static void emitProof(uint32_t z, uint32_t seq, uint32_t nonce, uint32_t salt, uint32_t jobSeed, const uint8_t digest[32]) {
  char digestHex[65];
  digestToHex(digest, digestHex);

  Serial.print("{\"kind\":\"JANUS_P4_PROOF\",\"ver\":\"");
  Serial.print(JANUS_P4_VERSION);
  Serial.print("\",\"node_id\":\"");
  Serial.print(gNodeId);
  Serial.print("\",\"role\":\"");
  Serial.print(roleName(gRole));
  Serial.print("\",\"z\":");
  Serial.print(z);
  Serial.print(",\"seq\":");
  Serial.print(seq);
  Serial.print(",\"nonce\":");
  Serial.print(nonce);
  Serial.print(",\"salt\":\"0x");
  Serial.print(salt, HEX);
  Serial.print("\",\"job_seed\":\"0x");
  Serial.print(jobSeed, HEX);
  Serial.print("\",\"digest\":\"");
  Serial.print(digestHex);
  Serial.println("\"}");

#if JANUS_USE_UART_BRIDGE
  JanusBridge.print("{\"kind\":\"JANUS_P4_PROOF\",\"node_id\":\"");
  JanusBridge.print(gNodeId);
  JanusBridge.print("\",\"role\":\"");
  JanusBridge.print(roleName(gRole));
  JanusBridge.print("\",\"z\":");
  JanusBridge.print(z);
  JanusBridge.print(",\"seq\":");
  JanusBridge.print(seq);
  JanusBridge.print(",\"nonce\":");
  JanusBridge.print(nonce);
  JanusBridge.print(",\"salt\":\"0x");
  JanusBridge.print(salt, HEX);
  JanusBridge.print("\",\"job_seed\":\"0x");
  JanusBridge.print(jobSeed, HEX);
  JanusBridge.print("\",\"digest\":\"");
  JanusBridge.print(digestHex);
  JanusBridge.println("\"}");
#endif
}

// ============================================================
// COMMANDS
// ============================================================

static void setJob(uint32_t seed, uint32_t targetZ, const char *jobId) {
  portENTER_CRITICAL(&gMux);
  gJob.seed = seed;
  gJob.target_z = targetZ;
  gJob.started_ms = millis();
  gJob.ttl_ms = JANUS_JOB_TIMEOUT_MS;
  if (jobId && *jobId) {
    strncpy(gJob.job_id, jobId, sizeof(gJob.job_id) - 1);
    gJob.job_id[sizeof(gJob.job_id) - 1] = '\0';
  } else {
    snprintf(gJob.job_id, sizeof(gJob.job_id), "%08lx", (unsigned long)seed);
  }
  portEXIT_CRITICAL(&gMux);
  gResetBest = true;
}

static void handleCommand(char *line, Stream &out) {
  while (*line == ' ' || *line == '\t') line++;
  if (*line == '\0') return;

  if (!strncmp(line, "JP4,", 4)) {
    bool ok = parseJP4Frame(line);
    if (!ok) {
      out.println("{\"kind\":\"JANUS_P4_FRAME_RX\",\"ok\":false}");
    }
    return;
  }

  portENTER_CRITICAL(&gMux);
  gStats.cmd_count++;
  portEXIT_CRITICAL(&gMux);

  if (!strcasecmp(line, "help") || !strcmp(line, "?")) {
    printHelp(out);
    return;
  }

  if (!strcasecmp(line, "status")) {
    emitStatus(out, "JANUS_P4_STATUS");
    return;
  }

  if (!strcasecmp(line, "reset")) {
    gResetBest = true;
    out.println("{\"kind\":\"JANUS_P4_CMD\",\"cmd\":\"reset\",\"ok\":true}");
    return;
  }

  if (!strncasecmp(line, "role ", 5)) {
    gRole = parseRoleName(line + 5);
    out.print("{\"kind\":\"JANUS_P4_CMD\",\"cmd\":\"role\",\"ok\":true,\"role\":\"");
    out.print(roleName(gRole));
    out.println("\"}");
    return;
  }

  if (!strncasecmp(line, "mode ", 5)) {
    gMode = parseModeName(line + 5);
    out.print("{\"kind\":\"JANUS_P4_CMD\",\"cmd\":\"mode\",\"ok\":true,\"mode\":\"");
    out.print(modeName(gMode));
    out.println("\"}");
    return;
  }

  if (!strncasecmp(line, "seed ", 5)) {
    uint32_t v = 0;
    if (parseHex32(line + 5, &v)) {
      gExtraSalt ^= v;
      out.print("{\"kind\":\"JANUS_P4_CMD\",\"cmd\":\"seed\",\"ok\":true,\"seed\":\"0x");
      out.print(v, HEX);
      out.println("\"}");
    } else {
      portENTER_CRITICAL(&gMux);
      gStats.bad_cmd_count++;
      portEXIT_CRITICAL(&gMux);
      out.println("{\"kind\":\"JANUS_P4_CMD\",\"cmd\":\"seed\",\"ok\":false}");
    }
    return;
  }

  if (!strncasecmp(line, "job ", 4)) {
    char *save = nullptr;
    char *cmd = strtok_r(line, " ", &save);
    char *seedText = strtok_r(nullptr, " ", &save);
    char *targetText = strtok_r(nullptr, " ", &save);
    char *jobIdText = strtok_r(nullptr, " ", &save);
    (void)cmd;

    uint32_t seed = 0;
    uint32_t target = JANUS_PROOF_Z_THRESHOLD;
    if (!seedText || !parseHex32(seedText, &seed)) {
      out.println("{\"kind\":\"JANUS_P4_CMD\",\"cmd\":\"job\",\"ok\":false,\"error\":\"bad_seed\"}");
      return;
    }
    if (targetText) {
      parseU32(targetText, &target);
    }
    setJob(seed, target, jobIdText);
    out.print("{\"kind\":\"JANUS_P4_CMD\",\"cmd\":\"job\",\"ok\":true,\"seed\":\"0x");
    out.print(seed, HEX);
    out.print("\",\"target_z\":");
    out.print(target);
    out.println("}");
    return;
  }

  portENTER_CRITICAL(&gMux);
  gStats.bad_cmd_count++;
  portEXIT_CRITICAL(&gMux);
  out.println("{\"kind\":\"JANUS_P4_CMD\",\"ok\":false,\"error\":\"unknown_command\"}");
}

static void pumpLineInput(Stream &in, char *buf, size_t &len, Stream &out) {
  while (in.available() > 0) {
    char c = (char)in.read();
    if (c == '\r') continue;
    if (c == '\n') {
      buf[len] = '\0';
      handleCommand(buf, out);
      len = 0;
      continue;
    }
    if (len + 1 < 256) {
      buf[len++] = c;
    } else {
      len = 0;
      out.println("{\"kind\":\"JANUS_P4_CMD\",\"ok\":false,\"error\":\"line_too_long\"}");
    }
  }
}

// ============================================================
// ADAPTIVE MODE
// ============================================================

static JanusMode activeWorkerMode() {
  JanusMode mode = gMode;
  if (mode != MODE_AUTO) return mode;

  if (gRole == ROLE_P4_A) {
    return MODE_SCOUT;
  }

  if (gRole == ROLE_P4_B) {
    uint32_t age = peerAgeMs();
    if (age < JANUS_PEER_TIMEOUT_MS) return MODE_VERIFY;
    return MODE_MIRROR;
  }

  return MODE_RELAY;
}

static uint32_t currentHashBatch(JanusMode mode) {
  if (mode == MODE_MIRROR || mode == MODE_VERIFY) return JANUS_HASH_BATCH_MIRROR;
  if (mode == MODE_IDLE || mode == MODE_RELAY) return 64;
  return JANUS_HASH_BATCH_FAST;
}

// ============================================================
// HASH WORKER
// ============================================================

static void hashWorkerTask(void *param) {
  (void)param;

  uint32_t seq = esp_random();
  uint32_t nonce = esp_random();
  uint32_t salt = fnv1a32Text(roleName(gRole)) ^ esp_random();
  uint8_t digest[32];

  gWorkerRunning = true;

  for (;;) {
    if (gResetBest) {
      gResetBest = false;
      resetBestOnly();
    }

    JanusMode workMode = activeWorkerMode();

    if (workMode == MODE_VERIFY || workMode == MODE_MIRROR || workMode == MODE_RELAY) {
      verifyOneCandidate();
    }

    if (workMode == MODE_IDLE || workMode == MODE_RELAY) {
      vTaskDelay(5);
      continue;
    }

    JanusJob job;
    JanusPeer peer;
    portENTER_CRITICAL(&gMux);
    memcpy(&job, &gJob, sizeof(job));
    memcpy(&peer, &gPeer, sizeof(peer));
    portEXIT_CRITICAL(&gMux);

    if (millis() - job.started_ms > job.ttl_ms) {
      uint32_t newSeed = xorshift32(job.seed ^ esp_random() ^ (uint32_t)esp_timer_get_time());
      setJob(newSeed, job.target_z, "local");
      job.seed = newSeed;
    }

    uint32_t batch = currentHashBatch(workMode);
    uint32_t hashes = 0;

    for (uint32_t i = 0; i < batch; i++) {
      nonce++;
      if (nonce == 0) {
        seq++;
        salt = xorshift32(salt ^ seq ^ esp_random());
      }

      uint32_t modeSalt = salt ^ gExtraSalt;
      uint32_t jobSeed = job.seed;

      if (workMode == MODE_MIRROR || workMode == MODE_VERIFY) {
        modeSalt ^= peer.salt ^ 0xB16B00B5UL;
        jobSeed ^= peer.job_seed ^ 0x51A7E11AUL;
      }

      uint8_t z = hashCandidate(
        gBaseMac,
        roleName(gRole),
        jobSeed,
        modeSalt,
        seq,
        nonce,
        digest
      );

      hashes++;

      bool newBest = false;
      portENTER_CRITICAL(&gMux);
      if (z > gStats.best_z) {
        gStats.best_z = z;
        gStats.best_seq = seq;
        gStats.best_nonce = nonce;
        gStats.best_salt = modeSalt;
        gStats.best_job_seed = jobSeed;
        memcpy(gStats.best_digest, digest, 32);
        newBest = true;
        if (z >= job.target_z || z >= JANUS_PROOF_Z_THRESHOLD) {
          gStats.local_proofs++;
        }
      }
      portEXIT_CRITICAL(&gMux);

      if (newBest) {
        rememberAnchor(
          true,
          true,
          gNodeId,
          roleName(gRole),
          z,
          seq,
          nonce,
          modeSalt,
          jobSeed,
          digest
        );

        if (z >= job.target_z || z >= JANUS_PROOF_Z_THRESHOLD) {
          emitProof(z, seq, nonce, modeSalt, jobSeed, digest);
        }
      }
    }

    portENTER_CRITICAL(&gMux);
    gStats.total_hashes += hashes;
    gStats.rounds++;
    portEXIT_CRITICAL(&gMux);

    vTaskDelay(1);
  }
}

// ============================================================
// ARDUINO
// ============================================================

void setup() {
  Serial.begin(115200);
  delay(250);

  readBaseMac();

  if (!strcmp(JANUS_DEFAULT_ROLE, "P4_B")) gRole = ROLE_P4_B;
  else if (!strcmp(JANUS_DEFAULT_ROLE, "P4_C")) gRole = ROLE_P4_C;
  else gRole = ROLE_P4_A;

  memset(&gStats, 0, sizeof(gStats));
  memset(&gPeer, 0, sizeof(gPeer));
  memset(gAnchors, 0, sizeof(gAnchors));
  memset(gVerifyQueue, 0, sizeof(gVerifyQueue));

  uint32_t bootSeed = fnv1a32Text(gNodeId) ^ esp_random() ^ (uint32_t)esp_timer_get_time();
  memset(&gJob, 0, sizeof(gJob));
  setJob(bootSeed, JANUS_PROOF_Z_THRESHOLD, "boot");

#if JANUS_HEARTBEAT_LED_PIN >= 0
  pinMode(JANUS_HEARTBEAT_LED_PIN, OUTPUT);
  digitalWrite(JANUS_HEARTBEAT_LED_PIN, !JANUS_HEARTBEAT_ACTIVE);
#endif

#if JANUS_USE_UART_BRIDGE
  JanusBridge.begin(JANUS_UART_BAUD, SERIAL_8N1, JANUS_UART_RX_PIN, JANUS_UART_TX_PIN);
#endif

  printBootInfo();

#if JANUS_PIN_WORKER_TO_CORE
  xTaskCreatePinnedToCore(
    hashWorkerTask,
    "janus_p4_core",
    JANUS_WORKER_STACK,
    nullptr,
    1,
    nullptr,
    JANUS_WORKER_CORE
  );
#else
  xTaskCreate(
    hashWorkerTask,
    "janus_p4_core",
    JANUS_WORKER_STACK,
    nullptr,
    1,
    nullptr
  );
#endif
}

void loop() {
  static uint32_t lastTelemetryMs = 0;
  static uint32_t lastFrameMs = 0;
  static uint32_t lastBlinkMs = 0;
  static bool ledState = false;

  pumpLineInput(Serial, usbLine, usbLineLen, Serial);

#if JANUS_USE_UART_BRIDGE
  pumpLineInput(JanusBridge, bridgeLine, bridgeLineLen, JanusBridge);
#endif

  uint32_t now = millis();

  if (now - lastBlinkMs >= 500) {
    lastBlinkMs = now;
    ledState = !ledState;
#if JANUS_HEARTBEAT_LED_PIN >= 0
    digitalWrite(JANUS_HEARTBEAT_LED_PIN, ledState ? JANUS_HEARTBEAT_ACTIVE : !JANUS_HEARTBEAT_ACTIVE);
#endif
  }

  if (now - lastTelemetryMs >= JANUS_TELEMETRY_MS) {
    uint32_t elapsed = now - lastTelemetryMs;
    lastTelemetryMs = now;

    uint64_t total;
    uint64_t last;
    portENTER_CRITICAL(&gMux);
    total = gStats.total_hashes;
    last = gStats.last_total_hashes;
    gStats.last_total_hashes = total;
    portEXIT_CRITICAL(&gMux);

    if (elapsed > 0 && total >= last) {
      gLastHps = ((float)(total - last) * 1000.0f) / (float)elapsed;
    }

    updateAdaptiveFields();
    emitStatus(Serial, "JANUS_P4_TELEMETRY");

#if JANUS_USE_UART_BRIDGE
    emitStatus(JanusBridge, "JANUS_P4_TELEMETRY");
#endif
  }

  if (now - lastFrameMs >= JANUS_FRAME_MS) {
    lastFrameMs = now;
    emitJP4Frame(Serial);
#if JANUS_USE_UART_BRIDGE
    emitJP4Frame(JanusBridge);
#endif
  }

  delay(5);
}
