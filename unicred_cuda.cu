// UNICRED GPU miner core (CUDA, keccak-256).
//
// Verified formula (from digestOf(bytes32,bytes32,address,uint256) at 0x3a39a703):
//   digest = keccak256(abi.encode(anchorHash, prev, miner, nonce))
//          = 5 * 32 = 160 bytes = 1 rate block (136) + 24-byte tail + padding
//
// Layout of 272-byte block (2 keccak rates of 136 bytes each):
//   byte 0..31   : anchorHash        (bytes32)   [static per anchor]
//   byte 32..63  : prev              (bytes32)   [static per session]
//   byte 64..95  : miner (padded)    (address)   [static per miner]
//   byte 96..127 : nonce high 128b   (uint256)   [zero — we vary low 8 bytes]
//   byte 128..135: nonce hi 8 bytes  [pos 0..7 in block 2] — varied per attempt
//   byte 136..143: nonce mid 8 bytes [pos 8..15 in block 2]
//   byte 144..151: nonce lo 16 bytes [pos 16..31 in block 2]
//   byte 152..159: (still nonce, part of the uint256 nonce word)
//   ...
// So nonce (32 bytes big-endian) occupies bytes 128..159. We vary low 8 bytes (152..159).
//
// Block 1 (bytes 0..135) is CONSTANT for a given anchor — midstate optimization.
// Block 2 varies only in the last 8 bytes of the nonce.
//
// Validation: hash (as 256-bit big-endian) < target.
//
// Protocol (stdin/stdout, line based):
//   JOB <anchor_hex64> <prev_hex64> <miner_hex40> <target_hex64>
//   -> RATE <hashes_per_sec>
//      FOUND <nonce_hex64> <hash_hex64>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <chrono>
#include <random>
#include <mutex>
#include <cstdarg>
#include <sys/select.h>
#include <unistd.h>
#include <cuda_runtime.h>

static std::mutex g_out;
static void out(const char *fmt, ...) {
  std::lock_guard<std::mutex> lk(g_out);
  va_list ap; va_start(ap, fmt); vprintf(fmt, ap); va_end(ap);
}

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)

// After first-block absorbtion + permutation, we have a "midstate" (25 lanes).
// Second block adds 17 more lanes (bytes 136..271) then permutes.
// Bytes 136..151 = nonce middle 16 bytes (in block 2, positions 0..15) = lanes 0, 1
// Bytes 152..159 = nonce low 8 bytes (positions 16..23) = lane 2
// Byte 160 = 0x01 (padding start, message length = 160)
// Byte 271 = 0x80 (rate marker for block 2)
//
// So block 2 template:
//   lane 0: nonce bytes at msg positions 128..135 (upper part) — from prev block absorb? No wait...
//
// Actually let me recount. Message is 160 bytes total:
//   0..31 anchor, 32..63 prev, 64..95 miner_padded, 96..127 zero, 128..159 nonce (BE uint256)
//
// keccak absorbs 136 bytes per block. Block 1 = bytes 0..135 = anchor+prev+miner+zero+first 8 of nonce.
// Block 2 = bytes 136..271, with real content only in bytes 136..159 (=nonce bytes 8..31).
// So the last 24 bytes of nonce (bytes 152..159 are the LOW 8, we vary them) split across blocks!
//
// Actually the low 8 nonce bytes (152..159) are in block 2 at positions 16..23 = lane 2 (bytes 16..23).
// The mid 16 bytes of nonce (136..151) are in block 2 positions 0..15 = lanes 0, 1.
// The upper 8 bytes of nonce (128..135) are in block 1 positions 128..135 = lanes 16 last 8 bytes.
//
// To keep midstate valid across nonce changes, we vary ONLY bytes in block 2 = only bytes 136..159.
// That gives us 24 bytes (192 bits) of nonce freedom — plenty.
// Bytes 128..135 (upper 8 nonce bytes) stay CONSTANT per session (part of midstate template).
//
// Layout in block 2:
//   lane 0 (bytes 136..143 = nonce bytes 8..15)  — session constant "hi_session"
//   lane 1 (bytes 144..151 = nonce bytes 16..23) — session constant "mid_session"
//   lane 2 (bytes 152..159 = nonce bytes 24..31) — VARIED per attempt (uint64 BE)
//   lane 3..16 = zero
//   lane 17 = padding+rate marker (position 136 has 0x01... wait no, position 24-135 of block 2 is zero)

// Simplification: we'll vary lane 2 only (last 8 bytes of nonce). Lanes 0, 1 fixed per session.

__constant__ uint64_t c_midstate[25];       // state after first block absorb+permute
__constant__ uint64_t c_block2_tpl[17];     // block 2 template lanes (lane 2 = 0 for us to fill)
__constant__ uint64_t c_target[4];          // target as 4 big-endian 64-bit words

__device__ __forceinline__ uint64_t rotl64(uint64_t x, int n) { return (x << n) | (x >> (64 - n)); }
__device__ __forceinline__ uint64_t bswap64(uint64_t x) {
  return ((x & 0x00000000000000FFULL) << 56) | ((x & 0x000000000000FF00ULL) << 40) |
         ((x & 0x0000000000FF0000ULL) << 24) | ((x & 0x00000000FF000000ULL) << 8) |
         ((x & 0x000000FF00000000ULL) >> 8)  | ((x & 0x0000FF0000000000ULL) >> 24) |
         ((x & 0x00FF000000000000ULL) >> 40) | ((x & 0xFF00000000000000ULL) >> 56);
}

__constant__ uint64_t RC[24] = {
  0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
  0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
  0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
  0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
  0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
  0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL };

__device__ __forceinline__ void keccakf(uint64_t s[25]) {
  uint64_t t, bc[5];
  #pragma unroll 1
  for (int r = 0; r < 24; r++) {
    for (int i = 0; i < 5; i++) bc[i] = s[i] ^ s[i + 5] ^ s[i + 10] ^ s[i + 15] ^ s[i + 20];
    for (int i = 0; i < 5; i++) {
      t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
      for (int j = 0; j < 25; j += 5) s[j + i] ^= t;
    }
    t = s[1];
    s[1] = rotl64(s[6], 44);  s[6] = rotl64(s[9], 20);  s[9] = rotl64(s[22], 61); s[22] = rotl64(s[14], 39);
    s[14] = rotl64(s[20], 18); s[20] = rotl64(s[2], 62); s[2] = rotl64(s[12], 43); s[12] = rotl64(s[13], 25);
    s[13] = rotl64(s[19], 8);  s[19] = rotl64(s[23], 56); s[23] = rotl64(s[15], 41); s[15] = rotl64(s[4], 27);
    s[4] = rotl64(s[24], 14);  s[24] = rotl64(s[21], 2);  s[21] = rotl64(s[8], 55); s[8] = rotl64(s[16], 45);
    s[16] = rotl64(s[5], 36);  s[5] = rotl64(s[3], 28);   s[3] = rotl64(s[18], 21); s[18] = rotl64(s[17], 15);
    s[17] = rotl64(s[11], 10); s[11] = rotl64(s[7], 6);   s[7] = rotl64(s[10], 3);  s[10] = rotl64(t, 1);
    for (int j = 0; j < 25; j += 5) {
      for (int i = 0; i < 5; i++) bc[i] = s[j + i];
      for (int i = 0; i < 5; i++) s[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
    }
    s[0] ^= RC[r];
  }
}

__global__ void mine_kernel(uint32_t hi, uint32_t lo_base, uint32_t per_thread,
                             uint64_t *out_nonce_lo, uint64_t *out_hash, unsigned int *out_cnt) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t lo = lo_base + tid * per_thread;
  for (uint32_t k = 0; k < per_thread; k++, lo++) {
    // Build state = midstate XOR block2 lanes (with our nonce in lane 2)
    uint64_t s[25];
    #pragma unroll
    for (int i = 0; i < 25; i++) s[i] = c_midstate[i];
    #pragma unroll
    for (int i = 0; i < 17; i++) s[i] ^= c_block2_tpl[i];

    // Nonce low 8 bytes (BE uint64) = hi<<32 | lo, placed as big-endian in lane 2
    // Lane is stored as little-endian uint64 in memory, so to write BE bytes we bswap
    uint64_t nonce_lo64 = ((uint64_t)hi << 32) | lo;
    uint64_t nonce_be_lane = bswap64(nonce_lo64);
    s[2] ^= nonce_be_lane;

    keccakf(s);

    // Check hash < target (256-bit big-endian compare)
    // s[0] is LE lane, corresponds to bytes 0..7 of hash. bswap gives BE value.
    uint64_t h0 = bswap64(s[0]);
    if (h0 > c_target[0]) continue;
    if (h0 == c_target[0]) {
      uint64_t h1 = bswap64(s[1]);
      if (h1 > c_target[1]) continue;
      if (h1 == c_target[1]) {
        uint64_t h2 = bswap64(s[2]);
        if (h2 > c_target[2]) continue;
        if (h2 == c_target[2] && bswap64(s[3]) >= c_target[3]) continue;
      }
    }

    unsigned int idx = atomicAdd(out_cnt, 1u);
    if (idx < 8) {
      out_nonce_lo[idx] = nonce_lo64;
      out_hash[idx * 4 + 0] = h0;
      out_hash[idx * 4 + 1] = bswap64(s[1]);
      out_hash[idx * 4 + 2] = bswap64(s[2]);
      out_hash[idx * 4 + 3] = bswap64(s[3]);
    }
  }
}

static int hexval(char c) { if (c >= '0' && c <= '9') return c - '0'; c |= 0x20; if (c >= 'a' && c <= 'f') return c - 'a' + 10; return -1; }
static bool hex2bytes(const std::string &h, uint8_t *out, size_t n) {
  std::string s = h; if (s.rfind("0x", 0) == 0) s = s.substr(2);
  if (s.size() != n * 2) return false;
  for (size_t i = 0; i < n; i++) { int a = hexval(s[2*i]), b = hexval(s[2*i+1]); if (a < 0 || b < 0) return false; out[i] = (uint8_t)(a * 16 + b); }
  return true;
}

// keccak permutation on host (for computing midstate)
static void keccakf_host(uint64_t s[25]) {
  static const uint64_t hRC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL };
  uint64_t t, bc[5];
  for (int r = 0; r < 24; r++) {
    for (int i = 0; i < 5; i++) bc[i] = s[i] ^ s[i+5] ^ s[i+10] ^ s[i+15] ^ s[i+20];
    for (int i = 0; i < 5; i++) {
      t = bc[(i+4)%5] ^ ((bc[(i+1)%5] << 1) | (bc[(i+1)%5] >> 63));
      for (int j = 0; j < 25; j += 5) s[j+i] ^= t;
    }
    t = s[1];
    #define ROTL(x, n) (((x) << (n)) | ((x) >> (64-(n))))
    s[1]=ROTL(s[6],44); s[6]=ROTL(s[9],20); s[9]=ROTL(s[22],61); s[22]=ROTL(s[14],39);
    s[14]=ROTL(s[20],18); s[20]=ROTL(s[2],62); s[2]=ROTL(s[12],43); s[12]=ROTL(s[13],25);
    s[13]=ROTL(s[19],8); s[19]=ROTL(s[23],56); s[23]=ROTL(s[15],41); s[15]=ROTL(s[4],27);
    s[4]=ROTL(s[24],14); s[24]=ROTL(s[21],2); s[21]=ROTL(s[8],55); s[8]=ROTL(s[16],45);
    s[16]=ROTL(s[5],36); s[5]=ROTL(s[3],28); s[3]=ROTL(s[18],21); s[18]=ROTL(s[17],15);
    s[17]=ROTL(s[11],10); s[11]=ROTL(s[7],6); s[7]=ROTL(s[10],3); s[10]=ROTL(t,1);
    #undef ROTL
    for (int j = 0; j < 25; j += 5) {
      for (int i = 0; i < 5; i++) bc[i] = s[j+i];
      for (int i = 0; i < 5; i++) s[j+i] ^= (~bc[(i+1)%5]) & bc[(i+2)%5];
    }
    s[0] ^= hRC[r];
  }
}

static bool build_job(const std::string &anchor, const std::string &prev,
                       const std::string &miner_hex, const std::string &target_hex) {
  // Build 272-byte message padded to 2 blocks
  uint8_t msg[272]; memset(msg, 0, sizeof msg);
  if (!hex2bytes(anchor, msg, 32)) { out("ERR bad anchor\n"); return false; }
  if (!hex2bytes(prev, msg + 32, 32)) { out("ERR bad prev\n"); return false; }
  // miner: address is 20 bytes, in abi.encode it's padded to 32 bytes (address goes to LOW bytes)
  // padding: 12 zero bytes then 20 address bytes
  uint8_t miner_bytes[20];
  if (!hex2bytes(miner_hex, miner_bytes, 20)) { out("ERR bad miner\n"); return false; }
  memcpy(msg + 64 + 12, miner_bytes, 20);
  // nonce (bytes 128..159) stays zero for template — we'll vary the low 8 bytes
  // padding: byte 160 = 0x01, byte 271 (last byte of 2nd block, rate 136) = 0x80
  msg[160] = 0x01;
  msg[271] = 0x80;

  // Absorb block 1 (bytes 0..135) into fresh state, then permute → midstate
  uint64_t s[25]; memset(s, 0, sizeof s);
  for (int i = 0; i < 17; i++) {
    uint64_t v = 0;
    for (int b = 0; b < 8; b++) v |= (uint64_t)msg[i*8+b] << (8*b);
    s[i] ^= v;
  }
  keccakf_host(s);
  CK(cudaMemcpyToSymbol(c_midstate, s, sizeof s));

  // Block 2 template lanes (17 lanes from bytes 136..271)
  uint64_t b2[17]; memset(b2, 0, sizeof b2);
  for (int i = 0; i < 17; i++) {
    uint64_t v = 0;
    for (int b = 0; b < 8; b++) v |= (uint64_t)msg[136 + i*8 + b] << (8*b);
    b2[i] = v;
  }
  // We'll XOR c_block2_tpl into state and then XOR the nonce bytes into lane 2.
  // Lane 2 (bytes 152..159) is currently zero in our template — good, we'll fill it per attempt.
  CK(cudaMemcpyToSymbol(c_block2_tpl, b2, sizeof b2));

  // Target as 4 big-endian words
  uint8_t tb[32]; if (!hex2bytes(target_hex, tb, 32)) { out("ERR bad target\n"); return false; }
  uint64_t tw[4];
  for (int i = 0; i < 4; i++) { uint64_t v = 0; for (int b = 0; b < 8; b++) v = (v << 8) | tb[i*8+b]; tw[i] = v; }
  CK(cudaMemcpyToSymbol(c_target, tw, sizeof tw));

  out("INFO job accepted\n");
  return true;
}

static bool stdin_ready() {
  fd_set fds; FD_ZERO(&fds); FD_SET(0, &fds);
  struct timeval tv = {0, 0};
  return select(1, &fds, nullptr, nullptr, &tv) > 0;
}

int main() {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  int dev = 0; CK(cudaSetDevice(dev));
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));
  printf("INFO device %s sm=%d.%d SMs=%d\n", p.name, p.major, p.minor, p.multiProcessorCount);

  uint64_t *d_nonce, *d_hash; unsigned int *d_cnt;
  CK(cudaMalloc(&d_nonce, 8 * sizeof(uint64_t)));
  CK(cudaMalloc(&d_hash, 32 * sizeof(uint64_t)));
  CK(cudaMalloc(&d_cnt, sizeof(unsigned int)));

  const int threads = 256;
  const int blocks = p.multiProcessorCount * 64;
  const uint32_t per_thread = 256;
  const uint64_t per_launch = (uint64_t)blocks * threads * per_thread;

  std::mt19937_64 rng(std::chrono::steady_clock::now().time_since_epoch().count() ^ getpid());
  uint32_t hi = (uint32_t)rng();
  uint32_t lo = (uint32_t)rng();

  bool have_job = false;
  std::string line;
  uint64_t hashes = 0; auto t0 = std::chrono::steady_clock::now();

  while (true) {
    while (stdin_ready() || !have_job) {
      char lb[1024];
      if (!fgets(lb, sizeof lb, stdin)) { printf("INFO stdin closed\n"); return 0; }
      line = lb; while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
      char cmd[16], a[128], b[128], c[128], d[128];
      int nf = sscanf(line.c_str(), "%15s %127s %127s %127s %127s", cmd, a, b, c, d);
      if (nf == 5 && strcmp(cmd, "JOB") == 0) {
        // JOB <anchor> <prev> <miner> <target>
        if (build_job(a, b, c, d)) {
          have_job = true;
          hi = (uint32_t)rng();
          lo = (uint32_t)rng();
        }
      } else if (strcmp(line.c_str(), "STOP") == 0) { have_job = false; out("INFO stopped\n"); }
      else if (!line.empty()) out("ERR unknown: %s\n", line.c_str());
      if (!have_job) continue;
    }

    CK(cudaMemset(d_cnt, 0, sizeof(unsigned int)));
    mine_kernel<<<blocks, threads>>>(hi, lo, per_thread, d_nonce, d_hash, d_cnt);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    unsigned int cnt = 0; CK(cudaMemcpy(&cnt, d_cnt, sizeof cnt, cudaMemcpyDeviceToHost));
    if (cnt > 0) {
      uint64_t hn[8], hh[32];
      CK(cudaMemcpy(hn, d_nonce, sizeof hn, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(hh, d_hash, sizeof hh, cudaMemcpyDeviceToHost));
      for (unsigned int i = 0; i < cnt && i < 8; i++) {
        out("FOUND %016llx %016llx%016llx%016llx%016llx\n",
            (unsigned long long)hn[i],
            (unsigned long long)hh[i*4], (unsigned long long)hh[i*4+1],
            (unsigned long long)hh[i*4+2], (unsigned long long)hh[i*4+3]);
      }
    }
    lo += (uint32_t)per_launch;
    hashes += per_launch;
    auto t1 = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();
    if (dt >= 2.0) { out("RATE %.0f\n", hashes / dt); hashes = 0; t0 = t1; }
  }
}
