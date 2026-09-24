"""
UNICRED native GPU miner — orchestrator.
Unichain (chain 130), contract 0xf60de24F228dc7Ca6fF025958d2eE3A956ED88E5.

Formula (verified via digestOf):
    digest = keccak256(abi.encode(anchorHash, prev, miner, nonce))
    valid if digest < targetFor(miner)

anchorHash = blockhash(anchorBlock), where anchorBlock is periodically updated by contract.
We take anchorBlock from state(miner) — the field at index 7 (0x38bd171 in test call).

mine(uint256 anchorBlock, uint256 nonce, uint256 maxPrice) payable
    selector 0x106c9da1

ENV:
    MINER_PRIVATE_KEY  0x...
    RPC_URLS           default: mainnet.unichain.org + backups
    CUDA_BIN           default: ./unicred_cuda
    REFRESH_SEC        default: 1.0
"""
import os
import random
import subprocess
import sys
import threading
import time
from queue import Queue, Empty

import requests
from eth_account import Account
from web3 import Web3

PRIVATE_KEY = os.environ.get("MINER_PRIVATE_KEY", "").strip()

DEFAULT_RPCS = ",".join([
    "https://mainnet.unichain.org",
    "https://unichain.drpc.org",
    "https://unichain-mainnet.g.alchemy.com/public",
])
RPC_URLS = [u.strip() for u in os.environ.get("RPC_URLS", DEFAULT_RPCS).split(",") if u.strip()]
random.shuffle(RPC_URLS)

CUDA_BIN     = os.environ.get("CUDA_BIN", "./unicred_cuda")
REFRESH_SEC  = float(os.environ.get("REFRESH_SEC", "1.0"))

CHAIN_ID  = 130   # Unichain
CONTRACT  = Web3.to_checksum_address("0xf60de24F228dc7Ca6fF025958d2eE3A956ED88E5")

# Selectors
SEL_STATE       = "0x31e658a5"   # state(address who) → tuple with anchorBlock at index 7
SEL_PREVWORK    = "0xa4da5da2"   # prevWork()
SEL_TARGETFOR   = "0x16ccc8c0"   # targetFor(address)
SEL_TOTALMINTED = "0xa2309ff8"
SEL_ANCHOR_WIN  = "0x14d0c338"   # ANCHOR_WINDOW

# mine(uint256 anchorBlock, uint256 nonce, uint256 maxPrice) payable
MINE_SELECTOR   = "0x106c9da1"

if not (PRIVATE_KEY.startswith("0x") and len(PRIVATE_KEY) == 66):
    print("MINER_PRIVATE_KEY missing or invalid"); sys.exit(1)

acct = Account.from_key(PRIVATE_KEY)
ADDR = acct.address


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------- RPC ----------------
class Rpc:
    def __init__(self, urls):
        self.urls = urls; self.i = 0; self.sess = requests.Session()

    @property
    def url(self): return self.urls[self.i]

    def rotate(self): self.i = (self.i + 1) % len(self.urls)

    def call(self, method, params, tries=6, timeout=10):
        last = None
        for _ in range(tries):
            try:
                r = self.sess.post(self.url, json={"jsonrpc":"2.0","id":1,"method":method,"params":params}, timeout=timeout)
                r.raise_for_status()
                j = r.json()
                if "error" in j: raise RuntimeError(str(j["error"])[:200])
                return j["result"]
            except Exception as e:
                last = e
                self.rotate()
                time.sleep(0.2)
        raise RuntimeError(f"RPC dead: {last}")


rpc = Rpc(RPC_URLS)


def read_state():
    """Reads state(me), prevWork, targetFor(me), current anchor block info."""
    to = CONTRACT
    addr_word = ADDR[2:].lower().rjust(64, "0")
    state = rpc.call("eth_call", [{"to": to, "data": SEL_STATE + addr_word}, "latest"])
    prev  = rpc.call("eth_call", [{"to": to, "data": SEL_PREVWORK}, "latest"])
    tgt   = rpc.call("eth_call", [{"to": to, "data": SEL_TARGETFOR + addr_word}, "latest"])
    tm    = rpc.call("eth_call", [{"to": to, "data": SEL_TOTALMINTED}, "latest"])
    gas   = rpc.call("eth_gasPrice", [])
    txn   = rpc.call("eth_getTransactionCount", [ADDR, "pending"])
    head  = rpc.call("eth_blockNumber", [])

    # Parse state tuple. From your test: state has 8 fields, price at idx 1, anchorBlock at idx 7
    s = state[2:]
    def word(k): return s[k*64:(k+1)*64]
    price        = int(word(1), 16)
    anchor_block = int(word(7), 16)
    if anchor_block == 0:
        # fallback: use latest block minus small offset
        anchor_block = int(head, 16) - 1

    # fetch blockhash for anchor
    blk = rpc.call("eth_getBlockByNumber", [hex(anchor_block), False])
    anchor_hash = blk["hash"]

    return {
        "prev": prev, "target": tgt, "price": price,
        "anchor_block": anchor_block, "anchor_hash": anchor_hash,
        "gas_price": int(gas, 16), "tx_nonce": int(txn, 16), "head": int(head, 16),
        "total_minted": int(tm, 16),
    }


def get_balance():
    return int(rpc.call("eth_getBalance", [ADDR, "latest"]), 16)


# ---------------- verification (recompute hash on CPU) ----------------
def local_hash(anchor_hash: str, prev: str, miner: str, nonce_hex: str) -> int:
    def w(x):
        x = x[2:] if x.startswith("0x") else x
        return bytes.fromhex(x.rjust(64, "0"))
    data = w(anchor_hash) + w(prev) + w(miner) + w(nonce_hex)
    return int.from_bytes(Web3.keccak(data), "big")


# ---------------- tx ----------------
TX_NONCE = None
GAS_LIMIT = 400_000


def refresh_tx_nonce():
    global TX_NONCE
    TX_NONCE = int(rpc.call("eth_getTransactionCount", [ADDR, "pending"]), 16)


def _broadcast(raw_hex):
    import concurrent.futures as _cf
    errs = []
    def send(u):
        r = requests.post(u, json={"jsonrpc":"2.0","id":1,"method":"eth_sendRawTransaction","params":[raw_hex]}, timeout=6)
        j = r.json()
        if j.get("result"): return (j["result"], u)
        raise RuntimeError(str(j.get("error"))[:80])
    with _cf.ThreadPoolExecutor(max_workers=len(RPC_URLS)) as ex:
        futs = {ex.submit(send, u): u for u in RPC_URLS}
        for f in _cf.as_completed(futs):
            try: return f.result()
            except Exception as e: errs.append(str(e))
    for e in errs:
        if "already known" in e.lower():
            return ("0x" + Web3.keccak(hexstr=raw_hex).hex().replace("0x",""), "already-known")
    raise RuntimeError("; ".join(errs)[:200])


def submit_mine(anchor_block: int, nonce_hex: str, max_price: int, value: int, gas_price: int) -> str:
    global TX_NONCE
    if TX_NONCE is None: refresh_tx_nonce()

    # ABI-encode: MINE_SELECTOR + anchor_block(32) + nonce(32) + max_price(32)
    def w32(x):
        if isinstance(x, int): return x.to_bytes(32, "big").hex()
        x = x[2:] if x.startswith("0x") else x
        return x.rjust(64, "0")
    data = MINE_SELECTOR + w32(anchor_block) + w32(nonce_hex) + w32(max_price)

    for attempt in range(3):
        tx = {
            "to": CONTRACT, "from": ADDR, "value": value, "data": data,
            "chainId": CHAIN_ID, "nonce": TX_NONCE, "gas": GAS_LIMIT,
            "maxFeePerGas": max(gas_price * 2, Web3.to_wei(0.001, "gwei")),
            "maxPriorityFeePerGas": 0, "type": 2,
        }
        signed = acct.sign_transaction(tx)
        raw = "0x" + signed.raw_transaction.hex().replace("0x", "")
        try:
            h, via = _broadcast(raw)
            TX_NONCE += 1
            log(f"[TX] sent {h} value={value/1e18:.6f} nonce={tx['nonce']} via {via.split('//')[-1][:24]}")
            return h
        except Exception as e:
            m = str(e).lower()
            if "nonce" in m or "already known" in m:
                log(f"[TX] nonce issue: {str(e)[:80]}, refetching")
                refresh_tx_nonce(); continue
            raise
    raise RuntimeError("submit: 3 nonce conflicts")


# ---------------- CUDA process ----------------
class Cuda:
    def __init__(self):
        self.q: Queue = Queue(); self.proc = None; self.start()

    def start(self):
        env = dict(os.environ, MINER_PRIVATE_KEY=PRIVATE_KEY)
        self.proc = subprocess.Popen([CUDA_BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
        log(f"[cuda] pid={self.proc.pid}")

    def _reader(self, p):
        for line in p.stdout: self.q.put(line.rstrip("\n"))
        self.q.put(None)

    def send(self, line):
        try: self.proc.stdin.write(line + "\n"); self.proc.stdin.flush()
        except Exception as e: log(f"[cuda] write fail: {e}")

    def restart(self):
        try: self.proc.kill()
        except: pass
        time.sleep(2); self.start()


def bits_of(target_int: int) -> int:
    return 256 - target_int.bit_length()


def main():
    log(f"[*] Unicred miner {ADDR}")
    log(f"[*] rpcs {RPC_URLS}")

    try:
        bal = get_balance()
        log(f"[*] balance {bal/1e18:.5f} ETH")
        if bal < int(0.02e18):
            log("[!] balance < 0.02 ETH — mints will fail")
    except Exception as e:
        log(f"[!] balance err: {e}")

    if not os.path.exists(CUDA_BIN):
        log(f"[!] no binary at {CUDA_BIN}"); sys.exit(1)

    refresh_tx_nonce()
    cuda = Cuda()

    state = None
    last_refresh = 0
    counters = {"mined": 0, "fails": 0}
    clock = threading.Lock()

    # nonce_high — 24 leading bytes of nonce, fixed per session; we vary low 8 bytes only
    session_nonce_hi = int.from_bytes(os.urandom(24), "big")

    def send_job(st):
        # Full nonce template = session_nonce_hi (24 bytes BE) || low 8 bytes (varied by GPU)
        # We build a "session nonce" placeholder — GPU varies only low 8 bytes,
        # so upper 24 bytes must go into the block1/2 template.
        # For CUDA build_job, message layout uses these bytes at positions 128..151 of message.
        # Since GPU currently varies only lane 2 of block2 (=bytes 152..159 of message = LOW 8 nonce bytes),
        # the upper 24 nonce bytes MUST be embedded in build_job before absorbing block 1.
        # We pass nonce as target-format hex to build_job's target arg... no wait, we need a different way.
        #
        # Simplification: pass nonce_high=0 for now. GPU varies full 8-byte space (2^64), 
        # more than enough for 34-bit target. If we hit collisions across GPUs, later add hi_word.
        anchor_hex = st["anchor_hash"][2:]
        prev_hex   = st["prev"][2:]
        miner_hex  = ADDR[2:]
        target_hex = st["target"][2:]
        cuda.send(f"JOB {anchor_hex} {prev_hex} {miner_hex} {target_hex}")

    def submit_worker(nonce_hex, st):
        try:
            # maxPrice = 2x current price for safety
            max_price = st["price"] * 2
            txh = submit_mine(st["anchor_block"], nonce_hex, max_price, st["price"], st["gas_price"])
            log(f"[TX] {txh} — waiting receipt")
            # wait receipt via w3
            w3 = Web3(Web3.HTTPProvider(rpc.url, request_kwargs={"timeout": 60}))
            rc = w3.eth.wait_for_transaction_receipt(txh, timeout=180)
            with clock:
                if rc["status"] == 1:
                    counters["mined"] += 1
                    log(f"[OK] UNICRED MINTED! total: {counters['mined']} block={rc['blockNumber']}")
                else:
                    counters["fails"] += 1
                    log(f"[FAIL] tx reverted block={rc['blockNumber']}")
        except Exception as e:
            with clock: counters["fails"] += 1
            log(f"[FAIL] submit: {str(e)[:200]}")

    while True:
        now = time.time()
        if now - last_refresh >= REFRESH_SEC:
            last_refresh = now
            try:
                st = read_state()
                prev_changed = state is None or st["prev"] != state["prev"]
                anchor_changed = state is None or st["anchor_block"] != state["anchor_block"]
                target_changed = state is not None and st["target"] != state["target"]

                if prev_changed or anchor_changed or target_changed:
                    tbits = bits_of(int(st["target"], 16))
                    log(f"[STATE] prev/anchor/target refresh | minted={st['total_minted']}/4444 | "
                        f"bits={tbits} | price={st['price']/1e18:.5f} ETH | anchor_block={st['anchor_block']}")
                    state = st
                    send_job(st)
                else:
                    state["price"] = st["price"]; state["gas_price"] = st["gas_price"]; state["tx_nonce"] = st["tx_nonce"]
                    state["total_minted"] = st["total_minted"]
            except Exception as e:
                log(f"[!] state err: {str(e)[:100]}")

        try: line = cuda.q.get(timeout=0.3)
        except Empty: continue

        if line is None:
            log("[!] cuda died"); cuda.restart()
            if state: send_job(state)
            continue

        if line.startswith("RATE"):
            rate = float(line.split()[1])
            b = bits_of(int(state["target"], 16)) if state else 0
            eta = (2**b) / rate / 60 if rate and b else 0
            log(f"[RATE] {rate/1e9:.2f} GH/s | bits={b} | eta ~{eta:.1f} min | mined={counters['mined']} fails={counters['fails']}")
        elif line.startswith("FOUND"):
            _, nonce_lo_hex, hash_hex = line.split()
            # nonce_lo_hex is 16 hex chars = 8 bytes (the varied low part)
            # full nonce = 24 zero bytes + this 8 = uint256 with only low 8 bytes set
            nonce_full_hex = "0" * 48 + nonce_lo_hex
            if state is None: continue

            # verify on CPU
            lh = local_hash(state["anchor_hash"], state["prev"], ADDR, nonce_full_hex)
            if f"{lh:064x}" != hash_hex:
                log(f"[!] hash mismatch: gpu={hash_hex[:16]}... cpu={lh:016x}...")
                continue
            tgt = int(state["target"], 16)
            if lh >= tgt:
                log(f"[!] hash not below target — skip")
                continue

            log(f"[FOUND] nonce=0x{nonce_full_hex} bits={bits_of(lh)}")
            threading.Thread(target=submit_worker, args=(nonce_full_hex, dict(state)), daemon=True).start()

        elif line.startswith("INFO") or line.startswith("ERR"):
            log("[cuda]", line)


if __name__ == "__main__":
    main()
