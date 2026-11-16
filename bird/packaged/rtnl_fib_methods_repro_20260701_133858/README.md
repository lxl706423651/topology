# RTNL/FIB Methods Reproduction Package

This package contains the four Linux-main-FIB experiment methods used in the
current SeedEMU/BIRD RTNL study:

- `F`: stock-kernel host writer. Initial full FIB write uses controlled
  `setns + ip -batch`; reconvergence uses delta route diff.
- `K1`: private `BIRD_RT_BULK` API under global RTNL.
- `K2c`: K1 plus per-netns route lock, no-notify, and `fib_info_hash_mutex`.
- `K3c`: K2c plus cached-`fib_info` grouped bulk fast path.

The comparison/design document is `0626kernelDesign.md`.

## 1. Layout

```text
README.md
0626kernelDesign.md
1buildFastDockerImages.py
F/
K1/
K2c/
K3c/
```

Each method directory is self-contained for running from that directory. The
same scripts also exist under each method's `scripts/` directory as a copy, but
the recommended entrypoint is the method directory root, for example:

```bash
cd K3c
./run_3083_k3c_delta_experiment.sh
```

This matters because the runners use paths relative to `SCRIPT_DIR`, such as
`./kernel/tools` and `../1buildFastDockerImages.py`.

## 2. Common Variables

Set these for every experiment:

```bash
export VM_HOST=192.168.122.60
export VM_USER=lxl
export VM_DOMAIN=bird-rtnl-dev
export OUTPUT_DIR=/absolute/path/to/compiled/seedemu/output
export PROJECT_NAME=topology-3083
```

Switching topology scale only requires changing:

```bash
export OUTPUT_DIR=/path/to/another/output
export PROJECT_NAME=topology-1897   # choose a unique compose project name
```

The `OUTPUT_DIR` must contain the compiled SeedEMU `docker-compose.yml` and
node build contexts. The scripts sync this output to the VM and build/start
containers there.

Use `PROJECT_NAME` to avoid collisions with old Docker Compose projects. If
you reuse a project name, reset the VM/Docker first.

## 3. VM And Docker Preparation

The scripts assume:

- The host can run `virsh destroy/start ${VM_DOMAIN}`.
- The host can SSH to `${VM_USER}@${VM_HOST}`.
- The VM user has passwordless `sudo`.
- Docker, Docker Compose, and buildx are installed in the VM.
- `/home/${VM_USER}/bird/reset_docker_vm.sh` exists in the VM.

If this is a new machine, first copy `reset_docker_vm.sh` to the VM:

```bash
rsync -a F/reset_docker_vm.sh ${VM_USER}@${VM_HOST}:/home/${VM_USER}/bird/reset_docker_vm.sh
ssh ${VM_USER}@${VM_HOST} 'chmod +x /home/${USER}/bird/reset_docker_vm.sh'
```

For a clean start:

```bash
virsh destroy ${VM_DOMAIN} || true
virsh start ${VM_DOMAIN}
ssh ${VM_USER}@${VM_HOST} '/home/lxl/bird/reset_docker_vm.sh --skip-rebuild-seedemu-images'
```

If the host already has SeedEMU base images, import them into the VM:

```bash
docker save handsonsecurity/seedemu-base:2.0 handsonsecurity/seedemu-router:2.0 ubuntu:24.04 \
  | ssh ${VM_USER}@${VM_HOST} 'sudo -n docker load'
ssh ${VM_USER}@${VM_HOST} \
  'sudo -n docker tag handsonsecurity/seedemu-base:2.0 handsonsecurity/seedemu-base:latest && sudo -n docker tag handsonsecurity/seedemu-router:2.0 handsonsecurity/seedemu-router:latest'
```

## 4. Installing Prebuilt Kernels

K1/K2c/K3c include prebuilt kernel/module tarballs:

```text
K1/kernel/prebuilt/6.8.12-birdrtbulk-k1-global-rtnl.tar.gz
K2c/kernel/prebuilt/6.8.12-birdrtbulk-k2c.tar.gz
K3c/kernel/prebuilt/6.8.12-birdrtbulk-k3c.tar.gz
```

Install one on the VM like this:

```bash
# Example: K3c
KERNEL=6.8.12-birdrtbulk-k3c
TARBALL=K3c/kernel/prebuilt/6.8.12-birdrtbulk-k3c.tar.gz
rsync -a "${TARBALL}" ${VM_USER}@${VM_HOST}:/tmp/
ssh ${VM_USER}@${VM_HOST} "sudo -n tar -C / -xzf /tmp/$(basename "${TARBALL}") && sudo -n update-initramfs -c -k ${KERNEL} && sudo -n update-grub"
```

Then boot into it. The runners can do this through GRUB if the entry is
resolvable. If automatic GRUB detection fails, pass `K1_GRUB_ENTRY_ID`.

Check the running kernel:

```bash
ssh ${VM_USER}@${VM_HOST} 'uname -r'
```

Expected names:

```text
K1:  6.8.12-birdrtbulk
K2c: 6.8.12-birdrtbulk-k2c
K3c: 6.8.12-birdrtbulk-k3c
```

## 5. Method F

F does not require a modified kernel.

Run in phases:

```bash
cd F

# 1. Reset VM/Docker.
IMPORT_SEEDEMU_IMAGES_FROM_HOST=1 ./run_f_real_main_fib_vm.sh reset

# 2. Sync OUTPUT_DIR to VM, build images, and docker compose up.
./run_f_real_main_fib_vm.sh run

# 3. Start BIRD only. This does not rewrite kernel.conf to export all.
START_BIRD_JOBS=1 START_BIRD_DELAY=0.1 ./run_f_real_main_fib_vm.sh start-bird

# 4. Initial full RIB-to-FIB write using F.
F_SHARDS=8 REAL_DUMP_WORKERS=16 LIMIT_CONTAINERS=0 ./run_f_real_main_fib_vm.sh writeFIB_F

# 5. Reconvergence with delta routes.
DOWN_COUNT=10 DUMP_WORKERS=32 FIB_DUMP_WORKERS=32 F_SHARDS=8 ./reconfergence_delta_F.sh run
```

`LIMIT_CONTAINERS=0` means no artificial limit. Set it only for small smoke
tests, for example `LIMIT_CONTAINERS=20`.

## 6. K1

K1 uses the private bulk API but keeps global RTNL.

Fast full workflow with delta reconvergence:

```bash
cd K1
RESET_VM=1 \
OUTPUT_DIR="${OUTPUT_DIR}" \
PROJECT_NAME="${PROJECT_NAME}" \
INITIAL_WRITE_JOBS=1 \
K1_DUMP_JOBS=16 \
K1_BATCH=256 \
DELTA_WRITE_JOBS=1 \
OFFSET_JOBS=16 \
K1_DELTA_BATCH=256 \
RUN_FULL_COMPARISON=0 \
./run_3083_k1_delta_reconvergence.sh
```

Manual initial phases:

```bash
cd K1
K1_KERNEL=6.8.12-birdrtbulk ./run_3083_k1_bulk_real.sh prepare-k1-reset
./run_3083_k1_bulk_real.sh run
START_BIRD_JOBS=1 START_BIRD_DELAY=0.1 ./run_3083_k1_bulk_real.sh start-bird
./run_3083_k1_bulk_real.sh verify-start
K1_JOBS=1 K1_DUMP_JOBS=16 K1_BATCH=256 ./run_3083_k1_bulk_real.sh writeK1
./run_3083_k1_bulk_real.sh verify-fib
```

Manual delta reconvergence after initial FIB verification:

```bash
DELTA_JOBS=16 DELTA_IGNORE_INITIAL_FEED=1 ./install_bird_delta_vm.sh enable-delta
WRITE_BACKEND=k1-bulk \
WRITE_JOBS=1 \
K1_DELTA_JOBS=1 \
K1_DELTA_BATCH=256 \
OFFSET_JOBS=16 \
LOAD_ONLY_STABLE=1 \
STABLE_SECONDS=0 \
DOWN_COUNT=10 \
./reconvergence_bird_delta.sh run
```

## 7. K2c

K2c uses the same userspace writers as K1, but the VM must boot
`6.8.12-birdrtbulk-k2c`.

Recommended one-command run:

```bash
cd K2c
RESET_VM=1 \
OUTPUT_DIR="${OUTPUT_DIR}" \
PROJECT_NAME="${PROJECT_NAME}" \
RUN_FULL_COMPARISON=0 \
./run_3083_k2c_delta_optimized.sh
```

Manual initial phases:

```bash
cd K2c
K1_KERNEL=6.8.12-birdrtbulk-k2c ./run_3083_k1_bulk_real.sh prepare-k1-reset
./run_3083_k1_bulk_real.sh run
START_BIRD_JOBS=1 START_BIRD_DELAY=0.1 ./run_3083_k1_bulk_real.sh start-bird
./run_3083_k1_bulk_real.sh verify-start
K1_JOBS=8 K1_DUMP_JOBS=16 K1_BATCH=2000 ./run_3083_k1_bulk_real.sh writeK1
./run_3083_k1_bulk_real.sh verify-fib
```

Manual delta reconvergence:

```bash
DELTA_JOBS=16 DELTA_IGNORE_INITIAL_FEED=1 ./install_bird_delta_vm.sh enable-delta
WRITE_BACKEND=k1-bulk \
K1_DELTA_MIXED_BATCHES=1 \
WRITE_JOBS=1 \
K1_DELTA_JOBS=1 \
K1_DELTA_BATCH=2000 \
OFFSET_JOBS=16 \
LOAD_ONLY_STABLE=1 \
STABLE_SECONDS=0 \
DOWN_COUNT=10 \
./reconvergence_bird_delta.sh run
```

## 8. K3c

K3c is the fastest current controlled kernel prototype. It requires
`6.8.12-birdrtbulk-k3c`.

Recommended one-command run:

```bash
cd K3c
RESET_VM=1 \
OUTPUT_DIR="${OUTPUT_DIR}" \
PROJECT_NAME="${PROJECT_NAME}" \
INITIAL_WRITE_JOBS=16 \
K1_DUMP_JOBS=16 \
K1_BATCH=16000 \
K1_OPS_PER_ATTR=2000 \
DELTA_WRITE_JOBS=8 \
OFFSET_JOBS=16 \
K1_DELTA_BATCH=16000 \
RUN_FULL_COMPARISON=0 \
./run_3083_k3c_delta_experiment.sh
```

If the VM is already booted into K3c and Docker state is already prepared, use:

```bash
RESET_VM=0 ./run_3083_k3c_delta_experiment.sh
```

Manual initial phases are possible through the K1 phase runner while targeting
the K3c kernel:

```bash
cd K3c
K1_KERNEL=6.8.12-birdrtbulk-k3c ./run_3083_k1_bulk_real.sh prepare-k1-reset
./run_3083_k1_bulk_real.sh run
START_BIRD_JOBS=1 START_BIRD_DELAY=0.1 ./run_3083_k1_bulk_real.sh start-bird
./run_3083_k1_bulk_real.sh verify-start
K1_JOBS=16 K1_DUMP_JOBS=16 K1_BATCH=16000 ./run_3083_k1_bulk_real.sh writeK1
./run_3083_k1_bulk_real.sh verify-fib
```

For the exact K3c full + delta timing path, prefer the one-command runner above
because it passes the grouped-bulk settings used in the documented experiment.

K3c also includes:

```text
kernel/k3c_overlay/
```

This contains the final touched kernel files for reference. K3c currently has
prebuilt artifacts and an overlay, not a clean standalone patch file. K3c builds
on the K1 -> K2c patch sequence plus the cached-`fib_info` changes described in
`0626kernelDesign.md`.

## 9. Kernel Rebuild Notes

For K1:

```bash
cd linux-source-6.8.0-124.124
git apply /path/to/K1/kernel/patches/0001-bird-rtbulk-global-rtnl.patch
scripts/config --set-str CONFIG_LOCALVERSION -birdrtbulk
make olddefconfig
make -j"$(nproc)" bzImage modules
```

For K2c, apply patches `0001` through `0004` in order and use
`CONFIG_LOCALVERSION=-birdrtbulk-k2c`.

For K3c, the fastest path is to install `kernel/prebuilt/6.8.12-birdrtbulk-k3c.tar.gz`.
If rebuilding, apply K1-K2c patches first, then port the files in
`K3c/kernel/k3c_overlay/` onto the same Ubuntu 6.8.0-124.124 source tree and
use `CONFIG_LOCALVERSION=-birdrtbulk-k3c`.

Build logs are included under each method's `kernel/prebuilt/` directory.

## 10. Output And Summaries

Local summaries are written under each method's `runs/` directory. Important
files:

```text
main_summary.json
verify_initial_fib.json
reconvergence/summary.json
comparison_summary.json
```

VM summaries are usually under:

```text
/home/lxl/bird/runs/<run-name>
```

Use status commands:

```bash
./run_f_real_main_fib_vm.sh status
./run_3083_k1_bulk_real.sh status
./reconvergence_bird_delta.sh status
```

## 11. Notes On Semantics

- `restart_bird` in JSON summaries is a historical stage name. It means
  "start or restart BIRD processes inside router containers", not VM/container
  reset.
- K2c and K3c are no-notify controlled prototypes. They do not emit normal
  route monitor events for the bulk fast path.
- Reconvergence in this package should use delta route mode. Full
  post-reconvergence writes are only for comparison and are disabled by
  `RUN_FULL_COMPARISON=0`.
- For another topology scale, do not edit scripts. Set `OUTPUT_DIR` and
  `PROJECT_NAME`.
