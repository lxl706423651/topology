# bird-rtnl-dev VM

This VM is prepared for BIRD/RTNL experiments.

## Spec

- Name: `bird-rtnl-dev`
- OS image: Ubuntu 24.04 Noble cloud image
- CPU: 30 vCPU
- Memory: 200 GiB (`204800` MiB / `209715200` KiB)
- Disk: 300 GiB qcow2
- Network: libvirt `default` NAT network
- User: `lxl`
- SSH key: host `~/.ssh/id_ed25519.pub`

## Files

- Base image: `bird/vm/images/noble-server-cloudimg-amd64.img`
- VM disk: `bird/vm/bird-rtnl-dev.qcow2`
- Cloud-init seed ISO: `bird/vm/bird-rtnl-dev-seed.iso`
- Cloud-init user-data: `bird/vm/user-data`
- Cloud-init meta-data: `bird/vm/meta-data`

## Access

Current DHCP address:

```bash
ssh lxl@192.168.122.60
```

If the DHCP lease changes:

```bash
virsh domifaddr bird-rtnl-dev --source lease
```

## Lifecycle

```bash
virsh start bird-rtnl-dev
virsh shutdown bird-rtnl-dev
virsh destroy bird-rtnl-dev
virsh dominfo bird-rtnl-dev
```

The VM is persistent but autostart is disabled.

## Docker

The VM has the same Docker component versions as the host at setup time:

- Docker Engine/CLI: `29.0.0`
- containerd: `2.1.5`
- Docker Compose plugin: `2.40.3`
- Docker Buildx plugin: `0.29.1`

The packages are held with `apt-mark hold` to avoid accidental upgrades.

Docker daemon config:

```bash
cat /etc/docker/daemon.json
```

It includes the configured registry mirrors, BuildKit support, `overlay2`, and the systemd cgroup driver.

## SeedEMU Docker Runner

The helper script is available in this repo at:

```bash
bird/run.py
```

The end-to-end helper that also starts BIRD and writes `summary.json` is:

```bash
bird/run_all.py
```

It has also been copied to the VM:

```bash
/home/lxl/bird/run.py
/home/lxl/bird/run_all.py
```

Usage inside the VM:

```bash
/home/lxl/bird/run.py /path/to/seedemu/output
/home/lxl/bird/run_all.py /path/to/seedemu/output
```

The runner's build phase follows the logic from
`/home/lxl/seed-emulator/topology/buildTest1.sh`: warm up the first Compose
service, then run batched parallel `docker compose build` jobs with default
batch size `50` and build parallelism `16`.

The runner's up phase follows the logic from
`/home/lxl/seed-emulator/topology/dockerUp.sh`: fetch Compose services and run
batched parallel `docker compose up -d` jobs with default batch size `50` and up
parallelism `4`.

For the BIRD/RTNL experiment, both runners intentionally skip
`seedemu_internet_map` during build/up. SeedEMU generates that service with the
fixed global name `seedemu_internet_map`; stale stopped containers from older
runs can otherwise block a new `docker compose up`. The map container is not
needed for the router convergence experiment.

From the host, the same runner can now synchronize an output directory and
invoke the VM runner automatically:

```bash
python3 bird/run.py output/ --vm 192.168.122.60
python3 bird/run_all.py output/ --vm 192.168.122.60
```

In `--vm` mode, the host-side script copies itself to `/home/lxl/bird/run.py`
or `/home/lxl/bird/run_all.py`, creates a per-run directory under:

```bash
/home/lxl/bird/runs/topology-<service-count>_<YYYYmmdd_HHMMSS>_<project>/
```

It then saves a copy of the runner and `run_manifest.json` in that directory,
syncs the local output directory to the directory's `output/` child, and runs
the per-run runner copy against that VM-local path.

Without `--vm`, `run.py` is local-only: it operates on the filesystem and Docker
daemon of the machine where it is executed.

`run_all.py` has the same locality rule, but it also runs the start-bird and
start-kernel phases after up. The start-kernel phase rewrites router
`/etc/bird/conf/kernel.conf` to an explicit `protocol kernel kernel1` with
`export all`, validates the full BIRD config with `bird -p -c
/etc/bird/bird.conf`, then applies it with `birdc configure`. It waits for
1-minute load average below `50.0` between router updates by default and writes
`<output>/summary.json` with build, up, start-bird, and start-kernel durations
and success status. In VM mode, the remote summary is copied back to the
host-side output directory.

The VM run directory for `run_all.py` is organized like this:

```bash
/home/lxl/bird/runs/topology-1899_20260609_081500_topology-291e63c1/
├── output/
│   └── run_logs/
├── logs/dockerBuild/
├── run_all.py
├── run_manifest.json
├── summary.json
└── container_pids.json
```

The Compose project label is the reliable way to inspect containers for one
run:

```bash
docker ps --filter label=com.docker.compose.project=topology-291e63c1
```

The VM Docker reset helper is:

```bash
/home/lxl/bird/reset_docker_vm.sh
```

It is destructive: it removes Docker/containerd runtime data and recreates
Docker objects, BuildKit/cache state, and `/etc/docker/daemon.json`. It verifies
the fresh Docker daemon has zero containers/images/volumes/custom networks
before rebuilding images. No arguments are equivalent to
`--yes --rebuild-seedemu-images --seedemu-root /home/lxl/seed-emulator`, so it
also rebuilds the SeedEMU base/router images by default.

Use `--skip-rebuild-seedemu-images` only when you want an empty Docker daemon
without rebuilding the SeedEMU base/router images.

For example, the A00 sample output was synchronized to the VM and started with:

```bash
/home/lxl/bird/run.py /home/lxl/seed-emulator/examples/basic/A00_simple_as/output
```

The script writes logs under the target output directory:

```bash
<output>/run_logs/
```

It also writes build/up logs compatible with the shell scripts under the parent
directory's `logs/dockerBuild/`.
