# Start-BIRD A/B Experiment Notes

## 2026-06-15 4192-Scale Diagnosis

Goal: isolate why 4192/4954 scale runs hit very high load or SSH timeout around `restart_bird`.

Topology source:

- `bird/runs/F_reconf_multiscale_20260614_094101/scale_4192/output`
- Compose service count: 4194
- `seedemu_internet_map` skipped
- Router/container target: 4192

## Results

| Case | Image | Start method | Result | Key timing | Evidence |
| --- | --- | --- | --- | --- | --- |
| 1 | stock BIRD from PPA | start-only via `nsenter ... bird` | PASS | build `202.696s`, up `7151.448s`, start-bird `7099.132s`, RIB stable `275.247s` | `bird/runs/start-bird-ab-4194_20260615_123054_ab4192-stock-start/summary.json` |
| 2 | stock BIRD from PPA | full-restart via `install_bird_delta_vm.sh restart-bird`, jobs=2 | FAIL after restart | build `248.482s`, up `7431.238s`, restart `760.365s`; then SSH banner timeout while reading loadavg | `bird/runs/start-bird-ab-4194_20260615_165816_ab4192-stock-restart/summary.json` |
| 3a | patched BIRD | start-only via bare `bird` | INVALID/FAIL | stopped after about 800 triggers | bare `bird` used `/usr/local/var/run/bird.ctl` and exited: `Cannot create control socket ... No such file or directory` |
| 3b | patched BIRD | start-only via `bird -s /run/bird/bird.ctl -c /etc/bird/bird.conf` | PASS | patched build `222.398s`, up `7344.319s`, explicit start-bird `6859.322s`, RIB stable `197.643s` | `bird/runs/start-bird-ab-4194_20260615_192741_ab4192-patched-start/start_only_explicit_summary.json`, `explicit_wait_validate_summary.json` |

## Interpretation

The main high-load/SSH-timeout problem is not caused by the patched BIRD binary itself. It is reproducible with stock BIRD when using the full-restart path.

The old start-only path is slow because it serially enters 4192 namespaces and starts one BIRD at a time, but it does not overload the VM. Stock start-only passed, and patched start-only also passed once the control socket path was explicit.

The full-restart path is the dangerous path at 4192 scale. In the stock full-restart run, `restart-bird` processed 3825 brdnodes and returned success, then the next SSH command to read `/proc/loadavg` failed 5 times with `Connection timed out during banner exchange`.

Patched BIRD adds one separate pitfall: because it was built from source, the bare `bird` command defaults to `/usr/local/var/run/bird.ctl`, unlike the packaged BIRD workflow. Any patched-BIRD start script must use:

```bash
mkdir -p /run/bird
bird -s /run/bird/bird.ctl -c /etc/bird/bird.conf
```

## Operational Notes

- Do not use `docker compose down` on thousands of already-converged containers as a routine transition. During one attempted setup cleanup, `compose down` pushed VM load above 2000 while slowly removing containers. Use the VM reset cycle instead when starting a fresh 4192-scale experiment.
- The start-only trigger delay `0.02s` is not the dominant cost. The dominant cost is the per-container `nsenter`/Docker interaction, roughly several seconds per small batch at this scale.
- For future patched-BIRD experiments, either set `--bird-start-command 'mkdir -p /run/bird; bird -s /run/bird/bird.ctl -c /etc/bird/bird.conf'` or make that the patched-image default.

## 2026-07-02 IDC 3083 K1 Restart-BIRD Recovery Note

The IDC bare-metal K1 run reproduced the same dangerous shape at 3083 scale:
`install_bird_delta_vm.sh restart-bird` processed thousands of brdnodes and then
the next load check failed through SSH banner timeout. The host still had many
BIRD processes in `D` state and load1 above 1400 before the manual power reset.

The root cause is not only that `RESTART_DELAY=0.1` is small. The risky
combination is full restart of every container BIRD process, no load gate during
the serial loop, Docker/SSH command-entry overhead, and post-stage load sampling
through `ssh seed@127.0.0.1`. A fixed delay cannot adapt to BGP convergence
backlog. For IDC runs, use serial restart with a periodic local load gate, for
example `RESTART_LOAD_GATE_EVERY=50 RESTART_LOAD_THRESHOLD=50`, and read
`/proc/loadavg` locally when the runner is already executing on the target host.

Entry-method guidance is environment-specific. The 2026-06-10 VM benchmark found
parallel `docker exec` faster than repeated `sudo nsenter`, but that result must
not be assumed for the IDC bare metal. After the next successful 3083 up/start,
run `entry_method_benchmark.py` on the IDC project and record the real-machine
`docker exec` versus `nsenter` numbers before changing the default entry path.
