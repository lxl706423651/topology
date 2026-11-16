# F/Reconvergence Multi-Scale Experiment

Generated at: 2026-06-14T21:38:00+08:00

VM: `192.168.122.60`

Run root: `bird/runs/F_reconf_multiscale_20260614_094101`

## Timing Table

| Scale | Services | Build(s) | Up(s) | Start-BIRD(s) | RIB stable wait(s) | BIRD check | Start-kernel/F(s) | F commands | Kernel check | Reconf wait(s) | Reconf write(s) | Reconf commands | Reconf check | Success |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 2599 | 2601 | 125.355 | 2284.493 | 350.413 | 672.252 | 10/10 3.068s | 287.089 | 4142556 | 10/10 5.490s | 140.017 | 88.079 | 128477 | 10/10 0.266s | yes |
| 3083 | 3085 | 182.336 | 3160.299 | 337.945 | 300.207 | 10/10 3.383s | 579.296 | 6131994 | 10/10 3.004s | 143.579 | 225.951 | 198830 | 10/10 0.281s | yes |
| 4192 | 4194 | 223.800 | 6464.404 | 710.797 | failed: load >1000 / SSH timeout | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | no |
| 4954 | 4956 | 281.132 | 10150.515 | 1680.201 | failed: load >1400 / SSH timeout | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | no |

## Artifacts

- `2599`: `/data/lxl/seed-emulator/topology/bird/runs/F_reconf_multiscale_20260614_094101/scale_2599`; main `bird/runs/F_reconf_multiscale_20260614_094101/scale_2599/main_summary_merged.json`; reconvergence `bird/runs/F_reconf_multiscale_20260614_094101/scale_2599/reconvergence/summary.json`
- `3083`: `bird/runs/F_reconf_multiscale_20260614_094101/scale_3083`; main `bird/runs/F_reconf_multiscale_20260614_094101/scale_3083/main_summary.json`; reconvergence `bird/runs/F_reconf_multiscale_20260614_094101/scale_3083/reconvergence/summary.json`
- `4192`: `bird/runs/F_reconf_multiscale_20260614_094101/scale_4192`; main `bird/runs/F_reconf_multiscale_20260614_094101/scale_4192/main_summary_failed_highload_20260614_173810.json`; reconvergence `None`
- `4954`: `bird/runs/F_reconf_multiscale_20260614_094101/scale_4954`; main `bird/runs/F_reconf_multiscale_20260614_094101/scale_4954/main_summary_failed_highload_20260614_211250.json`; reconvergence `None`

## Check Coverage

| Scale | Post Start-BIRD Check | Post Start-kernel/F Check | Post Reconvergence Check |
|---:|:---:|:---:|:---:|
| 2599 | 10/10 passed | 10/10 passed | 10/10 passed |
| 3083 | 10/10 passed | 10/10 passed | 10/10 passed |
| 4192 | not reached: VM high load/SSH timeout before check | not reached | not reached |
| 4954 | not reached: VM high load/SSH timeout before check | not reached | not reached |

## Notes

- `Start-kernel/F` is the host-setns writer stage that writes BIRD-selected routes to main FIB. By default it uses the old F fast path (`birdc show route primary` -> `ip -batch`) for the initial full FIB sync.
- `Reconf write` excludes the RIB/load stable wait and measures only the delta FIB update.
- Each scale records 10-brdnode checks in `validate_bird`, `validate`, and reconvergence `validate` stages.
- `4192` and `4954` completed build/up/start-BIRD, but did not reach BIRD check, initial F write, kernel check, or reconvergence. Both runs failed while waiting for post-start-BIRD load/RIB stability because the VM became SSH-unresponsive under very high control-plane load.
- After the `4954` failure, the VM stayed SSH-unresponsive for roughly 30 minutes from the first high-load sample. It was recovered with `virsh destroy bird-rtnl-dev`, `virsh start bird-rtnl-dev`, then `/home/lxl/bird/reset_docker_vm.sh`.
- Post-reset VM verification: containers `0`, running containers `0`, volumes `0`, custom Docker networks `0`; `handsonsecurity/seedemu-base:{2.0,latest}` and `handsonsecurity/seedemu-router:{2.0,latest}` were rebuilt and available.
