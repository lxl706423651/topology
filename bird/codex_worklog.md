## 2026-07-15 23:05 - Clean Upstream 7.1.3 Per-Netns RTNL 3083 Comparison

- User intent: continue the clean upstream 7.1/standard rtnetlink experiment and compare it with K3c.
- Scope: VM `bird-rtnl-dev`, project `topology-3083-upstream713`, scripts `bird/rtnl_solution_lab.py`, `bird/run_f_real_main_fib_vm.sh`, docs `bird/0715upstreamPernetCompare.md`, `bird/0713kernelCompare.md`.
- Changes:
  - Fixed `rtnl_solution_lab.py` so `--limit-containers` defaults to `0` instead of `32`.
  - Changed `run_f_real_main_fib_vm.sh` so it always passes `--limit-containers`, preventing accidental 32-container runs.
  - Added `bird/0715upstreamPernetCompare.md` and appended the clean upstream 7.1.3 comparison to `bird/0713kernelCompare.md`.
- Commands:
  - Booted VM with `7.1.3-upstream-pernet-rtnl`, a clean upstream-derived kernel with no `BIRD_RT_BULK`/K3c code.
  - Ran `bird/run_f_real_main_fib_vm.sh reset` with cached SeedEMU image import.
  - Ran `bird/run_f_real_main_fib_vm.sh run` for `/data/lxl/seed-emulator/topology/output_3083`.
  - Ran `START_BIRD_JOBS=1 START_BIRD_DELAY=0.1 bird/run_f_real_main_fib_vm.sh start-bird`.
  - Ran `bird/run_f_real_main_fib_vm.sh writeFIB_F`; first attempt exposed the old 32-container default, then the fixed full run used `LIMIT_CONTAINERS=0`.
  - Ran post-start 10-AS protocol verification and post-FIB per-AS route-count verification.
- Validation:
  - Build/load: `185.592s`.
  - Compose up: `2251.902s`.
  - Start BIRD: `2757/2757` brdnodes, about `1304s`.
  - Full standard F writer: `6,131,994` route ops in `408.185s`, `15,022 ops/s`, max load1 `15.507`.
  - Post-FIB deterministic AS verifier: `312/312` AS representatives passed in `116.384s`.
  - VM remained running with kernel `7.1.3-upstream-pernet-rtnl`; containers and FIB state were left in place for manual inspection.
- Notes:
  - Clean upstream per-netns RTNL standard route path is much slower than K3c on the same 3083-scale full write: `408.185s` vs K3c 7.1.3 `7.061s`.
  - The result supports the conclusion that upstream per-netns RTNL is useful but does not replace K3c's bulk submission and cached `fib_info` fast path.

## 2026-07-15 23:13 - Audit Clean 7.1.3 Versus F/K1/K2c/K3c

- User intent: compare the latest clean 7.1.3 full-FIB result with F, K1, K2c, and K3c; determine whether the 8-versus-16 worker difference explains the gap; document the performance and semantic tradeoffs.
- Scope: experiment summaries and Linux route-lock source were read only; created `bird/0715RTM.md`; no VM workload, reset, container change, or kernel change was performed.
- Changes:
  - Added the 3083-scale full-write table for F, clean 7.1.3, K1, K2c, K3c 6.8, and K3c 7.1.3.
  - Corrected the earlier interpretation of the clean 7.1.3 result: the source contains per-netns RTNL migration scaffolding, but its build has `CONFIG_DEBUG_NET_SMALL_RTNL` disabled, so `rtnl_net_lock(net)` still maps to global RTNL.
  - Documented that enabling the current debug option still nests the per-netns mutex inside global RTNL; it is a migration/correctness aid, not a completed scalability mode.
  - Separated worker-count effects from bulk submission, no-notify, per-NS route locking, and K3c cached-`fib_info` effects.
- Commands:
  - Used `rg` and `sed` on `bird/design.md`, `bird/0617Experiment.md`, `bird/0708infocom.md`, `bird/0713kernelCompare.md`, and the Linux 6.8/7.1.3 source trees to trace method semantics and lock behavior.
  - Used `jq` on the F/K1/K2c/K3c run summaries to verify route count, workers, batch sizes, dump/write wall times, and failure counts.
  - Used `awk` to recompute throughput and speedup ratios from `6,131,994` route ops.
  - Used read-only SSH to confirm the VM is running `7.1.3-upstream-pernet-rtnl`; `/boot/config-7.1.3-upstream-pernet-rtnl` was not installed in the VM, so the build-tree `.config` is the retained config evidence.
- Validation:
  - Confirmed clean F uses 8 combined dump/write workers; its `REAL_DUMP_WORKERS=16` argument is not used by `real_solution_f()`.
  - Confirmed K2c also uses 8 write workers and is `5.74x` faster than clean 7.1.3, ruling out worker count as the primary cause.
  - Confirmed K3c 7.1.3 uses 16 workers and completes the same `6,131,994` ops in `7.061s`; even an idealized 2x clean-worker speedup would leave clean about `28.9x` slower.
  - `git diff --check -- bird/0715RTM.md` passed with no whitespace errors.
- Notes:
  - The 23:05 entry's phrase "clean upstream per-netns RTNL standard route path" is superseded by this source/config audit. The measured path is clean upstream standard rtnetlink with global RTNL; 7.1.3 has preparatory per-netns APIs but not completed cross-netns RTNL concurrency in this build.
  - No clean 7.1.3 `F_SHARDS=4/8/12/16` sweep was run. The `204.1s` 16-worker value in `0715RTM.md` is explicitly labeled as an idealized upper-bound calculation, not a measurement.
  - The repository's `.gitignore` contains a top-level `*` rule, so both Markdown files are present on disk but omitted from ordinary `git status` output.

## 2026-07-16 06:05 - Ubuntu 6.8 Upstream Per-Netns RTNL Backport Gate

- User intent: boot the RTNL VM and continue the upstream per-netns RTNL experiment with `start-bird timeout=3600s` and serial `start-kernel delay=0.2s, timeout=3600s`; only enter the 3083 baseline/export-all stages after smoke proves real cross-netns route-update concurrency.
- Scope: VM `bird-rtnl-dev`, Ubuntu 6.8 kernel source `/data/lxl/kernel-work/linux-source-6.8.0-124.124-pernet-v3`, upstream 2025-02-28 IPv4 FIB 12-patch series, `bird/pernet_rtnl_smoke.py`, `bird/0715RTM.md`.
- Changes:
  - Applied the full 12-patch series and required preparatory commits to Ubuntu 6.8, including IPv4 `fib_info` hash namespacification and `RTM_NEWROUTE`/`RTM_DELROUTE` per-netns handler registration.
  - Built and preserved both `6.8.12-ubuntu124-pernet-v3-debug` and final non-debug `6.8.12-ubuntu124-pernet-v3`; stock/recovery kernels were not removed.
  - Added the debug/production smoke comparison and the 3083 gate decision to `bird/0715RTM.md`; added machine-readable `gate_summary.json`.
- Commands:
  - Built final kernel with `make -j30 bzImage modules`; wall time `371.23s`.
  - Installed modules, regenerated initrd, updated libvirt direct-kernel paths, and cold-started the VM with `virsh destroy/start`.
  - Ran `pernet_rtnl_smoke.py --namespaces 8 --routes-per-namespace 5000` on debug and production builds.
- Validation:
  - Final VM kernel: `6.8.12-ubuntu124-pernet-v3`; Docker state remains 0 containers with cached SeedEMU base/router images present.
  - Debug build observed 40,000 `rtnl_net_lock` calls but max concurrent `fib_table_insert=1`.
  - Production build wrote all 40,000 routes correctly, with max concurrent `fib_table_insert=1`; no Oops, WARNING, refcount, lockdep, or other risk-log hit.
  - Source audit confirms debug mode nests the netns mutex under global RTNL, while the default helper falls back to global RTNL.
- Notes:
  - The real-concurrency gate failed, so the 3083 baseline and export-all runs were intentionally not started. Running them would measure another global-RTNL baseline and violate the agreed experiment gate.
  - The VM is left running, idle, and Docker-clean on the final 6.8 backport kernel for further inspection.

## 2026-07-16 06:25 - Clarify Global RTNL Status and References

- User intent: make the Markdown record explicitly state that the current standard `RTM_NEWROUTE` / `RTM_DELROUTE` path still uses global RTNL and has not obtained cross-netns concurrency; add references supporting the claim.
- Scope: documentation only; updated `bird/0715RTM.md` and this worklog. No VM, Docker, kernel, or experiment state was changed.
- Changes:
  - Added an explicit distinction between `RTNL_FLAG_DOIT_PERNET`, the `rtnl_net_lock(net)` migration interface, and actual removal of global RTNL.
  - Stated that both tested clean 7.1.3 and the Ubuntu 6.8 full-series backport remain globally serialized for the standard IPv4 route-update path.
  - Added upstream commit, netdev patch-cover, final patch, Kconfig, and local kprobe summaries as references.
- Validation:
  - Cross-checked the prose against upstream commit `76aed95319da`, the 2025-02-28 v3 12-patch cover/final patch, and both retained local smoke summaries.
  - The document does not claim that per-NS RTNL is impossible; it states precisely that the referenced upstream series alone does not complete or enable it.

## 2026-07-16 06:45 - Verify the Direct Post-Patch Kernel Version

- User intent: determine whether the earlier `v6.13-rc7` Kconfig citation predates and was superseded by the 2025-02-28 IPv4 FIB series; identify and inspect the kernel version directly containing that series.
- Scope: upstream source/release research and documentation only; no VM, Docker, kernel build, or experiment state was changed.
- Findings:
  - Compared Torvalds Linux `v6.14` and `v6.15`: the former lacks the IPv4 handler lock/flag conversion, while the latter contains both `rtnl_net_lock(net)` and `RTNL_FLAG_DOIT_PERNET`.
  - Confirmed final commit `1dd2af7963e9` is an ancestor of `v6.15`; Linux 6.15 is the first formal mainline release directly containing the series.
  - Confirmed Linux 6.15 retains the same Kconfig migration warning as the earlier prerequisite version.
  - Confirmed the actual Linux 6.15 function bodies: debug mode acquires global RTNL and then the namespace mutex; non-debug mode maps directly to global RTNL.
- Changes:
  - Added a 6.14/6.15 version-boundary section to `bird/0715RTM.md`.
  - Replaced the earlier Kconfig-only reference with direct Linux 6.15 release, handler, flag, Kconfig, and lock-helper source references.
- Conclusion:
  - The global-serialization result is not based on a stale pre-series Kconfig statement. It remains true in the first mainline release containing the patch series.

## 2026-07-16 08:37 - Verify Latest Mainline RTNL Migration Status

- User intent: cite Linux `v6.19-rc8` and `v7.2-rc3` and clarify whether current `rtnl_net_lock()` provides true per-network-namespace RTNL concurrency.
- Scope: documentation and upstream source inspection only; updated `bird/0715RTM.md` and this worklog. No VM, Docker, kernel, or experiment state was changed.
- Changes:
  - Added a latest-mainline subsection and the two requested Kconfig references.
  - Clarified that the `rtnl_net_lock()` API exists, but its global-RTNL-free per-NS locking semantics and scalability are not implemented yet.
  - Added direct `v7.2-rc3` links for both the debug implementation and non-debug fallback.
- Commands:
  - Fetched `v6.19-rc8/net/Kconfig.debug` and `v7.2-rc3/net/Kconfig.debug` from Torvalds' Linux repository and compared `CONFIG_DEBUG_NET_SMALL_RTNL`.
  - Fetched `v7.2-rc3/net/core/rtnetlink.c` and `include/linux/rtnetlink.h` to inspect the actual helper implementations.
- Validation:
  - Both Kconfig versions retain the same migration warning: current debug locking combines global RTNL with a per-netns mutex, and per-netns scalability depends on future removal of global RTNL.
  - `v7.2-rc3` debug code calls `rtnl_lock()` before `__rtnl_net_lock(net)`; its non-debug fallback calls only `rtnl_lock()`.
- Notes: “not implemented” refers to true per-NS-only RTNL concurrency, not absence of the `rtnl_net_lock()` function.

## 2026-07-16 16:01 - Add RTNL Version Boundary and Paper Citations

- User intent: append one paper-ready paragraph to `bird/0715RTM.md` that identifies the kernel versions containing the per-netns RTNL foundation and IPv4 route-handler conversion, establishes that global RTNL remains, and records the supporting literature and BibTeX.
- Scope: documentation only; updated `bird/0715RTM.md` and this worklog. No VM, Docker, kernel build, experiment, or runtime state was changed.
- Changes:
  - Added one continuous paragraph distinguishing the `rtnl_net_lock()` foundation first released in Linux 6.13 from the IPv4 `RTM_NEWROUTE`/`RTM_DELROUTE` conversion first released in Linux 6.15.
  - Summarized the conclusions of Westphal's RTNL contention paper, the Linux netdev locking documentation, Iwashima's 2024 foundation commit, and the 2025 IPv4 FIB patch series.
  - Added five BibTeX entries immediately after the paragraph and stated precisely that default and debug conversion builds still retain global RTNL serialization.
- Commands:
  - Inspected `0715RTM.md`, the local prerequisite patch, and the retained 2025 v3 patch-series mbox.
  - Compared upstream `v6.12`/`v6.13` helper declarations and `v6.14`/`v6.15` IPv4 handler sources.
  - Inspected Linux 6.15 Kconfig and helper bodies, Linux networking documentation, and the Netdev 2.2 paper.
- Validation:
  - Confirmed Linux 6.12 lacks and Linux 6.13 contains the helper interface; Linux 6.14 lacks and Linux 6.15 contains the IPv4 handler conversion.
  - Confirmed normal builds map `rtnl_net_lock(net)` to global `rtnl_lock()`, while debug conversion builds acquire global RTNL plus the per-netns mutex.
- Notes: the version statement deliberately separates introduction of the migration foundation (Linux 6.13) from migration of the standard IPv4 route-update handlers (Linux 6.15).
