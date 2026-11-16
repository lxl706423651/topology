# 0703 K3c netdevice leak / fib_info refcount problem

## Problem summary

In the K3c kernel (`6.8.12-birdrtbulk-k3c`), large SeedEMU/BIRD RTNL/FIB
runs can leave network namespaces unable to cleanly exit. The kernel log seen on
the IDC machine after K3c runs contained:

- `refcount_t: underflow; use-after-free`
- stack frames under `bird_rtbulk_apply_many_v4()` / `bird_rtbulk_apply_v4()`
- repeated `unregister_netdevice: waiting for <dev> to become free` for devices
  such as `ix10`, `net_7_10`, and `net_10_11`

This was observed in K3c runs at 3083, 3821, and 4192 scale. K1/K2c boot logs
checked for the same experiment window did not show this specific K3c
`bird_rtbulk` refcount/netdevice problem.

The visible symptom is not just a warning. A leaked netdevice reference can
delay or block Docker/netns teardown, leave stale namespace resources behind,
and make the next experiment look stuck in Docker cleanup or network creation.

## Root cause

K3c introduced a cached `fib_info` fast path in:

- `/data/lxl/kernel-work/linux-source-6.8.0-124.124/net/ipv4/fib_bulk.c`
- packaged mirror:
  `/home/lxl/seed-emulator/topology/bird/packaged/rtnl_fib_methods_repro_20260701_133858/K3c/kernel/k3c_overlay/net/ipv4/fib_bulk.c`

The fast path in `bird_rtbulk_apply_many_v4()` creates one shared `fib_info`
with `fib_create_info(&base_cfg, extack)` and then inserts many prefixes with
`fib_table_insert_info(net, tb, &cfg, fi, extack)`.

The bug is that the code called `fib_info_hold(fi)` before each insert. That
increments `fib_clntref`, not `fib_treeref`.

For entries stored in the FIB trie, the relevant ownership is `fib_treeref`.
`fib_table_insert()` normally creates or finds a `fib_info` with one tree
reference and consumes that reference into a `fib_alias` on success. On
duplicate, replace-no-op, allocation failure, notifier failure, or another
error path, the insert path drops that tree reference with `fib_release_info()`.

The K3c cached insert helper follows the same contract:

- success path: stores `fi` in `new_fa->fa_info` and returns without releasing it
- error/no-op path: calls `fib_release_info(fi)`

Because K3c was not adding one `fib_treeref` per alias, many aliases could
point at the same `fib_info` while only the original single tree reference
existed. Later route deletion, replacement, or namespace teardown releases
those aliases one by one, causing `fib_treeref` underflow and stale nexthop /
netdevice references. That matches the IDC log signature.

## Fix

Before each `fib_table_insert_info()` attempt in the cached fast path, add one
tree reference:

```c
refcount_inc(&fi->fib_treeref);
err = fib_table_insert_info(net, tb, &cfg, fi, extack);
```

Keep the final `fib_release_info(fi)` after the loop. That final release drops
the original tree reference returned by `fib_create_info()`.

With this ownership model:

- successful insert: the new `fib_alias` owns the per-attempt `fib_treeref`
- failed/no-op insert: `fib_table_insert_info()` drops the per-attempt
  `fib_treeref`
- after the batch: the original setup reference is released

This is a narrow fix and should not materially change write latency. It adds
one atomic refcount increment per cached route insert, but it keeps the K3c
optimization that avoids per-route `fib_create_info()` and avoids adding new
RTNL locking, notifier, or full FIB construction work.

## Implementation status on 2026-07-03

The local kernel source and the packaged K3c overlay have been changed:

- `/data/lxl/kernel-work/linux-source-6.8.0-124.124/net/ipv4/fib_bulk.c`
- `/home/lxl/seed-emulator/topology/bird/packaged/rtnl_fib_methods_repro_20260701_133858/K3c/kernel/k3c_overlay/net/ipv4/fib_bulk.c`

Both files now use a small helper, `bird_rtbulk_fib_info_tree_hold()`, before
calling `fib_table_insert_info()`. The previous `fib_info_hold(fi)` call is no
longer present in the K3c cached insert path.

A side-by-side fixed kernel package was built locally:

- kernel release: `6.8.12-birdrtbulk-k3c-fibref`
- local tarball:
  `/data/lxl/kernel-work/6.8.12-birdrtbulk-k3c-fibref.tar.gz`
- packaged prebuilt copy:
  `/home/lxl/seed-emulator/topology/bird/packaged/rtnl_fib_methods_repro_20260701_133858/K3c/kernel/prebuilt/6.8.12-birdrtbulk-k3c-fibref.tar.gz`
- sha256:
  `605cde762d438ced00e8d15c777659a0bbda44c49b3eb6af0494f53017f4ffe5`

The local/package tarball was repacked after the first IDC upload so that tar
metadata is root-owned and installable directories are `0755`. This avoids
carrying local stage ownership or restrictive umask metadata into `/boot` or
`/lib/modules`.

The fixed tarball was uploaded to IDC as:

- `/home/seed/rtnl_packages/rtnl_fib_methods_repro_20260701_133858_seedlocal/K3c/kernel/prebuilt/6.8.12-birdrtbulk-k3c-fibref.tar.gz`

The uploaded IDC copy was the earlier package hash:

- IDC uploaded sha256:
  `7640404ca7c81577fe0c2215a85e7db4b1e7d3f556b7d881dc15fb137a562875`

After SSH is restored, replace the IDC tarball with the local/package hash
`605cde762d438ced00e8d15c777659a0bbda44c49b3eb6af0494f53017f4ffe5` before
trying the install again.

The IDC side-by-side install did not complete in this session. The install
attempt likely got past the silent `sudo tar -C / -xzf ...` step and then
failed at the next `sudo update-initramfs ...` line, with:

```text
bash: line 12: /usr/bin/sudo: Permission denied
```

Immediately after that, SSH public-key authentication for `seed`, `lxl`, and
`root` started returning `Permission denied (publickey,password)`, so `/boot`
and `/lib/modules` could not be rechecked. Based on the command line numbering,
the fixed kernel should be treated as potentially partially unpacked on IDC:
`/boot/vmlinuz-6.8.12-birdrtbulk-k3c-fibref` and
`/lib/modules/6.8.12-birdrtbulk-k3c-fibref` may exist, but initramfs and GRUB
were not confirmed.

Do not reboot into `6.8.12-birdrtbulk-k3c-fibref` until login is restored and
the side-by-side install is explicitly verified.

When login is restored, first inspect the remote state before taking any action:

```bash
uname -r
ls -ld /lib /lib/modules /usr/lib /usr/bin/sudo /home/seed /home/seed/.ssh
ls -lh /boot/vmlinuz-6.8.12-birdrtbulk-k3c-fibref \
       /boot/initrd.img-6.8.12-birdrtbulk-k3c-fibref \
       /boot/System.map-6.8.12-birdrtbulk-k3c-fibref \
       /boot/config-6.8.12-birdrtbulk-k3c-fibref 2>/dev/null || true
test -d /lib/modules/6.8.12-birdrtbulk-k3c-fibref && echo modules-present
```

The safer reinstall command, after replacing the IDC tarball with the current
local hash, is:

```bash
KERNEL=6.8.12-birdrtbulk-k3c-fibref
TARBALL=/home/seed/rtnl_packages/rtnl_fib_methods_repro_20260701_133858_seedlocal/K3c/kernel/prebuilt/6.8.12-birdrtbulk-k3c-fibref.tar.gz
sudo tar --no-overwrite-dir -C / -xzf "$TARBALL"
sudo update-initramfs -c -k "$KERNEL"
sudo update-grub
```

## Validation plan

1. Build-check `net/ipv4/fib_bulk.o` in the local kernel tree. Done.
2. Build a new side-by-side K3c fixed kernel package with a distinct local
   version, `6.8.12-birdrtbulk-k3c-fibref`. Done locally.
3. Install on IDC only side-by-side. Do not remove old kernels or modules.
   Earlier tarball was uploaded; install may be partial and is not confirmed
   because SSH access failed. Replace the uploaded tarball with the current
   root-owned package before retrying.
4. Reboot into the fixed K3c kernel only after explicit maintenance-window
   confirmation.
5. Run the same K3c workflow and verify:
   - no `refcount_t: underflow`
   - no `use-after-free`
   - no `unregister_netdevice: waiting for ... to become free`
   - Docker project/networks cleanly tear down
   - K3c write time remains close to previous K3c runs

## Residual risk

This fixes the ownership mismatch in the cached FIB insert path. It does not
prove there are no other K3c bugs until the fixed kernel is booted and exercised
on IDC with the 3083/3821/4192 workflows.
