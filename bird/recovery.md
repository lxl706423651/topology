# IDC SSH 恢复操作说明

日期：2026-07-03

背景：在 IDC 机器上解压第一版 `6.8.12-birdrtbulk-k3c-fibref` tarball 之后，`sudo` 返回 `Permission denied`，随后 `seed@10.202.191.39` 的 SSH 公钥登录也开始返回 `Permission denied (publickey,password)`。这最可能说明 tar 解压时改坏了持久化的文件系统元数据，例如 `/`、`/usr`、`/usr/bin`、`/usr/lib`、`/usr/bin/sudo` 或 `/home/seed/.ssh` 的属主/权限。普通重启通常不能修复这类问题，因为损坏的权限已经写在磁盘上。

下面的命令请在 IDC 的 root console/rescue shell 中执行。不要切到新的 `fibref` 内核。

## 0. 确认根文件系统可写

```bash
whoami
hostname
uname -r
findmnt -no SOURCE,FSTYPE,OPTIONS /
mount -o remount,rw / || true
findmnt -no SOURCE,FSTYPE,OPTIONS /
```

预期结果：

- `whoami` 输出 `root`。
- `/` 已经是读写挂载，或者被成功 remount 成读写。
- 当前 kernel 仍可能是 `6.8.12-birdrtbulk-k3c`，这是正常的。

## 1. 先检查当前损坏状态

如果可以，请保存这段输出。它有助于判断到底是哪一级目录权限坏了。

```bash
stat -c '%a %U:%G %n' \
  / /usr /usr/bin /usr/lib /boot /home /home/seed \
  /home/seed/.ssh /home/seed/.ssh/authorized_keys \
  /usr/bin/sudo 2>&1 | tee /root/idc_recovery_stat_before.txt

ls -ld / /usr /usr/bin /usr/lib /boot /home /home/seed /home/seed/.ssh 2>&1 \
  | tee -a /root/idc_recovery_stat_before.txt

ls -l /usr/bin/sudo /home/seed/.ssh/authorized_keys 2>&1 \
  | tee -a /root/idc_recovery_stat_before.txt

namei -om /home/seed/.ssh/authorized_keys 2>&1 \
  | tee -a /root/idc_recovery_stat_before.txt
```

异常信号：

- `/`、`/usr`、`/usr/bin`、`/usr/lib` 或 `/boot` 不是 `root:root`。
- `/home/seed/.ssh/authorized_keys` 路径上的任何父目录是 group-writable 或 world-writable。
- `/usr/bin/sudo` 不可执行，或者丢失 setuid 权限。正常的 sudo 通常是 `root:root`，权限里有 `s`，显示类似 `-rwsr-xr-x`。

## 2. 修复核心系统目录和 sudo 权限

这段命令不会删除文件，只恢复关键目录和 `sudo` 的属主/权限。

```bash
chown root:root /
chmod 755 /

for d in /usr /usr/bin /usr/lib /boot /etc /var /var/log; do
  if [ -e "$d" ]; then
    chown root:root "$d"
    chmod 755 "$d"
  fi
done

if [ -e /usr/bin/sudo ]; then
  chown root:root /usr/bin/sudo
  chmod 4755 /usr/bin/sudo
fi

stat -c '%a %U:%G %n' / /usr /usr/bin /usr/lib /boot /usr/bin/sudo
ls -l /usr/bin/sudo
```

预期结果：

- `/`、`/usr`、`/usr/bin`、`/usr/lib`、`/boot` 都是 `755 root:root`。
- `/usr/bin/sudo` 是 `root:root`，并带 setuid，例如 `-rwsr-xr-x`。

## 3. 修复 seed 用户的 SSH 路径

```bash
getent passwd seed
id seed

chown root:root /home
chmod 755 /home

chown seed:seed /home/seed
chmod go-w /home/seed
chmod 755 /home/seed

install -d -o seed -g seed -m 700 /home/seed/.ssh

if [ -f /home/seed/.ssh/authorized_keys ]; then
  chown seed:seed /home/seed/.ssh/authorized_keys
  chmod 600 /home/seed/.ssh/authorized_keys
fi

chown seed:seed /home/seed/.ssh
chmod 700 /home/seed/.ssh

stat -c '%a %U:%G %n' /home /home/seed /home/seed/.ssh /home/seed/.ssh/authorized_keys
namei -om /home/seed/.ssh/authorized_keys
wc -l /home/seed/.ssh/authorized_keys
ssh-keygen -l -f /home/seed/.ssh/authorized_keys || true
```

预期结果：

- `/home` 是 `755 root:root`。
- `/home/seed` 不是 group-writable，也不是 world-writable。
- `/home/seed/.ssh` 是 `700 seed:seed`。
- `/home/seed/.ssh/authorized_keys` 是 `600 seed:seed`。
- `authorized_keys` 里至少还有一行 key。

## 4. 校验并重启 SSH 服务

Ubuntu 通常使用 `ssh` 这个 service 名称，而不是 `sshd`。

```bash
command -v sshd || command -v /usr/sbin/sshd
/usr/sbin/sshd -t

systemctl restart ssh || systemctl restart sshd
systemctl status ssh --no-pager -l || systemctl status sshd --no-pager -l

journalctl -u ssh -n 80 --no-pager || journalctl -u sshd -n 80 --no-pager
tail -n 80 /var/log/auth.log 2>/dev/null || true
```

预期结果：

- `/usr/sbin/sshd -t` 退出码为 0，正常情况下不输出内容。
- SSH 服务状态是 active。
- 日志里不应再出现 `/`、`/home/seed`、`.ssh`、`authorized_keys` 的 `bad ownership or modes`。

## 5. 从本机测试 SSH 登录

这段在本机执行，不是在 IDC root console 里执行：

```bash
ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null seed@10.202.191.39 \
  'hostname; whoami; uname -r; stat -c "%a %U:%G %n" / /usr /usr/bin /usr/bin/sudo /home/seed /home/seed/.ssh /home/seed/.ssh/authorized_keys; sudo -n true && echo sudo-ok'
```

预期结果：

- 能成功以 `seed` 登录。
- 输出 `sudo-ok`。如果 SSH 登录恢复了，但没有输出 `sudo-ok`，保持 root console 开着，再用交互式 `seed` shell 检查 `sudo -l`。

## 6. 检查 fibref 内核是否处于半安装状态

这段可以在 root console 中执行，也可以在 SSH 恢复后执行。它只做检查和元数据修复，不会重启，也不会切换内核。

```bash
KERNEL=6.8.12-birdrtbulk-k3c-fibref

ls -ld \
  /lib/modules/$KERNEL \
  /usr/lib/modules/$KERNEL \
  /boot/vmlinuz-$KERNEL \
  /boot/initrd.img-$KERNEL \
  /boot/System.map-$KERNEL \
  /boot/config-$KERNEL 2>&1 | tee /root/idc_fibref_install_state.txt

if [ -d "/lib/modules/$KERNEL" ] || [ -d "/usr/lib/modules/$KERNEL" ]; then
  MODDIR="$(readlink -f /lib/modules/$KERNEL 2>/dev/null || readlink -f /usr/lib/modules/$KERNEL)"
  echo "MODDIR=$MODDIR"
  chown -R root:root "$MODDIR"
  find "$MODDIR" -type d -exec chmod 755 {} +
  find "$MODDIR" -type f -exec chmod 644 {} +
fi

for f in /boot/vmlinuz-$KERNEL /boot/System.map-$KERNEL /boot/config-$KERNEL /boot/initrd.img-$KERNEL; do
  if [ -e "$f" ]; then
    chown root:root "$f"
  fi
done

grep -n "menuentry 'Ubuntu, with Linux" /boot/grub/grub.cfg | head -n 80 || true
```

注意：

- 现在不要重启进入 `6.8.12-birdrtbulk-k3c-fibref`。
- IDC 上已经上传的那份 tarball 可能是旧的不安全打包版本。重新安装前必须先用本机安全版本替换。
- 本机安全 tarball 的 sha256 是：

```text
605cde762d438ced00e8d15c777659a0bbda44c49b3eb6af0494f53017f4ffe5
```

## 7. SSH 恢复后，替换 IDC 上的 fibref tarball

SSH 恢复后，在本机执行：

```bash
LOCAL_TARBALL=/home/lxl/seed-emulator/topology/bird/packaged/rtnl_fib_methods_repro_20260701_133858/K3c/kernel/prebuilt/6.8.12-birdrtbulk-k3c-fibref.tar.gz
REMOTE_TARBALL=/home/seed/rtnl_packages/rtnl_fib_methods_repro_20260701_133858_seedlocal/K3c/kernel/prebuilt/6.8.12-birdrtbulk-k3c-fibref.tar.gz

sha256sum "$LOCAL_TARBALL"
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$LOCAL_TARBALL" seed@10.202.191.39:"$REMOTE_TARBALL"
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null seed@10.202.191.39 "sha256sum '$REMOTE_TARBALL'"
```

远端 hash 应该是：

```text
605cde762d438ced00e8d15c777659a0bbda44c49b3eb6af0494f53017f4ffe5
```

只有确认安全 tarball 已经在 IDC 上，才能继续完成 side-by-side 安装：

```bash
ssh -t -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null seed@10.202.191.39
```

然后在 IDC 上执行：

```bash
KERNEL=6.8.12-birdrtbulk-k3c-fibref
TARBALL=/home/seed/rtnl_packages/rtnl_fib_methods_repro_20260701_133858_seedlocal/K3c/kernel/prebuilt/6.8.12-birdrtbulk-k3c-fibref.tar.gz

sha256sum "$TARBALL"
sudo tar --no-overwrite-dir -C / -xzf "$TARBALL"
sudo update-initramfs -c -k "$KERNEL"
sudo update-grub
grep -n "menuentry 'Ubuntu, with Linux" /boot/grub/grub.cfg | head -n 80
```

到这里先停下，汇报 GRUB entries。重启进入 `fibref` 仍然需要明确的维护窗口确认，因为切内核会断开 SSH。

## 8. SSH 健康后可选执行 Docker Reset

只有 SSH 和 `sudo` 都恢复健康后再执行。这是用户指定的 IDC 恢复步骤，和恢复 SSH 本身无关。

```bash
ssh -t -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null seed@10.202.191.39
```

然后在 IDC 上执行：

```bash
sudo /home/seed/lxl_topology/autocoder_test/reset_docker.sh

docker info --format 'Containers={{.Containers}} Running={{.ContainersRunning}} Images={{.Images}} NFd={{.NFd}} NGoroutines={{.NGoroutines}}'
docker ps -a | head
docker network ls | head
```

不要执行 `docker system prune`，也不要删除 `/var/lib/docker`，除非再次明确确认。

