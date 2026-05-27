import psutil
import subprocess
import time
import os
from datetime import datetime
import signal
import sys
import json
import re

class DockerCommandMonitor:
    def __init__(self, command_idx, interval=1, batch_size=50, parallel_jobs=8, post_monitor=False):
        """
        初始化Docker命令监控器
        :param command_idx: 命令编号（1或2）
        :param interval: 监控间隔（秒）
        :param batch_size: xargs的-n参数
        :param parallel_jobs: xargs的-P参数
        :param post_monitor: 是否在up结束后继续持续监控 (True/False)
        """
        
        self.batch_size = batch_size
        self.parallel_jobs = parallel_jobs
        self.interval = interval
        self.post_monitor = post_monitor  # 新增控制参数
        
        self.compose_file = "docker-compose.yml" 
        self.output_dir = os.path.join(os.getcwd(), "output")

        # 0. 获取节点规模 (用于文件名)
        self.node_scale = self._get_node_scale()
        print(f"检测到节点规模 (output下子目录数): {self.node_scale}")

        # 构建 'config' 命令
        config_cmd = f"docker compose -f {self.compose_file} config --services"

        # 1. 构建 'build' 命令
        # 注意：这里去掉了 export DOCKER_BUILDKIT=0，因为有些环境可能需要它，
        # 如果你确定需要禁用 BuildKit，可以加回去。
        # 关键修改：为了捕捉错误，不能在这里直接重定向 > /dev/null，我们要在 Popen 中接管输出
        build_xargs = f"xargs -n {self.batch_size} -P {self.parallel_jobs}"
        build_suffix = f"docker compose -f {self.compose_file} build" # --progress=plain 输出太多，建议去掉或保留看需求
        
        # 2. 构建 'up' 命令
        up_xargs = f"xargs -n {self.batch_size} -P {self.parallel_jobs}"
        up_suffix = f"docker compose -f {self.compose_file} up -d"
        
        self.commands = {
            1: f"{config_cmd} | {build_xargs} {build_suffix}",
            2: f"{config_cmd} | {up_xargs} {up_suffix}"
        }
        
        if command_idx not in self.commands:
            raise ValueError(f"命令编号无效，只能是1或2（1=build，2=up）")
            
        self.command = self.commands[command_idx]
        self.command_name = "build" if command_idx == 1 else "up"
        self.process = None
        self.running = False

        # --- 初始化速率计算所需的上一时刻状态 ---
        self.last_io_time = time.time()
        self.last_net_io = None
        self.last_disk_io = None
        
        # --- 监控日志命名 ---
        self._init_log_file()
        
        # --- [新增] 命令执行输出日志 (用于排查 123 错误) ---
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.cmd_log_filename = f"logs/dockerBuild_Up/cmd_output_{self.command_name}_{timestamp}.log"
        self.cmd_log_path = os.path.join(os.getcwd(), self.cmd_log_filename)

    def _init_log_file(self, suffix=""):
        """初始化或重置监控数据日志文件路径"""
        params_str = f"scale{self.node_scale}_n{self.batch_size}_p{self.parallel_jobs}_i{self.interval}"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 监控数据日志 (CSV)
        name_part = self.command_name if not suffix else f"{self.command_name}_{suffix}"
        self.log_filename = f"logs/dockerBuild_Up/docker_{name_part}_{params_str}_{timestamp}.log"
        self.log_path = os.path.join(os.getcwd(), self.log_filename)
        
        self._write_log_header()

    def _get_node_scale(self):
        """获取output文件夹下的子文件夹数量作为节点规模"""
        try:
            if not os.path.exists(self.output_dir):
                return 0
            # 统计目录下是文件夹的项目
            count = len([name for name in os.listdir(self.output_dir) 
                         if os.path.isdir(os.path.join(self.output_dir, name))])
            return count
        except Exception:
            return 0

    def _write_log_header(self):
        """写入日志头部"""
        with open(self.log_path, 'w', encoding='utf-8') as f:
            f.write("--- 系统全局监控任务开始 ---\n")
            f.write(f"任务类型: {self.command_name}\n")
            f.write(f"节点规模: {self.node_scale}\n")
            f.write(f"Post-Monitor: {self.post_monitor}\n") # 记录参数
            f.write(f"监控间隔: {self.interval}s\n")
            f.write(f"完整命令: {self.command}\n")
            f.write("----------------------------------\n\n")
            
            headers = [
                "时间",
                "Load_1m", "Load_5m", "Load_15m",
                "CPU_Total%", "CPU_User%", "CPU_Sys%", "CPU_IOWait%", "CPU_Per_Core%",
                "Mem_Used%", "Mem_Used_MB", "Mem_Cached_MB", "Mem_Buffers_MB", "Swap_Used%",
                "PID_Count", "Thread_Count",
                "FD_Used", "FD_Ratio%", 
                "TCP_Alloc", "TCP_InUse",
                "Conntrack_Used", "Conntrack_Ratio%",
                "Net_Recv_MB/s", "Net_Sent_MB/s", "Net_Packets_Recv/s", "Net_Packets_Sent/s",
                "Disk_Read_MB/s", "Disk_Write_MB/s", "Disk_Read_IOPS", "Disk_Write_IOPS",
                "Disk_Root_Used%",
                "Docker_Stat_Info"
            ]
            f.write(",".join(headers) + "\n")

    def _read_sys_file(self, path):
        try:
            with open(path, 'r') as f:
                return f.read().strip()
        except:
            return None

    def _get_global_metrics(self):
        metrics = {}
        current_time = time.time()
        time_delta = current_time - self.last_io_time
        if time_delta <= 0: time_delta = 1.0
        
        # 1. CPU & Load
        try:
            cpu_times = psutil.cpu_times_percent(interval=None) 
            metrics['cpu_total'] = 100.0 - cpu_times.idle
            metrics['cpu_user'] = cpu_times.user
            metrics['cpu_sys'] = cpu_times.system
            metrics['cpu_iowait'] = getattr(cpu_times, 'iowait', 0.0)
            
            per_core = psutil.cpu_percent(interval=None, percpu=True)
            metrics['cpu_per_core'] = "[" + "|".join([str(int(x)) for x in per_core]) + "]"
            
            load_avg = os.getloadavg()
            metrics['load_1'], metrics['load_5'], metrics['load_15'] = load_avg
        except Exception:
            metrics.update({'cpu_total':0,'cpu_user':0,'cpu_sys':0,'cpu_iowait':0,'cpu_per_core':"[]",'load_1':0,'load_5':0,'load_15':0})

        # 2. Memory
        try:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            metrics['mem_percent'] = mem.percent
            metrics['mem_used_mb'] = mem.used / (1024 * 1024)
            metrics['mem_cached_mb'] = getattr(mem, 'cached', 0) / (1024 * 1024)
            metrics['mem_buffers_mb'] = getattr(mem, 'buffers', 0) / (1024 * 1024)
            metrics['swap_percent'] = swap.percent
        except:
            metrics.update({'mem_percent':0, 'mem_used_mb':0, 'mem_cached_mb':0, 'mem_buffers_mb':0, 'swap_percent':0})

        # 3. PIDs
        try:
            pids = psutil.pids()
            metrics['pid_count'] = len(pids)
            metrics['thread_count'] = 0 
        except:
            metrics['pid_count'] = 0
            metrics['thread_count'] = 0

        # 4. FD & TCP
        try:
            fd_data = self._read_sys_file('/proc/sys/fs/file-nr')
            if fd_data:
                parts = fd_data.split()
                fd_alloc = int(parts[0])
                fd_max = int(parts[2])
                metrics['fd_used'] = fd_alloc
                metrics['fd_ratio'] = (fd_alloc / fd_max) * 100 if fd_max > 0 else 0
            
            sock_data = self._read_sys_file('/proc/net/sockstat')
            metrics['tcp_alloc'] = 0
            metrics['tcp_inuse'] = 0
            if sock_data:
                for line in sock_data.splitlines():
                    if line.startswith("TCP:"):
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "inuse":
                                metrics['tcp_inuse'] = int(parts[i+1])
                            elif part == "alloc":
                                metrics['tcp_alloc'] = int(parts[i+1])
                        break
        except:
             metrics.update({'fd_used':0, 'fd_ratio':0, 'tcp_alloc':0, 'tcp_inuse':0})

        # 5. Conntrack
        try:
            ct_count_str = self._read_sys_file('/proc/sys/net/netfilter/nf_conntrack_count')
            ct_max_str = self._read_sys_file('/proc/sys/net/netfilter/nf_conntrack_max')
            if ct_count_str and ct_max_str:
                ct_used = int(ct_count_str)
                ct_max = int(ct_max_str)
                metrics['ct_used'] = ct_used
                metrics['ct_ratio'] = (ct_used / ct_max) * 100 if ct_max > 0 else 0
            else:
                metrics.update({'ct_used':0, 'ct_ratio':0})
        except:
            metrics.update({'ct_used':0, 'ct_ratio':0})

        # 6. Network I/O
        try:
            net_io = psutil.net_io_counters()
            if self.last_net_io:
                sent_bytes = net_io.bytes_sent - self.last_net_io.bytes_sent
                recv_bytes = net_io.bytes_recv - self.last_net_io.bytes_recv
                sent_pkts = net_io.packets_sent - self.last_net_io.packets_sent
                recv_pkts = net_io.packets_recv - self.last_net_io.packets_recv
                
                metrics['net_recv_mb_s'] = (recv_bytes / time_delta) / (1024 * 1024)
                metrics['net_sent_mb_s'] = (sent_bytes / time_delta) / (1024 * 1024)
                metrics['net_pkts_recv_s'] = recv_pkts / time_delta
                metrics['net_pkts_sent_s'] = sent_pkts / time_delta
            else:
                metrics.update({'net_recv_mb_s':0, 'net_sent_mb_s':0, 'net_pkts_recv_s':0, 'net_pkts_sent_s':0})
            
            self.last_net_io = net_io
        except:
             metrics.update({'net_recv_mb_s':0, 'net_sent_mb_s':0, 'net_pkts_recv_s':0, 'net_pkts_sent_s':0})

        # 7. Disk I/O
        try:
            disk_io = psutil.disk_io_counters()
            if self.last_disk_io and disk_io:
                read_bytes = disk_io.read_bytes - self.last_disk_io.read_bytes
                write_bytes = disk_io.write_bytes - self.last_disk_io.write_bytes
                read_count = disk_io.read_count - self.last_disk_io.read_count
                write_count = disk_io.write_count - self.last_disk_io.write_count

                metrics['disk_read_mb_s'] = (read_bytes / time_delta) / (1024 * 1024)
                metrics['disk_write_mb_s'] = (write_bytes / time_delta) / (1024 * 1024)
                metrics['disk_read_iops'] = read_count / time_delta
                metrics['disk_write_iops'] = write_count / time_delta
            else:
                metrics.update({'disk_read_mb_s':0, 'disk_write_mb_s':0, 'disk_read_iops':0, 'disk_write_iops':0})
            
            self.last_disk_io = disk_io
        except:
            metrics.update({'disk_read_mb_s':0, 'disk_write_mb_s':0, 'disk_read_iops':0, 'disk_write_iops':0})

        # 8. Disk Space
        try:
            usage = psutil.disk_usage('/')
            metrics['disk_root_used_percent'] = usage.percent
        except:
            metrics['disk_root_used_percent'] = 0

        self.last_io_time = current_time
        return metrics

    def _get_docker_count(self):
        try:
            count_cmd = ""
            count_name = ""
            if self.command_name == "build":
                count_cmd = "docker images -q | wc -l"
                count_name = "ImgCount"
            else:
                count_cmd = "docker ps -q | wc -l"
                count_name = "ContainerCount"

            result = subprocess.run(count_cmd, shell=True, capture_output=True, text=True)
            count = result.stdout.strip()
            return f"{count_name}:{count}"
        except Exception:
            return f"StatError"

    def _monitor_loop(self, check_process=True):
        last_stat_time = 0
        stat_interval = 300 
        
        psutil.cpu_times_percent(interval=None)
        self.last_net_io = psutil.net_io_counters()
        self.last_disk_io = psutil.disk_io_counters()
        self.last_io_time = time.time()
        
        time.sleep(0.1)

        while self.running:
            if check_process and self.process and self.process.poll() is not None:
                break

            current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            m = self._get_global_metrics()
            
            stat_info = ""
            current_ts = time.time()
            if current_ts - last_stat_time >= stat_interval:
                stat_info = self._get_docker_count()
                last_stat_time = current_ts
            
            line_data = [
                current_time_str,
                f"{m['load_1']:.2f}", f"{m['load_5']:.2f}", f"{m['load_15']:.2f}",
                f"{m['cpu_total']:.1f}", f"{m['cpu_user']:.1f}", f"{m['cpu_sys']:.1f}", f"{m['cpu_iowait']:.1f}", m['cpu_per_core'],
                f"{m['mem_percent']:.1f}", f"{m['mem_used_mb']:.0f}", f"{m['mem_cached_mb']:.0f}", f"{m['mem_buffers_mb']:.0f}", f"{m['swap_percent']:.1f}",
                f"{m['pid_count']}", f"{m['thread_count']}",
                f"{m['fd_used']}", f"{m['fd_ratio']:.2f}",
                f"{m['tcp_alloc']}", f"{m['tcp_inuse']}",
                f"{m['ct_used']}", f"{m['ct_ratio']:.2f}",
                f"{m['net_recv_mb_s']:.2f}", f"{m['net_sent_mb_s']:.2f}", f"{m['net_pkts_recv_s']:.0f}", f"{m['net_pkts_sent_s']:.0f}",
                f"{m['disk_read_mb_s']:.2f}", f"{m['disk_write_mb_s']:.2f}", f"{m['disk_read_iops']:.0f}", f"{m['disk_write_iops']:.0f}",
                f"{m['disk_root_used_percent']:.1f}",
                stat_info
            ]
            
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(",".join(line_data) + "\n")
            
            time.sleep(self.interval)

    # ----------------------------------------------------------------
    # 验证逻辑：Build
    # ----------------------------------------------------------------
    def _verify_build_success(self):
        print("开始验证镜像构建结果...")
        try:
            config_cmd = ["docker", "compose", "-f", self.compose_file, "config", "--format", "json"]
            result = subprocess.run(config_cmd, cwd=self.output_dir, capture_output=True, text=True, encoding='utf-8', check=True)
            config_data = json.loads(result.stdout)
            
            project_name = config_data.get("name")
            if not project_name:
                project_name = os.path.basename(self.output_dir)

            images_cmd = ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"]
            result = subprocess.run(images_cmd, capture_output=True, text=True, encoding='utf-8', check=True)
            available_images = set(result.stdout.strip().split("\n"))
            
            services = config_data.get("services", {})
            if not services: return True

            missing_services = []
            checked_count = 0

            for service_name, service_data in services.items():
                if "build" in service_data:
                    checked_count += 1
                    if "image" in service_data:
                        image_name = service_data.get("image")
                        if ":" not in image_name: image_name += ":latest"
                        if image_name not in available_images:
                            missing_services.append(f"服务 '{service_name}' (期望: {image_name})")
                        continue

                    candidates = [
                        f"{project_name}-{service_name}:latest", 
                        f"{project_name}_{service_name}:latest"
                    ]
                    if not any(cand in available_images for cand in candidates):
                           missing_services.append(f"服务 '{service_name}' (期望: {candidates[0]} 或 {candidates[1]})")

            if checked_count == 0: return True

            if not missing_services:
                print(f"✅ 验证成功：所有 {checked_count} 个应构建的镜像均已在本地找到。")
                return True
            else:
                print(f"❌ 验证失败：以下 {len(missing_services)} 个服务的镜像在本地未找到：")
                for msg in missing_services[:10]:
                    print(f"  - {msg}")
                return False

        except Exception as e:
            print(f"验证错误: {e}")
            return False

    # ----------------------------------------------------------------
    # 验证逻辑：Up
    # ----------------------------------------------------------------
    def _verify_up_success(self):
        print("开始验证容器和网络启动结果...")
        log_msgs = []
        is_success = True

        try:
            config_cmd = ["docker", "compose", "-f", self.compose_file, "config", "--format", "json"]
            result = subprocess.run(config_cmd, cwd=self.output_dir, capture_output=True, text=True, encoding='utf-8', check=True)
            config_data = json.loads(result.stdout)
            
            project_name = config_data.get("name")
            if not project_name:
                project_name = os.path.basename(self.output_dir).lower()
                project_name = ''.join(c for c in project_name if c.isalnum() or c in '_-')

            # --- 验证容器 ---
            all_services = set(config_data.get("services", {}).keys())
            
            ps_cmd = ["docker", "compose", "-f", self.compose_file, "ps", "--services", "--filter", "status=running"]
            ps_result = subprocess.run(ps_cmd, cwd=self.output_dir, capture_output=True, text=True, encoding='utf-8', check=True)
            running_services = set(ps_result.stdout.strip().split("\n"))
            if "" in running_services: running_services.remove("")

            missing_containers = all_services - running_services
            
            if not missing_containers:
                msg = f"✅ 容器验证通过：所有 {len(all_services)} 个服务均在运行。"
                print(msg)
                log_msgs.append(msg)
            else:
                is_success = False
                msg = f"❌ 容器验证失败：以下 {len(missing_containers)} 个服务未运行: {', '.join(list(missing_containers)[:10])}..."
                print(msg)
                log_msgs.append(msg)

            # --- 验证网络 ---
            defined_networks = config_data.get("networks", {})
            expected_networks = []
            
            if not defined_networks:
                expected_networks.append(f"{project_name}_default")
            else:
                for net_name, net_conf in defined_networks.items():
                    is_external = False
                    if isinstance(net_conf, dict) and net_conf.get("external") is True:
                        is_external = True
                        ext_name = net_conf.get("name", net_name)
                        expected_networks.append(ext_name)
                    else:
                        expected_networks.append(f"{project_name}_{net_name}")

            net_ls_cmd = ["docker", "network", "ls", "--format", "{{.Name}}"]
            net_result = subprocess.run(net_ls_cmd, capture_output=True, text=True, encoding='utf-8', check=True)
            existing_networks = set(net_result.stdout.strip().split("\n"))

            missing_networks = []
            for net in expected_networks:
                if net not in existing_networks:
                    alt_name = net.replace("_", "-")
                    alt_name2 = net.replace("-", "_")
                    if alt_name in existing_networks: continue
                    if alt_name2 in existing_networks: continue
                    missing_networks.append(net)

            if not missing_networks:
                msg = f"✅ 网络验证通过：所有预期网络 ({len(expected_networks)}个) 均存在。"
                print(msg)
                log_msgs.append(msg)
            else:
                is_success = False
                msg = f"❌ 网络验证失败：未找到网络: {', '.join(missing_networks)}"
                print(msg)
                log_msgs.append(msg)

            return is_success, "\n".join(log_msgs)

        except Exception as e:
            err_msg = f"验证过程发生异常: {str(e)}"
            print(err_msg)
            return False, err_msg

    def start(self):
        """启动命令并监控"""
        if not os.path.exists(self.output_dir):
            print(f"错误：output目录不存在（路径：{self.output_dir}）")
            return
        
        if not os.path.exists(os.path.join(self.output_dir, self.compose_file)):
             print(f"错误：在 output 目录中未找到 {self.compose_file}")
             return
        
        # 打开日志文件准备记录命令输出
        cmd_log_file = open(self.cmd_log_path, 'w', encoding='utf-8')

        print("=================================================")
        print(f"正在后台执行命令...") 
        print(f"任务类型: {self.command_name}")
        print(f"节点规模: {self.node_scale}")
        print(f"监控日志: {self.log_path}")
        print(f"命令输出: {self.cmd_log_path} (遇到错误请查看此文件)")
        print(f"持续监控: {'开启' if self.post_monitor and self.command_name=='up' else '关闭'}")
        print("=================================================")
        
        start_time = time.time()
        
        try:
            # 关键修改：stdout和stderr重定向到文件
            self.process = subprocess.Popen(
                self.command,
                shell=True,
                cwd=self.output_dir,
                stdout=cmd_log_file,  
                stderr=cmd_log_file, 
                text=True,
                preexec_fn=os.setsid
            )
            
            self.running = True
            
            # 阶段1：命令执行期间的监控
            self._monitor_loop(check_process=True)
            
            self.process.wait() # 等待命令结束

        finally:
            # 确保文件句柄关闭
            cmd_log_file.close()

        end_time = time.time()
        
        # ----------------------------------------------------------------
        # 总结与验证
        # ----------------------------------------------------------------
        main_return_code = self.process.returncode
        verification_passed = None
        verification_details = ""
        
        print("\n任务结束，正在执行最终验证...")

        if main_return_code != 0:
             verification_passed = False
             verification_details = f"主命令执行失败 (Return Code {main_return_code})。\n请检查日志文件: {self.cmd_log_path}"
             
             # 尝试读取最后几行错误信息并打印出来
             print(f"\n❌ 检测到错误 (Code {main_return_code})。日志最后 10 行:")
             print("-" * 50)
             try:
                 with open(self.cmd_log_path, 'r', encoding='utf-8', errors='ignore') as f:
                     lines = f.readlines()
                     for line in lines[-10:]:
                         print(line.strip())
             except Exception:
                 print("无法读取日志文件。")
             print("-" * 50)

        else:
            if self.command_name == "build":
                verification_passed = self._verify_build_success()
                verification_details = "Build Verification Completed"
            elif self.command_name == "up":
                verification_passed, verification_details = self._verify_up_success()
        
        duration = int(end_time - start_time)
        hours = duration // 3600
        minutes = (duration % 3600) // 60
        seconds = duration % 60
        
        print("\n-------------------------------------")
        print("执行结束。")
        print(f"主命令返回码：{self.process.returncode}")
        
        status_text = "未知"
        if verification_passed is not None:
            status_text = "成功 ✅" if verification_passed else "失败 ❌"
        print(f"最终验证结果：{status_text}")
        print(f"Total runtime: {hours}h {minutes}m {seconds}s")
        print("=================================================")
        
        # 将总结写入监控日志
        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write("\n\n--- 监控任务结束 ---\n")
                f.write(f"主命令返回码: {self.process.returncode}\n")
                f.write(f"最终验证结果: {status_text}\n")
                if verification_details:
                    f.write(f"验证详情:\n{verification_details}\n")
                f.write(f"Total runtime: {hours}h {minutes}m {seconds}s\n")
                f.write("----------------------------------\n")
        except Exception as e:
            pass

        # ----------------------------------------------------------------
        # 阶段2：持续监控模式 (Post-Execution) - 仅针对 'up' 命令且参数开启
        # ----------------------------------------------------------------
        if self.command_name == "up" and self.post_monitor:
            print("\n进入持续监控模式 (Post-Execution Monitoring)...")
            print("监控间隔调整为: 60秒")
            print("日志文件将重新生成，监控参数保持一致。")
            print("按 Ctrl+C 停止监控")

            # 调整参数
            self.interval = 60
            
            # 重新生成日志文件名 (加 _post 后缀以便区分，同时更新时间戳)
            self._init_log_file(suffix="post")
            
            # 继续监控 (不再检查进程状态，因为进程已经结束了)
            self._monitor_loop(check_process=False)
        elif self.command_name == "up" and not self.post_monitor:
            print(f"\nPost-Monitor 参数为 0 (关闭)，监控结束。")
        else:
            print(f"\n任务 {self.command_name} 已完成，监控结束。")

    def stop(self):
        """强制停止"""
        self.running = False
        if self.process and self.process.poll() is None:
            print("\n强制终止命令...")
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except Exception:
                pass


if __name__ == "__main__":
    # 参数检查逻辑修改：允许 4个参数 或 5个参数
    if len(sys.argv) < 5 or len(sys.argv) > 6:
        print("用法：python test.py <命令编号> <监控间隔秒数> <批次大小n> <并行作业P> [可选:自动检测开启(1/0)]")
        print("1 = build, 2 = up")
        sys.exit(1)
    
    try:
        command_idx = int(sys.argv[1])
        interval = float(sys.argv[2])
        batch_size = int(sys.argv[3])
        parallel_jobs = int(sys.argv[4])
        
        # 处理可选参数，默认 0 (False)
        post_monitor_flag = 0
        if len(sys.argv) == 6:
            post_monitor_flag = int(sys.argv[5])
        
        post_monitor = True if post_monitor_flag == 1 else False
        
        if batch_size <= 0: batch_size = 50
        if parallel_jobs <= 0: parallel_jobs = 8
    except ValueError:
        print("错误：参数必须是数字")
        sys.exit(1)
    
    monitor = None
    try:
        monitor = DockerCommandMonitor(command_idx, interval, batch_size, parallel_jobs, post_monitor)
        monitor.start()
    except KeyboardInterrupt:
        if monitor:
            monitor.stop()
        print("\n监控已停止")
    except Exception as e:
        print(f"执行失败：{e}")
        if monitor:
            monitor.stop()