import psutil
import subprocess
import time
import os
from datetime import datetime
import signal
import sys

class DockerCommandMonitor:
    def __init__(self, command_idx, interval=1):
        """
        初始化Docker命令监控器
        :param command_idx: 命令编号（1或2）
        :param interval: 监控间隔（秒）
        """
        # 预设两条命令
        self.commands = {
            1: "export DOCKER_BUILDKIT=0; docker compose config --services | xargs -n100 docker compose build",
            2: "export DOCKER_BUILDKIT=0; docker compose config --services | xargs -n100 docker compose up -d"
        }
        # 验证命令编号
        if command_idx not in self.commands:
            raise ValueError(f"命令编号无效，只能是1或2（1=build，2=up）")
        self.command = self.commands[command_idx]
        self.command_name = "build" if command_idx == 1 else "up"  # 用于日志命名
        self.interval = interval
        self.process = None  # 子进程对象
        self.running = False  # 监控状态
        self.output_dir = os.path.join(os.getcwd(), "output1")  # output子目录路径
        
        # 生成日志文件名（格式：命令类型_时间.log）
        self.log_filename = f"docker_{self.command_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.log_path = os.path.join(os.getcwd(), self.log_filename)  # 日志保存在当前目录
        self._write_log_header()

    def _write_log_header(self):
        """写入日志头部"""
        with open(self.log_path, 'w', encoding='utf-8') as f:
            f.write("时间,主进程ID,总CPU使用率(%),总内存占用(MB),系统内存使用率(%)\n")

    def _get_process_resources(self, pid):
        """获取进程及其子进程的总资源占用"""
        try:
            main_process = psutil.Process(pid)
            all_processes = [main_process] + main_process.children(recursive=True)  # 包含所有子进程
            
            total_cpu = 0.0
            total_memory = 0.0
            
            for p in all_processes:
                try:
                    total_cpu += p.cpu_percent(interval=0.01)  # 累加CPU使用率
                    total_memory += p.memory_info().rss / (1024 **2)  # 累加内存（MB）
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue  # 忽略已结束或无权限的进程
            
            sys_mem_percent = psutil.virtual_memory().percent  # 系统内存使用率
            return total_cpu, total_memory, sys_mem_percent
        
        except psutil.NoSuchProcess:
            return 0.0, 0.0, 0.0  # 主进程已结束
        except Exception as e:
            print(f"获取资源失败：{e}")
            return 0.0, 0.0, 0.0

    def _monitor_loop(self):
        """监控循环"""
        while self.running and self.process.poll() is None:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # 毫秒级时间
            pid = self.process.pid
            cpu, mem, sys_mem = self._get_process_resources(pid)
            
            # 写入日志
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(f"{current_time},{pid},{cpu:.2f},{mem:.2f},{sys_mem:.2f}\n")
            
            time.sleep(self.interval)

    def start(self):
        """启动命令并监控"""
        # 检查output目录是否存在
        if not os.path.exists(self.output_dir):
            print(f"错误：output目录不存在（路径：{self.output_dir}）")
            return
        
        print(f"将在目录 {self.output_dir} 中执行命令：")
        print(f"{self.command}\n")
        print(f"监控日志将保存至：{self.log_path}")
        
        # 在output目录中执行命令
        self.process = subprocess.Popen(
            self.command,
            shell=True,
            cwd=self.output_dir,  # 切换到output目录执行
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True
        )
        
        self.running = True
        self._monitor_loop()  # 开始监控
        
        # 命令执行完毕，获取输出
        stdout, stderr = self.process.communicate()
        if stdout:
            print(f"\n命令输出：\n{stdout[:500]}...")  # 只显示前500字符，避免过长
        if stderr:
            print(f"\n命令错误：\n{stderr}")
        
        print(f"\n命令执行结束，返回码：{self.process.returncode}")
        print(f"完整日志：{self.log_path}")

    def stop(self):
        """强制停止命令和监控"""
        if self.running and self.process.poll() is None:
            print("\n强制终止命令...")
            try:
                # 终止主进程及其所有子进程
                main_process = psutil.Process(self.process.pid)
                for p in main_process.children(recursive=True):
                    p.send_signal(signal.SIGTERM)
                main_process.send_signal(signal.SIGTERM)
            except Exception as e:
                print(f"终止进程失败：{e}")
        self.running = False


if __name__ == "__main__":
    # 解析命令行参数
    if len(sys.argv) != 3:
        print("用法：python test.py <命令编号> <监控间隔秒数>")
        print("命令编号：1 = build 命令，2 = up 命令")
        print("示例：")
        print("  执行build命令，每1秒监控一次：python test.py 1 1")
        print("  执行up命令，每2秒监控一次：python test.py 2 2")
        sys.exit(1)
    
    try:
        command_idx = int(sys.argv[1])
        interval = float(sys.argv[2])
    except ValueError:
        print("错误：参数必须是数字（命令编号为1或2，间隔为正数）")
        sys.exit(1)
    
    # 启动监控
    try:
        monitor = DockerCommandMonitor(command_idx, interval)
        monitor.start()
    except KeyboardInterrupt:
        monitor.stop()
        print("\n用户中断，监控已停止")
    except Exception as e:
        print(f"执行失败：{e}")