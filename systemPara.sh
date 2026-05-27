echo "=== PID Usage ==="
printf "Used: %s / Max: %s\n" $(ps -eLf | wc -l) $(cat /proc/sys/kernel/pid_max)

echo -e "\n=== ARP Table Usage ==="
printf "Entries: %s / Max(gc_thresh3): %s\n" $(ip neigh | wc -l) $(sysctl -n net.ipv4.neigh.default.gc_thresh3)

echo -e "\n=== File Descriptors (System) ==="
usage=$(cat /proc/sys/fs/file-nr | awk '{print $1}')
limit=$(cat /proc/sys/fs/file-nr | awk '{print $3}')
printf "Used: %s / Max: %s\n" $usage $limit

echo -e "\n=== Conntrack Usage ==="
if [ -f /proc/sys/net/netfilter/nf_conntrack_count ]; then
    printf "Used: %s / Max: %s\n" $(cat /proc/sys/net/netfilter/nf_conntrack_count) $(cat /proc/sys/net/netfilter/nf_conntrack_max)
else
    echo "Conntrack module not loaded or not available."
fi
