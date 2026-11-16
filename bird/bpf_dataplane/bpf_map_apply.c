// SPDX-License-Identifier: GPL-2.0
// Fast userspace updater for SeedEMU BPF dataplane route maps.
//
// The Python controller writes fixed-size binary records and this helper applies
// them with direct bpf(2) syscalls. This avoids bpftool's text batch parser in
// reconvergence experiments while keeping the TC program and map layout simple.

#include <errno.h>
#include <fcntl.h>
#include <linux/bpf.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef BPF_ANY
#define BPF_ANY 0
#endif

struct route_key {
	uint32_t prefixlen;
	uint32_t addr;
};

struct route_val {
	uint32_t ifindex;
	uint32_t nh;
};

struct route_op {
	uint8_t op;
	uint8_t pad[3];
	struct route_key key;
	struct route_val value;
};

static int bpf_sys(enum bpf_cmd cmd, union bpf_attr *attr)
{
	return (int)syscall(__NR_bpf, cmd, attr, sizeof(*attr));
}

static int bpf_obj_get_path(const char *path)
{
	union bpf_attr attr;

	memset(&attr, 0, sizeof(attr));
	attr.pathname = (uint64_t)(uintptr_t)path;
	return bpf_sys(BPF_OBJ_GET, &attr);
}

static int bpf_update(int map_fd, const struct route_key *key,
		      const struct route_val *value)
{
	union bpf_attr attr;

	memset(&attr, 0, sizeof(attr));
	attr.map_fd = (uint32_t)map_fd;
	attr.key = (uint64_t)(uintptr_t)key;
	attr.value = (uint64_t)(uintptr_t)value;
	attr.flags = BPF_ANY;
	return bpf_sys(BPF_MAP_UPDATE_ELEM, &attr);
}

static int bpf_delete(int map_fd, const struct route_key *key)
{
	union bpf_attr attr;

	memset(&attr, 0, sizeof(attr));
	attr.map_fd = (uint32_t)map_fd;
	attr.key = (uint64_t)(uintptr_t)key;
	return bpf_sys(BPF_MAP_DELETE_ELEM, &attr);
}

int main(int argc, char **argv)
{
	const char *map_path;
	const char *ops_path;
	struct route_op op;
	uint64_t updates = 0;
	uint64_t deletes = 0;
	uint64_t delete_misses = 0;
	int map_fd;
	int ops_fd;

	if (argc != 3) {
		fprintf(stderr, "usage: %s <pinned-map-path> <ops-file>\n", argv[0]);
		return 2;
	}

	map_path = argv[1];
	ops_path = argv[2];
	map_fd = bpf_obj_get_path(map_path);
	if (map_fd < 0) {
		fprintf(stderr, "BPF_OBJ_GET %s failed: %s\n", map_path, strerror(errno));
		return 1;
	}

	ops_fd = open(ops_path, O_RDONLY | O_CLOEXEC);
	if (ops_fd < 0) {
		fprintf(stderr, "open %s failed: %s\n", ops_path, strerror(errno));
		close(map_fd);
		return 1;
	}

	for (;;) {
		ssize_t n = read(ops_fd, &op, sizeof(op));

		if (n == 0)
			break;
		if (n < 0) {
			fprintf(stderr, "read %s failed: %s\n", ops_path, strerror(errno));
			close(ops_fd);
			close(map_fd);
			return 1;
		}
		if (n != (ssize_t)sizeof(op)) {
			fprintf(stderr, "short record in %s: %zd bytes\n", ops_path, n);
			close(ops_fd);
			close(map_fd);
			return 1;
		}

		if (op.op == 1) {
			if (bpf_update(map_fd, &op.key, &op.value) < 0) {
				fprintf(stderr, "BPF_MAP_UPDATE_ELEM failed after %llu updates: %s\n",
					(unsigned long long)updates, strerror(errno));
				close(ops_fd);
				close(map_fd);
				return 1;
			}
			updates++;
		} else if (op.op == 2) {
			if (bpf_delete(map_fd, &op.key) < 0) {
				if (errno == ENOENT) {
					delete_misses++;
				} else {
					fprintf(stderr, "BPF_MAP_DELETE_ELEM failed after %llu deletes: %s\n",
						(unsigned long long)deletes, strerror(errno));
					close(ops_fd);
					close(map_fd);
					return 1;
				}
			}
			deletes++;
		} else {
			fprintf(stderr, "invalid op %u in %s\n", op.op, ops_path);
			close(ops_fd);
			close(map_fd);
			return 1;
		}
	}

	close(ops_fd);
	close(map_fd);
	printf("updates=%llu deletes=%llu delete_misses=%llu\n",
	       (unsigned long long)updates,
	       (unsigned long long)deletes,
	       (unsigned long long)delete_misses);
	return 0;
}
