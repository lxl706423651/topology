// SPDX-License-Identifier: GPL-2.0
// Minimal TC dataplane for SeedEMU feasibility tests.
//
// The program performs IPv4 LPM lookup in a BPF map and redirects the packet to
// a configured egress ifindex using bpf_redirect_neigh(). It intentionally does
// not update Linux main FIB.

#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/in.h>
#include <linux/ip.h>
#include <linux/pkt_cls.h>
#include <stddef.h>

#ifndef AF_INET
#define AF_INET 2
#endif

#ifndef bpf_htons
#define bpf_htons(x) __builtin_bswap16(x)
#endif

#ifndef BPF_F_RECOMPUTE_CSUM
#define BPF_F_RECOMPUTE_CSUM (1ULL << 0)
#endif

#ifndef __section
#define __section(NAME) __attribute__((section(NAME), used))
#endif

#ifndef SEC
#define SEC(NAME) __section(NAME)
#endif

#ifndef __uint
#define __uint(name, val) int (*name)[val]
#endif

#ifndef __type
#define __type(name, val) typeof(val) *name
#endif

struct route_key {
	__u32 prefixlen;
	__u32 addr;
};

struct route_val {
	__u32 ifindex;
	__be32 nh;
};

struct counters {
	__u64 packets;
	__u64 redirects;
	__u64 misses;
	__u64 drops;
	__u64 ttl_drops;
	__u64 csum_drops;
	__u64 store_drops;
};

struct {
	__uint(type, BPF_MAP_TYPE_LPM_TRIE);
	__uint(max_entries, 262144);
	__uint(map_flags, BPF_F_NO_PREALLOC);
	__type(key, struct route_key);
	__type(value, struct route_val);
} routes SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, __u32);
	__type(value, struct counters);
} stats SEC(".maps");

static void *(*bpf_map_lookup_elem)(void *map, const void *key) =
	(void *)BPF_FUNC_map_lookup_elem;
static long (*bpf_redirect_neigh)(__u32 ifindex, struct bpf_redir_neigh *params,
				  int plen, __u64 flags) =
	(void *)BPF_FUNC_redirect_neigh;
#ifdef SEED_BPF_DECREMENT_TTL
static long (*bpf_skb_store_bytes)(struct __sk_buff *skb, __u32 offset,
				   const void *from, __u32 len, __u64 flags) =
	(void *)BPF_FUNC_skb_store_bytes;
#endif

static __always_inline void bump_counter(__u32 field)
{
	__u32 key = 0;
	struct counters *c = bpf_map_lookup_elem(&stats, &key);

	if (!c)
		return;

	if (field == 0)
		__sync_fetch_and_add(&c->packets, 1);
	else if (field == 1)
		__sync_fetch_and_add(&c->redirects, 1);
	else if (field == 2)
		__sync_fetch_and_add(&c->misses, 1);
	else if (field == 3)
		__sync_fetch_and_add(&c->drops, 1);
	else if (field == 4)
		__sync_fetch_and_add(&c->ttl_drops, 1);
	else if (field == 5)
		__sync_fetch_and_add(&c->csum_drops, 1);
	else if (field == 6)
		__sync_fetch_and_add(&c->store_drops, 1);
}

SEC("tc/ingress")
int tc_router(struct __sk_buff *skb)
{
	void *data = (void *)(long)skb->data;
	void *data_end = (void *)(long)skb->data_end;
	struct ethhdr *eth = data;
	struct iphdr *ip;
	struct route_key key = {};
	struct route_val *val;
	struct bpf_redir_neigh neigh = {};
#ifdef SEED_BPF_DECREMENT_TTL
	__u8 ttl;
#endif

	bump_counter(0);

	if ((void *)(eth + 1) > data_end)
		return TC_ACT_OK;

	if (eth->h_proto != __constant_htons(ETH_P_IP))
		return TC_ACT_OK;

	ip = (void *)(eth + 1);
	if ((void *)(ip + 1) > data_end)
		return TC_ACT_OK;

#ifdef SEED_BPF_DECREMENT_TTL
	if (ip->ttl <= 1) {
		bump_counter(3);
		bump_counter(4);
		return TC_ACT_SHOT;
	}
#endif

	key.prefixlen = 32;
	key.addr = ip->daddr;
	val = bpf_map_lookup_elem(&routes, &key);
	if (!val) {
		bump_counter(2);
		return TC_ACT_OK;
	}

#ifdef SEED_BPF_DECREMENT_TTL
	ttl = ip->ttl - 1;
	if (bpf_skb_store_bytes(skb, ETH_HLEN + offsetof(struct iphdr, ttl),
				&ttl, sizeof(ttl), BPF_F_RECOMPUTE_CSUM) < 0) {
		bump_counter(3);
		bump_counter(6);
		return TC_ACT_SHOT;
	}
#endif

	neigh.nh_family = AF_INET;
	neigh.ipv4_nh = val->nh;
	bump_counter(1);
	return bpf_redirect_neigh(val->ifindex, &neigh, sizeof(neigh), 0);
}

char _license[] SEC("license") = "GPL";
