// SPDX-License-Identifier: GPL-2.0
#include <linux/bird_rtbulk.h>
#include <linux/ktime.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/rtnetlink.h>
#include <net/genetlink.h>
#include <net/ip_fib.h>
#include <net/net_namespace.h>
#include <net/netns/generic.h>
#include <net/netlink.h>
#include "fib_lookup.h"

struct bird_rtbulk_stats {
	u32 success_count;
	u32 fail_count;
	u32 first_fail_index;
	s32 first_fail_errno;
	u64 apply_nsec;
};

static struct genl_family bird_rtbulk_genl_family;

struct bird_rtbulk_net {
	struct mutex route_lock;
};

static unsigned int bird_rtbulk_net_id;

static struct bird_rtbulk_net *bird_rtbulk_pernet(struct net *net)
{
	return net_generic(net, bird_rtbulk_net_id);
}

void bird_rtbulk_route_lock(struct net *net)
{
	mutex_lock(&bird_rtbulk_pernet(net)->route_lock);
}

void bird_rtbulk_route_unlock(struct net *net)
{
	mutex_unlock(&bird_rtbulk_pernet(net)->route_lock);
}

bool bird_rtbulk_k3a_route_allowed(const struct fib_config *cfg)
{
	return cfg->fc_table == RT_TABLE_MAIN &&
	       cfg->fc_protocol == RTPROT_BIRD &&
	       cfg->fc_priority == 232 &&
	       !cfg->fc_nh_id &&
	       !cfg->fc_mp &&
	       !cfg->fc_mx &&
	       !cfg->fc_encap &&
	       cfg->fc_type != RTN_LOCAL;
}

static int __net_init bird_rtbulk_net_init(struct net *net)
{
	mutex_init(&bird_rtbulk_pernet(net)->route_lock);
	return 0;
}

static struct pernet_operations bird_rtbulk_net_ops = {
	.init = bird_rtbulk_net_init,
	.id = &bird_rtbulk_net_id,
	.size = sizeof(struct bird_rtbulk_net),
};

static const struct nla_policy bird_rtbulk_policy[BIRD_RTBULK_A_MAX + 1] = {
	[BIRD_RTBULK_A_OPS] = { .type = NLA_BINARY },
};

static int bird_rtbulk_validate_op_v4(const struct bird_rtbulk_op_v1 *op)
{
	if (op->dst_len > 32)
		return -EINVAL;

	if (op->dst_len < 32 && (ntohl(op->dst) << op->dst_len))
		return -EINVAL;

	return 0;
}

static void bird_rtbulk_build_cfg_v4(struct net *net,
				     const struct bird_rtbulk_op_v1 *op,
				     struct fib_config *cfg)
{
	u8 type = op->type;

	if (!type && op->op != BIRD_RTBULK_OP_DEL)
		type = RTN_UNICAST;

	memset(cfg, 0, sizeof(*cfg));
	cfg->fc_dst_len = op->dst_len;
	cfg->fc_dst = op->dst;
	cfg->fc_table = op->table ? op->table : RT_TABLE_MAIN;
	cfg->fc_protocol = op->protocol ? op->protocol : RTPROT_STATIC;
	cfg->fc_scope = op->scope;
	cfg->fc_type = type;
	cfg->fc_skip_notify = true;
	cfg->fc_flags = op->flags;
	cfg->fc_priority = op->metric;
	cfg->fc_oif = op->ifindex;
	cfg->fc_nlflags = NLM_F_REQUEST;
	cfg->fc_nlinfo.nl_net = net;

	if (op->gateway) {
		cfg->fc_gw_family = AF_INET;
		cfg->fc_gw4 = op->gateway;
	}
}

static bool bird_rtbulk_cfg_same_info(const struct fib_config *a,
				      const struct fib_config *b)
{
	return a->fc_table == b->fc_table &&
	       a->fc_protocol == b->fc_protocol &&
	       a->fc_scope == b->fc_scope &&
	       a->fc_type == b->fc_type &&
	       a->fc_flags == b->fc_flags &&
	       a->fc_priority == b->fc_priority &&
	       a->fc_oif == b->fc_oif &&
	       a->fc_gw_family == b->fc_gw_family &&
	       a->fc_gw4 == b->fc_gw4 &&
	       a->fc_nh_id == b->fc_nh_id &&
	       a->fc_mp == b->fc_mp &&
	       a->fc_mx == b->fc_mx &&
	       a->fc_encap == b->fc_encap &&
	       a->fc_prefsrc == b->fc_prefsrc;
}

static int bird_rtbulk_apply_one_v4(struct net *net,
				    const struct bird_rtbulk_op_v1 *op,
				    struct netlink_ext_ack *extack)
{
	struct fib_config cfg;
	struct fib_table *tb;
	int err;

	err = bird_rtbulk_validate_op_v4(op);
	if (err)
		return err;

	bird_rtbulk_build_cfg_v4(net, op, &cfg);

	switch (op->op) {
	case BIRD_RTBULK_OP_ADD:
		cfg.fc_nlflags |= NLM_F_CREATE | NLM_F_EXCL;
		tb = fib_new_table(net, cfg.fc_table);
		if (!tb)
			return -ENOBUFS;
		err = fib_table_insert(net, tb, &cfg, extack);
		break;
	case BIRD_RTBULK_OP_REPLACE:
		cfg.fc_nlflags |= NLM_F_CREATE | NLM_F_REPLACE;
		tb = fib_new_table(net, cfg.fc_table);
		if (!tb)
			return -ENOBUFS;
		err = fib_table_insert(net, tb, &cfg, extack);
		break;
	case BIRD_RTBULK_OP_DEL:
		tb = fib_get_table(net, cfg.fc_table);
		if (!tb)
			return -ESRCH;
		err = fib_table_delete(net, tb, &cfg, extack);
		break;
	default:
		err = -EINVAL;
		break;
	}

	if (!err && cfg.fc_type == RTN_LOCAL)
		net->ipv4.fib_has_custom_local_routes = true;

	return err;
}

static bool bird_rtbulk_can_use_cached_fi(struct net *net,
					  const struct bird_rtbulk_op_v1 *ops,
					  int count,
					  struct fib_config *base_cfg)
{
	struct fib_config cfg;
	int i;

	if (count <= 1)
		return false;

	if (ops[0].op != BIRD_RTBULK_OP_ADD &&
	    ops[0].op != BIRD_RTBULK_OP_REPLACE)
		return false;

	if (bird_rtbulk_validate_op_v4(&ops[0]))
		return false;

	bird_rtbulk_build_cfg_v4(net, &ops[0], base_cfg);
	for (i = 1; i < count; i++) {
		if (ops[i].op != ops[0].op)
			return false;
		if (bird_rtbulk_validate_op_v4(&ops[i]))
			return false;
		bird_rtbulk_build_cfg_v4(net, &ops[i], &cfg);
		if (!bird_rtbulk_cfg_same_info(base_cfg, &cfg))
			return false;
	}

	return true;
}

static void bird_rtbulk_fib_info_tree_hold(struct fib_info *fi)
{
	/*
	 * fib_table_insert_info() consumes one fib_treeref: successful
	 * inserts attach it to the new fib_alias, while error/no-op paths
	 * drop it with fib_release_info().
	 */
	refcount_inc(&fi->fib_treeref);
}

static void bird_rtbulk_apply_many_v4(struct net *net,
				      const struct bird_rtbulk_op_v1 *ops,
				      int count,
				      struct netlink_ext_ack *extack,
				      struct bird_rtbulk_stats *stats,
				      int *global_index)
{
	struct fib_config base_cfg;
	struct fib_info *fi;
	struct fib_table *tb;
	int i;

	if (!bird_rtbulk_can_use_cached_fi(net, ops, count, &base_cfg))
		goto slow_path;

	base_cfg.fc_nlflags |= ops[0].op == BIRD_RTBULK_OP_ADD ?
			       NLM_F_CREATE | NLM_F_EXCL :
			       NLM_F_CREATE | NLM_F_REPLACE;
	tb = fib_new_table(net, base_cfg.fc_table);
	if (!tb)
		goto slow_path;

	fi = fib_create_info(&base_cfg, extack);
	if (IS_ERR(fi))
		goto slow_path;

	for (i = 0; i < count; i++, (*global_index)++) {
		struct fib_config cfg;
		int err;

		bird_rtbulk_build_cfg_v4(net, &ops[i], &cfg);
			cfg.fc_nlflags |= ops[i].op == BIRD_RTBULK_OP_ADD ?
					  NLM_F_CREATE | NLM_F_EXCL :
					  NLM_F_CREATE | NLM_F_REPLACE;
			bird_rtbulk_fib_info_tree_hold(fi);
			err = fib_table_insert_info(net, tb, &cfg, fi, extack);
			if (err) {
				stats->fail_count++;
			if (!stats->first_fail_errno) {
				stats->first_fail_index = *global_index;
				stats->first_fail_errno = err;
			}
		} else {
			stats->success_count++;
		}
	}
	fib_release_info(fi);
	return;

slow_path:
	for (i = 0; i < count; i++, (*global_index)++) {
		int err;

		err = bird_rtbulk_apply_one_v4(net, &ops[i], extack);
		if (err) {
			stats->fail_count++;
			if (!stats->first_fail_errno) {
				stats->first_fail_index = *global_index;
				stats->first_fail_errno = err;
			}
		} else {
			stats->success_count++;
		}
	}
}

static int bird_rtbulk_reply(struct genl_info *info,
			     const struct bird_rtbulk_stats *stats)
{
	struct sk_buff *msg;
	void *hdr;
	int err;

	msg = genlmsg_new(NLMSG_DEFAULT_SIZE, GFP_KERNEL);
	if (!msg)
		return -ENOMEM;

	hdr = genlmsg_put_reply(msg, info, &bird_rtbulk_genl_family, 0,
				BIRD_RTBULK_CMD_APPLY_V4);
	if (!hdr) {
		nlmsg_free(msg);
		return -ENOMEM;
	}

	err = nla_put_u32(msg, BIRD_RTBULK_A_SUCCESS_COUNT,
			  stats->success_count);
	if (err)
		goto cancel;
	err = nla_put_u32(msg, BIRD_RTBULK_A_FAIL_COUNT, stats->fail_count);
	if (err)
		goto cancel;
	err = nla_put_u32(msg, BIRD_RTBULK_A_FIRST_FAIL_INDEX,
			  stats->first_fail_index);
	if (err)
		goto cancel;
	err = nla_put_s32(msg, BIRD_RTBULK_A_FIRST_FAIL_ERRNO,
			  stats->first_fail_errno);
	if (err)
		goto cancel;
	err = nla_put_u64_64bit(msg, BIRD_RTBULK_A_APPLY_NSEC,
				stats->apply_nsec, BIRD_RTBULK_A_UNSPEC);
	if (err)
		goto cancel;

	genlmsg_end(msg, hdr);
	return genlmsg_reply(msg, info);

cancel:
	genlmsg_cancel(msg, hdr);
	nlmsg_free(msg);
	return err;
}

static int bird_rtbulk_apply_v4(struct sk_buff *skb, struct genl_info *info)
{
	const struct bird_rtbulk_op_v1 *ops;
	struct bird_rtbulk_stats stats = {};
	struct net *net;
	struct nlattr *attr;
	ktime_t start;
	int remaining;
	int global_index = 0;
	int count;

	if (!info->attrs[BIRD_RTBULK_A_OPS])
		return -EINVAL;

	net = genl_info_net(info);
	start = ktime_get();

	/*
	 * K2 experiment: intentionally avoid the global RTNL in this private
	 * synthetic path and serialize only writers targeting the same netns.
	 *
	 * fib_table_insert/delete are documented as RTNL-protected upstream
	 * APIs, so this mode is not a production replacement.  It is only for
	 * isolated SeedEMU synthetic netns tests with no concurrent device,
	 * nexthop, VRF, or fib-rules mutation.
	 */
	bird_rtbulk_route_lock(net);
	nla_for_each_attr(attr, nlmsg_attrdata(info->nlhdr, GENL_HDRLEN),
			  nlmsg_attrlen(info->nlhdr, GENL_HDRLEN), remaining) {
		if (nla_type(attr) != BIRD_RTBULK_A_OPS)
			continue;
		if (nla_len(attr) % sizeof(*ops)) {
			stats.fail_count++;
			if (!stats.first_fail_errno) {
				stats.first_fail_index = global_index;
				stats.first_fail_errno = -EINVAL;
			}
			continue;
		}
		count = nla_len(attr) / sizeof(*ops);
		ops = nla_data(attr);
		if (count) {
			bird_rtbulk_apply_many_v4(net, ops, count, info->extack,
						 &stats, &global_index);
		}
	}
	bird_rtbulk_route_unlock(net);

	stats.apply_nsec = ktime_to_ns(ktime_sub(ktime_get(), start));
	return bird_rtbulk_reply(info, &stats);
}

static const struct genl_small_ops bird_rtbulk_ops[] = {
	{
		.cmd = BIRD_RTBULK_CMD_APPLY_V4,
		.validate = GENL_DONT_VALIDATE_STRICT | GENL_DONT_VALIDATE_DUMP,
		.doit = bird_rtbulk_apply_v4,
		.flags = GENL_ADMIN_PERM,
	},
};

static struct genl_family bird_rtbulk_genl_family __ro_after_init = {
	.name = BIRD_RTBULK_GENL_NAME,
	.version = BIRD_RTBULK_GENL_VERSION,
	.maxattr = BIRD_RTBULK_A_MAX,
	.policy = bird_rtbulk_policy,
	.netnsok = true,
	.parallel_ops = true,
	.module = THIS_MODULE,
	.small_ops = bird_rtbulk_ops,
	.n_small_ops = ARRAY_SIZE(bird_rtbulk_ops),
	.resv_start_op = __BIRD_RTBULK_CMD_MAX,
};

static int __init bird_rtbulk_init(void)
{
	int err;

	err = register_pernet_subsys(&bird_rtbulk_net_ops);
	if (err)
		return err;

	err = genl_register_family(&bird_rtbulk_genl_family);
	if (err)
		unregister_pernet_subsys(&bird_rtbulk_net_ops);

	return err;
}
subsys_initcall(bird_rtbulk_init);
