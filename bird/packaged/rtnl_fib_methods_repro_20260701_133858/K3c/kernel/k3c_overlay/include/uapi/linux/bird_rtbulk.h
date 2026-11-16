/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
#ifndef _UAPI_LINUX_BIRD_RTBULK_H
#define _UAPI_LINUX_BIRD_RTBULK_H

#include <linux/types.h>

#define BIRD_RTBULK_GENL_NAME "BIRD_RT_BULK"
#define BIRD_RTBULK_GENL_VERSION 1

enum bird_rtbulk_cmd {
	BIRD_RTBULK_CMD_UNSPEC,
	BIRD_RTBULK_CMD_APPLY_V4,
	__BIRD_RTBULK_CMD_MAX,
};
#define BIRD_RTBULK_CMD_MAX (__BIRD_RTBULK_CMD_MAX - 1)

enum bird_rtbulk_attr {
	BIRD_RTBULK_A_UNSPEC,
	BIRD_RTBULK_A_OPS,
	BIRD_RTBULK_A_SUCCESS_COUNT,
	BIRD_RTBULK_A_FAIL_COUNT,
	BIRD_RTBULK_A_FIRST_FAIL_INDEX,
	BIRD_RTBULK_A_FIRST_FAIL_ERRNO,
	BIRD_RTBULK_A_APPLY_NSEC,
	__BIRD_RTBULK_A_MAX,
};
#define BIRD_RTBULK_A_MAX (__BIRD_RTBULK_A_MAX - 1)

enum bird_rtbulk_op {
	BIRD_RTBULK_OP_UNSPEC,
	BIRD_RTBULK_OP_REPLACE,
	BIRD_RTBULK_OP_ADD,
	BIRD_RTBULK_OP_DEL,
};

/*
 * One compact IPv4 route operation.
 *
 * The first experiment intentionally avoids nested netlink attributes so user
 * space can send a dense binary array.  Generic-netlink attributes still have
 * a 16-bit length field, so practical request size is around 1024-2048 ops per
 * message depending on struct growth.
 */
struct bird_rtbulk_op_v1 {
	__u8 op;          /* enum bird_rtbulk_op */
	__u8 dst_len;     /* 0..32 */
	__u8 type;        /* RTN_*; 0 means RTN_UNICAST for add/replace */
	__u8 scope;       /* RT_SCOPE_* */
	__u8 protocol;    /* RTPROT_*; 0 means RTPROT_STATIC */
	__u8 pad[3];
	__be32 dst;
	__be32 gateway;   /* 0 for no gateway */
	__s32 ifindex;    /* 0 for no explicit output device */
	__u32 table;      /* 0 means RT_TABLE_MAIN */
	__u32 metric;
	__u32 flags;      /* RTNH_F_ONLINK, etc.; copied to fib_config.fc_flags */
};

#endif /* _UAPI_LINUX_BIRD_RTBULK_H */
