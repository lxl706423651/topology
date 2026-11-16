/*
 *	BIRD -- Selected Route Delta Event Protocol
 *
 *	This protocol subscribes to a routing table and writes selected-route
 *	change events to a tab-separated file. It intentionally does not touch
 *	the kernel FIB; an external writer can consume this file and batch
 *	updates with ip -force -batch to avoid per-router RTNL bursts.
 *
 *	Can be freely distributed and used under the terms of the GNU GPL.
 */

#undef LOCAL_DEBUG

#include <errno.h>
#include <stdio.h>

#include "nest/bird.h"
#include "nest/cli.h"
#include "nest/iface.h"
#include "nest/protocol.h"
#include "nest/route.h"
#include "conf/conf.h"
#include "filter/filter.h"
#include "lib/string.h"
#include "lib/timer.h"

#include "delta.h"

static inline struct delta_proto *
delta_proto(struct proto *P)
{
  return (struct delta_proto *) P;
}

static const char *
delta_table_name(struct channel *c)
{
  return (c && c->table && c->table->name) ? c->table->name : "-";
}

static void
delta_write_marker(struct delta_proto *p, struct channel *c, const char *marker, int initial)
{
  if (!p->event_file)
    return;

  fprintf(p->event_file, "%lld\tM\t%s\t%s\t%s\t%d\n",
	  (long long) current_real_time(), p->p.name, delta_table_name(c), marker, initial);
  p->markers_written++;
}

static void
delta_write_route(struct delta_proto *p, struct channel *c, net *net, rte *new, rte *old UNUSED)
{
  char prefix[NET_MAX_TEXT_LENGTH + 1];
  const char *action = new ? "replace" : "delete";
  rte *src = new ? new : old;
  rta *a = src ? src->attrs : NULL;
  uint nh_count = 0;

  if (!p->event_file)
    return;

  bsnprintf(prefix, sizeof(prefix), "%N", net->n.addr);

  if (a && (a->dest == RTD_UNICAST))
    for (struct nexthop *nh = &a->nh; nh; nh = nh->next)
      nh_count++;

  fprintf(p->event_file, "%lld\tR\t%s\t%s\t%s\t%s\t%s\t%u\t%u",
	  (long long) current_real_time(),
	  p->p.name,
	  delta_table_name(c),
	  action,
	  prefix,
	  a ? rta_dest_name(a->dest) : "-",
	  p->metric,
	  nh_count);

  if (a && (a->dest == RTD_UNICAST))
    for (struct nexthop *nh = &a->nh; nh; nh = nh->next)
    {
      char gw[IPA_MAX_TEXT_LENGTH + 1];
      const char *iface = nh->iface ? nh->iface->name : "-";
      uint weight = nh->weight + 1;

      if (ipa_nonzero(nh->gw))
	bsnprintf(gw, sizeof(gw), "%I", nh->gw);
      else
	bsnprintf(gw, sizeof(gw), "-");

      fprintf(p->event_file, "\t%s\t%s\t%u\t%u", gw, iface, (uint) nh->flags, weight);
    }

  fprintf(p->event_file, "\n");
  p->events_written++;
}

static void
delta_rt_notify(struct proto *P, struct channel *c, net *net, rte *new, rte *old)
{
  struct delta_proto *p = delta_proto(P);

  if (p->skipping_initial_feed)
  {
    p->initial_events_skipped++;
    return;
  }

  delta_write_route(p, c, net, new, old);
}

static void
delta_feed_begin(struct channel *c, int initial)
{
  struct delta_proto *p = delta_proto(c->proto);

  p->skipping_initial_feed = p->ignore_initial_feed && initial;
  delta_write_marker(p, c, "feed_begin", initial);
}

static void
delta_feed_end(struct channel *c)
{
  struct delta_proto *p = delta_proto(c->proto);

  delta_write_marker(p, c, "feed_end", p->skipping_initial_feed);
  p->skipping_initial_feed = 0;
}

static struct proto *
delta_init(struct proto_config *CF)
{
  struct proto *P = proto_new(CF);
  struct delta_proto *p = delta_proto(P);
  struct delta_config *cf = (struct delta_config *) CF;

  P->main_channel = proto_add_channel(P, proto_cf_main_channel(CF));
  P->rt_notify = delta_rt_notify;
  P->feed_begin = delta_feed_begin;
  P->feed_end = delta_feed_end;

  p->event_file_name = cf->event_file;
  p->metric = cf->metric;
  p->ignore_initial_feed = cf->ignore_initial_feed;

  return P;
}

static int
delta_start(struct proto *P)
{
  struct delta_proto *p = delta_proto(P);

  p->event_file = fopen(p->event_file_name, "a");
  if (!p->event_file)
  {
    log(L_ERR "Delta %s: unable to open event file %s: %m", P->name, p->event_file_name);
    return PS_DOWN;
  }

  setvbuf(p->event_file, NULL, _IOLBF, 0);
  fprintf(p->event_file, "# bird-delta-v1 ts_us type proto table action prefix dest metric nh_count [gw iface flags weight]...\n");
  delta_write_marker(p, P->main_channel, "start", 0);

  return PS_UP;
}

static int
delta_shutdown(struct proto *P)
{
  struct delta_proto *p = delta_proto(P);

  delta_write_marker(p, P->main_channel, "shutdown", 0);

  if (p->event_file)
  {
    fclose(p->event_file);
    p->event_file = NULL;
  }

  return PS_DOWN;
}

static int
delta_reconfigure(struct proto *P, struct proto_config *CF)
{
  struct delta_proto *p = delta_proto(P);
  struct delta_config *cf = (struct delta_config *) CF;

  if (bstrcmp(p->event_file_name, cf->event_file) ||
      (p->metric != cf->metric) ||
      (p->ignore_initial_feed != cf->ignore_initial_feed))
    return 0;

  if (!proto_configure_channel(P, &P->main_channel, proto_cf_main_channel(CF)))
    return 0;

  p->event_file_name = cf->event_file;
  p->metric = cf->metric;
  p->ignore_initial_feed = cf->ignore_initial_feed;

  return 1;
}

static void
delta_postconfig(struct proto_config *CF)
{
  struct delta_config *cf = (struct delta_config *) CF;

  if (cf->p.class == SYM_TEMPLATE)
    return;

  if (!proto_cf_main_channel(CF))
    cf_error("Channel not specified");

  if (cf->merge_paths)
  {
    struct channel_config *cc = proto_cf_main_channel(CF);
    cc->ra_mode = RA_MERGED;
    cc->merge_limit = cf->merge_paths;
  }
}

static void
delta_copy_config(struct proto_config *dest, struct proto_config *src)
{
  struct delta_config *d = (struct delta_config *) dest;
  struct delta_config *s = (struct delta_config *) src;

  d->event_file = s->event_file;
  d->metric = s->metric;
  d->merge_paths = s->merge_paths;
  d->ignore_initial_feed = s->ignore_initial_feed;
}

static void
delta_show_proto_info(struct proto *P)
{
  struct delta_proto *p = delta_proto(P);

  cli_msg(-1006, "  Event file: %s", p->event_file_name);
  cli_msg(-1006, "  FIB metric: %u", p->metric);
  cli_msg(-1006, "  Ignore initial feed: %s", p->ignore_initial_feed ? "yes" : "no");
  cli_msg(-1006, "  Route events written: %u", p->events_written);
  cli_msg(-1006, "  Initial route events skipped: %u", p->initial_events_skipped);
  cli_msg(-1006, "  Marker events written: %u", p->markers_written);
}

struct protocol proto_delta = {
  .name =		"Delta",
  .template =		"delta%d",
  .class =		PROTOCOL_DELTA,
  .channel_mask =	NB_IP,
  .proto_size =		sizeof(struct delta_proto),
  .config_size =	sizeof(struct delta_config),
  .postconfig =		delta_postconfig,
  .init =		delta_init,
  .start =		delta_start,
  .shutdown =		delta_shutdown,
  .reconfigure =	delta_reconfigure,
  .copy_config =	delta_copy_config,
  .show_proto_info =	delta_show_proto_info,
};

void
delta_build(void)
{
  proto_build(&proto_delta);
}
