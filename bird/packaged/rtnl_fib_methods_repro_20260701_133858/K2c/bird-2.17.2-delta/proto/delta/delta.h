/*
 *	BIRD -- Selected Route Delta Event Protocol
 *
 *	(c) 2026 local RTNL experiment
 *
 *	Can be freely distributed and used under the terms of the GNU GPL.
 */

#ifndef _BIRD_DELTA_H_
#define _BIRD_DELTA_H_

#include <stdio.h>

struct delta_config {
  struct proto_config p;
  const char *event_file;
  uint metric;
  uint merge_paths;
  uint ignore_initial_feed;
};

struct delta_proto {
  struct proto p;
  FILE *event_file;
  const char *event_file_name;
  uint metric;
  uint ignore_initial_feed;
  uint skipping_initial_feed;
  uint events_written;
  uint initial_events_skipped;
  uint markers_written;
};

#endif
