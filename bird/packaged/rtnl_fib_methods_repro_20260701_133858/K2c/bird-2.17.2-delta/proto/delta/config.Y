/*
 *	BIRD -- Selected Route Delta Event Protocol Configuration
 *
 *	(c) 2026 local RTNL experiment
 *
 *	Can be freely distributed and used under the terms of the GNU GPL.
 */

CF_HDR

#include "filter/filter.h"
#include "proto/delta/delta.h"

CF_DEFINES

#define DELTA_CFG ((struct delta_config *) this_proto)

CF_DECLS

CF_KEYWORDS(DELTA, EVENT, MERGE, PATHS, LIMIT, METRIC, IGNORE, INITIAL, FEED)

%type <i> delta_mp_limit

CF_GRAMMAR

proto: delta_proto '}' ;

delta_proto_start: proto_start DELTA
{
  this_proto = proto_config_new(&proto_delta, $1);
  DELTA_CFG->event_file = "/run/bird/delta-events.tsv";
  DELTA_CFG->metric = 32;
  DELTA_CFG->merge_paths = 16;
  DELTA_CFG->ignore_initial_feed = 0;
};

delta_proto:
   delta_proto_start proto_name '{'
 | delta_proto delta_proto_item ';'
 | delta_proto proto_item ';'
 ;

delta_mp_limit:
   /* empty */ { $$ = 16; }
 | LIMIT expr  { $$ = $2; if (($2 <= 0) || ($2 > 255)) cf_error("Merge paths limit must be in range 1-255"); }
 ;

delta_proto_item:
   proto_channel { this_proto->net_type = $1->net_type; }
 | EVENT text { DELTA_CFG->event_file = $2; }
 | METRIC expr { DELTA_CFG->metric = $2; }
 | MERGE PATHS bool delta_mp_limit { DELTA_CFG->merge_paths = $3 ? $4 : 0; }
 | IGNORE INITIAL FEED bool { DELTA_CFG->ignore_initial_feed = $4; }
 ;


CF_CODE

CF_END
