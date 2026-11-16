
# fullMeshExample

This directory is a self-contained 1078-scale connectivity example based on
`RR_1897.py` and `real_topology_1078.txt`.

## Flow

```bash
cd /home/lxl/seed-emulator/topology/fullMeshExample

# Generate real_topology_1078_connected.json and compile ./output.
./compileFullMesh.sh

# Build images from ./output/docker-compose.yml.
./buildFullMesh.sh

# Start containers from ./output/docker-compose.yml.
./dockerUpFullMesh.sh

# Test brdnode reachability. rs_ix_ix* services are ignored.
python3 testBrdConnectivity.py --mode root --target-address dummy --wait 60
```

`connectTopology.py` defaults to the `provider-star` strategy. It keeps the
original router/IX membership graph shape: it does not add `ix_ix_transit_edges`
or new IX memberships. It only adds or promotes AS-AS relationships on existing
shared IXs.

`patchRuntimeRouting.py` is run by `compileFullMesh.sh` after SeedEMU compile.
It patches generated brdnode BIRD configs so dummy0 `/32` router addresses are
the ping targets, not IX LAN addresses. The patch also adds stable OSPF timers,
iBGP `next hop self`, and adjacent dummy static routes for iBGP loopback
resolution.

## Faster Build/Up

The generated SeedEMU compose builds one image per node. For this 1078-scale
example that can be very slow. The fast path builds one shared runtime image
and bind-mounts each generated node's config files:

```bash
./buildFastFullMesh.sh
./dockerUpFastFullMesh.sh
python3 testBrdConnectivity.py --compose output/docker-compose-fast.yml --mode root --target-address dummy --wait 60
```

Use `python3 testBrdConnectivity.py --mode all-pairs` only when you need the
full ping matrix; it is much slower at 1078 scale.

## Current Validation Notes

For the 1078 input, the connected JSON keeps the topology node count stable:
880 `brdnode_*` routers plus 198 `rs_ix_ix*` route servers. The report shows
`ix_ix_transit_edges` unchanged at 13470 and `brd_nodes` unchanged at 880.

The fast build/up flow has been run successfully with 1078/1078 topology
containers up and 0 exited containers. Full brdnode reachability is not yet
proven: live tests showed that AS-AS route relationships are added without
node growth, but large-AS internal iBGP/OSPF forwarding still causes dummy0
ping failures on some multi-hop paths.
