# Purpose: shared runtime image for generated SeedEMU brdnode and rs_ix nodes.
# Inputs: bind-mounted node files from output/<service>/ at container runtime.
# Outputs: seedemu-fullmesh-node:<tag> built by buildFastFullMesh.sh.
# Side effects: installs iperf3 and prepares the bird example config path.
# Expected context: build from fullMeshExample.

FROM handsonsecurity/seedemu-router:2.0

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends iperf3 \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /usr/share/doc/bird2/examples/ \
    && touch /usr/share/doc/bird2/examples/bird.conf

CMD ["/bin/bash", "/start.sh"]
