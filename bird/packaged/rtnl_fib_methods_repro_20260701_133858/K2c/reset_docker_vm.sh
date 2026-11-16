#!/usr/bin/env bash
# Reset Docker state inside the bird-rtnl-dev VM.
#
# Inputs:
#   No arguments                  Reset Docker and rebuild SeedEMU base/router images.
#   --yes                         Accepted for compatibility; destructive reset is default.
#   --rebuild-seedemu-images      Accepted for compatibility; image rebuild is default.
#   --seedemu-root PATH           Optional SeedEMU repo path, default /home/lxl/seed-emulator.
#
# Outputs:
#   Recreated /etc/docker/daemon.json and fresh Docker/containerd data dirs.
#
# Side effects:
#   DANGEROUS: removes all Docker containers, images, networks, volumes,
#   BuildKit cache, Docker/containerd runtime data, and Docker config.
#   Run this inside the VM, not on the host, unless you explicitly intend to
#   destroy the host Docker state.

set -Eeuo pipefail

REBUILD_SEEDEMU_IMAGES=1
SEEDEMU_ROOT="/home/lxl/seed-emulator"
DEFAULT_USER="${SUDO_USER:-${USER:-lxl}}"
DOCKER_TIMEOUT_SECONDS=15

usage() {
    cat <<'EOF'
Usage:
  ./reset_docker_vm.sh [--seedemu-root PATH] [--skip-rebuild-seedemu-images]
  ./reset_docker_vm.sh --yes --rebuild-seedemu-images --seedemu-root /home/lxl/seed-emulator

No arguments are equivalent to:
  ./reset_docker_vm.sh --yes --rebuild-seedemu-images --seedemu-root /home/lxl/seed-emulator

This script is destructive. It first removes Docker objects through the Docker
API when possible, then stops Docker/containerd, removes runtime/cache data,
recreates /etc/docker/daemon.json for the BIRD/RTNL VM, verifies the daemon is
empty, and rebuilds SeedEMU base/router images by default.
EOF
}

log() {
    printf '[INFO] %s\n' "$*"
}

err() {
    printf '[ERROR] %s\n' "$*" >&2
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --yes)
            shift
            ;;
        --rebuild-seedemu-images)
            REBUILD_SEEDEMU_IMAGES=1
            shift
            ;;
        --skip-rebuild-seedemu-images)
            REBUILD_SEEDEMU_IMAGES=0
            shift
            ;;
        --seedemu-root)
            SEEDEMU_ROOT="${2:?missing value for --seedemu-root}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            err "Unknown argument: $1"
            usage
            exit 2
            ;;
    esac
done

if ! sudo -n true >/dev/null 2>&1; then
    err "Passwordless sudo is required inside the VM."
    exit 1
fi

docker_ready() {
    sudo timeout "${DOCKER_TIMEOUT_SECONDS}s" docker info >/dev/null 2>&1
}

docker_cmd() {
    sudo timeout "${DOCKER_TIMEOUT_SECONDS}s" docker "$@"
}

docker_count() {
    local what="$1"
    case "$what" in
        containers)
            { docker_cmd ps -aq 2>/dev/null || true; } | wc -l
            ;;
        running)
            { docker_cmd ps -q 2>/dev/null || true; } | wc -l
            ;;
        images)
            { docker_cmd images -aq 2>/dev/null || true; } | sort -u | wc -l
            ;;
        volumes)
            { docker_cmd volume ls -q 2>/dev/null || true; } | wc -l
            ;;
        custom_networks)
            { docker_cmd network ls --filter type=custom -q 2>/dev/null || true; } | wc -l
            ;;
        *)
            echo 0
            ;;
    esac
}

print_docker_counts() {
    if ! docker_ready; then
        log "Docker daemon is not ready; object counts unavailable."
        return
    fi
    log "Docker counts: containers=$(docker_count containers), running=$(docker_count running), images=$(docker_count images), volumes=$(docker_count volumes), custom_networks=$(docker_count custom_networks)"
}

docker_api_cleanup() {
    if ! docker_ready; then
        log "Docker daemon is not reachable; skipping API cleanup before runtime reset."
        return
    fi

    log "Remove containers through Docker API..."
    { docker_cmd ps -aq 2>/dev/null || true; } | xargs -r sudo timeout "${DOCKER_TIMEOUT_SECONDS}s" docker rm -f >/dev/null 2>&1 || true

    log "Remove user-defined Docker networks through Docker API..."
    { docker_cmd network ls --filter type=custom -q 2>/dev/null || true; } | xargs -r sudo timeout "${DOCKER_TIMEOUT_SECONDS}s" docker network rm >/dev/null 2>&1 || true

    log "Remove Docker volumes through Docker API..."
    { docker_cmd volume ls -q 2>/dev/null || true; } | xargs -r sudo timeout "${DOCKER_TIMEOUT_SECONDS}s" docker volume rm -f >/dev/null 2>&1 || true

    log "Remove Docker images through Docker API..."
    { docker_cmd images -aq 2>/dev/null || true; } | sort -u | xargs -r sudo timeout "${DOCKER_TIMEOUT_SECONDS}s" docker image rm -f >/dev/null 2>&1 || true

    log "Prune Docker build/cache state through Docker API..."
    docker_cmd buildx prune -af >/dev/null 2>&1 || true
    docker_cmd builder prune -af >/dev/null 2>&1 || true
    docker_cmd system prune -af --volumes >/dev/null 2>&1 || true
}

stop_docker_services() {
    log "Stop Docker services and remaining runtime processes..."
    sudo systemctl stop docker.socket docker.service containerd.service 2>/dev/null || true
    sleep 2
    sudo pkill -TERM -x docker-proxy 2>/dev/null || true
    sudo pkill -TERM -x buildkitd 2>/dev/null || true
    sudo pkill -TERM -x dockerd 2>/dev/null || true
    sudo pkill -TERM -x containerd-shim-runc-v2 2>/dev/null || true
    sudo pkill -TERM -x containerd-shim 2>/dev/null || true
    sudo pkill -TERM -x containerd 2>/dev/null || true
    sleep 2
    sudo pkill -KILL -x docker-proxy 2>/dev/null || true
    sudo pkill -KILL -x buildkitd 2>/dev/null || true
    sudo pkill -KILL -x dockerd 2>/dev/null || true
    sudo pkill -KILL -x containerd-shim-runc-v2 2>/dev/null || true
    sudo pkill -KILL -x containerd-shim 2>/dev/null || true
    sudo pkill -KILL -x containerd 2>/dev/null || true
}

unmount_runtime_tree() {
    local root="$1"
    local targets
    if command -v findmnt >/dev/null 2>&1; then
        targets="$(findmnt -Rno TARGET "$root" 2>/dev/null || true)"
        printf '%s\n' "$targets" | sort -r | while IFS= read -r target; do
            [ -n "$target" ] || continue
            sudo umount -lf "$target" 2>/dev/null || true
        done
    else
        targets="$(mount | awk -v root="$root" '$3 ~ "^" root {print $3}' || true)"
        printf '%s\n' "$targets" | sort -r | while IFS= read -r target; do
            [ -n "$target" ] || continue
            sudo umount -lf "$target" 2>/dev/null || true
        done
    fi
}

remove_runtime_data() {
    log "Unmount Docker/containerd runtime mount points..."
    unmount_runtime_tree /var/lib/docker
    unmount_runtime_tree /var/lib/containerd

    log "Remove Docker/containerd runtime data, configs, sockets, and BuildKit metadata..."
    sudo rm -rf /var/lib/docker
    sudo rm -rf /var/lib/containerd
    sudo rm -rf /var/lib/buildkit
    sudo rm -rf /etc/docker
    sudo rm -rf /run/docker
    sudo rm -rf /run/containerd
    sudo rm -rf /var/run/docker
    sudo rm -rf /var/run/containerd
    sudo rm -rf /root/.docker/buildx
    sudo rm -rf "/home/${DEFAULT_USER}/.docker/buildx" 2>/dev/null || true
}

wait_for_docker() {
    log "Wait for Docker daemon..."
    for _ in $(seq 1 60); do
        if docker_ready; then
            return 0
        fi
        sleep 1
    done
    return 1
}

verify_empty_runtime() {
    local containers images volumes networks
    containers="$(docker_count containers)"
    images="$(docker_count images)"
    volumes="$(docker_count volumes)"
    networks="$(docker_count custom_networks)"
    print_docker_counts
    if [ "$containers" -ne 0 ] || [ "$images" -ne 0 ] || [ "$volumes" -ne 0 ] || [ "$networks" -ne 0 ]; then
        err "Docker runtime is not empty after reset: containers=${containers}, images=${images}, volumes=${volumes}, custom_networks=${networks}"
        exit 1
    fi
}

resolve_seedemu_image_dir() {
    local image_dir="$1"
    local candidate
    for candidate in \
        "${SEEDEMU_ROOT}/docker_images/${image_dir}" \
        "${SEEDEMU_ROOT}/docker_images/amd64/${image_dir}" \
        "${SEEDEMU_ROOT}/docker_images/multiarch/${image_dir}"; do
        if [ -d "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

log "Target host: $(hostname)"
log "No-argument default: reset Docker and rebuild SeedEMU images from ${SEEDEMU_ROOT}."
log "1. Current Docker state before reset..."
print_docker_counts

log "2. Clear Docker objects through the daemon when possible..."
docker_api_cleanup

log "3. Stop Docker/containerd and remove runtime data..."
stop_docker_services
remove_runtime_data

log "4. Recreate /etc/docker/daemon.json for VM..."
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json >/dev/null <<'EOF'
{
  "registry-mirrors": [
    "https://docker.1ms.run",
    "https://docker.anyhub.us.kg",
    "https://dockerhub.icu",
    "https://docker.1panel.live"
  ],
  "features": {
    "buildkit": true
  },
  "storage-driver": "overlay2",
  "exec-opts": [
    "native.cgroupdriver=systemd"
  ],
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "20m",
    "max-file": "3"
  }
}
EOF

log "5. Restart Docker services..."
sudo systemctl daemon-reload
sudo systemctl start containerd
sudo systemctl start docker

if ! wait_for_docker; then
    err "Docker did not become ready."
    sudo systemctl --no-pager status docker || true
    exit 1
fi

docker --version || true
docker compose version || true
docker buildx version || true

log "6. Verify Docker runtime is empty before image rebuild..."
verify_empty_runtime

if [ "$REBUILD_SEEDEMU_IMAGES" -eq 1 ]; then
    log "7. Rebuild SeedEMU base/router images from ${SEEDEMU_ROOT}..."
    for image_dir in seedemu-base seedemu-router; do
        if ! build_dir="$(resolve_seedemu_image_dir "$image_dir")"; then
            err "Missing build directory for ${image_dir} under ${SEEDEMU_ROOT}/docker_images"
            exit 1
        fi
        log "Building ${image_dir} from ${build_dir}..."
        (cd "$build_dir" && docker compose build)
        case "$image_dir" in
            seedemu-base)
                log "Tag handsonsecurity/seedemu-base:2.0 as :latest for local router builds..."
                docker tag handsonsecurity/seedemu-base:2.0 handsonsecurity/seedemu-base:latest
                ;;
            seedemu-router)
                log "Tag handsonsecurity/seedemu-router:2.0 as :latest for compatibility..."
                docker tag handsonsecurity/seedemu-router:2.0 handsonsecurity/seedemu-router:latest
                ;;
        esac
    done
else
    log "7. Skip SeedEMU image rebuild. Use --rebuild-seedemu-images if needed."
fi

log "8. Docker state after reset/rebuild..."
print_docker_counts
log "Docker reset completed."
