# ZED camera via Docker (Fedora host)

Stereolabs only ships the ZED SDK as an Ubuntu `.run` installer, and it hard-requires
CUDA — there's no native Fedora/RPM path. This runs the SDK inside Stereolabs' own
`stereolabs/zed` Docker image (Ubuntu-based, CUDA + SDK preinstalled) instead, so the
container gets the ABI it expects and the Fedora host doesn't need to.

## What's here
- `Dockerfile` — `stereolabs/zed` base + `pyzed` bindings + the light python deps
  `inference_ur10/CamZED.py` needs.
- `docker-compose.yml` — GPU passthrough + USB passthrough + bind-mounts
  `conformal_VLA/` into the container at `/workspace`.
- `test_zed_connection.py` — dependency-light smoke test (raw `pyzed`, no repo
  imports). Run this **first**.
- `test_camzed_driver.py` — exercises the actual `CamZED` driver + `CamZED.yaml`
  config that `run_inference.py` uses. Run this **second**.

## Prerequisites on this host (verified 2026-08-28)

This laptop has an NVIDIA RTX 4060 Max-Q. All of the below is already in place on
this machine — kept here as a checklist in case you rebuild on a different host.

1. **NVIDIA driver** (RPM Fusion, standard Fedora path) — installed:
   `akmod-nvidia 610.57.04`, `nvidia-smi` reports max supported CUDA **13.3**.
   ```bash
   sudo dnf install https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm \
                     https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm
   sudo dnf install akmod-nvidia
   # wait for the akmod to build (`akmods --status`), then reboot
   nvidia-smi   # should now list your GPU + "CUDA Version: X.Y" (the MAX your driver supports)
   ```
2. **NVIDIA Container Toolkit** (so `docker run --gpus all` / the compose GPU block work) —
   installed: `nvidia-container-toolkit 1.20.0`, `nvidia` runtime shows up in `docker info`.
   ```bash
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
       | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
   sudo dnf install nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi   # sanity check
   ```
   Re-ran this sanity check on 2026-08-28 — GPU passthrough confirmed working.
3. **Docker permissions** — user is in the `docker` group, `docker ps` works without `sudo`.
   ```bash
   sudo usermod -aG docker $USER   # then fully log out/in (a new shell isn't enough)
   ```
4. **Match the image tag to your driver.** `nvidia-smi`'s reported "CUDA Version" is
   the *max* CUDA your driver supports — pick a `stereolabs/zed` tag with a CUDA
   version at or below that. This driver's 13.3 ceiling covers the whole
   `stereolabs/zed` CUDA range, so the `Dockerfile` `ARG` defaults (ZED SDK 4.2 /
   CUDA 12.1 / Ubuntu 22.04) were left as-is rather than bumped — see the comment
   block at the top of `Dockerfile` for the verification details. Browse tags at
   https://hub.docker.com/r/stereolabs/zed/tags if you need to change them (e.g. for
   a different/older driver).

## Build

```bash
cd real/zed_docker
docker compose build
```

## Connect the camera and test

Plug the ZED into a USB3 port (USB2 will run but at reduced resolution/fps), then:

```bash
docker compose run --rm zed bash
```

Inside the container:
```bash
python3 /workspace/real/zed_docker/test_zed_connection.py
```
This lists detected ZED devices, opens the camera, grabs one frame, prints depth
stats, and saves `zed_test_snapshot.png` — no display needed, safe over SSH.

If that passes, check the actual driver used by `run_inference.py`:
```bash
python3 /workspace/real/zed_docker/test_camzed_driver.py
```

## GUI (optional — `CamZED.realtime_show()` / `cv2.imshow`)

This host runs a Wayland session but `$DISPLAY` is set (GNOME's XWayland is active),
so X11 forwarding works with one extra step on the **host** (not in the container),
once per login:
```bash
xhost +local:docker
```
`docker-compose.yml` already forwards `DISPLAY` and mounts `/tmp/.X11-unix`. If
`cv2.imshow` still fails to open a window, run `echo $DISPLAY` on the host and make
sure it matches what's visible inside the container (`docker compose run --rm zed env | grep DISPLAY`).

## Notes / gotchas

- **`--privileged` + `/dev:/dev`**: the ZED SDK uses its own USB3 protocol, not plain
  UVC, so narrower `--device=...` mappings are unreliable — this mirrors Stereolabs'
  own Docker guidance rather than being sloppy about container security. Fine for a
  personal dev box; don't ship this compose file as-is to a shared/production host.
- **Hand-eye calibration still applies**: none of this changes the fact that
  `T_base_in_cam` (`inference_ur10/interfaces.py`) needs to be recalibrated for the
  ZED's physical mount — a working camera connection isn't a working extrinsic.
- **Scope**: this container only proves the ZED connection + `CamZED.py` driver. It
  does **not** include torch/transformers/onnxruntime, so you can't run the full
  DINO-SAM/LaMa/policy stack inside it yet — add those to `requirements.txt` (and
  probably a CUDA-matched `torch` wheel) if you decide to run all of `run_inference.py`
  inside this container rather than just using it to validate the camera.
