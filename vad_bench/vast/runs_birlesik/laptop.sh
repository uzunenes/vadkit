#!/usr/bin/env bash
# LAPTOP side of the combined-streaming run (docs/tez_notlari/birlesik_hiz_on_kayit_2026-11-08.md). Needs vastai with
# the API key set, ~/.ssh/vast registered on the account and gh logged in (the token stays on the laptop).
#
#   bash vad_bench/vast/runs_birlesik/laptop.sh offers          # list RTX 3090 offers (verified, >= 16 CPU cores)
#   bash vad_bench/vast/runs_birlesik/laptop.sh start OFFER_ID  # rent, copy the repo (git HEAD), download, setup, run
#   bash vad_bench/vast/runs_birlesik/laptop.sh status          # tail of run.log
#   bash vad_bench/vast/runs_birlesik/laptop.sh collect         # wait for ALL_DONE, copy out/ back, destroy if complete
#
# The instance id and ssh address are kept in /tmp/msc_birlesik_instance. Only that instance is ever destroyed.
set -euo pipefail
REPO=uzunenes/cctv-aean
ROOT=$(git rev-parse --show-toplevel)
DEST=$ROOT/paper/ieee_access/kanit/birlesik_20261108/vast_out
STATE=/tmp/msc_birlesik_instance
LABEL=msc-birlesik-20261108
IMAGE=pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

ssh_cmd() { . "$STATE"; ssh -i "$HOME/.ssh/vast" -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p "$PORT" "root@$HOST" "$@"; }

case "${1:-}" in
offers)
  vastai search offers 'gpu_name=RTX_3090 num_gpus=1 verified=true reliability>0.98 cpu_cores_effective>=16 disk_space>=150 inet_down>=500 direct_port_count>=1' \
    -o 'dph_total' | head -20 ;;
start)
  [ -n "${2:-}" ] || { echo "usage: $0 start OFFER_ID"; exit 1; }
  id=$(vastai create instance "$2" --image $IMAGE --disk 150 --ssh --direct --label $LABEL --raw |
       python3 -c "import json,sys; print(json.load(sys.stdin)['new_contract'])")
  echo "instance $id"
  for i in $(seq 1 60); do
    read -r host port < <(vastai show instance "$id" --raw | python3 -c "
import json,sys; i=json.load(sys.stdin); p=(i.get('ports') or {}).get('22/tcp')
print(i.get('public_ipaddr') or '', p[0]['HostPort'] if p else '')")
    if [ -n "$host" ] && [ -n "$port" ] && ssh -i "$HOME/.ssh/vast" -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
         -o BatchMode=yes -p "$port" "root@$host" true 2>/dev/null; then break; fi
    sleep 15
  done
  printf 'ID=%s\nHOST=%s\nPORT=%s\n' "$id" "$host" "$port" > "$STATE"
  echo "ssh ok: $host:$port"
  # repository at HEAD (committed files only), then the private release assets through short-lived signed URLs
  git -C "$ROOT" archive --format=tar HEAD | ssh_cmd 'mkdir -p /workspace/cctv_aean /workspace/data /workspace/logs && tar xf - -C /workspace/cctv_aean'
  urls=$(gh api "repos/$REPO/releases/tags/veri-2026-11-02" --jq '.assets[] | "\(.id) \(.name)"' |
    while read -r aid name; do
      [[ "$name" =~ ^(UCSD_Anomaly_Dataset.tar.gz|Avenue_Dataset.zip|ground_truth_demo.zip|checkpoints_2026-11-02.tar)$ ]] || continue
      echo "$name $(curl -s -o /dev/null -w '%{redirect_url}' -H "Authorization: Bearer $(gh auth token)" \
            -H 'Accept: application/octet-stream' "https://api.github.com/repos/$REPO/releases/assets/$aid")"
    done)
  echo "$urls" | ssh_cmd 'cat > /workspace/data/urls.txt && bash /workspace/cctv_aean/vad_bench/vast/runs_velpilot/dl.sh /workspace/data/urls.txt' |
    tee /tmp/msc_birlesik_dl.log
  grep -q FAIL /tmp/msc_birlesik_dl.log && { echo "download failed; instance $id left running"; exit 1; }
  ssh_cmd 'cd /workspace/cctv_aean && bash vad_bench/vast/runs_hakem/setup.sh > /workspace/logs/setup.log 2>&1; tail -1 /workspace/logs/setup.log'
  ssh -n -f -i "$HOME/.ssh/vast" -o StrictHostKeyChecking=no -p "$port" "root@$host" \
    'cd /workspace/cctv_aean && nohup setsid bash vad_bench/vast/runs_birlesik/run.sh > /workspace/logs/run.log 2>&1 < /dev/null &'
  echo "run.sh started on $id; later: $0 collect" ;;
status)
  ssh_cmd 'tail -5 /workspace/logs/run.log; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader' ;;
collect)
  . "$STATE"
  for i in $(seq 1 300); do
    ssh_cmd 'test -f /workspace/logs/ALL_DONE' 2>/dev/null && break
    sleep 60
  done
  mkdir -p "$DEST"
  ssh_cmd 'cd /workspace && tar czf - --exclude out/feat --exclude out/state/*/features* out logs' | tar xzf - -C "$DEST"
  if [ -f "$DEST/out/stream/avenue/result.json" ] && [ -f "$DEST/out/grid/check_avenue.json" ] && \
     [ "$(ls "$DEST"/out/bench/*.json | wc -l)" -ge 24 ]; then
    lbl=$(vastai show instance "$ID" --raw | python3 -c "import json,sys; print(json.load(sys.stdin).get('label'))")
    [ "$lbl" = "$LABEL" ] && vastai destroy instance "$ID" -y && echo "copied to $DEST; instance $ID destroyed"
  else
    echo "copy incomplete; instance $ID left running"
  fi ;;
*)
  sed -n 2,12p "$0" ;;
esac
