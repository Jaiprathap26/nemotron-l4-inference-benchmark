#!/bin/bash
set -euo pipefail
PROJECT_ID=$(gcloud config get-value project)
ZONE="us-central1-a"
INSTANCE_NAME="nemotron-inference-lab"
IMAGE_FAMILY=$(gcloud compute images list --project=deeplearning-platform-release --filter="family ~ common-cu12.*ubuntu-2204" --format="value(family)" | sort -r | head -n 1)

gcloud compute instances create "$INSTANCE_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family="${IMAGE_FAMILY:-common-cu124-ubuntu-2204-nvidia-550}" \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-balanced \
  --metadata="install-nvidia-driver=True"
