gcloud ai-platform jobs submit training $JOB_NAME \
    --package-path trainer \
    --module-name trainer.task \
    --region $REGION \
    --python-version 3.7 \
    --runtime-version 1.15 \
    --job-dir $JOB_DIR \
    --scale-tier custom \
    --master-machine-type n1-standard-8 \
    --master-accelerator=type=nvidia-tesla-p4,count=1 \
    --stream-logs

