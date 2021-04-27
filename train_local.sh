source set_env.sh && gcloud ai-platform local train \
    --package-path trainer \
    --module-name trainer.task \
    --job-dir $JOB_DIR