python train.py \
--data_dir /data/ephemeral/home/data/train/images \
--model_dir /data/ephemeral/home/project/repo/level1-imageclassification-cv-06/ \
--model_type mask_model \
--model MyModel \
--dataset MaskModelDataset \
--criterion focal \
--epochs 5 \