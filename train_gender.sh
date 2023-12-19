python train.py \
--data_dir /data/ephemeral/home/data/train/images \
--model_dir /data/ephemeral/home/project/level1-imageclassification-cv-06 \
--model_type gender_model \
--model  EfficientNetb0Custom \
--dataset GenderModelDataset \
--criterion focal \
--epochs 25 \
--augmentation BaseAugmentation \
--resize 224 224 \
--batch_size 64 \
--valid_batch_size 100 \
--optimizer Adam \
--lr 1e-3 \
--val_ratio 0.2

# python train.py \
# --data_dir /data/ephemeral/home/data/train/images \
# --model_dir /data/ephemeral/home/project/repo/level1-imageclassification-cv-06/ \
# --model_type gender_model \
# --model ResNet50Model \
# --dataset MaskBaseDataset \
# --criterion focal