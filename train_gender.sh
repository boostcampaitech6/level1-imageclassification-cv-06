python train.py \
--data_dir /data/ephemeral/home/data/train/images \
--model_dir /data/ephemeral/home/repo/level1-imageclassification-cv-06 \
--model_type gender_model \
--model EfficientNetb0Custom \
--dataset GenderModelDataset \
--criterion focal \
--optimizer Adam \
--lr 0.0001 \
--resize 128 96 \
--augmentation BaseAugmentation \
--name hn \
--epochs 15

# --data_dir /data/ephemeral/home/data/train/images \
# --model_dir /data/ephemeral/home/project/level1-imageclassification-cv-06 \
# --model_type gender_model \
# --model  EfficientNetb0Custom \
# --dataset GenderSplitByProfileDataset \
# --criterion focal \
# --epochs 25 \
# --augmentation CustomAugmentation \
# --resize 224 224 \
# --batch_size 64 \
# --valid_batch_size 100 \
# --optimizer Adam \
# --lr 1e-3 \
# --val_ratio 0.2

# python train.py \
# --data_dir /data/ephemeral/home/data/train/images \
# --model_dir /data/ephemeral/home/project/repo/level1-imageclassification-cv-06/ \
# --model_type gender_model \
# --model ResNet50Model \
# --dataset MaskBaseDataset \
# --criterion focal