python train.py \
--data_dir /data/ephemeral/home/data/train/images \
--model_dir /data/ephemeral/home/project/repo/level1-imageclassification-cv-06/ \
--model_type age_model \
--model MyModel \
--dataset AgeModelDataset \
--criterion focal \
--optimizer Adam \
--epoch 5
# --k_fold 5 \