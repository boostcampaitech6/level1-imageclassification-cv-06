python train.py \
--data_dir /data/ephemeral/home/data/train/images \
--model_dir /data/ephemeral/home/level1-imageclassification-cv-06/ \
--model_type gender_model \
--model MyModel \
--dataset GenderModelDataset \
--criterion focal \
--optimizer Adam \
--k_fold 5 \
--epoch 5