python train.py \
--data_dir /data/ephemeral/home/data/train/images \
--model_dir /data/ephemeral/home/project/repo/level1-imageclassification-cv-06/ \
--model_type age_model \
--model MyModel \
--dataset AgeModelDataset \
--criterion focal \
--optimizer Adam \
#"k for (Stratified) K-fold Cross Validation"
--k_fold 5 \
#"0: No CV, 1: K-fold, 2: Stratified K-fold"
--k_fold_type 1 \
--epochs 5 \