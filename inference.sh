cat best_model/mask_model_best.tar* | tar xvf - -C ./best_model/ \
&& \
python inference.py \
--data_dir /data/ephemeral/home/data/eval \
--model_dir /data/ephemeral/home/level1-imageclassification-cv-06/best_model \
--output_dir ./ \
--age_model MyModel \
--gender_model EfficientNetb0Custom \
--mask_model MaskCustomModel \
--age_augmentation BaseAugmentation \
--gender_augmentation BaseAugmentation \
--mask_augmentation MaskAugmentation \
&& \
rm -f best_model/mask_model_best.pth \
