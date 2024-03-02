# DEVICE=0 DATASET=Mini-WebVision EPOCH=200 BATCH_SIZE=64 ARCH=InceptionResNetV2 sh ./scripts/rough/high_priority/template_mmix.sh
# DEVICE=0 DATASET=WebVision1.0 EPOCH=60 BATCH_SIZE=512 ARCH=InceptionResNetV2 sh ./scripts/rough/high_priority/template_mmix.sh
# DEVICE=0 DATASET=Clothing1M EPOCH=10 BATCH_SIZE=512 ARCH=ResNet50 PRETRAIN_PATH=$NANO_HOME/label_robustness/ImageNet1k_final-v5/train-ResNet50-MisLabeled0.0/resnet50-pretrained.pt sh ./scripts/rough/high_priority/template_mmix.sh

for seed in 32087 35416 12484
do 
    for arch in $ARCH
    do
        for lr in 1e-2
        do 
            for wd in 4e-5
            do 
                for bsz in $BATCH_SIZE
                do 
                    for mgp in 0.85
                    do
                        for ma in 0.2
                        do
                            # MentorMix
                            if [[ -z "${PRETRAIN_PATH}" ]]; then
                                CUDA_VISIBLE_DEVICES=$DEVICE python ./train.py --dataset $DATASET --data-path $LOCAL_HOME/data/$DATASET \
                                --arch $arch --seed $seed --model-path $NANO_HOME/label_robustness --entity-name purdue-nrl-msee286 --project-name final-v5  --use-valset 0.1 \
                                --batch-size $bsz --weight-decay $wd --lr $lr --epoch $EPOCH  --mnet-gamma-p $mgp --mmix-alpha $ma 
                            else
                                CUDA_VISIBLE_DEVICES=$DEVICE python ./train.py --dataset $DATASET --data-path $LOCAL_HOME/data/$DATASET \
                                --arch $arch --seed $seed --model-path $NANO_HOME/label_robustness --entity-name purdue-nrl-msee286 --project-name final-v5  --use-valset 0.1 \
                                --batch-size $bsz --weight-decay $wd --lr $lr --epoch $EPOCH  --mnet-gamma-p $mgp --mmix-alpha $ma \
                                --load-loc $PRETRAIN_PATH 
                            fi
                        done
                    done
                done
            done
        done
    done
done