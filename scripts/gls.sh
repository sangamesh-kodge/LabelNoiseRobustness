### Requires defining DEVICE ARCH EPOCH DATASET DATA_PATH SAVE_PATH  
# eg. DEVICE=0 ARCH=InceptionResNetV2 EPOCH=200 DATASET=Mini-WebVision DATA_PATH=<path-to-dataset> SAVE_PATH=<path-to-save-model> sh ./scripts/gls.sh
for seed in 32087 35416 12484
do 
    for arch in $ARCH
    do
        for lr in 1e-2
        do 
            for wd in 4e-5
            do 
                for bsz in 256
                do 
                    for gs in 0.2
                    do
                        # MixUp
                        CUDA_VISIBLE_DEVICES=$DEVICE python3 ./train.py --dataset $DATASET --data-path $DATA_PATH \
                        --arch $arch --seed $seed --model-path $SAVE_PATH --project-name final  \
                        --batch-size $bsz --weight-decay $wd --lr $lr --gls-smoothing $gs --epoch $EPOCH
                    done
                done
            done
        done
    done
done
