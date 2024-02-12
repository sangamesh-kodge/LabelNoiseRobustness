# DEVICE=3 DATASET=Mini-WebVision EPOCH=200 sh ./scripts/rough/sam.sh
# DEVICE=3 DATASET=WebVision1.0 EPOCH=45 sh ./scripts/rough/sam.sh

for seed in 32087 35416 12484
do 
    for arch in InceptionResNetV2
    do
        for lr in 1e-2
        do 
            for wd in 4e-5
            do 
                for bsz in 256
                do 
                    for sr in 0.1
                    do
                        # SAM
                        CUDA_VISIBLE_DEVICES=$DEVICE python ./train.py --dataset $DATASET --data-path /local/scratch/a/skodge/data/$DATASET \
                        --arch $arch --seed $seed --model-path /home/nano01/a/skodge/label_robustness --entity-name purdue-nrl-msee286 --project-name final \
                        --batch-size $bsz --weight-decay $wd --lr $lr  --sam-rho $sr --epoch $EPOCH 
                    done
                done
            done
        done
    done
done