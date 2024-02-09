for seed in 32087 #35416 12484
do 
    for arch in resnet18
    do
        for lr in 1e-2
        do 
            for wd in 1e-5
            do 
                for bsz in 128
                do 
                    for sr in 0.1 0.05
                    do
                        # SAM
                        CUDA_VISIBLE_DEVICES=$DEVICE python ./train.py --dataset Mini-WebVision --data-path /local/scratch/a/skodge/data/Mini-WebVision \
                        --arch $arch --seed $seed --model-path /home/nano01/a/skodge/label_robustness --entity-name nrl \
                        --batch-size $bsz --weight-decay $wd --lr $lr  --use-rho $sr
                    done
                done
            done
        done
    done
done