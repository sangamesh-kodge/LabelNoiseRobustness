### This script will tune the hyperparameter for vanialla model. We use the best parameter obtained from this script in all out experiments. Run the below line from teminal. 
# DEVICE=0 sh./scripts/tune.sh
for seed in 42
do 
    for arch in resnet18
    do
        for lr in 0.1 0.01 0.001 0.0005  0.0002 0.0001 
        # best lr:1e-2
        do 
            for wd in 1e-2 1e-3 1e-4 1e-5  
            # best wd:1e-5
            do 
                for bsz in 64 128 256 512 
                # best bsz:128
                do 
                    # VANILLA
                    CUDA_VISIBLE_DEVICES=$DEVICE python3 ./train.py --dataset Mini-WebVision --data-path /local/scratch/a/skodge/data/Mini-WebVision \
                    --arch $arch --seed $seed --model-path /home/nano01/a/skodge/label_robustness/tuning --entity-name nrl --epoch 20 --project-name hp_tune --do-not-save \
                    --batch-size $bsz --weight-decay $wd --lr $lr                  
                done
            done
        done

    done
done