mkdir assets

# Baseline
# python main.py --n-epochs 10 --batch-size 16 --learning-rate 0.0001 --hidden-dim 10 --drop-rate 0.9 --do-train

# python main.py --n-epochs 10 --batch-size 16 --learning-rate 0.0001 --hidden-dim 150 --drop-rate 0.8 --do-train
# python main.py --n-epochs 10 --batch-size 32 --learning-rate 0.0001 --hidden-dim 150 --drop-rate 0.8 --do-train
# python main.py --n-epochs 10 --batch-size 64 --learning-rate 0.0001 --hidden-dim 150 --drop-rate 0.8 --do-train
# python main.py --n-epochs 10 --batch-size 16 --learning-rate 0.0001 --hidden-dim 250 --drop-rate 0.5 --do-train
# python main.py --n-epochs 10 --batch-size 32 --learning-rate 0.0001 --hidden-dim 250 --drop-rate 0.5 --do-train
# python main.py --n-epochs 10 --batch-size 64 --learning-rate 0.0001 --hidden-dim 250 --drop-rate 0.5 --do-train

# SWA + LLRD
# python main.py --n-epochs 10 --do-train --task custom --reinit_n_layers 3
# python main.py --hidden-dim 50 --debug --task custom --do-train
# python main.py --hidden-dim 150 --debug --task custom --do-train
# python main.py --hidden-dim 50 --debug --task custom --do-train --drop-rate 0.5
# python main.py --hidden-dim 150 --debug --task custom --do-train --drop-rate 0.5

# SupCon
# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 5 --batch-size 32
# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 10 --batch-size 32
# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 15 --batch-size 32
# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 20 --batch-size 32

# python main.py --hidden-dim 250 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 5 --batch-size 32
# python main.py --hidden-dim 250 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 10 --batch-size 32
# python main.py --hidden-dim 250 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 15 --batch-size 32
# python main.py --hidden-dim 250 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 20 --batch-size 32

# python main.py --hidden-dim 100 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 5 --batch-size 32
# python main.py --hidden-dim 100 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 10 --batch-size 32
# python main.py --hidden-dim 100 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 15 --batch-size 32
# python main.py --hidden-dim 100 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 20 --batch-size 32

# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 512 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 5 --batch-size 32
# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 512 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 10 --batch-size 32
# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 512 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 15 --batch-size 32
# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 512 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 20 --batch-size 32

# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 64 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 5 --batch-size 32
# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 64 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 10 --batch-size 32
# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 64 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 15 --batch-size 32
# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 64 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 20 --batch-size 32

# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 768 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 5 --batch-size 32
# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 768 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 10 --batch-size 32
# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 768 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 15 --batch-size 32
# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 768 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 20 --batch-size 32

# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-6 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 5 --batch-size 32
# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-6 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 10 --batch-size 32
# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-6 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 15 --batch-size 32
# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-6 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 20 --batch-size 32

# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-4 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 5 --batch-size 32
# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-4 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 10 --batch-size 32
# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-4 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 15 --batch-size 32
# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-4 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 20 --batch-size 32

# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 25 --cl-learning "SimCLR" --n-epochs 5 --batch-size 32
# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 25 --cl-learning "SimCLR" --n-epochs 10 --batch-size 32
# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 25 --cl-learning "SimCLR" --n-epochs 15 --batch-size 32
# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 25 --cl-learning "SimCLR" --n-epochs 20 --batch-size 32

# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 15 --cl-learning "SimCLR" --n-epochs 5 --batch-size 32
# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 15 --cl-learning "SimCLR" --n-epochs 10 --batch-size 32
# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 15 --cl-learning "SimCLR" --n-epochs 15 --batch-size 32
# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 15 --cl-learning "SimCLR" --n-epochs 20 --batch-size 32

# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 5 --batch-size 16
# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 10 --batch-size 16
# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 15 --batch-size 16
# python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 20 --batch-size 16

