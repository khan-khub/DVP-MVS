sudo docker run --gpus all -it --rm     -v "$(pwd)":/workspace     -v /home/kwk/Desktop/EECE7150/project/textureless/ETH3D/:/data     apde-mvs-new

python3 colmap2mvsnet.py --dense_folder /data/multi_view_training_dslr_undistorted/office_original/ --save_folder /data/office_DVP_MVS --scale_factor 2 

python3 colmap2mvsnet.py     --dense_folder /data/multi_view_training_dslr_undistorted/office     --save_folder /workspace/office_DVP_MVS     --model_dir sparse     --scale_factor 1.0     --model_ext .txt



pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip3 install -e .
pip3 install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
pip3 install -e ".[app]" 
pip3 install -e ".[all]"
pip3 install pygltflib


python3 export_dmb.py     --input_dir /workspace/office_DVP_MVS/images     --output_dir /workspace/office_DVP_MVS/dep/
