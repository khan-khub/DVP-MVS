# Build and Run 

Run DA3 docker
```bash
sudo docker run --gpus all -it --rm     -v "$PWD":/workspace     da3-final
```

Navigate to DA3 repo with `export_dmb.py` and run to get dmb depths
```bash
python3 export_dmb.py --input_dir /workspace/test/office/images/ --output_dir /workspace/test/office/dep/
```

Run the DVP-MVS docker which is same as APDe-MVS docker 
```bash
sudo docker run --gpus all -it --rm     -v "$(pwd)":/workspace     apde-mvs-5080
```

Prepare data using 
```bash 
python3 colmap2mvsnet.py --dense_folder /workspace/data/multi_view_training_dslr_undistorted/office/ --save_folder /workspace/test/office --scale_factor 2 
```

Build code using 
```bash
mkdir build 
cd build 
cmake ..
make 
```

Run on data using 
```bash 
./APD /workspace/test/office/
```