# [CSD: Consistency-based Semi-supervised learning for object Detection (NeurIPS 2019)](https://papers.nips.cc/paper/9259-consistency-based-semi-supervised-learning-for-object-detection) 

By [Jisoo Jeong](http://mipal.snu.ac.kr/index.php/Jisoo_Jeong), Seungeui Lee, [Jee-soo Kim](http://mipal.snu.ac.kr/index.php/Jee-soo_Kim), [Nojun Kwak](http://mipal.snu.ac.kr/index.php/Nojun_Kwak)

---

## Updated guide for running on off-the-shelf Ubuntu 18.04

### Preparing AWS instances
Launch standard Ubuntu 18.04 (spot) instance with at least 200 GB of storage.

Then run:

```
# Prepare environment
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt-get install nvidia-driver-418 gcc-6 g++-6
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-6 10
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 10

# Download CUDA
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/patches/1/cuda_9.0.176.1_linux-run
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/patches/2/cuda_9.0.176.2_linux-run
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/patches/3/cuda_9.0.176.3_linux-run
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/patches/4/cuda_9.0.176.4_linux-run

# Install
chmod +x cuda*
sudo sh cuda_9.0.176_384.81_linux-run --verbose --silent --toolkit --override
sudo sh cuda_9.0.176.1_linux-run --silent --accept-eula
sudo sh cuda_9.0.176.2_linux-run --silent --accept-eula
sudo sh cuda_9.0.176.3_linux-run --silent --accept-eula
sudo sh cuda_9.0.176.4_linux-run --silent --accept-eula

# Add to PATH
sudo sh -c "echo '/usr/local/cuda-9.0/lib64' >> /etc/ld.so.conf"
sudo ldconfig

# Here must intervene manually
vim .bashrc

# Add the following:
export PATH=/usr/local/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH

source .bashrc

# Download CUDNN
wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7_7.6.5.32-1+cuda9.0_amd64.deb
wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7-dev_7.6.5.32-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7_7.6.5.32-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-dev_7.6.5.32-1+cuda9.0_amd64.deb

# Get Conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Here must intervene manually, agree to set up Conda in the very end
sh Miniconda3-latest-Linux-x86_64.sh

# Reboot to finish CUDA installation
sudo reboot
```

### CSD-RFCN config

```
# Env
conda create -n pytorch0.4 python=3.6
conda activate pytorch0.4
conda install pytorch=0.4.0 cuda90 -c pytorch

# Get code
git clone https://github.com/soo89/CSD-RFCN
pip install -r requirements.txt
pip install scipy==1.1.0 scikit-image torchvision==0.1.6 Cython

# Compile CUDA code
cd CSD-RFCN/lib

# Manual intervention: see https://github.com/princewang1994/R-FCN.pytorch#compilation
vim make.sh
# Configure your arch etc.

sh make.sh

# Download VOC data
cd ~
mkdir data
git clone https://github.com/amdegroot/ssd.pytorch
sh ssd.pytorch/data/scripts/VOC2007.sh ~/data
sh ssd.pytorch/data/scripts/VOC2012.sh ~/data
rm -rf ssd.pytorch

# Link data (here I use /home/ubuntu/data, note: must use absolute path)
sed -i 's/\/home\/soo\/data/\/home\/ubuntu\/data/g' lib/model/utils/config.py

# Download resnet
mkdir pretrained_model && cd pretrained_model
pip install gdown
gdown https://drive.google.com/uc?id=1I4Jmh2bU6BJVnwqfg5EDe8KGGdec2UE8
mv resnet101_caffe.pth resnet101_rcnn.pth
```

Now, change some lines as advised here (both of the two comments): [https://github.com/jwyang/faster-rcnn.pytorch/issues/147#issuecomment-386213465](https://github.com/jwyang/faster-rcnn.pytorch/issues/147#issuecomment-386213465)

Then in `/lib/model/rfcn/rfcn_consistency.py` in lines `111, 144, 156` replace `int(semi_check)` with `int(semi_check.sum())`.

Finally, you are ready to run this!

### Outcome using one V100 w/ 16GB

Even reducing images to 100x100px and batch_size to 16 (from 256) I get out-of-memory error.
(most of modifications in `config.py` are unnecessary, only `data` path should be changed given you have enough memory)

---

## Installation & Preparation
We experimented with CSD using the RFCN pytorch framework. To use our model, complete the installation & preparation on the [RFCN pytorch homepage](https://github.com/princewang1994/R-FCN.pytorch)

## Check list
```Shell
check DATA_DIR in '/lib/model/utils/config.py'
```

## Supervised learning
```Shell
python train_rfcn.py
```

## CSD training
```Shell
python train_csd.py
```

## Evaluation
```Shell
python test_net.py
```
