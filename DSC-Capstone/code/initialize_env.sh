
sudo yum install git -y
sudo yum install docker -y

git clone https://github.com/louiecai/DSC-180B-Hardware-Acceleration.git
cd DSC-180B-Hardware-Acceleration

sudo systemctl start docker
sudo docker build -t dsc180b-hardware-acceleration .

docker run -it -v $(pwd):/root/home/DSC-180B-Hardware-Acceleration dsc180b-hardware-acceleration /bin/zsh

pip3 install gdown
python3 download_model.py
