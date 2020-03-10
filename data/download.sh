wget https://www.dropbox.com/s/a2a0bll17f5dvhr/Mini-ImageNet.zip?dl=0
mv Mini-ImageNet.zip?dl=0 Mini-ImageNet.zip
unzip Mini-ImageNet.zip
rm Mini-ImageNet.zip
rm -r Mini-ImageNet/train_val Mini-ImageNet/train_test
mv  Mini-ImageNet/train_train Mini-ImageNet/train 

wget https://www.dropbox.com/s/t36y8ng47wlcxw0/netFeatBest64.653.pth?dl=0
mv netFeatBest64.653.pth?dl=0 netFeatBest64.653.pth
mkdir ../ckpts
mkdir ../ckpts/Mini-ImageNet
mv netFeatBest64.653.pth ../ckpts/Mini-ImageNet/
