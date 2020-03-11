wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1FzkXVCIA8VbhcOoKQPJKX5OVe7Q0aFKp' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1FzkXVCIA8VbhcOoKQPJKX5OVe7Q0aFKp" -O Mini-ImageNet.zip && rm -rf /tmp/cookies.txt
unzip Mini-ImageNet.zip
rm Mini-ImageNet.zip
rm -r Mini-ImageNet/train_val Mini-ImageNet/train_test
mv  Mini-ImageNet/train_train Mini-ImageNet/train 
mv  Mini-ImageNet data/miniImageNet

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ia1mUe9OVIVfs_ggjiycBS6ttQd8Ke1i' -O val1000Episode_5_way_5_shot.json
mv val1000Episode_5_way_5_shot.json data/miniImageNet/

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1G206V-8_Ls5QH05KKFtuWXmJ3iv-MRQn' -O val1000Episode_5_way_1_shot.json
mv val1000Episode_5_way_1_shot.json data/miniImageNet/
