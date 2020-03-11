wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1bL_l3OtOM-bYlIDpsODVmbfhbXhA2i6X' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1bL_l3OtOM-bYlIDpsODVmbfhbXhA2i6X" -O netFeatBest.pth && rm -rf /tmp/cookies.txt
mkdir ckpts
mkdir ckpts/miniImageNet
mv netFeatBest.pth ckpts/miniImageNet/

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1M7zZRI65TTUF9f3_EFuU_Eo_SxDoT3Cj' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1M7zZRI65TTUF9f3_EFuU_Eo_SxDoT3Cj" -O e3bm_ckpt.pth && rm -rf /tmp/cookies.txt
mv e3bm_ckpt.pth ckpts/miniImageNet/
