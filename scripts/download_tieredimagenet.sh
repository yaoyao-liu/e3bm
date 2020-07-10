wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1T-4NVTSa5T6CXKSRbymYLnWp_OrtF-mo' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1T-4NVTSa5T6CXKSRbymYLnWp_OrtF-mo" -O tiered_imagenet.tar && rm -rf /tmp/cookies.txt
tar xvf tiered_imagenet.tar
rm tiered_imagenet.tar
mv tiered_imagenet ./data/