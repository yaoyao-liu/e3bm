wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1vv3m14kusJcRpCsG-brG_Xk9MnetY9Bt' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1vv3m14kusJcRpCsG-brG_Xk9MnetY9Bt" -O miniimagenet.tar && rm -rf /tmp/cookies.txt
tar zxvf miniimagenet.tar
rm miniimagenet.tar
mv miniimagenet ./data/