wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=13pzlvn9s4psbZlGpIsYCi9fwQnWeSIkP' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=13pzlvn9s4psbZlGpIsYCi9fwQnWeSIkP" -O pretrain_model.tar && rm -rf /tmp/cookies.txt
tar zxvf pretrain_model.tar
rm pretrain_model.tar
