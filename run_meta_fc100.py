import os

def run_exp(num_batch=1000, shot=1, query=15, lr=0.0001, lr2=0.001, lr3=0.00001, lr4=0.00001, base_lr=0.01, update_step=10, gamma=0.5, gen_softweight=0.1, index=0):
    #num_batch = 1000
    max_epoch = 100
    #shot = 1
    #query = 15
    way = 5
    #lr = 0.0001
    #lr2 = 0.001
    step_size = 10
    #gamma = 0.5
    gpu = 0
    label = 'exp' + str(index)
    init_weights = '/BS/sun_project_multimodal/work/yyliu_project/lcc-new/Mtl-PyTorch-20-0-0/logs/pre/FC100_ResNet_batchsize128_lr0.1_gamma0.2_step30_maxepoch110_3/max_acc.pth'
    
    the_command = 'python3 main.py' \
        + ' --max_epoch=' + str(max_epoch) \
        + ' --num_batch=' + str(num_batch) \
        + ' --shot=' + str(shot) \
        + ' --train_query=' + str(query) \
        + ' --way=' + str(way) \
        + ' --lr=' + str(lr) \
        + ' --lr2=' + str(lr2) \
        + ' --step_size=' + str(step_size) \
        + ' --gamma=' + str(gamma) \
        + ' --gpu=' + str(gpu) \
        + ' --base_lr=' + str(base_lr) \
        + ' --update_step=' + str(update_step) \
        + ' --lr3=' + str(lr3) \
        + ' --lr4=' + str(lr4) \
        + ' --label=' + label \
        + ' --init_weights=' + init_weights \
        + ' --dataset=FC100' \
        + ' --gen_softweight=' + str(gen_softweight)

    os.system(the_command)

the_index=0
run_exp(num_batch=100, shot=1, query=20, lr=0.0001, lr2=0.001, lr3=0.00001, lr4=0.00001, base_lr=0.01, update_step=100, gamma=0.5, gen_softweight=0.001, index=the_index+15)
run_exp(num_batch=100, shot=5, query=20, lr=0.0001, lr2=0.001, lr3=0.00001, lr4=0.00001, base_lr=0.01, update_step=100, gamma=0.5, gen_softweight=0.001, index=the_index)

