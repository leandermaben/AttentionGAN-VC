import os
import pandas as pd
import shutil
import sys
from metrics.evaluate_lsd import main as lsd
from metrics.mssl import main as mssl

def run(command):
    print(command)
    exit_status = os.system(command)
    if exit_status > 0:
        exit(1)

def log(path,name,comment,avg_lsd,std_lsd,avg_mssl,std_mssl,male_avg_lsd,male_std_lsd,male_avg_mssl,male_std_mssl,female_avg_lsd,female_std_lsd,female_avg_mssl,female_std_mssl):
    """
    Created by Leander Maben.
    """
    df=pd.read_csv(path)
    df.loc[len(df.index)] = {'name':name,'comment':comment,'avg_lsd':avg_lsd,'std_lsd':std_lsd,'avg_mssl':avg_mssl,'std_mssl':std_mssl,'male_avg_lsd':male_avg_lsd,'male_std_lsd':male_std_lsd,'male_avg_mssl':male_avg_mssl,'male_std_mssl':male_std_mssl,'female_avg_lsd':female_avg_lsd,'female_std_lsd':female_std_lsd,'female_avg_mssl':female_avg_mssl,'female_std_mssl':female_std_mssl}
    df.to_csv(path,index=False)

##TODO: Modify functions to handle male,female,std dev

## Experiment with data quantity
def vary_data(train_percents, test_percent, names, csv_path, use_cycled_discriminators=True, use_phase=False, use_mask=False, lambda_cyc=10, lambda_identity=0.5, n_epochs=100, n_epochs_decay=0, data_cache='/content/AttentionGAN-VC/data_cache',results_dir='/content/AttentionGAN-VC/results'):
    """
    Experiment with different size training sets and document lsd and mssl scores.

    Created by Leander Maben.
    """

    input_nc = 1
    output_nc = 1

    bool_args_train = ''
    bool_args_test = ''

    if use_cycled_discriminators:
        bool_args_train+= '--use_cycled_discriminators'

    if use_phase:
        bool_args_train+= '--use_phase'
        bool_args_test+= '--use_phase'
        input_nc = 2
        output_nc = 2

    if use_mask:
        bool_args_train+= '--use_mask'
        bool_args_test+= '--use_mask'

    for train_p, name in zip(train_percents,names):
        print('#'*25)
        print(f'Training {name} with train_percent {train_p}')

        run(f'python datasets/fetchData.py --train_percent {train_p} --test_percent {test_percent}')
        run(f'python train.py --dataroot {data_cache} --name {name} --model attention_gan --dataset_mode audio --pool_size 50 --no_dropout --norm instance --lambda_A {lambda_cyc} --lambda_B {lambda_cyc} --lambda_identity {lambda_identity} --load_size 128 --crop_size 128 --batch_size 4 --niter {n_epochs} --niter_decay {n_epochs_decay} --gpu_ids 0 --display_id 0 --display_freq 100 --print_freq 100 --input_nc {input_nc} --output_nc {output_nc}'+bool_args_train)
        run(f'python test.py --dataroot {data_cache} --name {name} --model attention_gan --dataset_mode audio --norm instance --phase test --no_dropout --load_size 128 --crop_size 128 --batch_size 1 --gpu_ids 0 --input_nc {input_nc} --output_nc {output_nc}'+bool_args_test)
        avg_lsd,std_lsd,male_avg_lsd,male_std_lsd,female_avg_lsd,female_std_lsd = lsd(os.path.join(data_cache,'noisy','test'),os.path.join(results_dir,name,'test_latest','audios','fake_B'),use_gender=True)
        avg_mssl,std_mssl,male_avg_mssl,male_std_mssl,female_avg_mssl,female_std_mssl = mssl(os.path.join(data_cache,'noisy','test'),os.path.join(results_dir,name,'test_latest','audios','fake_B'),use_gender=True)
        log(csv_path, name,f'Training with train_percent {train_p} for {n_epochs} without decay and {n_epochs_decay} with decay and {bool_args_train} and lambda',avg_lsd,std_lsd,avg_mssl,std_mssl,male_avg_lsd,male_std_lsd,male_avg_mssl,male_std_mssl,female_avg_lsd,female_std_lsd,female_avg_mssl,female_std_mssl)
    
        shutil.rmtree(data_cache)
        print(f'Finished experiment with {name}')
        print('#'*25)
        
## Experiment with lambda_L1
def vary_lambdaL1(lamba_L1s,train_percent ,test_percent, names, csv_path, n_epochs=75, n_epochs_decay=25, data_cache='/content/Pix2Pix-VC/data_cache',results_dir='/content/Pix2Pix-VC/results'):
    """
    Experiment with different lambda_L1s and document lsd scores.

    Created by Leander Maben.
    """
    
    for lambd, name in zip(lamba_L1s,names):
        print('#'*25)
        print(f'Training {name} with Lambda_l1 {lambd}')
        run(f'python datasets/fetchData.py --train_percent {train_percent} --test_percent {test_percent}')
        run(f'python train.py --dataroot {data_cache} --name {name} --model pix2pix --n_epochs {n_epochs} --n_epochs_decay {n_epochs_decay} --direction AtoB --input_nc 1 --output_nc 1 --lambda_L1 {lambd} --netG resnet_9blocks   --dataset_mode audio')
        run(f'python test.py --dataroot {data_cache} --name {name} --model pix2pix --direction AtoB --netG resnet_9blocks  --input_nc 1 --output_nc 1 --dataset_mode audio')
        avg_lsd, min_lsd = lsd(os.path.join(data_cache,'noisy','test'),os.path.join(results_dir,name,'test_latest','audios','fake_B'))
        log(csv_path, name,f'Training with {lambd} lambda_L1 for {n_epochs} without decay and {n_epochs_decay} with decay',avg_lsd,min_lsd)
        shutil.rmtree(data_cache)
        print(f'Finished experiment with {name}')
        print('#'*25)
    
def run_eval(checkpoints_dir='/content/Pix2Pix-VC/checkpoints', data_cache='/content/Pix2Pix-VC/data_cache', results_dir='/content/Pix2Pix-VC/results'):
    """
    Run evaluation on all stored models

    Created by Leander Maben
    """
    #run(f'python datasets/fetchData.py --train_percent {10} --test_percent {15}')
    for name in os.listdir(checkpoints_dir):       
        run(f'python test.py --dataroot {data_cache} --name {name} --model pix2pix --direction AtoB --netG resnet_9blocks  --input_nc 1 --output_nc 1 --dataset_mode audio')
        avg_lsd, min_lsd = lsd(os.path.join(data_cache,'noisy','test'),os.path.join(results_dir,name,'test_latest','audios','fake_B'))


def use_codecs(codecs, train_percent, test_percent, names, csv_path, n_epochs=75, n_epochs_decay=25, data_cache='/content/Pix2Pix-VC/data_cache',results_dir='/content/Pix2Pix-VC/results'):
    """
    Experiment with different codecs and document lsd scores.

    Created by Leander Maben.
    """

    for codec, name in zip(codecs,names):
        print('#'*25)
        print(f'Training {name} with codec {codec}')
        run(f'python datasets/fetchData.py --train_percent {train_percent} --test_percent {test_percent} --transfer_mode codec --codec_name {codec}')
        run(f'python train.py --dataroot {data_cache} --name {name} --model pix2pix --n_epochs {n_epochs} --n_epochs_decay {n_epochs_decay} --direction AtoB --input_nc 1 --output_nc 1 --lambda_L1 200 --netG resnet_9blocks   --dataset_mode audio')
        run(f'python test.py --dataroot {data_cache} --name {name} --model pix2pix --direction AtoB --netG resnet_9blocks  --input_nc 1 --output_nc 1 --dataset_mode audio')
        avg_lsd,std_lsd,male_avg_lsd,male_std_lsd,female_avg_lsd,female_std_lsd = lsd(os.path.join(data_cache,'noisy','test'),os.path.join(results_dir,name,'test_latest','audios','fake_B'),use_gender=True)
        avg_mssl,std_mssl,male_avg_mssl,male_std_mssl,female_avg_mssl,female_std_mssl = mssl(os.path.join(data_cache,'noisy','test'),os.path.join(results_dir,name,'test_latest','audios','fake_B'),use_gender=True)
        log(csv_path, name,f'Training with codec {codec} for {n_epochs} without decay and {n_epochs_decay} with decay',avg_lsd,std_lsd,avg_mssl,std_mssl,male_avg_lsd,male_std_lsd,male_avg_mssl,male_std_mssl,female_avg_lsd,female_std_lsd,female_avg_mssl,female_std_mssl)
        shutil.rmtree(data_cache)
        print(f'Finished experiment with {name}')
        print('#'*25)


if __name__ == '__main__':
    csv_path = '/content/drive/MyDrive/NTU - Speech Augmentation/att.csv'
    if not os.path.exists(csv_path):
        cols=['name','comment','avg_lsd','std_lsd','avg_mssl','std_mssl','male_avg_lsd','male_std_lsd','male_avg_mssl','male_std_mssl','female_avg_lsd','female_std_lsd','female_avg_mssl','female_std_mssl']
        df=pd.DataFrame(columns=cols)
        df.to_csv(csv_path,index=False)
    
    #log(csv_path,'Dummy', "Logging default parameters used - 200 lambda_L1, 15% test size",0,0)

    train_percents =[5,10,25,50]
    vary_data(train_percents,15,[f'pix_noisy_{i}' for i in train_percents], csv_path)

    train_percents =[5,10,25,50]
    vary_data(train_percents,15,[f'pix_noisy_{i}' for i in train_percents], csv_path,use_mask =True)

    train_percents =[5,10,25,50]
    vary_data(train_percents,15,[f'pix_noisy_{i}' for i in train_percents], csv_path,use_phase =True)

    # lambda_L1s = [2000]
    # vary_lambdaL1(lambda_L1s ,10,15,[f'pix_noisy_lambdL1_{i}' for i in lambda_L1s], csv_path)
    #run_eval()
    # codecs = ['g726','gsm','ogg','g723_1']
    # use_codecs(codecs , 10, 15, [f'pix_{c}' for c in codecs], csv_path)
    
