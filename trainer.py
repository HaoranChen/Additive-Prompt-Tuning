import os
import sys
import argparse
import torch
import numpy as np
import random
from random import shuffle
from collections import OrderedDict
import dataloaders
from dataloaders.utils import *
from torch.utils.data import DataLoader
import learners
from utils.calc_forgetting import calc_coda_forgetting, calc_general_forgetting

class Trainer:

    def __init__(self, args, seed, cur_iter, metric_keys, save_keys):

        # process inputs
        self.seed = seed
        self.cur_iter = cur_iter
        self.metric_keys = metric_keys
        self.save_keys = save_keys
        self.log_dir = args.log_dir
        self.batch_size = args.batch_size
        self.workers = args.workers
        
        # model load directory
        self.model_top_dir = args.log_dir

        # select dataset
        self.grayscale_vis = True
        self.top_k = 1
        if args.dataset == 'CIFAR10':
            Dataset = dataloaders.iCIFAR10
            num_classes = 10
            self.dataset_size = [32,32,3]
        elif args.dataset == 'CIFAR100':
            Dataset = dataloaders.iCIFAR100
            num_classes = 100
            self.dataset_size = [32,32,3]
        elif args.dataset == 'ImageNet_R':
            Dataset = dataloaders.iIMAGENET_R
            num_classes = 200
            self.dataset_size = [224,224,3]
        elif args.dataset == 'DomainNet':
            Dataset = dataloaders.iDOMAIN_NET
            num_classes = 200
            self.dataset_size = [224,224,3]
        elif args.dataset == 'CUB200':
            Dataset = dataloaders.iCUB200
            num_classes = 200
            self.dataset_size = [224,224,3]

        elif args.dataset == 'StanfordCars':
            Dataset = dataloaders.iStanfordCars
            num_classes = 196
            self.dataset_size = [224, 224, 3]

        elif args.dataset == 'StanfordDogs':
            Dataset = dataloaders.iStanfordDogs
            num_classes = 120
            self.dataset_size = [224, 224, 3]

        elif args.dataset == 'NABirds':
            Dataset = dataloaders.iNABirds
            num_classes = 55
            self.dataset_size = [224, 224, 3]
        elif args.dataset == 'EuroSAT':
            Dataset = dataloaders.iEuroSAT
            num_classes = 10
            self.dataset_size = [224, 224, 3]
        elif args.dataset == 'ISIC':
            Dataset = dataloaders.iISIC
            num_classes = 6
            self.dataset_size = [224, 224, 3]
        elif args.dataset == 'RESISC45':
            Dataset = dataloaders.iRESISC45
            num_classes = 45
            self.dataset_size = [224, 224, 3]
        elif args.dataset == 'CropDisease':
            Dataset = dataloaders.iCropDisease
            num_classes = 35
            self.dataset_size = [224, 224, 3]
        else:
            raise ValueError('Dataset not implemented!')

        # upper bound flag
        if args.upper_bound_flag:
            args.other_split_size = num_classes
            args.first_split_size = num_classes

        # load tasks
        class_order = np.arange(num_classes).tolist()
        class_order_logits = np.arange(num_classes).tolist()
        if self.seed >= 0 and args.rand_split:
            print('=============================================')
            print('Shuffling....seed is',self.seed)
            print('pre-shuffle:' + str(class_order))
            random.seed(self.seed)
            random.shuffle(class_order)
            print('post-shuffle:' + str(class_order))
            print('=============================================')
        self.tasks = []
        self.tasks_logits = []
        p = 0
        while p < num_classes and (args.max_task == -1 or len(self.tasks) < args.max_task):
            inc = args.other_split_size if p > 0 else args.first_split_size
            self.tasks.append(class_order[p:p+inc])
            self.tasks_logits.append(class_order_logits[p:p+inc])
            p += inc
        self.num_tasks = len(self.tasks)
        self.task_names = [str(i+1) for i in range(self.num_tasks)]

        # number of tasks to perform
        if args.max_task > 0:
            self.max_task = min(args.max_task, len(self.task_names))
        else:
            self.max_task = len(self.task_names)

        # datasets and dataloaders
        k = 1 # number of transforms per image
        if args.model_name.startswith('vit'):
            resize_imnet = True
        else:
            resize_imnet = False
        train_transform = dataloaders.utils.get_transform(dataset=args.dataset, phase='train', aug=args.train_aug, resize_imnet=resize_imnet)
        test_transform  = dataloaders.utils.get_transform(dataset=args.dataset, phase='test', aug=args.train_aug, resize_imnet=resize_imnet)
        self.train_dataset = Dataset(args.dataroot, train=True, lab = True, tasks=self.tasks,
                            download_flag=True, transform=train_transform, 
                            seed=self.seed, rand_split=args.rand_split, validation=args.validation)
        self.test_dataset  = Dataset(args.dataroot, train=False, tasks=self.tasks,
                                download_flag=False, transform=test_transform, 
                                seed=self.seed, rand_split=args.rand_split, validation=args.validation)

        # for oracle
        self.oracle_flag = args.oracle_flag
        self.add_dim = 0

        # Prepare the self.learner (model)
        self.learner_config = {'num_classes': num_classes,
                        'lr': args.lr,
                        'debug_mode': args.debug_mode == 1,
                        'momentum': args.momentum,
                        'weight_decay': args.weight_decay,
                        'schedule': [args.schedule],
                        'schedule_type': args.schedule_type,
                        'model_type': args.model_type,
                        'model_name': args.model_name,
                        'optimizer': args.optimizer,
                        'gpuid': args.gpuid,
                        'memory': args.memory,
                        'temp': args.temp,
                        'out_dim': num_classes,
                        'overwrite': args.overwrite == 1,
                        'DW': args.DW,
                        'batch_size': args.batch_size,
                        'upper_bound_flag': args.upper_bound_flag,
                        'tasks': self.tasks_logits,
                        'top_k': self.top_k,
                        'prompt_param':[self.num_tasks,args.prompt_param],
                        'ema_coeff': args.ema_coeff
                        }
        self.learner_type, self.learner_name = args.learner_type, args.learner_name
        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)

    def task_eval(self, t_index, local=False, task='acc'):

        val_name = self.task_names[t_index]
        print('validation split name:', val_name,f"local = {local}")
        
        # eval
        self.test_dataset.load_dataset(t_index, train=True)
        test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=self.workers)
        if local:
            return self.learner.validation(test_loader, task_in = self.tasks_logits[t_index], task_metric=task)
        else:
            return self.learner.validation(test_loader, task_metric=task)

    def train(self, avg_metrics):
        
        # temporary results saving
        temp_table = {}
        for mkey in self.metric_keys: temp_table[mkey] = []
        temp_dir = self.log_dir + '/temp/'
        if not os.path.exists(temp_dir): os.makedirs(temp_dir)

        # for each task
        for i in range(self.max_task):

            # save current task index
            self.current_t_index = i

            # print name
            train_name = self.task_names[i]
            print('======================', train_name, '=======================')

            # load dataset for task
            task = self.tasks_logits[i]
            if self.oracle_flag:
                self.train_dataset.load_dataset(i, train=False)
                self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)
                self.add_dim += len(task)
            else:
                self.train_dataset.load_dataset(i, train=True)
                self.add_dim = len(task)

            # set task id for model (needed for prompting)
            try:
                self.learner.model.module.task_id = i
            except:
                self.learner.model.task_id = i

            # add valid class to classifier
            self.learner.add_valid_output_dim(self.add_dim)

            # load dataset with memory
            self.train_dataset.append_coreset(only=False)

            # load dataloader
            train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=int(self.workers))

            # increment task id in prompting modules
            if i > 0:
                try:
                    if self.learner.model.module.prompt is not None:
                        self.learner.model.module.prompt.process_task_count()
                except:
                    if self.learner.model.prompt is not None:
                        self.learner.model.prompt.process_task_count()

            # learn
            self.test_dataset.load_dataset(i, train=False)
            test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=self.workers)
            model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.cur_iter+1)+'/task-'+self.task_names[i]+'/'
            if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
            avg_train_time = self.learner.learn_batch(train_loader, self.train_dataset, model_save_dir)

            # save model
            self.learner.save_model(model_save_dir)
            
            # evaluate acc
            acc_table = []
            self.reset_cluster_labels = True
            for j in range(i+1):
                acc_table.append(self.task_eval(j))
            temp_table['acc'].append(np.mean(np.asarray(acc_table)))

            # save temporary acc results
            for mkey in ['acc']:
                save_file = temp_dir + mkey + '.csv'
                np.savetxt(save_file, np.asarray(temp_table[mkey]), delimiter=",", fmt='%.2f')  

            if avg_train_time is not None: avg_metrics['time']['global'][i] = avg_train_time


        return avg_metrics 
    
    def summarize_acc(self, acc_dict, acc_table):

        # unpack dictionary
        avg_acc_all = acc_dict['global']
        avg_acc_pt = acc_dict['pt']

        if self.max_task > 1:
            forgetting_table = np.zeros((1,self.max_task,self.max_task)) 
        # Calculate average performance across self.tasks
        # Customize this part for a different performance metric
        avg_acc_history = [0] * self.max_task
        for i in range(self.max_task):
            train_name = self.task_names[i]
            cls_acc_sum = 0
            for j in range(i+1):
                val_name = self.task_names[j]
                cls_acc_sum += acc_table[val_name][train_name]
                avg_acc_pt[j,i,self.cur_iter] = acc_table[val_name][train_name]
                # avg_acc_pt_local[j,i,self.cur_iter] = acc_table_pt[val_name][train_name]
                if self.max_task > 1:
                    forgetting_table[0][i][j] = acc_table[val_name][train_name]
            avg_acc_history[i] = cls_acc_sum / (i + 1)

        avg_acc_all[:,self.cur_iter] = avg_acc_history
        
        if self.max_task > 1:
            coda_forgetting = calc_coda_forgetting(forgetting_table)
            general_forgetting = calc_general_forgetting(forgetting_table)
            print("coda_forgetting=",coda_forgetting)
            print("general_forgetting=",general_forgetting)

        drop_array = []
        print("acctable,",acc_table)
        for i in range(self.max_task): # 0 1 2 3 4 5 6 7 8 9 10
            train_name = self.task_names[i]
            drop_i = []
            for j in range(i+1,self.max_task+1):
                val_name = self.task_names[j-1]
                drop_i.append(acc_table[train_name][val_name])
            drop_array.append(drop_i)
        
        print("dropmatrix:", drop_array)

        # repack dictionary and return
        if  self.max_task > 1:
            return {'global': avg_acc_all,'pt': avg_acc_pt}
        else:
            return {'global': avg_acc_all,'pt': avg_acc_pt}
   
    def evaluate(self, avg_metrics):

        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)

        # store results
        metric_table = {}
        metric_table_local = {}
        for mkey in self.metric_keys:
            metric_table[mkey] = {}
            metric_table_local[mkey] = {}
            
        for i in range(self.max_task):

            # increment task id in prompting modules
            if i > 0:
                try:
                    if self.learner.model.module.prompt is not None:
                        self.learner.model.module.prompt.process_task_count()
                except:
                    if self.learner.model.prompt is not None:
                        self.learner.model.prompt.process_task_count()

            # load model
            model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.cur_iter+1)+'/task-'+self.task_names[i]+'/'
            self.learner.task_count = i 
            self.learner.add_valid_output_dim(len(self.tasks_logits[i]))
            self.learner.pre_steps()
            self.learner.load_model(model_save_dir)
            
            # set task id for model (needed for prompting)
            try:
                self.learner.model.module.task_id = i
            except:
                self.learner.model.task_id = i

            # evaluate acc
            metric_table['acc'][self.task_names[i]] = OrderedDict()
            metric_table_local['acc'][self.task_names[i]] = OrderedDict()
            self.reset_cluster_labels = True
            print("===global validation====")
            for j in range(i+1):
                val_name = self.task_names[j]
                print(f"test task {val_name}, using model {self.task_names[i]}")
                metric_table['acc'][val_name][self.task_names[i]] = self.task_eval(j)
            # print("===local validation====")
            # for j in range(i+1):
            #     val_name = self.task_names[j]
            #     print(f"test task {val_name}, using model {self.task_names[i]}")
            #     metric_table_local['acc'][val_name][self.task_names[i]] = self.task_eval(j, local=True)
                
        # summarize metrics
        avg_metrics['acc'] = self.summarize_acc(avg_metrics['acc'], metric_table['acc'])
        
        return avg_metrics