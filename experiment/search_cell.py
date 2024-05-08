import os
import sys
import yaml
import time
import shutil
import argparse
from tqdm import tqdm
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transform
from torch.cuda.amp import GradScaler

sys.path.append('..')
from util.loss.loss import SegmentationLosses
from util.datasets import get_dataset, datasets
from util.utils import get_logger, save_checkpoint
from util.utils import average_meter, calc_time
from util.utils import get_gpus_memory_info, calc_parameters_count
from util.optimizers import get_optimizer
from util.metrics import *
from search.backbone.nas_unet_search import NasUnetSearch, Architecture
import wandb

def probability_combinations(array):
    # Get the shape of the input array
    array = prune_array(array)
    shape = array.shape
    out = []
    for j in range(shape[1]):
        cur = array[0, j]
        for i in range(1, shape[0]):
            cur = np.outer(cur, array[i, :])
        out = out + cur.flatten().tolist()
    return out

def ranked_list(array):
    rank_weights = probability_combinations(array)
    ranked = sorted(enumerate(rank_weights), key=lambda x: x[1], reverse=True)
    return [i for i, _ in ranked]

def prune_array(array):
    return np.array([sorted(row, reverse=True)[:2] for row in array])

def array_to_string(array):
    rows = []
    for row in array:
        row_str = ','.join(map(str, row))
        rows.append(row_str)
    return '|'.join(rows)

class SearchNetwork(object):

    def __init__(self):
        self._init_configure()
        self._init_logger()
        self._init_device()
        self._init_dataset()
        self._init_model()
        self._check_resume()

    def _init_configure(self):
        parser = argparse.ArgumentParser(description='config')
        parser.add_argument('--config', nargs='?',type=str,default='../configs/nas_unet/nas_unet_voc.yml',
                            help='Configuration file to use (default is nas_unet_voc.yml)')
        parser.add_argument('--quantize', action='store_true',
                    help='Whether to quantize or not (default is False)')
        parser.add_argument('--mixed_precision', action='store_true',
                    help='Whether to use mixed precision or not (default is False)')
        parser.add_argument('--low_prec_optim', action='store_true',
                    help='Whether to use a low precision optimizer or not (default is False)')

        self.args = parser.parse_args()

        if self.args.mixed_precision:
            self.grad_scaler = GradScaler()

        with open(self.args.config) as fp:
            self.cfg = yaml.safe_load(fp)
            print('load configure file at {}'.format(self.args.config))

    def _init_logger(self):
        log_dir = '../logs/nasunet/search' + '/{}'.format(self.cfg['data']['dataset']) +\
                  '/search-{}'.format(time.strftime('%Y%m%d-%H%M%S'))
        self.logger = get_logger(log_dir)
        print('RUNDIR: {}'.format(log_dir))
        shutil.copy(self.args.config, log_dir)
        self.logger.info('Nas-Search')
        self.save_path = log_dir
        self.save_tbx_log = self.save_path + '/tbx_log'
        #self.writer = SummaryWriter(self.save_tbx_log)
        # Initialize Weights and Biases
        wandb.init()
        wandb.log({"args": vars(self.args)})

    def _init_device(self):
        self.device = torch.device("cuda" if self.cfg['searching']['gpu'] else "cpu")
        np.random.seed(self.cfg.get('seed', 1337))
        torch.manual_seed(self.cfg.get('seed', 1337))
        if self.cfg['searching']['gpu'] and torch.cuda.is_available() :
            self.device_id, _ = get_gpus_memory_info()
            self.device = torch.device('cuda:{}'.format(0 if self.cfg['searching']['multi_gpus'] else self.device_id))
            torch.cuda.manual_seed(self.cfg.get('seed', 1337))
            torch.cuda.set_device(self.device)
            cudnn.enabled = True
            cudnn.benchmark = True
        else:
            self.logger.info('No gpu devices available!, we will use cpu')
            self.device = torch.device('cpu')
            self.device_id = 0

    def _init_dataset(self):
        trainset = get_dataset(self.cfg['data']['dataset'], split='train', mode='train')

        num_train = len(trainset)
        print(f"Num train: {num_train}")
        indices = list(range(num_train))
        split = int(np.floor(self.cfg['searching']['train_portion'] * num_train))
        self.n_classes = trainset.num_class
        self.in_channels = trainset.in_channels
        kwargs = {'num_workers': self.cfg['searching']['n_workers'], 'pin_memory': True}
        self.train_queue = data.DataLoader(trainset, batch_size=self.cfg['searching']['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               indices[:split]), **kwargs)

        self.valid_queue = data.DataLoader(trainset, batch_size=self.cfg['searching']['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               indices[split:num_train]),**kwargs)

    def _init_model(self):

        # Read the configure
        init_channel = self.cfg['searching']['init_channels']
        depth = self.cfg['searching']['depth']
        meta_node_num = self.cfg['searching']['meta_node_num']

        # Setup loss function
        self.criterion = SegmentationLosses(name=self.cfg['searching']['loss']['name']).to(self.device)
        self.logger.info("Using loss {}".format(self.cfg['searching']['loss']['name']))

        # Setup Model
        model = NasUnetSearch(self.in_channels, init_channel, self.n_classes, depth,
                              meta_node_num=meta_node_num, use_sharing=self.cfg['searching']['sharing_normal'],
                              double_down_channel=self.cfg['searching']['double_down_channel'],
                              multi_gpus=self.cfg['searching']['multi_gpus'],
                              device=self.device,
                              use_quantization=self.args.quantize)

        if self.device.type == 'cuda':
            if torch.cuda.device_count() > 1 and self.cfg['searching']['multi_gpus']:
                self.logger.info('use: %d gpus', torch.cuda.device_count())
                self.model = nn.DataParallel(model)
            elif torch.cuda.is_available():
                self.logger.info('gpu device = %d' % self.device_id)
                torch.cuda.set_device(self.device)

        self.model = model.to(self.device)
        self.logger.info('param size = %fMB', calc_parameters_count(model))

        # Setup optimizer, lr_scheduler and loss function for model
        optimizer_cls1 = get_optimizer(self.cfg, phase='searching', optimizer_type='model_optimizer', low_prec_optim=self.args.low_prec_optim)
        optimizer_params1 = {k: v for k, v in self.cfg['searching']['model_optimizer'].items()
                            if k != 'name'}

        self.model_optimizer = optimizer_cls1(self.model.parameters(), **optimizer_params1)
        self.logger.info("Using model optimizer {}".format(self.model_optimizer))

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.model_optimizer, self.cfg['searching']['epoch'], eta_min=1.0e-3)

        # Setup optimizer, lr_scheduler and loss function for architecture
        optimizer_cls2 = get_optimizer(self.cfg, phase='searching', optimizer_type='arch_optimizer', low_prec_optim=self.args.low_prec_optim)
        optimizer_params2 = {k: v for k, v in self.cfg['searching']['arch_optimizer'].items()
                            if k != 'name'}

        self.arch_optimizer = optimizer_cls2(self.model.alphas(), **optimizer_params2)

        self.architect = Architecture(self.model, arch_optimizer=self.arch_optimizer,
                                      criterion=self.criterion)

    def _check_resume(self):
        self.dur_time = 0
        self.start_epoch = 0
        self.cur_count = 0
        self.geno_type = ''
        # optionally resume from a checkpoint for model
        if self.cfg['searching']['resume'] is not None:
            if os.path.isfile(self.cfg['searching']['resume']):
                self.logger.info(
                    "Loading model and optimizer from checkpoint '{}'".format(
                        self.cfg['searching']['resume']
                    )
                )
                checkpoint = torch.load(self.cfg['searching']['resume'], map_location=self.device)
                self.start_epoch = checkpoint['epoch']
                self.dur_time = checkpoint['dur_time']
                self.cur_count = 0
                self.geno_type = checkpoint['geno_type']
                self.architect.optimizer.load_state_dict(checkpoint['arch_optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                self.model.load_state_dict(checkpoint['model_state'])
                self.model_optimizer.load_state_dict(checkpoint['model_optimizer'])
                self.arch_optimizer.load_state_dict(checkpoint['arch_optimizer'])
                self.model.load_alphas(checkpoint['alphas_dict'])
            else:
                self.logger.info("No checkpoint found at '{}'".format(self.cfg['searching']['resume']))

    def run(self):
        self.logger.info('args = {}'.format(self.cfg))
        # Setup Metrics
        self.metric_train = SegmentationMetric(self.n_classes)
        self.metric_val = SegmentationMetric(self.n_classes)
        self.train_loss_meter = average_meter()
        self.val_loss_meter = average_meter()
        run_start = time.time()
        with open("alphas_file.txt", "w+") as OUT:
            for epoch in range(self.start_epoch, self.cfg['searching']['epoch']):
                self.epoch = epoch

                # update scheduler
                self.scheduler.step()
                self.logger.info('epoch %d / %d lr %e', self.epoch,
                                 self.cfg['searching']['epoch'], self.scheduler.get_lr()[-1])

                # get genotype
                genotype = self.model.genotype()
                self.logger.info('genotype = %s', genotype)

                alphas_normal_down = F.softmax(self.model.alphas_normal_down, dim=-1).cpu().detach().numpy()
                alphas_down = F.softmax(self.model.alphas_down, dim=-1).cpu().detach().numpy()
                alphas_normal_up = F.softmax(self.model.alphas_normal_up, dim=-1).cpu().detach().numpy()
                alphas_up = F.softmax(self.model.alphas_up, dim=-1).cpu().detach().numpy()

                # Note the ordering of how weights are scaled does not matter, as long as it is consistent

                OUT.write(f"{array_to_string(alphas_normal_down)}[]")
                OUT.write(f"{array_to_string(alphas_down)}[]")
                OUT.write(f"{array_to_string(alphas_normal_up)}[]")
                OUT.write(f"{array_to_string(alphas_up)}\n")

                # the performance may be unstable, before train in a degree
                if self.epoch >= self.cfg['searching']['alpha_begin']:
                    # check whether the genotype has changed
                    if self.geno_type == genotype:
                        self.cur_count += 1
                    else:
                        self.cur_count = 0
                        self.geno_type = genotype

                    self.logger.info('curr_cout = {}'.format(self.cur_count))

                    if self.cur_count >= self.cfg['searching']['max_patience']:
                        self.logger.info('Reach the max patience! \n best genotype {}'.format(genotype))
                        break

                # train and search the model
                self.train()

                # validate the model
                #self.infer()

                self.model.alphas_dict()

                save_checkpoint({
                    'epoch': epoch + 1,
                    'dur_time': self.dur_time + time.time() - run_start,
                    'cur_count': self.cur_count,
                    'geno_type': self.geno_type,
                    'model_state': self.model.state_dict(),
                    'arch_optimizer': self.arch_optimizer.state_dict(),
                    'model_optimizer': self.model_optimizer.state_dict(),
                    'alphas_dict': self.model.alphas_dict(),
                    'scheduler': self.scheduler.state_dict()
                },False, self.save_path)
                self.logger.info('save checkpoint (epoch %d) in %s  dur_time: %s'
                            , epoch, self.save_path, calc_time(self.dur_time + time.time() - run_start))

                self.metric_train.reset()
                self.metric_val.reset()
                self.val_loss_meter.reset()
                self.train_loss_meter.reset()

        # export scalar data to JSON for external processing
        wandb.log({"args": self.args})
        #self.writer.export_scalars_to_json(self.save_tbx_log + "/all_scalars.json")
        #self.writer.close()

    def train(self):

        training_time = 0.0
        loading_time = 0.0

        start_time = time.perf_counter()
        end_prev_iter = time.perf_counter()

        self.model.train()
        tbar = tqdm(self.train_queue)
        for step, (input, target) in enumerate(tbar):

            input = input.to(self.device)
            target = target.to(self.device)

            start_cur_iter = time.perf_counter()
            loading_time += start_cur_iter - end_prev_iter
            start_training_time = time.perf_counter()

            # Get a random mini-batch from the search queue
            input_valid, target_valid = next(iter(self.valid_queue))
            input_valid = input_valid.to(self.device)
            target_valid = target_valid.to(self.device)

            # Update the architecture parameters first!
            # Update the architecture parameters when the model weights
            # trained in a degree
            if self.epoch >= self.cfg['searching']['alpha_begin']:
                self.architect.step(input_valid, target_valid)

            self.model_optimizer.zero_grad()
            torch.cuda.synchronize()
            start_training = time.perf_counter()

            predicts = None 
            train_loss = None 

            if self.args.mixed_precision:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    predicts = self.model(input)
                    train_loss = self.criterion(predicts, target)
            else:
                predicts = self.model(input)
                train_loss = self.criterion(predicts, target)

            self.train_loss_meter.update(train_loss.item())
            self.metric_train.update(target, predicts)

            # Update the network parameters
            if self.args.mixed_precision:
                self.grad_scaler.scale(train_loss).backward()
                # This leads to issues with learning; comment out clipping code.
                #nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['searching']['grad_clip'])
                self.grad_scaler.step(self.model_optimizer)
                self.grad_scaler.update()
                self.model_optimizer.zero_grad(set_to_none=True)
            else:
                train_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.cfg['searching']['grad_clip'])
                self.model_optimizer.step()

            torch.cuda.synchronize()
            end_training = time.perf_counter()

            if step % self.cfg['searching']['report_freq'] == 0:
                pixAcc, mIoU = self.metric_train.get()
                tbar.set_description('loss: %.5f; pixAcc: %.3f; mIoU: %.5f' % (train_loss, pixAcc, mIoU))

            training_time += end_training - start_training
            end_prev_iter = time.perf_counter()

        #self.writer.add_scalar('Train/Loss', self.train_loss_meter.avg, self.epoch)
        wandb.log({"Train/Loss": self.train_loss_meter.avg, "Epoch": self.epoch})
        end_time = time.perf_counter()
        epoch_time = end_time - start_time

        print(f"Loading time for epoch: {loading_time}")
        print(f"Training time for epoch: {training_time}")
        print(f"Time for epoch: {epoch_time}")

        wandb.log({"Loading time": loading_time, "Training time": training_time, "Total Time": epoch_time, "Epoch": self.epoch})

    def infer(self):

        self.model.eval()
        tbar = tqdm(self.valid_queue)
        with torch.no_grad():
            for step, (input, target) in enumerate(tbar):
                input = input.to(self.device)
                target = target.to(self.device)
                predicts = self.model(input)

                val_loss = self.criterion(predicts, target)
                self.val_loss_meter.update(val_loss.item())

                self.metric_val.update(target, predicts)
                if step % self.cfg['searching']['report_freq'] == 0:
                    pixAcc, mIoU = self.metric_val.get()
                    loss_v = self.val_loss_meter.avg
                    self.logger.info('Val loss: %.6f; pixAcc: %.3f; mIoU: %.5f' % (loss_v, pixAcc, mIoU))
                    tbar.set_description('Val loss: %.6f; pixAcc: %.3f; mIoU: %.5f' % (loss_v, pixAcc, mIoU))

        pixAcc, mIoU = self.metric_val.get()
        cur_loss = self.val_loss_meter.mloss
        #self.writer.add_scalar('Val/pixAcc', pixAcc, self.epoch)
        #self.writer.add_scalar('Val/mIoU', mIoU, self.epoch)
        #self.writer.add_scalar('Val/loss', cur_loss, self.epoch)
        wandb.log({'Val/pixAcc': pixAcc, 'Val/mIoU': mIoU, 'Val/loss': loss, "Epoch": self.epoch})

if __name__ == '__main__':
    search_network = SearchNetwork()
    search_network.run()
