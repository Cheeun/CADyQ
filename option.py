import argparse

parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--gpu_id', default="0",type=str,
                    help='the gpu id of using')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--threshld_ratio', type=float, default=1.0,
                    help='random seed')
# Data specifications
parser.add_argument('--dir_data', type=str, default='/mnt/disk1/cheeun914/datasets/',
                    help='dataset image directory')
parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='Set5',
                    help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-800/801-810',
                    help='train/test data range')
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--scale', type=str, default='4',
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=96,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')
parser.add_argument('--kl', action='store_true',
                    help='use kl')
parser.add_argument('--conv_idx', type=str, default='22',
                    help='vgg index')
                    
# Model specifications
parser.add_argument('--model', default='EDSR',
                    help='model name')
parser.add_argument('--pix_type', default='l1',
                    help='model name')
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--teacher_weights', type=str, default=None,
                    help='pretrained model directory for teacher initialization')
parser.add_argument('--student_weights', type=str, default=None,
                    help='pretrained model directory for student initialization')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')
parser.add_argument('--k_bits', type=int, default=32,
                    help='The k_bits of the quantize')

# Option for Residual dense network (RDN)
parser.add_argument('--G0', type=int, default=64,
                    help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='B',
                    help='parameters config of RDN. (Use in RDN)')

# Option for Residual channel attention network (RCAN)
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')

#---------------------IDN-------------------------
parser.add_argument('--idn_d', type=int, default=16)
parser.add_argument('--idn_s', type=int, default=4)
# use n_feats of above : default = 64

#---------------------CARN-------------------------
parser.add_argument('--multi_scale',action='store_true')
parser.add_argument('--group', type=int, default=1)


# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=30,
                    help='number of epochs to train')
parser.add_argument('--ema_epoch', type=int, default=1,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--w_l1', type=float, default=1.0, help='learning rate for L1')
parser.add_argument('--w_at', type=float, default=1e+3, help='learning rate for distillation loss') 
parser.add_argument('--w_bit', type=float, default=0.5, help='learning rate for bit regularization loss') 

parser.add_argument('--decay', type=str, default='150',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--nesterov', type=bool, default=False,
                    help='nesterov')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=5,
                    help='gradient clipping threshold (0 = no clipping)')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e8',
                    help='skipping batch that has large error')
parser.add_argument('--loss_kd', action='store_true', help='trainning with knowledge distillation loss') 
parser.add_argument('--loss_kdf', action='store_true', help='trainning with feature knowledge distillation loss') 


# Log specifications
parser.add_argument('--suffix', default=None, type=str, 
                    help='suffix to help you remember what experiment you ran')
parser.add_argument('--save', type=str, default='test',
                    help='file name to save')
parser.add_argument('--load', type=str, default='',
                    help='file name to load')
parser.add_argument('--resume', type=str, default=None,
                    help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')


parser.add_argument('--cadyq', action='store_true', help='mixed precision for layer') 
parser.add_argument('--search_space', type=str, default='32', help='bit search space')


parser.add_argument('--bitsel_lr', type=float, default=1e-4)
parser.add_argument('--bitsel_decay', type=str, default=150)
parser.add_argument('--w_bit_decay', type=float, default=2e-6)

parser.add_argument('--test_patch', action='store_true', help='testing patch-wise') 
parser.add_argument('--step_size', type=int, default=28, help='step size for combining patches')
parser.add_argument('--save_patch', action='store_true',help='save patch results')

# parser.add_argument('--linq', action='store_true', help='linq') 
# parser.add_argument('--fully', action='store_true', help='full quantization') 
# parser.add_argument('--train_full', action='store_true', help='pretraining 32 bit model') 
# parser.add_argument('--lpips', action='store_true', help='use lpips loss for optimization') 


# FSRCNN
parser.add_argument('--m', type=int, default=4, help='m')

args = parser.parse_args()

args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')

args.search_space = list(map(lambda x: int(x), args.search_space.split('+')))
# [2,4,8]
if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

