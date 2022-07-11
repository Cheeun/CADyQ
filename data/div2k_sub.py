import os
from data import srdata

class DIV2K_sub(srdata.SRData):
    def __init__(self, args, name='DIV2K_sub', train=True, benchmark=True):
        # data_range = [r.split('-') for r in args.data_range.split('/')]
       
        # if train:
        #     data_range = data_range[0]
        # else:
        #     if args.test_only and len(data_range) == 1:
        #         data_range = data_range[0]
        #     else:
        #         data_range = data_range[1]
        # self.begin, self.end = list(map(lambda x: int(x), data_range))
        self.data_class = args.data_class
        print("Training data : Loading from DIV2K_sub_std_class"+str(self.data_class))

        super(DIV2K_sub, self).__init__(
            args, name=name, train=train, benchmark=True
        )
        

    # def _scan(self):
    #     names_hr, names_lr = super(DIV2K_sub, self)._scan()
    #     names_hr = names_hr[self.begin - 1:self.end]
    #     names_lr = [n[self.begin - 1:self.end] for n in names_lr]

    #     return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        # super(DIV2K_sub, self)._set_filesystem(dir_data)
        self.apath = os.path.join(dir_data, 'DIV2K')
        self.dir_hr = os.path.join(self.apath, 'DIV2K_scale_sub_std_GT_class'+str(self.data_class))
        self.dir_lr = os.path.join(self.apath, 'DIV2K_scale_sub_std_LR_class'+str(self.data_class))
        self.ext = ('', '.png')


