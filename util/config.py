device = 'cuda'
debug = True
num_workers = 0
inp_shape = (128, 128)
discr_out = (1, 14, 14)
AVAIL_GPUS = 1 if device == 'cuda' else None
BATCH_SIZE = 256

