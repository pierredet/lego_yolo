import ConfigParser


def read_cfg(cfg_file):
    cfg = ConfigParser.ConfigParser()
    cfg.read(cfg_file)

    ann_path = cfg.get('general', 'ann_path')
    val_ann_path = cfg.get('general', 'val_ann_path')
    ckpt_path = cfg.get('general', 'ckpt_path')
    labels = cfg.get('general', 'labels').split()
    exclusive = cfg.getboolean('general', 'exclusive')

    batch = cfg.getint('net', 'batch')
    epoch = cfg.getint('general', 'epoch')
    lr = cfg.getfloat('general', 'learning_rate')
    save_iter = cfg.getint('general', 'save_iter')

    # load detection metaparameter from the config file
    meta = {}
    for key, value in cfg.items('detection'):
        if '.' in value:
            meta[key] = float(value)
        else:
            meta[key] = int(value)
    meta['inp_size'] = [cfg.getint('net', 'height'),
                        cfg.getint('net', 'width'),
                        cfg.getint('net', 'channels')]
    meta['labels'] = labels
    meta['lr'] = lr
    meta['batch'] = batch
    meta['model'] = ann_path.split('/')[-1]
    meta['ann_path'] = ann_path

    tu = (ann_path, val_ann_path, ckpt_path, labels, exclusive, batch, epoch,
          lr, save_iter, meta)
    return tu
