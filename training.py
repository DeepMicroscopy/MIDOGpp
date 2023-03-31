from torch.utils.data import SubsetRandomSampler
from slide.slide_helper import *
from slide.data_loader import *
from utils.detection_helper import create_anchors
from object_detection_fastai.loss.RetinaNetFocalLoss import RetinaNetFocalLoss
from object_detection_fastai.models.RetinaNet import RetinaNet
from utils.callbacks import PascalVOCMetricByDistance, BBMetrics
import wandb
from wandb.fastai import WandbCallback
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import StratifiedKFold
import pandas as pd

def get_y_func(x):
    return x.y

def training(cfg: DictConfig):
    # Confirm that you have a GPU!
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tumortypes =  [tumortype for tumortype in cfg.data.tumortypes.split(",")]
    sizes = [(cfg.retinanet.sizes, cfg.retinanet.sizes)]
    ratios = [cfg.retinanet.ratios]
    scales = [float(s) for s in cfg.retinanet.scales.split(",")]

    files, _, test_files = load_images(Path(cfg.files.image_path), cfg.files.annotation_file, level=cfg.data.level, patch_size=cfg.data.patch_size, categories=[1], tumortypes=tumortypes)
    with open('statistics_sdata.pickle', 'rb') as handle:
        statistics = pickle.load(handle)
    mean = np.array(np.mean(np.array([value for key,value in statistics['mean'].items() if tumortypes.__contains__(key)]), axis=(0,1)), dtype=np.float32)
    std = np.array(np.mean(np.array([value for key,value in statistics['std'].items() if tumortypes.__contains__(key)]), axis=(0,1)), dtype=np.float32)

    tfms = get_transforms(do_flip=True,
                          flip_vert=True,
                          max_lighting=0.5,
                          max_zoom=2,
                          max_warp=0.2,
                          p_affine=0.5,
                          p_lighting=0.5,
                          )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    bins = pd.qcut(pd.Series([len(t.y[1]) for t in files]), 3, labels=False)
    for train_index, val_index in skf.split(files, bins):
        cfg.update({'x-validation': {'train': json.dumps([files[i].file.name for i in train_index]), 'valid': json.dumps([files[i].file.name for i in val_index])}})
        run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
                         config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True), reinit=True)
        train_files = [files[i] for i in train_index]
        valid_files = [files[i] for i in val_index]

        # Create FastAi data bunch
        item_list = ItemLists(path=Path('.'), train=ObjectItemListSlide(train_files), valid=ObjectItemListSlide(valid_files))
        item_list = item_list.label_from_func(get_y_func, label_cls=SlideObjectCategoryList)
        data = item_list.transform(tfms, size=cfg.data.patch_size, tfm_y=True).databunch(bs=cfg.data.batch_size, no_check=True,num_workers=0, pin_memory=True,device=device, collate_fn=bb_pad_collate).normalize((mean,std))
        data.train_dl = data.train_dl.new(shuffle=False,sampler=SubsetRandomSampler(indices=create_indices(train_files, cfg.data.train_patches)))
        data.valid_dl = data.valid_dl.new(shuffle=False,sampler=SubsetRandomSampler(indices=create_indices(valid_files, cfg.data.valid_patches)))

        # TRAINING
        # Model Summary
        print("Input Size: {} x {}".format(cfg.data.patch_size, cfg.data.patch_size))
        print("Resolution Level: {}".format(cfg.data.level))
        print("Batch Size: {}".format(cfg.data.batch_size))
        print("Training Set: {} Slides with {} Patches".format(len(train_files), len(data.train_dl)*cfg.data.batch_size))
        print("Validation Set: {} Slides with {} Patches".format(len(valid_files), len(data.valid_dl)*cfg.data.batch_size))

        anchors = create_anchors(sizes=sizes, ratios=ratios, scales=scales)
        crit = RetinaNetFocalLoss(anchors=anchors)
        encoder = create_body(models.resnet18, True, -2)
        model = RetinaNet(encoder, n_classes=data.train_ds.c, n_anchors=len(scales) * len(ratios),sizes=[size[0] for size in sizes], chs=128, final_bias=-4., n_conv=3)
        voc = PascalVOCMetricByDistance(anchors, cfg.data.patch_size, [str(i) for i in data.train_ds.y.classes[1:]])
        learn = Learner(data, model, loss_func=crit, metrics=[voc], callback_fns=[BBMetrics])
        learn.model_dir = run.dir
        learn.fit_one_cycle(cfg.training.num_epochs, max_lr=slice(cfg.training.lr),callbacks=[WandbCallback(learn, save_model=True, mode='max', monitor='AP-mitotic figure')])
        wandb.join()




