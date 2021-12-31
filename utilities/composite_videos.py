import os
from os import path
import numpy as np
import cv2
from glob import glob


mode = 'composite'

def save_video(video,savepath,fps= 5):
    writer = cv2.VideoWriter(
        savepath,
        cv2.VideoWriter_fourcc(*"MP4V"),
        fps,
        (video.shape[2], video.shape[1]),
    )

    # writer = vio.FFmpegWriter(savename,inputdict=inputdict,outputdict=outputdict)

    for frame in video:
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)

    writer.release()




if mode == 'composite':

    prefix = 'ci-'


    #samples_dir = path.join(basedir,'generated',run_name,samples_dir,f'sid_{idx}')
    samples_dir = '/export/scratch/ablattma/poking_inn/iccv21/videos_supple/gui/plants'

    if "DATAPATH" in os.environ:
        samples_dir = os.path.join(os.environ["DATAPATH"], samples_dir[1:])

    row_dirs = glob(path.join(samples_dir,'id_*'))
    rows = []
    for rd in row_dirs:

        row = glob(path.join(rd,'vid_*.mp4'))

        row.insert(0,path.join(rd,'gt_vid.mp4'))
        row.insert(1, glob(path.join(rd, 'gt_poke_vid_*.mp4'))[0])

        # baseline_samples = glob(path.join(rd,f'{prefix}*.mp4'))
        #
        # row.extend(baseline_samples)

        loaded_vids = []
        for vid in row:
            act_sequence = []
            vidcap = cv2.VideoCapture(vid)
            success, image = vidcap.read()

            count = 0
            while success:
                # test = np.where(np.expand_dims(ep_img[:, :, -1], axis=-1), ep_img, image) if ep_img is not None else image
                act_sequence.append(image)
                success, image = vidcap.read()
                #print('Read a new frame: ', success)
                count += 1

            loaded_vids.append(np.stack(act_sequence))

        sample_row = np.concatenate(loaded_vids,axis=2)
        rows.append(sample_row)

    grid = np.concatenate(rows,axis=1)
    savename = path.join(samples_dir,f'ui_grid.mp4')
    print(f'save video to {savename}')
    save_video(grid,savename,3)
elif mode == 'transfer':
    samples_dir = '/export/scratch/ablattma/poking_inn/iccv21/videos_supple/iper/transfer'

    if "DATAPATH" in os.environ:
        samples_dir = os.path.join(os.environ["DATAPATH"], samples_dir[1:])

    transfer_cols = glob(path.join(samples_dir, 'transfer_*.mp4'))

    read_cols = []

    for vid in transfer_cols:
        act_sequence = []
        vidcap = cv2.VideoCapture(vid)
        success, image = vidcap.read()

        count = 0
        while success:
            # test = np.where(np.expand_dims(ep_img[:, :, -1], axis=-1), ep_img, image) if ep_img is not None else image
            act_sequence.append(image)
            success, image = vidcap.read()
            # print('Read a new frame: ', success)
            count += 1

        read_cols.append(np.stack(act_sequence))

    grid = np.concatenate(read_cols,axis=2)
    savename = path.join(samples_dir,f'transfer_grid.mp4')
    print(f'save video to {savename}')
    save_video(grid, savename, 3)



else:
    import yaml
    from data import get_dataset

    model_name = 'iper-16_10d1-bs40-lr1e-3-bn32-mfcf64-fullseq-ss64-mf10-endpoint-np5-mweight'
    # model_name = 'plants-16_10d1-bs20-lr1e-3-bn64-fullseq-mfc32-ss128-mf10-endpoint-np5'
    # model_name = 'plants-16_10d1-bs20-lr1e-3-bn64-fullseq-mfc32-ss128-mf10-endpoint-np5'
    # model_name = 'iper-16_10d1-bs96-lr1e-3-bn32-fmcf64-fullseq-ss128-mf10-endpoint-np5-complex'




    config_path = path.join(basedir,'config',model_name,'config.yaml')
    with open(config_path,'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['data']['filter']  = 'all'

    dset, transforms = get_dataset(config['data'])
    test_dataset = dset(transforms,["images","poke","flow",'sample_ids'],config['data'],train=False)



    basepath = '/export/scratch/ablattma/poking_inn/iccv21/videos_supple/iper/comparison_baselines/'

    if "DATAPATH" in os.environ:
        basepath = os.path.join(os.environ["DATAPATH"], basepath[1:])

    ids = np.asarray([int(idx_name.split('_')[-1]) for idx_name in os.listdir(basepath) if idx_name.startswith('id_')],dtype=int)

    image_names=test_dataset.datadict['img_path'][ids]

    savename = path.join(basepath,'image_files.txt')

    with open(savename,'w+') as file:
        for idx,imgname in zip(ids,image_names):
            img = cv2.imread(imgname)
            cv2.imwrite(path.join(basepath,f'src_img_{idx}.png'),img)
            file.writelines(str(imgname)+'\n')