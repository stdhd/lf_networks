import argparse
from os import path,makedirs,listdir
from glob import glob
import numpy as np
import torch
import cv2
from tqdm import tqdm
from natsort import natsorted
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
import seaborn as sns
import yaml
import os

from data import get_dataset

def vis_flow(flow_map, normalize=False):
    if isinstance(flow_map,torch.Tensor):
        flow_map = flow_map.cpu().numpy()
    flows_vis = []
    for flow in flow_map:
        hsv = np.zeros((*flow.shape[1:],3),dtype=np.uint8)
        hsv[...,1] = 255
        mag, ang = cv2.cartToPolar(flow[0], flow[1])
        # since 360 is not valid for uint8, 180° corresponds to 360° for opencv hsv representation. Therefore, we're dividing the angle by 2 after conversion to degrees
        hsv[...,0] = ang * 180 / np.pi / 2
        hsv[...,2] = cv2.normalize(mag,None,alpha=0,beta=255, norm_type=cv2.NORM_MINMAX)

        as_rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
        if normalize:
            as_rgb = as_rgb.astype(float) - as_rgb.min(axis=(0,1),keepdims=True)
            as_rgb = (as_rgb / as_rgb.max(axis=(0,1),keepdims=True)*255.).astype(np.uint8)
        flows_vis.append(as_rgb)

    return flows_vis

def vis_flow_dense(flow_map,**kwargs):
    if isinstance(flow_map,torch.Tensor):
        flow_map = flow_map.cpu().numpy()
    flows_vis = []
    for flow in flow_map:
        h, w = flow.shape[1:]
        fx, fy = flow[0], flow[1]
        ang = np.arctan2(fy, fx) + np.pi
        v = np.sqrt(fx * fx + fy * fy)
        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[..., 0] = ang * (180 / np.pi / 2)
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(v,None,alpha=0,beta=255, norm_type=cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        flows_vis.append(bgr)
    return flows_vis




def get_image(vidcap, frame_number,spatial_size=None):
    vidcap.set(1, frame_number)
    _, img = vidcap.read()
    if spatial_size is not None and spatial_size != img.shape[0]:
        img=cv2.resize(img,(spatial_size,spatial_size),interpolation=cv2.INTER_LINEAR)
    return img


def process_video(f_name, args,target_ids, dataset):
    from utilities.flownet_loader import FlownetPipeline
    from utilities.general import get_gpu_id_with_lowest_memory, get_logger_old
    flows = {}

    target_gpus = None if len(args.target_gpus) == 0 else args.target_gpus
    gpu_index = get_gpu_id_with_lowest_memory(target_gpus=target_gpus)
    torch.cuda.set_device(gpu_index)

    #f_name = vid_path.split(vid_path)[-1]

    logger = get_logger_old(f"{gpu_index}")

    extract_device = torch.device("cuda", gpu_index.index if isinstance(gpu_index,torch.device) else gpu_index)

    # load flownet
    pipeline = FlownetPipeline()
    flownet = pipeline.load_flownet(args, extract_device)

    # yield_ids = {idx: range(idx, idx + (dataset.min_frames + 15) * dataset.subsample_step + 1, dataset.subsample_step) \
    #     if dataset.var_sequence_length else range(idx, idx + dataset.max_frames * dataset.subsample_step + 1, dataset.subsample_step) for idx in target_ids}
    #
    # gt_imgs = {idx: [cv2.imread(test_dataset.datadict["img_path"]) for i in yield_ids[idx]] for idx in yield_ids}






    # open video
    base_raw_dir = args.raw_dir.split("*")[0]

    if not isinstance(f_name,list):
        f_name = [f_name]

    logger.info(f"Iterating over {len(f_name)} files...")
    for fn in tqdm(f_name,):
        vid_path = path.join(base_raw_dir, fn)
        # vid_path = f"Code/input/train_data/movies/{fn}"
        vidcap = cv2.VideoCapture()
        vidcap.open(vid_path)
        counter = 0
        while not vidcap.isOpened():
            counter += 1
            time.sleep(1)
            if counter > 10:
                raise Exception("Could not open movie")

        # get some metadata
        number_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #upright = height > width

        # create target path if not existent
        fname_split = fn.split("/")
        base_path = path.join(args.processed_dir, "/".join(fname_split[:-1]))
        vid_name = fname_split[-1].split(".")[0]
        if not base_path in flows:
            flows.update({"/".join(fname_split[:-1]):[]})


        # base_path = f"Code/input/train_data/images/{f_name.split('.')[0]}/"
        makedirs(base_path, exist_ok=True)

        delta = args.flow_delta
        #diff = args.flow_max

        # begin extraction

        # check for existence
        if args.all_frames_flow:

            for frame_number in range(0, number_frames):
                # check for existence
                first_fidx, second_fidx = 0, number_frames - frame_number
                image_target_file = path.join(base_path, f"frame_{frame_number}.png")

                img = get_image(vidcap, frame_number)
                if img is None:
                    raise FileNotFoundError("no img")


                    # if success:
                    #     logger.info(f'wrote img with shape {img.shape} to "{image_target_file}".')
                # FLOW

                if second_fidx < number_frames:
                    flow_target_file = path.join(
                        base_path, f"{vid_name}_prediction_{first_fidx}_{second_fidx}.flow"
                    )

                    img, img2 = (
                        get_image(vidcap, first_fidx),
                        get_image(vidcap, second_fidx),
                    )
                    # if upright:
                    #     img, img2 = cv2.transpose(img), cv2.transpose(img2)
                    sample = pipeline.preprocess_image(img, img2, "BGR", spatial_size=args.input_size).to(
                        extract_device
                    )
                    prediction = (
                        pipeline.predict(flownet, sample[None], spatial_size=args.spatial_size)
                            .cpu()
                            .detach()
                            .numpy()
                    )
                    np.save(flow_target_file, prediction)
                    flows["/".join(fname_split[:-1])].append(prediction)

            logger.info(
                f'Finish processing video sequence "{fn}".')


        else:
            first_fidx, second_fidx = 0, delta-1
            image_target_file = path.join(base_path, f"src_frame.png")
            # image_target_file = f"{base_path}frame_{frame_number}.png"
            # FRAME
            if not path.exists(image_target_file):
                # write frame itself
                img = get_image(vidcap, 0)
                if img is None:
                    continue
                # if upright:
                #     img = cv2.transpose(img)
                try:
                    if args.spatial_size is None:
                        success = cv2.imwrite(image_target_file, img)
                    else:
                        img_res = cv2.resize(img,(args.spatial_size,args.spatial_size), interpolation=cv2.INTER_LINEAR)
                        success = cv2.imwrite(image_target_file,img_res)
                except cv2.error as e:
                    print(e)
                    continue
                except Exception as ex:
                    print(ex)
                    continue

            if second_fidx < number_frames:
                flow_target_file = path.join(
                    base_path, f"{vid_name}.flow"
                )
                if not path.exists(flow_target_file + ".npy"):
                    # predict and write flow prediction
                    img, img2 = (
                        get_image(vidcap, first_fidx),
                        get_image(vidcap, second_fidx),
                    )
                    # if upright:
                    #     img, img2 = cv2.transpose(img), cv2.transpose(img2)
                    sample = pipeline.preprocess_image(img, img2, "BGR",spatial_size=args.input_size).to(
                        extract_device
                    )
                    prediction = (
                        pipeline.predict(flownet, sample[None],spatial_size=args.spatial_size)
                        .cpu()
                        .detach()
                        .numpy()
                    )
                    np.save(flow_target_file, prediction)
                    flows["/".join(fname_split[:-1])].append(prediction)

            logger.info(
                f'Finish processing video "{fn}".')

    return flows


def create_heatmaps(vars, valid_pixls, args, prefix, size, bp):
    ids = np.random.choice(valid_pixls, int(min(args.n_ex,valid_pixls.shape[0])), replace=False)
    for n, idx in enumerate(tqdm(ids)):
        cors = vars[idx]
        # normalize
        cors = cors / cors.max()
        hm = (1 - cors).reshape(args.spatial_size, args.spatial_size)
        hm = (hm * 255.).astype(np.uint8)
        hm = cv2.GaussianBlur(hm, (5, 5), 1.5)
        pixel_loc = (idx % size, int(idx / size))

        fig = plt.figure()
        ax = sns.heatmap((hm.astype(float) / 255.), cmap="viridis", xticklabels=False, yticklabels=False)
        ax.plot([pixel_loc[0]], [pixel_loc[1]], "rx")

        plt.savefig(path.join(args.processed_dir, bp, f"{prefix}-part_vis-legend-{n}.png"))
        plt.close()



        hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        #hm_smooth = cv2.GaussianBlur(hm, (3, 3), 1.5)
        #hm_smooth = cv2.circle(hm_smooth, pixel_loc, 2, (255, 0, 0), -1)

        overlay = cv2.addWeighted(img, 0., hm, 1., 0)
        overlay = cv2.circle(overlay, pixel_loc, 2, (0, 255, 0), -1)
        cv2.imwrite(path.join(args.processed_dir, bp, f"{prefix}-part_vis-{n}.png"), overlay)
        #cv2.imwrite(path.join(args.processed_dir, bp, f"{prefix}-part_vis-{n}_gauss.png"), hm_smooth)


if __name__ == '__main__':
    import time
    from utilities.general import get_logger_old

    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_max", type=float, default=1.0)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Run model in pseudo-fp16 mode (fp16 storage fp32 math).",
    )
    parser.add_argument(
        "--fp16_scale",
        type=float,
        default=1024.0,
        help="Loss scaling, positive power of 2 values can improve fp16 convergence.",
    )
    parser.add_argument(
        "--raw_dir",
        "-v",
        type=str,
        default="/export/data/ablattma/Datasets/plants/cropped/",
    )
    parser.add_argument(
        "--processed_dir",
        "-p",
        type=str,
        default="/export/scratch/ablattma/Datasets/plants/processed_256/",
    )
    parser.add_argument(
        "--flow_delta",
        "-fd",
        type=int,
        default=5,
        help="The number of frames between two subsequently extracted flows.",
    )
    #parser.add_argument("--flow_max", "-fm", type=int, default=5)
    parser.add_argument("--spatial_size", "-s", type=int, default=128, help="The desired spatial_size of the output.")
    parser.add_argument("--input_size", "-is", type=int, default=256, help="The input size for the flownet (images are resized to this size, if not divisible by 64.")
    parser.add_argument("--target_gpus", default=[], type=int,
                        nargs="+", help="GPU's to use.")
    parser.add_argument("--vis_flows",action="store_true",default=False,help="Whether or not to visualize flow estimations")
    parser.add_argument("--foreground_sep", action="store_true", default=False, help="Whether to pre-separate foreground and background")
    parser.add_argument("--n_ex", type=int, default=10, help="Number of examples to store per src_img")
    parser.add_argument("--estimate_flows", action="store_true",default=False,help="Whether or not to estimate flows")
    parser.add_argument("--n_clusters", type=int, default=11, help="The number of clusters which shall be found")
    parser.add_argument("--linkage_type", type=str, choices=["ward","complete","single","average"],default="complete",help="The linkage type for the agglomerative clustering procedure")
    parser.add_argument("--save_vars",action="store_true", default=False,help="Save sim matrix variances or not.")
    parser.add_argument("--frame_id", default=None, type=int,help="target frame id")
    parser.add_argument("-sf","--sep_flow",default=False, action="store_true",help="Whether or not to use (gt) optical flow for foreground background separation.")
    parser.add_argument("-c","--config",type=str,default="/export/scratch/ablattma/poking_inn/second_stage_video/config/plants-16_10d1-bs20-lr1e-3-bn64-fullseq-mfc32-ss128-mf10-endpoint-np5/config.yaml")
    parser.add_argument("-om", "--only_mask", default=False, action="store_true", help="Whether or not to only compute masks.")
    parser.add_argument("-of","--only_flow", action="store_true", default=False, help="Whether to only to estimate flow or not")
    parser.add_argument("-aff", "--all_frames_flow", action="store_true", default=False, help="Whether to estimate flow for all frames")
    parser.add_argument("-vdf", "--vis_flow_dense", action="store_true", default=False, help="Whether to use opencv tutorial code or not")



    args = parser.parse_args()
    # parameters for part vis for plants
    # --raw_dir "/export/scratch/ablattma/visual_poking/two_stage_model/generated/plants-two_stage-ablation-ldyn-128/gui/*" --processed_dir "/export/scratch/ablattma/visual_poking/two_stage_model/generated/plants-two_stage-ablation-ldyn-128/sim_mats_foreground" --target_gpus 1 --n_ex 10 --foreground_sep


    if "DATAPATH" in os.environ:
        args.raw_dir = os.path.join(os.environ["DATAPATH"],args.raw_dir[1:])
        args.config = os.path.join(os.environ["DATAPATH"],args.config[1:])

    # load config
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset, transforms = get_dataset(config["data"])

    logger = get_logger_old(args.raw_dir)
    logger.info("Start visualization")

    flow_width_factor = 5
    valid_h = (5,args.spatial_size-5)

    # check if raw dir has wildcards and extract base_raw_dir
    base_raw_dir = args.raw_dir.split("*")[0]
    assert path.isdir(base_raw_dir)


    # --raw_dir "/export/scratch/ablattma/visual_poking/sequence_poke_model/generated/iper-deep-maxf9-bs3-skip-128-ldl5-pdl0-vgg5-patchganorig_gp.0001-gantmpold.1gpfmap10-longseq20/gui/id_40318/*" --processed_dir "/export/scratch/ablattma/visual_poking/sequence_poke_model/generated/iper-deep-maxf9-bs3-skip-128-ldl5-pdl0-vgg5-patchganorig_gp.0001-gantmpold.1gpfmap10-longseq20/gui/id_40318/" --target_gpus 5 --n_ex 10 --foreground_sep -of --vis_flows -aff --config /export/scratch/ablattma/visual_poking/sequence_poke_model/config/iper-deep-maxf9-bs3-skip-128-ldl5-pdl0-vgg5-patchganorig_gp.0001-gantmpold.1gpfmap10-longseq20/config.yaml

    # target_ids = [7270,8208,2832,4430,207,6284]
    # these were for iper
    #target_ids = [40977,9747]

    # here are the target ids for plants
    target_ids = [4462]

    # only dummy dataset
    test_dataset = dataset(transforms, ["images", "poke", "flow"], config["data"], train=False)

    f_names = [p.split(base_raw_dir)[-1] for p in glob(path.join(args.raw_dir,"*.mp4")) ]
    logger.info(f'fnames are {f_names}; base raw dir {base_raw_dir}')
    #f_names = [fn for fn in f_names if int(fn.split("/")[-2].split("_")[-1]) in target_ids]
    # estimate flows and store them on disk
    if args.estimate_flows:
        flows= process_video(f_names,args,target_ids,test_dataset)
        flow_file_names = None
        #flow_file_names = {bp: [f.split("/")[-1].split(".")[0] for f in natsorted(glob(path.join(args.processed_dir, bp, "*.flow.npy")))] for bp in listdir(args.processed_dir)}
    else:
        flows = {bp: [np.load(f) for f in natsorted(glob(path.join(args.processed_dir,bp,"*.flow.npy")))] for bp in listdir(args.processed_dir) if path.isdir(path.join(args.processed_dir,bp))}
        flow_file_names = {bp: [f.split("/")[-1].split(".")[0] for f in natsorted(glob(path.join(args.processed_dir,bp,"*.flow.npy")))] for bp in listdir(args.processed_dir)}


    clusterer = AgglomerativeClustering(n_clusters=args.n_clusters,affinity="precomputed",linkage=args.linkage_type)

    for flow in flows:
        logger.info(f'flows-dict contains key: "{flow}"; flow_file_name {flow_file_names[flow]}')

    # bp2idx = {bp: int(base_raw_dir.split("/")[-1].split("_")[-1]) for bp in flows}
    bp2idx = {bp: int(bp.split("_")[-1]) for bp in flows}


    gt_flows = {bp: test_dataset._get_flow((bp2idx[bp], None)).numpy() for bp in bp2idx}

    #bp2idx = {bp:int(bp.split("/")[-1].split("_")[-1]) for bp in flows}
    # base_raw_dir = base_raw_dir[:-1] if base_raw_dir.endswith("/") else base_raw_dir



    for bp in tqdm(flows):
        files = natsorted(glob(path.join(args.processed_dir, bp, "*.flow.npy")))

        logger.info(f'Actual id is {bp}')

        gt_flow = gt_flows[bp]

        if len(flows[bp]) < len(files):
            curr_flows = [np.load(f) for f in files]
            flows[bp].extend(curr_flows)


        # get flows as 3-channel image where the first channel is the angle and the last the magnitude
        flow_fn = vis_flow_dense if args.vis_flow_dense else vis_flow
        flows_vis = flow_fn(flows[bp],normalize=False)
        flow_vis_gt = flow_fn([gt_flow],normalize=False)[0]
        cv2.imwrite(path.join(args.processed_dir, bp,f"flows_vis-gt.png"),flow_vis_gt)

        # visualize esimated flows if desired
        if args.vis_flows:
            for i, f in enumerate(flows_vis):
                save_name = flow_file_names[bp][i]+"_vis.png" if flow_file_names is not None else f"flows_vis-{i}.png"
                if args.vis_flow_dense:
                    save_name = save_name[:-4] + "_cv2" +save_name[-4:]
                cv2.imwrite(path.join(args.processed_dir, bp,save_name),f)


        # flow_vecs = [np.moveaxis(f,0,-1) for f in flows[bp]]
        # flow_vecs = [f.reshape(-1,f.shape[-1]) for f in flow_vecs]

        if not args.only_flow:
            img = cv2.imread(path.join(args.processed_dir, bp, "src_frame.png"))
            valid_pixls = None
            if args.foreground_sep and bp2idx[bp] in target_ids:

                amplitude = np.linalg.norm(gt_flow, 2, axis=0)
                amplitude -= amplitude.min()
                amplitude /= amplitude.max()

                # use only such regions where the amplitude is larger than mean + 1 * std
                mask = np.where(np.greater(amplitude, amplitude.mean()- .3 * amplitude.std()), np.ones_like(amplitude), np.zeros_like(amplitude)).astype(bool)
                cv2.imwrite(path.join(args.processed_dir, bp, f"fgbg_flow.png"), mask * 255)
                mask_flow = mask.reshape(-1).astype(bool)
                # valid_pixls = np.nonzero(mask)
                # valid_pixls = valid_pixls[0]


                mask = np.zeros(img.shape[:2], np.uint8)
                # rect defines starting background area

                rect = (int(img.shape[1] / 5), int(valid_h[0]), int((flow_width_factor - 2) / flow_width_factor * img.shape[1]), int(valid_h[1] - valid_h[0]))

                #rect = (0,15,128,133)
                # rect = (0,0,128,128)
                # initialize background and foreground models
                fgm = np.zeros((1, 65), dtype=np.float64)
                bgm = np.zeros((1, 65), dtype=np.float64)
                # apply grab cut algorithm
                mask_src, fgm, bgm = cv2.grabCut(img, mask, rect, fgm, bgm, 500, cv2.GC_INIT_WITH_RECT)
                mask_src = np.where((mask_src == 2) | (mask_src == 0), 0, 1)
                cv2.imwrite(path.join(args.processed_dir, bp, f"fgbg.png"), mask_src * 255)
                mask_src = mask_src.reshape(-1).astype(bool)
                mask_src = np.logical_or(mask_flow,mask_src)
                valid_pixls = np.nonzero(mask_src)
                valid_pixls = valid_pixls[0]


            if not args.only_mask:

                sim_mats = []
                sim_mats_filt = []
                sim_mats_polar = []
                for n, fm in enumerate(tqdm(flows[bp],desc=f"computing similarity matrices of {len(flows[bp])} samples....")):
                    magn = np.linalg.norm(fm, axis=0, )
                    magn -= magn.min()
                    magn /= magn.max()
                    std = magn.std()
                    mean = np.mean(magn)
                    flow_filt = np.where(np.greater(magn, mean + (std)), fm, np.zeros_like(fm))
                    magn_filt = np.where(np.greater(magn, mean + (std)), magn,   np.zeros_like(magn))


                    fv = np.moveaxis(fm,0,-1)
                    fv = fv.reshape(-1,fm.shape[0])
                    ff = np.moveaxis(flow_filt,0,-1)
                    ff = ff.reshape(-1,fm.shape[0])
                    fp = np.stack([np.sqrt(flow_filt[0]**2+flow_filt[1]**2),np.arctan2(flow_filt[1],flow_filt[0])],axis=-1).reshape(-1,flow_filt.shape[0])
                    sim_mats.append(pairwise_distances(fv,metric="l2"))
                    sim_mats_filt.append(pairwise_distances(ff,metric="l2"))
                    sim_mats_polar.append(pairwise_distances(fp,metric="l2"))

                    # make magnitude plot
                    magn = (magn * 255).astype(np.uint8)
                    magn_filt = (magn_filt * 255.).astype(np.uint8)
                    #magn_sep = np.where(magn > magn.mean(),255,0).astype(np.uint8)
                    magn = cv2.applyColorMap(magn, cv2.COLORMAP_VIRIDIS)
                    magn_filt = cv2.applyColorMap(magn_filt, cv2.COLORMAP_VIRIDIS)
                    cv2.imwrite(path.join(args.processed_dir, bp, f"magn-{n}.png"),magn)
                    cv2.imwrite(path.join(args.processed_dir, bp, f"magn_filt-{n}.png"), magn_filt)



                sim_mats = np.stack(sim_mats,axis=0)
                sim_mats_filt = np.stack(sim_mats_filt,axis=0)
                sim_mats_polar = np.stack(sim_mats_polar, axis=0)
                # get variance in all similarity maps
                logger.info("Compute variance of similarity matrices")
                vars = np.var(sim_mats,axis=0)
                vars_filt = np.var(sim_mats_filt, axis=0)
                vars_polar = np.var(sim_mats_polar, axis=0)
                #means = np.mean(sim_mats,axis=0)

                if args.save_vars:
                    sd = path.join(args.processed_dir,bp,"variances")
                    makedirs(sd, exist_ok=True)
                    np.save(path.join(sd,"vars_raw.npy"),vars)
                    np.save(path.join(sd, "sim_mats_raw.npy"), sim_mats)
                    np.save(path.join(sd, "vars_filt.npy"), vars_filt)
                    np.save(path.join(sd, "sim_mats_filt.npy"), sim_mats_filt)
                    np.save(path.join(sd, "vars_polar.npy"), vars_polar)
                    np.save(path.join(sd, "sim_mats_polar.npy"), sim_mats_polar)

                # #cluster
                # logger.info("Cluster pixels")
                # clusters_var = clusterer.fit_predict(vars)
                # clusters_mean = clusterer.fit_predict(means)
                #
                # # visualize
                # clustermap_means = clusters_mean.reshape(flows_vis[0].shape[0],flows_vis[0].shape[1])
                # clustermap_means = np.flipud(clustermap_means)
                # clustermap_vars = clusters_var.reshape(flows_vis[0].shape[0],flows_vis[0].shape[1])
                # clustermap_vars = np.flipud(clustermap_vars)
                #
                # fig, axs = plt.subplots(1,2,squeeze=True)
                # axs[0].pcolor(clustermap_means,cmap="hsv")
                # axs[0].axis("off")
                # axs[0].set_title("Segmentation based on means",fontsize="small")
                #
                # axs[1].pcolor(clustermap_vars, cmap="hsv")
                # axs[1].axis("off")
                # axs[1].set_title("Segmentation based on variances",fontsize="small")
                #
                # plt.savefig(path.join(args.processed_dir, bp, f"segmentation_{args.linkage_type}.png"), dpi="figure")
                # plt.close()
                # get foreground

                if valid_pixls is None:
                    valid_pixls = np.arange(vars.shape[0])

                create_heatmaps(vars,valid_pixls,args,"unfilt",flows_vis[0].shape[0],bp)
                create_heatmaps(vars_filt, valid_pixls, args, "filt", flows_vis[0].shape[0], bp)
                create_heatmaps(vars_polar, valid_pixls, args, "polar", flows_vis[0].shape[0], bp)

                # ang_vecs = [np.expand_dims(f[...,0].reshape(-1),axis=-1) for f in flows_vis]
                #
                #
                # sim_mat_ang = pairwise_distances(ang_vecs,metric="euclidean")



