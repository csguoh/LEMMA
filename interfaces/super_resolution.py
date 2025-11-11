import torch
import os
from datetime import datetime
import pickle
import copy
from utils import util, ssim_psnr
from IPython import embed
import torch.nn.functional as F
import cv2
from interfaces import base
from utils.metrics import get_string_aster, get_string_crnn, Accuracy,get_string_abinet,get_string_parseq
from utils.util import str_filt
import numpy as np
from ptflops import get_model_complexity_info
import editdistance
import time
import logging
from dataset.dataset import Lable2Tensor
from torch import optim as optim
import lpips
import setup
from model.parseq.parseq_tokenizer import get_parseq_tokenize
from PIL import Image
import torchvision.transforms as transforms

parseq_tokenizer = get_parseq_tokenize()
abi_charset = setup.CharsetMapper()
label2tensor = Lable2Tensor()
ssim = ssim_psnr.SSIM()
lpips_vgg = lpips.LPIPS(net="vgg")

class TextSR(base.TextBase):
    def loss_stablizing(self, loss_set, keep_proportion=0.7):
        # acsending
        sorted_val, sorted_ind = torch.sort(loss_set)
        batch_size = loss_set.shape[0]
        # print("batch_size:", loss_set, batch_size)
        loss_set[sorted_ind[int(keep_proportion * batch_size)]:] = 0.0
        return loss_set

    def model_inference(self, images_lr, images_hr, model_list, aster, i):
        ret_dict = {}
        ret_dict['duration']= 0.
        images_sr = []
        before = time.time()
        cascade_images, pos_prior = model_list[0](images_lr)
        after = time.time()
        # print("fps:", (after - before))
        ret_dict["duration"] += (after - before)
        images_sr.append(cascade_images)
        ret_dict["images_sr"] = images_sr

        return ret_dict



    def train(self):
        cfg = self.config.TRAIN
        train_dataset, train_loader = self.get_train_data()
        val_dataset_list, val_loader_list = self.get_val_data()
        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']
        model_list = [model]
        # ------- init text recognizer for eval here --------
        test_bible = {}
        if self.args.test_model == "CRNN":
            crnn, aster_info = self.CRNN_init()
            crnn.eval()
            test_bible["CRNN"] = {
                        'model': crnn,
                        'data_in_fn': self.parse_crnn_data,
                        'string_process': get_string_crnn
                    }
        elif self.args.test_model == "ASTER":
            aster_real, aster_real_info = self.Aster_init() # init ASTER model
            aster_info = aster_real_info
            test_bible["ASTER"] = {
                'model': aster_real,
                'data_in_fn': self.parse_aster_data,
                'string_process': get_string_aster
            }

        elif self.args.test_model == "MORAN":
            moran = self.MORAN_init()
            if isinstance(moran, torch.nn.DataParallel):
                moran.device_ids = [0]
            test_bible["MORAN"] = {
                'model': moran,
                'data_in_fn': self.parse_moran_data,
                'string_process': get_string_crnn
            }

        elif self.args.test_model == 'ABINet':
            abinet = self.ABINet_init()
            test_bible["ABINet"] = {
                'model': abinet,
                'data_in_fn': self.parse_abinet_data,
                'string_process': get_string_abinet
            }

        elif self.args.test_model == 'MATRN':
            matrn = self.MATRN_init()
            test_bible["MATRN"] = {
                'model': matrn,
                'data_in_fn': self.parse_abinet_data,
                'string_process': get_string_abinet
            }

        elif self.args.test_model == 'PARSeq':
            parseq = self.PARSeq_init()
            test_bible["PARSeq"] = {
                'model': parseq,
                'data_in_fn': self.parse_parseq_data,
                'string_process': get_string_parseq
            }

        optimizer_G = self.optimizer_init(model_list[0])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_G, 400,0.5)

        if not os.path.exists(cfg.ckpt_dir):
            os.makedirs(cfg.ckpt_dir)
        best_history_acc = dict(
            zip([val_loader_dir.split('/')[-1] for val_loader_dir in self.config.TRAIN.VAL.val_data_dir],
                [0] * len(val_loader_list)))
        best_model_acc = copy.deepcopy(best_history_acc)
        best_model_psnr = copy.deepcopy(best_history_acc)
        best_model_ssim = copy.deepcopy(best_history_acc)
        best_acc = 0
        converge_list = []

        for model in model_list:
            model.train()

        for epoch in range(cfg.epochs):
            # ----------------start training here -------------------
            for j, data in (enumerate(train_loader)):
                iters = len(train_loader) * epoch + j + 1
                for model in model_list:
                    for p in model.parameters():
                        p.requires_grad = True

                images_hr, _, images_lr, _, _, label_strs, label_vecs, weighted_mask, weighted_tics = data
                images_lr = images_lr.to(self.device) # [1,4,16,64]
                images_hr = images_hr.to(self.device) # [1,4,32,128]
                label_tensor = label2tensor.getlabletensor(label_strs)
                loss_img = 0.

                cascade_images, pos_prior = model_list[0](images_lr) # forward process
                loss, mse_loss, attention_loss, recognition_loss,finetune_loss = image_crit(cascade_images, images_hr, label_tensor,pos_prior)
                loss_img_each  =  loss.mean() * 100
                loss_img += loss_img_each
                loss_im = loss_img
                optimizer_G.zero_grad()
                loss_im.backward()
                for model in model_list:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer_G.step()


                if iters % cfg.displayInterval == 0:
                    logging.info('[{}]\t'
                                 'Epoch: [{}][{}/{}]\t'
                                 'total_loss {:.3f}\t'
                                 'mse_loss {:.3f}\t'
                                 'finetune_loss {:.3f}\t'
                                 'trans_attn {:.3f}\t'
                                 'trans_recg_loss {:.3f}\t'
                                 .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                         epoch, j + 1, len(train_loader),
                                         float(loss_im.data),
                                         mse_loss.mean().item() *100,
                                         finetune_loss.mean().item(),
                                         attention_loss.mean().item()*1000,
                                         recognition_loss.mean().item() * 100*0.0005,
                                         ))

                if iters % cfg.VAL.valInterval == 0:
                    logging.info('======================================================')
                    current_acc_dict = {}
                    psnr_dict={}
                    ssim_dict={}
                    for k, val_loader in enumerate(val_loader_list):
                        data_name = self.config.TRAIN.VAL.val_data_dir[k].split('/')[-1]
                        logging.info('evaling %s' % data_name)
                        for model in model_list:
                            model.eval()
                            for p in model.parameters():
                                p.requires_grad = False

                        metrics_dict = self.eval(
                            model_list,
                            val_loader,
                            image_crit,
                            iters,
                            [test_bible[self.args.test_model], None, None],
                            aster_info,
                            data_name
                        )
                        psnr_dict[data_name]=metrics_dict['psnr_avg']
                        ssim_dict[data_name]=metrics_dict['ssim_avg']

                        for model in model_list:
                            for p in model.parameters():
                                p.requires_grad = True
                            model.train()

                        converge_list.append({'iterator': iters,
                                              'acc': metrics_dict['accuracy'],
                                              'psnr': metrics_dict['psnr_avg'],
                                              'ssim': metrics_dict['ssim_avg']})
                        acc = metrics_dict['accuracy']
                        current_acc_dict[data_name] = float(acc)
                        if acc > best_history_acc[data_name]:
                            best_history_acc[data_name] = float(acc)
                            best_history_acc['epoch'] = epoch
                            logging.info('best_%s = %.2f%%*' % (data_name, best_history_acc[data_name] * 100))

                        else:
                            logging.info('best_%s = %.2f%%' % (data_name, best_history_acc[data_name] * 100))
                    if sum(current_acc_dict.values()) > best_acc:
                        best_acc = sum(current_acc_dict.values())
                        best_model_acc = current_acc_dict
                        best_model_acc['epoch'] = epoch
                        best_model_psnr[data_name] = metrics_dict['psnr_avg']
                        best_model_ssim[data_name] = metrics_dict['ssim_avg']
                        best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                        logging.info('saving best model')
                        logging.info('avg_acc {:.4f}%'.format(100*(current_acc_dict['easy']*1619
                                                              +current_acc_dict['medium']*1411
                                                              +current_acc_dict['hard']*1343)/(1343+1411+1619)))
                        logging.info('=============')
                        logging.info('bset psnr {:.4f}%'.format(1*(psnr_dict['easy']*1619
                                                              +psnr_dict['medium']*1411
                                                              +psnr_dict['hard']*1343)/(1343+1411+1619)))
                        logging.info('best ssim {:.4f}%'.format(1 * (ssim_dict['easy'] * 1619
                                                                + ssim_dict['medium'] * 1411
                                                                + ssim_dict['hard'] * 1343) / (1343 + 1411 + 1619)))
                        self.save_checkpoint(model_list, epoch, iters, best_history_acc, best_model_info, True, converge_list, recognizer=None)

                if iters % cfg.saveInterval == 0:
                    best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                    self.save_checkpoint(model_list, epoch, iters, best_history_acc, best_model_info, False, converge_list, recognizer=None)
            lr_scheduler.step()

    def eval(self, model_list, val_loader, image_crit, index, aster, aster_info, data_name=None):
        n_correct_lr = 0
        n_correct_hr = 0
        sum_images = 0
        metric_dict = {
                       'psnr_lr': [],
                       'ssim_lr': [],
                       'cnt_psnr_lr': [],
                       'cnt_ssim_lr': [],
                       'psnr': [],
                       'ssim': [],
                       'cnt_psnr': [],
                       'cnt_ssim': [],
                       'accuracy': 0.0,
                       'psnr_avg': 0.0,
                       'ssim_avg': 0.0,
                        'edis_LR': [],
                        'edis_SR': [],
                        'edis_HR': [],
                        'LPIPS_VGG_LR': [],
                        'LPIPS_VGG_SR': []
                       }

        counters = {0: 0}
        image_counter = 0
        rec_str = ""

        sr_infer_time = 0

        for i, data in (enumerate(val_loader)):
            images_hrraw, images_lrraw, images_HRy, images_lry, label_strs, label_vecs_gt = data
            # print("label_strs:", label_strs)
            images_lr = images_lrraw.to(self.device)
            images_hr = images_hrraw.to(self.device)


            val_batch_size = images_lr.shape[0]
            ret_dict = self.model_inference(images_lr, images_hr, model_list, aster, i)
            sr_infer_time += ret_dict["duration"]

            images_sr = ret_dict["images_sr"]

            # == 首先解析输入图像为文本识别器的输入形式 =======
            aster_dict_lr = aster[0]["data_in_fn"](images_lr[:, :3, :, :])
            aster_dict_hr = aster[0]["data_in_fn"](images_hr[:, :3, :, :])
            # ==== 之后是将解析后的图像用于文本识别  这里只对LR和HR进行识别====
            if self.args.test_model == "MORAN":
                # LR
                aster_output_lr = aster[0]["model"](
                    aster_dict_lr[0],
                    aster_dict_lr[1],
                    aster_dict_lr[2],
                    aster_dict_lr[3],
                    test=True,
                    debug=True
                )
                # HR
                aster_output_hr = aster[0]["model"](
                    aster_dict_hr[0],
                    aster_dict_hr[1],
                    aster_dict_hr[2],
                    aster_dict_hr[3],
                    test=True,
                    debug=True
                )
            else:
                aster_output_lr = aster[0]["model"](aster_dict_lr)
                aster_output_hr = aster[0]["model"](aster_dict_hr)
            # ============对SR进行图像解析和通过文本识别器 ==============
            if type(images_sr) == list:
                predict_result_sr = []
                image = images_sr[0]
                aster_dict_sr = aster[0]["data_in_fn"](image[:, :3, :, :])
                if self.args.test_model == "MORAN":
                    # aster_output_sr = aster[0]["model"](*aster_dict_sr)
                    aster_output_sr = aster[0]["model"](
                        aster_dict_sr[0],
                        aster_dict_sr[1],
                        aster_dict_sr[2],
                        aster_dict_sr[3],
                        test=True,
                        debug=True
                    )
                else:
                    aster_output_sr = aster[0]["model"](aster_dict_sr)
                # outputs_sr = aster_output_sr.permute(1, 0, 2).contiguous()
                if self.args.test_model == "CRNN":
                    predict_result_sr_ = aster[0]["string_process"](aster_output_sr)
                elif self.args.test_model == "ASTER":
                    predict_result_sr_, _ = aster[0]["string_process"](
                        aster_output_sr['output']['pred_rec'],
                        aster_dict_sr['rec_targets'],
                        dataset=aster_info
                    )
                elif self.args.test_model == "MORAN":
                    preds, preds_reverse = aster_output_sr[0]
                    _, preds = preds.max(1)
                    sim_preds = self.converter_moran.decode(preds.data, aster_dict_sr[1].data)
                    predict_result_sr_ = [pred.split('$')[0] for pred in sim_preds]

                elif self.args.test_model in ["ABINet",'MATRN']:
                    # 对识别字典做处理
                    predict_result_sr_ = aster[0]['string_process'](aster_output_sr, abi_charset)[0]

                elif self.args.test_model in ['PARSeq']:
                    # 对parseq的识别字典logits做处理
                    predict_result_sr_ = aster[0]['string_process'](aster_output_sr, parseq_tokenizer)

                predict_result_sr.append(predict_result_sr_)

                img_lr = torch.nn.functional.interpolate(images_lr, images_hr.shape[-2:], mode="bicubic")
                img_sr = torch.nn.functional.interpolate(images_sr[-1], images_hr.shape[-2:], mode="bicubic")

                metric_dict['psnr'].append(self.cal_psnr(img_sr[:, :3], images_hr[:, :3]))
                metric_dict['ssim'].append(self.cal_ssim(img_sr[:, :3], images_hr[:, :3]))

                metric_dict["LPIPS_VGG_SR"].append(lpips_vgg(img_sr[:, :3].cpu(), images_hr[:, :3].cpu()).data.numpy()[0].reshape(-1)[0])

                metric_dict['psnr_lr'].append(self.cal_psnr(img_lr[:, :3], images_hr[:, :3]))
                metric_dict['ssim_lr'].append(self.cal_ssim(img_lr[:, :3], images_hr[:, :3]))

                metric_dict["LPIPS_VGG_LR"].append(lpips_vgg(img_lr[:, :3].cpu(), images_hr[:, :3].cpu()).data.numpy()[0].reshape(-1)[0])

            else:
                aster_dict_sr = aster[0]["data_in_fn"](images_sr[:, :3, :, :])
                if self.args.test_model == "MORAN":
                    # aster_output_sr = aster[0]["model"](*aster_dict_sr)
                    aster_output_sr = aster[0]["model"](
                        aster_dict_sr[0],
                        aster_dict_sr[1],
                        aster_dict_sr[2],
                        aster_dict_sr[3],
                        test=True,
                        debug=True
                    )
                else:
                    aster_output_sr = aster[0]["model"](aster_dict_sr) # 对超分结果进行识别
                # outputs_sr = aster_output_sr.permute(1, 0, 2).contiguous()
                if self.args.test_model == "CRNN":
                    predict_result_sr = aster[0]["string_process"](aster_output_sr)
                elif self.args.test_model == "ASTER":
                    predict_result_sr, _ = aster[0]["string_process"](
                        aster_output_sr['output']['pred_rec'],
                        aster_dict_sr['rec_targets'],
                        dataset=aster_info
                    )
                elif self.args.test_model == "MORAN":
                    preds, preds_reverse = aster_output_sr[0]
                    _, preds = preds.max(1)
                    sim_preds = self.converter_moran.decode(preds.data, aster_dict_sr[1].data)
                    predict_result_sr = [pred.split('$')[0] for pred in sim_preds]
                elif self.args.test_model in ['ABINet','MATRN']:
                    # 对识别字典做处理
                    predict_result_sr = aster[0]['string_process'](aster_output_sr,abi_charset)[0]
                elif self.args.test_model in ['PARSeq']:
                    predict_result_sr = aster[0]['string_process'](aster_output_sr, parseq_tokenizer)

                img_lr = torch.nn.functional.interpolate(images_lr, images_sr.shape[-2:], mode="bicubic")
                metric_dict['psnr'].append(self.cal_psnr(images_sr[:, :3], images_hr[:, :3]))
                metric_dict['ssim'].append(self.cal_ssim(images_sr[:, :3], images_hr[:, :3]))

                metric_dict["LPIPS_VGG_SR"].append(lpips_vgg(images_sr[:, :3].cpu(), images_hr[:, :3].cpu()).data.numpy()[0].reshape(-1)[0])

                metric_dict['psnr_lr'].append(self.cal_psnr(img_lr[:, :3], images_hr[:, :3]))
                metric_dict['ssim_lr'].append(self.cal_ssim(img_lr[:, :3], images_hr[:, :3]))

                metric_dict["LPIPS_VGG_LR"].append(lpips_vgg(img_lr[:, :3].cpu(), images_hr[:, :3].cpu()).data.numpy()[0].reshape(-1)[0])

            if self.args.test_model == "CRNN":#之前是只对SR的识别结果进行的处理，这里将HR和LR同样进行处理
                predict_result_lr = aster[0]["string_process"](aster_output_lr)
                predict_result_hr = aster[0]["string_process"](aster_output_hr)
                # print(predict_result_hr)
            elif self.args.test_model == "ASTER":
                predict_result_lr, _ = aster[0]["string_process"](
                    aster_output_lr['output']['pred_rec'],
                    aster_dict_lr['rec_targets'],
                    dataset=aster_info
                )
                predict_result_hr, _ = aster[0]["string_process"](
                    aster_output_hr['output']['pred_rec'],
                    aster_dict_hr['rec_targets'],
                    dataset=aster_info
                )
            elif self.args.test_model == "MORAN":
                ### LR ###
                preds, preds_reverse = aster_output_lr[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, aster_dict_lr[1].data)
                predict_result_lr = [pred.split('$')[0] for pred in sim_preds]

                ### HR ###
                preds, preds_reverse = aster_output_hr[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, aster_dict_hr[1].data)
                predict_result_hr = [pred.split('$')[0] for pred in sim_preds]

            elif self.args.test_model in  ['ABINet','MATRN']:
                predict_result_lr = aster[0]["string_process"](aster_output_lr, abi_charset)[0]
                predict_result_hr = aster[0]["string_process"](aster_output_hr, abi_charset)[0]
            elif self.args.test_model in ['PARSeq']:
                predict_result_lr = aster[0]["string_process"](aster_output_lr, parseq_tokenizer)
                predict_result_hr = aster[0]["string_process"](aster_output_hr, parseq_tokenizer)
            filter_mode = 'lower'

            for batch_i in range(images_lr.shape[0]):
                label = label_strs[batch_i]
                image_counter += 1
                rec_str += str(image_counter) + ".jpg," + label + "\n"

                if str_filt(predict_result_sr[0][batch_i], filter_mode) == str_filt(label, filter_mode):
                    counters[0] += 1
                if str_filt(predict_result_lr[batch_i], filter_mode) == str_filt(label, filter_mode):
                    n_correct_lr += 1
                if str_filt(predict_result_hr[batch_i], filter_mode) == str_filt(label, filter_mode):
                    n_correct_hr += 1
            sum_images += val_batch_size
            torch.cuda.empty_cache()

        # 已经把整个测试集跑完
        psnr_avg = sum(metric_dict['psnr']) / (len(metric_dict['psnr']) + 1e-10)
        ssim_avg = sum(metric_dict['ssim']) / (len(metric_dict['psnr']) + 1e-10)

        psnr_avg_lr = sum(metric_dict['psnr_lr']) / (len(metric_dict['psnr_lr']) + 1e-10)
        ssim_avg_lr = sum(metric_dict['ssim_lr']) / (len(metric_dict['ssim_lr']) + 1e-10)
        lpips_vgg_lr = sum(metric_dict["LPIPS_VGG_LR"]) / (len(metric_dict['LPIPS_VGG_LR']) + 1e-10)
        lpips_vgg_sr = sum(metric_dict["LPIPS_VGG_SR"]) / (len(metric_dict['LPIPS_VGG_SR']) + 1e-10)

        logging.info('[{}]\t'
              'loss_rec {:.3f}| loss_im {:.3f}\t'
              'PSNR {:.2f} | SSIM {:.4f}\t'
              'LPIPS {:.4f}\t'
              .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      0, 0,
                      float(psnr_avg), float(ssim_avg), float(lpips_vgg_sr)))

        logging.info('[{}]\t'
              'PSNR_LR {:.2f} | SSIM_LR {:.4f}\t'
              'LPIPS_LR {:.4f}\t'
              .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      float(psnr_avg_lr), float(ssim_avg_lr), float(lpips_vgg_lr)))

        logging.info('save display images')

        accuracy = round(counters[0] / sum_images, 4)# loader已经跑完了，把全部的测试集图片和全部正确个数相除

        accuracy_lr = round(n_correct_lr / sum_images, 4)
        accuracy_hr = round(n_correct_hr / sum_images, 4)
        psnr_avg = round(psnr_avg.item(), 6)
        ssim_avg = round(ssim_avg.item(), 6)


        logging.info('sr_accuray_iter' + ': %.2f%%' % (accuracy * 100))


        logging.info('lr_accuray: %.2f%%' % (accuracy_lr * 100))
        logging.info('hr_accuray: %.2f%%' % (accuracy_hr * 100))
        metric_dict['accuracy'] = accuracy
        metric_dict['psnr_avg'] = psnr_avg
        metric_dict['ssim_avg'] = ssim_avg

        inference_time = sum_images / sr_infer_time
        logging.info("AVG inference:{}".format(inference_time))
        logging.info("sum_images:{}".format(sum_images))

        return metric_dict



    def test(self):
        total_acc = {}
        val_dataset_list, val_loader_list = self.get_val_data()
        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']
        model_list = [model]
        logging.info('Using text recognizer {}'.format(self.args.test_model))
        test_bible = {}
        aster_info = None
        if self.args.test_model == "CRNN":
            crnn, aster_info = self.CRNN_init()
            crnn.eval()
            test_bible["CRNN"] = {
                'model': crnn,
                'data_in_fn': self.parse_crnn_data,
                'string_process': get_string_crnn
            }
        elif self.args.test_model == "ASTER":
            aster_real, aster_real_info = self.Aster_init()  # init ASTER model
            aster_info = aster_real_info
            test_bible["ASTER"] = {
                'model': aster_real,
                'data_in_fn': self.parse_aster_data,
                'string_process': get_string_aster
            }

        elif self.args.test_model == "MORAN":
            moran = self.MORAN_init()
            aster_info = None
            if isinstance(moran, torch.nn.DataParallel):
                moran.device_ids = [0]
            test_bible["MORAN"] = {
                'model': moran,
                'data_in_fn': self.parse_moran_data,
                'string_process': get_string_crnn
            }
        elif self.args.test_model == 'ABINet':
            abinet = self.ABINet_init()
            test_bible["ABINet"] = {
                'model': abinet,
                'data_in_fn': self.parse_abinet_data,
                'string_process': get_string_abinet
            }
        elif self.args.test_model == 'MATRN':
            matrn = self.MATRN_init()
            test_bible["MATRN"] = {
                'model': matrn,
                'data_in_fn': self.parse_abinet_data,
                'string_process': get_string_abinet
            }
        elif self.args.test_model == 'PARSeq':
            parseq = self.PARSeq_init()
            test_bible["PARSeq"] = {
                'model': parseq,
                'data_in_fn': self.parse_parseq_data,
                'string_process': get_string_parseq
            }


        for k, val_loader in enumerate(val_loader_list):
            data_name = self.config.TRAIN.VAL.val_data_dir[k].split('/')[-1]
            logging.info('testing %s' % data_name)
            for model in model_list:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False

            metrics_dict = self.eval(
                model_list,
                val_loader,
                image_crit,
                0,
                [test_bible[self.args.test_model], None, None],
                aster_info,
                data_name
            )
            acc = float(metrics_dict['accuracy'])
            total_acc[data_name]=acc
            logging.info('best_%s = %.2f%%' % (data_name, acc * 100))
        logging.info('avg_acc(Easy,Medium,Hard) is {:.3f}'.format(100*(total_acc['easy']*1619
                                                              +total_acc['medium']*1411
                                                              +total_acc['hard']*1343)/(1343+1411+1619)))
        logging.info('Test with recognizer {} finished!'.format(self.args.test_model))

    def inference(self, image_path, recognizer_name=None):
        """
        Run inference on a single image file.

        Args:
            image_path (str): Path to the input PNG or JPG image.
            recognizer_name (str, optional): Name of the text recognizer to use
                (e.g., "CRNN", "ASTER", "ABINet"). If None, uses self.args.test_model.

        Returns:
            tuple: (predicted_text, sr_image_pil)
                - predicted_text (str): The recognized text string.
                - sr_image_pil (PIL.Image): The super-resolved image.
        """
        # 1. Set default recognizer if not provided
        if recognizer_name is None:
            recognizer_name = self.args.test_model
        logging.info(f'Running inference on {image_path} using recognizer {recognizer_name}')

        # 2. Load Super-Resolution (SR) model
        model_dict = self.generator_init()
        sr_model = model_dict['model']
        sr_model.eval()
        model_list = [sr_model] # Consistent with eval/test

        # 3. Load Text Recognizer model
        test_bible = {}
        aster_info = None
        if recognizer_name == "CRNN":
            crnn, aster_info = self.CRNN_init()
            crnn.eval()
            test_bible["CRNN"] = {
                'model': crnn,
                'data_in_fn': self.parse_crnn_data,
                'string_process': get_string_crnn
            }
        elif recognizer_name == "ASTER":
            aster_real, aster_real_info = self.Aster_init()
            aster_info = aster_real_info
            aster_real.eval()
            test_bible["ASTER"] = {
                'model': aster_real,
                'data_in_fn': self.parse_aster_data,
                'string_process': get_string_aster
            }
        elif recognizer_name == "MORAN":
            moran = self.MORAN_init()
            if isinstance(moran, torch.nn.DataParallel):
                moran.device_ids = [0]
            moran.eval()
            test_bible["MORAN"] = {
                'model': moran,
                'data_in_fn': self.parse_moran_data,
                'string_process': get_string_crnn
            }
        elif recognizer_name == 'ABINet':
            abinet = self.ABINet_init()
            abinet.eval()
            test_bible["ABINet"] = {
                'model': abinet,
                'data_in_fn': self.parse_abinet_data,
                'string_process': get_string_abinet
            }
        elif recognizer_name == 'MATRN':
            matrn = self.MATRN_init()
            matrn.eval()
            test_bible["MATRN"] = {
                'model': matrn,
                'data_in_fn': self.parse_abinet_data,
                'string_process': get_string_abinet
            }
        elif recognizer_name == 'PARSeq':
            parseq = self.PARSeq_init()
            parseq.eval()
            test_bible["PARSeq"] = {
                'model': parseq,
                'data_in_fn': self.parse_parseq_data,
                'string_process': get_string_parseq
            }
        else:
            raise ValueError(f"Unknown text recognizer: {recognizer_name}")

        recognizer_bundle = test_bible[recognizer_name]
        recognizer_model = recognizer_bundle['model']
        data_in_fn = recognizer_bundle['data_in_fn']
        string_process_fn = recognizer_bundle['string_process']

        lr_h, lr_w = 16, 64
        
        img_pil = Image.open(image_path).convert('RGBA')

        transform = transforms.Compose([
            transforms.Resize((lr_h, lr_w), Image.BICUBIC),
            transforms.ToTensor(),
        ])
        
        images_lr = transform(img_pil).unsqueeze(0).to(self.device)

        predicted_text = ""
        sr_image_pil = None

        with torch.no_grad():
            # Run Super-Resolution
            cascade_images, pos_prior = sr_model(images_lr)
            images_sr = cascade_images # This is the SR image tensor

            #  Run Text Recognition on the SR image
            images_sr_for_rec = images_sr[:, :3, :, :] 
            rec_input_dict = data_in_fn(images_sr_for_rec)

            # Run the recognizer
            if recognizer_name == "MORAN":
                aster_output_sr = recognizer_model(
                    rec_input_dict[0],
                    rec_input_dict[1],
                    rec_input_dict[2],
                    rec_input_dict[3],
                    test=True,
                    debug=True
                )
            else:
                aster_output_sr = recognizer_model(rec_input_dict)

            # Decode the output
            predict_result_sr_ = []
            if recognizer_name == "CRNN":
                predict_result_sr_ = string_process_fn(aster_output_sr)
            
            elif recognizer_name == "ASTER":
               
                predict_result_sr_, _ = string_process_fn(
                    aster_output_sr['output']['pred_rec'],
                    rec_input_dict['rec_targets'], 
                    dataset=aster_info
                )
            
            elif recognizer_name == "MORAN":
                preds, preds_reverse = aster_output_sr[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, rec_input_dict[1].data)
                predict_result_sr_ = [pred.split('$')[0] for pred in sim_preds]

            elif recognizer_name in ["ABINet", 'MATRN']:
                predict_result_sr_ = string_process_fn(aster_output_sr, abi_charset)[0]
            
            elif recognizer_name in ['PARSeq']:
                predict_result_sr_ = string_process_fn(aster_output_sr, parseq_tokenizer)
            
            if predict_result_sr_:
                predicted_text = predict_result_sr_[0] 

          
            sr_tensor_for_pil = images_sr.squeeze(0).cpu().detach().clamp(0, 1)
            pil_transform = transforms.ToPILImage()
            sr_image_pil = pil_transform(sr_tensor_for_pil)

            sr_image_pil.save("./super_resolved_output.png")

        logging.info(f'Inference complete. Predicted text: "{predicted_text}"')
        
        return predicted_text, sr_image_pil