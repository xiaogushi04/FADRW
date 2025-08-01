from globals import *  # 导入全局变量和参数配置，如args_global等
from method.models import GraphSAINT  # 导入GraphSAINT图神经网络模型类
from method.minibatch import Minibatch  # 导入用于子图采样和批处理的Minibatch类
from utils import *  # 导入工具函数，如printf、to_numpy等
from metric import *  # 导入评估指标相关函数，如calc_f1、calc_metrics等
from method.utils import *  # 导入方法相关的工具函数

import warnings  # 导入warnings模块，用于控制警告信息
warnings.filterwarnings("ignore")  # 忽略所有警告信息，避免输出干扰
import torch  # 导入PyTorch深度学习框架
import time  # 导入time模块，用于计时
from method.sam import SAM  # 导入SAM优化器
#from scipy.interpolate import interp1d  # 插值函数

def evaluate_full_batch(model, minibatch, mode='val'):
    """
    Full batch evaluation: for validation and test sets only.
        When calculating the F1 score, we will mask the relevant root nodes
        (e.g., those belonging to the val / test sets).
    """
    loss,preds,labels = model.eval_step(*minibatch.one_batch(mode=mode))  # 获取一个批次的数据并进行模型评估，返回损失、预测和标签
    if mode == 'val':  # 如果是验证集
        node_target = [minibatch.node_val]  # 只评估验证集节点
    elif mode == 'test':  # 如果是测试集
        node_target = [minibatch.node_test]  # 只评估测试集节点
    else:  # 如果是同时评估验证集和测试集
        assert mode == 'valtest'  # 断言mode必须为'valtest'
        node_target = [minibatch.node_val, minibatch.node_test]  # 同时评估验证集和测试集节点
    f1mic, f1mac = [], []  # 初始化微平均和宏平均F1分数列表
    acc, f1m, prec, rec, f1 = [], [], [], [], []  # 初始化准确率、宏F1、精确率、召回率、F1分数列表
    for n in node_target:  # 遍历所有目标节点集
        f1_scores = calc_f1(to_numpy(labels[n]), to_numpy(preds[n]), model.sigmoid_loss)  # 计算F1分数
        accuracy, macro_f1, precision, recall, f1_score = calc_metrics(to_numpy(labels[n]), to_numpy(preds[n]), model.sigmoid_loss)  # 计算各项评估指标
        f1mic.append(f1_scores[0])  # 添加微平均F1
        f1mac.append(f1_scores[1])  # 添加宏平均F1
        acc.append(accuracy)  # 添加准确率
        prec.append(precision)  # 添加精确率
        rec.append(recall)  # 添加召回率
        f1.append(f1_score)  # 添加F1分数
        f1m.append(macro_f1)  # 添加宏F1
    f1mic = f1mic[0] if len(f1mic)==1 else f1mic  # 如果只有一个，取第一个，否则保留列表
    f1mac = f1mac[0] if len(f1mac)==1 else f1mac  # 同上
    acc = acc[0] if len(acc) == 1 else acc  # 同上
    prec = prec[0] if len(prec) ==1 else prec  # 同上
    rec = rec[0] if len(rec) == 1 else rec  # 同上
    f1 = f1[0] if len(f1) ==1 else f1  # 同上
    f1m = f1m[0] if len(f1m) == 1 else f1m  # 同上

    # loss is not very accurate in this case, since loss is also contributed by training nodes
    # on the other hand, for val / test, we mostly care about their accuracy only.
    # so the loss issue is not a problem.
    return loss, f1mic, f1mac, acc, prec, rec, f1, f1m  # 返回损失和各项评估指标

# 在这里修改了prepare函数，添加了SAM优化器
def prepare(train_data,train_params,arch_gcn):
    """
    Prepare some data structure and initialize model / minibatch handler before
    the actual iterative training taking place.
    """
    adj_full, adj_train, feat_full, class_arr,role = train_data  # 解包训练数据，包括全图邻接矩阵、训练子图邻接矩阵、特征、类别、角色划分
    adj_full = adj_full.astype(np.int32)  # 转为int32类型，节省内存
    adj_train = adj_train.astype(np.int32)  # 同上
    adj_full_norm = adj_norm(adj_full)  # 对全图邻接矩阵归一化
    tr_ids = role['tr']
    stego_tr_ids = [int(i) for i in tr_ids if class_arr[i, 1] == 1]
    normal_tr_ids = [int(i) for i in tr_ids if class_arr[i, 1] == 0]

    selected_stego = np.random.choice(stego_tr_ids, 190, replace=False)
    role['tr'] = list(selected_stego) + normal_tr_ids
    num_classes = class_arr.shape[1]  # 获取类别数
    cls_num_list = class_arr.sum(axis=0).tolist()  # 获取每个类别的样本数


    minibatch = Minibatch(adj_full_norm, adj_train, role, train_params)  # 初始化训练用采样器
    model = GraphSAINT(num_classes, arch_gcn, train_params, feat_full, class_arr)  # 初始化模型
    model.cls_num_list = cls_num_list
    model.loss_type = args_global.loss_type
    printf("TOTAL NUM OF PARAMS = {}".format(sum(p.numel() for p in model.parameters())), style="yellow")  # 打印模型参数总数numral()返回张量中元素总数，parameters()返回模型所有参数
    minibatch_eval=Minibatch(adj_full_norm, adj_train, role, train_params, cpu_eval=True)  # 初始化评估用采样器（CPU）
    model_eval=GraphSAINT(num_classes, arch_gcn, train_params, feat_full, class_arr, cpu_eval=True)  # 初始化评估用模型（CPU）
    print(model.cls_num_list)
    print(f"args_global.gpu: {args_global.gpu}")
    if args_global.gpu >= 0:  # 如果使用GPU
        model = model.cuda()  # 将模型转移到GPU
    return model, minibatch, minibatch_eval, model_eval  # 返回训练和评估用的模型与采样器


def train(train_phases, model, minibatch, minibatch_eval, model_eval, eval_val_every):
    if not args_global.cpu_eval:  # 如果不在CPU上评估
        minibatch_eval=minibatch  # 评估采样器等于训练采样器
    epoch_ph_start = 0  # 当前阶段起始epoch
    f1mic_best, ep_best = 0, -1  # 最佳微F1和对应epoch
    f1_best = 0  # 最佳F1分数
    # loss_best, ep_best = 100000, -1
    time_train = 0  # 训练总时间
    dir_saver = '{}/pytorch_models'.format(args_global.dir_log)  # 模型保存目录
    path_saver = '{}/pytorch_models/saved_model_{}.pkl'.format(args_global.dir_log, timestamp)  # 模型保存路径
    early_stop = 0  # 早停计数器
    for ip, phase in enumerate(train_phases):  # 遍历所有训练阶段 enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
        printf('START PHASE {:4d}'.format(ip),style='underline')  # 打印第几阶段信息
        minibatch.set_sampler(phase)  # 设置采样器参数
        num_batches = minibatch.num_training_batches()  # 获取训练批次数

        for e in range(epoch_ph_start, int(phase['end'])):  # 遍历当前阶段的所有epoch
            printf('Epoch {:4d}'.format(e),style='bold')  # 打印当前epoch
            model.current_epoch = e
            #adjust_learning_rate(model, e, args_global)
            #adjust_rho(model, e, args_global)
            minibatch.shuffle()  # 打乱采样顺序
            l_loss_tr, l_f1mic_tr, l_f1mac_tr = [], [], []  # 初始化训练损失和F1分数列表
            l_acc_tr, l_f1m_tr, l_prec_tr, l_rec_tr, l_f1_tr = [],[],[],[],[]  # 初始化其他指标列表
            time_train_ep = 0  # 当前epoch训练时间
            while not minibatch.end():  # 遍历所有批次
                t1 = time.time()  # 记录起始时间
                loss_train,preds_train,labels_train = model.train_step(*minibatch.one_batch(mode='train'),current_epoch=e)  # 训练一步，返回损失、预测、标签
                time_train_ep += time.time() - t1  # 累加训练时间
                if not minibatch.batch_num % args_global.eval_train_every:  # 每隔eval_train_every批次评估一次
                    f1_mic, f1_mac = calc_f1(to_numpy(labels_train),to_numpy(preds_train),model.sigmoid_loss)  # 计算F1分数
                    accuracy, macro_f1, precision, recall, f1_score = calc_metrics(to_numpy(labels_train),to_numpy(preds_train),model.sigmoid_loss)  # 计算其他指标
                    l_loss_tr.append(loss_train)  # 记录损失
                    l_f1mic_tr.append(f1_mic)  # 记录微F1
                    l_f1mac_tr.append(f1_mac)  # 记录宏F1
                    l_acc_tr.append(accuracy)  # 记录准确率
                    l_prec_tr.append(precision)  # 记录精确率
                    l_rec_tr.append(recall)  # 记录召回率
                    l_f1_tr.append(f1_score)  # 记录F1
                    l_f1m_tr.append(macro_f1)  # 记录宏F1
            if (e+1)%eval_val_every == 0:  # 每隔eval_val_every个epoch在验证集评估
                if args_global.cpu_eval:  # 如果在CPU上评估
                    torch.save(model.state_dict(),'tmp.pkl')  # 保存模型参数到临时文件
                    model_eval.load_state_dict(torch.load('tmp.pkl',map_location=lambda storage, loc: storage))  # 加载到评估模型
                else:
                    model_eval = model  # 直接用当前模型评估
                loss_val, f1mic_val, f1mac_val,acc_val, prec_val, rec_val, f1_val, f1m_val = evaluate_full_batch(model_eval, minibatch_eval, mode='val')  # 在验证集评估
                printf(' TRAIN (Ep avg): loss = {:.4f}\tmic = {:.4f}\tmac = {:.4f}\tacc = {:.4f}\tprec = {:.4f}\trec = {:.4f}\tf1 = {:.4f}\tf1m = {:.4f}\ttrain time = {:.4f} sec'\
                        .format(f_mean(l_loss_tr), f_mean(l_f1mic_tr), f_mean(l_f1mac_tr),f_mean(l_acc_tr),f_mean(l_prec_tr), f_mean(l_rec_tr), f_mean(l_f1_tr), f_mean(l_f1m_tr) , time_train_ep))  # 打印训练集平均指标
                printf(' VALIDATION:     loss = {:.4f}\tmic = {:.4f}\tmac = {:.4f}\tacc = {:.4f}\tprec = {:.4f}\trec = {:.4f}\tf1 = {:.4f}\tf1m = {:.4f}'\
                        .format(loss_val, f1mic_val, f1mac_val, acc_val,prec_val, rec_val, f1_val, f1m_val), style='yellow')  # 打印验证集指标
                # if f1mic_val > f1mic_best: # equals best accuracy
                #     f1mic_best, ep_best = f1mic_val, e
                if f1_val > f1_best:  # 如果当前F1更优
                    f1_best = f1_val  # 更新最佳F1
                    ep_best = e  # 记录最佳epoch
                    if not os.path.exists(dir_saver):  # 如果保存目录不存在
                        os.makedirs(dir_saver)  # 创建目录
                    printf('  Saving model ...', style='yellow')  # 打印保存信息
                    torch.save(model.state_dict(), path_saver)  # 保存模型参数
                    early_stop = 0  # 早停计数器归零
                else:
                    early_stop += 1  # 早停计数器加一
                if early_stop >= 20:  # 如果连续20次未提升
                    print("     Early Stop   ")  # 打印早停信息
                    break  # 跳出epoch循环
            time_train += time_train_ep  # 累加训练时间
        epoch_ph_start = int(phase['end'])  # 更新阶段起始epoch
    printf("Optimization Finished!", style="yellow")  # 打印训练结束
    if ep_best >= 0:  # 如果有最佳epoch
        if args_global.cpu_eval:  # 如果在CPU上评估
            model_eval.load_state_dict(torch.load(path_saver, map_location=lambda storage, loc: storage))  # 加载最佳模型到评估模型
        else:
            model.load_state_dict(torch.load(path_saver))  # 加载最佳模型到当前模型
            model_eval=model  # 评估模型等于当前模型
        printf('  Restoring model ...', style='yellow')  # 打印恢复模型信息
    loss, f1mic_both, f1mac_both, acc_both, prec_both, rec_both, f1_both, f1m_both = evaluate_full_batch(model_eval, minibatch_eval, mode='valtest')  # 在验证集和测试集做最终评估
    f1mic_val, f1mic_test = f1mic_both  # 分别获取验证集和测试集微F1
    f1mac_val, f1mac_test = f1mac_both  # 分别获取宏F1
    acc_val, acc_test = acc_both  # 分别获取准确率
    prec_val, prec_test = prec_both  # 分别获取精确率
    rec_val, rec_test = rec_both  # 分别获取召回率
    f1_val, f1_test = f1_both  # 分别获取F1
    f1m_val, f1m_test = f1m_both  # 分别获取宏F1

    printf("Full validation (Epoch {:4d}): \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}\tAcc = {:.4f}\tPrec = {:.4f}\tRec = {:.4f}\tF1 = {:.4f}\tF1m = {:.4f}"\
            .format(ep_best, f1mic_val, f1mac_val,acc_val,prec_val, rec_val, f1_val, f1m_val ), style='red')  # 打印验证集最终指标
    printf("Full test stats: \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}\tAcc = {:.4f}\tPrec = {:.4f}\tRec = {:.4f}\tF1 = {:.4f}\tF1m = {:.4f}"\
            .format(f1mic_test, f1mac_test, acc_test, prec_test, rec_test, f1_test, f1m_test), style='red')  # 打印测试集最终指标
    printf("Total training time: {:6.2f} sec".format(time_train), style='red')  # 打印总训练时间
    return f1mic_test, f1mac_test, acc_test, prec_test, rec_test, f1_test, f1m_test  # 返回测试集各项指标

def adjust_learning_rate(model, epoch, args):
    """调整学习率"""
    epoch = epoch + 1
    if epoch <= 5:  # 前5个epoch线性增加学习率
        lr = args_global.lr * epoch / 5
    elif epoch > 180:  # 最后阶段使用很小的学习率
        lr = args_global.lr * 0.0001
    elif epoch > 160:  # 倒数第二阶段
        lr = args_global.lr * 0.01
    else:  # 中间阶段使用原始学习率
        lr = args_global.lr
    model.lr = lr

def adjust_rho(model, epoch, args):
    """调整SAM优化器的rho参数"""
    epoch = epoch + 1
    if args_global.sam_rho_schedule == 'step':  # 阶段性调整
        if epoch <= 5:
            rho = 0.05
        elif epoch > 180:
            rho = 0.6
        elif epoch > 160:
            rho = 0.5 
        else:
            rho = 0.1
        model.rho = rho
    elif args_global.sam_rho_schedule == 'linear':  # 线性调整
        X = [1, 200]
        Y = [args_global.sam_min_rho, args_global.sam_max_rho]
        y_interp = interp1d(X, Y)
        rho = y_interp(epoch)
        model.rho = np.float16(rho)
    elif args_global.sam_rho_schedule == 'none':  # 固定rho值
        rho = args_global.sam_rho
        model.rho = rho

if __name__ == '__main__':  # 主程序入口
    log_dir = log_dir(args_global.train_config, args_global.data_prefix, git_branch, git_rev, timestamp)  # 生成日志目录
    result_json_path = log_dir + "result.json"  # 结果json文件路径
    train_params, train_phases, train_data, arch_gcn = parse_n_prepare(args_global)  # 解析参数并准备数据和模型结构
    if 'eval_val_every' not in train_params:  # 如果未指定验证间隔
        train_params['eval_val_every'] = EVAL_VAL_EVERY_EP  # 使用默认值
    f1mic_tests, f1mac_tests, acc_tests, prec_tests, rec_tests, f1_tests, f1m_tests = [], [], [], [], [], [], []  # 初始化多次实验结果列表
    for _ in range(args_global.repeat_time):  # 重复多次实验
        model, minibatch, minibatch_eval, model_eval = prepare(train_data, train_params, arch_gcn)  # 每次都重新初始化模型和采样器
        f1mic_test, f1mac_test, acc_test, prec_test, rec_test, f1_test, f1m_test = train(train_phases, model, minibatch, minibatch_eval, model_eval, train_params['eval_val_every'])  # 训练并评估
        f1mic_tests.append(f1mic_test)  # 记录微F1
        f1mac_tests.append(f1mac_test)  # 记录宏F1
        acc_tests.append(acc_test)  # 记录准确率
        prec_tests.append(prec_test)  # 记录精确率
        rec_tests.append(rec_test)  # 记录召回率
        f1_tests.append(f1_test)  # 记录F1
        f1m_tests.append(f1m_test)  # 记录宏F1

    json.dump({"repeat_time":args_global.repeat_time,"f1mac":f_mean(f1mac_tests), "f1mac_std":np.std(f1mac_tests),
               "f1mic":f_mean(f1mic_tests), "f1mic_std":np.std(f1mic_tests),
               "acc":f_mean(acc_tests),"acc_std":np.std(acc_tests),
               "prec":f_mean(prec_tests), "prec_std":np.std(prec_tests),
               "rec":f_mean(rec_tests),"rec_std":np.std(rec_tests),
               "f1":f_mean(f1_tests),"f1_std":np.std(f1_tests),
               "f1m":f_mean(f1m_tests), "f1m_std":np.std(f1m_tests)},
              open(result_json_path,"w", encoding="utf-8"))  # 保存所有实验的均值和标准差到json文件

