# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif, focal_loss
from pytorch_pretrained_bert.optimization import BertAdam
import test


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()  # start batch normalization and dropout
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels, feats, graphs) in enumerate(train_iter):
            torch.cuda.empty_cache()
            (issue_labels, solution_labels) = labels
            issue_label_tensor = torch.tensor(issue_labels)
            solution_label_tensor = torch.tensor(solution_labels)
            issue_outputs, solution_outputs = model(trains, graphs, feats)
            # model.zero_grad()
            # document_label_tensor = document_label_tensor.cuda()
            list_issue_outputs = issue_outputs.cpu().detach().numpy()
            list_issue_outputs = list_issue_outputs.flatten()
            issue_predicted = np.where(list_issue_outputs > 0.5, 1, 0)
            # issue_predicted = np.argmax(list_issue_outputs, axis=1)
            # issue_predicted = issue_predicted.reshape(len(issue_predicted), 1)
            issue_outputs, issue_label_tensor = torch.flatten(issue_outputs).cuda(), torch.flatten(
                issue_label_tensor).cuda()
            # issue_label_tensor = torch.flatten(issue_label_tensor).cuda()
            # loss_issue = F.cross_entropy(issue_outputs, issue_label_tensor).cuda()
            loss_issue = focal_loss(issue_outputs, issue_label_tensor, config.focal_alpha, config.focal_gamma,
                                    config.focal_logits, config.focal_reduce).cuda()

            list_solution_outputs = solution_outputs.cpu().detach().numpy()
            list_solution_outputs = list_solution_outputs.flatten()
            solution_predicted = np.where(list_solution_outputs > 0.5, 1, 0)
            # solution_predicted = np.argmax(list_solution_outputs, axis=1)
            # solution_predicted = solution_predicted.reshape(len(solution_predicted), 1)
            solution_outputs, solution_label_tensor = torch.flatten(solution_outputs).cuda(), torch.flatten(
                solution_label_tensor).cuda()
            # solution_label_tensor = torch.flatten(solution_label_tensor).cuda()
            # loss_solution = F.cross_entropy(solution_outputs, solution_label_tensor).cuda()
            loss_solution = focal_loss(solution_outputs, solution_label_tensor, config.focal_alpha, config.focal_gamma,
                                       config.focal_logits, config.focal_reduce).cuda()

            # total_loss = 0.5 * loss_issue + 0.5 * loss_solution
            total_loss = ((loss_issue ** 2 + loss_solution ** 2) / 2).sqrt()
            print("Iter: {},Issue Loss: {}, Solution Loss: {}, Total Loss: {}.".format(i, loss_issue, loss_solution,
                                                                                       total_loss))
            total_loss.backward()
            optimizer.step()
            if total_batch % 50 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                # true = torch.tensor(labels).cuda()
                # labels_output = np.array(labels).flatten()
                labels_output = np.concatenate(labels).flatten()
                outputs = np.concatenate((issue_predicted, solution_predicted), axis=0).flatten()
                labels_output = np.where(labels_output == 1, 1, 0)
                outputs = np.where(outputs == 1, 1, 0)
                # predic = torch.max(outputs.data, 1)[1].cuda()

                # train_acc = metrics.accuracy_score(labels_output, outputs)
                # train_pre = metrics.precision_score(labels_output, outputs)
                # train_rec = metrics.recall_score(labels_output, outputs)
                # train_f1 = metrics.f1_score(labels_output, outputs)
                # train_issue_pre = metrics.precision_score(labels_output[:int(len(labels_output) / 2)],
                #                                           outputs[:int(len(outputs) / 2)])
                # train_issue_rec = metrics.recall_score(labels_output[:int(len(labels_output) / 2)],
                #                                        outputs[:int(len(outputs) / 2)])
                # train_issue_f1 = metrics.f1_score(labels_output[:int(len(labels_output) / 2)],
                #                                        outputs[:int(len(outputs) / 2)])
                # train_solution_pre = metrics.precision_score(labels_output[int(len(labels_output) / 2):],
                #                                           outputs[int(len(outputs) / 2):])
                # train_solution_rec = metrics.recall_score(labels_output[int(len(labels_output) / 2):],
                #                                        outputs[int(len(outputs) / 2):])
                # train_solution_f1 = metrics.f1_score(labels_output[int(len(labels_output) / 2):],
                #                                   outputs[int(len(outputs) / 2):])

                dev_acc, dev_iss_pre, dev_iss_rec, dev_iss_f1, dev_sol_pre, dev_sol_rec, dev_sol_f1, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Dev Loss: {2:>5.2},  Dev Iss Pre: {3:>6.2%},  Dev Iss Rec: {4:>6.2%},  Dev Iss F1: {5:>6.2%},  ' \
                      'Dev Sol Pre: {6:>6.2%},  Dev Sol Rec: {7:>6.2%},  Dev Sol F1: {8:>6.2%},  Time: {9} {10}'
                print(msg.format(total_batch, total_loss.item(), dev_loss, dev_iss_pre, dev_iss_rec, dev_iss_f1, dev_sol_pre,
                                 dev_sol_rec, dev_sol_f1, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    # test(config, model, test_iter)


def evaluate(config, model, dev_iter):
    with torch.no_grad():  # Necessary!!!
        dev_acc, dev_loss = 0.0, 0.0
        dev_pre, dev_rec, dev_f1 = 0.0, 0.0, 0.0
        dev_iss_pre, dev_iss_rec, dev_iss_f1 = 0.0, 0.0, 0.0
        dev_sol_pre, dev_sol_rec, dev_sol_f1 = 0.0, 0.0, 0.0
        dev_batch = 0
        # dev_iter.batch_size = 16

        for j, (devs, dev_labels, dev_feats, dev_graphs) in enumerate(dev_iter):
            torch.cuda.empty_cache()
            # print("Valid Batches: {}".format(j))
            # if j > 3:
            #     break
            # labels_tensor = torch.tensor(dev_labels)
            # labels_tensor = labels_tensor.cpu()
            # document_label_tensor, sentence_label_tensor = labels_tensor.split([1, 10], dim=1)
            # document_label_tensor = torch.squeeze(document_label_tensor, dim=1)
            (issue_labels, solution_labels) = dev_labels
            issue_label_tensor = torch.tensor(issue_labels)
            solution_label_tensor = torch.tensor(solution_labels)
            issue_dev_outputs, solution_dev_outputs = model(devs, dev_graphs, dev_feats)
            model.zero_grad()
            # document_label_tensor = document_label_tensor.cuda()

            list_issue_outputs = issue_dev_outputs.cpu().detach().numpy()
            list_issue_outputs = list_issue_outputs.flatten()
            issue_predicted = np.where(list_issue_outputs > 0.5, 1, 0)
            # issue_predicted = np.argmax(list_issue_outputs, axis=1)
            # issue_predicted = issue_predicted.reshape(len(issue_predicted), 1)
            issue_dev_outputs, issue_label_tensor = torch.flatten(issue_dev_outputs).cuda(), torch.flatten(
                issue_label_tensor).cuda()
            # issue_label_tensor = torch.flatten(issue_label_tensor).cuda()
            # loss_issue = F.cross_entropy(issue_dev_outputs, issue_label_tensor).cuda()
            loss_issue = focal_loss(issue_dev_outputs, issue_label_tensor, config.focal_alpha, config.focal_gamma,
                                    config.focal_logits, config.focal_reduce).cuda()

            list_solution_outputs = solution_dev_outputs.cpu().detach().numpy()
            list_solution_outputs = list_solution_outputs.flatten()
            solution_predicted = np.where(list_solution_outputs > 0.5, 1, 0)
            # solution_predicted = np.argmax(list_solution_outputs, axis=1)
            # solution_predicted = solution_predicted.reshape(len(solution_predicted), 1)
            # solution_label_tensor = torch.flatten(solution_label_tensor).cuda()
            solution_dev_outputs, solution_label_tensor = torch.flatten(solution_dev_outputs).cuda(), torch.flatten(
                solution_label_tensor).cuda()
            # loss_solution = F.cross_entropy(solution_dev_outputs, solution_label_tensor).cuda()
            loss_solution = focal_loss(solution_dev_outputs, solution_label_tensor, config.focal_alpha,
                                       config.focal_gamma,
                                       config.focal_logits, config.focal_reduce).cuda()

            # each_loss = 0.5 * loss_issue + 0.5 * loss_solution
            each_loss = ((loss_issue ** 2 + loss_solution ** 2) / 2).sqrt()

            labels_output = np.array(dev_labels).flatten()
            outputs = np.concatenate((issue_predicted, solution_predicted), axis=0).flatten()
            labels_output = np.where(labels_output == 1, 1, 0)
            outputs = np.where(outputs == 1, 1, 0)

            each_accuracy = metrics.accuracy_score(labels_output, outputs)
            each_pre = metrics.precision_score(labels_output, outputs)
            each_rec = metrics.recall_score(labels_output, outputs)
            each_f1 = metrics.f1_score(labels_output, outputs)
            each_issue_pre = metrics.precision_score(labels_output[:int(len(labels_output) / 2)],
                                                      outputs[:int(len(outputs) / 2)])
            each_issue_rec = metrics.recall_score(labels_output[:int(len(labels_output) / 2)],
                                                   outputs[:int(len(outputs) / 2)])
            each_issue_f1 = metrics.f1_score(labels_output[:int(len(labels_output) / 2)],
                                              outputs[:int(len(outputs) / 2)])
            each_solution_pre = metrics.precision_score(labels_output[int(len(labels_output) / 2):],
                                                         outputs[int(len(outputs) / 2):])
            each_solution_rec = metrics.recall_score(labels_output[int(len(labels_output) / 2):],
                                                      outputs[int(len(outputs) / 2):])
            each_solution_f1 = metrics.f1_score(labels_output[int(len(labels_output) / 2):],
                                                 outputs[int(len(outputs) / 2):])
            dev_batch += 1
            dev_acc += each_accuracy
            dev_pre += each_pre
            dev_rec += each_rec
            dev_f1 += each_f1
            dev_iss_pre += each_issue_pre
            dev_iss_rec += each_issue_rec
            dev_iss_f1 += each_issue_f1
            dev_sol_pre += each_solution_pre
            dev_sol_rec += each_solution_rec
            dev_sol_f1 += each_solution_f1
            dev_loss += each_loss
        dev_acc = dev_acc / dev_batch
        dev_pre = dev_pre / dev_batch
        dev_rec = dev_rec / dev_batch
        dev_f1 = dev_f1 / dev_batch
        dev_iss_pre = dev_iss_pre / dev_batch
        dev_iss_rec = dev_iss_rec / dev_batch
        dev_iss_f1 = dev_iss_f1 / dev_batch
        dev_sol_pre = dev_sol_pre / dev_batch
        dev_sol_rec = dev_sol_rec / dev_batch
        dev_sol_f1 = dev_sol_f1 / dev_batch
        dev_loss = dev_loss / dev_batch
    return dev_acc, dev_iss_pre, dev_iss_rec, dev_iss_f1, dev_sol_pre, dev_sol_rec, dev_sol_f1, dev_loss
