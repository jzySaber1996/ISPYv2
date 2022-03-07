import json
import xlrd, xlwt
from utils import fy
import random
from importlib import import_module



def sheet_capture(sheetname):
    workbook = xlrd.open_workbook('../data/Issue-Threat (1).xls')
    sheet_data = workbook.sheet_by_name(sheetname)
    return sheet_data


def j_reformat(sheet_data, j_set, divide_type=0):
    model_name = "model"
    x = import_module('model.' + model_name)
    # x = import_module(model_name)
    config = x.Config()
    rows_num = sheet_data.nrows
    cols_value, cols_relation = sheet_data.col_values(1), sheet_data.col_values(2)
    cols_value, cols_relation = cols_value[1:], cols_relation[1:]
    count = 0
    COPY_K = 3
    for (dialogue, relation) in zip(cols_value, cols_relation):
        mark_issue_label, mark_solution_label = False, False
        cal_issue_label, cal_solution_label = 0, 0
        print(count)
        count += 1
        utterances = dialogue.split('\n')
        j_dialogue, j_utters = dict(), list()
        count_utter = 0
        for i_utter, utterance in enumerate(utterances):
            if utterance != '':
                count_utter += 1
                data_dict = dict()
                label = ''
                if divide_type == 0:
                    label = utterance[utterance.index('[') - 1]
                    data_dict['speaker'] = utterance[utterance.index('<') + 1:utterance.index('>')]
                    data_dict['sentence'] = utterance[utterance.index('>') + 2:]
                elif divide_type == 1:
                    label = utterance[utterance.index(' ') + 1]
                    data_dict['speaker'] = utterance[utterance.index('@') + 1:utterance.index(':')]
                    data_dict['sentence'] = utterance[utterance.index(':') + 1:]
                data_dict['issue_label'], data_dict['solution_label'] = 0, 0
                if label == '-':
                    if i_utter <= config.sentence_pad_size:
                        mark_issue_label = True
                        cal_issue_label += 1
                    data_dict['issue_label'] = 1
                if label == '+':
                    if i_utter <= config.sentence_pad_size:
                        mark_solution_label = True
                        cal_solution_label += 1
                    data_dict['solution_label'] = 1
                j_utters.append(data_dict)
        relation_matrix = [[1 if c_index == r_index else 0 for c_index in range(count_utter)]
                           for r_index in range(count_utter)]
        relation_set = relation.split('\n')
        for cor_relation in relation_set:
            if cor_relation != 'none' and cor_relation != '':
                x_cor = int(cor_relation[cor_relation.index('[') + 1:cor_relation.index(',')])
                y_cor = int(cor_relation[cor_relation.index(',') + 1:cor_relation.index(']')])
                relation_matrix[x_cor][y_cor] = 1
        j_dialogue['sentence_samples'] = j_utters
        j_dialogue['relations'] = relation_matrix
        # j_set.append(j_dialogue)
        if not mark_issue_label and not mark_solution_label:
            randdata_none = random.uniform(0, 1)
            if randdata_none > 0.7:
                j_set.append(j_dialogue)
        elif mark_issue_label and mark_solution_label:
            for i in range(COPY_K):
                j_temp = dict()
                j_temp['sentence_samples'] = []
                for i_sentence, j_sen in enumerate(j_dialogue['sentence_samples']):
                    temp_sen = dict()
                    temp_sen['speaker'] = j_sen['speaker']
                    temp_sen['sentence'] = ((j_sen['sentence'])) + 'e' + 'm' * (i * 5)
                    temp_sen['issue_label'] = j_sen['issue_label']
                    temp_sen['solution_label'] = j_sen['solution_label']
                    j_temp['sentence_samples'].append(temp_sen)
                j_temp['relations'] = j_dialogue['relations']
                j_set.append(j_temp)
        else:
            j_set.append(j_dialogue)
    return j_set


if __name__ == '__main__':
    # res = fy('这是一只狗')
    # print(res)
    project_dict = {'angular-others': 0, 'appium-others': 1, 'docker-others': 0, 'deeplearning4j-others': 1,
                    'typescript-others': 1}
    j_set = list()
    for project in project_dict.keys():
        sheet_data = sheet_capture(project)
        j_set = j_reformat(sheet_data, j_set, divide_type=project_dict[project])
    with open('../data/data_raw.json', 'w', encoding='utf8') as json_file:
        json.dump(j_set, json_file, ensure_ascii=False)
        print('Transfer JSON finished!')
