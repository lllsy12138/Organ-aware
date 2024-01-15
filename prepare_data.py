import json
import re
import numpy.random as random
from tqdm import tqdm
random.seed(12345678)

# 清理mimic数据集原始报告（来自R2GEN）
def clean_report_mimic_cxr(report):
    report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
        .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
        .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
        .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
        .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                    .replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    report = ' . '.join(tokens) + ' .'
    return report

def clean_report_iu_xray(report):
    report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    report = ' . '.join(tokens) + ' .'
    return report

dict_ = []

# 提取每个区域的正常描述句子
def analyze_normal(name,data_path):
    ann = json.loads(open(data_path, 'r').read())
    list_for = [[], [], [], [], [], [], [], []]
    cnt_norab = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
    for item in tqdm(ann[name]):
        dict_sample = {}
        dict_sample['id'] = item['id']
        item['report'] = clean_report_mimic_cxr(item['report'])
        report = item['report']
        list_report = report.split('. ')

        len_list_report = len(list_report)
        list_report[len_list_report-1] = list_report[len_list_report-1][:len(list_report[len_list_report-1])-1]

        region_list = ["trachea","thoracic_aorta", "airspace","heart", "pleural", "bone", "lung"]
        string_list = ["", "", "", "", "", "", "", ""]
        tag_list = [0, 0, 0, 0, 0, 0, 0, 0]
        list_keyword = [["trachea"," tube"," picc ","catheter"," course ","pacemaker","port-a-cath",
    "device"," clip"," pacer ","stent ","wires ","foreign "," ij "," svc "," ett ","wiring"," aicd ",
                         "pacing ","hardware ","prosthetic ","mitral valvular replacement","wedge","wire-like"],
        ["thoracic aorta","aorta","aortic","vascula","aortic","pulmonary hypertension","vessels"],
        ["airspace","air space","air ","opacity"],
        ["heart","cardiomediastinal","cardiac","cardiomegaly","cardial",
    "mediastinum","mediastinal","hemidiaphragm","diaphragm","hernia","costophrenic","cardio"],
        ["pleural","pneumothorax","effusion","thickening","pneumothoraces"],
        ["bone","bony","osseous","skeletal","spine","spondylosis","osseus","fracture"," rib ",
    "vertebrae","degenerative "," ribs"],
        ["lung","pulmonary edema","pneumonia","granulomatous","granuloma","emphysema",
    "atelecta","edema","nodules","masses","hyperinflated ",
    "lobe","opacity ","opacit","opacification","perihilar","consolidation","hilar","hilum ",
         "hilus ","hila ","pulmonary "]]
        list_normalword = [["there are no ", "no radiopaque"],
                           ["normal", "stable", "midline", "unremarkable",
                            "no free", "no ", "not "],
                           ["normal", "stable", "midline", "unremarkable",
                            "no free", "unchanged", "no ", "not "],
                           ["normal", "stable", "unremarkable", "unchanged", "clear", "not"],
                           ["without", "no ", "negative", "clear"],
                           ["no ", "unchanged", "intact", "unremarkable"],
                           ["clear", "free of", "normal", "no focal", "unremarkable",
                            "no ", "unchanged", "well"]]

        list_ = ""
        #标注每一句话属于的模块类别
        for report_sentence in list_report:
            fg = 0
            cnt_use = 0
            for i in range(len(list_keyword)):
                for keyword in list_keyword[i]:
                    if report_sentence.find(keyword) != -1:
                        string_list[i] = string_list[i] + report_sentence + '. '
                        tag = 1
                        cnt_use += 1
                        for normalword in list_normalword[i]:
                            if report_sentence.find(normalword) != -1:
                                tag = 0
                                break
                        tag_list[i] |= tag
                        fg = 1

                        list_ += str(i)
                        break
                if fg == 1:
                    break

        dict_.append(list_)

        for i in range(len(tag_list)):
            if tag_list[i] == 0 and len(string_list[i]) > 10:
                #if i == len(tag_list)-2 and len(string_list[i]) < 70:
                #    continue

            #if len(string_list[i]) > 10:
                cnt_norab[i][tag_list[i]] += 1
                list_for[i].append(string_list[i])

    for i in range(len(list_for)):
        #print(len(list_for[i]))
        list_for[i] = list(set(list_for[i]))
        #print(len(list_for[i]))

    return list_for

# 生成每个区域的报告标注
def analyze(name,data_path):
    ann = json.loads(open(data_path, 'r').read())
    list = []
    cnt = 0
    cnt_all = [0, 0, 0, 0, 0, 0, 0, 0]
    cnt_allab = [0, 0, 0, 0, 0, 0, 0, 0]
    cnt_list = [0, 0, 0, 0, 0, 0, 0, 0]
    for item in tqdm(ann[name]):
        dict_sample = {}
        dict_sample['id'] = item['id']
        #dict_sample['study_id'] = item['study_id']
        #dict_sample['subject_id'] = item['subject_id']
        #dict_sample['tag'] = item['tag']
        dict_sample['report'] = item['report']
        item['report'] = clean_report_mimic_cxr(item['report'])

        dict_sample['image_path'] = item['image_path']
        original_report = item['report']


        list_report = original_report.split('. ')
        #list_report = list_report[:len(list_report)-1]
        len_list_report = len(list_report)
        list_report[len_list_report-1]=list_report[len_list_report-1][:len(list_report[len_list_report-1])-1]
        #print(list_report)
        region_list = ["trachea", "thoracic_aorta", "airspace", "heart", "pleural", "bone", "lung"]
        string_list = ["", "", "", "", "", "", "", ""]
        tag_list = [0, 0, 0, 0, 0, 0, 0, 0]
        list_keyword = [["trachea", " tube", " picc ", "catheter", " course ", "pacemaker", "port-a-cath",
                         "device", "clip", " pacer ", "stent ", "wires ", "foreign ", " ij ", " svc ", " ett ",
                         "wiring", " aicd ", "pacing ", "hardware ","prosthe","mitral valvular replacement",
                         "wedge","wire-like"],
                        ["thoracic aorta", "aorta", "aortic", "vascula", "aortic", "pulmonary hypertension","vessels"],
                        ["airspace", "air space", "air ", "opacity"],
                        ["heart", "cardiomediastinal", "cardiac", "cardiomegaly", "cardial",
                         "mediastinum", "mediastinal", "hemidiaphragm", "diaphragm", "hernia", "costophrenic","cardio"],
                        ["pleural", "pneumothorax", "effusion", "thickening", "pneumothoraces"],
                        ["bone", "bony", "osseous", "skeletal", "spine", "spondylosis", "osseus", "fracture", " rib ",
                         "vertebrae","degenerative "," ribs"],
                        ["lung", "pulmonary edema", "pneumonia", "granulomatous", "granuloma", "emphysema",
                         "atelecta", "edema", "nodules", "masses", "hyperinflated ",
                         "lobe", "opacity ", "opacit", "opacification", "perihilar", "consolidation", "hilar",
                         "hilum ","hilus ","hila ","pulmonary "]]
        list_normalword = [["there are no ", "no radiopaque"],
                           ["normal", "stable", "midline", "unremarkable",
                            "no free", "no ", "not "],
                           ["normal", "stable", "midline", "unremarkable",
                            "no free", "unchanged", "no ", "not "],
                           ["normal", "stable", "unremarkable", "unchanged", "clear", "not"],
                           ["without", "no ", "negative", "clear"],
                           ["no ", "unchanged", "intact", "unremarkable"],
                           ["clear", "free of", "normal", "no focal", "unremarkable",
                            "no ", "unchanged", "well"]]
        string_not_use = ""
        for report_sentence in list_report:
            count_use = 0
            fg = 0
            for i in range(len(list_keyword)):
                for keyword in list_keyword[i]:
                    if report_sentence.find(keyword) != -1:
                        string_list[i] = string_list[i] + report_sentence + '. '
                        tag = 1
                        count_use = 1
                        for normalword in list_normalword[i]:
                            if string_list[i].find(normalword) != -1:
                                tag = 0
                                break
                        cnt_all[i] += 1
                        cnt_allab[i] += tag
                        tag_list[i] |= tag
                        fg = 1
                        break
                if fg == 1:
                    break
            if count_use == 0:
                string_not_use = string_not_use + report_sentence + '. '
        #这里可以改进
        for i in range(len(string_list)-1):
            if string_list[i] == "":
                x = random.randint(0, len(list_for[i]))
                string_list[i] = list_for[i][x]
        if string_not_use == "":
            string_not_use = 'pa and lateral views of the chest provided . '
        dict_sample['others'] = string_not_use
        dict_sample['tag'] = tag_list
        for i in range(len(cnt_list)):
            cnt_list[i] = cnt_list[i] + tag_list[i]

        for i in range(len(region_list)):
            list_report = string_list[i].split(' ')
            cnt = max(cnt, len(list_report))
            #if len(list_report) > 30:
            #    print(string_list[i])
            #if string_list[i] == "":
            #    cnt_list[i] = cnt_list[i] + 1
            dict_sample[region_list[i]] = string_list[i]

        list.append(dict_sample)
    p = []
    for i in range(len(cnt_list)-1):
        p.append(cnt_list[i]/(len(ann[name])-cnt_list[i]))

    for i in range(len(cnt_list) - 1):
        print(cnt_list[i]/len(ann[name]))
    print(',')
    for i in range(len(cnt_list) - 1):
        print(cnt_all[i]/len(ann[name]))
    print(',')
    for i in range(len(cnt_list) - 1):
        print(cnt_allab[i]/len(ann[name]))
    #print(p)

    return list

data_path = '/mnt/lustrenew/lishiyu/datasets/report_gen/mimic_cxr/annotation.json'
list_for = analyze_normal('train',data_path)
new_dict = {}
list_train = analyze('train',data_path)
list_val = analyze('val',data_path)
list_test = analyze('test',data_path)
new_dict['train'] = list_train
new_dict['val'] = list_val
new_dict['test'] = list_test
with open("/mnt/lustrenew/lishiyu/datasets/report_gen/mimic_cxr/mimic_annotation.json", "w") as f:
    json.dump(new_dict, f)
