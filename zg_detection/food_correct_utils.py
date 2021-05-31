# -*- coding: utf-8 -*-
# @Time    : 2021/5/17
# @Author  : sunyihuan
# @File    : food_correct_utils.py

'''
检测结果后针对特殊食材、特殊处理
'''


def correct_bboxes(bboxes_pr):
    num_label = len(bboxes_pr)
    # 未检测食材
    if num_label == 0:
        return bboxes_pr

    # 检测到一个食材
    elif num_label == 1:
        if bboxes_pr[0][4] < 0.45:
            if int(bboxes_pr[0][5]) == 19:  # 低分花生米
                bboxes_pr[0][4] = 0.75
            elif int(bboxes_pr[0][5]) == 24:  # 低分整鸡
                bboxes_pr[0][4] = 0.75
            elif int(bboxes_pr[0][5]) == 37:  # 低分nofood
                bboxes_pr[0][4] = 0.85
            else:
                del bboxes_pr[0]

        # 低分shrimp、cabbage_heart、fish、strand、chips,crab针对性提升分值
        if int(bboxes_pr[0][5]) in [26, 2, 15, 27, 7, 10]:
            bboxes_pr[0][4] = bboxes_pr[0][4] * 1.3

        return bboxes_pr

    # 检测到多个食材
    else:
        new_bboxes_pr = []
        for i in range(len(bboxes_pr)):
            if bboxes_pr[i][4] >= 0.3:
                new_bboxes_pr.append(bboxes_pr[i])

        new_num_label = len(new_bboxes_pr)
        same_label = True
        for i in range(new_num_label):
            if i == (new_num_label - 1):
                break
            if new_bboxes_pr[i][5] == new_bboxes_pr[i + 1][5]:
                continue
            else:
                same_label = False

        sumProb = 0.
        # 多个食材，同一标签
        if same_label:
            if new_num_label > 3:
                new_bboxes_pr[0][4] = 0.98
            return new_bboxes_pr

        # 多个食材，非同一标签
        else:
            new_bboxes_pr = sorted(new_bboxes_pr, key=lambda x: x[4], reverse=True)
            problist = list(map(lambda x: x[4], new_bboxes_pr))
            labellist = list(map(lambda x: x[5], new_bboxes_pr))

            labeldict = {}
            for key in labellist:
                labeldict[key] = labeldict.get(key, 0) + 1
                # 按同种食材label数量降序排列
            s_labeldict = sorted(labeldict.items(), key=lambda x: x[1], reverse=True)

            n_name = len(s_labeldict)
            name1 = int(s_labeldict[0][0])
            num_name1 = s_labeldict[0][1]
            name2 = int(s_labeldict[1][0])
            num_name2 = s_labeldict[1][1]

            # 优先处理食材特例
            if n_name == 2:
                # 如果鸡翅中检测到了排骨，默认单一食材为鸡翅
                if (name1 == 5 and name2 == 23) or (name1 == 23 and name2 == 5):
                    for i in range(new_num_label):
                        new_bboxes_pr[i][5] = 5
                    return new_bboxes_pr

                # 如果鸡翅根检测到了排骨，默认单一食材为鸡翅根
                # if (name1 == 4 and name2 == 23) or (name1 == 23 and name2 == 4):
                #         for i in range(new_num_label):
                #             new_bboxes_pr[i][5] = 23
                #         return new_bboxes_pr

                # 如果菜心中检测到遮挡黑暗，默认为异常场景-遮挡黑暗
                if (name1 == 2 and name2 == 36) or (name1 == 36 and name2 == 2):
                    for i in range(new_num_label):
                        new_bboxes_pr[i][5] = 36
                    return new_bboxes_pr

            # 数量最多label对应的食材占比0.7以上
            if num_name1 / new_num_label > 0.7:
                name1_bboxes_pr = []
                for i in range(new_num_label):
                    if name1 == new_bboxes_pr[i][5]:
                        name1_bboxes_pr.append(new_bboxes_pr[i])

                # name1_bboxes_pr[0][4] = 0.95
                return name1_bboxes_pr
            # else:
            #     return new_bboxes_pr

            # 按各个label的probability降序排序
            else:
                for i in range(len(new_bboxes_pr)):
                    new_bboxes_pr[i][4] = new_bboxes_pr[i][4] * 0.9
                return new_bboxes_pr
