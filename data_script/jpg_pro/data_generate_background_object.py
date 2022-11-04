# -*- coding: utf-8 -*-
# @Time    : 2022/9/19
# @Author  : sunyihuan
# @File    : data_generate_background_object.py

'''
新采集图片中背景作为老数据目标物背景
'''

from xml.dom.minidom import *
from data_script.jpg_pro.jpg_boxes_fill import *
from tqdm import tqdm


def generate_xml(img_size, bboxes):
    (img_width, img_height, img_channel) = img_size
    # 创建一个文档对象
    doc = Document()

    # 创建一个根节点
    root = doc.createElement('annotation')

    # 根节点加入到tree
    doc.appendChild(root)

    # 创建二级节点
    fodler = doc.createElement('fodler')
    fodler.appendChild(doc.createTextNode('1'))  # 添加文本节点

    filename = doc.createElement('filename')
    filename.appendChild(doc.createTextNode('xxxx.jpg'))  # 添加文本节点

    path = doc.createElement('path')
    path.appendChild(doc.createTextNode('./xxxx.jpg'))  # 添加文本节点

    source = doc.createElement('source')
    name = doc.createElement('database')
    name.appendChild(doc.createTextNode('Unknown'))  # 添加文本节点
    source.appendChild(name)  # 添加文本节点

    size = doc.createElement('size')
    width = doc.createElement('width')
    width.appendChild(doc.createTextNode(str(img_width)))  # 添加图片width
    height = doc.createElement('height')
    height.appendChild(doc.createTextNode(str(img_height)))  # 添加图片height
    channel = doc.createElement('depth')
    channel.appendChild(doc.createTextNode(str(img_channel)))  # 添加图片channel
    size.appendChild(height)
    size.appendChild(width)
    size.appendChild(channel)

    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode('0'))
    root.appendChild(fodler)  # fodler加入到根节点
    root.appendChild(filename)  # filename加入到根节点
    root.appendChild(path)  # path加入到根节点
    root.appendChild(source)  # source加入到根节点
    root.appendChild(size)  # source加入到根节点
    root.appendChild(segmented)  # segmented加入到根节点

    for i in range(len(bboxes)):
        object = doc.createElement('object')
        name = doc.createElement('name')
        print(str(bboxes[i][4]))
        name.appendChild(doc.createTextNode(str(bboxes[i][4])))
        object.appendChild(name)
        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode("Unspecified"))
        object.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode("0"))
        object.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode("0"))
        object.appendChild(difficult)
        bndbox = doc.createElement('bndbox')
        xmin = doc.createElement("xmin")
        xmin.appendChild(doc.createTextNode(str(bboxes[i][0])))
        bndbox.appendChild(xmin)
        ymin = doc.createElement("ymin")
        ymin.appendChild(doc.createTextNode(str(bboxes[i][1])))
        bndbox.appendChild(ymin)
        xmax = doc.createElement("xmax")
        xmax.appendChild(doc.createTextNode(str(bboxes[i][2])))
        bndbox.appendChild(xmax)
        ymax = doc.createElement("ymax")
        ymax.appendChild(doc.createTextNode(str(bboxes[i][3])))
        bndbox.appendChild(ymax)
        object.appendChild(bndbox)

        root.appendChild(object)  # object加入到根节点
    return doc


def fii(o_w, o_h, image0, n_b, old_object, typ=2):
    w = o_w
    h = o_h
    b = n_b
    # 方法1：按原目标框比例压缩
    if typ == 0:
        old_object_img = np.array(cv2.resize(old_object[0], (w, h)))
        image0[int(b[1]):int(b[3]), int(b[0]):int(b[2]), 0] = old_object_img[:, :, 0]
        image0[int(b[1]):int(b[3]), int(b[0]):int(b[2]), 1] = old_object_img[:, :, 1]
        image0[int(b[1]):int(b[3]), int(b[0]):int(b[2]), 2] = old_object_img[:, :, 2]

    # 方法2：先将目标物填充为灰色，再左上角为基点填充目标框
    if typ == 1:
        fi_array = 123 * np.ones((h, w, 3))
        image0[int(b[1]):int(b[3]), int(b[0]):int(b[2]), 0] = fi_array[:, :, 0]
        image0[int(b[1]):int(b[3]), int(b[0]):int(b[2]), 1] = fi_array[:, :, 1]
        image0[int(b[1]):int(b[3]), int(b[0]):int(b[2]), 2] = fi_array[:, :, 2]
        # #
        old_object_img = old_object[0]
        old_ob_size = old_object_img.shape
        si = min(h / old_ob_size[0], w / old_ob_size[1])
        shape_si_h = int(old_ob_size[0] * si)
        shape_si_w = int(old_ob_size[1] * si)

        old_object_img0 = np.array(cv2.resize(old_object_img, (shape_si_w, shape_si_h)))

        image0[int(b[1]):int(b[1]) + shape_si_h, int(b[0]):int(b[0]) + shape_si_w, 0] = old_object_img0[:, :, 0]
        image0[int(b[1]):int(b[1]) + shape_si_h, int(b[0]):int(b[0]) + shape_si_w, 1] = old_object_img0[:, :, 1]
        image0[int(b[1]):int(b[1]) + shape_si_h, int(b[0]):int(b[0]) + shape_si_w, 2] = old_object_img0[:, :, 2]

    # 方法3：先将目标物填充为灰色，再左下角为基点填充目标框
    if typ == 2:
        fi_array = 123 * np.ones((h, w, 3))

        image0[int(b[1]):int(b[3]), int(b[0]):int(b[2]), 0] = fi_array[:, :, 0]
        image0[int(b[1]):int(b[3]), int(b[0]):int(b[2]), 1] = fi_array[:, :, 1]
        image0[int(b[1]):int(b[3]), int(b[0]):int(b[2]), 2] = fi_array[:, :, 2]

        old_object_img = old_object[0]
        old_ob_size = old_object_img.shape
        si = min(h / old_ob_size[0], w / old_ob_size[1])
        shape_si_h = int(old_ob_size[0] * si)
        shape_si_w = int(old_ob_size[1] * si)

        old_object_img0 = np.array(cv2.resize(old_object_img, (shape_si_w, shape_si_h)))

        image0[(int(b[3]) - shape_si_h):int(b[3]), int(b[0]):int(b[0]) + shape_si_w, 0] = old_object_img0[:, :, 0]
        image0[(int(b[3]) - shape_si_h):int(b[3]), int(b[0]):int(b[0]) + shape_si_w, 1] = old_object_img0[:, :, 1]
        image0[(int(b[3]) - shape_si_h):int(b[3]), int(b[0]):int(b[0]) + shape_si_w, 2] = old_object_img0[:, :, 2]
    return image0


def fill_object(img_new, new_xml_path, img, xml_path, img_save, xml_save_name):
    # 读取新图片
    image0 = cv2.imread(img_new)
    image0 = np.array(image0)

    # 获取新图片中的目标框
    new_bboxes = get_bboxes(new_xml_path)

    # 目标框按面积排序
    ss = {}
    for j, nn in enumerate(new_bboxes):
        b_area = (int(nn[2]) - int(nn[0])) * (int(nn[3]) - int(nn[1]))
        ss[j] = b_area
    ss = sorted(ss.items(), key=lambda x: x[1], reverse=True)
    bb_area_sort = []
    for s in ss:
        i = list(s)[0]
        bb_area_sort.append(new_bboxes[i])

    new_bboxes = bb_area_sort

    # 读取老图片
    image1 = cv2.imread(img)
    image1 = np.array(image1)

    # 获取老图片中的目标框
    old_bboxes = get_bboxes(xml_path)
    bb_imgs = []
    for bb in old_bboxes:
        bb_imgs.append([image1[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])], bb[4]])

    # 针对每个新目标框，填充老目标物
    generate_bboxes = []
    for b in new_bboxes:
        old_object = choice(bb_imgs)  # 随机选取一个目标物
        object_cls = old_object[1]

        w, h = int(b[2] - b[0]), int(b[3] - b[1])
        if (w * h) / (int(b[2]) - int(b[0])) * (int(b[3]) - int(b[1])) > 0.5 and (w * h) / (int(b[2]) - int(b[0])) * (
                int(b[3]) - int(b[1])) < 2:
            image0 = fii(w, h, image0, b, old_object, typ=2)
        else:
            old_object = choice(bb_imgs)  # 随机选取一个目标物
            object_cls = old_object[1]
            w, h = int(b[2] - b[0]), int(b[3] - b[1])
            image0 = fii(w, h, image0, b, old_object, typ=2)

        b[4] = object_cls
        generate_bboxes.append(b)

    # 生成图片保存
    cv2.imwrite(img_save, image0)

    # 存成xml文件
    img_height, img_width, img_channel = image0.shape
    img_size = (img_height, img_width, img_channel)
    doc = generate_xml(img_size, generate_bboxes)

    # 保存xml文件
    fp = open(xml_save_name, 'w', encoding='utf-8')
    doc.writexml(fp, indent='', addindent='\t', newl='\n', encoding='utf-8')
    fp.close()


if __name__ == "__main__":
    new_data_dir = "F:\\ObjectRecognition\\data\\all_model_data\\imageset2022\\2022imageset"
    new_image_dir = new_data_dir + "\\JPGImages"
    new_xml_dir = new_data_dir + "\\Annotations"

    old_data_dir = "F:\\ObjectRecognition\\data\\all_model_data\\imageset2021"
    old_list = [o.split(".")[0] for o in os.listdir(old_data_dir + "/JPGImages")]

    save_dir = "F:\\ObjectRecognition\\data\\all_model_data\\ge_data"

    for ii in tqdm(os.listdir(new_image_dir)):
        i_i = ii.split(".")[0]
        new_image_path = new_image_dir + "\\" + i_i + ".jpg"
        new_xml_path = new_xml_dir + "\\" + i_i + ".xml"
        if os.path.exists(new_image_path) and os.path.exists(new_xml_path):
            old_d = choice(old_list)
            old_img_path = old_data_dir + "\\JPGImages\\" + old_d + ".jpg"
            old_xml_path = old_data_dir + "\\Annotations\\" + old_d + ".xml"

            img_save = save_dir + "\\JPGImages1\\" + i_i + "_" + old_d + ".jpg"
            xml_save_name = save_dir + "\\Annotations1\\" + i_i + "_" + old_d + ".xml"

            fill_object(new_image_path, new_xml_path, old_img_path, old_xml_path, img_save, xml_save_name)
