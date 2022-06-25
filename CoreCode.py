import enum
import sys
from cv2 import sqrt
import numpy as np
import cv2
import random
'''LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL                      
Note 
(1)block : 40*80  (pixel size )
QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ'''

'''LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL                      
【函数名】
【功  能】set
QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ'''
# the threshold for smallest area of a instance
threshold_area = 700

# the threshold for smallest occlude area to define if two instance are occluded
threshold_occlude = 45

# the threshold for most gap of one layer in a graph
threshold_same_layer_gap = 35

# the threshold for gap between two layers
threshold_two_layer_gap = 135


'''LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL                      
【函数名】
【功  能】load data
QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ'''
print("[....................................load data ....................................]")
# test image
test_img_name = 910
print("test_img_name:", test_img_name)

# instance data
instance = np.load("JYImage2Graph/data/bcnet/instance"+str(test_img_name)+".npy",allow_pickle=True)

# box data
ins_boxes = np.load("JYImage2Graph/data/bcnet/ins_boxes"+str(test_img_name)+".npy",allow_pickle=True)
print(len(instance),len(ins_boxes))

# input RGB image
img_origin = cv2.imread("test_img/"+str(test_img_name)+".jpg")

# background
img = cv2.imread("JYImage2Graph/image/PictureWhite.png")
img = cv2.resize(img, (640, 480))

graph_bg = img.copy()
cv2.rectangle(graph_bg, (638,0),(640,480),(71,123,176))
img_draw = img.copy()

# input D image
depth_image = np.load("JYImage2Graph/data/depth/"+str(test_img_name)+".npy")
depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.4), cv2.COLORMAP_JET)
print("image:",img.shape )
print("depth_image:", depth_image.shape)
# cv2.imshow('RealSense', depth_colormap)
# cv2.waitKey(0)

'''LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL                      
【函数名】
【功  能】create color
QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ'''
print("[....................................create color....................................]")

# make a batch of RGB colors
colors = []
Rs = random.sample(range(10, 250), 200)
Gs = random.sample(range(10, 250), 200)
Bs = random.sample(range(10, 250), 200)
for i in range(0,200): 
    grayLevel =  Rs[i] * 0.299 + Gs[i] * 0.587 + Bs[i] * 0.114
    if grayLevel >= 100: colors.append((int(Bs[i]), int(Gs[i]), int(Rs[i])))



# draw each instance and then put the points to "ins_points[]"
num_instance = len(instance)
print("num_instance:",num_instance)
ins_points = [[]for _ in range(num_instance)]
for i, mask in enumerate(instance):
    pre_block_bg = cv2.imread("JYImage2Graph/image/PictureWhite.png")
    pre_block_bg = cv2.resize(pre_block_bg, (640, 480))
    cv2.fillConvexPoly(pre_block_bg, mask, colors[i])
    for x in range(0, 480):
        for y in range(0, 640): 
            bg_color = (pre_block_bg[x,y,0], pre_block_bg[x,y,1], pre_block_bg[x,y,2])
            if bg_color != (255,255,255): ins_points[i].append((x,y))
    # cv2.imshow("img",pre_block_bg)
    # cv2.waitKey()
    cv2.fillConvexPoly(img_draw, mask, colors[i])

# print(len(ins_points[0]), len(ins_points[1]), len(ins_points[2]))
# cv2.imshow("img",img_draw)
# cv2.waitKey()

'''LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL                      
【函数名】
【功  能】detect color
QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ'''
print("[....................................detect color ....................................]")
ins_area = [[] for _ in range(num_instance)]
for x in range(0, 480):
    for y in range(0, 640):
        bgr_now = (img_draw[x,y,0], img_draw[x,y,1], img_draw[x,y,2])
        for i in range(0, num_instance):
            bgr_drawed = colors[i]
            if bgr_drawed == bgr_now: ins_area[i].append((x,y))

ins_empty_area = []
ins_area_final = []
ins_points_final = []
for i, area in enumerate(ins_area):
    print(i, "'s area:", len(area))
    if len(area) == 0: ins_empty_area.append(i)
    if len(area) > threshold_area: 
        ins_area_final.append(area)
        ins_points_final.append(ins_points[i])

num_instance = len(ins_area_final)                                                                                                                                                         #确定最终的物体
print("num_instance_final: ", num_instance)
print("ins_empty_area: ",ins_empty_area)
'''LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL                      
【函数名】
【功  能】find occ points
QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ'''
print("[....................................find occ points ....................................]")
occ_points = []
occ_of_ins = {}
for x in range(0, 480):
    for y in range(0, 640):
        occ_of_ins[str((x,y))] = []

for x in range(0, 480):
    for y in range(0, 640):
        if img_draw[x, y][0] != 255 and img_draw[x, y][1] != 255 and img_draw[x, y][2] != 255:
            for i, points in enumerate(ins_points_final):
                    if (x, y) in points:
                        occ_of_ins[str((x,y))].append(i)

connected_matrix = np.zeros((num_instance, num_instance), dtype= int)
for x in range(0, 480):
    for y in range(0, 640):
        occ_of_ins_now = occ_of_ins[str((x,y))]
        if len(occ_of_ins_now) > 1: 
            occ_points.append((x,y))
            for i, val1 in enumerate(occ_of_ins_now):
                for val2 in occ_of_ins_now[i+1:]:
                    connected_matrix[val1, val2] += 1
                    connected_matrix[val2, val1] += 1

print("connected_matrix:", connected_matrix)
print("len(occ_points):", len(occ_points))

'''LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL                      
【函数名】
【功  能】find occ relation
QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ'''
print("[....................................find occ relation ....................................]")
occ_relation = []
for row in range(num_instance):
    for col in range(num_instance):
        if connected_matrix[row, col] > threshold_occlude and col >= row: 
            occ_relation.append((row, col))

print("occ_relation:", occ_relation)

'''LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL                      
【函数名】
【功  能】draw color & detect depth
QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ'''
print("[....................................detect depth ....................................]")
ins_depth = [[] for _ in range(num_instance)]
for i, area in enumerate(ins_area_final):
    for point in area:
        if point not in occ_points:
            cv2.circle(img, (point[1], point[0]), 1, colors[i])
            ins_depth[i].append(depth_image[point[0]-6 , point[1] - 26 ])
# cv2.imshow("img",img)
# cv2.waitKey(0)

ins_depth_final = []
for i, depth in enumerate(ins_depth):
    depth_median = np.median(depth)
    ins_depth_final.append(depth_median)
    print(i, "'s depth:", depth_median)

# cv2.imshow("img",img)
# cv2.waitKey(0)

'''LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL                      
【函数名】
【功  能】detect center of boxes
QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ'''
print("[....................................detect center ....................................]")
ins_boxes = ins_boxes.tolist()

for i in ins_empty_area:
    ins_boxes.insert(i, [0,0,0,0])

ins_center = []
for i, box in enumerate(ins_boxes):
    y, x = int((box[0] + box[2])/2), int((box[1] + box[3])/2)
    ins_center.append((x, y))
    print(i, "'s center:", (x,y))

ins_center_final = []
for i, area in enumerate(ins_area_final):
    if len(area) > threshold_area: 
        ins_center_final.append(ins_center[i])

for i, center in enumerate(ins_center_final):
    text_color = (colors[i][0]-40, colors[i][1]-40, colors[i][2]-40)
    cv2.circle(img, (center[1], center[0]), 5, text_color, thickness= 4)
    cv2.putText(img, str(i), (center[1], center[0]), cv2.FONT_ITALIC, 0.65,(0,0,0), 2)

'''LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL                      
【函数名】
【功  能】put rgbd and center to block
QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ'''
print("[....................................create block ....................................]")
blocks_list = []
for i in range(num_instance):
    block_new = {"id":i,"xy":ins_center_final[i],"depth":ins_depth_final[i],"layer":0,"BGR":colors[i]}
    blocks_list.append(block_new)
    # print("block: ",block_new)

'''LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL                      
【函数名】layer_gap = 7
【功  能】put layer to block
QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ'''
print("[....................................put layer....................................]")
depth_for_layer = ins_depth_final[:]

layer_gap = threshold_same_layer_gap
layer_now = 1
while depth_for_layer:
    depth_layer_min = np.min(depth_for_layer)

    # get to the next layer
    if layer_now != 1 \
        and depth_layer_min - depth_layer_min_last > threshold_two_layer_gap:
          layer_now += 1          
             
    # get to now layer                                                                                    是否跨层
    depth_layer_max = depth_layer_min + layer_gap
    print('depth_min,max:', depth_layer_min, depth_layer_max)
    print("layer:",layer_now)
    for i, depth in enumerate(ins_depth_final):
        if depth <= depth_layer_max and depth >= depth_layer_min:
            blocks_list[i]["layer"] = layer_now
            print("block:",i,"depth:",depth)
            for d in depth_for_layer:
                if d == depth:
                    depth_for_layer.remove(d)

    print("remain",depth_for_layer)
    layer_now += 1
    print("|||")
    depth_layer_min_last = depth_layer_min

layer_max = layer_now - 1  #最大层数

for block in blocks_list:
    print(block)

'''LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL                      
【函数名】
【功  能】create graph
QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ'''
print("[....................................create graph....................................]")
block_per_layer = [0 for _ in range(5)]
block_of_layer = [[] for _ in range(5)]
block_graph_pos = [0 for _ in range(num_instance)]

for i, block in enumerate(blocks_list):
    block_layer = block["layer"]
    block_of_layer[block_layer - 1].append(block)
    layer_block_num = block_per_layer[block_layer - 1]
    pos = (80*(layer_block_num+1) , 80*block_layer)
    block_graph_pos[i] = pos
    cv2.circle(graph_bg, pos, 14, block["BGR"], -1)
    cv2.putText(graph_bg, str(block["id"]), (80*(layer_block_num+1)-6 , 80*block_layer+6), cv2.FONT_ITALIC, 0.6,[255,255,255], 2)
    block_per_layer[block_layer - 1] += 1

print("block_num_per_layer:",block_per_layer)
for i,blocks in enumerate(block_of_layer):
    if blocks:
        print("layer:",i+1)
        for block in blocks:
            print(block)
        print("|||")

occ_relation_final = []
for relation in occ_relation:                                                                                                                                                                      #建立边
    block_1, block_2 = blocks_list[relation[0]],  blocks_list[relation[1]]
    edge_val = abs(block_1["layer"] - block_2["layer"])
    cirle_flag = 0
    if edge_val > 1:
        occ_relation_copy = occ_relation[:]
        val1, val2 = relation[0], relation[1]
        print(val1, val2)
        occ_relation_copy.remove((val1, val2))
        print(occ_relation_copy)
        val1_other = []
        val2_other = []
        for rela in occ_relation_copy:
            if val1 in rela:
                val1_index = rela.index(val1)
                if val1_index == 0: val1_other.append(rela[1])
                else: val1_other.append(rela[0])
            if val2 in rela:
                val2_index = rela.index(val2)
                if val2_index == 0: val2_other.append(rela[1])
                else: val2_other.append(rela[0])
        print(val2_other)
        for val in val1_other:
            if val in val2_other:
                cirle_flag = 1
                print("circle!")
    if edge_val == 0: cirle_flag = 1
    if not cirle_flag:
        occ_relation_final.append([[relation[0], relation[1]],edge_val])
        cv2.line(
            graph_bg, 
            block_graph_pos[relation[0]],block_graph_pos[relation[1]],
            (150,150,150),1,cv2.LINE_AA
            )
        cv2.putText(
            graph_bg, 
            str(edge_val), 
            (int((block_graph_pos[relation[0]][0] + block_graph_pos[relation[1]][0])/2) + 5,
            int((block_graph_pos[relation[0]][1] + block_graph_pos[relation[1]][1])/2)), 
            cv2.FONT_ITALIC, 0.5,(150,150,150), 1
            )
cv2.imwrite("for_paper/"+str(test_img_name)+"graphy.jpg", graph_bg[50:350,:400])
'''LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL                      
【函数名】
【功  能】make graph has dir
QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ'''
for i, relation in enumerate(occ_relation_final):
    id_1 = relation[0][0]
    id_2 = relation[0][1]
    block_1, block_2 = blocks_list[relation[0][0]],  blocks_list[relation[0][1]]
    if block_1["layer"] > block_2["layer"]:
        occ_relation_final[i][0][0] = id_2
        occ_relation_final[i][0][1] = id_1
    
print("occ_relation_final: ",occ_relation_final)
edge_for_net = occ_relation_final[:]

'''LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL                      
【函数名】
【功  能】make ins order
QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ'''
#先找不动点
ins_order = [0 for _ in range(num_instance)]
for i, relation in enumerate(occ_relation_final):
    if relation[1] > 1: ins_order[relation[0][1]] = -1

order = 1
while occ_relation_final:
    print("/////////////////////////////////////////////////////////////")
    not_move_block = []
    for i, relation in enumerate(occ_relation_final):
        not_move_block.append(relation[0][1])

    for i in range(num_instance):
        if (i not in not_move_block) and (ins_order[i] != -1) and(ins_order[i] == 0):
            ins_order[i] = order

    move_block = []
    for j,o in enumerate(ins_order):
        if o == order:
            print(j,o)
            for i, relation in enumerate(occ_relation_final):
                print(relation)
                if j == relation[0][0]: move_block.append(relation)
                   
    for relation in move_block:
        occ_relation_final.remove(relation)

    order += 1
    print("not_move_block", not_move_block)
    print("occ_relation_final: ", occ_relation_final)

    print("ins_order: ",ins_order)

for j,o in enumerate(ins_order):
    if o == 0: ins_order[j] = order


print("ins_order: ",ins_order)

for i, order in enumerate(ins_order):
    blocks_list[i]["order"] = order

for i, block in enumerate(blocks_list):
    print(block)
    cv2.putText(
            graph_bg, 
            str(block["order"]), 
            (block_graph_pos[i][0] - 5, block_graph_pos[i][1] + 36),
            cv2.FONT_ITALIC, 0.55,(211,0,148), 2
            )
cv2.imwrite("for_paper/"+str(test_img_name)+"graphy_order1.jpg", graph_bg[50:350,:400])
'''LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL                      
【函数名】
【功  能】reverse layers
QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ'''
print("[....................................reverse layers....................................]")
print("layer_max: ", layer_max)
for block in blocks_list:
    block["layer"] = layer_max -  block["layer"] +1
    print(block)

'''LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL                      
【函数名】
【功  能】go net
QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ'''
print("[....................................go net....................................]")
print("edge_for_net: ", edge_for_net)

output_order_now = 1

print("np.max(ins_order): ",np.max(ins_order))
while output_order_now <= np.max(ins_order):
    input_not_block  = []
    for e in edge_for_net:
        input_not_block.append(e[0][0])

    input_block, output_block = [], []
    for  block in blocks_list:
        if block["id"] not in input_not_block:  input_block.append(block)
        if block["order"] == output_order_now: output_block.append(block)

    print("input_block: ", input_block)
    print("output_block: ", output_block)

    output_id, input_id = [], []
    for block in input_block: input_id.append(block["id"])
    for block in output_block: output_id.append(block["id"])
    print("input_id: ", input_id)
    print("output_id: ", output_id)

    x = [1 for _ in range(len(input_block))]
    y = [0 for _ in range(len(output_block))]

    def dfs(stack,node, output, node_end):
        for e in edge_for_net:
            if node == e[0][1]:
                node_find = e[0][0]
                if node_find not in stack: 
                    stack.append(node_find)
                    if node_find == node_end: 
                        output.append(stack[:])
                    dfs(stack, node_find, output, node_end)


    net_path = []
    for block_start in input_block:
        for block_end in output_block:
            node_start = block_start["id"]
            node_end = block_end["id"]
            if node_start == node_end: net_path.append([node_start, node_end])
            print(node_start, node_end)
            stack = []
            stack.append(node_start)
            dfs(stack, node_start, net_path, node_end)
            print(net_path)

    #权值1
    print("//////////////////")
    w1 = [0 for _ in range(len(output_block))]                                                                                                           
    for path in net_path:
        for i, out_id in enumerate(output_id):
            if path[-1] == out_id: w1[i] += ((output_block[i]["layer"])*(output_block[i]["layer"] + 1))/2

    print("w1: ",w1)

    #权值2
    print("//////////////////")
    w2 = [0 for _ in range(len(output_block))]
    for i, out_id_1 in enumerate(output_id):
        dis_list = []
        for j, out_id_2 in enumerate(output_id):
            if out_id_2 != out_id_1:
                dis_x = int((output_block[i]['xy'][0] - output_block[j]['xy'][0])**2)
                dis_y = int((output_block[i]['xy'][1] - output_block[j]['xy'][1])**2)
                dis_xy = dis_x+dis_y
                dis_list.append(np.sqrt(dis_xy))
        if len(output_block) == 1: w2[i] = 1
        else: w2[i] = np.mean(dis_list)
    print("w2: ",w2)

    #最终权值
    print("//////////////////")
    w_final = [0 for _ in range(len(output_block))]
    order_son = [0 for _ in range(len(output_block))]
    for i in range(len(output_block)):
        w_final[i] = w2[i]/w1[i]
    print("w_final: ",w_final)


    #最终子顺序
    print("//////////////////")
    w_final_argsort = np.argsort(w_final)
    w_final_argsort = w_final_argsort[::-1]

    for i, idx in enumerate(w_final_argsort):
        order_son[idx] = i + 1

    print("order_son: ", order_son)


    for i, block in enumerate(output_block):
        blocks_list[block["id"]]["order_son"] = order_son[i]
        cv2.putText(
                graph_bg, 
                "."+str(order_son[i]), 
                (block_graph_pos[block["id"]][0] +7, block_graph_pos[block["id"]][1] + 36),
                cv2.FONT_ITALIC, 0.55,(211,0,148), 2
                )

    output_order_now += 1


for block in blocks_list:
    print(block)



cv2.imwrite("for_paper/"+str(test_img_name)+"graphy_order2.jpg", graph_bg[50:350,:400])





'''LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL                      
【函数名】
【功  能】result show
QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ'''
print('[....................................success!......................................]')
img_show =  np.concatenate((graph_bg, img, img_origin),axis=1) 
cv2.imwrite("for_paper/"+str(test_img_name)+"graphy_order_final.jpg", img_show)
cv2.imshow("img",img_show)
cv2.waitKey(0)
