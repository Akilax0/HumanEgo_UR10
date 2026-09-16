#!/usr/bin/env python3
"""Scrub through Aria VRS RGB frames with timestamps. 
Keys: a/d = prev/next, A/D = jump 30, m = mark timestamp, q = quit."""
import cv2
import numpy as np
from pathlib import Path
from projectaria_tools.core import data_provider

VRS = "/home/ax0/Documents/aria_data/Bellpepper_grasp_retake/vrs/Grasp_Bellpepper_Red_7_20260908_170304.vrs"
MARKS_FILE = Path("/home/ax0/Documents/aria_data/Bellpepper_grasp_retake/marks_files/Grasp_BellPepper_Red_7_marks.txt")

MARKS_FILE.parent.mkdir(parents=True, exist_ok=True)  # make the ..._marks_files/ dir
MARKS_FILE.touch(exist_ok=True)

provider = data_provider.create_vrs_data_provider(VRS)
rgb = provider.get_stream_id_from_label("camera-rgb")
n = provider.get_num_data(rgb)

# First timestamp for relative time display
t0_ns = provider.get_image_data_by_index(rgb, 0)[1].capture_timestamp_ns

marks = []
i = 0

cv2.namedWindow("aria", cv2.WINDOW_NORMAL)
while True:
    data = provider.get_image_data_by_index(rgb, i)
    img = np.rot90(data[0].to_numpy_array(), 0).copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    ts_ns = data[1].capture_timestamp_ns
    ts_us = ts_ns // 1000
    rel_s = (ts_ns - t0_ns) / 1e9

    label = f"frame {i}/{n-1}  |  {rel_s:8.3f}s  |  device: {ts_us} us"
    cv2.putText(img, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 2)

    cv2.imshow("aria", img)
    k = cv2.waitKey(0) & 0xFF
    if k == ord('q'): break
    elif k == ord('d'): i = min(i + 1, n - 1)
    elif k == ord('a'): i = max(i - 1, 0)
    elif k == ord('f'): i = min(i + 30, n - 1)
    elif k == ord('w'): i = max(i - 30, 0)
    elif k == ord('m'):
        # marks.append(ts_us)
        marks.append((i, ts_us, rel_s))
        with open(MARKS_FILE, "a") as f:
            f.write(f"{ts_us}\t{rel_s:.3f}\t{i}\n")
        print(f"marked: {ts_us} us  ({rel_s:.3f}s)  -> {MARKS_FILE}")
        # print(f"marked: {ts_us} us  ({rel_s:.3f}s)")

cv2.destroyAllWindows()
print(f"\n{len(marks)} marks saved to {MARKS_FILE}")
print("\nSegment boundaries (us):", marks)

"""
Red_9
marked: 1206743659 us  (32.999s)
marked: 1221143111 us  (47.398s)
marked: 1227942172 us  (54.197s)
marked: 1241141182 us  (67.396s)
marked: 1247240838 us  (73.496s)
marked: 1272138750 us  (98.394s)
marked: 1277138594 us  (103.394s)
marked: 1299436690 us  (125.692s)
marked: 1304036512 us  (130.292s)
marked: 1319435612 us  (145.691s)
marked: 1324434955 us  (150.690s)
marked: 1340533991 us  (166.789s)
marked: 1345933315 us  (172.189s)
marked: 1362232392 us  (188.488s)
marked: 1366931958 us  (193.187s)
marked: 1379531434 us  (205.787s)
marked: 1385230595 us  (211.486s)
marked: 1404029565 us  (230.285s)
marked: 1409428545 us  (235.684s)
marked: 1432527361 us  (258.783s)
marked: 1437226198 us  (263.481s)
marked: 1455124827 us  (281.380s)
marked: 1459824631 us  (286.080s)
marked: 1479623087 us  (305.878s)
marked: 1485222815 us  (311.478s)
marked: 1504221407 us  (330.477s)
marked: 1512220445 us  (338.476s)
marked: 1531118993 us  (357.374s)
marked: 1536918749 us  (363.174s)
marked: 1557116993 us  (383.372s)
marked: 1566716620 us  (392.972s)
marked: 1583814937 us  (410.070s)
"""

"""
Red_8
marked: 83537990 us  (60.495s)
marked: 101937058 us  (78.894s)
marked: 107235832 us  (84.193s)
marked: 125334470 us  (102.291s)
marked: 132134090 us  (109.091s)
marked: 154332283 us  (131.289s)
marked: 159632312 us  (136.589s)
marked: 179030416 us  (155.987s)
marked: 185030050 us  (161.987s)
marked: 208728822 us  (185.686s)
marked: 226526812 us  (203.484s)
marked: 246026143 us  (222.983s)
marked: 252025174 us  (228.982s)
marked: 267023728 us  (243.981s)
"""



"""
Red_2
marked: 1960785309 us  (26.997s)
marked: 1971484880 us  (37.696s)
marked: 1976684360 us  (42.896s)
marked: 1986083425 us  (52.295s)
marked: 1992482947 us  (58.694s)
marked: 2009981638 us  (76.193s)
marked: 2017081264 us  (83.293s)
marked: 2037879821 us  (104.091s)
marked: 2043679107 us  (109.890s)
marked: 2063877584 us  (130.089s)
"""

"""
Red_3
marked: 2956790346 us  (33.497s)
marked: 2971289074 us  (47.996s)
marked: 2979288299 us  (55.995s)
marked: 3002086587 us  (78.793s)
marked: 3013785709 us  (90.492s)
marked: 3038183937 us  (114.890s)
marked: 3045483323 us  (122.190s)
marked: 3064382060 us  (141.089s)
marked: 3071481358 us  (148.188s)
marked: 3090279933 us  (166.986s)
marked: 3097279402 us  (173.986s)
marked: 3112278263 us  (188.985s)
marked: 3116877913 us  (193.584s)
marked: 3128677015 us  (205.384s)
marked: 3150375361 us  (227.082s)
marked: 3164574278 us  (241.281s)

"""

"""
Red_4
marked: 3944280714 us  (35.998s)
marked: 3962879607 us  (54.597s)
marked: 3972278617 us  (63.996s)
marked: 3988477400 us  (80.195s)
marked: 3995076904 us  (86.794s)
marked: 4010375751 us  (102.093s)
marked: 4021274929 us  (112.992s)
marked: 4044973245 us  (136.690s)
marked: 4054172440 us  (145.890s)
marked: 4072371382 us  (164.089s)
marked: 4081370598 us  (173.088s)
marked: 4100968995 us  (192.686s)
"""

"""
Red_5
marked: 6153886901 us  (17.100s)
marked: 6165485788 us  (28.699s)
marked: 6171085024 us  (34.298s)
marked: 6184384241 us  (47.597s)
marked: 6195083235 us  (58.296s)
marked: 6215881679 us  (79.095s)
marked: 6226180907 us  (89.394s)
marked: 6245679537 us  (108.892s)
marked: 6252578924 us  (115.792s)
marked: 6273577341 us  (136.790s)
marked: 6285176554 us  (148.390s)
marked: 6303675274 us  (166.888s)
marked: 6309674612 us  (172.888s)
marked: 6326973300 us  (190.186s)
marked: 6333972769 us  (197.186s)
marked: 6352571445 us  (215.784s)
marked: 6359071191 us  (222.284s)
marked: 6376970183 us  (240.183s)
marked: 6382669072 us  (245.882s)
marked: 6398368122 us  (261.581s)
marked: 6404367412 us  (267.580s)
marked: 6423165977 us  (286.379s)
marked: 6429265511 us  (292.478s)
marked: 6442864472 us  (306.077s)
marked: 6452763714 us  (315.977s)
marked: 6468762825 us  (331.976s)
"""

"""
Red_6
marked: 7102785112 us  (43.498s)
marked: 7115083883 us  (55.796s)
marked: 7119883518 us  (60.596s)
marked: 7134182661 us  (74.895s)
marked: 7137182165 us  (77.895s)
marked: 7146181673 us  (86.894s)
marked: 7152081047 us  (92.794s)
marked: 7170179685 us  (110.892s)
marked: 7177479136 us  (118.192s)
marked: 7191778531 us  (132.491s)
marked: 7199677461 us  (140.390s)
marked: 7215376274 us  (156.089s)
marked: 7222675722 us  (163.388s)
marked: 7236574669 us  (177.287s)
marked: 7243774123 us  (184.487s)
marked: 7256973949 us  (197.686s)
marked: 7260272946 us  (200.985s)
marked: 7272672004 us  (213.385s)
marked: 7277571556 us  (218.284s)
marked: 7289070681 us  (229.783s)
marked: 7292970428 us  (233.683s)
marked: 7305269991 us  (245.983s)
marked: 7313468822 us  (254.181s)
marked: 7326767808 us  (267.480s)
marked: 7330367533 us  (271.080s)
marked: 7345766358 us  (286.479s)
marked: 7354565698 us  (295.278s)
marked: 7369364554 us  (310.077s)
marked: 7374664225 us  (315.377s)
marked: 7386963903 us  (327.676s)
marked: 7390962902 us  (331.675s)
marked: 7403661930 us  (344.374s)
marked: 7408061612 us  (348.774s)
marked: 7419361075 us  (360.074s)
"""

"""
Red_7
marked: 8665837647 us  (30.098s)
marked: 8679736754 us  (43.997s)
marked: 8685236122 us  (49.497s)
marked: 8697335218 us  (61.596s)
marked: 8704334693 us  (68.595s)
marked: 8718333644 us  (82.594s)
marked: 8724033362 us  (88.294s)
marked: 8735232672 us  (99.493s)
marked: 8739832030 us  (104.093s)
marked: 8751831127 us  (116.092s)
marked: 8756430781 us  (120.691s)
marked: 8768529869 us  (132.791s)
marked: 8776829242 us  (141.090s)
marked: 8788228381 us  (152.489s)
marked: 8792228079 us  (156.489s)
marked: 8803527224 us  (167.788s)
marked: 8807526921 us  (171.788s)
marked: 8816926209 us  (181.187s)
marked: 8826025519 us  (190.286s)
marked: 8838224593 us  (202.485s)
marked: 8843324206 us  (207.585s)
marked: 8856023386 us  (220.284s)
marked: 8862022785 us  (226.283s)
marked: 8873621902 us  (237.883s)
marked: 8878721514 us  (242.982s)
marked: 8890820592 us  (255.081s)
marked: 8899420038 us  (263.681s)
marked: 8908619735 us  (272.880s)
marked: 8915918679 us  (280.179s)
marked: 8926417877 us  (290.679s)
marked: 8935017221 us  (299.278s)
marked: 8944916464 us  (309.177s)
marked: 8951715945 us  (315.977s)
marked: 8963015081 us  (327.276s)
"""