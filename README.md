# pclines_python
that famous pclines in python

Here, you can find implementation of three papers:

1. http://www.fit.vutbr.cz/research/groups/graph/pclines/pub_page.php?id=2013-BMVC-VanPoints
2. http://www.fit.vutbr.cz/research/groups/graph/pclines/pub_page.php?id=2014-ITS-MotionVanPoints
3. https://www.zpascal.net/cvpr2014/Lezama_Finding_Vanishing_Points_2014_CVPR_paper.pdf

using traffic road and cars, current algorithm calibrates the camera, and saves them.

yolo_detection file uses that calibration file and finds distance based on that.

in order to run it, change path in main_using_pc_lines.py file and run :
<pre>
python main_using_pc_lines.py
</pre>
missing files in the folder:
<pre>
['yolo_cpp_dll.pdb',
 'yolo_cpp_dll.exp',
 'pthreadVC2.dll',
 'yolo_cpp_dll.iobj',
 'calibration.pickle',
 'opencv_world346.dll',
 'yolo_cpp_dll.ipdb',
 'cudnn64_7.dll',
 'yolo_cpp_dll.lib',
 'libdarknet.so',
 'yolov3.weights',
 'yolo_cpp_dll.dll',
 'pthreadGC2.dll']
 </pre>
code is messy, as i am. try to use it on your own.
