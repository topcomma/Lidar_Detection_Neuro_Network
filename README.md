# Lidar_Detection_Neuro_Network

基于激光雷达点云的3D目标检测算法论文总结


前言 
过去很多年激光雷达的车规标准和高昂价格是阻碍其量产落地的主要因素，最近两三年随着速腾、禾赛、大疆、图达通、Luminar等厂家混合固态激光雷达的量产，新势力车企、互联网车企陆续发布与交付了基于激光雷达的车型，比如：小鹏P5、蔚来ET7/ET5、集度概念车、威马M7、智己L7、高合HiPhiZ、沙龙机甲龙、极狐HBT，混合固态激光雷达即将进入批量量产的前夜。后续随着各大厂商智能电动车型的大规模量产与交付，混合固态激光雷达可能将会是主流车型的标配。
LiDAR感知、定位、建图、预测算法功能的开发将在车企/供应商ADAS团队中占比越来越多，不再仅仅是一个辅助/真值系统的存在。最近疫情在家，对过去几年学习、积累的LiDAR目标检测算法(不包含传统算法、车道线、FreeSpace检测)论文做了总结，共计有54篇论文及代码，有些是基础网络算法，有些经典的、最新的算法也可作为工程落地的参考方案。
基于激光雷达点云的3D目标检测算法有很多种方法：传统聚类方法，点云、体素化、柱状化，RangeView、BirdEyeView，多帧、多视图，OneStage、TwoStage，AnchorBased、AnchroFree、关键点、中心点、Voting、与分割结合、结合反射强度与线束角、转为深度图，知识蒸馏、Transformer、Atteintion、半监督，2DCNN、3D稀疏卷积、图卷积，与Camera图像数据数据融合、特征融合。
从现阶段角度，激光雷达本身还有很多工程问题(布置、噪声、标定、同步、畸变、补偿、安全)需要尝试和解决，还有一个难点是网络模型在嵌入式平台的部署与优化。但是对于目标检测算法本身，还是先基于CNN、BEV、AnchorBased/中心点为基础算法完成工程落地，后续逐渐升级到以Transformer/Fusion框架的大感知框架。先以LiDAR/Camera后融合为主，可能的话，逐渐走向前融合的方案。
算法论文
3DSSD 
题目：3DSSD: Point-based 3D Single Stage Object Detector
名称：3DSSD：基于点的 3D 单级物体检测器
论文：https://arxiv.org/abs/2002.10187
代码：https://github.com/tomztyang/3DSSD
AFDet 
题目：AFDet: Anchor Free One Stage 3D Object Detection
名称：AFDet：无锚的一级 3D 对象检测
论文：https://arxiv.org/abs/2006.12671
Associate-3DDet 
题目：Associate-3Ddet: Perceptual-to-Conceptual Association for 3D Point Cloud Object Detection
名称：Associate-3Ddet：3D 点云对象检测的感知到概念关联
论文：https://arxiv.org/abs/2006.04356
BackReality 
题目：Back to Reality: Weakly-supervised 3D Object Detection with Shape-guided Label Enhancement
名称：回到现实：带有形状引导标签增强的弱监督 3D 对象检测
论文：https://arxiv.org/abs/2203.05238
代码：https://github.com/wyf-ACCEPT/BackToReality
BEVDetNet 
题目：BEVDetNet: Bird's Eye View LiDAR Point Cloud based Real-time 3D Object Detection for Autonomous Driving
名称：BEVDetNet：基于鸟瞰 LiDAR 点云的自动驾驶实时 3D 对象检测
论文：https://arxiv.org/abs/2104.10780
BirdNet 
题目：BirdNet: a 3D Object Detection Framework from LiDAR information
名称：BirdNet：来自 LiDAR 信息的 3D 对象检测框架
论文：https://arxiv.org/abs/1805.01195
BirdNet+ 
题目：BirdNet+: End-to-End 3D Object Detection in LiDAR Bird's Eye View
名称：BirdNet+：LiDAR 鸟瞰图中的端到端 3D 对象检测
论文：https://arxiv.org/abs/2003.04188
CanonicalVoting 
题目：Canonical Voting: Towards Robust Oriented Bounding Box Detection in 3D Scenes
名称：规范投票：在 3D 场景中实现稳健的定向边界框检测
论文：https://arxiv.org/abs/2011.12001
代码：https://github.com/qq456cvb/CanonicalVoting
CenterNet3D 
题目：CenterNet3D: An Anchor Free Object Detector for Point Cloud
名称：用于自动驾驶的无锚物体检测器
论文：https://arxiv.org/abs/2007.07214
代码：https://github.com/wangguojun2018/CenterNet3d
CenterPoint 
题目：Center-based 3D Object Detection and Tracking
名称：基于中心的3D目标检测和跟踪
论文：https://arxiv.org/abs/2006.11275
代码：https://github.com/tianweiy/CenterPoint
CG-SSD 
题目：CG-SSD: Corner Guided Single Stage 3D Object Detection from LiDAR Point Cloud
名称：CG-SSD：来自 LiDAR 点云的角引导单级 3D 对象检测
论文：https://arxiv.org/abs/2202.11868
CIA-SSD 
题目：CIA-SSD: Confident IoU-Aware Single-Stage Object Detector From Point Cloud
名称：CIA-SSD：来自点云的自信的 IoU 感知单级目标检测器
论文：https://arxiv.org/abs/2012.03015
代码：https://github.com/Vegeta2020/CIA-SSD
ClassBalanced-GS 
题目：Class-balanced Grouping and Sampling for Point Cloud 3D Object Detection
名称：用于点云 3D 对象检测的类平衡分组和采样
论文：https://arxiv.org/abs/1908.09492
Complex-YOLO 
题目：Complex-YOLO: Real-time 3D Object Detection on Point Clouds
名称：Complex-YOLO：点云上的实时 3D 对象检测
论文：https://arxiv.org/abs/1803.06199
代码：https://github.com/AI-liu/Complex-YOLO
CT3D 
题目：Improving 3D Object Detection with Channel-wise Transformer
名称：使用 Channel-wise Transformer 改进 3D 对象检测
论文：https://arxiv.org/abs/2108.10723
Deformable-PV-RCNN 
题目：Deformable PV-RCNN: Improving 3D Object Detection with Learned Deformations
名称：可变形 PV-RCNN：通过学习变形改进 3D 对象检测
论文：https://arxiv.org/abs/2008.08766
E2E-PL 
题目：End-to-End Pseudo-LiDAR for Image-Based 3D Object Detection
名称：用于基于图像的 3D 对象检测的端到端伪激光雷达
论文：https://arxiv.org/abs/2004.03080
代码：https://github.com/mileyan/pseudo-LiDAR_e2e
Fast-Point-RCNN 
题目：Fast Point R-CNN
名称：快速点 R-CNN
论文：https://arxiv.org/abs/1908.02990
FVNet 
题目：FVNet: 3D Front-View Proposal Generation for Real-Time Object Detection from Point Clouds
名称：FVNet：用于从点云进行实时对象检测的 3D 前视图建议生成
论文：https://arxiv.org/abs/1903.10750
Hollow3D-RCNN 
题目：From Multi-View to Hollow-3D: Hallucinated Hollow-3D R-CNN for 3D Object Detection
名称：从多视图到 Hollow-3D：用于 3D 对象检测的幻觉 Hollow-3D R-CNN
论文：https://arxiv.org/abs/2107.14391
HotSpotNet 
题目：Object as Hotspots: An Anchor-Free 3D Object Detection Approach via Firing of Hotspots
名称：对象即热点：通过触发热点的无锚 3D 对象检测方法
论文：https://arxiv.org/abs/1912.12791
HVPR 
题目：HVPR: Hybrid Voxel-Point Representation for Single-stage 3D Object Detection
名称：HVPR：用于单级 3D 对象检测的混合体素点表示
论文：https://arxiv.org/abs/2104.00902
IS-SSD 
题目：Not All Points Are Equal: Learning Highly Efficient Point-based Detectors for 3D LiDAR Point Clouds
名称：并非所有点都是平等的：学习用于 3D LiDAR 点云的高效基于点的检测器
论文：https://arxiv.org/abs/2203.11139
代码：https://github.com/yifanzhang713/IA-SSD
LaserNet 
题目：LaserNet: An Efficient Probabilistic 3D Object Detector for Autonomous Driving
名称：LaserNet：用于自动驾驶的高效概率 3D 对象检测器
论文：https://arxiv.org/abs/1903.08701
Lidar-RCNN 
名称：LiDAR R-CNN：一种高效且通用的 3D 物体检测器
论文：https://arxiv.org/abs/2103.15297
代码：https://github.com/tusimple/LiDAR_RCNN
MLCVNet 
题目：MLCVNet: Multi-Level Context VoteNet for 3D Object Detection
名称：MLCVNet：用于三维目标检测的多级上下文VoteNet
论文：https://openaccess.thecvf.com/content_CVPR_2020/papers/Xie_MLCVNet_Multi-Level_Context_VoteNet_for_3D_Object_Detection_CVPR_2020_paper.pdf
代码：https://github.com/NUAAXQ/MLCVNet
MVF 
题目：End-to-End Multi-View Fusion for 3D Object Detection in LiDAR Point Clouds
名称：用于 LiDAR 点云中 3D 对象检测的端到端多视图融合
论文：https://arxiv.org/abs/1910.06528
PartA2Net 
题目：From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
名称：从点到部分：使用部分感知和部分聚合网络从点云进行 3D 对象检测
论文：https://arxiv.org/abs/1907.03670
代码：https://github.com/sshaoshuai/PointCloudDet3D
PIXOR 
题目：PIXOR: Real-time 3D Object Detection from Point Clouds
名称：PIXOR：点云的实时 3D 对象检测
论文：https://arxiv.org/abs/1902.06326
Pointformer
题目：3D Object Detection with Pointformer
名称：3D Object Detection with Pointformer
论文：https://arxiv.org/abs/2012.11409
代码：https://github.com/Vladimir2506/Pointformer
Point-GNN 
题目：Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud
名称：Point-GNN：用于点云中 3D 对象检测的图神经网络
论文：https://arxiv.org/abs/2003.01251
代码：https://github.com/WeijingShi/Point-GNN
PointPillars 
题目：PointPillars: Fast Encoders for Object Detection from Point Clouds
名称：PointPillars：点云目标检测的快速编码器
论文：https://arxiv.org/abs/1812.05784
论文：https://openaccess.thecvf.com/content_CVPR_2019/papers/Lang_PointPillars_Fast_Encoders_for_Object_Detection_From_Point_Clouds_CVPR_2019_paper.pdf
代码：https://github.com/nutonomy/second.pytorch
PointRCNN 
题目：PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud
名称：PointRCNN：来自点云的 3D 对象建议生成和检测
论文：https://arxiv.org/abs/1812.04244
代码：https://github.com/sshaoshuai/PointRCNN
Pseudo-LiDAR 
题目：Pseudo-LiDAR from Visual Depth Estimation: Bridging the Gap in 3D Object Detection for Autonomous Driving
名称：来自视觉深度估计的伪激光雷达：弥合自动驾驶 3D 对象检测的差距
论文：https://arxiv.org/abs/1812.07179
代码：https://github.com/mileyan/pseudo_lidar
PU-Net 
题目：PU-Net: Point Cloud Upsampling Network
名称：PU-Net：点云上采样网络
论文：https://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_PU-Net_Point_Cloud_CVPR_2018_paper.pdf
代码：https://github.com/yulequan/PU-Net
Point-Voxel
题目：Point-Voxel CNN for Efficient 3D Deep Learning
名称：用于高效 3D 深度学习的点体素 CNN
论文：https://arxiv.org/abs/1907.03739
主页：https://pvcnn.mit.edu/
项目：https://developer.nvidia.com/blog/point-voxel-cnn-3d/
PV-RCNN 
题目：PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection
名称：PV-RCNN：用于 3D 对象检测的点体素特征集抽象
论文：https://arxiv.org/abs/1912.13192
代码：https://github.com/open-mmlab/OpenPCDet
PV-RCNN++
题目：PV-RCNN++: Point-Voxel Feature Set Abstraction With Local Vector Representation for 3D Object Detection
名称：PV-RCNN++：用于 3D 对象检测的具有局部向量表示的点体素特征集抽象
论文：https://arxiv.org/abs/2102.00463
代码：https://github.com/open-mmlab/OpenPCDet
RangeDet 
题目：RangeDet:In Defense of Range View for LiDAR-based 3D Object Detection
名称：RangeDet：为基于 LiDAR 的 3D 对象检测保护范围视图
论文：https://arxiv.org/abs/2103.10039
SA-Det3D 
题目：SA-Det3D: Self-Attention Based Context-Aware 3D Object Detection
名称：SA-Det3D：基于自注意力的上下文感知 3D 对象检测
论文：https://arxiv.org/abs/2101.02672
代码：https://github.com/AutoVision-cloud/SA-Det3D
SASA 
题目：SASA: Semantics-Augmented Set Abstraction for Point-based 3D Object Detection
名称：SASA：基于点的 3D 对象检测的语义增强集抽象
论文：https://arxiv.org/pdf/2201.01976.pdf
代码：https://github.com/blakechen97/SASA
SA-SSD
题目：Structure Aware Single-stage 3D Object Detection from Point Cloud
名称：基于点云的结构感知单级三维目标检测
论文：https://www4.comp.polyu.edu.hk/~cslzhang/paper/SA-SSD.pdf
代码：https://github.com/skyhehe123/SA-SSD
SECOND 
题目：SECOND: Sparsely Embedded Convolutional Detection
名称：第二：稀疏嵌入卷积检测
论文：https://www.mdpi.com/1424-8220/18/10/3337
SE-SSD
题目：SE-SSD: Self-Ensembling Single-Stage Object Detector From Point Cloud
名称：SE-SSD：来自点云的自集成单级目标检测器
论文：https://arxiv.org/abs/2104.09804
代码：https://github.com/Vegeta2020/SE-SSD
SIENet 
题目：SIENet: Spatial Information Enhancement Network for 3D Object Detection from Point Cloud
名称：SIENet：用于从点云进行 3D 对象检测的空间信息增强网络
论文：https://arxiv.org/abs/2103.15396
SS3D 
题目：SS3D: Single Shot 3D Object Detector
名称：SS3D：单次 3D 物体检测器
论文：https://arxiv.org/abs/2004.14674
相关课程：国内首个3D缺陷检测教程：理论、源码与实战
SST 
题目：Embracing Single Stride 3D Object Detector with Sparse Transformer
名称：使用 Sparse Transformer 拥抱单步 3D 对象检测器
论文：https://arxiv.org/abs/2112.06375
代码：https://github.com/TuSimple/SST
STD 
题目：STD: Sparse-to-Dense 3D Object Detector for Point Cloud
名称：STD：点云的稀疏到密集 3D 对象检测器
论文：https://arxiv.org/abs/1907.10471
TANet 
题目：TANet: Robust 3D Object Detection from Point Clouds with Triple Attention
名称：TANet：来自具有三重注意力的点云的稳健 3D 对象检测
论文：https://arxiv.org/abs/1912.05163
VoteNet 
题目：Deep Hough Voting for 3D Object Detection in Point Clouds
名称：用于点云中 3D 对象检测的深度霍夫投票
论文：https://arxiv.org/abs/1904.09664
代码：https://github.com/facebookresearch/votenet
VOTR
题目：Voxel Transformer for 3D Object Detection
名称：用于 3D 对象检测的体素转换器
论文：https://arxiv.org/abs/2109.02497
代码：https://github.com/PointsCoder/VOTR
Voxel-FPN
题目：Voxel-FPN: multi-scale voxel feature aggregation in 3D object detection from point clouds
名称：Voxel-FPN：点云 3D 对象检测中的多尺度体素特征聚合
论文：https://arxiv.org/abs/1907.05286
VoxelNet 
题目：VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection
名称：VoxelNet：基于点云的 3D 对象检测的端到端学习
论文：https://arxiv.org/abs/1711.06396
代码：https://github.com/qianguih/voxelnet
Voxel-RCNN 
题目：Voxel R-CNN: Towards High Performance Voxel-based 3D Object Detection
名称：Voxel R-CNN：迈向高性能基于体素的 3D 对象检测
论文：https://arxiv.org/abs/2012.15712
