/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv/cv.h>


namespace ORB_SLAM2
{
/**
 * 该类中定义了四叉树创建的函数以及树中结点的属性
 * bool bNoMore： 根据该结点中被分配的特征点的数目来决定是否继续对其进行分割
 * DivisionNode()：实现如何对一个结点进行分割
 * vKeys：用来存储被分配到该结点区域内的所有特征点
 * UL, UR, BL, BR：四个点定义了一个结点的区域
 * lit:list的迭代器，遍历所有生成的节点
 * 
*/
class ExtractorNode
{
public:
    ExtractorNode():bNoMore(false){}

    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<ExtractorNode>::iterator lit;
    bool bNoMore;
};

class ORBextractor
{
public:
    
    enum {HARRIS_SCORE=0, FAST_SCORE=1 };
    /**
     * nfeatures ORB特征点数量
     * scaleFactor相邻层的放大倍数
     * nlevels层数
     * iniThFAST 提取FAST角点时初始阈值
     * minThFAST 提取FAST角点时更小的阈值
     * 注意：设置两个阈值的原因是在FAST提取角点进行分块后有可能在某个块中在原始阈值情况下提取不到角点，使用更小的阈值进一步提取
     * */
    ORBextractor(int nfeatures, float scaleFactor, int nlevels,
                 int iniThFAST, int minThFAST);

    ~ORBextractor(){}

    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an octree.
    // Mask is ignored in the current implementation.
    void operator()( cv::InputArray image, cv::InputArray mask,
      std::vector<cv::KeyPoint>& keypoints,
      cv::OutputArray descriptors);

    int inline GetLevels(){
        return nlevels;}

    float inline GetScaleFactor(){
        return scaleFactor;}

    std::vector<float> inline GetScaleFactors(){
        return mvScaleFactor;
    }

    std::vector<float> inline GetInverseScaleFactors(){
        return mvInvScaleFactor;
    }

    std::vector<float> inline GetScaleSigmaSquares(){
        return mvLevelSigma2;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares(){
        return mvInvLevelSigma2;
    }
    //图像金字塔 存放各层的图片
    std::vector<cv::Mat> mvImagePyramid;

protected:
    //计算高斯金字塔
    void ComputePyramid(cv::Mat image);
    //计算关键点并用四叉树进行存储
    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
    //将关键点分配到四叉树 
    std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                           const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level);

    void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
    //存储关键点附近patch的点对
    std::vector<cv::Point> pattern;
    //提取特征点的最大数量
    int nfeatures;
    //每层之间的缩放比例
    double scaleFactor;
    //高斯金字塔的层数
    int nlevels;
    //iniThFAST提取FAST角点时初始阈值
    int iniThFAST;
    //minThFAST提取FAST角点时更小的阈值
    int minThFAST;
    //每层的特征数量
    std::vector<int> mnFeaturesPerLevel;
    //Patch圆的最大坐标
    std::vector<int> umax;
    //每层的相对于原始图像的缩放比例
    std::vector<float> mvScaleFactor;
    //每层的相对于原始图像的缩放比例的倒数
    std::vector<float> mvInvScaleFactor;
    //mvScaleFactor的平方  
    std::vector<float> mvLevelSigma2;
    //mvScaleFactor的平方的倒数
    std::vector<float> mvInvLevelSigma2;
};

} //namespace ORB_SLAM

#endif

