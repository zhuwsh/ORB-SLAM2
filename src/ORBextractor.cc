/**
* This file is part of ORB-SLAM2.
* This file is based on the file orb.cpp from the OpenCV library (see BSD license below).
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
/**
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*/


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#include "ORBextractor.h"


using namespace cv;
using namespace std;

namespace ORB_SLAM2
{

const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;

/**
 * image 为某个金字塔层的图像
 * pt    为该图中一个关键点
 * u_max 为之前计算出来的一个数组长度为16的数值
 * umax[16]中所有数据，依次是：15 15 15 15 14 14 14 13 13 12 11 10 9 8 6 3
 * 对照着高博的《视觉SLAM十四讲》中的第七讲理解方向的计算，但说实话这块代码我还是没有完全理解，还需要继续理解
*/
static float IC_Angle(const Mat& image, Point2f pt,  const vector<int> & u_max)
{
    int m_01 = 0, m_10 = 0;
    //opencv3中图形存储基本为Mat格式，如果我们想获取像素点的灰度值或者RGB值，可以通过image.at<uchar>(i,j)的方式轻松获取。
    const uchar* center = &image.at<uchar> (cvRound(pt.y), cvRound(pt.x));

    // Treat the center line differently, v=0
    //u从-15到15之间
    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
        m_10 += u * center[u];

    // Go line by line in the circuI853lar patch
    int step = (int)image.step1();
    for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
    {
        // Proceed over the two lines
        int v_sum = 0;
        int d = u_max[v];
        for (int u = -d; u <= d; ++u)
        {
            int val_plus = center[u + v*step], val_minus = center[u - v*step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }
    //返回计算的角度
    return fastAtan2((float)m_01, (float)m_10);
}

//弧度制和角度制之间的转换
const float factorPI = (float)(CV_PI/180.f);
//针对每一个关键点计算ORB描述子
static void computeOrbDescriptor(const KeyPoint& kpt,
                                 const Mat& img, const Point* pattern,
                                 uchar* desc)
{
    //获取关键点的弧度表示下的角度值
    float angle = (float)kpt.angle*factorPI;
    //考虑关键点的方向
    float a = (float)cos(angle), b = (float)sin(angle);

    const uchar* center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
    const int step = (int)img.step;
    //注意这里在pattern里找点的时候，是利用关键点的方向信息进行了旋转矫正的
    #define GET_VALUE(idx) \
        center[cvRound(pattern[idx].x*b + pattern[idx].y*a)*step + \
               cvRound(pattern[idx].x*a - pattern[idx].y*b)]


    for (int i = 0; i < 32; ++i, pattern += 16)
    {
        int t0, t1, val;
        t0 = GET_VALUE(0); t1 = GET_VALUE(1);
        val = t0 < t1;
        t0 = GET_VALUE(2); t1 = GET_VALUE(3);
        val |= (t0 < t1) << 1;
        t0 = GET_VALUE(4); t1 = GET_VALUE(5);
        val |= (t0 < t1) << 2;
        t0 = GET_VALUE(6); t1 = GET_VALUE(7);
        val |= (t0 < t1) << 3;
        t0 = GET_VALUE(8); t1 = GET_VALUE(9);
        val |= (t0 < t1) << 4;
        t0 = GET_VALUE(10); t1 = GET_VALUE(11);
        val |= (t0 < t1) << 5;
        t0 = GET_VALUE(12); t1 = GET_VALUE(13);
        val |= (t0 < t1) << 6;
        t0 = GET_VALUE(14); t1 = GET_VALUE(15);
        val |= (t0 < t1) << 7;

        desc[i] = (uchar)val;
    }

    #undef GET_VALUE
}

//计算描述子的pattern,高斯分布，也可以使用其他定义的pattern
static int bit_pattern_31_[256*4] =
{
    8,-3, 9,5/*mean (0), correlation (0)*/,
    4,2, 7,-12/*mean (1.12461e-05), correlation (0.0437584)*/,
    -11,9, -8,2/*mean (3.37382e-05), correlation (0.0617409)*/,
    7,-12, 12,-13/*mean (5.62303e-05), correlation (0.0636977)*/,
    2,-13, 2,12/*mean (0.000134953), correlation (0.085099)*/,
    1,-7, 1,6/*mean (0.000528565), correlation (0.0857175)*/,
    -2,-10, -2,-4/*mean (0.0188821), correlation (0.0985774)*/,
    -13,-13, -11,-8/*mean (0.0363135), correlation (0.0899616)*/,
    -13,-3, -12,-9/*mean (0.121806), correlation (0.099849)*/,
    10,4, 11,9/*mean (0.122065), correlation (0.093285)*/,
    -13,-8, -8,-9/*mean (0.162787), correlation (0.0942748)*/,
    -11,7, -9,12/*mean (0.21561), correlation (0.0974438)*/,
    7,7, 12,6/*mean (0.160583), correlation (0.130064)*/,
    -4,-5, -3,0/*mean (0.228171), correlation (0.132998)*/,
    -13,2, -12,-3/*mean (0.00997526), correlation (0.145926)*/,
    -9,0, -7,5/*mean (0.198234), correlation (0.143636)*/,
    12,-6, 12,-1/*mean (0.0676226), correlation (0.16689)*/,
    -3,6, -2,12/*mean (0.166847), correlation (0.171682)*/,
    -6,-13, -4,-8/*mean (0.101215), correlation (0.179716)*/,
    11,-13, 12,-8/*mean (0.200641), correlation (0.192279)*/,
    4,7, 5,1/*mean (0.205106), correlation (0.186848)*/,
    5,-3, 10,-3/*mean (0.234908), correlation (0.192319)*/,
    3,-7, 6,12/*mean (0.0709964), correlation (0.210872)*/,
    -8,-7, -6,-2/*mean (0.0939834), correlation (0.212589)*/,
    -2,11, -1,-10/*mean (0.127778), correlation (0.20866)*/,
    -13,12, -8,10/*mean (0.14783), correlation (0.206356)*/,
    -7,3, -5,-3/*mean (0.182141), correlation (0.198942)*/,
    -4,2, -3,7/*mean (0.188237), correlation (0.21384)*/,
    -10,-12, -6,11/*mean (0.14865), correlation (0.23571)*/,
    5,-12, 6,-7/*mean (0.222312), correlation (0.23324)*/,
    5,-6, 7,-1/*mean (0.229082), correlation (0.23389)*/,
    1,0, 4,-5/*mean (0.241577), correlation (0.215286)*/,
    9,11, 11,-13/*mean (0.00338507), correlation (0.251373)*/,
    4,7, 4,12/*mean (0.131005), correlation (0.257622)*/,
    2,-1, 4,4/*mean (0.152755), correlation (0.255205)*/,
    -4,-12, -2,7/*mean (0.182771), correlation (0.244867)*/,
    -8,-5, -7,-10/*mean (0.186898), correlation (0.23901)*/,
    4,11, 9,12/*mean (0.226226), correlation (0.258255)*/,
    0,-8, 1,-13/*mean (0.0897886), correlation (0.274827)*/,
    -13,-2, -8,2/*mean (0.148774), correlation (0.28065)*/,
    -3,-2, -2,3/*mean (0.153048), correlation (0.283063)*/,
    -6,9, -4,-9/*mean (0.169523), correlation (0.278248)*/,
    8,12, 10,7/*mean (0.225337), correlation (0.282851)*/,
    0,9, 1,3/*mean (0.226687), correlation (0.278734)*/,
    7,-5, 11,-10/*mean (0.00693882), correlation (0.305161)*/,
    -13,-6, -11,0/*mean (0.0227283), correlation (0.300181)*/,
    10,7, 12,1/*mean (0.125517), correlation (0.31089)*/,
    -6,-3, -6,12/*mean (0.131748), correlation (0.312779)*/,
    10,-9, 12,-4/*mean (0.144827), correlation (0.292797)*/,
    -13,8, -8,-12/*mean (0.149202), correlation (0.308918)*/,
    -13,0, -8,-4/*mean (0.160909), correlation (0.310013)*/,
    3,3, 7,8/*mean (0.177755), correlation (0.309394)*/,
    5,7, 10,-7/*mean (0.212337), correlation (0.310315)*/,
    -1,7, 1,-12/*mean (0.214429), correlation (0.311933)*/,
    3,-10, 5,6/*mean (0.235807), correlation (0.313104)*/,
    2,-4, 3,-10/*mean (0.00494827), correlation (0.344948)*/,
    -13,0, -13,5/*mean (0.0549145), correlation (0.344675)*/,
    -13,-7, -12,12/*mean (0.103385), correlation (0.342715)*/,
    -13,3, -11,8/*mean (0.134222), correlation (0.322922)*/,
    -7,12, -4,7/*mean (0.153284), correlation (0.337061)*/,
    6,-10, 12,8/*mean (0.154881), correlation (0.329257)*/,
    -9,-1, -7,-6/*mean (0.200967), correlation (0.33312)*/,
    -2,-5, 0,12/*mean (0.201518), correlation (0.340635)*/,
    -12,5, -7,5/*mean (0.207805), correlation (0.335631)*/,
    3,-10, 8,-13/*mean (0.224438), correlation (0.34504)*/,
    -7,-7, -4,5/*mean (0.239361), correlation (0.338053)*/,
    -3,-2, -1,-7/*mean (0.240744), correlation (0.344322)*/,
    2,9, 5,-11/*mean (0.242949), correlation (0.34145)*/,
    -11,-13, -5,-13/*mean (0.244028), correlation (0.336861)*/,
    -1,6, 0,-1/*mean (0.247571), correlation (0.343684)*/,
    5,-3, 5,2/*mean (0.000697256), correlation (0.357265)*/,
    -4,-13, -4,12/*mean (0.00213675), correlation (0.373827)*/,
    -9,-6, -9,6/*mean (0.0126856), correlation (0.373938)*/,
    -12,-10, -8,-4/*mean (0.0152497), correlation (0.364237)*/,
    10,2, 12,-3/*mean (0.0299933), correlation (0.345292)*/,
    7,12, 12,12/*mean (0.0307242), correlation (0.366299)*/,
    -7,-13, -6,5/*mean (0.0534975), correlation (0.368357)*/,
    -4,9, -3,4/*mean (0.099865), correlation (0.372276)*/,
    7,-1, 12,2/*mean (0.117083), correlation (0.364529)*/,
    -7,6, -5,1/*mean (0.126125), correlation (0.369606)*/,
    -13,11, -12,5/*mean (0.130364), correlation (0.358502)*/,
    -3,7, -2,-6/*mean (0.131691), correlation (0.375531)*/,
    7,-8, 12,-7/*mean (0.160166), correlation (0.379508)*/,
    -13,-7, -11,-12/*mean (0.167848), correlation (0.353343)*/,
    1,-3, 12,12/*mean (0.183378), correlation (0.371916)*/,
    2,-6, 3,0/*mean (0.228711), correlation (0.371761)*/,
    -4,3, -2,-13/*mean (0.247211), correlation (0.364063)*/,
    -1,-13, 1,9/*mean (0.249325), correlation (0.378139)*/,
    7,1, 8,-6/*mean (0.000652272), correlation (0.411682)*/,
    1,-1, 3,12/*mean (0.00248538), correlation (0.392988)*/,
    9,1, 12,6/*mean (0.0206815), correlation (0.386106)*/,
    -1,-9, -1,3/*mean (0.0364485), correlation (0.410752)*/,
    -13,-13, -10,5/*mean (0.0376068), correlation (0.398374)*/,
    7,7, 10,12/*mean (0.0424202), correlation (0.405663)*/,
    12,-5, 12,9/*mean (0.0942645), correlation (0.410422)*/,
    6,3, 7,11/*mean (0.1074), correlation (0.413224)*/,
    5,-13, 6,10/*mean (0.109256), correlation (0.408646)*/,
    2,-12, 2,3/*mean (0.131691), correlation (0.416076)*/,
    3,8, 4,-6/*mean (0.165081), correlation (0.417569)*/,
    2,6, 12,-13/*mean (0.171874), correlation (0.408471)*/,
    9,-12, 10,3/*mean (0.175146), correlation (0.41296)*/,
    -8,4, -7,9/*mean (0.183682), correlation (0.402956)*/,
    -11,12, -4,-6/*mean (0.184672), correlation (0.416125)*/,
    1,12, 2,-8/*mean (0.191487), correlation (0.386696)*/,
    6,-9, 7,-4/*mean (0.192668), correlation (0.394771)*/,
    2,3, 3,-2/*mean (0.200157), correlation (0.408303)*/,
    6,3, 11,0/*mean (0.204588), correlation (0.411762)*/,
    3,-3, 8,-8/*mean (0.205904), correlation (0.416294)*/,
    7,8, 9,3/*mean (0.213237), correlation (0.409306)*/,
    -11,-5, -6,-4/*mean (0.243444), correlation (0.395069)*/,
    -10,11, -5,10/*mean (0.247672), correlation (0.413392)*/,
    -5,-8, -3,12/*mean (0.24774), correlation (0.411416)*/,
    -10,5, -9,0/*mean (0.00213675), correlation (0.454003)*/,
    8,-1, 12,-6/*mean (0.0293635), correlation (0.455368)*/,
    4,-6, 6,-11/*mean (0.0404971), correlation (0.457393)*/,
    -10,12, -8,7/*mean (0.0481107), correlation (0.448364)*/,
    4,-2, 6,7/*mean (0.050641), correlation (0.455019)*/,
    -2,0, -2,12/*mean (0.0525978), correlation (0.44338)*/,
    -5,-8, -5,2/*mean (0.0629667), correlation (0.457096)*/,
    7,-6, 10,12/*mean (0.0653846), correlation (0.445623)*/,
    -9,-13, -8,-8/*mean (0.0858749), correlation (0.449789)*/,
    -5,-13, -5,-2/*mean (0.122402), correlation (0.450201)*/,
    8,-8, 9,-13/*mean (0.125416), correlation (0.453224)*/,
    -9,-11, -9,0/*mean (0.130128), correlation (0.458724)*/,
    1,-8, 1,-2/*mean (0.132467), correlation (0.440133)*/,
    7,-4, 9,1/*mean (0.132692), correlation (0.454)*/,
    -2,1, -1,-4/*mean (0.135695), correlation (0.455739)*/,
    11,-6, 12,-11/*mean (0.142904), correlation (0.446114)*/,
    -12,-9, -6,4/*mean (0.146165), correlation (0.451473)*/,
    3,7, 7,12/*mean (0.147627), correlation (0.456643)*/,
    5,5, 10,8/*mean (0.152901), correlation (0.455036)*/,
    0,-4, 2,8/*mean (0.167083), correlation (0.459315)*/,
    -9,12, -5,-13/*mean (0.173234), correlation (0.454706)*/,
    0,7, 2,12/*mean (0.18312), correlation (0.433855)*/,
    -1,2, 1,7/*mean (0.185504), correlation (0.443838)*/,
    5,11, 7,-9/*mean (0.185706), correlation (0.451123)*/,
    3,5, 6,-8/*mean (0.188968), correlation (0.455808)*/,
    -13,-4, -8,9/*mean (0.191667), correlation (0.459128)*/,
    -5,9, -3,-3/*mean (0.193196), correlation (0.458364)*/,
    -4,-7, -3,-12/*mean (0.196536), correlation (0.455782)*/,
    6,5, 8,0/*mean (0.1972), correlation (0.450481)*/,
    -7,6, -6,12/*mean (0.199438), correlation (0.458156)*/,
    -13,6, -5,-2/*mean (0.211224), correlation (0.449548)*/,
    1,-10, 3,10/*mean (0.211718), correlation (0.440606)*/,
    4,1, 8,-4/*mean (0.213034), correlation (0.443177)*/,
    -2,-2, 2,-13/*mean (0.234334), correlation (0.455304)*/,
    2,-12, 12,12/*mean (0.235684), correlation (0.443436)*/,
    -2,-13, 0,-6/*mean (0.237674), correlation (0.452525)*/,
    4,1, 9,3/*mean (0.23962), correlation (0.444824)*/,
    -6,-10, -3,-5/*mean (0.248459), correlation (0.439621)*/,
    -3,-13, -1,1/*mean (0.249505), correlation (0.456666)*/,
    7,5, 12,-11/*mean (0.00119208), correlation (0.495466)*/,
    4,-2, 5,-7/*mean (0.00372245), correlation (0.484214)*/,
    -13,9, -9,-5/*mean (0.00741116), correlation (0.499854)*/,
    7,1, 8,6/*mean (0.0208952), correlation (0.499773)*/,
    7,-8, 7,6/*mean (0.0220085), correlation (0.501609)*/,
    -7,-4, -7,1/*mean (0.0233806), correlation (0.496568)*/,
    -8,11, -7,-8/*mean (0.0236505), correlation (0.489719)*/,
    -13,6, -12,-8/*mean (0.0268781), correlation (0.503487)*/,
    2,4, 3,9/*mean (0.0323324), correlation (0.501938)*/,
    10,-5, 12,3/*mean (0.0399235), correlation (0.494029)*/,
    -6,-5, -6,7/*mean (0.0420153), correlation (0.486579)*/,
    8,-3, 9,-8/*mean (0.0548021), correlation (0.484237)*/,
    2,-12, 2,8/*mean (0.0616622), correlation (0.496642)*/,
    -11,-2, -10,3/*mean (0.0627755), correlation (0.498563)*/,
    -12,-13, -7,-9/*mean (0.0829622), correlation (0.495491)*/,
    -11,0, -10,-5/*mean (0.0843342), correlation (0.487146)*/,
    5,-3, 11,8/*mean (0.0929937), correlation (0.502315)*/,
    -2,-13, -1,12/*mean (0.113327), correlation (0.48941)*/,
    -1,-8, 0,9/*mean (0.132119), correlation (0.467268)*/,
    -13,-11, -12,-5/*mean (0.136269), correlation (0.498771)*/,
    -10,-2, -10,11/*mean (0.142173), correlation (0.498714)*/,
    -3,9, -2,-13/*mean (0.144141), correlation (0.491973)*/,
    2,-3, 3,2/*mean (0.14892), correlation (0.500782)*/,
    -9,-13, -4,0/*mean (0.150371), correlation (0.498211)*/,
    -4,6, -3,-10/*mean (0.152159), correlation (0.495547)*/,
    -4,12, -2,-7/*mean (0.156152), correlation (0.496925)*/,
    -6,-11, -4,9/*mean (0.15749), correlation (0.499222)*/,
    6,-3, 6,11/*mean (0.159211), correlation (0.503821)*/,
    -13,11, -5,5/*mean (0.162427), correlation (0.501907)*/,
    11,11, 12,6/*mean (0.16652), correlation (0.497632)*/,
    7,-5, 12,-2/*mean (0.169141), correlation (0.484474)*/,
    -1,12, 0,7/*mean (0.169456), correlation (0.495339)*/,
    -4,-8, -3,-2/*mean (0.171457), correlation (0.487251)*/,
    -7,1, -6,7/*mean (0.175), correlation (0.500024)*/,
    -13,-12, -8,-13/*mean (0.175866), correlation (0.497523)*/,
    -7,-2, -6,-8/*mean (0.178273), correlation (0.501854)*/,
    -8,5, -6,-9/*mean (0.181107), correlation (0.494888)*/,
    -5,-1, -4,5/*mean (0.190227), correlation (0.482557)*/,
    -13,7, -8,10/*mean (0.196739), correlation (0.496503)*/,
    1,5, 5,-13/*mean (0.19973), correlation (0.499759)*/,
    1,0, 10,-13/*mean (0.204465), correlation (0.49873)*/,
    9,12, 10,-1/*mean (0.209334), correlation (0.49063)*/,
    5,-8, 10,-9/*mean (0.211134), correlation (0.503011)*/,
    -1,11, 1,-13/*mean (0.212), correlation (0.499414)*/,
    -9,-3, -6,2/*mean (0.212168), correlation (0.480739)*/,
    -1,-10, 1,12/*mean (0.212731), correlation (0.502523)*/,
    -13,1, -8,-10/*mean (0.21327), correlation (0.489786)*/,
    8,-11, 10,-6/*mean (0.214159), correlation (0.488246)*/,
    2,-13, 3,-6/*mean (0.216993), correlation (0.50287)*/,
    7,-13, 12,-9/*mean (0.223639), correlation (0.470502)*/,
    -10,-10, -5,-7/*mean (0.224089), correlation (0.500852)*/,
    -10,-8, -8,-13/*mean (0.228666), correlation (0.502629)*/,
    4,-6, 8,5/*mean (0.22906), correlation (0.498305)*/,
    3,12, 8,-13/*mean (0.233378), correlation (0.503825)*/,
    -4,2, -3,-3/*mean (0.234323), correlation (0.476692)*/,
    5,-13, 10,-12/*mean (0.236392), correlation (0.475462)*/,
    4,-13, 5,-1/*mean (0.236842), correlation (0.504132)*/,
    -9,9, -4,3/*mean (0.236977), correlation (0.497739)*/,
    0,3, 3,-9/*mean (0.24314), correlation (0.499398)*/,
    -12,1, -6,1/*mean (0.243297), correlation (0.489447)*/,
    3,2, 4,-8/*mean (0.00155196), correlation (0.553496)*/,
    -10,-10, -10,9/*mean (0.00239541), correlation (0.54297)*/,
    8,-13, 12,12/*mean (0.0034413), correlation (0.544361)*/,
    -8,-12, -6,-5/*mean (0.003565), correlation (0.551225)*/,
    2,2, 3,7/*mean (0.00835583), correlation (0.55285)*/,
    10,6, 11,-8/*mean (0.00885065), correlation (0.540913)*/,
    6,8, 8,-12/*mean (0.0101552), correlation (0.551085)*/,
    -7,10, -6,5/*mean (0.0102227), correlation (0.533635)*/,
    -3,-9, -3,9/*mean (0.0110211), correlation (0.543121)*/,
    -1,-13, -1,5/*mean (0.0113473), correlation (0.550173)*/,
    -3,-7, -3,4/*mean (0.0140913), correlation (0.554774)*/,
    -8,-2, -8,3/*mean (0.017049), correlation (0.55461)*/,
    4,2, 12,12/*mean (0.01778), correlation (0.546921)*/,
    2,-5, 3,11/*mean (0.0224022), correlation (0.549667)*/,
    6,-9, 11,-13/*mean (0.029161), correlation (0.546295)*/,
    3,-1, 7,12/*mean (0.0303081), correlation (0.548599)*/,
    11,-1, 12,4/*mean (0.0355151), correlation (0.523943)*/,
    -3,0, -3,6/*mean (0.0417904), correlation (0.543395)*/,
    4,-11, 4,12/*mean (0.0487292), correlation (0.542818)*/,
    2,-4, 2,1/*mean (0.0575124), correlation (0.554888)*/,
    -10,-6, -8,1/*mean (0.0594242), correlation (0.544026)*/,
    -13,7, -11,1/*mean (0.0597391), correlation (0.550524)*/,
    -13,12, -11,-13/*mean (0.0608974), correlation (0.55383)*/,
    6,0, 11,-13/*mean (0.065126), correlation (0.552006)*/,
    0,-1, 1,4/*mean (0.074224), correlation (0.546372)*/,
    -13,3, -9,-2/*mean (0.0808592), correlation (0.554875)*/,
    -9,8, -6,-3/*mean (0.0883378), correlation (0.551178)*/,
    -13,-6, -8,-2/*mean (0.0901035), correlation (0.548446)*/,
    5,-9, 8,10/*mean (0.0949843), correlation (0.554694)*/,
    2,7, 3,-9/*mean (0.0994152), correlation (0.550979)*/,
    -1,-6, -1,-1/*mean (0.10045), correlation (0.552714)*/,
    9,5, 11,-2/*mean (0.100686), correlation (0.552594)*/,
    11,-3, 12,-8/*mean (0.101091), correlation (0.532394)*/,
    3,0, 3,5/*mean (0.101147), correlation (0.525576)*/,
    -1,4, 0,10/*mean (0.105263), correlation (0.531498)*/,
    3,-6, 4,5/*mean (0.110785), correlation (0.540491)*/,
    -13,0, -10,5/*mean (0.112798), correlation (0.536582)*/,
    5,8, 12,11/*mean (0.114181), correlation (0.555793)*/,
    8,9, 9,-6/*mean (0.117431), correlation (0.553763)*/,
    7,-4, 8,-12/*mean (0.118522), correlation (0.553452)*/,
    -10,4, -10,9/*mean (0.12094), correlation (0.554785)*/,
    7,3, 12,4/*mean (0.122582), correlation (0.555825)*/,
    9,-7, 10,-2/*mean (0.124978), correlation (0.549846)*/,
    7,0, 12,-2/*mean (0.127002), correlation (0.537452)*/,
    -1,-6, 0,-11/*mean (0.127148), correlation (0.547401)*/
};
/**
 * 这个是类ORBextractor的带参构造函数，并且使用初始化列表对该类中的这5个变量赋值
 * ORB特征点提取器
 * nt _nfeatures       特征点个数，单目mono_kitti中该值为2000
 * float _scaleFactor  金字塔中相邻层图像的比例系数1.2
 * int _nlevels        构造金字塔的层数 8
 * int _iniThFAST      检测FAST角点的阈值 20
 * int _minThFAST      在iniThFAST没有检测到角点的前提下，降低的阈值 7
 * 
*/
ORBextractor::ORBextractor(int _nfeatures, float _scaleFactor, int _nlevels,
         int _iniThFAST, int _minThFAST):
    nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
    iniThFAST(_iniThFAST), minThFAST(_minThFAST)
{
    //计算每一层相对于原始图片的放大倍数
    //mvScaleFactor是用来存储金字塔中每层图像对应的尺度因子的vector变量
    mvScaleFactor.resize(nlevels);
    //mvLevelSigma2是该层尺度因子的平方，有点表示面积的意思，具体含义等遇到了再解释
    mvLevelSigma2.resize(nlevels);
    mvScaleFactor[0]=1.0f;
    mvLevelSigma2[0]=1.0f;
    /**
     * 计算金字塔中每一层图像对应的尺度因子和尺度因子的平方。
     * 可以发现金字塔中第0层的尺度因子是1,然后每向上高一层，图像的尺度因子是在上一层图像的尺度因子 
     * 上乘以scaleFactor，在本工程下该值为1.2
     * 1   1*1.2   1*1.2*1.2    1*1.2*1.2*1.2   ...
     *  
    */
    for(int i=1; i<nlevels; i++)
    {
        mvScaleFactor[i]=mvScaleFactor[i-1]*scaleFactor;
        mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
    }
    //计算每一层相对于原始图片放大倍数的逆，为正向尺度的倒数
    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);
    for(int i=0; i<nlevels; i++)
    {
        //1  1/1.2  1/1.2^2  1/1.2^3  1/1.2^4  1/1.2^5  1/1.2^6  1/1.2^7
        mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
        mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
    }
    //mvImagePyramid是一个存储金字塔图像的vector，vector中的每一个元素是用Mat数据类型表示的图
    //像, std::vector<cv::Mat> mvImagePyramid;
    mvImagePyramid.resize(nlevels);
    //mnFeaturesPerLevel是一个存储金字塔中每层图像应该提取的特征点的个数
    mnFeaturesPerLevel.resize(nlevels);
    //那在这里factor = 1/1.2;
    float factor = 1.0f / scaleFactor;
    //nDesiredFeaturesPerScale是根据总的要在整幅图像中提取的特征点数nFeatures(在这里是2×2000)
    //根据下面的程序可以得知nDesiredFeaturesPerScale是第0层图像上应该提取的特征点的个数
    //nDesiredFeaturesPerScale=4000*(1-1.0/1.2)/(1-(1/1.2)^8)=868.698
    float nDesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));

    int sumFeatures = 0;
    for( int level = 0; level < nlevels-1; level++ )
    {
        //cvRound()：返回跟参数最接近的整数值，即四舍五入。即对上层所包含的特征点数进行求整并存储
        //那么mnFeaturesPerLevel[0]就是上面计算出来的这个值喽，实际值cvRound后为869
        mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
        //计算除去最后一层的其他层所有特征点总数
        //由于四舍五入的原因将实际计算出的数值加到一起与预设的4000比较
        //869+724+603+503+419+349+291=3758
        sumFeatures += mnFeaturesPerLevel[level];
        //下一层所含有的特征点数
        //层数越高则需要提取的特征的个数就越少，并且相邻层图像上提取的特征点的个数存在1.2倍关系
        //这里是我计算的结果
        //869+724+603+503+419+349+291=3758
        //factor=1/1.2
        nDesiredFeaturesPerScale *= factor;
    }
    //最大层需要的特征点数=max(总共需要的特征点数(4000)-其他所有层的特征点总合(3758), 0) = 242
    //看到这里就会知道上面为啥要把计算出来的特征点个数求和的原因了，就是来决定最后一层图像应该提取的特征的个数了，根据计算最后一层要提取242个特征值
    mnFeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);
    //准备计算关键点keypoint的brief描述子时所需要的pattern
    //这个pattern一共有512个点对；
    const int npoints = 512;

    //bit_pattern_31_是一个1024维的数组，其信息是512对点对的相对中心点的像素坐标
    //这里将bit_pattern_31_里的信息以Point的形式存储在了std::vector<cv::Point> pattern里;
    //最后pattern储存了512个Point
    const Point* pattern0 = (const Point*)bit_pattern_31_;
    //复制训练的模板pattern0到pattern中
    std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));

    //This is for orientation
    // pre-compute the end of a row in a circular patch
    //这里是为了计算关键点方向的准备工作
    //我们是要在以关键点keypoint像素坐标点为中心半径为HALF_PATCH_SIZE的patch圆内计算关键点keypoint的方向。
    //那如何描述这个patch圆的范围呢？这里选择的是储存不同v所对应的的umax来描述这个patch圆的范围。
    umax.resize(HALF_PATCH_SIZE + 1);
    //cvFloor()：返回不大于参数的最大整数值，即向下取整
    //这里计算得vmax=11
    int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
    //cvCeil()：返回不小于参数的最小整数值，即向上取整
    //这里计算得vmin=11
    int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
    //hp2=225
    const double hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;
    //v从0到11依次遍历，umax中存储12个数：15 15 15 15 14 14 14 13 13 12 11 10
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt(hp2 - v * v));

    // Make sure we are symmetric
    //下面是算当v=vmin至HALF_PATCH_SIZE时的umax[v]，那就是v从15到11
    //计算得umax[15]-umax[11]依次是3 6 8 9 10 
    //至此，也就算出了umax[16]中所有数据，依次是：15 15 15 15 14 14 14 13 13 12 11 10 9 8 6 3
    for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }
}
/**
 * 计算关键点的方向
*/
static void computeOrientation(const Mat& image, vector<KeyPoint>& keypoints, const vector<int>& umax)
{
    //遍历关键点
    for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
         keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
    {
        keypoint->angle = IC_Angle(image, keypoint->pt, umax);
    }
}
/**
 * 一个节点分割成4个子节点
 * 
*/
void ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4)
{
    const int halfX = ceil(static_cast<float>(UR.x-UL.x)/2);
    const int halfY = ceil(static_cast<float>(BR.y-UL.y)/2);

    //Define boundaries of childs
    n1.UL = UL;
    n1.UR = cv::Point2i(UL.x+halfX,UL.y);
    n1.BL = cv::Point2i(UL.x,UL.y+halfY);
    n1.BR = cv::Point2i(UL.x+halfX,UL.y+halfY);
    n1.vKeys.reserve(vKeys.size());

    n2.UL = n1.UR;
    n2.UR = UR;
    n2.BL = n1.BR;
    n2.BR = cv::Point2i(UR.x,UL.y+halfY);
    n2.vKeys.reserve(vKeys.size());

    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = BL;
    n3.BR = cv::Point2i(n1.BR.x,BL.y);
    n3.vKeys.reserve(vKeys.size());

    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = BR;
    n4.vKeys.reserve(vKeys.size());

    //Associate points to childs
    //根据特征点的坐标来将特征点分配到不同的新结点区域
    for(size_t i=0;i<vKeys.size();i++)
    {
        const cv::KeyPoint &kp = vKeys[i];
        if(kp.pt.x<n1.UR.x)
        {
            if(kp.pt.y<n1.BR.y)
                n1.vKeys.push_back(kp);
            else
                n3.vKeys.push_back(kp);
        }
        else if(kp.pt.y<n1.BR.y)
            n2.vKeys.push_back(kp);
        else
            n4.vKeys.push_back(kp);
    }
    //最后根据每个结点中分得的特征点的数目来设定bNoMore变量的真假
    //如果此节点包含的关键点数量为一，那么就不分裂
    if(n1.vKeys.size()==1)
        n1.bNoMore = true;
    if(n2.vKeys.size()==1)
        n2.bNoMore = true;
    if(n3.vKeys.size()==1)
        n3.bNoMore = true;
    if(n4.vKeys.size()==1)
        n4.bNoMore = true;

}
/**
 * 利用四叉树来筛选提取的特征点，使得筛选后的特征点数量在预计的范围，并且分布均匀
 * vToDistributeKeys 每个小窗口中提取的关键点
 * minX maxX minY maxY：是该层图像去除了边界的区域
 * N: mnFeaturesPerLevel[i]表示该层图像上应该提取的特征点的个数
 * level 当前提取的关键点是金字塔的第level层
*/
vector<cv::KeyPoint> ORBextractor::DistributeOctTree(const vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                       const int &maxX, const int &minY, const int &maxY, const int &N, const int &level)
{
    // Compute how many initial nodes
    /**
     * 常用的相机kinect v1的分辨率是：640*480 kinect v2的分辨率是：1920*1080
     * 为了尽量使得每一个结点的区域形状接近正方形所以图像的长宽比决定了四叉树根节点的数目
     * 如果使用kinect v1那么只有一个根结点，如果使用kinect v2那么就会有两个根结点
     * */
    //计算根节点数量
    const int nIni = round(static_cast<float>(maxX-minX)/(maxY-minY));
    //计算根节点的X范围，其Y轴范围就是(maxY-minY).hX可以理解为一个根节点所占的宽度
    const float hX = static_cast<float>(maxX-minX)/nIni;
    //新建一个双向链表，用来存储生成的树结点，其实后面大部分操作就是改变这个
    list<ExtractorNode> lNodes;
    //vpIniNodes变量中存储的是结点的地址，vpIniNodes在将vToDistributeKeys分配到父节点中时起索引作用
    vector<ExtractorNode*> vpIniNodes;
    //vpIniNodes的大小先设置成根结点的个数
    vpIniNodes.resize(nIni);
    //遍历每个根节点，计算根节点领域的四个点UL，UR，BL，BR并其存入lNodes和vpIniNodes
    for(int i=0; i<nIni; i++)
    {
        ExtractorNode ni;
        /**
         * 四叉树是每次根据特定条件将一个结点分成四等分，四个区域左上(UL)，右上(UR)，左下(BL)，右下(BR)
         * 
        */
        //cv::Point2i(x,y)创建二维点对象
        ni.UL = cv::Point2i(hX*static_cast<float>(i),0);
        ni.UR = cv::Point2i(hX*static_cast<float>(i+1),0);
        ni.BL = cv::Point2i(ni.UL.x,maxY-minY);
        ni.BR = cv::Point2i(ni.UR.x,maxY-minY);

        //vKeys的大小为在上面的这个根节点范围内总共提取的特征点的个数
        ni.vKeys.reserve(vToDistributeKeys.size());
        //将创建的根节点插入到list lNodes中
        lNodes.push_back(ni);
        /**
         * 将lNodes变量中的最后存储的那个结点的地址存储到vector变量vpIniNodes中,暂时还不知道这个变量做何用
         * vpIniNodes总是把最后插入到lNodes中的结点的地址拿走，然后要为该结点的vKeys成员变量内部添加属于该结点的特征点。
         * */
        vpIniNodes[i] = &lNodes.back();
    }

    //Associate points to childs
    //将vToDistributeKeys按照位置分配在根节点中
    //要一直记得vToDistributeKeys变量中存储的是该层图像中提取的特征点
    //遍历在该层图像上提取的所有特征点
    for(size_t i=0;i<vToDistributeKeys.size();i++)
    {
        const cv::KeyPoint &kp = vToDistributeKeys[i];
        /**
         * 将所有提取的特征点根据坐标位置将其分配到对应的根节点中去
         * 如果使用kinect b=v1那么所有的kp.pt.x都小于hX,所以所有的特征点都被分配到
         * vpIniNodes的第0个元素中存储的结点指针所指向的空间中去了。
         * 到这里明白了这个四叉树的玩法了定义一个list变量，用来存储生成的树节点本身。
         * 定义一个vector变量，用来存储结点的指针，这个指针可以指向该结点区域被分配的特征点的内存
         * list是一个双向链表容器，可高效地进行插入删除元素。
         * vector是将元素置于一个动态数组中，vector可以随机存取元素，在头尾插入数据快，但是从中            
         * 间插入数据很慢正是利用了list和vector的特点，使得我们即可以快速高效的插入删除结点，又可以随机的存取
         * 被分配到某一个结点区域的的特征点。
         * 
         * 根据关键点在图像中的x坐标，将所有的关键点分配到指定节点下的vKeys容器中去去
        */
        vpIniNodes[kp.pt.x/hX]->vKeys.push_back(kp);
    }

    list<ExtractorNode>::iterator lit = lNodes.begin();
    /**
     * 现在lNodes中只放有根节点，变量根节点，如果根节点中关键点个数为1
     * 那么将这个节点的bNoMore设置为true，表示这个节点不能再分裂了。
     * 如果为空，那么就删除这个节点
     * 
    */
    while(lit!=lNodes.end())
    {
        //如果判断在一个结点里面只有一个特征点
        if(lit->vKeys.size()==1)
        {
            //将该结点的bNoMore属性设置为true，表示不再对这个结点进行分割
            lit->bNoMore=true;
            lit++;
        }
        //如果判断这个结点中没有被分配到任何的特征点那么就将这个结点删除
        else if(lit->vKeys.empty())
            lit = lNodes.erase(lit);
        else
            lit++;
    }
    /**
     * lNodes中的结点和 vpIniNodes中的结点指针是同步的，只有在 vpIniNodes中存储的结点中存储了 
     * 特征点，才能根据特征点的数目来决定如何处理这个结点
     * 那如果在lNodes中删除一个没有特征点的结点，那么在 vpIniNodes中对应的这个地址也会被销毁吗？
    */
    //后面大循环结束标志位
    bool bFinish = false;

    int iteration = 0;

    vector<pair<int,ExtractorNode*> > vSizeAndPointerToNode;
    vSizeAndPointerToNode.reserve(lNodes.size()*4);
    //现在lNodes里只有根节点，且每个节点的关键点数量都不为0
    while(!bFinish)
    {
        iteration++;

        int prevSize = lNodes.size();

        lit = lNodes.begin();
        //待分裂的节点数nToExpand
        int nToExpand = 0;

        vSizeAndPointerToNode.clear();
        //遍历双向链表lNodes中的节点
        while(lit!=lNodes.end())
        {
            if(lit->bNoMore)
            {
                // If node only contains one point do not subdivide and continue
                //如果此节点包含的关键点数量为一，那么就不分裂
                lit++;
                continue;
            }
            else
            {
                // If more than one point, subdivide
                //如果此节点关键点数量大于1，那么就分裂此节点，将此节点中的关键点通过DivideNode分配到n1,n2,n3,n4中
                //并且将此被分裂的节点删除
                ExtractorNode n1,n2,n3,n4;
                lit->DivideNode(n1,n2,n3,n4);

                // Add childs if they contain points
                //如果节点n1的关键点数量大于0，将其插入lNodes的头部
                //如果节点n1的关键点数量大于1，将其插入vSizeAndPointerToNode中表示待分裂，并且接下来待分裂的节点个数做+1操作
                //后面n2,n3,n4以此类推
                if(n1.vKeys.size()>0)
                {
                    lNodes.push_front(n1);                    
                    if(n1.vKeys.size()>1)
                    {
                        nToExpand++;
                        //变量vSizeAndPointerToNode中存储的是每一个结点的地址以及该结点中被分配到的特征点的个数。
                        vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                        //每插入一个结点就要更新list的开始结点的位置
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n2.vKeys.size()>0)
                {
                    lNodes.push_front(n2);
                    if(n2.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n3.vKeys.size()>0)
                {
                    lNodes.push_front(n3);
                    if(n3.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n4.vKeys.size()>0)
                {
                    lNodes.push_front(n4);
                    if(n4.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                lit=lNodes.erase(lit);
                continue;
            }
        }       

        // Finish if there are more nodes than required features
        // or all nodes contain just one point
        //当创建的结点的数目比要求的特征点还要多，或者每个结点中都只有一个特征点的时候就停止分割
        if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
        {
            bFinish = true;
        }
        /**
         * 待分裂的节点数vSizeAndPointerToNode分裂后，一般会增加nToExpand*3，如果增加nToExpand*3后大于N，则进入此分支。
         * 如果现在生成的结点全部进行分割后生成的结点满足大于需求的特征点的数目，但是不继续分割又不能满足大于N的要求时。
         * 这里为什么是乘以3，这里也正好印证了上面所说的当一个结点被分割成四个新的结点时，
         * 这个结点是要被删除的，其实总的结点是增加了3个。
         * 
        */
        else if(((int)lNodes.size()+nToExpand*3)>N)
        {

            while(!bFinish)
            {

                prevSize = lNodes.size();
                //这里将已经创建好的结点放到一个新的容器中
                vector<pair<int,ExtractorNode*> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                vSizeAndPointerToNode.clear();
                //根据vPrevSizeAndPointerToNode.first大小来排序，也就是根据待分裂节点的关键点数量大小来有小到大排序。
                //vPrevSizeAndPointerToNode.second为节点
                /**
                 * 根据结点中被分配到的特征点的数目对结点进行排序,
                 * 这里为何要排序，我们想要的结果是想让尽可能多的特征点均匀的分布在图像上
                 * 如果前面的特征分布相对均匀的结点中的特征点数目已经达到了指标那么就可以将
                 * 后面那些分布密集的特征点去掉了。
                */
                sort(vPrevSizeAndPointerToNode.begin(),vPrevSizeAndPointerToNode.end());
                //倒序遍历vPrevSizeAndPointerToNode，就是先分裂待分裂节点的关键点数量大的
                for(int j=vPrevSizeAndPointerToNode.size()-1;j>=0;j--)
                {
                    ExtractorNode n1,n2,n3,n4;
                    //second其实就是pair中第二个元素，也就是节点
                    vPrevSizeAndPointerToNode[j].second->DivideNode(n1,n2,n3,n4);
                    // Add childs if they contain points
                    if(n1.vKeys.size()>0)
                    {
                        lNodes.push_front(n1);
                        if(n1.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n2.vKeys.size()>0)
                    {
                        lNodes.push_front(n2);
                        if(n2.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n3.vKeys.size()>0)
                    {
                        lNodes.push_front(n3);
                        if(n3.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n4.vKeys.size()>0)
                    {
                        lNodes.push_front(n4);
                        if(n4.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    //删除被分裂的节点
                    lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);
                    //如果所有的结点还没有被分割完就已经达到了大于N的要求那么就直接跳出循环
                    if((int)lNodes.size()>=N)
                        break;
                }

                if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
                    bFinish = true;

            }
        }
    }

    // Retain the best point in each node
    //取出每个节点中响应最大的特征点
    vector<cv::KeyPoint> vResultKeys;
    vResultKeys.reserve(nfeatures);
    //遍历创建的所有结点
    for(list<ExtractorNode>::iterator lit=lNodes.begin(); lit!=lNodes.end(); lit++)
    {
        //获取每个结点下的特征点
        vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
        cv::KeyPoint* pKP = &vNodeKeys[0];
        float maxResponse = pKP->response;
        //在每个结点中找到那个最强壮的特征点进行保存
        for(size_t k=1;k<vNodeKeys.size();k++)
        {
            if(vNodeKeys[k].response>maxResponse)
            {
                pKP = &vNodeKeys[k];
                maxResponse = vNodeKeys[k].response;
            }
        }
        //只将每个结点下最强壮的的特征点保存
        vResultKeys.push_back(*pKP);
    }

    return vResultKeys;
}
/**
 * 功能：提取图像金字塔中各层图像的关键点并储存在allKeypoints里
 * 算法具体步骤描述：
 * 1.先是计算提取关键点的边界，太边缘的地方放弃提取关键点；
 * 2.将图像分割为W*W的小图片，遍历这些分割出来的小图片并提取其关键点；
 * 3.将提取的关键点交给DistributeOctTree()利用四叉树以及非极大值抑制算法进行筛选；
 * 4.计算关键点的相关信息：像素位置，patch尺度，方向，所在高斯金字塔层数；
 * 注意：在分割图片的时候，小窗之间是有重叠的
*/
void ORBextractor::ComputeKeyPointsOctTree(vector<vector<KeyPoint> >& allKeypoints)
{
    allKeypoints.resize(nlevels);
    //暂定的分割窗口的大小
    const float W = 30;
    //对高斯金字塔mvImagePyramid中每层图像提取orb关键点
    for (int level = 0; level < nlevels; ++level)
    {
        //计算边界，在这个边界范围内计算FAST关键点
        const int minBorderX = EDGE_THRESHOLD-3;
        const int minBorderY = minBorderX;
        const int maxBorderX = mvImagePyramid[level].cols-EDGE_THRESHOLD+3;
        const int maxBorderY = mvImagePyramid[level].rows-EDGE_THRESHOLD+3;
        //用这个存储待筛选的orb关键点
        vector<cv::KeyPoint> vToDistributeKeys;
        //reserve()用来保留（扩充）容量，他并不改变容器的有限元素个数；
        //这里nfeatures=4000
        vToDistributeKeys.reserve(nfeatures*10);
        //计算图像宽度和高度
        const float width = (maxBorderX-minBorderX);
        const float height = (maxBorderY-minBorderY);
        //将图像划分为W*W=30*30的小窗口，计算行列各有多少个
        const int nCols = width/W;//列数
        const int nRows = height/W;//行数
        //实际每个小窗口的宽高 宽:wCell 高：hCell
        const int wCell = ceil(width/nCols);
        const int hCell = ceil(height/nRows);
        //按行进行遍历，使用两个for循环遍历每个窗口，首先先计算窗口的四个坐标
        for(int i=0; i<nRows; i++)
        {
            //iniY,maxY为窗口的行上坐标和下坐标
            //计算(iniY,maxY)坐标
            const float iniY =minBorderY+i*hCell;
            //这里多取了6行
            float maxY = iniY+hCell+6;
            //窗口的行上坐标超出边界，则放弃此行
            if(iniY>=maxBorderY-3)
                continue;
            //窗口的行下坐标超出边界，则将窗口的行下坐标设置为边界
            if(maxY>maxBorderY)
                maxY = maxBorderY;
            //遍历每一列
            for(int j=0; j<nCols; j++)
            {
                //iniX,maxX为窗口的列左坐标和右坐标
            	//计算(iniX,maxX)坐标
                const float iniX =minBorderX+j*wCell;
                float maxX = iniX+wCell+6;
                if(iniX>=maxBorderX-6)
                    continue;
                if(maxX>maxBorderX)
                    maxX = maxBorderX;
                //每个小窗里的关键点KeyPoint将存在这里
                vector<cv::KeyPoint> vKeysCell;
                /**
                 * 提取FAST角点放入vKeysCell中
                 * 输入参数:
                 * mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX) level层的图片中在行范围(iniY,maxY),列范围(iniX,maxX)的取件进行角点提取
                 * vKeysCell 储存提取的fast关键点
                 * iniThFAST 提取角点的阈值 这里为20，提取不到的时候改为用minThFAST来提取，值为7
                 * true 是否开启非极大值抑制算法
                 * 
                */
                FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                     vKeysCell,iniThFAST,true);
                // 如果没有找到FAST关键点，就降低阈值重新计算FAST，minThFAST=7
                if(vKeysCell.empty())
                {
                    FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                         vKeysCell,minThFAST,true);
                }
                //如果找到的点不为空，就加入到vToDistributeKeys
                if(!vKeysCell.empty())
                {
                    for(vector<cv::KeyPoint>::iterator vit=vKeysCell.begin(); vit!=vKeysCell.end();vit++)
                    {
                        //pt是关键点的坐标。这里是计算关键点在该层图片中的实际坐标值
                        (*vit).pt.x+=j*wCell;
                        (*vit).pt.y+=i*hCell;
                        vToDistributeKeys.push_back(*vit);
                    }
                }

            }
        }
        //经DistributeOctTree筛选后的关键点存储在这里
        vector<KeyPoint> & keypoints = allKeypoints[level];
        keypoints.reserve(nfeatures);
        //筛选vToDistributeKeys中的关键点存入keypoints中，mnFeaturesPerLevel中存放每层图中的特征点数
        //对获取的特征点进行过滤，只选择反应比较大的关键点
        keypoints = DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX,
                                      minBorderY, maxBorderY,mnFeaturesPerLevel[level], level);
        /**
         * 计算在本层提取出的关键点对应的Patch大小，称为scaledPatchSize 
         * 你想想，本层的图像是缩小的，而你在本层提取的orb特征点，计算orb的方向，描述子的时候根据 
         * 的PATCH大小依旧是PATCH_SIZE。 
         * 而你在本层提取的orb是要恢复到原始图像上去的，所以其特征点的size（代表特征点的尺度信息）需要放大。
        */
        const int scaledPatchSize = PATCH_SIZE*mvScaleFactor[level];

        // Add border to coordinates and scale information
        const int nkps = keypoints.size();
        for(int i=0; i<nkps ; i++)
        {
            /**
             * keypoint各项属性解释如下：
             * pt(x,y):关键点的点坐标；
             * size():该关键点邻域直径大小；
             * angle:角度，表示关键点的方向，值为[零,三百六十)，负值表示不使用。
             * response:响应强度，网络上有如下解释：
             * 1)”The response, by which the strongest keypoints have been selected.”
             * 2)”Responseis indeed an indicator of “how good” (roughly speaking, in terms of corner-ness) a point is. 
             * The higher, the better. The strongest keypoints means the ones that are the best.“
             * octacv:从哪一层金字塔得到的此关键点。
             * class_id:当要对图片进行分类时，用class_id对每个关键点进行区分，默认为-1。
            */
            keypoints[i].pt.x+=minBorderX;
            keypoints[i].pt.y+=minBorderY;
            keypoints[i].octave=level;
            keypoints[i].size = scaledPatchSize;
        }
    }

    // compute orientations
    //遍历每一层图像，计算该层图像上每个特征点的方向
    for (int level = 0; level < nlevels; ++level)
        computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
}

void ORBextractor::ComputeKeyPointsOld(std::vector<std::vector<KeyPoint> > &allKeypoints)
{
    allKeypoints.resize(nlevels);

    float imageRatio = (float)mvImagePyramid[0].cols/mvImagePyramid[0].rows;

    for (int level = 0; level < nlevels; ++level)
    {
        const int nDesiredFeatures = mnFeaturesPerLevel[level];

        const int levelCols = sqrt((float)nDesiredFeatures/(5*imageRatio));
        const int levelRows = imageRatio*levelCols;

        const int minBorderX = EDGE_THRESHOLD;
        const int minBorderY = minBorderX;
        const int maxBorderX = mvImagePyramid[level].cols-EDGE_THRESHOLD;
        const int maxBorderY = mvImagePyramid[level].rows-EDGE_THRESHOLD;

        const int W = maxBorderX - minBorderX;
        const int H = maxBorderY - minBorderY;
        const int cellW = ceil((float)W/levelCols);
        const int cellH = ceil((float)H/levelRows);

        const int nCells = levelRows*levelCols;
        const int nfeaturesCell = ceil((float)nDesiredFeatures/nCells);

        vector<vector<vector<KeyPoint> > > cellKeyPoints(levelRows, vector<vector<KeyPoint> >(levelCols));

        vector<vector<int> > nToRetain(levelRows,vector<int>(levelCols,0));
        vector<vector<int> > nTotal(levelRows,vector<int>(levelCols,0));
        vector<vector<bool> > bNoMore(levelRows,vector<bool>(levelCols,false));
        vector<int> iniXCol(levelCols);
        vector<int> iniYRow(levelRows);
        int nNoMore = 0;
        int nToDistribute = 0;


        float hY = cellH + 6;

        for(int i=0; i<levelRows; i++)
        {
            const float iniY = minBorderY + i*cellH - 3;
            iniYRow[i] = iniY;

            if(i == levelRows-1)
            {
                hY = maxBorderY+3-iniY;
                if(hY<=0)
                    continue;
            }

            float hX = cellW + 6;

            for(int j=0; j<levelCols; j++)
            {
                float iniX;

                if(i==0)
                {
                    iniX = minBorderX + j*cellW - 3;
                    iniXCol[j] = iniX;
                }
                else
                {
                    iniX = iniXCol[j];
                }


                if(j == levelCols-1)
                {
                    hX = maxBorderX+3-iniX;
                    if(hX<=0)
                        continue;
                }


                Mat cellImage = mvImagePyramid[level].rowRange(iniY,iniY+hY).colRange(iniX,iniX+hX);

                cellKeyPoints[i][j].reserve(nfeaturesCell*5);

                FAST(cellImage,cellKeyPoints[i][j],iniThFAST,true);

                if(cellKeyPoints[i][j].size()<=3)
                {
                    cellKeyPoints[i][j].clear();

                    FAST(cellImage,cellKeyPoints[i][j],minThFAST,true);
                }


                const int nKeys = cellKeyPoints[i][j].size();
                nTotal[i][j] = nKeys;

                if(nKeys>nfeaturesCell)
                {
                    nToRetain[i][j] = nfeaturesCell;
                    bNoMore[i][j] = false;
                }
                else
                {
                    nToRetain[i][j] = nKeys;
                    nToDistribute += nfeaturesCell-nKeys;
                    bNoMore[i][j] = true;
                    nNoMore++;
                }

            }
        }


        // Retain by score

        while(nToDistribute>0 && nNoMore<nCells)
        {
            int nNewFeaturesCell = nfeaturesCell + ceil((float)nToDistribute/(nCells-nNoMore));
            nToDistribute = 0;

            for(int i=0; i<levelRows; i++)
            {
                for(int j=0; j<levelCols; j++)
                {
                    if(!bNoMore[i][j])
                    {
                        if(nTotal[i][j]>nNewFeaturesCell)
                        {
                            nToRetain[i][j] = nNewFeaturesCell;
                            bNoMore[i][j] = false;
                        }
                        else
                        {
                            nToRetain[i][j] = nTotal[i][j];
                            nToDistribute += nNewFeaturesCell-nTotal[i][j];
                            bNoMore[i][j] = true;
                            nNoMore++;
                        }
                    }
                }
            }
        }

        vector<KeyPoint> & keypoints = allKeypoints[level];
        keypoints.reserve(nDesiredFeatures*2);

        const int scaledPatchSize = PATCH_SIZE*mvScaleFactor[level];

        // Retain by score and transform coordinates
        for(int i=0; i<levelRows; i++)
        {
            for(int j=0; j<levelCols; j++)
            {
                vector<KeyPoint> &keysCell = cellKeyPoints[i][j];
                KeyPointsFilter::retainBest(keysCell,nToRetain[i][j]);
                if((int)keysCell.size()>nToRetain[i][j])
                    keysCell.resize(nToRetain[i][j]);


                for(size_t k=0, kend=keysCell.size(); k<kend; k++)
                {
                    keysCell[k].pt.x+=iniXCol[j];
                    keysCell[k].pt.y+=iniYRow[i];
                    keysCell[k].octave=level;
                    keysCell[k].size = scaledPatchSize;
                    keypoints.push_back(keysCell[k]);
                }
            }
        }

        if((int)keypoints.size()>nDesiredFeatures)
        {
            KeyPointsFilter::retainBest(keypoints,nDesiredFeatures);
            keypoints.resize(nDesiredFeatures);
        }
    }

    // and compute orientations
    //为了使得提取的特征点具有旋转不变性，需要计算每个特征点的方向。
    //方法是计算以特征点为中心以像素为权值的圆形区域上的重心，以中心和重心的连线作为该特征点的方向。
    for (int level = 0; level < nlevels; ++level)
        computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
}
/**
 * image       图像金字塔中每一层里的图像
 * keypoints   图像金字塔中每一层图像里的关键点
 * descriptors 出参，需要计算的描述子
 * pattern     计算描述子需要的模式
 * 
*/
static void computeDescriptors(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors,
                               const vector<Point>& pattern)
{
    //Mat::zeros第一个参数是行数，第二个列数，第三个类型
    descriptors = Mat::zeros((int)keypoints.size(), 32, CV_8UC1);

    for (size_t i = 0; i < keypoints.size(); i++)
        //descriptors.ptr((int)i)用于获取descriptors的第i行
        computeOrbDescriptor(keypoints[i], image, &pattern[0], descriptors.ptr((int)i));
}
/**
 * 重载了括号运算符，对输入的图像计算关键点和描述子
 * 具体流程：
 * 1.对于输入的图片，计算其图像金字塔；
 * 2.提取图像金字塔中各层图像的关键点；
 * 3.计算提取出的关键点对应的描述子；
 * _image:获取的灰度图像
 * _mask:掩码
 * 传入引用参数 _keypoints, _descriptors 用于存储计算得到的特征点及其描述子
*/
void ORBextractor::operator()( InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints,
                      OutputArray _descriptors)
{ 
    if(_image.empty())
        return;

    Mat image = _image.getMat();
    //判断通道是否为灰度图
    assert(image.type() == CV_8UC1 );

    // Pre-compute the scale pyramid
    //计算图像高斯金字塔
    ComputePyramid(image);
    //vector类型的allKeyPonts里每个元素又是一个类型为KeyPoint的vector
    vector < vector<KeyPoint> > allKeypoints;
    //对金字塔图像进行角点检测，提取图像金字塔中各层图像的关键点并储存在allKeypoints里
    ComputeKeyPointsOctTree(allKeypoints);
    //ComputeKeyPointsOld(allKeypoints);

    Mat descriptors;
    //算出实际提取出来的关键点数量nkeypoints
    //新建descriptors对象，特征点的描述子将存在这里
    //descriptors就好似nkeypoints*32维的矩阵，矩阵里存着一个uchar，其单位为8bit
    //这样对于每个关键点，就对应有32*8=256bit的二进制向量作为描述子
    int nkeypoints = 0;
    for (int level = 0; level < nlevels; ++level)
        nkeypoints += (int)allKeypoints[level].size();
    if( nkeypoints == 0 )
        _descriptors.release();
    else
    {
        //为_descriptors分配空间
        _descriptors.create(nkeypoints, 32, CV_8U);
        descriptors = _descriptors.getMat();
    }

    _keypoints.clear();
    _keypoints.reserve(nkeypoints);

    int offset = 0;
    //遍历高斯金字塔每层，计算其提取出的关键点的描述子储存在descriptors里
    for (int level = 0; level < nlevels; ++level)
    {
        vector<KeyPoint>& keypoints = allKeypoints[level];
        int nkeypointsLevel = (int)keypoints.size();

        if(nkeypointsLevel==0)
            continue;

        // preprocess the resized image
        //获取金字塔中level层的图像
        Mat workingMat = mvImagePyramid[level].clone();
        //使用高斯模糊
        GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);

        // Compute the descriptors
        //计算描述子，其计算所需的点对分布采用的是高斯分布，储存在pattern里
        Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
        //计算所得描述子存放在desc中
        computeDescriptors(workingMat, keypoints, desc, pattern);
        //修改offset的值
        offset += nkeypointsLevel;

        // Scale keypoint coordinates
        // 对关键点的位置做尺度恢复，恢复到原图的位置
        if (level != 0)
        {
            //mvScaleFactor值为：1   1*1.2  1.2^2  1.2^3  1.2^4  1.2^5  1.2^6  1.2^7
            float scale = mvScaleFactor[level]; //getScale(level, firstLevel, scaleFactor);
            for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                 keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
                //将关键点的坐标恢复到原始图像中的位置上
                keypoint->pt *= scale;
        }
        // And add the keypoints to the output
        //将恢复完坐标的关键点插入到_keypoints中，所有金字塔层遍历完后，_keypoints中就存储了所有的图像关键点
        _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
    }
}
/**
 * 建立图像金字塔
 * 将原始图像一级级缩小并依次存在mvImagePyramid里
 * 
*/
void ORBextractor::ComputePyramid(cv::Mat image)
{
    // 计算n个level尺度的图片
    for (int level = 0; level < nlevels; ++level)
    {
        //获取缩放尺度,mvInvScaleFactor的值为1  1/1.2  1/1.2^2  1/1.2^3  1/1.2^4  1/1.2^5  1/1.2^6  1/1.2^7
        float scale = mvInvScaleFactor[level];
        //计算当前层图片尺寸
        Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));
        //裁剪前的图片加上边框
        Size wholeSize(sz.width + EDGE_THRESHOLD*2, sz.height + EDGE_THRESHOLD*2);
        //新建一个temp，大小为wholeSize
        Mat temp(wholeSize, image.type()), masktemp;
        //从temp中裁剪（起点为EDGE_THRESHOLD, EDGE_THRESHOLD，大小为sz.width, 			
	    //sz.height）存入mvImagePyramid[level]中，这里就将level层的图片裁剪存入完成了
        mvImagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

        // Compute the resized image
        if( level != 0 )
        {
            //从上一级图像mvImagePyramid[level-1]中缩小图像至sz大小，并存入mvImagePyramid[level]
            resize(mvImagePyramid[level-1], mvImagePyramid[level], sz, 0, 0, INTER_LINEAR);
            /**
             * level!=0时调用调用copyMakeBorder毫无意义啊，上边的resize函数已经完成了此功能
            */
            copyMakeBorder(mvImagePyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           BORDER_REFLECT_101+BORDER_ISOLATED);            
        }
        else
        {
            //将图片image上下左右边界扩充EDGE_THRESHOLD个像素，放入temp中，temp为输出
            /**
             * 函数原型：
             * void copyMakeBorder( const Mat& src, Mat& dst,
             * int top, int bottom, int left, int right,
             * int borderType, const Scalar& value=Scalar())
            */
            copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           BORDER_REFLECT_101);
            //BORDER_REFLECT_101:以最边缘像素为轴，进行对称填充
            //相关学习：https://blog.csdn.net/guyuealian/article/details/78113325          
        }
    }
    /**
     * 总结：
     * 1.先用resize()将图片缩放至sz大小，放入 mvImagePyramid[i]；
     * 2.接着用copyMakeBorder()扩展边界至wholeSize大小放入temp；
     * 3.注意temp和mvImagePyramid[i]公用相同数据区，改变temp会改变mvImagePyramid[i]。
    */
}

} //namespace ORB_SLAM
