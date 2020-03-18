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


#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>

#include<mutex>


using namespace std;

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];
    /**
     * 1.构造相机内参矩阵
     * | fx 0  cx |
     * | f0 fy cy |
     * | 0  0  1  |
    */
    cv::Mat K = cv::Mat::eye(3,3,CV_32F);//创建了3*3的矩阵，元素使用32位单精度浮点型
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);//mK中存放的是相机的内参矩阵
    //2.创建了4*1的矩阵，用于存放相机去畸变的参数。mono_kitti不需要去畸变，所以这些参数都为0
    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);//mDistCoef中存放的是相机图片去畸变的参数

    mbf = fSettings["Camera.bf"];
    //Camera frames per second
    float fps = fSettings["Camera.fps"];
    if(fps==0)//从配置文件中读取的fps=10.0
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;

    //3.读取颜色序列0: BGR, 1: RGB.配置文件中读取的是1
    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters
    //4.加载金字塔图像提取器的参数
    int nFeatures = fSettings["ORBextractor.nFeatures"];//每张图片中提取的ORB特征数(2000)
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];//ORB Extractor: Scale factor between levels in the scale pyramid(1.2)
    int nLevels = fSettings["ORBextractor.nLevels"];//ORB Extractor: Number of levels in the scale pyramid(8)
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];//(10)
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];//(5)

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
    //5.构建特征提取器。单目传入的特征数为其他的2倍，这里是2*2000=4000
    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}


cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}

/**
 * 函数功能：1.将图像转为mImGray并初始化mCurrentFrame;
 *         2.进行tracking过程，输出世界坐标系到该帧相机坐标系的变换矩阵
 * im:       输入图像
 * timestamp:时间戳
 * 
*/
cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    mImGray = im;
    //1：将RGB或RGBA图像转变为灰度图
    if(mImGray.channels()==3)
    {
        //mbRGB从配置文件中读取：Camera.RGB: 1
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }
    //2：使用生成的灰度图构造Frame
    //没有成功初始化的前一个状态就是NO_IMAGES_YET，调用Tracking构造函数的时候赋给mState的值就是NO_IMAGES_YET
    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    else
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    //3：跟踪，也算是Tracking线程的主方法
    Track();

    return mCurrentFrame.mTcw.clone();
}
/**
 * 相当于Tracking线程中的主要方法
 * track包含两部分：估计运动、跟踪局部地图
 * tracking过程的状态机：
 * SYSTEM_NOT_READY=-1,
 * NO_IMAGES_YET=0,
 * NOT_INITIALIZED=1,
 * OK=2,
 * LOST=3
 * 如果图像复位过、或者第一次运行，则为NO_IMAGE_YET状态
*/
void Tracking::Track()
{
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }
    //mLastProcessedState存储了Tracking最新的状态，用于FrameDrawer中的绘制
    mLastProcessedState=mState;
    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
    //1.初始化：没有初始化的话会先进入初始化分支
    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD)
            StereoInitialization();
        else
            //1.单目相机初始化
            MonocularInitialization();

        mpFrameDrawer->Update(this);

        if(mState!=OK)
            return;
    }
    else//2.跟踪估计相机位姿：初始化完成后会进入跟踪分支
    {
        // System is initialized. Track Frame.
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        //使用运动模型或者重定位来初始化相机位姿
        //在viewer中有个开关menuLocalizationMode，由它控制是否ActivateLocalizationMode，并最终管控mbOnlyTracking
        //mbOnlyTracking等于false表示正常VO模式（有地图更新），mbOnlyTracking等于true表示用户手动选择定位模式
        if(!mbOnlyTracking)
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.
            //正常初始化成功
            if(mState==OK)
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                //检查并更新上一帧被替换的MapPoints,更新Fuse函数和SearchAndFuse函数替换的MapPoints
                CheckReplacedInLastFrame();
                //上一帧速度为0 || 当前帧ID与重定位帧ID之间相差大于2。跟踪上一帧或者参考帧或者重定位
                if(mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId+2)
                {
                    /**
                     * 将上一帧的位姿作为当前帧的初始位姿,通过BoW的方式在参考帧中找当前帧特征点的匹配点,
                     * 优化每个特征点对应3D点重投影误差即可得到当前帧的位姿
                    */
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    /**
                     * 根据恒速模型设定当前帧的初始位姿,通过投影的方式在参考帧中找当前帧特征点的匹配点,优化每个特征点所对应3D点的投影误差即可得到位姿
                    */
                    bOK = TrackWithMotionModel();
                    if(!bOK)
                        //TrackReferenceKeyFrame是跟踪参考帧，不能根据固定运动速度模型预测当前帧的位姿态，通过bow加速匹配（SearchByBow）最后通过优化得到位姿
                        bOK = TrackReferenceKeyFrame();
                }
            }
            else
            {
                //重定位,BOW搜索，PnP求解位姿
                bOK = Relocalization();
            }
        }
        else//只跟踪不定位，不插入关键帧，局部地图不工作
        {
            // Localization Mode: Local Mapping is deactivated
            if(mState==LOST)
            {
                //丢失了需要重定位
                bOK = Relocalization();
            }
            else
            {
                //mbVO在Tracking线程初始化的时候初始值为false
                if(!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map
                    //若上一帧有速度，使用TrackWithMotionModel
                    //若上一帧没有速度，使用TrackReferenceKeyFrame
                    if(!mVelocity.empty())
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.
                    //在最后一帧，我们跟踪了主要的视觉里程计点
                    //我们计算两个相机位姿，一个来自运动模型一个做重定位。如果重定位成功我们就选择这个思路，否则我们继续使用视觉里程计思路

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    cv::Mat TcwMM;
                    if(!mVelocity.empty())
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    bOKReloc = Relocalization();

                    if(bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if(mbVO)
                        {
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc)
                    {
                        //重定位成功的时候，就使用重定位的思路，所以标记mbVO要设置为false
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }//else end(mbOnlyTracking==true)
        //将最新的关键帧作为reference frame
        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        /**
         * 在帧间匹配得到初始的姿态后，现在对local map进行跟踪得到更多的匹配，并优化当前位姿
         * local map:当前帧、当前帧的MapPoints、当前关键帧与其它关键帧共视关系
         * 在步骤2.1中主要是两两跟踪（恒速模型跟踪上一帧、跟踪参考帧），这里搜索局部关键帧后搜集所有局部MapPoints，
         * 然后将局部MapPoints和当前帧进行投影匹配，得到更多匹配的MapPoints后进行Pose优化
        */
        if(!mbOnlyTracking)
        {
            if(bOK)
                bOK = TrackLocalMap();
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            // mbVO值为true表示有很少的匹配点和地图中的MapPoint相匹配，我们无法检索本地地图因此此时不调用TrackLocalMap函数。
            // 一旦系统重新定位相机位置，我们将再次使用本地地图
            if(bOK && !mbVO)
                bOK = TrackLocalMap();//局部地图跟踪
        }

        if(bOK)
            mState = OK;
        else
            mState=LOST;

        // Update drawer
        mpFrameDrawer->Update(this);

        // If tracking were good, check if we insert a keyframe
        if(bOK)
        {
            // Update motion model
            if(!mLastFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                //当前帧的位姿和上一个帧的位姿相乘获得速度
                mVelocity = mCurrentFrame.mTcw*LastTwc;
            }
            else
                mVelocity = cv::Mat();
            //地图中设置当前帧的相机位姿
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    //如果该MapPoint没有被其他帧观察到，则将该MapPoint设置为NULL,也就是去掉
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // Delete temporal MapPoints 删除暂时的MapPoint
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

            // Check if we need to insert a new keyframe
            // 判断是否需要插入关键帧
            if(NeedNewKeyFrame())
                CreateNewKeyFrame();

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            //抛弃无用的点
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        //如果初始化后不久相机丢失，则进行重置
        if(mState==LOST)
        {
            if(mpMap->KeyFramesInMap()<=5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
        //Tracking完成的时候，记录当前帧为上一帧
        mLastFrame = Frame(mCurrentFrame);
    }//跟踪end
    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    //3：存储帧的位姿信息，稍后用来重现完整的相机运动轨迹
    if(!mCurrentFrame.mTcw.empty())
    {
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }
}


void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)
    {
        // Set Frame pose to the origin
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPoint(pNewMP,i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);
        //设置当前帧的相机位姿
        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState=OK;
    }
}
/**
 * 单目相机初始化函数
 * 功能：创建初始化器，并对前两个关键点数大于100的帧进行特征点匹配，根据匹配结果计算出当前帧的变换矩阵并在窗口中显示
 * 1. 第一次进入该方法,如果当前帧关键点数>100,将当前帧保存为初始帧和最后一帧，并创建一个初始化器；
 * 2. 第二次进入该方法的时候，已经有初始化器了，如果当前帧中的关键点数>100；
 * 3. 利用ORB匹配器，对当前帧和初始帧进行匹配，匹配关键点数小于100时失败；
 * 4. 利用匹配的关键点信息进行单应矩阵和基础矩阵的计算，进而计算出相机位姿的旋转矩阵和平移矩阵；
 * 5. 进行三角化判断，删除不能三角化的无用特征关键点；
 * 6. 由旋转矩阵和平移矩阵构造变换矩阵；
 * 7. 将三角化得到的3D点包装成MapPoints，在地图中显示；
*/
void Tracking::MonocularInitialization()
{
    //如果单目初始器还没有被创建，则创建单目初始器
    if(!mpInitializer)
    {
        // Set Reference Frame
        //step 1：第一次进入该方法,如果当前帧关键点数>100,将当前帧保存为初始帧和最后一帧，并创建一个初始化器
        if(mCurrentFrame.mvKeys.size()>100)
        {
            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            //mvbPrevMatched的大小设置为已经提取的关键点的个数，mvbPrevMatched最大的情况就是所有特征点都被跟踪上
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                //mvbPrevMatched中存储关键点的坐标，用于新的一帧到来时进行匹配
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            if(mpInitializer)
                delete mpInitializer;
            //由当前帧构造初始器 sigma:1.0 iterations:200
            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);
            //mvIniMatches中所有的元素值设置为-1
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            return;
        }
    }
    else//如果是第二次进入，这时候已经创建了初始器
    {
        // Try to initialize
        /**
         * step 2：第二次进入该方法的时候，已经有初始化器了，如果当前帧中的关键点数>100，则继续进行匹配工作。
         * 如果当前帧特征点太少，释放初始化器。因此只有连续两帧的特征点个数都大于100时，才能继续进行初始化过程
        */
        if((int)mCurrentFrame.mvKeys.size()<=100)
        {
            //特征点数少于100，此时删除初始化器
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

        // Find correspondences
        //创建ORBmatcher
        ORBmatcher matcher(0.9,true);
        //计算当前帧和初始化帧之间的匹配关系
        /**
         * step 3：在mInitialFrame与mCurrentFrame中找匹配的特征点对
         * mvbPrevMatched为前一帧的特征点的坐标，存储了mInitialFrame中哪些点将进行接下来的匹配。SearchForInitialization调用完后mvbPrevMatched中
         * 保存的值为F1中对应关键点和F2中哪个关键点匹配的该关键点的(x,y)坐标
         * mvIniMatches用于存储mInitialFrame,mCurrentFrame之间匹配的特征点
        */
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        // Check if there are enough correspondences
        if(nmatches<100)
        {
            //释放初始化器
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

        cv::Mat Rcw; // Current Camera Rotation 定义相机旋转矩阵
        cv::Mat tcw; // Current Camera Translation 定义相机平移矩阵
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)三角化对应关系
        //step 4：利用匹配的关键点信息进行单应矩阵和基础矩阵的计算，进而计算出相机位姿的旋转矩阵Rcw和平移矩阵tcw
        //在该函数中创建了计算单应矩阵和基础矩阵的两个线程，计算的旋转和平移量值存放在Rcw和tcw中
        //mvIniMatches[i]中i为前一帧匹配的关键点的index，值为当前帧的匹配的关键点的index
        if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            //step 5：删除那些无法进行三角化的匹配点
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                //判断该点是否可以三角化
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    //表示两帧对应的关键点不再匹配
                    mvIniMatches[i]=-1;
                    //关键点匹配个数-1
                    nmatches--;
                }
            }

            // Set Frame Poses
            //将初始化的第一帧作为世界坐标系，因此第一帧变换矩阵为单位矩阵
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            /**
             * step 6：由旋转矩阵和平移矩阵构造变换矩阵
             * 由Rcw和tcw构造Tcw,并赋值给mTcw，mTcw为世界坐标系到该帧的变换矩阵
             * 这里构造出来的Tcw为一个4*4的矩阵，其中的Rcw为3*3，tcw为3*1如下所示：
             * |Rcw  tcw|
             * |0     1 |
            */
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);
            /**
             * step 7：将三角化得到的3D点包装成MapPoints，在地图中显示
             * Initialize函数会得到mvIniP3D，
             * mvIniP3D是cv::Point3f类型的一个容器，是个存放3D点的临时变量，
             * CreateInitialMapMonocular将3D点包装成MapPoint类型存入KeyFrame和Map中
            */
            CreateInitialMapMonocular();
        }
    }
}
/**
 * 功能：为关键帧初始化生成对应的MapPoints
*/
void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    //创建初始帧和当前帧
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    //1：计算初始化关键帧和当前关键帧的BOW
    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    //2.将基础关键帧和当前关键帧插入地图中，地图中就会显示
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    //3.遍历所有匹配的关键点，创建对应的mapPoint
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        //用已经初始化好的3D点来创建world坐标
        cv::Mat worldPos(mvIniP3D[i]);
        // 3.1：用3D点构造MapPoint
        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);
        //i为pMP这个MapPoint在pKFini中对应的关键点的index值
        pKFini->AddMapPoint(pMP,i);
        //mvIniMatches[i]为pMP这个MapPoint在pKFcur这个关键帧中对应的关键点的index值
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);
        /**
         * 3.2：为该MapPoint添加相关属性
        */
        //1）.哪些关键帧可以观测到该MapPoint
        //pKFini这个关键帧可以观测到pMP这个MapPoint，这个MapPoint对应pKFini中特征点的index值为i
        pMP->AddObservation(pKFini,i);
        //mvIniMatches[i]为pMP这个MapPoint在pKFcur这个关键帧中对应的关键点的index值
        pMP->AddObservation(pKFcur,mvIniMatches[i]);
        //2).从众多观测到该MapPoint的特征点中挑选区分度最高的描述子
        pMP->ComputeDistinctiveDescriptors();
        //3).更新该MapPoint平均观测方向以及观测距离的范围
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        //mvIniMatches[i]为pMP这个MapPoint在pKFcur这个关键帧中对应的关键点的index值
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        //将mappoint插入到地图中
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    //4：更新关键帧间的连接关系
    //在3D点和关键帧之间建立边，每个边有一个权重，边的权重是该关键帧与当前帧公共3D点的个数
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    //mpMap->MapPointsInMap()返回的是map中的mappoint数目
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;
    //5：进行BA优化
    Optimizer::GlobalBundleAdjustemnt(mpMap,20);

    // Set median depth to 1
    //6：将MapPoints的中值深度归一化到1，并归一化两帧之间变换,评估关键帧场景深度，q=2表示中值
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f / medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    //将Tc2w中的平移向量t进行了修改 
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
        }
    }
    //往LocalMapper中插入关键帧，此时LocalMapping线程中就会处理插入的关键帧
    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    //将当前关键帧设置为参考关键帧
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;
    //更新mLastFrame
    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
    //设置当前帧的相机位姿
    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);
    //初始化完成，修改标记值
    mState=OK;
}

void Tracking::CheckReplacedInLastFrame()
{
    //遍历关键点，获取其对应的MapPoint
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            //用新的替换该点的MapPoint替换掉当前的MapPoint
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}

/**
 * 函数功能：
 * 1. 计算当前帧的词包，将当前帧的特征点分到特定层的nodes上
 * 2. 对属于同一node的描述子进行匹配
 * 3. 根据匹配对估计当前帧的姿态
 * 4. 根据姿态剔除误匹配
 * 具体步骤：
 * 1. 按照关键帧进行Track的方法和运动模式恢复相机运动位姿的方法接近。首先求解当前帧的BOW向量。 
 * 2. 再搜索当前帧和关键帧之间的关键点匹配关系，如果这个匹配关系小于15对的话，就Track失败了。 
 * 3. 接着将当前帧的位置假定到上一帧的位置那里 
 * 4. 并通过最小二乘法优化相机的位姿。 
 * 5. 最后依然是抛弃无用的杂点，当match数大于等于10的时候，返回true成功。 
*/
bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    //步骤1：将当前帧的描述子转化为BoW向量
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    //构建ORBmatcher
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;
    //步骤2：通过特征点的BoW加快当前帧与参考帧之间的特征点匹配,对上一个关键帧进行BOW搜索匹配点
    //特征点的匹配关系由MapPoints进行维护.
    /**
     * vpMapPointMatches中存储的是当前帧特征点的index，值为mpReferenceKF参考帧中对应的MapPoint
    */
    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);
    //匹配数小于15，返回false
    if(nmatches<15)
        return false;
    //步骤3:将和参考关键帧匹配的MapPoint列表用于更新当前帧的MapPoints列表，将上一帧的位姿态作为当前帧位姿的初始值
    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    //用上一次的Tcw设置位姿初值，在PoseOptimization可以收敛快一些
    mCurrentFrame.SetPose(mLastFrame.mTcw);
    //步骤4:通过优化3D-2D的重投影误差来获得位姿。通过PoseOptimization优化相机的位姿
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    //抛弃杂点.步骤5：剔除优化后的outlier匹配点（MapPoints）
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            //判断是否为outlier匹配点
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                //将为outlier匹配点的MapPoint设置为NULL
                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                //表示这个MapPoint不在可视化窗口中跟踪
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }
    //若被观察的点数大于10返回true
    return nmatchesMap>=10;
}
/**
 * 1.更新最近一帧的位姿
 * 2.对于双目或rgbd摄像头，为上一帧临时生成新的MapPoints,注意这些MapPoints不加入到Map中，
 * 在tracking的最后会删除,跟踪过程中需要将上一帧的MapPoints投影到当前帧可以缩小匹配范围，加快当前帧与上一帧进行特征点匹配
*/
void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    //步骤1：更新最近一帧的位姿
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame.SetPose(Tlr*pRef->GetPose());
    //如果上一帧为关键帧，或者单目的情况，则退出
    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    //创建视觉里程计地图点，我们根据双目或者RGB-D传感器的测量深度对点进行排序
    /**
     *步骤2：对于双目或rgbd摄像头，为上一帧临时生成新的MapPoints
     注意这些MapPoints不加入到Map中，在tracking的最后会删除
     跟踪过程中需要将上一帧的MapPoints投影到当前帧可以缩小匹配范围，加快当前帧与上一帧进行特征点匹配
     步骤2.1：得到上一帧有深度值的特征点
    */
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;
    //步骤2.2：按照深度从小到大排序
    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    //步骤2.3：将距离比较近的点包装成MapPoints
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            // 这些生成MapPoints后并没有通过：
            // a.AddMapPoint、
            // b.AddObservation、
            // c.ComputeDistinctiveDescriptors、
            // d.UpdateNormalAndDepth添加属性，
            // 这些MapPoint仅仅为了提高双目和RGBD的跟踪成功率
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

            mLastFrame.mvpMapPoints[i]=pNewMP;// 添加新的MapPoint
            //标记为临时添加的MapPoint，之后在CreateNewKeyFrame之前会全部删除
            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }
}
/**
 * 步骤：
 * 1. 首先通过上一帧的位姿和速度来设置当前帧相机的位姿
 * 2. 通过PnP方法估计相机位姿，再将上一帧的地图点投影到当前固定大小范围的帧平面上，如果匹配点少，那么扩大两倍的采点范围
 * 3. 然后进行一次BA算法，优化相机的位姿
 * 4. 优化位姿之后，对当前帧的关键点和地图点，抛弃无用的杂点，剩下的点供下一次操作使用
*/
bool Tracking::TrackWithMotionModel()
{
    //构建ORBmatcher
    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    // 根据参考关键帧更新上一帧的位姿。如果在定位模式中的话，创建视觉里程计点
    /**
     * 步骤1：对于双目或rgbd摄像头，根据深度值为上一关键帧生成新的MapPoints
     * （跟踪过程中需要将当前帧与上一帧进行特征点匹配，将上一帧的MapPoints投影到当前帧可以缩小匹配范围）
     * 在跟踪过程中，去除outlier的MapPoint，如果不及时增加MapPoint会逐渐减少
     * 这个函数的功能就是补充增加RGBD和双目相机上一帧的MapPoints数
    */
    UpdateLastFrame();
    //根据Const Velocity Model(认为这两帧之间的相对运动和之前两帧间相对运动相同)估计当前帧的位姿
    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);//当前帧位姿等于mVelocity*mLastFrame.mTcw

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    // 在前一帧观察投影点
    int th;
    if(mSensor!=System::STEREO)//非双目搜索范围系数设为15
        th=15;
    else
        th=7;
    //步骤2：根据匀速度模型进行对上一帧的MapPoints进行跟踪
    //根据上一帧特征点对应的3D点投影的位置缩小特征点匹配范围
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

    // If few matches, uses a wider window search
    //如果匹配对不到20个，扩大投影面积继续投影
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
    }
    //若匹配数依旧小于20返回false
    if(nmatches<20)
        return false;

    // Optimize frame pose with all matches
    //步骤3：优化位姿.通过PoseOptimization优化相机位姿
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    //步骤4：优化位姿后剔除outlier的mvpMapPoints.抛弃杂点
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    if(mbOnlyTracking)
    {
        //匹配数<10,则mbVO的值为true
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    return nmatchesMap>=10;
}
/**
 * 步骤：
 * 1. 更新Covisibility Graph， 更新局部关键帧
 * 2. 根据局部关键帧，更新局部地图点，接下来运行过滤函数 isInFrustum
 * 3. 将地图点投影到当前帧上，超出图像范围的舍弃
 * 4. 当前视线方向v和地图点云平均视线方向n, 舍弃n*v<cos(60)的点云
 * 5. 舍弃地图点到相机中心距离不在一定阈值内的点 
 * 6. 计算图像的尺度因子 isInFrustum 函数结束
 * 7. 进行非线性最小二乘优化
 * 8. 更新地图点的统计量
 * 
*/
bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.
    /**
     * 我们有了一个相机位姿的估计值和在帧中跟踪的一些MapPoint.我们检索本地地图并且尝试去发现在本地地图中的匹配点。
    */
    //步骤1：更新局部关键帧列表mvpLocalKeyFrames和局部地图点mvpLocalMapPoints
    UpdateLocalMap();
    //步骤2：在局部地图中查找与当前帧匹配的MapPoints，将其建立关联
    SearchLocalPoints();

    // Optimize Pose
    //在这个函数之前，在Relocalization、TrackReferenceKeyFrame、TrackWithMotionModel中都有位姿优化，
    //步骤3：更新局部所有MapPoints后对位姿再次优化
    Optimizer::PoseOptimization(&mCurrentFrame);//最小化投影误差优化位姿
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    //更新当前帧的MapPoints被观测程度，并统计跟踪局部地图的效果
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            //由于当前帧的MapPoints可以被当前帧观测到，其被观测统计量加1
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    //该MapPoint被其它关键帧观测到过
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    //记录当前帧跟踪到的MapPoints，用于统计跟踪效果
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    //步骤4：决定是否跟踪成功 mnLastRelocFrameId为上次重定位的帧id，mMaxFrames为一秒钟内上报的帧数目
    if(mCurrentFrame.mnId < mnLastRelocFrameId+mMaxFrames && mnMatchesInliers < 50)
        return false;

    if(mnMatchesInliers<30)
        return false;
    else
        return true;
}
/**
 * 函数功能：判断是否需要生成新的关键帧
 * 确定关键帧的标准：
 * 1.在上一全局重定位后，过了20帧；
 * 2.局部建图闲置了，或在上一关键帧插入后过了20帧；
 * 3.当前帧跟踪到大于50个点；
 * 4.当前帧跟踪到的比参考帧少了90%
 * 
*/

bool Tracking::NeedNewKeyFrame()
{
    /**
     * 步骤1：如果用户在界面上选择重定位，那么将不插入关键帧
     * 由于插入关键帧过程中会生成MapPoint，因此用户选择重定位后地图上的点云和关键帧都不会再增加
    */
    if(mbOnlyTracking)//如果仅跟踪，则不插入关键帧
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    //如果局部地图被闭环检测使用，则不插入关键帧
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();//关键帧数

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    /**
     * 步骤2：判断是否距离上一次插入关键帧的时间太短
     * mCurrentFrame.mnId是当前帧的ID
     * mnLastRelocFrameId是最近一次重定位帧的ID
     * mMaxFrames等于图像输入的帧率
     * 如果关键帧比较少，则考虑插入关键帧
     * 或距离上一次重定位超过1s，则考虑插入关键帧
     * kitti从单目配置文件中读取的mMaxFrames值为10
    */
    if(mCurrentFrame.mnId < mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    //步骤3：得到参考关键帧跟踪到的MapPoints数量
    //在UpdateLocalKeyFrames函数中会将与当前关键帧共视程度最高的关键帧设定为当前帧的参考关键帧
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);//获取参考关键帧跟踪到的MapPoints数量

    // Local Mapping accept keyframes?
    //步骤4：查询局部地图管理器是否繁忙
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    //步骤5：对于双目或RGBD摄像头，统计总的可以添加的MapPoints数量和跟踪到地图中的MapPoints数量
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    //非单目相机的处理
    if(mSensor!=System::MONOCULAR)
    {
        for(int i =0; i<mCurrentFrame.N; i++)//遍历当前帧所有匹配点
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)//map点的速度在合理范围内
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }

    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Thresholds
    //步骤6：决策是否需要插入关键帧
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    //自从上次之后已经有超过MaxFrames帧没有插入
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    //local mapping 处于空闲状态
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    //Condition 1c: tracking is weak
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        //如果mapping接受关键帧，则插入关键帧，否则发送信号到中断BA
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)
            {
                // 队列里不能阻塞太多关键帧
                // tracking插入关键帧不是直接插入，而且先插入到mlNewKeyFrames中，
                // 然后localmapper再逐个pop出来插入到mspKeyFrames
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}
/**
 * 1：用当前帧构造成关键帧
 * 2：将当前关键帧设置为当前帧的参考关键帧
 * 3：对于双目或rgbd摄像头，为当前帧生成新的MapPoints
*/
void Tracking::CreateNewKeyFrame()
{
    if(!mpLocalMapper->SetNotStop(true))
        return;
    //步骤1：用当前帧构造成关键帧
    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);
    //步骤2：将当前关键帧设置为当前帧的参考关键帧
    //在UpdateLocalKeyFrames函数中会将与当前关键帧共视程度最高的关键帧设定为当前帧的参考关键帧
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;
    // 步骤3：对于双目或rgbd摄像头，为当前帧生成新的MapPoints
    if(mSensor!=System::MONOCULAR)
    {
        //根据Tcw计算mRcw、mtcw和mRwc、mOw
        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        //步骤3.1：得到当前帧深度小于阈值的特征点
        //创建新的MapPoint, depth < mThDepth
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            //步骤3.2：按照深度从小到大排序
            sort(vDepthIdx.begin(),vDepthIdx.end());
            //步骤3.3：将距离比较近的点包装成MapPoints
            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    // 这些添加属性的操作是每次创建MapPoint后都要做的
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }
                // 这里决定了双目和rgbd摄像头时地图点云的稠密程度
                // 但是仅仅为了让地图稠密直接改这些不太好，
                // 因为这些MapPoints会参与之后整个slam过程
                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }
    /**
     * 往LocalMapping线程的mlNewKeyFrames队列中插入新生成的keyframe
     * LocalMapping线程中检测到有队列中有keyframe插入后线程会run起来
    */
    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}
/**
 * 函数功能：在局部地图中查找在当前帧视野范围内的点，将视野范围内的点和当前帧的特征点进行投影匹配
 * 步骤：
 * 1：遍历当前帧的mvpMapPoints，标记这些MapPoints不参与之后的搜索
 * 2：将所有局部MapPoints投影到当前帧。先判断是否在视野范围内，然后进行投影匹配
 * 3：对于双目或rgbd摄像头，为当前帧生成新的MapPoints
*/
void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    //步骤1：遍历当前帧的mvpMapPoints，标记这些MapPoints不参与之后的搜索
    //因为当前的mvpMapPoints一定在当前帧的视野中
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                // 更新能观测到该点的帧数加1
                pMP->IncreaseVisible();
                //标记该点被当前帧观测到
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                //标记该点将来不被投影，因为已经匹配过
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    //步骤2：将所有局部MapPoints投影到当前帧，判断是否在视野范围内，然后进行投影匹配
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        // 已经被当前帧观测到MapPoint不再判断是否能被当前帧观测到
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        //步骤2.1：判断LocalMapPoints中的点是否在在视野内
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            //观测到该MapPoint点的帧数加1，该MapPoint在某些帧的视野范围内
            pMP->IncreaseVisible();
            //只有在视野范围内的MapPoints才参与之后的投影匹配
            nToMatch++;
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        // If the camera has been relocalised recently, perform a coarser search
        //如果不久前进行过重定位，那么进行一个更加宽泛的搜索，阈值需要增大
        if(mCurrentFrame.mnId < mnLastRelocFrameId+2)
            th=5;
        //步骤2.2：对视野范围内的MapPoints通过投影进行特征点匹配
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }
}
/**
 * 更新局部地图
 * 局部地图包括：k1个关键帧、k2个临近关键帧和参考关键帧、由这些关键帧观测到的MapPoints
 * 
*/
void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // 更新局部关键帧列表
    UpdateLocalKeyFrames();
    // 更新局部点
    UpdateLocalPoints();
}
/**
 * 更新局部关键点
 * 局部关键帧mvpLocalKeyFrames的MapPoints，更新mvpLocalMapPoints
*/
void Tracking::UpdateLocalPoints()
{
    //步骤1：清空局部MapPoints
    mvpLocalMapPoints.clear();
    //步骤2：遍历局部关键帧mvpLocalKeyFrames
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}

/**
 * 更新局部关键帧列表，能看到当前关键帧中所有MapPoint的关键帧构成了一个局部关键帧列表mvpLocalKeyFrames
 * 遍历当前帧的MapPoints，将观测到这些MapPoints的关键帧和相邻的关键帧取出，更新mvpLocalKeyFrames
*/
void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    //步骤1：遍历当前帧的MapPoints，记录所有能观测到当前帧MapPoints的关键帧
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                // 获取能观测到当前帧MapPoints的关键帧
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    //keyframeCounter中最后存放的就是对应关键帧可以看到当前帧mCurrentFrame里多少个MapPoint
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);
    //步骤2：更新局部关键帧（mvpLocalKeyFrames），添加局部关键帧有三个策略
    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    //策略1：能观测到当前帧MapPoints的关键帧作为局部关键帧，存入局部关键帧列表中
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;
        //it->second表示pKF这个关键帧可以看到多少个MapPoint
        if(it->second > max)
        {
            max=it->second;
            //pKFmax表示能看到MapPoint点数最多的关键帧
            pKFmax=pKF;
        }
        //mvpLocalKeyFrames里边存放的是能看到当前帧对应的MapPoint的关键帧列表
        mvpLocalKeyFrames.push_back(it->first);
        //mnTrackReferenceForFrame防止重复添加局部关键帧
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }


    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    //策略2：与策略1得到的局部关键帧共视程度很高的关键帧作为局部关键帧
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;
        //策略2.1:最佳共视的10帧
        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                //mnTrackReferenceForFrame防止重复添加局部关键帧
                if(pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }
        //策略2.2:自己的子关键帧
        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }
        //策略2.3:自己的父关键帧
        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            //mnTrackReferenceForFrame防止重复添加局部关键帧
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }
    //步骤3：更新当前帧的参考关键帧，与自己共视程度最高的关键帧作为参考关键帧。
    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}
/**
 * 重定位，从之前的关键帧中找出与当前帧之间拥有充足匹配点的候选帧，利用Ransac迭代，通过PnP求解位姿。
 * 步骤：
 * 1. 先计算当前帧的BOW值，并从关键帧数据库中查找候选的匹配关键帧
 * 2. 构建PnP求解器，标记杂点，准备好每个关键帧和当前帧的匹配点集
 * 3. 用PnP算法求解位姿，进行若干次P4P Ransac迭代，并使用非线性最小二乘优化，直到发现一个有充足inliers支持的相机位置
 * 4. 返回成功或失败
*/
bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    //步骤1：计算当前帧特征点的Bow映射
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    //步骤2：找到与当前帧相似的候选关键帧
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);
    //如果未找到候选帧，返回false
    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    //我们首先执行与每个候选匹配的ORB匹配
    //如果找到足够的匹配，我们设置一个PNP解算器
    ORBmatcher matcher(0.75,true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;
    //遍历候选关键帧
    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;//去除不好的候选关键帧
        else
        {
            //步骤3：通过BoW进行匹配，计算出pKF中和mCurrentFrame匹配的关键点的MapPoint存入vvpMapPointMatches中。
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                //候选关键帧中匹配点数小于15的丢弃
                vbDiscarded[i] = true;
                continue;
            }
            else//用pnp求解
            {
                //候选关键帧中匹配点数大于15的构建PnP求解器。这个PnP求解器中的3D point为vvpMapPointMatches中的MapPoint，2D点为mCurrentFrame中的关键点
                //因为是重定位，所以就是求重定位的候选帧对应的MapPoint到当前帧的关键点之间的投影关系，通过投影关系确定当前帧的位姿，也就进行了重定位
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                //候选帧中匹配点数大于15的进行Ransac迭代
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    // 执行一些P4P RANSAC迭代，直到我们找到一个由足够的inliers支持的相机位姿
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;
            //步骤4：通过EPnP算法估计姿态
            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }
                //步骤5：通过PoseOptimization对姿态进行优化求解
                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                //步骤6：如果内点比较少，在一个大概的窗口中投影并再次进行位姿优化
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}

void Tracking::Reset()
{

    cout << "System Reseting" << endl;
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}



} //namespace ORB_SLAM
