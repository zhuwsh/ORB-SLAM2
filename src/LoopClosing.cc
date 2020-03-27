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

#include "LoopClosing.h"

#include "Sim3Solver.h"

#include "Converter.h"

#include "Optimizer.h"

#include "ORBmatcher.h"

#include<mutex>
#include<thread>


namespace ORB_SLAM2
{

LoopClosing::LoopClosing(Map *pMap, KeyFrameDatabase *pDB, ORBVocabulary *pVoc, const bool bFixScale):
    mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpMatchedKF(NULL), mLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true),
    mbStopGBA(false), mpThreadGBA(NULL), mbFixScale(bFixScale), mnFullBAIdx(0)
{
    mnCovisibilityConsistencyTh = 3;
}

void LoopClosing::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

/**
 * LoopClosing线程的执行函数
 * 在KeyFrameDataBase中查找与mlpLoopKeyFrameQueue中新加入的关键帧相似的闭环候选帧
 * 主要步骤：
 * 1.闭环检测，获得闭环候选帧；
 * 2.计算sim3,根据sim3的计算值更新地图点的位姿；
 * 3.进行地图点融合和位姿优化；
*/
void LoopClosing::Run()
{
    mbFinished =false;

    while(1)
    {
        // Check if there are keyframes in the queue
        if(CheckNewKeyFrames())
        {
            // Detect loop candidates and check covisibility consistency
            //检测闭环。在共视关系的关键帧中找到与当前关键帧Bow匹配最低得分minScore，在除去当前帧共视关系的关键帧数据库中，检测闭环候选帧
            if(DetectLoop())
            {
               // Compute similarity transformation [sR|t] 计算相似变换
               // In the stereo/RGBD case s=1
               //计算旋转平移的相似性，也就是相似变换
               if(ComputeSim3())
               {
                   // Perform loop fusion and pose graph optimization
                   // 进行闭环校正，执行闭环融合和位姿图优化
                   CorrectLoop();
               }
            }
        }       

        ResetIfRequested();

        if(CheckFinish())
            break;

        usleep(5000);
    }

    SetFinish();
}
/**
 * LocalMapping线程中调用该函数，将关键帧插入到LoopClosing线程的队列中
 * LoopClosing线程run函数检测到有关键帧插入后，会直行操作处理该关键帧
 * 
*/
void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    if(pKF->mnId!=0)
        mlpLoopKeyFrameQueue.push_back(pKF);
}

bool LoopClosing::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    return(!mlpLoopKeyFrameQueue.empty());
}
/**
 * 检测闭环
*/
bool LoopClosing::DetectLoop()
{
    //1.从队列中取出头部的关键帧
    {
        unique_lock<mutex> lock(mMutexLoopQueue);
        mpCurrentKF = mlpLoopKeyFrameQueue.front();
        //删除mlpLoopKeyFrameQueue中第一个元素
        mlpLoopKeyFrameQueue.pop_front();
        // Avoid that a keyframe can be erased while it is being process by this thread
        mpCurrentKF->SetNotErase();
    }

    //If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
    //2.判断距离上次闭环检测是否超过10帧。如果不超过10帧，则将当前帧添加到mpKeyFrameDB中的列表中
    if(mpCurrentKF->mnId < mLastLoopKFid+10)//超过10帧才进行后边的处理
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

    // Compute reference BoW similarity score
    // This is the lowest score to a connected keyframe in the covisibility graph
    // We will impose loop candidates to have a higher similarity than this
    //3.计算当前帧及其共视关键帧的词袋模型匹配得分，获得minScore
    const vector<KeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();
    //获取当前关键帧的词袋向量表
    const DBoW2::BowVector &CurrentBowVec = mpCurrentKF->mBowVec;
    float minScore = 1;
    //遍历和当前关键帧共视的关键帧列表，计算当前帧与和它共视的所有帧之间的Bow最低分值
    for(size_t i=0; i<vpConnectedKeyFrames.size(); i++)
    {
        KeyFrame* pKF = vpConnectedKeyFrames[i];
        if(pKF->isBad())
            continue;
        //获取共视的关键帧词袋向量
        const DBoW2::BowVector &BowVec = pKF->mBowVec;
        //计算当前帧和共视帧之间的词袋匹配分值
        float score = mpORBVocabulary->score(CurrentBowVec, BowVec);
        //minScore中存放的是当前关键帧和其共视关系的关键帧之间单词向量比较的最小分值
        if(score < minScore)
            minScore = score;
    }

    // Query the database imposing the minimum score
    /**
     * 在除去当前帧共视关系的关键帧数据中，检测闭环候选帧(这个函数在KeyFrameDatabase中)
     * 闭环候选帧删选过程： 
     * 1,BoW得分>minScore; 
     * 2,统计满足1的关键帧中有共同单词最多的单词数maxcommonwords 
     * 3,筛选出共同单词数大于mincommons(=0.8*maxcommons)的关键帧 
     * 4,相连的关键帧分为一组，计算总得分（总分）,得到最大总分bestAccScore,筛选出总分大于minScoreToRetain(=0.75*bestAccScore)的组 
     * 用得分最高的候选帧IAccScoreAndMathch代表该组，计算组得分的目的是剔除单独一帧得分较高，但是没有共视关键帧作为闭环来说不够鲁棒 
     * 对于通过了闭环检测的关键帧，还需要通过连续性检测(连续三帧都通过上面的筛选)，才能作为闭环候选帧
    */
    // 4.检测当前关键帧的闭环候选帧
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);

    // If there are no loop candidates, just add new keyframe and return false
    // 没有闭环候选帧
    if(vpCandidateKFs.empty())
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mvConsistentGroups.clear();
        mpCurrentKF->SetErase();
        return false;
    }

    // For each loop candidate check consistency with previous loop candidates
    // Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
    // A group is consistent with a previous group if they share at least a keyframe
    // We must detect a consistent loop in several consecutive keyframes to accept it
    //对每一个闭环候选帧和之前的闭环候选帧进行连续性性检查
    //每个候选帧扩展出了一个共视组(在共视图中关键帧和闭环候选帧相连接)
    //如果一个组和前一个组至少共享一个关键帧，则该组与前一个组是连续的。
    //我们必须在几个连续的关键帧中检测到一个连续的循环才能接受它
    mvpEnoughConsistentCandidates.clear();

    vector<ConsistentGroup> vCurrentConsistentGroups;
    vector<bool> vbConsistentGroup(mvConsistentGroups.size(),false);
    //5.遍历闭环候选帧，从中选出真正存在闭环的关键帧，这里主要是做连续性检测
    for(size_t i=0, iend=vpCandidateKFs.size(); i<iend; i++)//第一个循环
    {
        KeyFrame* pCandidateKF = vpCandidateKFs[i];
        //获取与该关键帧连接的关键帧集合
        set<KeyFrame*> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();
        //一个关键帧和与其相连的所有关键帧一起构成一个关键帧集合
        spCandidateGroup.insert(pCandidateKF);

        bool bEnoughConsistent = false;
        bool bConsistentForSomeGroup = false;
        for(size_t iG=0, iendG = mvConsistentGroups.size(); iG<iendG; iG++)//第二个循环
        {
            //获得一个关键帧组
            set<KeyFrame*> sPreviousGroup = mvConsistentGroups[iG].first;

            bool bConsistent = false;
            for(set<KeyFrame*>::iterator sit=spCandidateGroup.begin(), send=spCandidateGroup.end(); sit!=send;sit++)//第三个循环
            {
                //判断sit这个关键帧在sPreviousGroup中出现过，则认为具有连续性
                if(sPreviousGroup.count(*sit))
                {
                    bConsistent=true;
                    bConsistentForSomeGroup=true;
                    break;
                }
            }

            if(bConsistent)
            {
                int nPreviousConsistency = mvConsistentGroups[iG].second;
                int nCurrentConsistency = nPreviousConsistency + 1;
                if(!vbConsistentGroup[iG])//吴博说这里应该是vbConsistentGroup[i],而不应该是vbConsistentGroup[iG]？？？
                {
                    //将spCandidateGroup和其连续个数作为一个pair存入vCurrentConsistentGroups中
                    ConsistentGroup cg = make_pair(spCandidateGroup,nCurrentConsistency);
                    vCurrentConsistentGroups.push_back(cg);
                    vbConsistentGroup[iG] = true; //this avoid to include the same group more than once
                }
                //mnCovisibilityConsistencyTh=3
                if(nCurrentConsistency >= mnCovisibilityConsistencyTh && !bEnoughConsistent)
                {
                    //mvpEnoughConsistentCandidates中存储候选关键帧
                    mvpEnoughConsistentCandidates.push_back(pCandidateKF);
                    bEnoughConsistent=true; //this avoid to insert the same candidate more than once
                }
            }
        }

        // If the group is not consistent with any previous group insert with consistency counter set to zero
        //如果这个组和任何前边的组都没有连续性关系，则将连续性计数设置为0
        if(!bConsistentForSomeGroup)
        {
            ConsistentGroup cg = make_pair(spCandidateGroup,0);
            vCurrentConsistentGroups.push_back(cg);
        }
    }

    // Update Covisibility Consistent Groups
    mvConsistentGroups = vCurrentConsistentGroups;


    // Add Current Keyframe to database
    mpKeyFrameDB->add(mpCurrentKF);

    if(mvpEnoughConsistentCandidates.empty())
    {
        mpCurrentKF->SetErase();
        return false;
    }
    else
    {
        //当前候选帧和之前的帧之间存在闭环关系
        return true;
    }

    mpCurrentKF->SetErase();
    return false;
}
/**
 * 计算当前关键帧和闭环候选帧之间的Sim3,这个Sim3变换就是闭环前累计的尺度和位姿误差
 * 该误差也可以帮助检验该闭环在空间几何姿态上是否成立
*/
bool LoopClosing::ComputeSim3()
{
    // For each consistent loop candidate we try to compute a Sim3
    //对每一个连续的闭环候选帧，我们都尝试计算sim3
    //获得候选关键帧的个数
    const int nInitialCandidates = mvpEnoughConsistentCandidates.size();

    // We compute first ORB matches for each candidate
    // If enough matches are found, we setup a Sim3Solver
    // 我们为每一个候选帧计算第一个ORB匹配项
    // 如果足够的匹配项发现了，我们就启动Sim3Solver
    ORBmatcher matcher(0.75,true);

    vector<Sim3Solver*> vpSim3Solvers;
    vpSim3Solvers.resize(nInitialCandidates);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nInitialCandidates);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nInitialCandidates);

    int nCandidates=0; //candidates with enough matches
    //1.遍历候选关键帧，对每一个候选关键帧和当前关键帧之间匹配的特征点进行sim3求解
    for(int i=0; i<nInitialCandidates; i++)
    {
        KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

        // avoid that local mapping erase it while it is being processed in this thread
        pKF->SetNotErase();

        if(pKF->isBad())
        {
            vbDiscarded[i] = true;
            continue;
        }
        /**
         * 这里主要是通过SearchByBow搜索当前关键帧中和闭环候选帧匹配的地图点
         * BoW通过将单词聚类到树结构node的方法，这样可以加快搜索匹配速度
         * vvpMapPointMatches[i]用于存储当前关键帧和候选关键帧之间匹配的地图点
        */
        int nmatches = matcher.SearchByBoW(mpCurrentKF,pKF,vvpMapPointMatches[i]);//第一次匹配
        /**
         * 若nmatches<20，剔除该候选帧
         * 注意这里使用Bow匹配较快，但是会有漏匹配
        */
        if(nmatches<20)
        {
            vbDiscarded[i] = true;
            continue;
        }
        else
        {
            //构建Sim3求解器
            Sim3Solver* pSolver = new Sim3Solver(mpCurrentKF,pKF,vvpMapPointMatches[i],mbFixScale);
            pSolver->SetRansacParameters(0.99,20,300);
            vpSim3Solvers[i] = pSolver;
        }

        nCandidates++;
    }

    bool bMatch = false;

    // Perform alternatively RANSAC iterations for each candidate
    // until one is succesful or all fail
    /**
     * RANSAC：利用上面匹配上的地图点（虽然匹配上了，但是空间位置相差了一个Sim3），用RANSAC方法求解Sim3位姿
     * 这里有可能求解不出Sim3,也就是虽然匹配满足，但是空间几何姿态不满足vvpMapPointMatches
    */
    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nInitialCandidates; i++)
        {
            if(vbDiscarded[i])
                continue;

            KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            Sim3Solver* pSolver = vpSim3Solvers[i];
            //用RANSAC方法求解SIM3，一共迭代五次，可以提高优化结果准确性
            cv::Mat Scm  = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
            if(!Scm.empty())
            {
                vector<MapPoint*> vpMapPointMatches(vvpMapPointMatches[i].size(), static_cast<MapPoint*>(NULL));
                for(size_t j=0, jend=vbInliers.size(); j<jend; j++)
                {
                    if(vbInliers[j])
                       vpMapPointMatches[j]=vvpMapPointMatches[i][j];
                }
                //根据计算出的Sim3(s, R, t)，去找更多的匹配点(SearchBySim3),更新vpMapPointMatches
                cv::Mat R = pSolver->GetEstimatedRotation();
                cv::Mat t = pSolver->GetEstimatedTranslation();
                const float s = pSolver->GetEstimatedScale();
                //使用sim3求出来的s,R,t通过SearchBySim3得到更多匹配
                matcher.SearchBySim3(mpCurrentKF,pKF,vpMapPointMatches,s,R,t,7.5);//第二次匹配
                /**
                 * 使用更新过的匹配，使用g2o优化Sim3位姿，这是内点数nInliers>20,才说明通过。
                 * 一旦找到闭环帧mpMatchedKF,则break,跳过对其他候选帧的判断
                */
                g2o::Sim3 gScm(Converter::toMatrix3d(R),Converter::toVector3d(t),s);//当前帧到回环帧的sim3变换
                const int nInliers = Optimizer::OptimizeSim3(mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, mbFixScale);

                // If optimization is succesful stop ransacs and continue
                //如果当前帧和回环帧之间存在超过20个匹配点，则认为这个当前帧和回环帧的sim3变换是有效的
                if(nInliers>=20)
                {
                    bMatch = true;
                    /**
                     * 这里记录的是匹配的闭环关键帧
                    */
                    mpMatchedKF = pKF;
                    //当前帧的sim3，其中s=1.0
                    g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()), Converter::toVector3d(pKF->GetTranslation()), 1.0);
                    //mg2oScw保存的是通过sim3对当前帧的位姿进行修正后的位姿，这个位姿的保存形式也是sim3
                    mg2oScw = gScm*gSmw;
                    mScw = Converter::toCvMat(mg2oScw);

                    mvpCurrentMatchedPoints = vpMapPointMatches;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        for(int i=0; i<nInitialCandidates; i++)
             mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        return false;
    }

    // Retrieve MapPoints seen in Loop Keyframe and neighbors
    //将MatchedKF共视帧取出
    vector<KeyFrame*> vpLoopConnectedKFs = mpMatchedKF->GetVectorCovisibleKeyFrames();
    //匹配的关键帧mpMatchedKF加入到和它共视的关键帧列表中
    vpLoopConnectedKFs.push_back(mpMatchedKF);
    mvpLoopMapPoints.clear();
    for(vector<KeyFrame*>::iterator vit=vpLoopConnectedKFs.begin(); vit!=vpLoopConnectedKFs.end(); vit++)
    {
        KeyFrame* pKF = *vit;
        vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad() && pMP->mnLoopPointForKF != mpCurrentKF->mnId)
                {
                    mvpLoopMapPoints.push_back(pMP);
                    pMP->mnLoopPointForKF=mpCurrentKF->mnId;
                }
            }
        }
    }

    // Find more matches projecting with the computed Sim3
    /**
     * 获取mpMatchedKF及其相连关键帧对应的地图的地图点。将这些地图点通过上面优化得到的Sim3(gScm->mScw) 
     * 变换投影到当前关键帧进行匹配，若匹配点>=40个，则返回true,进行闭环调整，否则，返回false, 
     * 线程暂停5ms后继续接收Tracking发送来的关键帧队列 
     * 注意这里得到的当前关键帧中匹配上闭环关键帧共视地图点(mvpCurrentMatchedPoints)
     * 将用于后面CorrectLoop时当时关键帧地图点的冲突融合
     * 到这里，不仅确保了当前关键帧与闭环帧之间匹配度高， 
     * 而且保证了闭环帧的共视图中的地图点和当前帧的特征点匹配度更高,说明该闭环帧是正确的
    */
   //SearchByProjection得到更多匹配点，mvpCurrentMatchedPoints
    matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints,10);//第三次匹配

    // If enough matches accept Loop
    int nTotalMatches = 0;
    for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
    {
        if(mvpCurrentMatchedPoints[i])
            nTotalMatches++;
    }
    // 如果最终的匹配点超过40个，则认为当前帧和闭环帧确实存在闭环关系
    if(nTotalMatches>=40)
    {
        for(int i=0; i<nInitialCandidates; i++)
            if(mvpEnoughConsistentCandidates[i]!=mpMatchedKF)
                mvpEnoughConsistentCandidates[i]->SetErase();
        return true;
    }
    else
    {
        for(int i=0; i<nInitialCandidates; i++)
            mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        return false;
    }

}
/**
 * 闭环纠正时，LocalMapper和Global BA必须停止。注意Global BA必须停止。 
 * 注意Global BA使用的的是单独的线程
 * 分为两步，第一步LoopFusion,第二步Essential Graph优化 
 * 其中Essential Graph包含三部分：
 * 1，共视关系很好的关键帧；
 * 2, spanning tree连接关系(父子关系) 
 * 3，闭环关键帧连接关系
 * 
*/
void LoopClosing::CorrectLoop()
{
    cout << "Loop detected!" << endl;

    // Send a stop signal to Local Mapping
    // Avoid new keyframes are inserted while correcting the loop
    //1.暂停LocalMapping，防止在闭环校正过程中插入新的关键帧
    mpLocalMapper->RequestStop();

    // If a Global Bundle Adjustment is running, abort it
    //2.暂停全局BA优化
    if(isRunningGBA())
    {
        unique_lock<mutex> lock(mMutexGBA);
        mbStopGBA = true;

        mnFullBAIdx++;

        if(mpThreadGBA)
        {
            mpThreadGBA->detach();
            delete mpThreadGBA;
        }
    }

    // Wait until Local Mapping has effectively stopped
    while(!mpLocalMapper->isStopped())
    {
        usleep(1000);
    }

    // Ensure current keyframe is updated
    //3.更新当前帧的共视相连关系
    mpCurrentKF->UpdateConnections();

    // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
    //4.得到与当前帧相连关键帧(包括当前关键帧)
    mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    mvpCurrentConnectedKFs.push_back(mpCurrentKF);
    /**
     * 使用计算出的Sim3对当前位姿进行调整，并且传播到当前帧相连的的关键帧
     * (相连关键帧之间相对位姿是知道的，通过当前帧的Sim3计算相连关键帧的Sim3).
     * 这样回环的两侧关键帧就对齐了，利用调整过的位姿更新这些向量关键帧对应的地图点
    */
    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
    CorrectedSim3[mpCurrentKF]=mg2oScw;
    cv::Mat Twc = mpCurrentKF->GetPoseInverse();


    {
        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
        //5.计算当前关键帧有共视关系的关键帧的Sim3位姿
        for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKFi = *vit;
            //这里获得的Tiw是相机到世界坐标系的变换
            cv::Mat Tiw = pKFi->GetPose();

            if(pKFi!=mpCurrentKF)
            {
                //这里获得的是当前帧到pKFi帧的相对变换
                /**
                 * Tiw是pKFi的相机到世界坐标系的变换矩阵
                 * Twc是mpCurrentKF的世界到相机坐标系的变换矩阵
                 * 则Tic为pKFi相对于mpCurrentKF的位姿变换
                */
                cv::Mat Tic = Tiw*Twc;
                cv::Mat Ric = Tic.rowRange(0,3).colRange(0,3);
                cv::Mat tic = Tic.rowRange(0,3).col(3);
                //构造pKFi的sim3变换，s设置为1
                g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric),Converter::toVector3d(tic),1.0);
                //当前帧的位姿固定不动，其它的关键帧根据相对关系得到sim3更新后的位姿
                g2o::Sim3 g2oCorrectedSiw = g2oSic*mg2oScw;
                //Pose corrected with the Sim3 of the loop closure
                //得到闭环g2o优化后各个关键帧的位姿
                CorrectedSim3[pKFi]=g2oCorrectedSiw;
            }

            cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
            cv::Mat tiw = Tiw.rowRange(0,3).col(3);
            g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw),Converter::toVector3d(tiw),1.0);
            //Pose without correction
            //当前帧相连关键帧，没有进行闭环优化的位姿
            NonCorrectedSim3[pKFi]=g2oSiw;
        }

        // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
        //6.利用调整过的位姿更新这些相连关键帧对应的MapPoint
        for(KeyFrameAndPose::iterator mit=CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;
            g2o::Sim3 g2oCorrectedSiw = mit->second;
            g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

            g2o::Sim3 g2oSiw =NonCorrectedSim3[pKFi];

            vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();
            for(size_t iMP=0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++)
            {
                MapPoint* pMPi = vpMPsi[iMP];
                if(!pMPi)
                    continue;
                if(pMPi->isBad())
                    continue;
                if(pMPi->mnCorrectedByKF==mpCurrentKF->mnId)
                    continue;

                // Project with non-corrected pose and project back with corrected pose
                //将闭环帧及其相连帧的地图点都投影到当前帧以及相连帧上
                cv::Mat P3Dw = pMPi->GetWorldPos();
                /**
                 * 将地图点的世界坐标转换为eigen当中的矩阵Matrix形式
                */
                Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
                /**
                 * g2oSiw.map(eigP3Dw)：g2oSiw是每个关键帧最初的尺度为1的位姿，将eigP3Dw使用最初的位姿进行相似变换求解得到未进行sim3优化前关键帧中的投影
                 * g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw))：g2oCorrectedSwi是每个关键帧的sim3纠正后的带尺度的位姿的逆，map投影后就是将关键帧中的相机坐标投影到世界坐标
                 * eigCorrectedP3Dw就是地图点在世界坐标系当中的坐标矩阵
                */
                Eigen::Matrix<double,3,1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));
                /**
                 * eigCorrectedP3Dw转换为Mat形式
                */
                cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
                pMPi->SetWorldPos(cvCorrectedP3Dw);
                pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
                pMPi->mnCorrectedReference = pKFi->mnId;
                pMPi->UpdateNormalAndDepth();
            }

            // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
            //将Sim3转为SE3,并调整关键帧位姿
            Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
            Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
            double s = g2oCorrectedSiw.scale();
            //归一化处理
            eigt *=(1./s); //[R t/s;0 1]

            cv::Mat correctedTiw = Converter::toCvSE3(eigR,eigt);
            //调整关键帧位姿
            pKFi->SetPose(correctedTiw);

            // Make sure connections are updated
            //更新关键帧连接关系
            pKFi->UpdateConnections();
        }

        // Start Loop Fusion
        // Update matched map points and replace if duplicated
        //7.开始进行闭环融合。投影匹配上的和Sim3计算过的地图点进行融合(就是替换成高质量的)
        for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
        {
            if(mvpCurrentMatchedPoints[i])
            {
                MapPoint* pLoopMP = mvpCurrentMatchedPoints[i];
                MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i);
                if(pCurMP)
                    pCurMP->Replace(pLoopMP);//进行地图点替换
                else
                {
                    mpCurrentKF->AddMapPoint(pLoopMP,i);
                    pLoopMP->AddObservation(mpCurrentKF,i);
                    pLoopMP->ComputeDistinctiveDescriptors();
                }
            }
        }

    }

    // Project MapPoints observed in the neighborhood of the loop keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    //8.将这些已纠正位姿的MapPoints与闭环MapPoints融合
    //CorrectedSim3的index为关键帧，值为经过g2o优化后的sim3变换位姿
    SearchAndFuse(CorrectedSim3);


    // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
    //9.得到由闭环形成的连接关系，存储在LoopConnections中
    map<KeyFrame*, set<KeyFrame*> > LoopConnections;

    for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

        // Update connections. Detect new links.
        //更新关键帧连接关系
        pKFi->UpdateConnections();
        LoopConnections[pKFi]=pKFi->GetConnectedKeyFrames();
        for(vector<KeyFrame*>::iterator vit_prev=vpPreviousNeighbors.begin(), vend_prev=vpPreviousNeighbors.end(); vit_prev!=vend_prev; vit_prev++)
        {
            LoopConnections[pKFi].erase(*vit_prev);
        }
        for(vector<KeyFrame*>::iterator vit2=mvpCurrentConnectedKFs.begin(), vend2=mvpCurrentConnectedKFs.end(); vit2!=vend2; vit2++)
        {
            LoopConnections[pKFi].erase(*vit2);
        }
    }

    // Optimize graph
    //10.Essential Graph优化
    Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, mbFixScale);

    mpMap->InformNewBigChange();

    // Add loop edge
    mpMatchedKF->AddLoopEdge(mpCurrentKF);
    mpCurrentKF->AddLoopEdge(mpMatchedKF);

    // Launch a new thread to perform Global Bundle Adjustment
    // 11.启动一个新线程去进行全局BA优化
    mbRunningGBA = true;
    mbFinishedGBA = false;
    mbStopGBA = false;
    //新建一个线程进行全局BA优化，对于地图中地图点和关键帧位姿更新都是在这里进行的
    mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment,this,mpCurrentKF->mnId);

    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release();    
    //12.更新最后一个闭环关键帧id
    mLastLoopKFid = mpCurrentKF->mnId;   
}
/**
 * mvpLoopMapPoints中存放的是匹配闭环关键帧和其共视关键帧当中的地图点
 * CorrectedPosesMap中存放的是当前关键帧的共视关键帧和经过sim3纠正后的位姿
*/
void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap)
{
    ORBmatcher matcher(0.8);

    for(KeyFrameAndPose::const_iterator mit=CorrectedPosesMap.begin(), mend=CorrectedPosesMap.end(); mit!=mend;mit++)
    {
        KeyFrame* pKF = mit->first;

        g2o::Sim3 g2oScw = mit->second;
        cv::Mat cvScw = Converter::toCvMat(g2oScw);

        vector<MapPoint*> vpReplacePoints(mvpLoopMapPoints.size(),static_cast<MapPoint*>(NULL));
        /**
         * 计算mvpLoopMapPoints当中的地图点通过cvScw位姿投影有所得像素帧特征点是否在pKF帧内，并且投影特征点的描述子和地图点的描述子是否匹配
         * 如果匹配成功，此时如果关键帧pKF当中不存在该地图点，则将该地图点添加到关键帧中，并更新地图点的观测帧
         * vpReplacePoints返回值为pKF中已经存在的有效地图点
        */
        matcher.Fuse(pKF,cvScw,mvpLoopMapPoints,4,vpReplacePoints);

        // Get Map Mutex
        //更新pKF当中已经存在的有效地图点
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
        const int nLP = mvpLoopMapPoints.size();
        for(int i=0; i<nLP;i++)
        {
            MapPoint* pRep = vpReplacePoints[i];
            if(pRep)
            {
                pRep->Replace(mvpLoopMapPoints[i]);
            }
        }
    }
}


void LoopClosing::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
        unique_lock<mutex> lock2(mMutexReset);
        if(!mbResetRequested)
            break;
        }
        usleep(5000);
    }
}

void LoopClosing::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlpLoopKeyFrameQueue.clear();
        mLastLoopKFid=0;
        mbResetRequested=false;
    }
}

void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF)
{
    cout << "Starting Global Bundle Adjustment" << endl;

    int idx =  mnFullBAIdx;
    Optimizer::GlobalBundleAdjustemnt(mpMap,10,&mbStopGBA,nLoopKF,false);

    // Update all MapPoints and KeyFrames
    // Local Mapping was active during BA, that means that there might be new keyframes
    // not included in the Global BA and they are not consistent with the updated map.
    // We need to propagate the correction through the spanning tree
    {
        unique_lock<mutex> lock(mMutexGBA);
        if(idx!=mnFullBAIdx)
            return;

        if(!mbStopGBA)
        {
            cout << "Global Bundle Adjustment finished" << endl;
            cout << "Updating map ..." << endl;
            mpLocalMapper->RequestStop();
            // Wait until Local Mapping has effectively stopped

            while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
            {
                usleep(1000);
            }

            // Get Map Mutex
            unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

            // Correct keyframes starting at map first keyframe
            list<KeyFrame*> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(),mpMap->mvpKeyFrameOrigins.end());

            while(!lpKFtoCheck.empty())
            {
                KeyFrame* pKF = lpKFtoCheck.front();
                const set<KeyFrame*> sChilds = pKF->GetChilds();
                cv::Mat Twc = pKF->GetPoseInverse();
                for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
                {
                    KeyFrame* pChild = *sit;
                    if(pChild->mnBAGlobalForKF!=nLoopKF)
                    {
                        cv::Mat Tchildc = pChild->GetPose()*Twc;
                        pChild->mTcwGBA = Tchildc*pKF->mTcwGBA;//*Tcorc*pKF->mTcwGBA;
                        pChild->mnBAGlobalForKF=nLoopKF;

                    }
                    lpKFtoCheck.push_back(pChild);
                }

                pKF->mTcwBefGBA = pKF->GetPose();
                pKF->SetPose(pKF->mTcwGBA);
                lpKFtoCheck.pop_front();
            }

            // Correct MapPoints
            const vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();

            for(size_t i=0; i<vpMPs.size(); i++)
            {
                MapPoint* pMP = vpMPs[i];

                if(pMP->isBad())
                    continue;

                if(pMP->mnBAGlobalForKF==nLoopKF)
                {
                    // If optimized by Global BA, just update
                    pMP->SetWorldPos(pMP->mPosGBA);
                }
                else
                {
                    // Update according to the correction of its reference keyframe
                    KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

                    if(pRefKF->mnBAGlobalForKF!=nLoopKF)
                        continue;

                    // Map to non-corrected camera
                    cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0,3).colRange(0,3);
                    cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0,3).col(3);
                    cv::Mat Xc = Rcw*pMP->GetWorldPos()+tcw;

                    // Backproject using corrected camera
                    cv::Mat Twc = pRefKF->GetPoseInverse();
                    cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
                    cv::Mat twc = Twc.rowRange(0,3).col(3);

                    pMP->SetWorldPos(Rwc*Xc+twc);
                }
            }            

            mpMap->InformNewBigChange();

            mpLocalMapper->Release();

            cout << "Map updated!" << endl;
        }

        mbFinishedGBA = true;
        mbRunningGBA = false;
    }
}

void LoopClosing::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LoopClosing::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LoopClosing::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool LoopClosing::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}


} //namespace ORB_SLAM
