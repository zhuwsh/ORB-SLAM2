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

#include "KeyFrameDatabase.h"

#include "KeyFrame.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"

#include<mutex>

using namespace std;

namespace ORB_SLAM2
{

KeyFrameDatabase::KeyFrameDatabase (const ORBVocabulary &voc):
    mpVoc(&voc)
{
    mvInvertedFile.resize(voc.size());
}


void KeyFrameDatabase::add(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutex);

    for(DBoW2::BowVector::const_iterator vit= pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
        mvInvertedFile[vit->first].push_back(pKF);
}

void KeyFrameDatabase::erase(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutex);

    // Erase elements in the Inverse File for the entry
    for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
    {
        // List of keyframes that share the word
        list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

        for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
        {
            if(pKF==*lit)
            {
                lKFs.erase(lit);
                break;
            }
        }
    }
}

void KeyFrameDatabase::clear()
{
    mvInvertedFile.clear();
    mvInvertedFile.resize(mpVoc->size());
}

/**
 * 检测得到回环候选帧
 * pKF  当前关键帧
 * minScore 与当前帧共视的所有关键帧中Bow的最小分值
*/
vector<KeyFrame*> KeyFrameDatabase::DetectLoopCandidates(KeyFrame* pKF, float minScore)
{
    //spConnectedKeyFrames是一个set定义的变量，set中装的是在covisibility graph中与关键帧pKF相关联的关键帧。
    // 1.获取和当前关键帧有共视关系的关键帧
    set<KeyFrame*> spConnectedKeyFrames = pKF->GetConnectedKeyFrames();
    list<KeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current keyframes
    // Discard keyframes connected to the query keyframe
    //搜索所有和当前关键帧共享单词的关键帧，丢弃和查询的关键帧相连接的关键帧
    {
        unique_lock<mutex> lock(mMutex);
        //遍历当前帧的BowVector，找出与当前帧有公共单词的关键帧,但不包括与当前帧相连的关键帧
        //2.遍历当前关键帧的词向量，通过词向量查看这些词都在哪些关键帧中出现过
        for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++)
        {
            //在InvertedFile中查找该word都在哪些关键帧中出现过
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];
            //遍历和当前帧共享相同单词的关键帧列表
            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                /**
                 * 下面一段程序是计算pKFi和pKF共视的words个数
                 * 到了这里就是说pKF与pKFi有共视关系，但是并不是pKFi都可以被认为是闭环候选帧
                 * 只有当pKFi不是在covisibility graph中与pKF直接相连的关键帧才有机会入围
                */
                KeyFrame* pKFi=*lit;
                /**
                 * 如果pKFi是第一次被pKF查询，那么先初始化成员变量mnLoopwords为0
                 * 经判断pKFi确实不与pKF直接相连则将成员变量mnLoopQuery设置为pKF的Id号码
                 * 表示pKFi已经被pKF查询过，下次再次查询pKFi的时候
                 * pKFi->mnLoopQuery = pKF->mnId 则直接让 pKFi->mnLoopWords++;
                 * 并且将pKFi插入到lKFsSharingwords中去。
                */
                if(pKFi->mnLoopQuery != pKF->mnId)
                {
                    pKFi->mnLoopWords=0;
                    //判断pKFi没有在spConnectedKeyFrames容器中出现过，则将pKFi加入到共享词的帧列表中
                    //spConnectedKeyFrames.count(pKFi)是查询pKFi在spConnectedKeyFrames中出现的次数，返回值为0或者1
                    //进入以下分支，表示pKFi和pKF不是相连的关键帧
                    if(!spConnectedKeyFrames.count(pKFi))
                    {
                        pKFi->mnLoopQuery=pKF->mnId;
                        lKFsSharingWords.push_back(pKFi);
                    }
                }
                pKFi->mnLoopWords++;
            }
        }
    }
    //此时，lKFsSharingWords中存储的是与当前帧有公共单词，但是不与当前帧相连的关键帧
    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lScoreAndMatch;

    // Only compare against those keyframes that share enough words
    // 3.统计候选帧中与pKF具有共同单词最多的单词数
    int maxCommonWords=0;
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnLoopWords > maxCommonWords)
            maxCommonWords=(*lit)->mnLoopWords;
    }

    int minCommonWords = maxCommonWords*0.8f;

    int nscores=0;

    // Compute similarity score. Retain the matches whose score is higher than minScore
    //4. 筛选共有单词大于minCommonWords且Bow得分大于minscore的关键帧，连同分值存放在lScoreAndMatch中
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;

        if(pKFi->mnLoopWords > minCommonWords)
        {
            nscores++;

            float si = mpVoc->score(pKF->mBowVec,pKFi->mBowVec);

            pKFi->mLoopScore = si;
            if(si >= minScore)
                lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }
    //此时，lScoreAndMatch中存放的是和当前帧共有单词大于minCommonWords且Bow得分大于minscore的关键帧
    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = minScore;

    // Lets now accumulate score by covisibility
    /**
     * 5.遍历lScoreAndMatch列表，对其中每个关键帧计算其和共视关系最好的10个关键帧中的最高得分bestScore和他们的累计分值
     * lAccScoreAndMatch中存储累计分值和最高分值的关键帧
    */
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first;
        float accScore = it->first;
        KeyFrame* pBestKF = pKFi;
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            if(pKF2->mnLoopQuery == pKF->mnId && pKF2->mnLoopWords > minCommonWords)
            {
                accScore += pKF2->mLoopScore;
                if(pKF2->mLoopScore > bestScore)
                {
                    pBestKF = pKF2;
                    bestScore = pKF2->mLoopScore;
                }
            }
        }
        //pBestKF为得分最高的关键帧
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        if(accScore > bestAccScore)
            bestAccScore = accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    //得到阈值
    float minScoreToRetain = 0.75f*bestAccScore;

    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpLoopCandidates;
    vpLoopCandidates.reserve(lAccScoreAndMatch.size());
    // 6.遍历每组中得分最高的关键帧，对累计分值大于最好累计分值的75%的关键帧视为闭环候选关键帧，并加入到闭环候选关键帧列表中
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        //得分大于阈值
        if(it->first > minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            //该帧pKFi没有加入过spAlreadyAddedKF
            if(!spAlreadyAddedKF.count(pKFi))
            {
                vpLoopCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpLoopCandidates;
}
/**
 * 检测重定位候选关键帧
 * 因为当前的系统状态变为了LOST,所以需要进行重定位。
*/
vector<KeyFrame*> KeyFrameDatabase::DetectRelocalizationCandidates(Frame *F)
{
    list<KeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current frame
    //查找所有与当前帧共享同一个单词的关键帧
    {
        unique_lock<mutex> lock(mMutex);
        //遍历当前帧F的单词向量
        for(DBoW2::BowVector::const_iterator vit=F->mBowVec.begin(), vend=F->mBowVec.end(); vit != vend; vit++)
        {
            //获取和当前帧共享相同单词的关键帧列表
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];
            //遍历共享同一单词的关键帧
            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit;
                if(pKFi->mnRelocQuery!=F->mnId)
                {
                    pKFi->mnRelocWords=0;
                    pKFi->mnRelocQuery=F->mnId;
                    lKFsSharingWords.push_back(pKFi);
                }
                pKFi->mnRelocWords++;
            }
        }
    }
    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    // Only compare against those keyframes that share enough words
    //只比较共同含有足够多的单词的关键帧
    int maxCommonWords=0;
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnRelocWords > maxCommonWords)
            maxCommonWords=(*lit)->mnRelocWords;
    }

    int minCommonWords = maxCommonWords*0.8f;

    list<pair<float,KeyFrame*> > lScoreAndMatch;

    int nscores=0;

    // Compute similarity score.
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;

        if(pKFi->mnRelocWords > minCommonWords)
        {
            nscores++;
            //计算当前帧F和关键帧pKFi之间的词包向量的分值
            float si = mpVoc->score(F->mBowVec,pKFi->mBowVec);
            pKFi->mRelocScore=si;
            lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = 0;

    // Lets now accumulate score by covisibility
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first;
        float accScore = bestScore;
        KeyFrame* pBestKF = pKFi;
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            if(pKF2->mnRelocQuery!=F->mnId)
                continue;

            accScore+=pKF2->mRelocScore;
            if(pKF2->mRelocScore>bestScore)
            {
                pBestKF=pKF2;
                bestScore = pKF2->mRelocScore;
            }

        }
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    // 返回所有分值高于0.75×bestScore的关键帧
    float minScoreToRetain = 0.75f*bestAccScore;
    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpRelocCandidates;
    vpRelocCandidates.reserve(lAccScoreAndMatch.size());
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        const float &si = it->first;
        if(si > minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))
            {
                vpRelocCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpRelocCandidates;
}

} //namespace ORB_SLAM
