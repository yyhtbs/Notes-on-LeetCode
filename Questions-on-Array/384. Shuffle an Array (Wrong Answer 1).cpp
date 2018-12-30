class Solution {
public:
    vector<int> m_numsBackup;
    vector<int> m_nums;
    Solution(vector<int> nums) {
        m_numsBackup = nums; 
        m_nums = nums;
    }
    
    /** Resets the array to its original configuration and return it. */
    vector<int> reset() {
        return m_numsBackup;
    }
    
    /** Returns a random shuffling of the array. */
    vector<int> shuffle() {
        vector<int> outNums;
        while (m_nums.size() != 0) {
            if (rand() % 2 == 0) {
                outNums.push_back(m_nums[m_nums.size() - 1]);
                m_nums.pop_back();
            }
        }
        m_nums = outNums;
        return outNums;
    }
};
