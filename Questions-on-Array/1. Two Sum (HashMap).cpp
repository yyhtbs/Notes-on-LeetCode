class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, vector<int>> mapNums;
        // Add elements into a hashMap {value -> indices}
        // e.g. {value -> {idx1, idx2, ...}}
        for (int i = 0; i < nums.size(); i++) {
            if (mapNums.find(nums[i]) == mapNums.end()) {
                mapNums.insert(pair<int, vector<int>>
                           (nums[i], vector<int>(1, i)));
            }
            else {
                mapNums[nums[i]].push_back(i);
            }
        }
        vector<int> output;
        // traversing the list and calculate the residue, find if the residue does exist in the hashMap 
        for (int i = 0; i < nums.size(); i++) {
            int resi = target - nums[i];
            // Check if resi exists in the set
            if (mapNums.find(resi) != mapNums.end()) {
                // In case of the same Number
                // Check if the residue is the same as the value, [3, 2, 4] -> 6 = 3 + 3 is not a right answer
                if (resi == nums[i]) {
                // However, if the residue is the same as the value, and there does exist more than one index in the {indices}, 
                // they form a right answer
                    if (mapNums[resi].size() > 1) {
                      output.push_back(mapNums[resi][0]);
                      output.push_back(mapNums[resi][1]);
                      return output;
                    }
                    else {
                        continue;
                    }
                }
                else {
                    output.push_back(i);
                    output.push_back(mapNums[resi][0]);
                    return output;
                }
            }
            
        }
        
    }
};
