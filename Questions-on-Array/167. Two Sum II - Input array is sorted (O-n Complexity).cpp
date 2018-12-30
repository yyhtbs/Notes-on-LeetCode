// Assuming i and j are the positions (p) of two numbers from left and right
// there does exist a RULE that let i and j approach the suitable value such that i + j == t
// If the following two conditions hold,
//     C1. any p1 which first reaches one of the final value (p1*) will always stay at p1*, i.e. never pass it
//     C2. when p1* is reached, any p2 will converge to the final value (p2*)


// C2 constantly holds because the array is already sorted
// C1 holds , if increasing p1 when p1 + p2 < t and decreasing p2 when p1 + p2 > t
// Considering the case that p1 passes p1*, i.e. means p1 and p2 forms a number which is larger or smaller than t.
// If p1 and p2 are both smaller than p1* and p2*, i.e. p1 + p2 < p1* + p2* and p2 passes p2* first. 
// Based on R1, this can be avoided. Because moving p2 will always lower down the value
// when p1 < p1*, p2 = p2*, i.e. p1 + p2 < p1* + p2*, it will increase p1 instead of decreasing p2.


class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target) {
        vector<int> outNums;
        
        int i = 0;
        int j = numbers.size() - 1;
        
        int a = numbers[i];
        int b = numbers[j];
        
        while (a + b != target) {
            if (a + b < target) {
                i++;
                a = numbers[i];
            }
            else if (a + b > target) {
                j--;
                b = numbers[j];
            }
        }
        outNums.push_back(i + 1);
        outNums.push_back(j + 1);
        
        return outNums;
    }
};
