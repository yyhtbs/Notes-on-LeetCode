class Solution {
public:
    vector<int> sortArrayByParity(vector<int>& A) {
        vector<int> B(A);
        
        int j = 0, k = A.size() - 1;
        
        for (int i = 0; i < A.size(); i++) {
            if (A[i] % 2 == 0) {
                B[j] = A[i];
                j++;
            }
            else {
                B[k] = A[i];
                k--;
            }
        }
        return B;
    }
};
