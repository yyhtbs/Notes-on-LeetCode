class Solution {
public:
    vector<int> getRow(int rowIndex) {
        vector<int> outRows;
                
        for (int i = 0; i <= rowIndex; i++) {
            outRows.push_back(1);
            // Calculate the sum of the previous row
            for (int j = outRows.size() - 1 - 1; j >= 1; j--) {
                outRows[j] = outRows[j - 1] + outRows[j];
            }
            // add the first node; 1 ... * * * * 1 
            outRows[0] = 1;
        }
        return outRows;
    }
};
