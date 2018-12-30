class Solution {
public:
    vector<vector<int>> generate(int numRows) {
        vector<vector<int>> outRows;
        if (numRows == 0)
            return outRows;
        
        outRows.push_back(vector<int>(1, 1));
        
        for (int i = 1; i < numRows; i++) {
            // Generate row from prevRow
            vector<int>& tempRow = outRows[i - 1];
            vector<int> currRow;
            // add the first node; 1 * * * * ... 
            currRow.push_back(1);
            // Calculate the sum of the previous row
            for (int j = 0; j < tempRow.size() - 1; j++) {
                currRow.push_back(tempRow[j] + tempRow[j + 1]);
            }
            // add the first node; 1 ... * * * * 1 
            currRow.push_back(1);
            // append the new row
            outRows.push_back(currRow);
        }
        return outRows;
    }
};
