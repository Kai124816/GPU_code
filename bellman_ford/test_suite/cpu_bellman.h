#include <vector>
#include <limits>
#include <utility>

using namespace std;

const int INF = numeric_limits<int>::max();

vector<int> cpu_bellman(const vector<vector<pair<int, int>>>& adj_list);