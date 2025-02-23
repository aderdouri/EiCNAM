#ifndef BINOMIALTREE_HPP
#define BINOMIALTREE_HPP

#include <torch/torch.h>
#include <vector>

class BinomialTree
{
public:
    BinomialTree(torch::Tensor S0, torch::Tensor K, torch::Tensor r, torch::Tensor sigma, torch::Tensor T, int N);

    torch::Tensor price();

private:
    torch::Tensor S0, K, r, sigma, T;
    int N;
    torch::Tensor tree;
    torch::Tensor option_tree;

    void generateTree();
    torch::Tensor priceOption();
};

#endif // BINOMIALTREE_HPP
