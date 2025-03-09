#include <iostream>
#include <vector>
#include <memory>
#include <unordered_map>
#include <cmath>

// Define operation types for differentiation
enum class OpType
{
    NONE,
    ADD,
    MUL,
    LOG
};

struct Node
{
    OpType op;
    std::shared_ptr<Node> lhs, rhs;
    double value, adjoint;

    Node(OpType op, std::shared_ptr<Node> lhs, std::shared_ptr<Node> rhs, double val)
        : op(op), lhs(std::move(lhs)), rhs(std::move(rhs)), value(val), adjoint(0.0) {}
};

struct Tape
{
    std::vector<std::shared_ptr<Node>> nodes;

    std::shared_ptr<Node> addNode(OpType op, std::shared_ptr<Node> lhs, std::shared_ptr<Node> rhs, double value)
    {
        auto newNode = std::make_shared<Node>(op, lhs, rhs, value);
        nodes.push_back(newNode);
        return newNode;
    }

    void propagateAdjoints()
    {
        if (nodes.empty())
            return;
        nodes.back()->adjoint = 1.0;

        for (int i = nodes.size() - 1; i >= 0; --i)
        {
            auto &node = nodes[i];

            if (node->op == OpType::ADD)
            {
                if (node->lhs)
                    node->lhs->adjoint += node->adjoint;
                if (node->rhs)
                    node->rhs->adjoint += node->adjoint;
            }
            else if (node->op == OpType::MUL)
            {
                if (node->lhs)
                    node->lhs->adjoint += node->adjoint * node->rhs->value;
                if (node->rhs)
                    node->rhs->adjoint += node->adjoint * node->lhs->value;
            }
            else if (node->op == OpType::LOG && node->lhs)
            {
                node->lhs->adjoint += node->adjoint / node->lhs->value;
            }
        }
    }
};

struct Variable
{
    std::shared_ptr<Node> node;
    Tape *tape;

    Variable(double value, Tape *tape) : tape(tape)
    {
        node = tape->addNode(OpType::NONE, nullptr, nullptr, value);
    }

    Variable(std::shared_ptr<Node> node, Tape *tape) : node(std::move(node)), tape(tape) {}

    Variable operator+(const Variable &other) const
    {
        return Variable(tape->addNode(OpType::ADD, node, other.node, node->value + other.node->value), tape);
    }

    Variable operator*(const Variable &other) const
    {
        return Variable(tape->addNode(OpType::MUL, node, other.node, node->value * other.node->value), tape);
    }

    Variable log() const
    {
        return Variable(tape->addNode(OpType::LOG, node, nullptr, std::log(node->value)), tape);
    }

    friend Variable operator*(double scalar, const Variable &var)
    {
        auto scalar_node = var.tape->addNode(OpType::NONE, nullptr, nullptr, scalar);
        return Variable(var.tape->addNode(OpType::MUL, scalar_node, var.node, scalar * var.node->value), var.tape);
    }

    double value() const
    {
        return node->value;
    }

    double adjoint() const
    {
        return node->adjoint;
    }
};

// Template function f(T x[5]) with Fixed CSE
template <class T>
T f(T x[5])
{
    T y1 = x[2] * (5.0 * x[0] + x[1]);
    T y2 = y1.log();
    T y = (y1 + x[3] * y2) * (y1 + y2);
    return y;
}

int main()
{
    Tape tape;

    Variable x[5] = {Variable(1.0, &tape), Variable(2.0, &tape),
                     Variable(3.0, &tape), Variable(4.0, &tape),
                     Variable(5.0, &tape)};

    Variable y = f(x);

    tape.propagateAdjoints();

    std::cout << "Function Value: " << y.value() << std::endl;
    std::cout << "Gradients:" << std::endl;
    for (size_t i = 0; i < 5; ++i)
    {
        std::cout << "df/dx[" << i << "] = " << x[i].adjoint() << std::endl;
    }

    return 0;
}
