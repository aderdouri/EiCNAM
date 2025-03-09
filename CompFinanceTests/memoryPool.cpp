#include <iostream>
#include <vector>
#include <memory>
#include <unordered_map>
#include <cmath>
#include <cstdlib>

// Define operation types for differentiation
enum class OpType
{
    NONE,
    ADD,
    MUL,
    LOG
};

// Define a structure for nodes in the dynamic computation graph
struct Node
{
    OpType op;
    Node *lhs;
    Node *rhs;
    double value;
    double adjoint;
    Node *next_free; // Used for memory pooling

    Node(OpType op, Node *lhs, Node *rhs, double val)
        : op(op), lhs(lhs), rhs(rhs), value(val), adjoint(0.0), next_free(nullptr) {}
};

// **Custom Memory Pool for Node Allocations**
class NodePool
{
    static constexpr size_t POOL_SIZE = 10000; // Preallocate memory for 10,000 nodes
    Node *pool;
    Node *free_list;

public:
    NodePool()
    {
        pool = static_cast<Node *>(std::malloc(POOL_SIZE * sizeof(Node)));
        if (!pool)
            throw std::bad_alloc();

        // Initialize free list
        for (size_t i = 0; i < POOL_SIZE; ++i)
        {
            new (&pool[i]) Node(OpType::NONE, nullptr, nullptr, 0.0); // Placement new
            pool[i].next_free = (i == POOL_SIZE - 1) ? nullptr : &pool[i + 1];
        }
        free_list = pool;
    }

    ~NodePool() { std::free(pool); }

    Node *allocate(OpType op, Node *lhs, Node *rhs, double value)
    {
        if (!free_list)
        {
            std::cerr << "ERROR: Memory pool exhausted! Increase POOL_SIZE.\n";
            std::exit(1);
        }

        Node *new_node = free_list;
        free_list = free_list->next_free; // Move to next available node

        // Construct the new node
        new_node->op = op;
        new_node->lhs = lhs;
        new_node->rhs = rhs;
        new_node->value = value;
        new_node->adjoint = 0.0;
        new_node->next_free = nullptr;
        return new_node;
    }

    void deallocate(Node *node)
    {
        node->next_free = free_list;
        free_list = node;
    }
};

// Tape class with Memory Pool Integration
struct Tape
{
    std::vector<Node *> nodes;
    std::unordered_map<std::string, Node *> expression_cache; // CSE cache
    NodePool pool;

    // Generate a unique key for Common Subexpression Elimination
    std::string generateKey(OpType op, double lhs_value, double rhs_value)
    {
        return std::to_string(static_cast<int>(op)) + "_" +
               std::to_string(lhs_value) + "_" + std::to_string(rhs_value);
    }

    Node *addNode(OpType op, Node *lhs, Node *rhs, double value)
    {
        std::string key = generateKey(op, lhs ? lhs->value : 0.0, rhs ? rhs->value : 0.0);

        // **Ensure unique computation nodes (Fix CSE issue)**
        if (expression_cache.find(key) != expression_cache.end())
        {
            return expression_cache[key]; // Reuse existing node
        }

        Node *newNode = pool.allocate(op, lhs, rhs, value);
        nodes.push_back(newNode);
        expression_cache[key] = newNode; // Store node reference
        return newNode;
    }

    void propagateAdjoints()
    {
        if (nodes.empty())
            return;
        nodes.back()->adjoint = 1.0;

        for (int i = nodes.size() - 1; i >= 0; --i)
        {
            Node *node = nodes[i];

            if (!node)
                continue;

            if (node->op == OpType::ADD)
            {
                if (node->lhs)
                    node->lhs->adjoint += node->adjoint;
                if (node->rhs)
                    node->rhs->adjoint += node->adjoint;
            }
            else if (node->op == OpType::MUL)
            {
                if (node->lhs && node->rhs)
                {
                    node->lhs->adjoint += node->adjoint * node->rhs->value;
                    node->rhs->adjoint += node->adjoint * node->lhs->value;
                }
            }
            else if (node->op == OpType::LOG)
            {
                if (node->lhs)
                {
                    node->lhs->adjoint += node->adjoint / node->lhs->value;
                }
            }
        }
    }
};

// Variable class with Custom Memory Pool
struct Variable
{
    Node *node;
    Tape *tape;

    Variable(double value, Tape *tape) : tape(tape)
    {
        node = tape->addNode(OpType::NONE, nullptr, nullptr, value);
    }

    Variable(Node *node, Tape *tape) : node(node), tape(tape) {}

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

    double value() const { return node->value; }
    double adjoint() const { return node->adjoint; }
};

Variable operator*(double lhs, const Variable &rhs)
{
    return Variable(rhs.tape->addNode(OpType::MUL, rhs.node, nullptr, lhs * rhs.node->value), rhs.tape);
}

Variable operator+(double lhs, const Variable &rhs)
{
    return Variable(rhs.tape->addNode(OpType::ADD, rhs.node, nullptr, lhs + rhs.node->value), rhs.tape);
}

// **Function f(x) with Memory Pool Optimization**
template <class T>
T f(T x[5])
{
    T y1 = x[2] * (5.0 * x[0] + x[1]);
    T y2 = y1.log();
    T y = (y1 + x[3] * y2) * (y1 + y2);
    return y;
}

// **Main function for testing**
int main()
{
    Tape tape;

    // Define input variables dynamically
    Variable x[5] = {Variable(1.0, &tape), Variable(2.0, &tape),
                     Variable(3.0, &tape), Variable(4.0, &tape),
                     Variable(5.0, &tape)};

    // Compute function value dynamically
    Variable y = f(x);

    // Perform backpropagation
    tape.propagateAdjoints();

    // Output results
    std::cout << "Function Value: " << y.value() << std::endl;
    std::cout << "Gradients:\n";
    for (size_t i = 0; i < 5; ++i)
    {
        std::cout << "df/dx[" << i << "] = " << x[i].adjoint() << std::endl;
    }

    return 0;
}
