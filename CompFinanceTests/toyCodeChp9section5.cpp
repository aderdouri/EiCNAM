/*
	Toy code contained in chapter 9, sections 5 (pages 349 to 355).
*/
#include <memory>
#include <string>
#include <vector>
#include <queue>
#include <iostream>

using namespace std;

class Node
{
protected:
	vector<Node *> myArguments;

	double myResult;
	double myAdjoint = 0.0;

public:
	virtual ~Node() = default;

	// Access result
	double result()
	{
		return myResult;
	}

	// Access adjoint
	double &adjoint()
	{
		return myAdjoint;
	}

	void resetAdjoints()
	{
		for (auto argument : myArguments)
			argument->resetAdjoints();
		myAdjoint = 0.0;
	}

	virtual void propagateAdjoint() = 0;
};

class PlusNode : public Node
{
public:
	PlusNode(Node *lhs, Node *rhs)
	{
		myArguments.resize(2);
		myArguments[0] = lhs;
		myArguments[1] = rhs;

		// Eager evaluation
		myResult = lhs->result() + rhs->result();
	}

	void propagateAdjoint() override
	{
		myArguments[0]->adjoint() += myAdjoint;
		myArguments[1]->adjoint() += myAdjoint;
	}
};

class TimesNode : public Node
{
public:
	TimesNode(Node *lhs, Node *rhs)
	{
		myArguments.resize(2);
		myArguments[0] = lhs;
		myArguments[1] = rhs;

		// Eager evaluation
		myResult = lhs->result() * rhs->result();
	}

	void propagateAdjoint() override
	{
		myArguments[0]->adjoint() += myAdjoint * myArguments[1]->result();
		myArguments[1]->adjoint() += myAdjoint * myArguments[0]->result();
	}
};

class LogNode : public Node
{
public:
	LogNode(Node *arg)
	{
		myArguments.resize(1);
		myArguments[0] = arg;

		// Eager evaluation
		myResult = log(arg->result());
	}

	void propagateAdjoint() override
	{
		myArguments[0]->adjoint() += myAdjoint / myArguments[0]->result();
	}
};

class Leaf : public Node
{
public:
	Leaf(double val)
	{
		myResult = val;
	}

	double getVal()
	{
		return myResult;
	}

	void setVal(double val)
	{
		myResult = val;
	}

	void propagateAdjoint() override {}
};

class Number
{
	Node *myNode;

public:
	// The tape, as a public static member
	static vector<unique_ptr<Node>> tape;

	// Create node and put it on the tape
	Number(double val)
		: myNode(new Leaf(val))
	{
		tape.push_back(unique_ptr<Node>(myNode));
	}

	Number(Node *node)
		: myNode(node) {}

	Node *node()
	{
		return myNode;
	}

	void setVal(double val)
	{
		// Cast to Leaf, only leaves can be changed
		dynamic_cast<Leaf *>(myNode)->setVal(val);
	}

	double getVal()
	{
		// Only leaves can be read
		return dynamic_cast<Leaf *>(myNode)->getVal();
	}

	// Accessor/setter, from the inputs
	double &adjoint()
	{
		return myNode->adjoint();
	}

	// Propagator
	void propagateAdjoints()
	{
		myNode->resetAdjoints();
		myNode->adjoint() = 1.0;

		// Find my node on the tape, searching from the last
		auto it = tape.rbegin(); // last node on the tape
		while (it->get() != myNode)
			++it; // reverse iter: ++ means go back

		// Now it is on my node
		// Conduct propogation in reverse order
		while (it != tape.rend())
		{
			(*it)->propagateAdjoint();
			++it; // really means --
		}
	}
};

vector<unique_ptr<Node>> Number::tape;

Number operator+(Number lhs, Number rhs)
{
	// Create node: note eagerly computes result
	Node *n = new PlusNode(lhs.node(), rhs.node());
	// Put on tape
	Number::tape.push_back(unique_ptr<Node>(n));
	// Return result
	return n;
}

Number operator*(Number lhs, Number rhs)
{
	// Create node: note eagerly computes result
	Node *n = new TimesNode(lhs.node(), rhs.node());
	// Put on tape
	Number::tape.push_back(unique_ptr<Node>(n));
	// Return result
	return n;
}

Number log(Number arg)
{
	// Create node: note eagerly computes result
	Node *n = new LogNode(arg.node());
	// Put on tape
	Number::tape.push_back(unique_ptr<Node>(n));
	// Return result
	return n;
}

template <class T>
T f(T x[5])
{
	auto y1 = x[2] * (5.0 * x[0] + x[1]);
	auto y2 = log(y1);
	auto y = (y1 + x[3] * y2) * (y1 + y2);
	return y;
}

int main()
{
	Number x[5] = {1.0, 2.0, 3.0, 4.0, 5.0};

	// Evaluate and build the tape
	Number y = f(x);

	// Propagate adjoints through the tape in reverse order
	y.propagateAdjoints();

	// Get derivatives
	for (size_t i = 0; i < 5; ++i)
	{
		cout << "a" << i << " = " << x[i].adjoint() << endl;
	}

	// 950.736, 190.147, 443.677, 73.2041, 0
}
