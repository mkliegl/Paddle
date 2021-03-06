# Design Doc: Gradient Operators Registration


## The Problem Posed

In our current operator registration mechanism, for each operator, the programmer should register a *gradient operator creator* function, which takes a C++ operator instance, and returns the corresponding gradient instance.

However, as we decided to separate the *compilation* and *execution* of DL models, we need to reshape the creator to take a protobuf `OpDesc` message, and returns a corresponding message.

More than that, the new registration mechanism need to support the fact that an operators' gradient computation might be a composition of operators.

## Current Implementation

OpInfos store in a association map which key is the operator type. The `grad_op_type` indicate associated gradient operator type. Operator can create gradient operator by `OpInfo::creator_` of gradient. The pseudo code is

```cpp
struct OpInfo {
  std::function<OperatorBase*(...)> creator_;
  std::string grad_op_type_;
  ...
};

map<string, OpInfo> OpInfoMap;

OperatorBase* CreateGradientOperator(const OperatorBase& op) {
  return OpInfoMap.at(op.Type()).creator_(...);
}
```

## Proposed Solution

The mapping relationship between an operator and its gradient operators is a function. The interface of that function is:

```cpp
// (OpDesc) --> vector<OpDesc>
using GradOpDescMaker = std::function<std::vector<OpDesc>(const OpDesc&)>;
```

The function take a `OpDesc` of the forward operator and return one or many gradient operator descriptions.

The `GradOpDescMaker` will be registered in `OpInfo`, to replace `grad_op_type_` field. The `OpInfo` should be

```cpp
struct OpInfo {
  GradOpDescMaker grad_op_maker_;
  ...
};
```

The `grad_op_maker_ ` is `nullptr` if the operator does not have associated gradient operators.

We should chagne register macros at the same time. In the current solution, there is no difference between forwarding operators and backward operators. So `REGISTER_OP` just register one operator. If the `REGISTER_OPERATOR ` contains `OpProtoAndCheckerMaker` and `GradOpDescMaker`, we just list them in the same macro. It can be done by a macro contains `__VA_ARGS__`.

The user interface should be

```cpp
vector<OpDesc> MinusOpGradMaker(OpDesc) {...}
REGISTER_OPERATOR(minus, MinusOp, MinusOpProtoAndCheckerMaker, SumOpGradMaker);
// Developers can still manually implement gradient operator.
REGISTER_OPERATOR(minus_grad, MinusGradOp);
```

The interface of current `REGISTER_OP` macro could not be changed. In `REGISTER_OP`, it will invoke `REGISTER_OPERATOR` two times and generate GradOpDescMaker inside.

```cpp
REGISTER_OP(minus, MinusOp, MinusOpProtoAndCheckerMaker, minus_grad, MinusGradOp);
```
