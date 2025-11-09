"""
Safe expression evaluator for AI
"""
import ast, operator, math
from typing import Any

class SafeEvaluator:
    """Safe AST-based expression evaluator"""
    
    def __init__(self, calculator_engine):
        self.engine = calculator_engine
        self.safe_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos
        }
        
        self.safe_functions = {
            'sqrt': math.sqrt,
            'sin': lambda x: math.sin(math.radians(x)),
            'cos': lambda x: math.cos(math.radians(x)),
            'tan': lambda x: math.tan(math.radians(x)),
            'log': math.log10,
            'ln': math.log,
            'exp': math.exp,
            'abs': abs,
            'fact': self._safe_factorial,
            'pow': pow,
        }
        
        self.safe_constants = {
            'pi': math.pi,
            'e': math.e,
            'ans': self._get_ans,
            'mem': self._get_mem
        }
    
    def _safe_factorial(self, n):
        """Safe factorial with bounds checking"""
        if n < 0 or n != int(n):
            raise ValueError("Factorial of negative or non-integer")
        if n > 10000:
            raise ValueError("Number too large for factorial")
        return math.factorial(int(n))
    
    def _get_ans(self):
        """Get last result"""
        return self.engine.last_result or 0
    
    def _get_mem(self):
        """Get memory value"""
        return self.engine.memory
    
    def safe_eval(self, expr: str) -> Any:
        """Safely evaluate a mathematical expression - ORIGINAL IMPLEMENTATION"""
        try:
            tree = ast.parse(expr, mode='eval')
            return self._eval_node(tree.body)
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _eval_node(self, node):
        """Recursively evaluate AST nodes"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self.safe_operators[type(node.op)]
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self.safe_operators[type(node.op)]
            return op(operand)
        elif isinstance(node, ast.Call):
            func_name = node.func.id
            if func_name not in self.safe_functions:
                raise ValueError(f"Unknown function: {func_name}")
            args = [self._eval_node(arg) for arg in node.args]
            return self.safe_functions[func_name](*args)
        elif isinstance(node, ast.Name):
            if node.id in self.safe_constants:
                value = self.safe_constants[node.id]
                return value() if callable(value) else value
            else:
                raise ValueError(f"Unknown variable: {node.id}")
        else:
            raise ValueError(f"Unsupported expression type: {type(node)}")