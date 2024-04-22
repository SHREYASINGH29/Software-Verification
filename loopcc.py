# write lark parser for loopcc
import sys
from lark import Lark, Transformer, v_args
from lark.exceptions import LarkError
import subprocess

# define the grammar
grammar = r"""
%import common.CNAME  // Import common token CNAME for C-like identifiers
%import common.INT    // Import common token INT for integers
%import common.WS     // Import common token WS for whitespace
%import common.NEWLINE
%ignore WS           // Instruct Lark to ignore whitespaces
%ignore NEWLINE      // Ignore new lines for flexibility
COMMENT: /\/\/[^\n]*/  // Define comments style

?start: statement+

?statement: loop_declaration
          | seq_nest_main
          | array_operation
          | COMMENT

loop_declaration: "Loop" loop_name "(" stride ")" ":" range ";"

seq_nest_main: seq_nest ";"

seq_nest: seq_operation
          | nest_operation
		  | loop_name

seq_operation: "Seq" "(" content ")"
nest_operation: "Nest" "(" content ")"

content: seq_nest ("," seq_nest)*

array_operation: array_element "+=" array_element "*" array_element ";"

loop_name: CNAME
stride: expression
range: startidx "->" end
startidx: expression
end: expression

array_element: CNAME "[" index "]""[" index "]"
index: expression

expression: factor
          | expression "+" expression   -> add
          | expression "-" expression   -> subtract
          | expression "*" expression   -> multiply
          | expression "/" expression   -> divide

?factor: CNAME    -> var
       | INT      -> number
       | "(" expression ")"
"""

# define the transformer
@v_args(inline=True)
class LoopccTransformer(Transformer):
    def __init__(self, prefix, assignments={}, _vars=[], startIdx={}, loopDepth={}, operations=[], assumptions=[]):
        self.prefix = prefix
        self.assignments = assignments
        self._vars = _vars
        self.startIdx = startIdx
        self.loopDepth = {}
        self.current_depth = 0
        self.loopDepth = loopDepth
        self.operations = operations
        self.assumptions = assumptions

    def array_operation(self, *args):
        args = list(args)
        for arg in args:
            self.operations.append(arg)
        return None

    def array_element(self, *args):
        return [args[0].split(self.prefix)[-1], args[1], args[2]]

    def index(self, *args):
        return args[0]

    def seq_nest_main(self, *args):
        nest_seq = args[0]
        depth_stack = []
        nest_start = 0
        current_depth = 0
        current_loop = ""
        last_char = ""
        for idx in range(len(nest_seq)):
            if nest_seq[idx] == "[":
                nest_start = 1
                depth_stack.append(current_depth)
            elif nest_seq[idx] == "]":
                self.loopDepth[current_loop] = current_depth 
                current_depth = depth_stack.pop()
                nest_start = 0
            elif nest_seq[idx] == "(":
                nest_start = 0
            elif nest_seq[idx] == ")":
                self.loopDepth[current_loop] = current_depth
            elif nest_seq[idx] == ",":
                if last_char == "]" or last_char == ")":
                    pass
                else:
                    self.loopDepth[current_loop] = current_depth
                if nest_start == 1:
                    current_depth += 1
                current_loop = ""
            else:
                current_loop += nest_seq[idx]
            last_char = nest_seq[idx]
        return None

    def seq_nest(self, *args):
        return args[0]

    def seq_operation(self, *args):	
        return "(" + ",".join(args) + ")"

    def nest_operation(self, *args):
        return "[" + ",".join(args) + "]"

    def content(self, *args):
        return ",".join(args)

    def loop_declaration(self, loop_name, stride, range):
        start = range[0]
        end = range[1]
        self.startIdx[loop_name] = start
        self.assignments[loop_name] = "(" + end + " - " + start + ")" + " div " + stride
        self.assumptions.append(end + " mod " + stride + " == 0")
        self.assumptions.append(stride + " <= " + end + " div 2")
        self.assumptions.append(stride + " > 0")
        return None

    def loop_name(self, name_token):
        name = str(name_token[0])
        if name not in self.loopDepth:
            self.loopDepth[name] = self.current_depth
        return name

    def loop_name(self, name):
        return name

    def factor(self, value):
        return value

    def expression(self, *args):
        return args[0]

    def startidx(self, value):
        return value

    def end(self, value):
        return value

    def range(self, startidx, end):
        return [startidx, end]

    def multiply(self, left, right):
        return "(" + left + " * " + right + ")"

    def divide(self, left, right):
        return "(" + left + " div " + right + ")"

    def add(self, left, right):
        return "(" + left + " + " + right + ")"

    def subtract(self, left, right):
        return "(" + left + " - " + right + ")"

    def stride(self, value):
        return value

    def var(self, value):
        return value

    def number(self, value):
        return value

    def CNAME(self, value):
        name = self.prefix + str(value)
        if name not in self._vars:
            self._vars.append(name)
        return name

    def INT(self, value):
        return str(value)

# create parse object
def create_parser(prefix, assignments={}, _vars=[], startIdx={}, loopDepth={}, operations=[], assumptions=[]):
    return Lark(grammar, parser="lalr", transformer=LoopccTransformer(prefix, assignments, _vars, startIdx, loopDepth, operations, assumptions))

# make a class for storing boogie code
class BoogieCode:
    def __init__(self, assignments={}, _vars=[], startIdx={}, loopDepth={}, equality=[], assumptions=[], UniqueIdx=[], CIndex=[]):
        self.startIdx = startIdx
        self.vars = _vars
        self.code = ""
        self.loopDepth = loopDepth
        self.equality = equality
        self.assignments = assignments
        self.assumptions = assumptions
        self.UniqueIdx = UniqueIdx
        self.CIndex = CIndex

    def print_code(self, filename):
        self.epilogue()
        self.init_vars()
        self.init_overlap_vars()
        self.emit_assumptions()
        self.init_assignments()
        self.modify_assignments()
        self.init_overlap_assignments()
        self.init_equalities()
        self.emit_unique_idx()
        self.emit_same_area()
        self.prologue()
        # print(self.code)
        with open(filename, "w") as f:
            f.write(self.code)

    def epilogue(self):
        self.code += "procedure main()\n"
        self.code += "{\n"

    def prologue(self):
        self.code += "}\n"

    def init_vars(self):
        for var in self.vars:
            self.code += "var " + var + ": int;\n"

    def init_overlap_vars(self):
        self.code += "var __N1: int;\n"
        self.code += "var __N2: int;\n"
        self.code += "var __Data1StartI: [int]int;\n"
        self.code += "var __Data2StartI: [int]int;\n"
        self.code += "var __Data1SizeI: [int]int;\n"
        self.code += "var __Data2SizeI: [int]int;\n" 
        self.code += "var __Data1StartJ: [int]int;\n"
        self.code += "var __Data2StartJ: [int]int;\n"
        self.code += "var __Data1SizeJ: [int]int;\n"
        self.code += "var __Data2SizeJ: [int]int;\n" 

    def init_overlap_assignments(self):
        self.code += "__N1 := " + str(len(self.CIndex[0])) + ";\n"
        self.code += "__N2 := " + str(len(self.CIndex[1])) + ";\n"

        for i in range(len(self.CIndex[0])):
            self.code += "__Data1StartI[" + str(i) + "] := " + self.startIdx[self.CIndex[0][i][0]] + ";\n"
            self.code += "__Data1SizeI[" + str(i) + "] := " + self.CIndex[0][i][0] + ";\n"

        for i in range(len(self.CIndex[1])):
            self.code += "__Data2StartI[" + str(i) + "] := " + self.startIdx[self.CIndex[1][i][0]] + ";\n"
            self.code += "__Data2SizeI[" + str(i) + "] := " + self.CIndex[1][i][0] + ";\n"

        for i in range(len(self.CIndex[0])):
            self.code += "__Data1StartJ[" + str(i) + "] := " + self.startIdx[self.CIndex[0][i][1]] + ";\n"
            self.code += "__Data1SizeJ[" + str(i) + "] := " + self.CIndex[0][i][1] + ";\n"

        for i in range(len(self.CIndex[1])):
            self.code += "__Data2StartJ[" + str(i) + "] := " + self.startIdx[self.CIndex[1][i][1]] + ";\n"
            self.code += "__Data2SizeJ[" + str(i) + "] := " + self.CIndex[1][i][1] + ";\n"

        self.code += "assert (forall p: int :: ((exists i: int :: (i >= 0 && i < __N1) && (p >= __Data1StartI[i] && p < __Data1StartI[i] + __Data1SizeI[i])) ==> (exists j: int :: (j >= 0 && j < __N2) && (p >= __Data2StartI[j] && p < __Data2StartI[j] + __Data2SizeI[j]))));\n"

        self.code += "assert (forall p: int :: ((exists j: int :: (j >= 0 && j < __N2) && (p >= __Data2StartI[j] && p < __Data2StartI[j] + __Data2SizeI[j])) ==> (exists i: int :: (i >= 0 && i < __N1) && (p >= __Data1StartI[i] && p < __Data1StartI[i] + __Data1SizeI[i]))));\n"


    def init_assignments(self):
        for key, value in self.assignments.items():
            self.code += key + " := " + value + ";\n"

    def modify_assignments(self):
        for var in self.vars:
            if var in self.startIdx and self.startIdx[var] in self.vars and self.startIdx[var] in self.loopDepth and var in self.loopDepth:
                if self.loopDepth[var] == self.loopDepth[self.startIdx[var]]:
                    # seq
                    # self.var = self.var + self.startIdx
                    self.code += var + " := " + var + " + " + self.startIdx[var] + ";\n"
                else:
                    # nest
                    self.code += var + " := " + var + " * " + self.startIdx[var] + ";\n"

    def emit_unique_idx(self):
        for idx in self.UniqueIdx[0]:
            self.code += "assert 1 != 0"
            for idx2 in self.UniqueIdx[1]:
                self.code += " || " + idx + " == " + idx2
            self.code += ";\n"

    def emit_same_area(self):
        self.code += "assert 1 "
        for idx in self.UniqueIdx[0]:
            self.code += " * " + idx
        self.code += " == 1 "
        for idx in self.UniqueIdx[1]:
            self.code += " * " + idx
        self.code += ";\n"

    def emit_assumptions(self):
        for assumption in self.assumptions:
            self.code += "assume " + assumption + ";\n"

    def init_equalities(self):
        for i in range(0, len(self.equality), 2):
            self.code += "assert " + self.equality[i] + " == " + self.equality[i + 1] + ";\n"

# test the parser
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python loopcc.py <file1> <file2>")
        sys.exit(1)
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]
    print("Parsing", filename1)
    print("Parsing", filename2)

    assignments = {}
    _vars = []
    startIdx = {}
    loopDepth = {}
    operations1 = []
    operations2 = []
    assumptions = []

    with open(filename1) as f:
        try:
            tree = create_parser(".", assignments, _vars, startIdx, loopDepth, operations1, assumptions).parse(f.read())
            # print(tree.pretty())
        except LarkError as e:
            print(e)
            sys.exit(1)

    with open(filename2) as f:
        try:
            tree = create_parser(".", assignments, _vars, startIdx, loopDepth, operations2, assumptions).parse(f.read())
            # print(tree.pretty())
        except LarkError as e:
            print(e)
            sys.exit(2)

    print("Loop DSL generated successfully")

    equality = []
    Unique1 = set()
    Unique2 = set()
    CIndex1 = []
    CIndex2 = []
    if len(operations1) == len(operations2) and len(operations1) == 3:
        # for every three operations, only print the second and third operations
        for i in range(0, len(operations1), 3):
            if operations1[i + 1][0] != operations2[i + 1][0] or operations1[i + 2][0] != operations2[i + 2][0]:
                print("The two expressions are not equivalent.")
                sys.exit(3)
        for i in range(0, len(operations1)):
            equality.append(operations1[i][1])
            equality.append(operations2[i][1])
            equality.append(operations1[i][2])
            equality.append(operations2[i][2])
            Unique1.add(operations1[i][1])
            Unique1.add(operations1[i][2])
            Unique2.add(operations2[i][1])
            Unique2.add(operations2[i][2])
    else:
        for i in range(0, len(operations1), 3):
            CIndex1.append([operations1[i][1], operations1[i][2]])
        for i in range(0, len(operations2), 3):
            CIndex2.append([operations2[i][1], operations2[i][2]])

    print("Boogie code generation started")
    BoogieCode(assignments, _vars, startIdx, loopDepth, equality, assumptions, [Unique1, Unique2], [CIndex1, CIndex2]).print_code("output.bpl")
    print("Boogie code generated successfully in output.bpl")

    print("Running Boogie")
    try:
        subprocess.run(["/home/shreya/software-verification/boogie-src/Source/BoogieDriver/bin/Debug/net8.0/BoogieDriver", "output.bpl"], check=True)
    except subprocess.CalledProcessError as e:
        print("Boogie failed")
        sys.exit(4)

# end of loopcc.py
