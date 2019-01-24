from enum import Enum
import pegpy.utils as u

class ParsingExpression(object):

    def __repr__(self):
        return self.__str__()

    def __or__(self,right):
        return Alt(self, ParsingExpression.new(right))

    def __and__(self,right):
        return Seq.new2(self, ParsingExpression.new(right))

    def __xor__(self,right):
        return Seq.new2(self,lfold("", ParsingExpression.new(right)))

    def __rand__(self,left):
        return Seq.new2(ParsingExpression.new(left), self)

    def __add__(self,right):
        return Seq.new2(Many1(self),ParsingExpression.new(right))

    def __mul__(self,right):
        return Seq.new2(Many(self),ParsingExpression.new(right))

    def __truediv__ (self, right):
        return Ore(self, ParsingExpression.new(right))

    def __invert__(self):
        return Not(self)

    def __neq__(self):
        return Not(self)

    def __pos__(self):
        return And(self)

    def __len__(self):
        if hasattr(self, 'inner'): return 1
        if hasattr(self, 'right'): return 2
        return 0

    def __getitem__(self, item):
        if hasattr(self, 'inner') and item == 0:
            return self.inner
        if hasattr(self, 'right'):
            if item == 0: return self.left
            if item == 1: return self.right
        raise IndexError

    def __iter__(self):
        if hasattr(self, 'inner'):
            yield self.inner
        if hasattr(self, 'right'):
            yield self.left
            yield self.right

    def setpeg(self, peg):
        for pe in self: pe.setpeg(peg)
        return self

    def deref(self):
        return self.inner

    @classmethod
    def new(cls, e):
        if e == 0 or e is None: return EMPTY
        if isinstance(e, str):
            if len(e) == 0:
                return EMPTY
            return Char(e)
        return e

def flatten(pe, ls, target = None):
    if target is None:
        ls.append(pe)
        for e in pe: flatten(e, ls, target)
    elif isinstance(pe, target):
        for e in pe: flatten(e, ls, target)
    else :
        ls.append(pe)
    return ls

def filter(f, pe, ls):
    if f(pe):
        ls.append(pe)
    for e in pe: filter(f, e, ls)
    return ls

def ref(name):
    if name.find('/') != -1:
        return Ore.new(list(map(ref, name.split('/'))))
    if name.find(' ') != -1:
        return Seq.new(list(map(ref, name.split(' '))))
    if name.startswith('$'):
        return LinkAs("", Ref(name[1:]))
    return Ref(name)

def lfold(ltag,e):
    if isinstance(e, Many) and isinstance(e.inner, TreeAs):
        return Many(lfold(ltag, e.inner))
    if isinstance(e, Many1) and isinstance(e.inner, TreeAs):
        return Many1(lfold(ltag, e.inner))
    if isinstance(e, Ore):
        return Ore(lfold(ltag, e.left), lfold(ltag, e.right))
    if isinstance(e, TreeAs):
        return FoldAs(ltag, e.name, ParsingExpression.new(e.inner))
    return e


def grouping(e, f):
    return '(' + str(e) + ')' if f(e) else str(e)


def inSeq(e):
    return isinstance(e, Ore) or isinstance(e, Alt)


def inUnary(e):
    return (isinstance(e, Ore) and e.right != EMPTY) \
           or isinstance(e, Seq) or isinstance(e, Alt) \
           or isinstance(e, LinkAs) or isinstance(e, FoldAs)

# PEG Grammar

class Empty(ParsingExpression):
    def __str__(self):
        return "''"

EMPTY = Empty()

class Char(ParsingExpression):
    __slots__ = ['a']

    def __init__(self, a):
        self.a = a

    def __str__(self):
        return "'" + u.quote_string(self.a) + "'"

    def setpeg(self, peg):
        for c in self.a:
            peg.min = min(ord(c), peg.min)
            peg.max = max(ord(c), peg.max)
        return self

class Range(ParsingExpression):
    __slots__ = ['chars', 'ranges']

    def __init__(self, chars: str, ranges: tuple):
        self.chars = chars
        self.ranges = ranges

    def __str__(self):
        l = tuple(map(lambda x: u.quote_string(x[0], ']') + '-' + u.quote_string(x[1], ']'), self.ranges))
        return "[" + ''.join(l) + u.quote_string(self.chars, ']') + "]"

    def setpeg(self, peg):
        for c in self.chars:
            peg.min = min(ord(c), peg.min)
            peg.max = max(ord(c)+1, peg.max)
        for r in self.ranges:
            peg.min = min(ord(r[0]), peg.min)
            peg.max = max(ord(r[1])+1, peg.max)
        return self

    def min(self):
        return min(list(map(lambda c: ord(c), self.chars)) + [ord(x[0]) for x in self.ranges])

    def max(self):
        return max(list(map(lambda c: ord(c), self.chars)) + [ord(x[1]) for x in self.ranges])

    @classmethod
    def new(cls, *ss):
        chars = []
        ranges = []
        for s in ss:
            if isinstance(s, tuple):
                ranges.append(s)
            elif len(s) == 3 and s[1] is '-':
                ranges.append((s[0], s[2]))
            else:
                for c in s:
                    chars.append(c)
        chars = ''.join(chars)
        if len(ranges) == 0 and len(chars) == 1:
            return Char(chars)
        return Range(chars, ranges)


class Any(ParsingExpression):
    def __str__(self):
        return '.'

ANY = Any()


class Seq(ParsingExpression):
    __slots__ = ['left', 'right']

    def __init__(self, left, right):
        self.left = ParsingExpression.new(left)
        self.right = ParsingExpression.new(right)

    def __str__(self):
        return grouping(self.left, inSeq) + ' ' + grouping(self.right, inSeq)

    def flatten(self, ls):
        if isinstance(self.left, Seq):
            self.left.flatten(ls)
        else:
            ls.append(self.left)
        if isinstance(self.right, Seq):
            self.right.flatten(ls)
        else:
            ls.append(self.right)
        return ls

    @classmethod
    def new2(cls, x, y):
        if x is None or y is None: return None
        if isinstance(y, Empty): return x
        if isinstance(x, Empty): return y
        if isinstance(x, Char) and isinstance(y, Char):
            return Char(x.a + y.a)
        return Seq(x, y)

    @classmethod
    def new(cls, es):
        if len(es) == 1: return es[0]
        if len(es) > 1:
            return Seq.new2(es[0], Seq.new(es[1:]))
        return EMPTY

class Ore(ParsingExpression):
    __slots__ = ['left', 'right']

    def __init__(self, left, right):
        self.left = ParsingExpression.new(left)
        self.right = ParsingExpression.new(right)

    def __str__(self):
        if self.right == EMPTY:
            return grouping(self.left, inUnary) + '?'
        return str(self.left) + ' / ' + str(self.right)

    def flatten(self, ls):
        if isinstance(self.left, Ore):
            self.left.flatten(ls)
        else:
            ls.append(self.left)
        if isinstance(self.right, Ore):
            self.right.flatten(ls)
        else:
            ls.append(self.right)
        return ls

    @classmethod
    def new2(cls, x, y):
        if x is None or y is None: return None
        if x == EMPTY: return EMPTY
        return Ore(x, y)

    @classmethod
    def new(cls, es):
        if len(es) == 1: return es[0]
        if len(es) > 1:
            return Ore.new2(es[0], Ore.new(es[1:]))
        return EMPTY

class Alt(ParsingExpression):
    __slots__ = ['left', 'right']
    def __init__(self, left, right):
        self.left = ParsingExpression.new(left)
        self.right = ParsingExpression.new(right)
    def __str__(self):
        return str(self.left) + ' | ' + str(self.right)

    def flatten(self, ls):
        if isinstance(self.left, Alt):
            self.left.flatten(ls)
        else:
            ls.append(self.left)
        if isinstance(self.right, Alt):
            self.right.flatten(ls)
        else:
            ls.append(self.right)
        return ls

    @classmethod
    def new2(cls, x, y):
        if x is None or y is None: return None
        return Alt(x, y)

    @classmethod
    def new(cls, es):
        if len(es) == 1: return es[0]
        if len(es) > 1:
            return Alt.new2(es[0], Alt.new(es[1:]))
        return EMPTY



class And(ParsingExpression):
    __slots__ = ['inner']

    def __init__(self, inner):
        self.inner = ParsingExpression.new(inner)

    def __str__(self):
        return '&' + grouping(self.inner, inUnary)


class Not(ParsingExpression):
    __slots__ = ['inner']

    def __init__(self, inner):
        self.inner = ParsingExpression.new(inner)

    def __str__(self):
        return '!' + grouping(self.inner, inUnary)

FAIL  = ~EMPTY

class Many(ParsingExpression):
    __slots__ = ['inner']

    def __init__(self, inner):
        self.inner = ParsingExpression.new(inner)

    def __str__(self):
        return grouping(self.inner, inUnary) + '*'


class Many1(ParsingExpression):
    __slots__ = ['inner']

    def __init__(self, inner):
        self.inner = ParsingExpression.new(inner)

    def __str__(self):
        return grouping(self.inner, inUnary) + '+'

## Tree Construction


class TreeAs(ParsingExpression):
    __slots__ = ['name', 'inner']

    def __init__(self, name = '', inner = EMPTY):
        self.name = name
        self.inner = ParsingExpression.new(inner)

    def __str__(self):
        tag = ' #' + self.name if self.name != '' else ''
        return '{ ' + str(self.inner) + tag + ' }'

class LinkAs(ParsingExpression):
    __slots__ = ['name', 'inner']

    def __init__(self, name = '', inner=None):
        self.name = name
        self.inner = ParsingExpression.new(inner)

    def __str__(self):
        if self.name == '':
            return '$' + grouping(self.inner, inUnary)
        return self.name + ': ' + grouping(self.inner, inUnary)

    def __le__(self, right):
        return LinkAs(self.name, right)

    def __ge__(self, right):
        return LinkAs(self.name, right)

    def __mod__(self, right):
        return ref(right)

    def __xor__(self,right):
        return lfold(self.name, ParsingExpression.new(right))

N = LinkAs("")


class FoldAs(ParsingExpression):
    __slots__ = ['left', 'name', 'inner']

    def __init__(self, left = '', name = '', inner = EMPTY):
        self.left = left
        self.name = name
        self.inner = ParsingExpression.new(inner)

    def __str__(self):
        prefix = '^' if self.name == '' else self.left + ':^ '
        tag = ' #' + self.name if self.name != '' else ''
        return prefix + '{ ' + str(self.inner) + tag + ' }'


class Detree(ParsingExpression):
    __slots__ = ['inner']

    def __init__(self, inner):
        self.inner = ParsingExpression.new(inner)

    def __str__(self):
        return '@detree(' + str(self.inner) + ')'

## Symbol

class State(ParsingExpression):
    __slots__ = ['func', 'name', 'inner', 'opt']

    def __init__(self, func, inner, name = None, opt = None):
        self.func = func
        self.inner = ParsingExpression.new(inner)
        self.name = str(self.inner) if name is None else name
        self.opt = opt

    def __str__(self):
        return  self.func + '(' + str(self.inner) + ')'

## Move

class Move(ParsingExpression):
    __slots__ = ['func']

    def __init__(self, func):
        self.func = func

    def __str__(self):
        return  self.func + '()'


# @on(flag, e)
class On(ParsingExpression):
    __slots__ = ['name', 'inner']

    def __init__(self, name, inner):
        self.name = name
        self.inner = ParsingExpression.new(inner)

    def __str__(self):
        return  '@on(' + str(self.inner) + ')'

# @off(flag, e)
class Off(ParsingExpression):
    __slots__ = ['name', 'inner']

    def __init__(self, name, inner):
        self.name = name
        self.inner = ParsingExpression.new(inner)

    def __str__(self):
        return  '@off(' + str(self.inner) + ')'

# @if(flag)
class If(ParsingExpression):
    __slots__ = ['name']

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return  '@if(' + str(self.name) + ')'

## Meta

class Meta(ParsingExpression):
    __slots__ = ['name', 'inner', 'opt']
    def __init__(self, name, inner, opt = None):
        self.name = name
        self.inner = ParsingExpression.new(inner)
        self.opt = opt
    def __str__(self):
        arg = ', ' + repr(self.opt) if self.opt != None else ''
        return self.tag + '(' + str(self.inner) + arg + ')'

# Rule

#class Ref(ParsingExpression, ast.SourcePosition):

class Ref(ParsingExpression):
    __slots__ = ['peg', 'name', 'pos3']

    def __init__(self, name, peg = None):
        self.name = name
        self.peg = peg
        self.pos3 = None

    def __str__(self):
        return str(self.name)

    def uname(self):
        return self.name if self.name.find(':') > 0 else self.peg.namespace() + ':' + self.name

    def setpeg(self, peg):
        self.peg = peg
        if self.name[0].islower() and self.name in peg:  # inlining
            # print('@inline', self.name)
            return peg[self.name].inner
        return self

    def isNonTerminal(self):
        return self.name in self.peg

    def deref(self):
        return self.peg[self.name].inner

    @classmethod
    def isPatternOnlyName(cls, name):
        return name.isupper()

    def prop(self):
        return getattr(self.peg, self.name)

    def getmemo(self, prefix: str, default = None):
        if self.name in self.peg:
            rule = self.peg[self.name]
            if hasattr(rule, prefix):
                return getattr(rule, prefix)
            return None
        else:
            return default

    def setmemo(self, prefix, value):
        rule = self.peg[self.name]
        setattr(rule, prefix, value)

class Rule(Ref):
    __slots__ = ['peg', 'name', 'inner', 'pos3']

    def __init__(self, peg, name, inner, pos3=None):
        super().__init__(name, peg)
        self.inner = ParsingExpression.new(inner)
        self.pos3 = pos3

    def __str__(self):
        return self.name + ' = ' + str(self.inner)

    def deref(self):
        return self.inner

    @classmethod
    def isInlineName(cls, name):
        return (name[0].islower() and not '.' in name)

    @classmethod
    def isUnitName(cls, name):
        return not name[0].isalpha() or (name[0].isupper() and not cls.isTreeName(name))

    @classmethod
    def isTreeName(cls, name):
        if name[0].isupper():
            for c in name:
                if c.islower(): return True
            return False
        return False

## Properties

def addmethod(*ctags):
    def _match(func):
        name = ctags[-1]
        for ctag in ctags[:-1]:
            setattr(ctag, name, func)
        return func
    return _match

def isAlwaysConsumed(pe: ParsingExpression):
    if not hasattr(Char, 'isAlwaysConsumed'):
        method = 'isAlwaysConsumed'
        @addmethod(Char, Any, Range, method)
        def char(pe): return True

        @addmethod(Many, Not, And, Empty, Move, method)
        def empty(pe): return False

        @addmethod(Many1, LinkAs, TreeAs, FoldAs, Detree, Meta, method)
        def unary(pe):
            return pe.inner.isAlwaysConsumed()

        @addmethod(Seq, method)
        def seq(pe):
            return pe.left.isAlwaysConsumed() or pe.right.isAlwaysConsumed()

        @addmethod(Ore, Alt, method)
        def ore(pe):
            return pe.left.isAlwaysConsumed() and pe.right.isAlwaysConsumed()

        @addmethod(State, method)
        def state(pe):
            return pe.inner.isAlwaysConsumed()

        @addmethod(Ref, method)
        def memo(pe: Ref):
            memoed = pe.getmemo('nonnull', True)
            if memoed == None:
                pe.setmemo('nonnull', True)
                memoed = isAlwaysConsumed(pe.deref())
                pe.setmemo('nonnull', memoed)
            return memoed

    return pe.isAlwaysConsumed()

def first(pe: ParsingExpression, max=255):
    if not hasattr(Char, 'first'):
        method = 'first'
        @addmethod(Char, Range, method)
        def first_char(pe, max):
            return 1 << ord(pe.a[0])

        @addmethod(Any, Move, method)
        def first_any(pe, max):
            return (1 << (max+1))-1

        @addmethod(Range, method)
        def first_range(pe, max):
            cs = 0
            for c in pe.chars:
                cs |= 1 << ord(c)
            for r in pe.ranges:
                for c in range(ord(r[0]), ord(r[1])+1):
                    cs |= 1 << c
            return cs

        @addmethod(Not, Empty, method)
        def first_empty(pe, max):
            return 0

        @addmethod(Many, Many1, And, LinkAs, TreeAs, FoldAs, Detree, Meta, method)
        def first_unary(pe, max):
            return pe.inner.first(max)

        @addmethod(Seq, method)
        def first_seq(pe, max):
            cs = pe.left.first(max)
            return cs if isAlwaysConsumed(pe.left) else cs | pe.right.first(max)

        @addmethod(Ore, Alt, method)
        def first_ore(pe, max):
            return first(pe.left, max) | pe.right.first(max)

        @addmethod(State, method)
        def first_state(pe, max):
            return pe.inner.first(max)

        @addmethod(Ref, method)
        def first_memo(pe: Ref, max):
            propname = 'first_bitset{}'.format(max)
            memoed = pe.getmemo(propname, None)
            if memoed == None:
                pe.setmemo(propname, first_any(pe, max))
                memoed = pe.deref().first(max)
                pe.setmemo(propname, memoed)
            return memoed
    if max > 255: max = 65535
    return pe.first(max)

def cdr(pe: ParsingExpression):
    if not hasattr(Char, 'cdr'):
        method = 'cdr'

        @addmethod(Char, method)
        def cdr_char(pe):
            if len(pe.a) > 2: return Char(pe.a[1:])
            return EMPTY

        @addmethod(Any, Range, method)
        def cdr_any(pe): return EMPTY

        @addmethod(Seq, method)
        def cdr_seq(pe):
            return Seq.new2(pe.left.cdr(), pe.right)

        @addmethod(Many1, method)
        def cdr_many1(pe, max):
            return Seq.new2(pe.inner.cdr(), Many(pe.inner))

        @addmethod(Ore, method)
        def cdr_ore(pe):
            return Ore.new2(pe.left.cdr(), pe.right.cdr())

        @addmethod(Alt, method)
        def cdr_alt(pe):
            return Alt.new2(pe.left.cdr(), pe.right.cdr())

        @addmethod(Not, Empty, Move, method)
        def cdr_empty(pe):
            return None

        @addmethod(Many, Many1, And, LinkAs, TreeAs, FoldAs, Detree, Meta, method)
        def cdr_unary(pe, max):
            return None

        @addmethod(State, method)
        def cdr_state(pe, max):
            return None

        @addmethod(Ref, method)
        def cdr_ref(pe: Ref):
            return pe.deref().cdr()
    return pe.cdr()


## TreeState

class T(Enum):
    Unit = 0
    Tree = 1
    Mut = 2
    Fold = 3

def treeState(pe):
    if not hasattr(Char, 'treeState'):
        method = 'treeState'
        @addmethod(Empty, Char, Any, Range, Not, Detree, Move, method)
        def stateUnit(pe):
            return T.Unit

        @addmethod(TreeAs, method)
        def stateTree(pe):
            return T.Tree

        @addmethod(LinkAs, method)
        def stateMut(pe):
            return T.Mut

        @addmethod(FoldAs, method)
        def stateFold(pe):
            return T.Fold

        @addmethod(Seq, method)
        def stateSeq(pe):
            ts0 = treeState(pe.left)
            return ts0 if ts0 != T.Unit else treeState(pe.right)

        @addmethod(Ore, Alt, method)
        def stateAlt(pe):
            ts0 = treeState(pe.left)
            if ts0 != T.Unit: return ts0
            ts1 = treeState(pe.right)
            return T.Mut if ts1 == T.Tree else ts1

        @addmethod(Many, Many1, And, method)
        def stateMany(pe):
            ts0 = treeState(pe.inner)
            return T.Mut if ts0 == T.Tree else ts0

        @addmethod(State, method)
        def state(pe):
            if pe.func == '@exists' or pe.func.startswith('@match'): return T.Unit
            return treeState(pe.inner)

        @addmethod(Ref, method)
        def stateRef(pe: Ref):
            memoed = pe.getmemo('ts', T.Unit)
            if memoed == None:
                pe.setmemo('ts', T.Unit)
                memoed = treeState(pe.deref())
                pe.setmemo('ts', memoed)
            return memoed

    return pe.treeState()

def checkTree(pe, inside):
    if not hasattr(LinkAs, 'checkTree'):
        method = 'checkTree'

        @addmethod(Empty, Char, Any, Range, Move, method)
        def term(pe, inside):
            return pe

        @addmethod(And, Not, Many, Many1, Rule, method)
        def unary(pe, inside):
            pe.inner = checkTree(pe.inner, inside)
            return pe

        @addmethod(Seq, Ore, Alt, method)
        def binary(pe, inside):
            pe.left = checkTree(pe.left, inside)
            pe.right = checkTree(pe.right, inside)
            return pe

        @addmethod(TreeAs, method)
        def tree(pe, inside):
            pe.inner = checkTree(pe.inner, TreeAs)
            return LinkAs('', pe) if inside == TreeAs else pe

        @addmethod(FoldAs, method)
        def tree(pe, inside):
            pe.inner = checkTree(pe.inner, TreeAs)
            return pe

        @addmethod(LinkAs, method)
        def fold(pe, inside):
            ts = treeState(pe.inner)
            if ts != T.Tree:
                pe.inner = TreeAs('', pe.inner)
            checkTree(pe.inner, LinkAs)
            return pe

        @addmethod(State, method)
        def state(pe, inside):
            pe.inner = checkTree(pe.inner, inside)
            return pe

        @addmethod(Ref, method)
        def ref(pe, inside):
            ts = treeState(pe)
            if ts == T.Tree and inside == TreeAs:
                return LinkAs('', pe)
            return pe

    return pe.checkTree(inside)

## Setup

def load_tpeg(g):
    # Preliminary
    __ = N % '__'
    _ = N % '_'
    EOS = N % 'EOS'
    #Range = Range.new

    g.Start = N%'__ Source EOF'
    g.EOF = ~ANY
    g.EOL = ParsingExpression.new('\n') | ParsingExpression.new('\r\n') | N%'EOF'
    g.COMMENT = '/*' & (~ParsingExpression.new('*/') & ANY)* 0 & '*/' | '//' & (~(N%'EOL') & ANY)* 0
    g._ = (Range.new(' \t') | N%'COMMENT')* 0
    g.__ = (Range.new(' \t\r\n') | N%'COMMENT')* 0
    g.S = Range.new(' \t')

    g.Source = TreeAs('Source', (N%'$Statement')*0)
    g.EOS = N%'_' & (';' & N%'_' | N%'EOL' & (N%'S' | N%'COMMENT') & N%'_' | N%'EOL')* 0

    left = LinkAs('left')
    right = LinkAs('right')
    name = LinkAs('name')
    ns = LinkAs('ns')
    inner = LinkAs('inner')

    g.Statement = N%'Import/Example/Rule'

    g.Rule = TreeAs('Rule', (name <= N%'Identifier __') & '=' & __ & (Range.new('/|') & __ |0) & (inner <= N%'Expression')) & EOS
    g.Identifier = TreeAs('Name', (Range.new('A-Z', 'a-z', '@_') & Range.new('A-Z', 'a-z', '0-9', '_.')*0
                                   | '"' & (ParsingExpression.new(r'\"') | ~Range.new('\\"\n') & ANY)* 0 & '"'))

    g.Example = TreeAs('Example', 'example' & N%'S _' & (name <= N%'Names') & (inner <= N%'Doc')) & EOS
    g.Names = TreeAs('', N%'$Identifier _' & (Range.new(',&') & N%'_ $Identifier _')*0)
    Doc1 = TreeAs("Doc", (~(N%'DELIM EOL') & ANY)* 0)
    Doc2 = TreeAs("Doc", (~Range.new('\r\n') & ANY)* 0)
    g.Doc = N%'DELIM' & (N%'S'*0) & N%'EOL' & Doc1 & N % 'DELIM' | Doc2
    g.DELIM = ParsingExpression.new("'''")

    g.Import = TreeAs('Import', 'import' & N%'S _' & (name <= N%'Char')
                      & ((N%'S _' & 'as' & N%'S _' & (ns <= N%'Identifier'))|0)) & EOS

    #g.Expression = N%'Choice' & (left ^ (TreeAs('Alt', __ & '|' & _ & (right <= N%'Expression'))|0))
    g.Expression = N%'Choice' & (left ^ TreeAs('Alt', __ & '|' & _ & (right <= N%'Expression'))|0)
    g.Choice = N%'Sequence' & (left ^ TreeAs('Ore', __ & '/' & _ & (right <= N%'Choice'))|0)
    g.SS = N%'S _' & ~(N%'EOL') | (N%'_ EOL')+0 & N%'S _'
    g.Sequence = N%'Predicate' &  (left ^ TreeAs('Seq', (right <= N%'SS Sequence'))|0)

    g.Predicate = TreeAs('Not', '!' & (inner <= N%'Predicate')) \
                  | TreeAs('And', '&' & (inner <= N%'Predicate')) \
                  | TreeAs('Append', '$' & ( inner <= N%'Predicate')) \
                  | N%'Suffix'
    g.Suffix = N%'Term' & ((inner ^ TreeAs('Many', '*')) | (inner ^ TreeAs('Many1', '+')) | (inner ^ TreeAs('Option', '?')) | 0)

    g.Term = N%'Group/Char/Class/Any/Tree/Fold/BindFold/Bind/Func/Ref'
    g.Group = '(' & __ & N%'Expression/Empty' & __ & ')'

    g.Empty = TreeAs('Empty', EMPTY)
    g.Any = TreeAs('Any', '.')
    g.Char = "'" & TreeAs('Char', ('\\' & ANY | ~Range.new("'\n") & ANY)*0) & "'"
    g.Class = '[' & TreeAs('Class', ('\\' & ANY | ~Range.new("]") & ANY)*0) & ']'

    g.Tag = '{' & __ & (('#' & (name <= N%'Identifier')) | 0) & __
    g.ETag = ('#' & (name <= N%'Identifier') | 0) & __ & '}'

    g.Tree = TreeAs('TreeAs', N%'Tag' & (inner <= (N%'Expression __' | N%'Empty')) & N%'ETag' )
    g.Fold = '^' & _ & TreeAs('Fold', N%'Tag' & (inner <= (N%'Expression __' | N%'Empty')) & N%'ETag' )
    g.Bind = TreeAs('LinkAs', (name <= N%'Var' & ':') & _ & (inner <= N%'_ Term'))
    g.BindFold = TreeAs('Fold', (left <= N%'Var' & ':^') & _ & N%'Tag' & (inner <= (N%'Expression __' | N%'Empty')) & N%'ETag')
    g.Var = TreeAs('Name', Range.new('a-z', '$') & Range.new('A-Z', 'a-z', '0-9', '_')*0)

    g.Func = TreeAs('Func', N%'$Identifier' & '(' & __ & (N%'$Expression _' & ',' & __)* 0 & (N%'$Expression __' | 0) & ')')
    g.Ref = TreeAs('Ref', N%'NAME')
    g.NAME = '"' & (ParsingExpression.new(r'\"') | ~Range.new('\\"\n') & ANY)* 0 & '"' | (~Range.new(' \t\r\n(,){};<>[|/*+?=^\'`#') & ANY)+0

    # Example
    #g.example("Ref", "abc")
    #g.example("Ref", '"abc"')
    g.example("COMMENT", "/*hoge*/hoge", "[# '/*hoge*/']")
    g.example("COMMENT", "//hoge\nhoge", "[# '//hoge']")

    g.example("Ref,Term,Expression", "a", "[#Ref 'a']")

    g.example("Char,Expression", "''", "[#Char '']")
    g.example("Char,Expression", "'a'", "[#Char 'a']")
    g.example("Char,Expression", "'ab'", "[#Char 'ab']")
    #g.example("Char,Expression", "'\\''", "[#Char \"'\"]")
    g.example("Char,Expression", "'\\\\a'", "[#Char '\\\\\\\\a']")
    g.example("Ref,Expression", "\"a\"", "[#Ref '\"a\"']")
    g.example("Class,Expression", "[a]", "[#Class 'a']")
    g.example("Func,Expression", "f(a)", "[#Func [#Name 'f'] [#Ref 'a']]")
    g.example("Func,Expression", "f(a,b)", "[#Func [#Name 'f'] [#Ref 'a'] [#Ref 'b']]")
    g.example("Func,Expression", "@symbol(a)", "[#Func [#Name '@symbol'] [#Ref 'a']]")
    g.example("Func,Expression", "@exists(a,'b')", "[#Func [#Name '@exists'] [#Ref 'a'] [#Char 'b']]")
    #g.example("Func,Expression", "$name(a)", "[#Func [#Name '$name'] [#Ref 'a']]")

    g.example("Predicate,Expression", "&a", "[#And inner=[#Ref 'a']]")
    g.example("Predicate,Expression", "!a", "[#Not inner=[#Ref 'a']]")
    g.example("Suffix,Expression", "a?", "[#Option inner=[#Ref 'a']]")
    g.example("Suffix,Expression", "a*", "[#Many inner=[#Ref 'a']]")
    g.example("Suffix,Expression", "a+", "[#Many1 inner=[#Ref 'a']]")
    g.example("Expression", "{}", "[#TreeAs inner=[#Empty '']]")
    g.example("Expression", "{ a }", "[#TreeAs inner=[#Ref 'a']]")
    g.example("Expression", "{ }", "[#TreeAs inner=[#Empty '']]")
    g.example("Expression", "()", "[#Empty '']")
    g.example("Expression", "&'a'", "[#And inner=[#Char 'a']]")

    g.example("Expression", "{a}", "[#TreeAs inner=[#Ref 'a']]")
    g.example("Expression", "{a #Int}", "[#TreeAs inner=[#Ref 'a'] name=[#Name 'Int']]")
    g.example("Expression", "{#Int a}", "[#TreeAs name=[#Name 'Int'] inner=[#Ref 'a']]")
    g.example("Expression", "^{a}", "[#Fold inner=[#Ref 'a']]")
    g.example("Expression", "^{ #Int a }", "[#Fold name=[#Name 'Int'] inner=[#Ref 'a']]")
    g.example("Expression", "name:^ {a}", "[#Fold left=[#Name 'name'] inner=[#Ref 'a']]")
    g.example("Expression", "name:^ {#Int a}", "[#Fold left=[#Name 'name'] name=[#Name 'Int'] inner=[#Ref 'a']]")
    g.example("Expression", "$a", "[#Append inner=[#Ref 'a']]")
    g.example("Expression", "$(a)", "[#Append inner=[#Ref 'a']]")
    g.example("Expression", "name: a", "[#LinkAs name=[#Name 'name'] inner=[#Ref 'a']]")
    g.example("Expression", "name: a a", "[#Seq left=[#LinkAs name=[#Name 'name'] inner=[#Ref 'a']] right=[#Ref 'a']]")
    #g.example("Expression", "name: a", "[#LinkAs name=[#Name 'name'] inner=[#Ref 'a']]")

    g.example("Expression", "a b", "[#Seq left=[#Ref 'a'] right=[#Ref 'b']]")
    g.example("Expression", "a b c", "[#Seq left=[#Ref 'a'] right=[#Seq left=[#Ref 'b'] right=[#Ref 'c']]]")
    g.example("Expression", "a/b / c", "[#Ore left=[#Ref 'a'] right=[#Ore left=[#Ref 'b'] right=[#Ref 'c']]]")
    g.example("Expression", "a|b | c", "[#Alt left=[#Ref 'a'] right=[#Alt left=[#Ref 'b'] right=[#Ref 'c']]]")
    g.example("Statement", "A=a", "[#Rule name=[#Name 'A'] inner=[#Ref 'a']]")

    g.example("Statement", '"A"=a', "[#Rule name=[#Name '\"A\"'] inner=[#Ref 'a']]")

    g.example("Statement", "example A,B abc \n", "[#Example name=[# [#Name 'A'] [#Name 'B']] inner=[#Doc 'abc ']]")
    g.example("Statement", "A = a\n  b", "[#Rule name=[#Name 'A'] inner=[#Seq left=[#Ref 'a'] right=[#Ref 'b']]]")
    g.example("Start", "A = a; B = b;;",
              "[#Source [#Rule name=[#Name 'A'] inner=[#Ref 'a']] [#Rule name=[#Name 'B'] inner=[#Ref 'b']]]")
    g.example("Start", "A = a\nB = b",
              "[#Source [#Rule name=[#Name 'A'] inner=[#Ref 'a']] [#Rule name=[#Name 'B'] inner=[#Ref 'b']]]")
    g.example("Start", "A = a //hoge\nB = b",
              "[#Source [#Rule name=[#Name 'A'] inner=[#Ref 'a']] [#Rule name=[#Name 'B'] inner=[#Ref 'b']]]")
    return g

# Loader loader

def setup_loader(Grammar, pgen):
    from pegpy.ast import ParseTreeConv

    class PEGConv(ParseTreeConv):
        def __init__(self, *args):
            super(PEGConv, self).__init__(*args)

        def Empty(self, t, out):
            return EMPTY

        def Any(self, t, out):
            return ANY

        def Char(self, t, out):
            s = t.asString()
            if s.find(r'\x') >= 0:
                sb = []
                s = s.encode('utf-8')
                while len(s) > 0:
                    c, s = u.unquote(bytes(s, 'ascii'))
                    sb.append(c)
                return ParsingExpression.new(b''.join(sb))
            else:
                sb = []
                while len(s) > 0:
                    c, s = u.unquote(s)
                    sb.append(c)
                return ParsingExpression.new(''.join(sb))

        def Class(self, t, out):
            s = t.asString()
            sb = []
            while len(s) > 0:
                c, s = u.unquote(s)
                if s.startswith('-') and len(s) > 1:
                    c2, s = u.unquote(s[1:])
                    sb.append((c, c2))
                else:
                    sb.append(c)
            return Range.new(*sb)

        def Option(self, t, out):
            inner = self.conv(t['inner'])
            return Ore(inner, EMPTY)

        def TreeAs(self, t, out):
            name = t['name'].asString() if 'name' in t else ''
            inner = self.conv(t['inner'])
            return TreeAs(name, inner)

        def LinkAs(self, t, ouy):
            name = t['name'].asString() if 'name' in t else ''
            inner = self.conv(t['inner'])
            #print('@LinkAs', LinkAs(name, inner))
            return LinkAs(name, inner)

        def Append(self, t, out):
            name = ''
            tsub = t['inner']
            if tsub == 'Func':
                a = tsub.asArray()
                name = a[0].asString()
                inner = self.conv(a[1])
            else:
                inner = self.conv(tsub)
            return LinkAs(name, inner)

        def Fold(self, t, out):
            left = t['left'].asString() if 'left' in t else ''
            name = t['name'].asString() if 'name' in t else ''
            inner = self.conv(t['inner'])
            return FoldAs(left, name, inner)

        def Func(self, t, out):
            a = t.asArray()
            funcname = a[0].asString()
            if len(a) > 2:
                if funcname == '@on':
                    return On(a[1].asString(), self.conv(a[2]))
                elif funcname == '@off':
                    return Off(a[1].asString(), self.conv(a[2]))
            if len(a) > 1:
                if funcname == '@if':
                    return If(a[1].asString())
                elif funcname == '@ref':
                    return State('@ref', self.conv(a[1]))
                elif (funcname == '@scope' or funcname == '@block'):
                    return State('@scope', self.conv(a[1]))
                elif funcname == '@symbol':
                    return State('@symbol', self.conv(a[1]))
                elif funcname == '@match':
                    return State('@match', self.conv(a[1]))
                elif funcname == '@matchin':
                    return State('@matchin', self.conv(a[1]))
                elif funcname == '@exists' or funcname == '@exist':
                    return State('@exists', EMPTY, str(self.conv(a[1])))
                elif funcname == '@equals':
                    return State('@equals', self.conv(a[1]))
                elif funcname == '@contains':
                    return State('@contains', self.conv(a[1]))
                elif funcname == '@dict':
                    return State('@dict', self.conv(a[1]))
                elif funcname == '@defdict':
                    return State('@defdict', self.conv(a[1]))
                else:
                    out.perror(a[0].pos3(), 'undefined annotation')
                    return self.conv(a[1])
            if funcname == '@skip':
                return Move('@skip')
            out.perror(a[0].pos3(), 'illegal annotation arguments')
            return EMPTY

    def checkRef(pe, consumed: bool, name: str, visited: dict, out):
        if isinstance(pe, Seq):
            pe.left = checkRef(pe.left, consumed, name, visited, out)
            if not consumed:
                consumed = isAlwaysConsumed(pe.left)
            pe.right = checkRef(pe.right, consumed, name, visited, out)
            return pe
        if isinstance(pe, Ore) or isinstance(pe, Alt):
            pe.left = checkRef(pe.left, consumed, name, visited, out)
            pe.right = checkRef(pe.right, consumed, name, visited, out)
            return pe
        if hasattr(pe, 'inner'):
            pe.inner = checkRef(pe.inner, consumed, name, visited, out)
            return pe
        if isinstance(pe, Ref):
            if not consumed and pe.name == name:
                out.perror(pe.pos3, msg='left recursion: ' + str(pe.name))
                return pe
            if not pe.isNonTerminal():
                if pe.name[0] == '"': return Char(pe.name[1:-2])
                if not pe.name[0].isupper(): return Char(pe.name)
                out.perror(pe.pos3, msg='undefined name: ' + str(pe.name))
                return FAIL if Rule.isUnitName(pe.name) else TreeAs('', FAIL)
            if Rule.isInlineName(pe.name):
                out.notice(pe.pos3, msg='inlining: ' + str(pe.name) + ' => ' + str(pe.deref()))
                return pe.deref()
            if not pe.name in visited:
                visited[pe.name] = True
                checkRef(pe.deref(), consumed, name, visited, out)
        return pe

    PEGconv = PEGConv(Ore, Alt, Seq, And, Not, Many, Many1, TreeAs, FoldAs, LinkAs, Ref)
    pegparser = pgen(load_tpeg(Grammar('tpeg')))

    def load_grammar(g, path, out = u.STDOUT):
        if path.find('=') == -1:
            f = u.find_path(path).open()
            data = f.read()
            f.close()
            t = pegparser(data, path)
        else:
            t = pegparser(path)
        if t == 'err':
            out.perror(t.pos3())
            return
        # load
        for _, stmt in t:
            if stmt == 'Rule':
                name = stmt['name'].asString()
                pexr = stmt['inner']
                pe = PEGconv.conv(pexr)
                g.add(name, pe, stmt['name'].pos3())
            elif stmt == 'Example':
                s, spos, epos = stmt['inner'].pos3()
                doc, _, _ = u.encpos3(s, spos, epos)
                #doc = stmt['inner'].asString()
                for n in stmt['name'].asArray():
                    g.example(n.asString(), doc)
            elif stmt == 'Import':
                importPath = stmt['name'].asString()
                if path.find('=') == -1:
                    importPath = u.find_importPath(path, importPath)
                if 'ns' in stmt:
                    ns = stmt['ns'].asString()
                    ng = Grammar()
                    load_grammar(ng, importPath)
                    g.rulemap[ns] = ng
                else:
                    load_grammar(g, importPath)

        g.forEachRule(lambda rule: checkRef(rule.inner, False, rule.name, {}, out))

        def treeRule(rule):
            pe = checkTree(rule.inner, None)
            if not Rule.isInlineName(rule.name):
                ts = treeState(rule.inner)
                if ts == T.Unit and not Rule.isUnitName(rule.name):
                    out.warning(rule.pos3, 'illegal naming: use {}'.format(rule.name.upper()))
                elif ts == T.Tree and not Rule.isTreeName(rule.name):
                    out.warning(rule.pos3, 'illegal naming: use camel case')
                elif ts == T.Mut or ts == T.Fold:
                    out.warning(rule.pos3, 'illegal naming: allowed only inline')
            return pe
        g.forEachRule(treeRule)

        '''
        def dummy(pe):
            print('@first', first(pe.inner, g.max))
            return pe.inner
        '''
        #g.forEachRule(dummy)

    Grammar.load = load_grammar
