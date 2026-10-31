"""Conservative numeric equality checker. No eval/exec, symbols, or reference access."""
import ast
from fractions import Fraction
import re


def normalize(s):
    s=s.strip().replace('−','-').replace('×','*').replace('÷','/')
    s=re.sub(r'\\(?:times|cdot)\b','*',s);s=re.sub(r'\\div\b','/',s)
    s=s.replace('\\left','').replace('\\right','')
    # Numeric/simple arithmetic fractions only; nested supported by repeated passes.
    for _ in range(4):
        t=re.sub(r'\\(?:dfrac|tfrac|frac)\{([^{}]+)\}\{([^{}]+)\}',r'((\1)/(\2))',s)
        t=re.sub(r'\\boxed\{([^{}]+)\}',r'(\1)',t)
        if t==s:break
        s=t
    # Remove digit-grouping commas only, never treat arbitrary commas as grouping.
    s=re.sub(r'(?<![\d.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d.,])',lambda m:m.group().replace(',',''),s)
    s=s.replace('\\,',' ').replace('\\!',' ').replace('\\;',' ')
    return s.strip()


def interval(source):
    if len(source)>200 or not re.fullmatch(r'[\d\s.+*/()\-]+',source):raise ValueError('unsupported expression')
    tree=ast.parse(source,mode='eval')
    if sum(1 for _ in ast.walk(tree))>100:raise ValueError('expression too long')
    def walk(n):
        if isinstance(n,ast.Constant) and type(n.value) in (int,float):
            literal=ast.get_source_segment(source,n)
            if not re.fullmatch(r'(?:\d+(?:\.\d*)?|\.\d+)',literal):raise ValueError('unsupported literal')
            v=Fraction(literal)
            # Decimal literals may be rounded: accept overlapping rounding intervals.
            uncertainty=Fraction(1,2*10**len(literal.split('.')[1])) if '.' in literal else Fraction(0)
            return v-uncertainty,v+uncertainty
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):
            lo,hi=walk(n.operand);return (lo,hi) if isinstance(n.op,ast.UAdd) else (-hi,-lo)
        if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
            l,h=walk(n.left);r,s=walk(n.right)
            if isinstance(n.op,ast.Add):return l+r,h+s
            if isinstance(n.op,ast.Sub):return l-s,h-r
            if isinstance(n.op,ast.Div):
                if r<=0<=s:raise ValueError('division interval includes zero')
                r,s=1/s,1/r
            vals=[l*r,l*s,h*r,h*s];return min(vals),max(vals)
        raise ValueError('unsupported syntax')
    return walk(tree.body)


def check_equation(raw):
    s=normalize(raw)
    if any(t in s for t in ['≈','\\approx','<','>','%']):return []
    parts=[p.strip() for p in s.split('=')]
    if len(parts)<2 or any(not p for p in parts):return []
    try:
        ranges=[interval(p) for p in parts]
        if not any(isinstance(n,ast.BinOp) for p in parts for n in ast.walk(ast.parse(p,mode='eval'))):return []
    except (ValueError,SyntaxError,ZeroDivisionError,OverflowError):return []
    return [{'raw':raw,'normalized':s,'left':parts[i],'right':parts[i+1],'contradiction':bool(max(ranges[i][0],ranges[i+1][0])>min(ranges[i][1],ranges[i+1][1]))} for i in range(len(parts)-1)]


def check_text(text):
    # Take complete math spans, not numeric substrings of symbolic/unit expressions.
    spans=re.findall(r'\$\$(.*?)\$\$|(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)|\\\[(.*?)\\\]|\\\((.*?)\\\)',text,flags=re.S)
    pieces=[next((s for s in match if s),'') for match in spans]
    # Standalone numeric equality lines; no removal of words/units to force a match.
    pieces += [line.strip().strip('$* ') for line in text.split('\n') if '=' in line and not re.match(r'^\s*[-•]\s',line)]
    records=[];seen=set()
    for piece in pieces:
        for fragment in re.split(r'\\\\|\n',piece):
            for r in check_equation(fragment.strip()):
                key=(r['left'],r['right'])
                if key not in seen:seen.add(key);records.append(r)
    return records
