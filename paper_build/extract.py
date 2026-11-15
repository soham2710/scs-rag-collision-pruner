import zipfile, re, json, html
from lxml import etree
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M='http://schemas.openxmlformats.org/officeDocument/2006/math'
def q(t): 
    p,n=t.split(':'); return '{%s}%s'%({'w':W,'m':M}[p],n)
SUP=str.maketrans("0123456789+-=()n","⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")
SUB=str.maketrans("0123456789+-=()","₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")
def esc(s): return html.escape(s)
def mt(e):
    """OMML element -> HTML string"""
    tag=etree.QName(e).localname
    kids=lambda x:''.join(mt(c) for c in x)
    def sub(name):
        c=e.find(q('m:'+name)); return kids(c) if c is not None else ''
    if tag=='t': return esc(e.text or '')
    if tag=='f': return f'<span class="frac"><span>{sub("num")}</span><span>{sub("den")}</span></span>'
    if tag=='sSup': return f'{sub("e")}<sup>{sub("sup")}</sup>'
    if tag=='sSub': return f'{sub("e")}<sub>{sub("sub")}</sub>'
    if tag=='sSubSup': return f'{sub("e")}<sub>{sub("sub")}</sub><sup>{sub("sup")}</sup>'
    if tag=='rad': return f'√<span class="ov">{sub("e")}</span>'
    if tag=='nary':
        ch=e.find('.//'+q('m:chr')); c=ch.get(q('m:val')) if ch is not None else '∫'
        return f'{c}<sub>{sub("sub")}</sub><sup>{sub("sup")}</sup>{sub("e")}'
    if tag=='d':
        pr=e.find(q('m:dPr')); b,eN='(',')'
        if pr is not None:
            bc=pr.find(q('m:begChr')); ec=pr.find(q('m:endChr'))
            if bc is not None: b=bc.get(q('m:val'))
            if ec is not None: eN=ec.get(q('m:val'))
        inner='，'.join(kids(x) for x in e.findall(q('m:e'))).replace('，',', ')
        return b+inner+eN
    if tag=='acc':
        ch=e.find('.//'+q('m:chr')); return f'<span class="acc">{sub("e")}{(ch.get(q("m:val")) if ch is not None else "^")}</span>'
    if tag=='bar': return f'<span class="ov">{sub("e")}</span>'
    if tag in ('func',): return f'{sub("fName")}{sub("e")}'
    if tag=='limLow': return f'{sub("e")}<sub>{sub("lim")}</sub>'
    if tag=='limUpp': return f'{sub("e")}<sup>{sub("lim")}</sup>'
    if tag=='eqArr': return '<br>'.join(kids(x) for x in e.findall(q('m:e')))
    if tag=='m': return ' ; '.join(kids(r) for r in e.findall(q('m:mr')))
    if tag=='groupChr': return sub('e')
    if tag in ('r','e','num','den','sub','sup','fName','lim','oMath','mr','deg'): return kids(e)
    if tag.endswith('Pr'): return ''
    return kids(e)
def para(p):
    out=[]
    for c in p:
        t=etree.QName(c).localname
        if t=='r':
            txt=''.join(x.text or '' for x in c.findall(q('w:t')))
            if c.find(q('w:drawing')) is not None: out.append('@@IMG@@')
            rp=c.find(q('w:rPr'))
            if txt:
                s=esc(txt)
                if rp is not None and rp.find(q('w:i')) is not None: s=f'<i>{s}</i>'
                if rp is not None and rp.find(q('w:b')) is not None: s=f'<b>{s}</b>'
                out.append(s)
        elif t=='oMath': out.append('<span class="m">'+mt(c)+'</span>')
        elif t=='oMathPara':
            out.append('<div class="eq">'+''.join(mt(x) for x in c.findall(q('m:oMath')))+'</div>')
        elif t in('hyperlink','smartTag','ins'): out.append(para(c))
    return ''.join(out)
z=zipfile.ZipFile('original_RAG.docx')
root=etree.fromstring(z.read('word/document.xml'))
blocks=[]
for p in root.find(q('w:body')):
    t=etree.QName(p).localname
    if t=='p':
        ps=p.find(q('w:pPr')); st=ps.find(q('w:pStyle')).get(q('w:val')) if ps is not None and ps.find(q('w:pStyle')) is not None else 'Normal'
        h=para(p).strip()
        if h: blocks.append((st,h))
    elif t=='tbl': blocks.append(('TABLE','[TABLE]'))
json.dump(blocks,open('blocks.json','w'),ensure_ascii=False,indent=1)
print(len(blocks))
for st,h in blocks[:40]: print(st,'|',re.sub('<[^>]+>','',h)[:400]);print()
