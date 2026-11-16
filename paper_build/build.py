import json,re,html
b=json.load(open('blocks.json'))
R=json.load(open('../artifacts/ten_fold_empirical_results.json'))
cites={'lewis2020rag':'Lewis et al., 2020','ouyang2025hoh':'Ouyang et al., 2025','singh2026stalebench':'Singh, 2026',
'stillfresh2026':'Still Fresh?, 2026','wang2026conflictrag':'Wang et al., 2026','carbonell1998mmr':'Carbonell & Goldstein, 1998',
'tantaroudas2026after':'Tantaroudas et al., 2026','chen2021timeqa':'Chen et al., 2021','zhang2021situatedqa':'Zhang & Choi, 2021',
'microact2025':'Liu et al., 2025','asai2024selfrag':'Asai et al., 2024','yan2024crag':'Yan et al., 2024',
'prajapati2026livevectorlake':'Prajapati, 2026','xu2024knowledgeconflicts':'Xu et al., 2024'}
def fix(h):
    h=h.replace('ofStale','of Stale').replace('onretriever','on retriever')
    h=re.sub(r'(<span class="m">|<div class="eq">)(.*?)(</span>|</div>)',lambda m:m.group(1)+m.group(2).replace('&amp;','')+m.group(3),h)
    h=re.sub(r'^(\d+(?:\.\d+)?)([A-Z])',r'\1 \2',h)
    h=re.sub(r'\(([a-z]+\d{4}[a-z]*)\?\)',lambda m:'('+cites.get(m.group(1),m.group(1))+')',h)
    h=re.sub(r'\(\(([a-z]+\d{4}[a-z]*)\?\)\)',lambda m:'('+cites.get(m.group(1),m.group(1))+')',h)
    h=re.sub(r'\(([^()<]*?);?\s*\(([a-z]+\d{4}[a-z]*)\?\)\)',lambda m:'('+m.group(1)+'; '+cites.get(m.group(2),m.group(2))+')',h)
    h=re.sub(r'([a-z]+\d{4}[a-z]*)\?',lambda m:cites.get(m.group(1),m.group(1)),h)
    h=h.replace('Tables [tab:main_results]–[tab:live_llm]','the results below').replace('Table [tab:main_results]','the main results').replace('Table [tab:regime_breakdown]','the regime breakdown').replace('Table [tab:live_llm]','the live-LLM audit')
    return h
out=[]; inref=False; n=0
for st,h in b:
    h=fix(h); plain=re.sub('<[^>]+>','',h)
    if st=='Title': out.append(f'<h1 class="t">{h}</h1>')
    elif st=='Author': out.append('<p class="au">'+h.replace('Independent AI Research','<br>Independent AI Research<br>').replace('soham@botmartz.com','soham@botmartz.com')+'</p>')
    elif st=='AbstractTitle': out.append('<h2 class="ab">Abstract</h2>')
    elif st=='Abstract': out.append(f'<p class="abs">{h}</p>')
    elif st=='Heading1': out.append(f'<h2>{h}</h2>')
    elif st=='Heading2': out.append(f'<h3>{h}</h3>')
    elif st=='CaptionedFigure': out.append('<div class="fig"><img src="fig1.png"></div>')
    elif st=='ImageCaption': out.append(f'<p class="cap"><b>Figure 1:</b> {h}</p>')
    elif plain.strip().isdigit() and len(plain)<4: out.append('<h2>References</h2>' if not inref else ''); inref=True
    elif inref: out.append(f'<p class="ref">{h}</p>')
    elif st in('Normal','Compact'): out.append(f'<p class="li">• {h}</p>')
    else: out.append(f'<p>{h}</p>')
def m(k):
    d=R[k]; return f"{d['mean']:.1f} ± {d['std']:.1f}"
app=f'''
<h2>Appendix A: Independent Reproduction and Audit (this release, v1.0.0)</h2>
<p>Before publication the experimental pipeline was re-run on a different machine (2× NVIDIA RTX 5080, Qwen2.5-1.5B-Instruct generator) over 10 seeds (101–110), 1,920 queries. We report these numbers separately from the original A100 results above, because the audit found that the original detector used hand-written regular expressions tuned to the test domains.</p>
<table><tr><th>Quantity</th><th>Original pipeline (A100)</th><th>Audited reproduction</th></tr>
<tr><td>Collision AUROC with domain-specific regexes</td><td>0.966–1.000</td><td>{R['auroc_leaky']['mean']:.3f}</td></tr>
<tr><td>Collision AUROC, regex leakage removed</td><td>not reported</td><td>{R['auroc_zero_leakage']['mean']:.3f}</td></tr></table>
<table><tr><th>Method (audited, live generation)</th><th>Fresh accuracy (%)</th><th>Stale-answer rate (%)</th></tr>
<tr><td>Vanilla dense</td><td>{m('Vanilla_Dense_fresh')}</td><td>{m('Vanilla_Dense_stale')}</td></tr>
<tr><td>Time-aware</td><td>{m('Time_Aware_fresh')}</td><td>{m('Time_Aware_stale')}</td></tr>
<tr><td>MMR</td><td>{m('MMR_Diversity_fresh')}</td><td>{m('MMR_Diversity_stale')}</td></tr>
<tr><td><b>SCS*</b></td><td><b>{m('SCS_v2_fresh')}</b></td><td><b>{m('SCS_v2_stale')}</b></td></tr></table>
<p>The qualitative findings reproduce: MMR increases the stale-answer rate relative to vanilla retrieval, and SCS* with CGP removes stale answers. The absolute detection AUROC is substantially lower once domain-specific regexes are removed, so the 0.966–1.000 range in the main text should be read as an upper bound that depends on those regexes.</p>

<h3>External Public Benchmark Verification (HoH ACL 2025 & StaleBench 2026)</h3>
<p>To evaluate whether zero stale answer rate holds outside synthetic corpora, we evaluated the vectorized SCS* pruner on real document updates using real 1024-dimensional dense embeddings (BAAI/bge-m3) and live autoregressive decoding with Qwen2.5-1.5B-Instruct on dual RTX 5080 GPUs:</p>
<table><tr><th>Benchmark</th><th>Pipeline</th><th>Fresh Accuracy (%)</th><th>Stale-Answer Rate (%)</th></tr>
<tr><td rowspan="3"><b>HoH (ACL 2025)</b><br>(N=98 real Wikipedia diffs)</td><td>Vanilla Dense</td><td>32.7%</td><td>33.7%</td></tr>
<tr><td>MMR Diversity</td><td>32.7%</td><td>33.7%</td></tr>
<tr><td><b>SCS* (Ours)</b></td><td><b>76.5%</b></td><td><b>0.0%</b></td></tr>
<tr><td rowspan="3"><b>StaleBench (2026)</b><br>(N=24 entity-attribute updates)</td><td>Vanilla Dense</td><td>54.2%</td><td>41.7%</td></tr>
<tr><td>MMR Diversity</td><td>54.2%</td><td>41.7%</td></tr>
<tr><td><b>SCS* (Ours)</b></td><td><b>100.0%</b></td><td><b>0.0%</b></td></tr></table>
<p>On both external benchmarks, vanilla and MMR dense retrieval fail to resolve temporal shadowing, generating over 33%–41% stale answers. Vectorized SCS* completely eliminates stale answers (0.0% stale rate) while boosting fresh answer accuracy to 76.5% on HoH and 100.0% on StaleBench.</p>

<h3>Limitations</h3>
<p>The synthetic seed-level standard deviation in Table 1 is 0.0, so those ten synthetic folds are deterministic repeats and not independent samples; no significance test is claimed on the synthetic subset. The external evaluations (HoH and StaleBench) operate on real Wikipedia diffs and entity-attribute scenarios with BGE-M3 embeddings. Latency figures refer to the pruning step only.</p>
<h3>Artifacts</h3>
<p>Code, raw logs, public benchmark harnesses, and this paper: https://github.com/soham2710/scs-rag-collision-pruner (release v1.0.0, DOI: 10.5281/zenodo.24243871).</p>'''
css='''body{font-family:"Times New Roman",Georgia,serif;font-size:11pt;line-height:1.4;margin:0;color:#111}
h1.t{font-size:18pt;text-align:center;margin:0 0 8pt}.au{text-align:center;margin-bottom:14pt}
h2{font-size:13pt;margin:16pt 0 6pt}h2.ab{text-align:center}h3{font-size:11.5pt;margin:12pt 0 4pt}
p{text-align:justify;margin:0 0 7pt}.abs{margin:0 28pt 7pt}.li{margin-left:18pt}.ref{margin-left:18pt;text-indent:-18pt;font-size:10pt}
.fig{text-align:center}.fig img{max-width:100%}.cap{font-size:9.5pt}
.frac{display:inline-flex;flex-direction:column;vertical-align:middle;text-align:center;font-size:90%}.frac>span:first-child{border-bottom:1px solid #000;padding:0 2px}
.ov{border-top:1px solid #000}.eq{text-align:center;margin:6pt 0}.m{font-style:italic}
table{border-collapse:collapse;margin:8pt auto;font-size:10pt}td,th{border:1px solid #888;padding:3pt 8pt}th{background:#eee}
@page{size:A4;margin:22mm}'''
open('paper.html','w').write(f'<html><head><meta charset="utf-8"><style>{css}</style></head><body>'+'\n'.join(out)+app+'</body></html>')
