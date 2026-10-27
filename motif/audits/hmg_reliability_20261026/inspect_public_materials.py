"""Read-only extraction/search of the downloaded primary article and supplement."""
import hashlib
from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path
import re

from pypdf import PdfReader

OUT=Path(__file__).resolve().parent
class Text(HTMLParser):
    def __init__(self):super().__init__();self.parts=[];self.hidden=0
    def handle_starttag(self,tag,attrs):
        if tag in ('script','style'):self.hidden+=1
    def handle_endtag(self,tag):
        if tag in ('script','style'):self.hidden=max(0,self.hidden-1)
    def handle_data(self,data):
        if not self.hidden:self.parts.append(data)

parser=Text();parser.feed((OUT/'paper_article.html').read_text(encoding='utf-8-sig'))
article=re.sub(r'\s+',' ',' '.join(parser.parts))
(OUT/'paper_fulltext.txt').write_text(article,encoding='utf-8')
reader=PdfReader(OUT/'btag600_supplementary_data.pdf')
pages=[page.extract_text(extraction_mode='layout') for page in reader.pages]
(OUT/'supplement_fulltext.txt').write_text('\n\n'.join(f'PAGE {i+1}\n{page}' for i,page in enumerate(pages)),encoding='utf-8')
patterns={'target_scaling':r'(?:target|label|response).{0,70}(?:normaliz|standardi|scal)',
          'epoch':r'epochs?', 'scheduler':r'scheduler|Noam|warm.?up|learning.rate',
          'scaffold':r'scaffold', 'descriptor_standardization':r'descriptor.{0,180}standardi'}
evidence={}
for name,pattern in patterns.items():
    hits=[]
    for label,text in [('article',article)]+[(f'supplement_page_{i+1}',re.sub(r'\s+',' ',page)) for i,page in enumerate(pages)]:
        for match in re.finditer(pattern,text,re.I):
            hits.append(dict(source=label,context=text[max(0,match.start()-80):min(len(text),match.end()+100)]))
    evidence[name]=hits
record=dict(article_url='https://academic.oup.com/bioinformatics/article/42/8/btag600/8758503',
    article_sha256=hashlib.sha256((OUT/'paper_article.html').read_bytes()).hexdigest(),
    supplement_sha256=hashlib.sha256((OUT/'btag600_supplementary_data.pdf').read_bytes()).hexdigest(),
    supplement_pages=len(pages),article_characters=len(article),search=evidence,
    limitation='No keyword hit is not proof of absence in author training code or unshared materials.')
(OUT/'public_materials_search.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(article_characters=len(article),supplement_pages=len(pages),
    hits={k:len(v) for k,v in evidence.items()}),ensure_ascii=False))
for name in ('target_scaling','epoch','scheduler','descriptor_standardization'):
    print(name)
    for hit in evidence[name]:print(json.dumps(hit,ensure_ascii=True))
if importlib.util.find_spec('pypdfium2'):
    import pypdfium2 as pdfium
    pdf=pdfium.PdfDocument(str(OUT/'btag600_supplementary_data.pdf'))
    for i in (1,11,12):
        image=pdf[i].render(scale=1.5).to_pil();image.save(OUT/f'supplement_page_{i+1}.png')
    print('Rendered supplementary pages 2, 12, 13')
