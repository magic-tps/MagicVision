"""Reproduce a small source inspection sheet; this does not approve annotations."""
from PIL import Image, ImageDraw
from campus_schema import ROOT, read_manifest, read_labels, CLASSES


def main():
    base=ROOT/'datasets/campus_additions_v5'
    road=read_manifest(base/'roadwork/manifest.jsonl')
    hazards=read_manifest(base/'hazards_robots/manifest.jsonl')
    selected=[]
    for cls in (20,23):
        selected.extend([r for r in road if cls in {b[0] for b in read_labels(r['label'])}][:2])
    for cls in ('cable','wet_floor','cone'):
        selected.extend([r for r in hazards if cls in r.get('candidate_classes',[])][:2])
    sheet=Image.new('RGB',(1000,340*((len(selected)+1)//2)),'white')
    for i,row in enumerate(selected):
        with Image.open(row['image']) as source:
            image=source.convert('RGB');image.thumbnail((490,300))
        draw=ImageDraw.Draw(image);w,h=image.size
        for cls,x,y,bw,bh in read_labels(row['label']):
            bounds=((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h)
            draw.rectangle(bounds,outline='red',width=2)
            draw.text(bounds[:2],CLASSES[cls],fill='red')
        x0=(i%2)*500;y0=(i//2)*340
        sheet.paste(image,(x0,y0+35))
        label=row['id']+' '+','.join(row.get('candidate_classes',[]))
        ImageDraw.Draw(sheet).text((x0+3,y0+3),label,fill='black')
        if row.get('candidate_classes'):
            ImageDraw.Draw(sheet).text((x0+3,y0+17),'Candidate only - no boxes',fill='black')
    sheet.save(ROOT/'reports/campus_v5_sources_sample.jpg',quality=90)


if __name__=='__main__':main()
