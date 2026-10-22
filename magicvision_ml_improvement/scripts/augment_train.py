"""Optional moderate photometric augmentation of a RELEASED train split only.

Preserves geometry/boxes and collection groups. Derived images never count as
newly collected scenes. Validation/test images are copied unchanged.
"""
import argparse
import json
import random
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
from data_pipeline import read_manifest, record, write_manifest, save_json, sha


def augment(source, output, seed=42):
    source=Path(source);output=Path(output)
    report=json.loads((source/'split_report.json').read_text())
    if not report.get('training_ready'):raise ValueError('Release reviewed dataset before augmentation')
    rows=read_manifest(source/'manifest.jsonl')
    if not rows or any(r['review_status']!='approved' or not r.get('coverage_complete') for r in rows):
        raise ValueError('Unreviewed annotations')
    for row in rows:
        if row.get('reviewed_image_sha256') and sha(row['image'])!=row['reviewed_image_sha256']:
            raise ValueError('Image changed after review')
        if row.get('reviewed_label_sha256') and sha(row['label'])!=row['reviewed_label_sha256']:
            raise ValueError('Label changed after review')
    if output.exists():raise ValueError('Use a new output directory')
    shutil.copytree(source,output)
    rng=random.Random(seed);result=[];derived=0
    for row in rows:
        image=output/Path(row['image']).relative_to(source.resolve())
        label=output/Path(row['label']).relative_to(source.resolve())
        result.append({**row,'image':str(image.resolve()),'label':str(label.resolve())})
        if row['split']!='train' or row.get('derived_from'):continue
        settings={'brightness':rng.uniform(.85,1.15),'contrast':rng.uniform(.9,1.1),
                  'blur_radius':rng.uniform(0,.6),'jpeg_quality':rng.randint(75,95)}
        target=image.with_name(image.stem+'_photo.jpg')
        target_label=label.with_name(label.stem+'_photo.txt')
        with Image.open(image) as im:
            im=ImageEnhance.Brightness(im.convert('RGB')).enhance(settings['brightness'])
            im=ImageEnhance.Contrast(im).enhance(settings['contrast'])
            im=im.filter(ImageFilter.GaussianBlur(settings['blur_radius']))
            im.save(target,quality=settings['jpeg_quality'])
        shutil.copy2(label,target_label)
        extra={k:v for k,v in row.items() if k not in {'id','image','label','source','group','width','height','sha256','phash'}}
        extra.update(derived_from=row['sha256'],augmentation=settings,split='train')
        if row.get('reviewed_image_sha256'):
            extra.update(reviewed_source_image_sha256=row['reviewed_image_sha256'],
                         reviewed_image_sha256=sha(target),review_derivation='photometric_only_geometry_and_labels_preserved')
        result.append(record(target,target_label,row['source'],group=row['group'],**extra));derived+=1
    import yaml
    data=yaml.safe_load((output/'data.yaml').read_text());data['path']=str(output.resolve())
    (output/'data.yaml').write_text(yaml.safe_dump(data,sort_keys=False),encoding='utf-8')
    write_manifest(output/'manifest.jsonl',result)
    report.update(augmentation_seed=seed,derived_training_images=derived,
                  original_manifest_sha256=sha(source/'manifest.jsonl'),
                  note='Only train augmented; derived images do not count as independent scenes')
    report['images']['train']=report['images'].get('train',0)+derived
    save_json(output/'split_report.json',report)
    return derived


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();print('Derived train images:',augment(a.dataset,a.output,a.seed))
