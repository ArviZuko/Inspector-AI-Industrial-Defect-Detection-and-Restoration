
from __future__ import annotations

import sys
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
import re
import gc
import json
import time
import traceback
from uuid import uuid4
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext

import cv2
import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F

from PyQt6.QtCore import (
    Qt, QObject, QThread, QTimer, pyqtSignal, QEvent, QSettings
)
from PyQt6.QtGui import QImage, QPixmap, QIntValidator, QIcon
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame,
    QLabel, QPushButton, QLineEdit, QComboBox,
    QListWidget, QListWidgetItem, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QHBoxLayout,
    QGridLayout, QFileDialog, QMessageBox,
    QDialog, QScrollArea, QSplitter, QSizePolicy,
    QHeaderView, QAbstractItemView, QInputDialog, QProgressBar, QSpinBox
)


# ============================================================
# Integrated inspection_reporting.py
# ============================================================
"""PDF reporting and measured batch statistics for Inspectra AI."""
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

DEFECTS = ('Blowhole', 'Crack', 'Fray', 'Break')
METRICS = {
 'brisque_score': ('BRISQUE', 'lower'),
 'psnr_db': ('PSNR (dB)', 'higher'), 'ssim': ('SSIM', 'higher'),
 'mse': ('MSE', 'lower'), 'mae': ('MAE', 'lower'),
 'sharpness_laplacian_variance': ('Sharpness', None),
 'entropy_bits': ('Entropy (bits)', None), 'contrast_std': ('Contrast', None),
 'processing_seconds': ('Stage time (s)', 'lower')}

def fmt(value):
 if value is None: return 'N/A'
 if isinstance(value, (float, int)): return f'{value:.5g}'
 return str(value)

def enrich_report(report, images, root):
 report['image_dimensions'] = {key: list(im.size) for key, im in images.items() if im is not None}
 dims=report['image_dimensions']; original=dims.get('original')
 sr_input=dims.get('restored', original); sr=dims.get('super_resolution')
 report['super_resolution_extent'] = ({'input_dimensions':sr_input, 'output_dimensions':sr,
   'width_scale':sr[0]/sr_input[0], 'height_scale':sr[1]/sr_input[1],
   'input_pixels':sr_input[0]*sr_input[1], 'output_pixels':sr[0]*sr[1],
   'pixel_increase_percent':100*(sr[0]*sr[1]/(sr_input[0]*sr_input[1])-1)} if sr and sr_input else None)
 rows=[]
 for defect in DEFECTS:
  models=[m for m in report['defect_models'] if m['defect'].lower()==defect.lower()]
  detections=[d for m in models for d in m['detections']]
  scores=[d['confidence']*100 for d in detections]
  rows.append({'defect':defect,'status':'Inspected' if models else 'Not inspected',
    'count':len(detections) if models else None,
    'confidence_mean':sum(scores)/len(scores) if scores else None,
    'confidence_min':min(scores) if scores else None,'confidence_max':max(scores) if scores else None,
    'summed_region_area_pixels':sum(d['area_pixels'] for d in detections)})
 report['defect_summary']=rows
 # A stable inspection ID prevents repeated saves from inflating batch statistics.
 records={report['inspection_id']:report}
 for path in Path(root).glob('*/inspection_report.json'):
  try:
   other=json.loads(path.read_text(encoding='utf-8'))
   if report.get('session_id') and other.get('session_id') != report['session_id']: continue
   if (other.get('batch_number'),other.get('product_id')) != (report['batch_number'],report['product_id']): continue
   if not other.get('inspection_id'): continue
   records.setdefault(other['inspection_id'],other)
  except (OSError,ValueError,KeyError): continue
 report['batch_inspection_count']=len(records)
 stats={}
 for stage,model_key in [('restored','restoration_model'),('super_resolution','super_resolution_model')]:
  model=report.get(model_key)
  if not model: continue
  compatible=[r for r in records.values() if r.get(model_key)==model]
  stats[stage]={}
  for metric,(label,direction) in METRICS.items():
   values=[]; infinite=0
   for r in compatible:
    value=r.get('quality_metrics',{}).get(stage,{}).get(metric)
    if value=='Infinity': infinite+=1;continue
    if isinstance(value,(int,float)) and math.isfinite(value): values.append(float(value))
   stats[stage][metric]={'n':len(values),'infinite_count':infinite,
    'average':sum(values)/len(values) if values else None,
    'min':min(values) if values else None,'max':max(values) if values else None,
    'best':(min(values) if direction=='lower' else max(values)) if values and direction else None,
    'worst':(max(values) if direction=='lower' else min(values)) if values and direction else None,
    'direction':direction}
 report['batch_metric_statistics']=stats
 return report

def write_pdf(path, report, images):
 from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as PDFImage, KeepTogether
 from reportlab.lib import colors
 from reportlab.lib.styles import getSampleStyleSheet
 from reportlab.lib.pagesizes import A4
 from io import BytesIO
 styles=getSampleStyleSheet()
 styles['Normal'].fontSize=9;styles['Normal'].leading=12
 styles['Title'].textColor=colors.HexColor('#12263e')
 doc=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=34,leftMargin=34,topMargin=42,bottomMargin=40,
   title='Inspectra AI - Assembly Line Inspection',author='Inspectra AI')
 story=[];width=A4[0]-68
 def p(text,style='Normal'): return Paragraph(escape(str(text)).replace('\n','<br/>'),styles[style])
 def heading(text): story.extend([p(text,'Heading2'),Spacer(1,5)])
 def table(rows,widths=None):
  t=Table([[p(c) for c in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
  t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eef5')),('VALIGN',(0,0),(-1,-1),'TOP'),
   ('GRID',(0,0),(-1,-1),.3,colors.HexColor('#cbd5e1')),('LEFTPADDING',(0,0),(-1,-1),6),
   ('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
  story.extend([t,Spacer(1,10)])
 story.extend([p('Inspectra AI','Title'),p('ASSEMBLY LINE INSPECTION REPORT','Heading2'),p('Per-sample inspection with measured batch statistics'),Spacer(1,12)])
 table([['Identification','Value'],['Inspection ID',report['inspection_id']],['Saved at (local time)',report['timestamp']],
  ['Worker / Batch',f"{report['worker_id']} / {report['batch_number']}"],['Product / Sample',f"{report['product_id']} / {report['sample_number']}"],
  ['Device / Total processing time',f"{report['device']} / {report['processing_time_seconds']} s"],
  ['Detection source',report['detection_source']]], [155,width-155])
 heading('Defect counts and confidence')
 table([['Defect','Status','Regions','Mean %','Lowest %','Highest %']]+[[r['defect'],r['status'],fmt(r['count']),fmt(r['confidence_mean']),fmt(r['confidence_min']),fmt(r['confidence_max'])] for r in report['defect_summary']], [75,90,60,75,75,width-375])
 story.append(p('Counts are model-detected regions, not a certified count of physical defects. Overlapping model results may count the same physical region more than once. Confidence is model probability, not a statistical confidence interval. No automatic pass/fail is assigned because acceptance limits have not been configured.'))
 heading('Processing warnings')
 story.append(p('\n'.join(report['warnings']) or 'None'))
 story.append(PageBreak())
 heading('Restoration and super-resolution performance')
 for stage,key in [('restored','restoration_model'),('super_resolution','super_resolution_model')]:
  model=report.get(key);m=report.get('quality_metrics',{}).get(stage,{})
  heading(stage.replace('_',' ').title())
  story.append(p('Model: '+(model['name'] if model else 'Not selected')))
  if not m: continue
  story.append(p(m.get('reference_status','')))
  table([['Metric','Current sample']]+[[label,fmt(m.get(metric))] for metric,(label,_) in METRICS.items()],[width*.6,width*.4])
  story.append(p('Protocol: '+m.get('protocol','')))
 extent=report.get('super_resolution_extent')
 if extent:
  heading('Super-resolution extent')
  story.append(p(f"Input: {extent['input_dimensions'][0]} x {extent['input_dimensions'][1]} px; output: {extent['output_dimensions'][0]} x {extent['output_dimensions'][1]} px. Scale: {extent['width_scale']:.2f}x width and {extent['height_scale']:.2f}x height. Pixels: {extent['input_pixels']:,} to {extent['output_pixels']:,} ({extent['pixel_increase_percent']:.1f}% increase). Upscaling does not establish that all added detail is real."))
 story.append(PageBreak())
 heading('Measured batch performance')
 story.append(p(f"Batch {report['batch_number']} / Product {report['product_id']}: {report['batch_inspection_count']} distinct saved inspections, including this inspection. Statistics use matching model name, architecture and checkpoint. Re-saving the same inspection does not increase the sample count."))
 for stage,stats in report['batch_metric_statistics'].items():
  heading(stage.replace('_',' ').title())
  table([['Metric','Valid N','Average','Lowest','Highest','Best / Worst']]+[
   [METRICS[k][0],v['n'],fmt(v['average']),fmt(v['min']),fmt(v['max']),fmt(v['best'])+' / '+fmt(v['worst']) if v['direction'] else 'Not ranked']
   for k,v in stats.items()],[100,48,72,72,72,width-364])
  perfect=stats.get('psnr_db',{}).get('infinite_count',0)
  if perfect: story.append(p(f'{perfect} perfect-match PSNR values are infinite and listed separately from finite averages.'))
 story.append(p('Higher PSNR/SSIM and lower MSE/MAE indicate better agreement with the loaded clean references. Stage time: lower is faster; hardware and image dimensions affect comparability. Sharpness, entropy and contrast are descriptive statistics, so no best/worst quality ranking is assigned. Missing reference metrics are excluded; different reference images and content can affect aggregate comparisons.'))
 story.append(PageBreak())
 heading('Image evidence - full inspection results')
 story.append(p('The report includes all inspected defect classes, independent of the on-screen visibility filters.'))
 for key,label in [('original','Original'),('restored','Restored'),('super_resolution','Super-resolution'),('combined_detection','Combined defect detection')]:
  im=images.get(key)
  if im is None: continue
  stream=BytesIO();im.convert('RGB').save(stream,format='PNG');stream.seek(0)
  scale=min((width-10)/im.width,220/im.height)
  story.append(KeepTogether([p(f'{label} - {im.width} x {im.height} px','Heading3'),PDFImage(stream,width=im.width*scale,height=im.height*scale),Spacer(1,10)]))
 story.append(PageBreak())
 heading('Detection region register')
 rows=[['#','Defect / Model','Conf. %','Area px','Box x1,y1,x2,y2']]
 for model in report['defect_models']:
  for d in model['detections']:
   rows.append([len(rows),d['defect']+' / '+model['name'],f"{d['confidence']*100:.2f}",fmt(d['area_pixels']),', '.join(map(str,d['box']))])
 if len(rows)>1: table(rows,[28,175,60,65,width-328])
 else: story.append(p('No detected regions were recorded. See the summary for classes that were not inspected.'))
 heading('Models and traceability')
 for model in [report.get('restoration_model'),report.get('super_resolution_model')]+report['defect_models']:
  if model: story.append(p(f"{model['name']} | {model['architecture']} | {model['checkpoint']}"))
 for stage,reference in report.get('metric_reference_files',{}).items(): story.append(p(stage+' reference: '+reference))
 def footer(canvas,document):
  canvas.saveState();canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#526579'))
  canvas.drawString(34,22,'Inspectra AI | '+str(report['sample_number']))
  canvas.drawRightString(A4[0]-34,22,'Page '+str(document.page));canvas.restoreState()
 doc.build(story,onFirstPage=footer,onLaterPages=footer)


# ============================================================
# Integrated batch_reporting.py
# ============================================================
"""Persistent assembly-line samples and collective batch PDF reports."""
import sqlite3,json,math
from pathlib import Path
from datetime import datetime
from uuid import uuid4
from collections import defaultdict
from xml.sax.saxutils import escape

class BatchStore:
 def __init__(self,root,session_id=None):
  self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
  self.session_id=session_id or uuid4().hex
  session_folder=self.root/'sessions'/self.session_id
  session_folder.mkdir(parents=True,exist_ok=True)
  self.database=session_folder/'assembly_line.sqlite3'
  with self.connect() as db:
   db.execute('''CREATE TABLE IF NOT EXISTS samples (
    id TEXT PRIMARY KEY, batch TEXT NOT NULL, product TEXT NOT NULL, worker TEXT NOT NULL,
    number INTEGER NOT NULL, captured TEXT NOT NULL, image_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'Captured', report_path TEXT,
    UNIQUE(batch,number))''')
   db.execute("CREATE TABLE IF NOT EXISTS batches (batch TEXT PRIMARY KEY, capacity INTEGER NOT NULL)")
 def capacity(self,batch):
  with self.connect() as db:
   row=db.execute("SELECT capacity FROM batches WHERE batch=?",(batch,)).fetchone()
   return row[0] if row else None
 def set_capacity(self,batch,capacity):
  if capacity < 1: raise ValueError("Number of samples must be at least 1.")
  with self.connect() as db:
   db.execute("BEGIN IMMEDIATE")
   row=db.execute("SELECT capacity FROM batches WHERE batch=?",(batch,)).fetchone()
   count=db.execute("SELECT COUNT(*) FROM samples WHERE batch=?",(batch,)).fetchone()[0]
   if row and row[0] != capacity and count:
    raise ValueError("The batch size cannot change after samples have been collected.")
   db.execute("INSERT INTO batches(batch,capacity) VALUES(?,?) ON CONFLICT(batch) DO UPDATE SET capacity=excluded.capacity",(batch,capacity))
 def connect(self):
  db=sqlite3.connect(self.database);db.row_factory=sqlite3.Row;return db
 def capture(self,image,metadata):
  sample_id=uuid4().hex
  folder=self.root/'batch_samples'/sample_id;folder.mkdir(parents=True)
  image_path=folder/'captured.png';image.save(image_path)
  with self.connect() as db:
   db.execute('BEGIN IMMEDIATE')
   number=db.execute('SELECT COALESCE(MAX(number),0)+1 FROM samples WHERE batch=?',(metadata['batch_number'],)).fetchone()[0]
   row=db.execute("SELECT capacity FROM batches WHERE batch=?",(metadata['batch_number'],)).fetchone()
   if row is None or number > row[0]:
    raise ValueError("Batch capacity is not set or the batch has reached its sample limit.")
   db.execute('INSERT INTO samples(id,batch,product,worker,number,captured,image_path) VALUES(?,?,?,?,?,?,?)',
     (sample_id,metadata['batch_number'],metadata['product_id'],metadata['worker_id'],number,datetime.now().isoformat(),str(image_path)))
  return self.get(sample_id)
 def get(self,sample_id):
  with self.connect() as db:return dict(db.execute('SELECT * FROM samples WHERE id=?',(sample_id,)).fetchone())
 def samples(self,batch):
  with self.connect() as db:return [dict(r) for r in db.execute('SELECT * FROM samples WHERE batch=? ORDER BY number',(batch,))]
 def inspected(self,sample_id,report_path,warnings):
  with self.connect() as db:db.execute('UPDATE samples SET status=?,report_path=? WHERE id=?',
   ('Inspected with warnings' if warnings else 'Inspected',str(report_path),sample_id))
 def failed(self,sample_id):
  with self.connect() as db:db.execute("UPDATE samples SET status='Failed' WHERE id=? AND report_path IS NULL",(sample_id,))

def batch_summary(samples):
 records=[]
 for sample in samples:
  if sample['report_path']:
   path=Path(sample['report_path'])
   record=json.loads(path.read_text(encoding='utf-8'));records.append((sample,record))
 counts={d:{'regions':0,'samples_with_regions':0,'inspected_samples':0} for d in DEFECTS}
 for _,report in records:
  for row in report['defect_summary']:
   if row['count'] is not None:
    c=counts[row['defect']];c['regions']+=row['count'];c['inspected_samples']+=1
    c['samples_with_regions']+=int(row['count']>0)
 groups={}
 for sample,report in records:
  for stage,key in [('restored','restoration_model'),('super_resolution','super_resolution_model')]:
   model=report.get(key);metrics=report.get('quality_metrics',{}).get(stage)
   if not model or not metrics:continue
   group_key=(sample['product'],stage,model['name'],model['architecture'],model['checkpoint'])
   group=groups.setdefault(group_key,{'samples':0,'metrics':defaultdict(list),'infinite_psnr':0})
   group['samples']+=1
   for name in METRICS:
    value=metrics.get(name)
    if value=='Infinity':group['infinite_psnr']+=1
    elif isinstance(value,(float,int)) and math.isfinite(value):group['metrics'][name].append(value)
 return records,counts,groups

def write_batch_pdf(path,samples,batch,capacity=None):
 """Exactly two pages of important averages; no images or sample appendices."""
 from reportlab.pdfgen import canvas
 from reportlab.platypus import Table,TableStyle,Paragraph
 from reportlab.lib import colors
 from reportlab.lib.styles import getSampleStyleSheet
 from reportlab.lib.pagesizes import A4
 records,counts,groups=batch_summary(samples)
 styles=getSampleStyleSheet();styles['Normal'].fontSize=9;styles['Normal'].leading=12
 width,height=A4;left=34;usable=width-68
 temp=Path(path).with_suffix('.tmp')
 c=canvas.Canvas(str(temp),pagesize=A4)
 c.setTitle('Inspectra AI - Two-page batch summary')
 def text(value,y,size=10,bold=False):
  c.setFont('Helvetica-Bold' if bold else 'Helvetica',size)
  c.setFillColor(colors.HexColor('#12263e'));c.drawString(left,y,str(value))
 def paragraph(value,y):
  item=Paragraph(escape(str(value)),styles['Normal']);_,h=item.wrap(usable,100)
  item.drawOn(c,left,y-h);return y-h-12
 def table(rows,widths,y):
  item=Table([[Paragraph(escape(str(cell)),styles['Normal']) for cell in row] for row in rows],colWidths=widths)
  item.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eef5')),
   ('GRID',(0,0),(-1,-1),.3,colors.HexColor('#cbd5e1')),('VALIGN',(0,0),(-1,-1),'TOP'),
   ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
  _,h=item.wrap(usable,height);item.drawOn(c,left,y-h);return y-h-20
 def header(title,page):
  text('Inspectra AI',height-45,20,True);text(title,height-69,12,True)
  label=str(batch)
  if len(label)>65:label=label[:62]+'...'
  text('Batch: '+label,height-91,10)
  text('Generated: '+datetime.now().strftime('%Y-%m-%d %H:%M:%S'),height-108,9)
  c.setFont('Helvetica',8);c.drawString(left,23,'Inspectra AI | Assembly line batch summary')
  c.drawRightString(width-left,23,f'Page {page} of 2')
 header('BATCH SAMPLE AND DEFECT SUMMARY',1)
 n=len(records);warnings=sum(bool(r['warnings']) for _,r in records)
 decisions={name:sum(r.get('inspector_decision')==name for _,r in records) for name in ('Accepted','Rejected','Needs Review')}
 y=table([['Sample coverage','Count'],['Planned samples in batch',capacity if capacity else 'Not specified'],['Captured samples',len(samples)],['Inspected samples',n],
  ['Accepted / Rejected / Needs Review',str(decisions['Accepted'])+' / '+str(decisions['Rejected'])+' / '+str(decisions['Needs Review'])],
  ['Awaiting inspector decision',sum(r.get('inspector_decision') not in decisions for _,r in records)],
  ['Pending / not yet inspected',len(samples)-n],['Failed without a saved result',sum(s['status']=='Failed' for s in samples)],
  ['Inspected samples with warnings',warnings],['Samples with detected regions',sum(r['total_model_detections']>0 for _,r in records)]],[usable*.73,usable*.27],height-130)
 text('Defect totals and averages',y,12,True);y-=14
 rows=[['Defect','Total regions','Samples inspected','Mean regions / sample','Mean confidence %']]
 for defect,values in counts.items():
  scores=[d['confidence']*100 for _,r in records for model in r['defect_models']
          for d in model['detections'] if d['defect']==defect]
  denominator=values['inspected_samples']
  rows.append([defect,values['regions'] if denominator else 'N/A',denominator,
    fmt(values['regions']/denominator) if denominator else 'N/A',fmt(sum(scores)/len(scores)) if scores else 'N/A'])
 y=table(rows,[78,75,95,135,usable-383],y)
 total_time=sum(r['processing_time_seconds'] for _,r in records)
 y=paragraph(f'Average inspection time: {total_time/n:.2f} s. Total processing time: {total_time:.2f} s.' if n else 'No inspected samples are available yet.',y)
 y=paragraph(f'Products represented: {len(set(s["product"] for s in samples))}. Workers represented: {len(set(s["worker"] for s in samples))}.',y)
 paragraph('Averages use the latest saved inspection of each captured sample. Re-inspection does not add a sample. Region averages use samples inspected for that defect class. Confidence is averaged across detected regions, not a statistical confidence interval. Accepted, Rejected and Needs Review are manual inspector decisions; model detections do not automatically assign acceptance.',y)
 c.showPage();header('AVERAGE IMAGE PROCESSING PERFORMANCE',2)
 pooled={stage:{key:[] for key in METRICS} for stage in ('restored','super_resolution')}
 stage_counts={stage:0 for stage in pooled};infinite={stage:0 for stage in pooled}
 for _,report in records:
  for stage in pooled:
   metrics=report.get('quality_metrics',{}).get(stage)
   if not metrics:continue
   stage_counts[stage]+=1
   for key in METRICS:
    value=metrics.get(key)
    if value=='Infinity':infinite[stage]+=1
    elif isinstance(value,(float,int)) and math.isfinite(value):pooled[stage][key].append(value)
 rows=[['Metric','Restoration mean','Valid N','Super-resolution mean','Valid N']]
 for key,(label,_) in METRICS.items():
  row=[label]
  for stage in pooled:
   values=pooled[stage][key];row.extend([fmt(sum(values)/len(values)) if values else 'N/A',len(values)])
  rows.append(row)
 y=table(rows,[110,135,48,135,usable-428],height-130)
 sr=[r['super_resolution_extent'] for _,r in records if r.get('super_resolution_extent')]
 text('Processing coverage and average upscaling',y,12,True);y-=14
 def mean(key):return fmt(sum(r[key] for r in sr)/len(sr)) if sr else 'N/A'
 rows=[['Summary measure','Batch value'],['Restored samples',stage_counts['restored']],['Super-resolved samples',stage_counts['super_resolution']],
  ['Average width / height scale',mean('width_scale')+'x / '+mean('height_scale')+'x'],
  ['Average increase in pixel count (%)',mean('pixel_increase_percent')]]
 y=table(rows,[usable*.7,usable*.3],y)
 y=paragraph('Valid N is the number of available finite measurements for that metric. PSNR, SSIM, MSE and MAE require aligned clean references; unavailable values are excluded. Perfect-match infinite PSNR values excluded from finite means: restoration '+str(infinite['restored'])+', super-resolution '+str(infinite['super_resolution'])+'.',y)
 paragraph('These are overall batch averages across the products and selected models. Differences in model, content, dimensions and hardware affect comparisons. Sharpness, entropy and contrast describe image content and do not alone establish better quality. Images, sample registers and detailed detection appendices are omitted.',y)
 c.showPage();c.save();temp.replace(path)


# ============================================================
# Integrated automatic_quality.py
# ============================================================
"""Offline blind-image scoring, isolated from the application's OpenCV install."""
from pathlib import Path
import io,json,math,subprocess,sys

# Embedded OpenCV BRISQUE scoring data; no external quality files are needed.

# Data source: https://github.com/opencv/opencv_contrib/tree/4.x/modules/quality/samples

_OPENCV_BRISQUE_LICENSE = '\n                                 Apache License\n                           Version 2.0, January 2004\n                        http://www.apache.org/licenses/\n\n   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION\n\n   1. Definitions.\n\n      "License" shall mean the terms and conditions for use, reproduction,\n      and distribution as defined by Sections 1 through 9 of this document.\n\n      "Licensor" shall mean the copyright owner or entity authorized by\n      the copyright owner that is granting the License.\n\n      "Legal Entity" shall mean the union of the acting entity and all\n      other entities that control, are controlled by, or are under common\n      control with that entity. For the purposes of this definition,\n      "control" means (i) the power, direct or indirect, to cause the\n      direction or management of such entity, whether by contract or\n      otherwise, or (ii) ownership of fifty percent (50%) or more of the\n      outstanding shares, or (iii) beneficial ownership of such entity.\n\n      "You" (or "Your") shall mean an individual or Legal Entity\n      exercising permissions granted by this License.\n\n      "Source" form shall mean the preferred form for making modifications,\n      including but not limited to software source code, documentation\n      source, and configuration files.\n\n      "Object" form shall mean any form resulting from mechanical\n      transformation or translation of a Source form, including but\n      not limited to compiled object code, generated documentation,\n      and conversions to other media types.\n\n      "Work" shall mean the work of authorship, whether in Source or\n      Object form, made available under the License, as indicated by a\n      copyright notice that is included in or attached to the work\n      (an example is provided in the Appendix below).\n\n      "Derivative Works" shall mean any work, whether in Source or Object\n      form, that is based on (or derived from) the Work and for which the\n      editorial revisions, annotations, elaborations, or other modifications\n      represent, as a whole, an original work of authorship. For the purposes\n      of this License, Derivative Works shall not include works that remain\n      separable from, or merely link (or bind by name) to the interfaces of,\n      the Work and Derivative Works thereof.\n\n      "Contribution" shall mean any work of authorship, including\n      the original version of the Work and any modifications or additions\n      to that Work or Derivative Works thereof, that is intentionally\n      submitted to Licensor for inclusion in the Work by the copyright owner\n      or by an individual or Legal Entity authorized to submit on behalf of\n      the copyright owner. For the purposes of this definition, "submitted"\n      means any form of electronic, verbal, or written communication sent\n      to the Licensor or its representatives, including but not limited to\n      communication on electronic mailing lists, source code control systems,\n      and issue tracking systems that are managed by, or on behalf of, the\n      Licensor for the purpose of discussing and improving the Work, but\n      excluding communication that is conspicuously marked or otherwise\n      designated in writing by the copyright owner as "Not a Contribution."\n\n      "Contributor" shall mean Licensor and any individual or Legal Entity\n      on behalf of whom a Contribution has been received by Licensor and\n      subsequently incorporated within the Work.\n\n   2. Grant of Copyright License. Subject to the terms and conditions of\n      this License, each Contributor hereby grants to You a perpetual,\n      worldwide, non-exclusive, no-charge, royalty-free, irrevocable\n      copyright license to reproduce, prepare Derivative Works of,\n      publicly display, publicly perform, sublicense, and distribute the\n      Work and such Derivative Works in Source or Object form.\n\n   3. Grant of Patent License. Subject to the terms and conditions of\n      this License, each Contributor hereby grants to You a perpetual,\n      worldwide, non-exclusive, no-charge, royalty-free, irrevocable\n      (except as stated in this section) patent license to make, have made,\n      use, offer to sell, sell, import, and otherwise transfer the Work,\n      where such license applies only to those patent claims licensable\n      by such Contributor that are necessarily infringed by their\n      Contribution(s) alone or by combination of their Contribution(s)\n      with the Work to which such Contribution(s) was submitted. If You\n      institute patent litigation against any entity (including a\n      cross-claim or counterclaim in a lawsuit) alleging that the Work\n      or a Contribution incorporated within the Work constitutes direct\n      or contributory patent infringement, then any patent licenses\n      granted to You under this License for that Work shall terminate\n      as of the date such litigation is filed.\n\n   4. Redistribution. You may reproduce and distribute copies of the\n      Work or Derivative Works thereof in any medium, with or without\n      modifications, and in Source or Object form, provided that You\n      meet the following conditions:\n\n      (a) You must give any other recipients of the Work or\n          Derivative Works a copy of this License; and\n\n      (b) You must cause any modified files to carry prominent notices\n          stating that You changed the files; and\n\n      (c) You must retain, in the Source form of any Derivative Works\n          that You distribute, all copyright, patent, trademark, and\n          attribution notices from the Source form of the Work,\n          excluding those notices that do not pertain to any part of\n          the Derivative Works; and\n\n      (d) If the Work includes a "NOTICE" text file as part of its\n          distribution, then any Derivative Works that You distribute must\n          include a readable copy of the attribution notices contained\n          within such NOTICE file, excluding those notices that do not\n          pertain to any part of the Derivative Works, in at least one\n          of the following places: within a NOTICE text file distributed\n          as part of the Derivative Works; within the Source form or\n          documentation, if provided along with the Derivative Works; or,\n          within a display generated by the Derivative Works, if and\n          wherever such third-party notices normally appear. The contents\n          of the NOTICE file are for informational purposes only and\n          do not modify the License. You may add Your own attribution\n          notices within Derivative Works that You distribute, alongside\n          or as an addendum to the NOTICE text from the Work, provided\n          that such additional attribution notices cannot be construed\n          as modifying the License.\n\n      You may add Your own copyright statement to Your modifications and\n      may provide additional or different license terms and conditions\n      for use, reproduction, or distribution of Your modifications, or\n      for any such Derivative Works as a whole, provided Your use,\n      reproduction, and distribution of the Work otherwise complies with\n      the conditions stated in this License.\n\n   5. Submission of Contributions. Unless You explicitly state otherwise,\n      any Contribution intentionally submitted for inclusion in the Work\n      by You to the Licensor shall be under the terms and conditions of\n      this License, without any additional terms or conditions.\n      Notwithstanding the above, nothing herein shall supersede or modify\n      the terms of any separate license agreement you may have executed\n      with Licensor regarding such Contributions.\n\n   6. Trademarks. This License does not grant permission to use the trade\n      names, trademarks, service marks, or product names of the Licensor,\n      except as required for reasonable and customary use in describing the\n      origin of the Work and reproducing the content of the NOTICE file.\n\n   7. Disclaimer of Warranty. Unless required by applicable law or\n      agreed to in writing, Licensor provides the Work (and each\n      Contributor provides its Contributions) on an "AS IS" BASIS,\n      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or\n      implied, including, without limitation, any warranties or conditions\n      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A\n      PARTICULAR PURPOSE. You are solely responsible for determining the\n      appropriateness of using or redistributing the Work and assume any\n      risks associated with Your exercise of permissions under this License.\n\n   8. Limitation of Liability. In no event and under no legal theory,\n      whether in tort (including negligence), contract, or otherwise,\n      unless required by applicable law (such as deliberate and grossly\n      negligent acts) or agreed to in writing, shall any Contributor be\n      liable to You for damages, including any direct, indirect, special,\n      incidental, or consequential damages of any character arising as a\n      result of this License or out of the use or inability to use the\n      Work (including but not limited to damages for loss of goodwill,\n      work stoppage, computer failure or malfunction, or any and all\n      other commercial damages or losses), even if such Contributor\n      has been advised of the possibility of such damages.\n\n   9. Accepting Warranty or Additional Liability. While redistributing\n      the Work or Derivative Works thereof, You may choose to offer,\n      and charge a fee for, acceptance of support, warranty, indemnity,\n      or other liability obligations and/or rights consistent with this\n      License. However, in accepting such obligations, You may act only\n      on Your own behalf and on Your sole responsibility, not on behalf\n      of any other Contributor, and only if You agree to indemnify,\n      defend, and hold each Contributor harmless for any liability\n      incurred by, or claims asserted against, such Contributor by reason\n      of your accepting any such warranty or additional liability.\n\n   END OF TERMS AND CONDITIONS\n\n   APPENDIX: How to apply the Apache License to your work.\n\n      To apply the Apache License to your work, attach the following\n      boilerplate notice, with the fields enclosed by brackets "[]"\n      replaced with your own identifying information. (Don\'t include\n      the brackets!)  The text should be enclosed in the appropriate\n      comment syntax for the file format. We also recommend that a\n      file or class name and description of purpose be included on the\n      same "printed page" as the copyright notice for easier\n      identification within third-party archives.\n\n   Copyright [yyyy] [name of copyright owner]\n\n   Licensed under the Apache License, Version 2.0 (the "License");\n   you may not use this file except in compliance with the License.\n   You may obtain a copy of the License at\n\n       http://www.apache.org/licenses/LICENSE-2.0\n\n   Unless required by applicable law or agreed to in writing, software\n   distributed under the License is distributed on an "AS IS" BASIS,\n   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n   See the License for the specific language governing permissions and\n   limitations under the License.\n'

_BRISQUE_SVM_DATA = (
    'c-oA;Tgz-qa;|s3e?{N;&OOQVq3>*CU)f;xHU?vi5Tb2_b)vVnz!?8`$CFi+@m5uQJ+&l|)?BkjjT$2}G7j(Izxv<)oB!#5{8Rpw{=*;t_{ab7'
    '&;Ro0fBL(B`<s9KhrjvH|Mri6`VW8j!yo?7fBwrq{_Vg1=?|5Eu|ND@{-1yO^Pm3k-~Z45>2Lni|N0;Kv;X_gfBC0B|HJva_Rqhb|NI~S+yCMD'
    '@Bi1|{^LLX?VtY8f2H?->Yx9Z3V-Ln`_mtCD(x5l|G(J(|M?fQ{l(0G{MSGK<sbj%@BZ@l_J9B0uK54{;m`l_pa1?J{`sH&^!1TsAE|%%|M<|q'
    '{q0}==I{Rbzx)$FU_5@o-~8)8|LfoW!=L^zXFGrJU;gD^{`oKe`Zxdf=fC^cfBwsVe*KO={^5WB!yo&vJ$vuFc5dRo{=*;huew_+b+zXolwWP<'
    'Ty}c=!SloX(|*-H%bu+~|LXiTcAGi1eJ_^lw4Xl5=V$G*v#h>ytB=21f30CpsrPv=*2XBU>>PYHr8Zh#{yl59y?0B`KN!E}v^#E#tChD8+b^!W'
    'r~14ue_(ep(@1)?{(8#q-#mVmGVQz04=U%wJLO&VYP*!W>-zd|{WTZ&T%M1s*u&X9x3M1o-hQRrXIuUGBl54&%37m+-$wmaOHH}$`FeP}jI#I2'
    '_fplan``;}sNG(hrKDP)KcLv-NV&AvqbtA4nEgK6x7WRg-Ok9bpIU#7zM8Gyb^oAd`_7ij*HbUQR-bjY{#<AMSFxAV?Ik`x+xST>=f1vvZTmH5'
    'w%2oRv7GP8bMKP-`S2h94_(mnmp#(9*Q)ozGq<(J>oMCK%CoLjO8@?az1Z*R<Fy&~^DgJS7wf|oX-!`b)-Ky>w;$A>uXouy=-s~OwN#e<jFr|5'
    'msM{)E!#W(J@)=)%k#AscCp{T`}b$pubyTZdHKfK)fU4XLvFg*g3ezzooz|lb1nDTEZcO|_R$*m4^o|L+YbwVu&sQ%@BaFM7CQ3YIjwaU@Ul(5'
    '%KBUmc1LyD9q)Dbt1XY5#@MgrF{(X{n!on1Ju17MoqM_qIJJ^%k{52@I_yD5D=AI;tVN2~9oA}3rqp@wWA8dG0Jg6cZ0jf$3s-60b<}&aji3AL'
    'Vx4tr_Z<YRVH>5`3-#4*3%Hj1oX4+ZJKc893YYXq_H4SZV2f@RvU9nAU=Pcdc$wcpXRX$At?Mr7UWY}jApvVz0Bbo~($n629na8jzcA-zXT>Xm'
    '*~OAYx-sh4R<fJU%XZ3G{*!yQb!c(>6==FWEc;#GugNZJ-(rE#S7Enhnp;VJv1&nUw;;-<VZp*Sn%v_|78<j!aod!uwXX<gE7%sp-nO4OU(e#Q'
    'Rk_lv#Y3D0u{`!Tf8%XUXImb&0Gc0Q)8gg4!Naaz79lPN0brH1EciWFwQU`H%=UP?ZxFj{+ax<U42IpZt&qFzTA#!oKf&v^*s}JD-mvY5G|IX+'
    'ZKpvlHQ!fW?HuVljo@L`zIqwe@6le;uuWa>#dr@EX!rM`UiL-Pj?$^$%+?#lfAY?ahZ?-&HSK`w9;a=~+ign>WZBwXhmYD0M_bnYb#`W39JU*$'
    'U(EhvjS>%whS@Tsl6~{E)#`jDIGmAv+vys_`2A}R`|Rx77bb~9dSyTaloRy77j-|Isio!jXyM)dV%3Y;kLtm0A~{&L?e3{&0n#>8Y4>XOJS`!x'
    '461%V=`!pWrPp%cwJqBbHTg+pKfdfWqJ?R@X;~UJUk|4?3wrB*FDGK+#27)OCbO_W?0M;4Da{)`9>2m)#4)}?we7rW@2KAVaM?SyC!e!#343&Q'
    'y|se%VR?#eo;05aj9slK+t%HqQRb-bar!ZKvQJB3CuCaa^5AVh+7hYUzaO@3o^GG^mpba$s}H}J#U9JwzH@OdJG$+j*Ne$yYt7=6$7)OK?G=CL'
    'glD(3L^9_u)nY>1uUv*q&yLvMTCl9E?L&Li-CwG0pV5;qH+IP0BPby3JYBXRJXTlRhjuYvZg!^Gj~XcmKZT6(k$FF|LiUq|_UDHuMS%sgbJ*9f'
    '?PTZW)GWsmv(-IR+j(j$FwbCpSlF{bmFp{16Xb27rW<>5w`8q-R{-Q6dlKYs%3#Z^SV`fl+zKjter<)NM;`VHyxOww7?zLUdZAd>ZZ9c1jO}lY'
    'S?23m*+SZ^=(|j(Jxxn|TOCoTw!LJ*^j25o1@^%9^z^N>3R3@CTbB6@Ta)X?3rnf9)heTS?qWrc<t{0D)V6|0FR8xX=srv7M9lhF9wYU<gWQGB'
    '+R<w%o1ac6y1v%*Iok&@VSfb>OA#`uN%BZ#)r+0b&Cj|#^C=U*7xgIl#Jziymde<IkR;o-c(<_FgOF)y;2M5*TfSl0nq^u0-gUO)i}(KhsIA(t'
    'jHCD=XzL+cy_)W4IbbZTMj5KrAC|tp!qg<W<+jINF%}YSZIv3V>P)^imM;f;06f_8{IIQiD`V}CauB!dAh*xnvwX7kY|GE;`C6Y=!_^Yx<aQ!h'
    '93Q@z?0BlfQYCLm6mpPJqgY)jBzp4|(d(RcGy8@v%fk8`tps_sWr6m%Jgc+<%XY-wB({}pBeSrB-)tY{$n^Zh5_J2fqEFi@)*9p>jihQ*(@HHt'
    'w9dX*i=|e!!@i3cyDoLJ2P-@6?A_(sCG=YunJqXOGYi?`!1hhR@ibEQ^2w@=X;C;x%NI}fakibPSqWv9X?ba3D?PVyBcrXSK5C893fm6$Ew^)2'
    'EGw%wmYt6ltCw2vQA?t1C*Q7v<=La<WJ&-11@`Oh&93_S>@176{8Fm0LLVXB{>-_UCEj)`VZ+9@lT{(>Rs7hgm-^biHMboDrLM3ju|lAv?J>a?'
    'hpm^oqF8M&sM=XBJHpO{I;Q`^UWy&i-?gH>gRv+7dl!<`%9;9DVyOMRyrOt}(vRugEt*AE!uuMxB-6Ii-eDDz*bWk_^P^{RI|=qito~U+N^ZN*'
    '4h0F-R{q(cFq1^1?K%3}D`m~u>z0Jn@_?0YVQm+)#bN=s1QByt_OZr&IBa-Jvh6R+Va{h$|IX(v+t?PA)@>uwk|GB9;wm#y9|h}yvbir`OSb+k'
    '(!ZJ-RP?qE?RP|<t&Pg5`f<!6tgqmDZhFm4A2*0RmJ#x_m%OG*65+o%S;pD(YQZ<t8!3-=*dx%4Le+X7m)*8S#W^02>njCWxwYFZKF7Ny<um!Q'
    'k!QE1S@ZXS+^p7bkCePTqOFxXmhX6-TF0<MH7Lxv>|EZKw&2n>+be5Yo#~52S(m+JTRy(5_eWY?X@c()P(L#e%W^CUxq;gLz(Q>=vaI(Hfto~M'
    '&>{6|(F%GCy~B^xxvK528WmodLa=Dxrp{{Tez|{O-)l+hcQ4Xvj%KmBRZoSx1s<}j``L6dgC2!-JGZkvo|_xnIoE7Q_{A)OTGR_ZOBv2_EG-Xv'
    'RNbCd@$%8~fSl(FBbRNK(Y@%fjaO|yNpQA9&I+*-RU!6Kd&|<vv_)*?=f1aXX=1Ubvj1JpYI0Po0*5piEr^yH1TQNUtXxX{_LuGNS>>Ajv-Sw='
    'I7m?j!Vb5hW#0nNc0?I@Na59ZPdiv4Wzz@E<);LsFc`A?ZegiuS+w1gm*)1y%bYE})t)`*RueMVP>;Ugqco=(v9h!$tOl{$E<w-8axtqGd%uCM'
    ')yz?OGmykVQug=A$PPw}w&Z6wpS8o=cEI%5Z)X!}>j>7hmHKw_rDyF`9<BV5y^mbjZ~Z%hEGAQyL5_7lI_*6z>CW5QX|_4Tgx+fZ-In*)2qzW@'
    'i}zm(<##lWI4!@q$IUpb_OY+kemxEB=cME-1+lEWk1?b>-7MRiZ*2?;{~8P=lD(BBoxN=3pFMl&r&N2BDd^skG`C`h@RHu8MfT=}*DSQ`Yt$9Y'
    'R<$SpyBwmlnCI|*O1DHJM_s`|L#$TG8tl0bk3M!#68Cpm!B8+Wg2+B4Dt12D&+rXmN4EW07<@KgvWFChR(Q_5yy-K~eWV^V$*TQ23+?&lAj8t3'
    'x$e~}!=QY-F&}xvW98k=Q#X^cdmNL{<ZUNT)VQ%*EcP(ut}R=$*pq!GNqaIVt~re?lbb#JG1%!a_dPU&->H=m>|=|pww#97cdMO{qTRYd!50K;'
    'TeYKUrfC@<#o=vFaD;KZz2+nDNAhT_i7r{HWrnqQ57yp=Ww<%2TKcpYch{FaFH-s>$1F(%t*)@+#4);bXv<oB<!qfDH)*5=#o{PqSj3TKyqBea'
    'rD`Q&uRRBI=928d9Vy+4)?Tp{!`15$+N@*b;9*tX@GT`80~>o!sl39Vty(*pYIK&<_NQ99O~Fxnwn%l`qTDS>$#;F%Ejs3Uqm=C~J34K{`zx|n'
    '%d9-A(JaZKLw|=`cII2MvG3PbD;6!ujDQd)!nZ748aqoZpIKhtGJy^99yZFd09T}JU{dhPM$h}VAm8R=(SBwhn<+_Ml9Jn{=-o<!Z-+(@d(Nob'
    'p3P1DEl~Gn+pGCPvwbyrk<nYSLcm^=BQ7fKet-7Uo?oXIyyFk2-RG7D;>zVtL#<xkr`;UIJd6FdgXK`PLafV*?xYy?ju}3-lKmhnhV#unEPJ<C'
    'TKulJz44S=4i*S&$=2nN=3Jo#El3i}3G8R3&)cq4zr*g@gR6Z_-P6lsQaO&c@QCQFTJMruMO%tjQuYT{k=$0qM~Q2v;(XObOywDy<kv>DzfgGa'
    'QM0JhwOHMUJr)s`9jN1#>vpPROB_oNz#`Mu(k#{2Qs)pXTl;(<d>*aMul$3CYnO4k#g1v`kuU3NG0|3bI83#&-cBb)Z48F|9=9eFom9qe6A+$3'
    'Az}<E#2PK_9`!&hDYQJoQ!G1Ja`Aq3qitO5Tx>zLcse&#TFf*uQ!n0`D?NpnCfhEA#GRzy%5-&SIjb!C+I{CJxwXx+i)4ycpuLv$v&&9XwC3<}'
    'fuL9?Xx7Jjv2)v=LiZSA$0qHn2vRX={lA)#_B}KOr;K6jQ6<o4QK^|$vHVu^CZJgNlq^i%hQ*_(S;ZW{*0s!g$gq5f1B8mHdD+2T+g_X(BXSn9'
    '<!2Uw1c73kyYC(XLF~{kHOUdx=}y$_kDnuZ_1V5_3Hvn`nIx~YBw(&P)nR;;Cdqa5ymlfOXFVuvR|trt5&Y|p29vh(Xl405KIVZ^@zq8rBd++s'
    'ZcWJ=cz(Ml4-JnAd>Dcf>%&tHkU@QdB=Pgjav!_57G|lIuH0S*ScYbK*Lv2EjGpXq*1W$FO7=pmTIznLSoF5qscZ?O_AtMPd>O1|<yi1CeDA#b'
    'UXpXx+((XvA-t4+*VX}dEqkd;;24!kVg>8co`R)Z;{|-7!nJduI+ly~v+Ibabe08FGDQ!scD92+yydbLx81@_xTZ&DoGuM(yG8H`awJ<?oL-yr'
    'z28LteMq@uIWQegZxh<HT06?BsZgdZ!gMi|7Wb|0m5he1W;K2kxtQZFKcL#PvzKo3%Kjtu<cO<9218ZGH5La)_Kw@5^o*vdR-ta-J-$AuLFVu='
    '6zJ1Ea`eU7s_^yb$d_lE#va!!l1Y(1uJ)CJWIk)_q4_(ux@QC8^E9GcEa_{75z`)3e&PU0hApSCU*T0(CuZev*ux^sH$GYkZ-v2rq$H0sV=&%W'
    's|QHm#+;+bYOrMqYQ)!zg$FyK&%&&q6)t_>orbPUJFEomL}^@Fl!K63&k2^&QK-|F*8?^13}n>#?fZnWg}G>d6>sd>d$SC0$;jM}Nz3b_&1fsS'
    'SZ;KW#RM{VB|psVoez5oGa5^-Y5T7E_ki}hLvETCRy*ffQj}TSTG&<43AW{D>1D?amAA!}2uq%oAJA#57t2DX&0y+<I(m;>lu8s^jybpeu%_%|'
    '3zrn;D=7J_Y@3g*_{hO4Wu-3=+sC$W!4C((mt`?8rB_1{bf3NNC2cP|25OJqlHJmt41n8>khL@RFZp&SJLBMyXIO_YE0Y4Kr8C8qu{9~(%26w`'
    'GD}~>JI<Ee+1*sX8bi&L*N9pm!|vJg_%4V7E3RkpFmPHTL8oim<48jQf13t>wv5zEB6~>_tjX7Iq)4#<>GLt$jSO3-Aa}un*`0-ZqCjbA;|&Kf'
    'wKr-9Ia7cr-Lq9^6|t74K!mfjuTK3L!1j2!mNt=G3)cXIM(#3t+QZ7RATI}pZ25*q$w|ID%IyYstDtKg<vZ6KMDZ@U@E~&CK0!V6$2Gr~HB-&P'
    'EZtKi+wOhbnH8^M*bD9}dRA7XSt7VD;LlPS3*F6+x((c^=N)Ds_I~qn_$-rptETF`Sm&(j(G0j2iumT8xlWX-NwB2F9{aOj`He)iDz|Q**d&r@'
    'Z%^1CGzKdx7@6}5EG_b2GiXpcqI`nYUSOAlSxD{4vY+8Yn9fY%p6=T|3uHLx{G95hTlW6e_O!IR+}d+Y`XD`wJ_op<vqdmCWy(QWXsM(C$H9?T'
    '^6=TB{aygCK7s+dZA&YhEuL(w46W~K&C?6Mwu!hQR<MFozo+7SFcaDoqDSu|y%MVr2&|><wtX&S#OfI~yliUAPA#}a#SbCMk_?|Mp5ET>Nt$j9'
    '8sf2nJ&l&7ShTvlmRzKXY@y$Bsi<dUfuZaz#g8TUR)}qnTjOvJat@cZ#U3kmq~N2YP>0`(J^9i`31UxWkWgBaL}|jIktJi>wY)&*ALv`~TdCcC'
    '+_t1CkK{8Q`|s^Jl*nIW@4Y;IIcVH>fVIW3YH3}xthYU98kUMCwH<YT1#>ONFyAQ%<UMc;)jKtIF-zH2l8adt9D(RT71{AoM%7fOT96m0VebH;'
    '15#D7TBkP&Aa)?^>Bt}wMSD#>=t$byv9db9U@<hxC$OOHzrsSBg`c1-KXa3Q2Q+6ml3?Jq8;$Ia!30K>BMURTDL;b@#5S(rSSc6Vca!WO$}8P{'
    '0U0GrNw#p17}1Rx&lcGM4kx7*S`q-C-_90K9?hnI0zO~(bNDeH=ov}`r}5c)RR6DV!=z|gN|Ak4TTr#^C_ERtEySgX^%<MIx^ZAe^L|a+KU;f`'
    'HffcXz{;7Xb<9C{+r}0gEEksZ*OGEO2b)v_maUDsM+n$Sb_7&uJ)|**56dzg%D#W#mDq8<Qz<@NoS?#`3uK!UmR9)9jE`S?oIO9G$5lz!EE)kP'
    'j4E?p@OWY$E$dtm7TP^F$OIbjpXhk9%r>Pd;dNPEn(B7^F^TaCUK1#p&PMe#+Eyj|N#NR<Id}@mFm~9I(fes`E3=F>J-xLJ0dlBE{o73N64n6F'
    'ge~6;DTk#WksZi&UVu%2X$>f9^5cCct3%=@$b=-5w8Oty*5!^CYqEFL!dN{l&9ucMjf@5`Q=VTvwdZ$}tTVV?t>q#=*j9%>0rs_By@u1aGXSdH'
    'E0_gEU~fSR@zPrka^r5_+RJQrEoIwZmXZis8}!pm5gx;WgXM{@A84V*R(Q`%Km%ZrSBrAvZL1@d?Q<84J9#OXCEJZyZ53_B&Ah@pHFTIQg{WKl'
    ')pad9pd;F@z5CXEH~>_$lF#_!f?6}55A-VqnO1sxrx%EleRoohTw_-E9j$3j9ipP7AdrRjmP5xX*F1uT^@J;t%4@|x`ewFCnjG)-@B~Y6O4t@S'
    'L_R-J4j?JrJpvA(?Y-j^gb|>pAXcVA2={X81nVHPwnL&=J)5*SWohD6j#qi$(xQpRL~$TB9}(JWV2UFcwh9~hyNlWnf^j(9&Ax|T&RgFc@MX<W'
    'c3eoY>Z@C8nGR2;N}@Qe^qzTztmo4<F6)Oe3>f0w_Z$)C#!4MKV!8q<EIy{;Bb>vA2^ewZG3iLs6QjJVLnxwFqe82W*|_@v)~XO&_X{%q*BM^S'
    'xzPX^(=mGqI7{!^Wwd3PqK&)~tY62yTY<M$N^hCLoTHZzvu97T*YN`zm%zx=tUKCG_J;eDNcDU?<KVyAJ6%1-lU8ODWeH{vR@O*G;Fg<$bOy9X'
    'E9b_eWmF%SJv0$+Z_mQsDTCJtI!1uz$Ai-KXt7!LLFOc&1IPfQ<==L!ZB-u-U-3RpDYWmt7Z-R&KrOG(%)sVMAbfY_io`+)=ddi_GAtp`Hp`eS'
    '!@V1VDs$ByV)iUF+S2g^3zX3A0;*qt>&gfc5DPw*6XhmAN;1b<%DyMeI2^C<h9n@WX7`z;+msiZUgquOCjqp91}k_?sqZ>O&C=x&Sh6?Ei1;X^'
    '<arBguN-jG6!h#$CnX7n0~RYl5O^95HzYX;AwQZRabM43O|tjs_<cJtKvPQdw^>O!b09Kdr6EY6adDU+lP`f1G%ba-BbMr(wqNoJ>pwd$YPlb6'
    '@ULI^6CNh<vyGs@&rHC3^^AfJX}Zkj(?Mg<7^Gi~W{UU7>#xIVehpF2s1OX|_hgiBnUyL%zhOXfQ})q<ypaPUA;Zmwn8cHsKMn#ZPCQn!m$$2Q'
    '#O<E9HdBGk21IR3@)-)W=6&k184{N)^ZkC%Y|$>Xl+{7Dr>C84S86tqPckxtGtKI$om9o<(We&9Gnja*DWY#?(ry1uD2rCiTW##!qw!2+qT_+~'
    'X>S5@VN|@eP=v&sf}RZ%1hAjYpHBxW{1R9>xo@^qWbOdrY)>+GR~xcFU}k+2PVjnw*lV=??L3>d!9CTe%n$1A!yr1d@=Jf<vq@FRhf)&+H-WlO'
    'c~K4caKzFa7Ad4BPMofGIE(NRBk<ZsD2xF+ba(H0+CsE?t$G)t)5U!9$Uav#{2jY*J&TbI>Ow@cCv8ba7%+38<X)Ev!-Q_?q8U&M_KH>~!2Ny?'
    'C!%k$<s|bQL>L%U2A_p=WRKIw9uM+LzmQhmx&5jVm5R2s@)({|Tkv#!hSO86aZb&g<47GiTgU#rRG*hKm9nqiH9}-{dsJBq0q}pevYoL+gRtgr'
    'bODkZcK%K2iqB$&;}UX)1tY5k!bHzP6FHz@RzZw3)lC2q7^DGAvPc=yl-%8Z@|;d{fcU+4)O-aPSRDY)_iVV^{e7?)A1K8V=q@t_O9?DwOl6eR'
    '!>x>R>UCy=t%$e-#j<E)P_&MKiA$cJ5WiB)%t@pCVsnk!XDIei0QIDL2V;p|Layy5=$0qv9<wBvbTv;HX!C(=TF%V1s(kDmq#q=`_B~~J7F_7L'
    '7p)>zESvCXVo+bKdzZEfXR1puCEXIK)FSx3RrHy#=+W7pfsm--G##cxu%8+OJ=JLGXv?%$0I`3!hiS2^2Y?DYJm%85Dde}}N<;`nSPC{)B-~`1'
    '$1<#vTrCc!PjyC)9g_oIaCAOFjM`lqAeO9H8Gbw6rP}c~@(8lT+K~vTastm?L_UczrPXlHap-5TKD1Vsv#@V&4d#RlaP0J}sRYtoj4OAl9`+s@'
    'VNx7axW*-b`jErepO2ZPlWB=ayP^d9LDG4#FF)rui{wv~c^=_43Ao8NSb$>P2PagjG-+&MF?y?_%#%ZU*Xc{aE_K|CRd0oUos0_Xqb-mysR$^Q'
    'zV;IM`BCO`pdhnj(wqqEXZh<Ba8;k|QhU%-qv$;OPm~PMtZq{@#q7A^!17#b7s$OcnoH8!FVc;q7$~l{vl+i5^^*g7Mgu<UZ5m!2D>F6eZCX0n'
    'e}LfU)7o>zfovyo70eCy1z04)fk3b?eiByt7`f->EhB(LI~e3L4%}O)ypP7`y82b@G5VD`l$uBfU{Ee9^U<p7OBiu=T*0DG5jJGgMi&7PLacWa'
    '_}pwMI#%Z#wu+U?YYFmP0wI13%p-txJ^F=$!=%?OA|K2%5~)=v(8rF*Hx;0yhckQ|0(>$U;j)zMp%(8}gAJzQ**8}216fS#8|_DTv|frLC~u4U'
    'U05X$!$z)w-dqJ2du#uC=G{F_fsfce-4WC#^N<mial*}}TAUuJEfFethR_0-<HZv@q;;TZDzDVah5RH?e+BAchgwb2!Z>if-Gtgfd7pL=s5hml'
    'jGZTN#4Q+~eU9s}jcNHMsqqaSR?b(j_A=Y(ZAA(<>o9JTGZWA{q50~<0PtYuYh?KeK(hkFwdvqlP9L{Cp#|RceoMIaP6(9>z|In^mPIzxqMu!='
    'L`gNz(E@9hIS=?e$$L6dMEiSfF{~tG7+G40WTRQ86L1hqjaqsz+0eFoJsf(z$%!$OuD8|CH5@7yBsWB=z_xU~I$YpMvIY1@p#7p*B0T0Qc}JcT'
    'rk`AF*0TtrqMr6=AKpAINQvn*7{q={KdkDC44?wQC)M`$x^<#1R!3AhRnmjy!v~sDwvIgl7NFM=CKC~~VGm>%LG6z!QlT7cwu<&x;gsHz|E?3J'
    'rylPe=2#-QBUIF8F}2ZUStt6`W&6`V^sXSQhL;UmMz&r9o~z{$q}5et#HUBdYqshMgVn|fjgQ}|CQNcB=IU>0PODW8b}-JV)EcUiLTMEo_E`OB'
    'jS8NVHg}4sbgM7;NXbd48$5Z=3PmpwV&?EVYAueupcAptKJcVq***?S&7=IruxHE6B*XRbHio4x_M%q{Q6OWYNa0(^7F(-6ru><>$XtFvmJ-}X'
    'JCjAP%q>T>+A8ngmW$vv?CCd2(g$tSwWA=?m@_vSnmqb(uZU#dYya#WNbOpLwVI=9b#1{4dXg`%Gj|KtH44?!B2E*;9ypd>^kSA7mzF~cO|zie'
    'MC4{#z$nRk?zWliXM6*Gkp+8<YrI%YfIRo~R>ZcTIigfx@TiLKc}si7lhgT(C=A|92wZ6haJ20G$aR=8%`>6O0$MHZgaSbxNd~qk)@8v5*a~Qo'
    '$TcMfF#S?R8DlJ~4yCDRKchZV+wZ0|q!BsK>3fwXQX<F~xAc89!hXv@fU1a?B4;5v=CE2ajZo2W6MpVHEk`i4>J)n+9GQY0!8Qd1A#9y>rY7ZL'
    '(wH)qNR8|4y#1zAWA|5D7nJupyqbuAyomEll8^}&Kw&WHXgl=~mW|+s!__)15F{=Hos^N9`Q8!_MbtKwE(zA4RMjT7{5CjLPeeqR-L+@eM2o$O'
    '(D6peDTGv6j9S$TG$yuWqze0oEBU}4e<`9a)K5fMvmnjmp}!mnx)kZSS}?VU#;EOPA@+)j;H?>Yw)*OlePNpb%=-jn9&o8h=OL_MmF7ypq&_3g'
    'o8>%TuyA@)sz^pCuKZR#6487Z*wdE`bgz|agkhZs-6sL=pJ078Zg|NSLskd^u_`0aD##TQmg;9)e-@tKr)oP~g5;WVb}^JndlMNobR$|wg&9<b'
    'S&#{K6R1-O@u34yDnU|jduNE0A4)peN=;GuvBgy`3;m-XdBSy?jzfjiCnNrX5_~j>K}a#7WGqQ4Io1RI`9$nZH0;Lrkl)hbX~DRQ3fLN_hb{PF'
    'ODpn=wEzB|3W{W@`?7`r`BY_lrs@c9Jf-q}%NnriVbT5kZL~&2%ub1!Mb6n0JbN?JuVeZY@OTenq=R_9K!0$fmU&jyd}(F398SG*ziH7UiPwqw'
    'sj#77=V%U-kPp&s9D%-W;eAf2Ki`)1S;R?%<Bx9ZQVIeRpR46CT4RH;Jhx6eQ@!d0E!npD-V+SC@{uLH@3_(xaB+Y}7n;^a*i!^E(;N|PnN*XO'
    '0_qV^Id_RM;kD+Se`XD^&U56ALURtr$jbUf=z8pb4{N~cHux*vbOz=<w`jaix;Z^2!&p|ot3RLlQ1)peo-mM2eJORmw~cJSlq}gWWp3+RW|IS*'
    '_8k#gY>C(c$S%u1DH@kvEFCAB;Ill9-h^xT@Plzw$w6NB&})E+F{1<zu=Y;!nB}*`MWo6Q0}zc%qV;w;k<FL~q*cU8fmQ~oMe_!$J4EM`8PN#T'
    'y=Yz(Cb=qEcVtXO$|YGiDl=*fZpJ&S-6H0U-jZ^N5o*mPX7EthN<DCB&_fu36FIy+R@4fqI0osc+I~aV>H;N7OCzGJzMyXGYCH#cT9b>T!*9;O'
    '?m{vQ8scq<lBp$HmOTlruR~IQWee{|<1tStNeHFe5_=7N$Ec`8%i5kb#g#~zZ2Q|?)iq|ZbEihhHRt?T!q|B%#c}&iqyjU5%N#=hycY!In2vJ7'
    'nDGq$W2QmtZwWOM80$=OV&x-q9SS~FaO$}$IUXmRBuuTt1<1k4i2nOrGJ66JDI!dk_}jrbqK>!yCp1AyQ_GQ~l`0(WMOrtaW3_Y3OQsMHyQ7&M'
    'Lu@H>9y6H4pjv8`ig;JqUrtpf!Bzs<R{nuKw175kQMiM!AdMB|BrJgnTOoFYB-C=n?g#}5Q8Zp4A?vqi!0F1_e;Lk);qSe48Xh?x!efz!3RD>j'
    '&cjh(7UV#(HdjXi7Q1H+or9KxufEi2o=C{Ha)@w%6Pu<kY}{c=lceVG5L&28gzj_~Nq`Btp$d)|1-;&4ZJ<$UY>I4*CGntgr8fYhdHz7I=WyJ2'
    'Vtv%YlaN;FuI3$T4coS#+wp9W+%qzqzAa#yTR{>k<y1Ik%XvEuOh}csd7^tiTJ)Q9qJ$7hFdzVL655Ra*X!~h`vP|1US5AJ`axA5|AH<0)JgkF'
    'MJ8l&`OC~ZN^joN0JGY13YuP)N-&1;hI0nZ*pqAv@b;yJ1jZu|l&<t6BnaE$Vp9RkKZO}K?W2pEd^`gR`d7+*)CgvO2>*R){>}r3-$K{2z|yA>'
    '7|$d=QjSSTI1Skq8<yk{fqM9A=m1CI{){)b4&KrkhlpkmxV#z^j*QtYPw{p=B22@pC|g2+dsZyvKAF*pkwn76q+f-xahQcUh5loV#%mHiM%0in'
    '1@T-Yq1&3G5CpNHcA@?OR47|mF-OFy*QhHBkO@Mcr-;YLrf6j_sA0?Zx~A>~*OLV*MW;eR6B8AiSYiy%z*(<54NbHo9JXZ9q?%s})t?KD6(O&c'
    'K(`-@<aM=SiB^hw(CN6FwJZ?vC2%LqvzX169@4!klCn)*$WU#Pi|rArSu>Tl&6tZ&>>>CKA<9572-gU}OuK6p>!SJ_j?@I8J6FUdWU$M@x^ynH'
    'UR(Eh%%fhmatRqo@M8KM>^58ndzFjXTD9CQia(a1*XbM@Y!xgpT=oWI2*_kL!j)0%WgYAuFhx~ctwS0wQMYZK^Uc{Q&>iO--cyB;!2=7CsjyG3'
    'bJ#KPq%@Gm-&+0HRDDl&H(b$2+E<<<<<UwPR}c{9qlSF*b(VE<vmydUs(!IX6}3_XOf86;cbN~KXuVw(ty$o*>@0=DMl1e`EKJaTqe)HiVCF`{'
    'Mugnnob(Q{tK=aUYxSCReKPt=@wN2;9)kS$ezqNo&?%!%4C01K)Tb<I20Jk&=d)5eQcCYnd9rpT95JNZCA(GDgz>p*3j^1|?RJ@1p-Z|2fn5&~'
    'JXFdUz=`CpJCG{3ttEv4n;HThI}M)N^@KY5nZhL8ION4tf_AVRQf8KoAghFwafXdMGSsjfq2;D*TAIjJ^}86bhM^0-)-EpR2J#^}!fi7g$rnHQ'
    '$ze=9jv0@=na#n{%Ok>D+h^N{hyZgG8ACm!Mxv>V6%cJ@fmf5m?<wtTJ3(x*uE=c?-%>mIq|;=Pxwr%uC_H-Okva%n?&7lYpe4y$#V}%^eKtbf'
    'E2KeeF<!V6M2DTm65{1>(UT9nHA2-xC>rl&W5xE*4uY}6<9Kiy`O>1Bn6e;>(=niJLE42K6#|4}>40o!5|=4E1<F%^^>b0rnXwv{yJn7}Sa<s{'
    '{yR}ULF0Bz>3ri;GMiX!bs7)5MPUH1%BB*Q=-67SBmPg=F_aLrIGIsZ(Z+ZPlodfdK2TsGK`!A4Ax+iu_+3`*XGaZsW|k<mA&TyUE}A1vj`5D2'
    'oh6$5uL{6=i~B1qS5JFcGzY^TVN*fL<hyepUctxDQJ@n*#&)!&GQ*HrJjQ}mZBEUBnui-?$YmnKs3m)rYe9sHb-CeDfvK``Xz8=clPmwM)fP=y'
    '>BTTSxKPedEhvmr#L>oLv%P9TrvV(dP*Ni<BVrqEC#MzLd!wf-nma&sRu6H6wjQ76eNV*IW=xnwhPT<#-9ww-hJNDAOTZDRaSPu)!(-Svib5UQ'
    '@DOChn_A6--dIzQGW+9qj#!Z3oKnvsrv;{yN;E()A4-Tl9`SF3R)rvmUk#>D8-Za1ezti^5k_3Hf^YxM1e$V9x~30IEpRpQx+?=bQ4Z8>(TGXq'
    'LOMFFn000BPZ#Z|4(A$r2-udX-r4(b_R1Jmj306@a;Iw*^mtab9<E^8ks3fIB?=-|Q`fzeFoO5P)OcnYLwfR?zl2gNLjD5mLXT;rDaPh`g1HI&'
    'rXH@>MbLbtu6?W0`*6<g2hb~T+syUilyFYn-cN#5m2n&jrS|}~<>7nrQRlfyXF;81_L%r((Y_~b&7rL!UBVOo+xN4kg@YP^+#^PBbBJ<kp|I&Q'
    'NRfdjU!Cnozl^TV>w9tQ!^-L#?MWJ9+iu87xUf6AEopRA8?N_XYNh8idevIx)OfX8W;n&t7`R8kC?*akO5%Ds=kM+Vho;kadDH`63>PMLUHsu#'
    'g6%S}|H`P5(FNG<)}02OFkj!Qzl)mq4wDep8~W8U9h``k2?b#`bWZp8To;3({m?otPrMT*=`v7;`+fmXpI{g`7PGkWHN=uFyCdT`QAfyRe7GO|'
    'fKH{ujI&#cMlvYZwkhUopmmZ%P7gFD^ocnM2n5}wLW4c!u1(2~3VH0Whe4+{acv8yC2|l)Ww*>guPYJl$`)DVgbsi#F7|1T82VfmV7RN%0z#TT'
    '&^o4o{fg^-WY%cs)g{R=7^9gk_W?()8Ff%F1*8ir?Ik;x2|Pn`8MJ>_K-aL$qKlW<g0N5a=6`_f*BT{^(gnQ}*FIokTB-R&XCUJf7`~^3>EFJ*'
    'K4>TWtDG2?7q1c7a`i_`VTyhM5Rx>dU49JKN{g^k2oTh@ey{<3l)Xf*z%3VjkYiya1%c&DyEdI8(YOv9Wvux6U>`ogWenfT5EU*v_)?UjqEl#Z'
    'an>PD=7||yl7tsbc+1g_vg2b&PKlVaJwY*%V}_7x<h*neYo8Vpg2w^%IG?HjTq}_$4}<HSm*`fsPKnU2&dC6ly~8=uMoO^ks?N<>#w5Ryb1FkH'
    '+W;jB*{;;~<iE4V_5+8{IK{JQBEeJy!Z2ZsedqM(MFId5(XK{$;CPs0O=;&=i4Z_zp>tlANN<;e6_qlEZ%#F&GbZ2;!VFXirU&)SB2W&?Rpdpg'
    'f}UF9SQ$CK+6R^x)!`&Gh<<d~L}!Slj>$@pSKBc#-5D?go@n`w0P$L+6B}WktoVV0DR~pWqPfoOQQcUP^HPC?pG7wp1YugD{%cjr3Wc*e^a#se'
    'r;DswG)h~9G%U5vx3BWy_=4U{YuZ$U%~>zX?Z}Dekhb0v5<}Ui8ku=|5>z0TF38u%=H}CmVd2E&BOaH2?-p~oY%4p+nnw>0wMo(jtuhKBAlMU|'
    '=@t&0&?2J&;Hef=Iikc12`kra=)ZD**C=BW)1ueJh-U$OEUNyyhQqKyNgSSSB)ta)u#2G$$gre3e&>{!E!69<{B=2E601=%HmOlNwC>!)oV<~9'
    'yq&=yYkXnic#|VB%IB&sE-S_=9YZ>}n*w!=DD>F6quu5ANWc8aA1N<qQE*5#I2aLns`+gKg5Gx@za<kKVAXo2g5@QIwDuGY4UU@^l<}*<2j_Z='
    '^zBAcv&aV98v*GZ;GF=kbmy2ca686;LgYB$_%=h6iPKZl5hHxiy5DF*m&g7={bJ@Z#QHrS)b+etNyubET8hS#7*KC9VYK4fiY?_C_aBT!X=#Fo'
    '%u69RbvlWAqbPVRQ6pWpHjbQ<2>(wQBjQ%Y_&f++_Jcr%t8tSBeL7#VvzdtJ9QD|a(fJYtVM3e9$vId$_SPes9Sm0-16vYPiEPRAxFMu(&{Mkw'
    'uxp5bvkIwy_Cb9qoW^~`{bS$kx3S>yRYYq{%&RQrt<UMjXAZ*jB)R_NA@^XeCp=$NpV#*To2L&;ilJn2djw11D0aw<hmGGS(T<B+pxdI<XMcA{'
    '4uYy(p8?kMI9=J}5vs;9_wvTIZ4)@Pb;j!pId#eSY)_!8Ii#QbgJ~V(2RqwxQlg!0lr4=}=uJp?g)A&v_FYA=4?6e-r^cEGVaQVy)OT!|J!afQ'
    '#P5!5`YHsV=LPOD2ZziNEn6+SNFs3dEw~N}?A=2HNbp7Jj>@s_rMM7XS+v|QkM#p(g=D19Mh8%{KJOl}1P0Bpmu>sU8wSeNL?jGZ?>%LjRp9VF'
    'gXR~C?_jqiTwCl%QQS8P{T=3*0TT*WuqK2?(iJ-h)pVlz$??}vd_e0{e|G7gxT$vBE+Qaar-|wY?w^izz+`Z3X_5Xf%BFLO3OHHyF0vJm?)#29'
    '63Fcf&vYM0wmJmB-0^gcEOTiYGFcFr&|ctrioE}^km@@SiEq5*DwQc>5wpk_<Bzx>ou&Gf{cx;yf%`36ZByJg7Y@y1MG`iaAwZ%lKf|sQ3N}%q'
    '*?%20))1?R0ZGFZ3yx88W~db5{{Yswqn6zziWG{<TSwfnl`suKF>(oQ^Vnz?(MG_4Czl;4Nfh$m)P)LVD+HvW2=k)M7eZ)G%Vn?Ic!Ksa#?>Py'
    'k^w-PQF5L5i)s>@57J3>LXLD(1d@W8K9*E?*w<m7DCK$!d)720EU51#+GNOXjP5f=CfIlJeh$x(I~ANXzrh}kzqql_Ldf3{1V6o21(0T^*jP>B'
    'xE2L>WQTZURgp#aI^ND>-0vh?q$Y@x5GMS#8n%K@_Gyp%2nY^M+k1vm@3ua4Ez$OvDsqbvrTx(k6_N%s+2#EjIE|x;3SYKx`>epf?7h}u<O{ps'
    'PSN5GRSs=fq2K7|an4(;(%kVI=bJtR%K{9vq^KW}n73_7YRS~|qi0^9-Lj2hM6^aaU`lVf9JTl9u60Ka9tuh;KTz%F5_;RAGfl}<e<JJBF01y@'
    'F8=YyOr$B4ZcqBXA=(nykdNmq)^&tVSNd(;cvd6imMq0Bx~#rKkrFGdL`)f=%~qaHp<RKs6y6P-oiiy6X=@Xmfgw}H4m?V#8_RP5A4YZp_81XF'
    'Z11DOEmyO^+H&M$K<cqE_ncB_qkyO8_J~N99C~~$VEdj=P+5<|i1N!VEXYcr?Q{enJtwx*?aZaNXXjQtqb42sD_8WIubF8%bNsEwv=#A<5kBmG'
    'Od<s;RNi3M$Grl(_1=Qq4KIl~)q|DNNYV9&)4oGjXiI<Da^Co*#?M9Qb_sc#IRnY}eD>;fE5}BB?{ORF%!4vGiWzw%+wr#Ah3;Ylg2`s}UknPV'
    '<P}I3p&3&0mdP>lxU;{+b&ta!;;|7}JfGqt-O~^!-Qn(Y$3$;Fhdf`)I;YbT^)FtTWa8>uW%LrGO2%$_5&GmntEgEGkshdt30n;f84F(14t#-<'
    '%1}b2P8Nm$tp{v7X6nW1g-6#9uCRZ0CJ@}ShuIfp_P$CMK-@-=iXS{qj{LXz>^yx>`N{1aoqELQas*6Uwz6IbnP+D|A-&aqyQu9^<COC@f&rC~'
    '1YSaawU!nNfkM)hE+0*WCzQ85Dl*F{1495-A#w)8CEIc3+Cw4_M6&T3kiFkBzG8leq995cbdOn{ofTxGheliE$JgH&B>O<@wz>p&2cwokK|0jR'
    'Z^8me2-U{ae~SmHSQcWgB8_YN^?Qi5Pp0+LV|6E6Gc+sw2t9P=Bo5*~F~829J&vny3Lj;PZ3D)0ql5y+v>)p2njDW1g{vb33;<~EuvZW6T^KEA'
    ';Jv+O1g28cigN3TtXh>qw=jvwVy^M%@Ra}`YN5d*&ASk=?YsSk@vJ=(4#Y};cqfRvPUvx{7Cg)&pi=@JEf6e9!`W-1=k5tfL-TvgNFMT#zS4BL'
    '5Y!EaDpGZ92};y*I-Hv%lGLX{$gh&<VF$M!=0=|+0`%#{S`QUm5|=Jr@J7^cGdf!al5Cox9#=>aOtGc5o=lrBWhbkSE&=!v0ybTg`rBy?Ct~lc'
    '9{0^XQcQv%u>CSaq6qz}^I~p$DknR}nh2)QfU>Qa!^t~hLPfkt5xUjx-X<ECF@hO^y;16pD#HjM_34K#@=lFGfO-r5-acA)mT~TetzA;++(I$X'
    '1aCM;v;4FbNmyZS#R)ybmzllAO$*>{9s>%(c2qQvfgnfU<kr*hm-g(^qio18hFJgt#7en@L!Mna8m=S?p)0CYGg$2<DxR=g^QwNeAM=lM*XXoK'
    't!g1dH1Yw)4VG(MF)|qO?bw;n)V%8w+ONI2-h+l0ps2oNAk!FVXHloYpdiGQe2y7`Ow1(RL70&ZEs~Jqi-4|!c>A$Jx=lGoMXQ3|2*K^EkL(y{'
    'Cg%xLp&u^x;mh1nO_WAGXLl7LiFRtZ5T2eWowA@vQAvIo8voUGjm$JXd_i|X8{%)P8PuE)Q1qvjNq&@zvVSJ0k<xPLi>y$R3R2n|Q1Z*-D;?j<'
    '(4)99M;${)D!BX!Dt8{|n4E(`qQk${{5|76t+wftJyB*jkAXL^9n;1nAN2>(j0aVErQAPkiUCK~dczm_N^_EKJxXMKhaj3v_ER0iHwO4Hp<v=i'
    'UsH?D)maZm&LJTWy^tccI*QIEoZG`>MD$$nZ@4YDEamnO!U~80Vw5Qsa^<Xa2?Hp<*vs27m=;%)=B6#UIb5kEG;b}gMv^iJ`GK|^9-kW~oN9{Q'
    ')`0sNfGq*>^~d<KJT-jRMx^!r!9t=T!2KM_#12Ftz3vh8uy~rB_CmYFfX}5_85F23fF)vAJN4+TwEd8||J;q1NwigF5d$<!;i~%9p#D|j`T3cM'
    '^)8cpT=c5YY>xn1`W`ca5Wwm|u*ZjLo{AV&^xPu7qNBfq;IW_&N5+(BCucK)IjJ@BWu1jSwP1!HVtgL2*_=DJq#qa(F6lm_bh)Ms)X#J$tQRc?'
    'l$zcwIui6U85|CfXJO`L1=PBt?&l9se|1+yY!*A%uz&>xUM8t+)jl%<sV|a+h!v-NI4)!dm?FTC4p3UUlpUwH6iW{E>YJXcVf^>1zTg8L8@Nb}'
    'ZzE~4M027Jq#tGY^+-cPvIc#h+68Rca7^mbb6bbp7_%8EJ)jdwgD#i8<YdKW1HeaMG@4M1Up_;b_@fTdj2n5h-s8eAAJ|8^oh{p(Yi41d)6Cf;'
    '!j!<I-fA)aFt2O3rvlDy*RW{pmXx7A>~&t}HWVm+WV2Vqhu)aGOy9^%v%1n7EA!wrqfR|GD`g6l4rf05ydTm$shVamkg*8)qP0C;NKl${I10WI'
    'jBG+EFFM!hN~LlrEIi}H7>dZ4Y7O0E3hD$&Y_jZ-xF$cyk;6C{IVu`07`CihmOxEs|Ip%FC7cH*7(K&N0jvwQ*95F;w&EC}QV4z0EQWLVzS;8X'
    'LV&RJIE9o|d!=R7q~mj82R!iY*zKf{keIF5Ef-mdR&4Q~eDoCh+9~4qTI`v5t^GrCO_u>pj?!XJB+EL)1!|zt;+!|^?f54L0?4cnhHk<nVD>97'
    '4y%J6n_XbtZaYdWw(NWs-AH;^Ok-Bi3PH4NC1D|;29w-+mz)k|ps@pK-F9dx)eWPP9kWS#KfslcQ;pSgY~fUKC$Dz;`lCMrg1be!bYSvf-6XMH'
    'p)D|^0v+d{(BA|3<i4~;4J~$x(SU`qmY#B8U;6C@z>=6i@1u^Ui3BZ<6xpo9QR%mJbfoDRGm0>cy*AJ;(QtH}Ziqme=%ehsT~eWp<FFTv&n;HZ'
    'y@WjYHt{AP@fXI}aTwSos~BIhaJxla=79+80d|=}q50_zAqIcPHUygr;b?@)2Bp(1^1+@}#mMiepnEO(#bkejm4OMlquz(-4I~Yb(dx%pWd1w!'
    '+m9MwU8*gE-bRTb#1@G6be%}77?H8r!WStg6N^Q-Hsz7P#EYGb{qF6*s^5-3PG9P1-w$HFE13)aRS$<TKm(hK50f)jDH_v}Nr+}<374@Z(JE8h'
    'jv_>Ygt~c|zqOA!6%=-qqD9QGIkY-%r-f6h823G3cdC+vsOi`_H1B2PCNeA=2cDk%X&(#S7@i8|xHZ+gXV9v4mULD|pwDz%Qu&FpCa4-A58SG8'
    'SfRVx*Xo2#LsBaYQ~I1uMBk3eR_}oT*KyFDH-$MH%5pwEISR6yDGHm_!u7m8gfGFeGeu6Jd<Eu{Oecl44l>o=)AC_Xr8bNdrRMS0s8SPW6r^ev'
    'qk#lM_XiUhi?B|QM=7<=v_%&1>9)~BNh#Vp!&5#IAQ8VH*FgDsg~f=4ObpO2wi3XxCa&Pr>S_@L2Lf{KqK@*$K}6&<Gq0=PIAEOia!FeXnFS7}'
    '$?<joCOZ_lMQOM{hHTGvy+z11#Kdm5Bli|gRq<|~GVVL*%iR?d3wo@NXhM-udJsj0el)~=A15a1vqX$Wa4J|BiBJX;zviQJ40y3b55v!KDlGCv'
    '*<x(zg2;j6oucq_07_fcQybKCi6F=AEBfYn0Pf{dn0Y+NmCdRW?l9U)inXqh(Gsz=d=h)XI25~U`#XWV346Ds=~ZLT2#HT|HyL1TR_M}d8^|)o'
    '_DYFuH<D3zhw&|jjV1KJvt^wf<v5@4RBwh@xw6pFit>=raBNE=o>!<5MMLN$*bxS19Q8>x{2n8(EXg{_&n0MmbS4Xv_Nh)=U_V>=tup~c6f4v&'
    'e&)GA`W6|DT(EiCJB&ZD_~%8Q*td}Ox(>#9RxL%8L>$S*N}F@z;DCp_mf@WjzF|wI@ptvr&=DVM;?lN}pj_WNd?am1h~(|!-VmEqCd9TIe>7t@'
    'Bd2XmlJ205RYbHkS(#mTz-C<bQF~x-Y{s@`SAgzu;f)YOlkSDHG$7eoPLt&i8tB3rYEyRb+3Hkr^vd?s-alFeJp{+2j@VUsFoW<<VsCjmD?JKc'
    'g*pG0j@LulLD`HzXhw-V(%Qt9uM3-TSZdxckmU%c28ME|TvaN{sYa>C493}vi~eK4@I4ZIm-l!c^!N~t2LxJ63)zgg53TF1ZZ$IuEMYSq5270N'
    't?j>1Wmjp};6GEZ!qiDYBG+Mn@O|XchZNLSB6iU&s>~+V2m#lFiXaO@05m`9hytRst({Yk0$0OO-aE~UzgWdtcjq2rmrsYv7S>fW&&PnNx@F)2'
    'h1e<|k&zwZY{`Kg3qcm~kU8Q9<I++;K_N#_)wLTMIMXiC_R&`P?(bD>-9~r!mCae6)0+htl1?x8!jJwQ;?hgd$M_I$uVK*sfVZ=P!IwQp<jP!l'
    'c{ikPL?_5zuNBE5u*)hII}_GTj^_ghw^h;4F5gS2e31d(Ch2fz6h$4(R}P0aUYHA7evxD^F(#9e=9qD$5bA`@M^?}?a}H-h9&<I@<~bBquuwQf'
    'IER^{or+obW<QB4;6(zibo;Zmg#|@O=z5dBc32P`_mq430`9^<MqpC*3uxYtkh8HNONpeyadZ|LDypv1M-N5>mXkvdQL-m?<Zb>|QXwB!@@TV3'
    'FV*UJXUm^D<nUm=qisTxbX1WsuopTWt<^$aV1dUr*Gk{TVn4pV^JSvT?;ThlW`*O=Hos*{Q3pT;{{@6C2%J)=%R7SpUV1bTVHmRNOFOh2zo9ad'
    'H$@UfMGvv82YT5>oTH!N?(oS1z@0op<lh{4^DY5_WppZ}N(VSvbIAB2Gh10??*NH0BohhY1hp%GSlVIGiW4RYf}Gn!#f%x4tdQN(GC%2t5B8CG'
    '>1{i!s9$z8h=E02tMHlM)lzY+;P{&1AjXd9*+sP%HbI_3v*|6C04@hUWr>O?{*q8O5v9o|e@zjHb{$B^6h`earcX(joQpwv`rZy!f#;QS_pNjq'
    ')I(UaAc|(_IEvG^9o^D+IM`n|^R(ArfgDHK2Sd3e8^B%(O~C1fc<HG3N33%z*yr2qx(rUW`g`d;;8e*LgcLgTad3uyw;E}2$9iC4sB$8foljAm'
    'KFcqUIwGn@f`&G(?iM3yEGJ~{SHVb)9pw;q)7{Mmf$M=G(z4UefG1V;+ep`T%PNw897&JYo+w9dA-55V0E6{+94Y5mhlb=?aaAvx_kCPZo?gDJ'
    'i>0TdiO@VQKqK$4ePYYMwIVVVM>SCj0XxU{3Uq1PF=d_)Kv}(Zk|MA0FtEt+e6c><p>J_#`4*5vQ8BedzC>xj5%BDDtSx94d}d2?|6|*&0ePEx'
    'J{{FrL0}+T)@7b_JT@W$a`ql=E>Zbl+GDwkXEOs+mJwyukj|}U^PXZmgKl`${|@QBVrO7`kkX#BTpCx-w(BUMGNO)p^o3PC1k1J>INZCe*HJ_$'
    'kpm2cAY)OLA%LKDg>M+dpwkq}+5&YcLxwmsVoKZPx9C0lFP8T9R*N`+K=~ITDzw>F!2LdYO`X<K7pE(VHbJGH`Qqzmy*KubMS<v4?9tFolljQ%'
    '=QW3c5AVGq<{sU`X1x{?2fI~v?a?@7K{j=ZvPT4Y>gJptd#y);G?o~`=}O#$Fp|rAieg5BVp;Zg-b8!nW{?jAd?$py!BWcTQCknD5?wCg)!Xk!'
    '%F>&tAjIu0_W;)pDQ`=eWw~rdTEKyTR@0;&GWyo0%b5j{X*pV|(i4Q8xpt^5+i1<^>P7qkTc=z^g0vkYzbJUkBvjsWyk%SIt&8;q@%p=ZLUZ;o'
    'E?i=iM}@wl!Y;92jsOmpM<7Aduf!M9-RlBV&i8Vnr*Bb7gqrgbJAjEK?UgJ#G6nl7fd9#gp-b+JPdqTKDs_7j6uTM<d{L|00>F`@_CP;DUU#9+'
    '+mT&D$^>&mfQ{nJo;YRi#V8$T4`X$q)#`wi<KBk9pG?si<39arzV)1}8cwQf!dggDmjiDofpsvGYFg9VhcJxQ?5kCkzyfaB5f-Fl%kcss9S8N^'
    '>k?30Ezt8~fkvy(@(@M;H_tgT5P&ATM0L5nUnZ~h&5RX#k7&DgkNfY|X4LCSC|LztqAlCn&aV;Z=N+dvdss_)p0sFaRy)qZ@}?TV*lefRau|<S'
    'TAo65sphss<HI80+<k|F{xn|7#eyJ+_{QIayOh>yp*!O7Jdw9jk$kR>v*v_l(iU}YIoiB}XXoK+FUrAZS&tTdbBA77!~g*;gq6gS2&!OyiZ*@5'
    'Y6^#Z%~p6|1DI$Ne^y*&VlJR<+{~clCc}1q&hBX71Gj>v8qJ-K>|+&U<z~T|^wIvHJ+8LUU<_T=fNzUv@h2}gAF%mOJv1=@>YK6x%GTlk2_}PX'
    '^$PTV?NJ~X;aQgzmMjV1C3xZWs1N}piKJ=ix05kZ&P+0T8x+Yk?{86SCb72pr2V*$C#~Ra7=Yyv;0=*~me+gAz;|o*EbKH!=WZ)XBO1v`6LI#6'
    'yZU^x%PFFJLBT%44WC!PH9Ov&$+aKQwty^Rrv81zl%wk-JpoJ`T2fhtRMX2J<Esq<qF{?)@*!xw6|X<f%pn=?B`<l=%$NeEw<9L^y~36GL|We<'
    'k{Lo4#W34x-^UDVuhXFNbmcUxy8txc-?xW0PWu{~S16$pbM2dwN+mrSg)1ze7_Sep=xH!FyLg!}PK)TZ2g+8COPr^7mJE!!|2;$Vf^?SRod*L9'
    'l^xxznG(j$_n~<`y-Q1eiqFg~sJ7O^Hk(!0^ecWCE}op%US#ChcbcS2%<2$M9F|wMDHC{${yChP0}^p1`-eaQb&P7_4j9<6dmUiCC6wuEQL?*P'
    'EYS3AKY9d(OhKEW47EMS@Ji|npnZ0C;Y8JfO@4vaDL@<XM&dViZ7G6PP!*q8@5$2xl;paSI49&*BXw&l)aLeO%$OU{lVVbTAQ0f6Mz2_!gZM+P'
    'DJ)W2fDGXye?!=BQ%2;=0`Nu5J%RnO0#+<PY#DrVLXJfNlK3I5XFxxpvtl}X`L&fn`@#aqiVrViMXDx*=LRBP0CQk#oAczUN9h4af)(fjq_=q8'
    'n!tY#F&JsV$gFf7^3G`DuWsM;roEcav?dJLU`DFbRq0-2D`SSLOEGU5(OeX&JchO&u#tZnIznQKm{tvO?Z_9DJ43}&cKdB(p|vb{l_+dx9@rAd'
    '&v7uiQ}c1m%XSLXL5SDWja?q_=zYYV<BNR|bV*nw{;UBbS+sfKSzB6}B6Gyvr%=a{(zJB3rK*}H>uUME*Nm0sviK-kdUgzf1Yx<-=qtiYqu^3*'
    'c8CN#wp^l@+MC_V4xxLaFnygq>0iW}DITDpe^Uer1Am8&lN|o1i=uMgKaLm=?{mRl7AzG?%|(F76B=m`!!=W*nM9RxwKIX0mc;D!stvr03*fy{'
    'Ra`6~)+3y5JMB_mq0Lvu0^#byQJ_FHPd+$jgfLf*aDT8s;9(04l%|}kDZtY2J%?taho2&RFX#fbipcg)c1Y!jotM<5aa|i^!Y*!9Jg&S#5;kJ`'
    'g46@?Q@oRni^f8#C766|3aDCiYKC?~b7Ux5-+hK^3df6}I58$|Dspvg-k8RO-ek7^i_Znmp)+x|m*Ur=ZH}nW?7cF?Fl!-Y!-P$8r?Y2r#I#34'
    'c&Ob1st&!RrLt!zPllaqG$^x66JR_zU`)2(f}AYpw^C%kS!IG8@l%?=f$3G~qhy<}<cm8w##~)sN=Hu&hEAvtfF*Vn&B4c*ad~Y+v%Qa$<B|Om'
    'Oe*L!JYZLkulA8jESxnJe2;a83%7ET)Q<36blZ5-)K_a^9gUolUez4PWO%_PlKa9x)>g4Idl?#VPP@hmh@u_YqYiTROQf^Jq8vSX&wi{_ry;hY'
    'oDCY{V=Pi1tUU1Scf3HuymkyHB4>Lb$MrT9Dqe=@tf$wRDs-=}-^)2as;Ctve3OfO@wC&tL@hqF8)*+qtR}D{wlBCgt94T7m2<=<+Eya3?T;xj'
    'g=R*tI~tzDX-IvE;5&jDwI!ME^qxvc8D&W?w+ehAlZ17?j^uLLVIu`+k>Z0?D%&-sBzP?AP&#WR+7(O2vS6f&<|SZb*NB<}1uu%OTJv<SwF`#B'
    'M(?3c2t&8D1QT!NIGB3*1EjM9+WDClGC?0%T+bQitFIBlAtS7mv6ZoSK7ODw(qTkNe2yp)!A+tJkteiAV_Q_UmF$O<9Hu~21q@id?wK&|>J;tu'
    'jd05dBeoR~R$5EeN6S9VB7`8c{8m8v>iO9Fsc6kHL=p5b&${Itx{{1L*M9_AEV8#U1DLnQ4Qvn;vIE;Q6G>etK6DFK4__6`SS^*u!zNU8uo%mH'
    'uQ2dQ3~puEh((E2grZMt+j~bwaD@YZ)+VyL_LEap%e|k5jqT`J?>Va~?Av;3T~=$&^?}o@pTGF<>0Y7;sO21{Rc+#qi>YIcxN7Lo*z-`qy@qI3'
    '_S0j`T5^icWpLO%3n>>=X3k<lN}Z5fU@pzm*dS3wV3qc55`Lpa3VHxF?1)U#ty&f8Z7W3Eb>Iq0j_ghsq+B(y!>hU%<(rY+&8Tw3=(AnGb%kxl'
    'NI8iuFG{9)gxC;f%EHi?r|0}y*mQ9cZW<Bf$r4dI9_sl$08xqv&~d|yhUk{?ua;Dy&5r5h{stQ-=d<9V$qe4N_04efP<rzl^sujQ%t3`qw6LP~'
    '^#BxUimI-~H%7D3hC0}g3+q1W^JA>4b8!HZGaRl^a+czHRB16qU+%1i=U$l+`o|V^y`<w4LBPT!Az>ih&Pn~<z(%bF4I4ikVa2+OC#R+5{?9r2'
    '7d2)Pa0MvQAqg%4TT(TS+5;Hyalb5AMrGsVO6x?>QVjPa=pyOYuJ}LRY^&12E>4ZHs^Q>1qa)q!XIw|eFV0sv7V3Mh2yl&2LGB_(WsA}-wufSe'
    'LKVKFiH}gL2w63N&Z7v9j~x;^v80B;wW#ntY^lX+WR3^qI<ncwl!;qJ={kY}LcX7P*_Nme1B}dt7V~i>Ok0dt_P}SOguZGAU9d#$=FG*Njo2*j'
    '-RlQdA5lqRyL8~I$jTLZlQ~i~u*>2V>FzN|VfQJqC;|RSKQI|lA8~IGPT5U+Rt!I|j|nUV+EE4+wRu4G$e7CH5=>v|p?eHN;ohY>JsC-s%svB1'
    'DJeAl1H)HD!FwDXCOOUV+UX<aF90y^a@&-0JylfzDsJ@nsda|iL-W370~u11_#7gi9cb~M{xZY~St6T6H7hZREfFNDB$CK0NpC+vc)I4D2lxgd'
    'DsU%4l#s5k2n~Xnv<gsQ_85*uhJo`Gm1Csw!@@q)bJ`Yb#Y}1|@5EItC!`+)sD-WQD50=QPKO97WMDiwlhjrAoNIkP>@S2W=#Vm%(=srz61--i'
    '!2cePY^1En23-Zg^1o6Yn6gpX8Z*Vzi+&u5+l7bBWA5(3pZqb85ysrI%h=)o^B%9knr!BNkyHrI<#`D~22sY<HQS!Cz9b=coj2o&5>f}=Mdo5V'
    '*C=due4bL)v8Mf8{&E-W(TTw|vVcJ$Le7N^xN;65uF$vGJvu1r!H0~i-kWGn`m#Ra6_%Bh2nsyG;&o9cW76ASZ7P~#NVrF)qen1n3Jwjap^lf_'
    '`0c7jTM3|fn&N4KfU86h>^*volaPj2-j>lD>Ju|(0S;I?+1K~E%JhflrR%URTGBF0N(JB6FbZHfaV<G$>eY$WpkIu*m}AMGBl^&VKve`NeWY_O'
    'Egzwiu083Q-GJ&40x<L#G1>P+k#`V_#b<9Ho|`{vd*3G#MiYtwCP_|(Y+qZF02u5!+d7Zj0vA8Hbn#WATb5+pfg<b5C4zg`CqC6>#I0Jd3M)QR'
    ';dRC~&FhT;ajYCJ$QD4=Y$fLP(XDBY^W=oV9w&8$FCP)UKsTKw_%gS9xY8|bf3{Gg!ROW-Pa6A@zsx-1h^&iLVn<nj1+Mf9I&Bs27#KiXT^<L@'
    '+au@Dl*C?Ce|2COY2nhDB%^^(mfH*o>s0Z{n>9xw<hHI4ZWf+e8081aj+}}JwhO@h<6`grh-lpuv6kjqG*D&oi0W=270r~$vN7$D*N%gDm-z%|'
    'Mp-!;1tASxG_NEuSJA9rI4#ErLOXlWZt-%_Qjoj^eQ2r6iqMC^(#T#!l6o@8C?a@mA(Zkedc{^GG_mYEI>~mIZAvSahf*TQ!1)i4XZ>tBReCi!'
    '3F66)kv2;Ry2h&AOOeAt163;0L&#C28R{%%2B2P>XUltVUy@l`<L**;ERtZ%YxRX+VjF^HA+F7oqZLBCy;QMSok!D2RXh<-Tq&}Lu4Gs~5~et}'
    'm@?Hz`1Dkw0XhSd^61PwDgLniTsC<=gi~6I%xq_`ZX{g*SXdN%%@`Pkbudd8iiqxWd#so5RGC-u3Yh)4Iz8owAACimBm<!9X<YS@#h!3uS+x5l'
    'Dq-qw6?OY2RFa5R=YUn<MrZ<y)O8zsrV=p{uujNV$+$vPonjb<pJ9OoTVvVP@x+tA%+l?aXADfWLV%h<KnW-r7jZSPmk)KmMj}CYJ;{0=<*1?i'
    'FKnl&Q6oiLfCU2O0w%umyJIT2!C6CMjRi-4p+^n9X6MEnDMa{H=Yh2^2DZI0RYEue$0=0M2g)@ZF@oqwb2<EGBg#+(o3eo)1wuc^RI#pnBhQ(<'
    'SZa3CtM=CffGhCE*_5$_X3)4g8yF>(zS;_Uuh~MDrM>d&@TPUvY{g`W$!Up(?nDa7Tl%SGdXsLOz3>-!;^SPxw5aL1glNsBiIf8XossI*VWPVH'
    '{9`I}Kp0iVDA>QO9|Uh;&ZQ=eDpU`}cgq%9zItaL&I^;ZGqg<XppE@-W-8xXiTcg6qF!D^NF;$}e=9nfpY@20I2zD(A+?of)8Zm|7QD3pUbUGc'
    '+Y^$8^IR5cT-D`+75OeU2tLSVc)8dHA>H+2m@Gz<&1}V3Z{m=FfcRKLB`Xk58$oon`f|!Zf`K-eCLL^6RMmI>K~}aG=jU3=v#Vzx)By+&hhRs`'
    '+RK$PeS*~y!_Q9GpiSDMI<cXv;RwThiofk}K5iOf!45P!CR%1YiwX&?+N70`g_HgCTmFjG1Kx$`o+AroN$?aiG<45%q{(lAxhLhd7S!QXxsWEz'
    '4u!7(M`L{LrZ^TsmcTnn)n*Q5_pq;6L$Za{Wf;gQQkmkW9Hbrz7g|T@!2Rh6q3Piy4EFaHYM)1KQznZAM_Up2u(G=ZpFOF@b}6__YqIC&SbS(P'
    'YnC3OSz&19e8~s7qc-$vXH=NZNUY7zvEo;t&IcEfyXc$(Ub(6RA2_3lx|kzQ4u$9@TF_Tu!G<R><LKOlyF;bBFMj!ee=Goo(uVXGuY<%n@ekop'
    'Y^o}>?k5cGD^7Q$fU0QY&^+Cuj*aZ;4kQ`tnOgvFabfjIRyT_<8F0qz(e%a6XS!(4jf3yzX(^4dbhPiHCO109iVl1mm>hSeJjUpuKq?T)EVZcR'
    'RjGN%|GAr2j|rpgU{SX@mS|fKHQ*JY!QMJ^ttlD;c-?x}g!7anz68cR88;U<;6OwK{rgcwAa$dS9VN2mVK6E!CX07<`nDXn*v}E2#2IUtEq)SX'
    'A7LFY8gcJiK@0N8DA<~vaxPW8xV8w3U}dm(j{qe2ex%x@ZW(ZORV)j}RGML&wqQwDUzF)BnaNekt)Fsg@&&8%RbCJ4!R-BV7C!oFHG;0JBh&th'
    'TadgD9#`Tty{;DW+boeoaZezU2j4W*$Ota?jzH<X(KZy%5k@a7ZI3DQH1t-)8MnAJlUsV?5j7BJ5(krl;?_v9)y%w5%##LNW?ycb9)pM(jMz?Z'
    '6RVlkFYIC;Rr*)x+7y^>NcH}$mI~$S^iI^4bKd@bi99_RbJbsGmUbmc8%1GVQ~5Cf1yYST2YDf$JKn?u*DC#LeWEW-fgO>DK-{U_90>5XTfG>T'
    '4g7WIP;74<@JG*riCPnb1g}S!;0Fm)!5{(!H1Yc`R?Q4GksIinhRA<713-~q@6$H?8P+;$09~pm=CO=w?oEI;5BUUN95C&#jowFsI6)wEY1MFy'
    'x$!s;G&VTL_>`81B(WGQ_p(G{<Dmf-q6`taPPZ`(F!?}7r2T?K`|9-@3V|jJkY9PQ>lo)bQGX1quOpNVFuvT^l@(40H`vv57Q@?JL)`5ETzm;)'
    '%p~?>;8oAAv;wjB(A}JXf}4||q1I8JdI2La*8~n4SRa*hNQ+nH22fuG=0_*Ev=&ZXdIk6>YAtw+wYuJh)y!Sycjp6K%upfI<NCIxfqR5v*P6fe'
    'h&gthX{+)%r(B&SN+2vIV^by4goohuN|Esmua=<nutY=>WVNkQQN6kW$b7T~1EzU9D7@xW!*el`jVc#Abg}R>?fGzDe7aK(BvLS`Pgl|Jc*zKQ'
    '>R_K!w{)FLLLMBuh7t$~hk|i~Eyl0)O;sP4z!<)9I9S<}oD-?9<jkRsv=Iq)l9?@t<T8V^LZ&De5j;~Jk+=^=5Tgl~j(YPg6hRn$4GX?_FbK*h'
    'CDbA-WG7Gj7C^G#j5y50ZE@OPDLv7;o2Of_zg~)HpMe^UHXqqE#9OKeIXvXF*N90p?WhT=gP?F%9@Y|8aE%^ii5TcCA8Mmf1%y=Wkssae9@Q_c'
    '+UD7!-f|iN*KhA2w-+GESbola@O4t<fQCC)#E&pRCk$96fPq$<Z&KUnU0q)aG!m;FBOwO<L+gga2`Sf-#nS?JlF(2v;VdzWr@?j5l&Ic!MW}G6'
    '8htZu<>ETI$Y|O<Hj!LERG7$M4``jG5h|dFv#MRr!!@Aq`~;SKSs&3DX<c|TXasCED?g)?6gWT^WZIk$&&-&|pnxbEPd`9q@WXzY*ij8z&GD2X'
    'xYkFY6C-31vUI5^lk5>-lQ6vS<BkUGb^!D@a9!aJT}zrJQ*)&Di;)c^(WOQ<X~4S>%2$SI7R{>3vAWyi_M;w9yW~E&ZR|jIm(9q+76+Y99+5>{'
    '+7?%ByM(#GA><Y@xx+9#CzrwmWZGvKLny5g1NJiof5JEU{)MPwFTv8ZWOyLpF9dC?Nt{mz4I3%RNVgjt-ctzid6O1<YLUqt@Pz(+$zLsyhTFFE'
    'e9>QAQ@9K8)EMK3$zq<RzxY2#r5BaFT19~lyhQoEk|a&NBedJ43)~p}MX&<kw3q$$^pPx0lq7AZmKyDehaz-R**t^@Ru@plBTvtf!6(B1ej@EN'
    '32s$#n$wMnSi~f4^tD%4gKh3o6xIf&Uf%@)kZevMfz2L=og}c?j2Ks1p*s!6qF3^o=~d>K5EvFZyfs8KoB{hSfWYi?u`Ad?dA02~*C<)EH;g#0'
    '79z+?H$SA$L^*V0wxEY5?|$={WlSnLO#*)yH}+82OQK<v8P6|i>{(Xdb|6)O)D$`(yNkMVwXYJ+5v{DtvH8d(Mg?cAMp~sj)u`%IA_=|w8hx0{'
    '_AF;|YJ%@4xyG-l1eA~kE!v`hWFdCLWjZ`_J`2WBW~X7oHP83d7l@IVAyHwX%|4=8=w#^OhPjQAE~!`-7UZcJ!)1+)+0H_3vM?}<x@_)#HCaAj'
    'p!O)bn>&#mY&!|>tc%$w256yU)1v)YQWfX}|FeZnHwVRGs9`w}3+=+l!fWe|7W>1(KE>+MF&6rvU8<C%#tXa6wnOiu1d^s^=_00z)T=kVGj4`C'
    'lF2~!Zu<}y{8UBmLu^n`EQ~ts^wE)Da4(IGNca$n#1T*2BXm6(P%pZ7<KDW=XIL^2(x}ly6QL?MLPXw%?w2SnK6sELkTDpRWH9md+2F85KTCJB'
    'p->m}*$W*+u!t9VWrq<?N7eO^fQjbivr&9^V<@YI1Q)O@;9`NW+<(WrkOVPstHL#`iVogpiC+eLw!26pm+zbtV~2@|z!KpEJbjv{zZ6@Sjr#t='
    '5R#o&MF6pCdPq~#>^(z_BbVLGJ(9XD;V>Qgn$z`NMF{v-KDIT4{n%*6rbtk6B#S#;0FhQD?Yd-T03}23J~Y(A;6*ydR)JXQAi6-dRMv{*U?*h#'
    ')QTV?(?dj3G@)4xEA(()?Efv63UpNF^|SxH+J~q|vA5VI@xi>HRndXz(EE4TL$STTcW$?}2d}r^NjXy}G=IxQG<_-f4``az;S;cB>h)|WOtB(I'
    'E*RDgyAGL~BJomWvWkfIEHyv}WF)eiw8b!Xd&6c}@Ff(5EOQS1lDr?kc-Bf?nkdMm2(`9gYpYTMQdv94jSoAc+FBxLY6nqmZF=d2eDoc{d{X~W'
    'iYgA$pycnG?m?QXbw-endZrWQzJ{5P1fzd;6McFQ#UD@bXhYa$`Aq=?CFT(f^tnf-$Wux>9KeUqQRZWU(<8(O6X{-6p{llY^NYvDX00R4qp_F}'
    'o{8mSb{N+to^3zn(NIwB1RUZ@ZTW#^0-}8nM(S?QPg?h=a1vfou``BWI$2o9Q@(|$9t%Z1{gg*%JX!Ini6*`Xy$+X^0OEq$hlE&a!xA*~5ebK)'
    'yu=WE;r%LvUCM~obLa<?Y+(sX7V`dK?^nUA7hH@(gg*92H9N+<Oms=bKBU1zj^KIYD9!p{-=Y9)a7Wl8QDsR|4|RYj=dx-_J(yR?b$~h=Q;|u*'
    'L@y@`Q5JPW4(lW*1pt!JI>U-Idsk9rY(xQ*B#M3T=}>I$_(paSCeB#r6~Sv_L}B^y3PI;~)aNo?5Y<w#arFtw6AMV|UPC2TET7ZuFy)PqhZx5$'
    '#5D4bM!tFPV;baX2K(ASTn<^i2KMWxKVHk8a^@0?yXJPN$kT`nkuD?^&0EbI6#7=f=$#N>39SZDH<wKC5ZSG|>S~NW5iD2{uPHH$@N(%8sk68V'
    '0}rhc*LZ?pJj6_~kwari7!sWhN+=we4zX4gqaSM1!|RT9*h+H*icMC8q7W-WWLUu!QQCYfH$UJ#$40BmYzkGcE`MN)2o-_|gF%DQg1bYo8u6yD'
    'BETYh$AQBer#~*1Yu|-!VZG!zh8Te+lF=wz`k*Lg0Pi`k;kv<4KuR4ogq~Vi;kkDK&$nt3SqbdR2<s7IZE#AdX8l0WQH1GFz}}%Q*AY2UpRvUr'
    '$ig=On1X9d=yr7UXNG?o9?h=rr^UQ(gzxo6PqSXlq2quD@N6bw7a+ja%5N1w3QC*y$*HqK`uCxe4$!b&1=2$b+|-v2DWn=O@Q)k@zet#<<0z;V'
    'x!Ms*2q!R|)<m1G)8Fweu6>J9UBO5_nnz^3_wLyMDtad5H6D@SuV^+w6-1KfviEVg1uW2#2pA+6{CQ8GC_P2!ckIXJb}WS*SY^#fosV)i*O%(;'
    'r*7mCfU-Xs;b)hfP*sO)baw2VZdV5po^dNJ5dX0ISr#@0QH?+St%8r}JN6zi?-s?t=9rD+Cy7w9hzdd+$kC}ht;BC{UM}PKA4xFFmIJRHBJ~bK'
    'u{eB^vTjSb#y~E`>eUmk_9lLkM~9~S^u=xqtHuZ+Y{hL_PPl+jKFL$HgE>P&`Jpz<F{Q&jfMpedMBABRyR${Tnln?J5-o7G9CZty9KqngaP{b_'
    'F$(p!WZ7_H#3^%UDySU=>v6*A@&P1u$9zMhuVUdma)j7c63!`zw>e>??JI9FpiT9fBEKUv%tn_mRR?gOH~Jb5BqbsmS9^V1aC(q#im;=x-{~c`'
    'J3_vq7O}qB2bd#cpq~&RX#7r6vV2D`_|eexmI)igxv=az9_4q)oS7ZaWLYwLSqOZkFsp9}k?kK##tBWe><|HhP{EWD&p14{xqoZoh1hfgb|K>-'
    'EZ)1QNZQ5lWk{hLC&G+62i?b?sEPTHef5ZJK&cpKQvrJmWvvNn7#*loT4)1`ypO=5H=KYa3UKx>cv3~o2DHhmGZL}$!HZ?x0~S}8x>{76TZz%i'
    'km4K|ieO+m&YtCxCqJ6<9MjTQJHEJOCG}oIIWug8oKe%-S`?!^SCh(KTf7XNPZJ_VI}4k9(L(6LeaT25_4->6(KEU_anLWe5hpJ#lt?H?^3N-`'
    'qh9K(dRy)XJ}bmEv47<8KshT-!r;SXw}&!dkYA?anrNcSgmLC0br@+?PGnSySi4%VMOs+YrW(q8IY=QeXtIP#OaQ^}JCBT+MzbPmM+NqgD!7%y'
    '5&FO{LB^t1GmqIqr!{Ei&JLt4*^W8ModxjocZuT1umi6x!6s!a;JX}Fh=`QHYOTRyXY!^Hdwzlh%xp{d1;YOtD3U@uT%8niSd7f7t7=c{1vCB7'
    'j3j4B%U5-QO}E5G$bK04Tj<w1Y@6QpC-_8A_aex#?8q4@G<j>F-MctiJv0*BKEt9Ql)$iiV{fnx;cwvu*QYvBY{8PeMG!0;`W7k_K8RzBkKe>K'
    'B^nmNv|FpD?bRX>*`yVpJxdE8EgD=9*7wWn>iLU%R|0KD3L*7ya)BSs<&I!Dn!#;pcXYMT$S>n&4Ln8+4$@I6@{SBctDHjM2~F6nJ~}vfrL5Y*'
    'SVN5<e`#0{K)&-C(E#l+Rjw0<$<clgAhXjftDy}ZHCr2PR(V=f4jm!r4sKDd-V>+Pq9d7k)EGnKA*eO|hHIQL=c55rUra^IL3WPaK(|<)uXn~Q'
    'ZG(w2Op2L`G}hF%*~)>kr6c-AdwgdqqAZ=V^wyDj9g>7h<R5k@;7XNHhA5+Sd*$$pM1-&(m+(NQ7og}8kr5OcYn5)!M94R?h{N2H+pJfC?7-fd'
    '04+N`s6Z0*h)7tTkVRFZ(OSQ}PGCDnd^iRR!^%@Fa(+XVrV$D^IS?K~(FWHXNl7tUlDY)yB5v$EE*H`N9Rg_vdAplRA2gr>C6EL?9v^9RUxy?d'
    'OKt^h?F^*cA0n06<@<2$DA=uOLQ_(WI?snd*4`xJ-*XTwn&_9e7+XyT;O%sS1uK)Cw~M`tf3OG@@k>(Cw*s~MK^H}EPNZ&%mIKRHp(qVw&VfQ`'
    'jZ{Y}@G(v4*#N1iZMB^SIi%o|g5(PjIuLFI?o^f23ZNR>Q!Z+UDq>^kv0tYohQajpLL*|@h1ne@Ptn#z&+i!bGBXIuMcfG1l6%DJ$<iUwY(Y%u'
    'IcWdJR%0nM#jvy%E5zIWtpF;LJ-BY9bk`w#lHJ7CZliwq(#!eu3(4q53r-et!Go(;pHef7N>_vtv~_sQBs@`iv@Pz`8A!^SJCxJ10NIM{TF&3C'
    '4e+;%y`uvVj(&uIm%>rU#Z@Ndc2wa;6XYON)_Dl&jWUEhOPaxbm$)*Yu$aXh4GHVz=<;^Zo}-1*s}E+EpzDwV=<`+PGt&S`yrXgeo#N_{?gt0g'
    'r%y1Q?N>a|i(f6&fpnROa)Y@M?^AGo2c;5peK6DB`Pu;^_qdK#p#UoF7eB2NzRhuMSx%`3;Jx=Mw5bdka!1k<4cAZSHG(*_&DoYiDaNr)4UN6i'
    'VY<Ic%k>~tuOStY1pp=_L7GV(oPBgspNe`Os!>2&aTA1geDkG5#B3DIzSs_3AOtj#fafeaZwU>T%X~Vh^h1shFxf{RckLv6L#pkU&;(I&^yhQ_'
    '^Z{ii?&SS`tQ_o+^cEQ-u^ifu7Mv~~KCXpDXpepNCD&T`XrvjbHlqk6sm6!+NBMO`=vNDcLpWG+(dJ2ZeC#o7%ShElWMzoE<!yqJH|oEFNQZ6o'
    'Bm;b#m|FtTgxQSVb8|Um+Q5w_djQ%Z^Q=t{h;lqd<Y(AkC^JfI;Rpk}5(MKoLZCuHE<o@{ijG99D9M-SPoqUR9W)4b$W<x`4XNlebB{Nl-2Ky%'
    'whCZOr?Me8N9R#|8&NFSc}GbA;vK2W)`4JcF>Z^hM29#Q<no=0ZHtslKdsv3znC&K&k;4N*Ws1-^=9a@r$nAkJnbWtAhmZ`9SicsE3c;~Z9gQO'
    'nGeLIWbk2!XT_TF_q_6>z@D=EY2v9?0g{V`I(W$v)rymHZ|yDnV&zQnD7_pwMInHlIY^sH3GBrl(VH#@(wR9LAk#o$h$B9!r9)J3&I3BOLFIHj'
    '=qNE#QzG5+DGY}$1Fzkp4$4jkjVPiHNQzIn+CzOI^xiqD6)IK)6>I9my>oW12&-!$r$3F^pJTa!hhM%ud%&n=wSXO-68fZBFzB#i?oa{@@jgSq'
    'PZ&A+`)$?-h3WCFpwL<qJ|E~6WI`e+2+W)7`;`HP+pQD<hTCdMdq?5A^A}U{bhK_95*%De@=;qGp=Vfy<R<C{wh;j%dkL~BNoGhb2OT)7h;Lb_'
    'f#WH};{t&wg0IsBtzE!Xxne|7rFUidcwHg293q<wvmk}I;qc46s~~_GP<W5?CTPGCJ6t@x*zw~kk)U3$*sn+e0sz{=zhqpheEor+TU(;_(X0Oi'
    'IzaL*kzONGj#tg~tS&H1`?_z?4!4tb&M%1v4nj*v^IksUwu7xTx8piitq~1({CX(V<_p~(4B8(pi}C_B<dEG-YkS(4yi^)KspZ@UzAh2(7(?r}'
    'p#FJyL>05|%e+3exG6=JPUFDbMaYYE%hE(4o7=-%PocaouZ)Jty#Pk&T(%^pgBxI4E<xwT(#hGYa-6W9c6mB7Wvtg{L}5ut%3ru)MPVzqArn7T'
    'Z+di?!=FTSIE#z<A`vcEUScutYN1yKoRfN`6&?Z#wvVO0eINi!5Na_EOsNT35!vCKb>1~m?CT-gy&Qg=2RWK>dK@RJ{$2uF=%#h6J07@)MRkw1'
    '_R`@LMkOklUb!s`+witnh9h+zgO(8`$Nd+y2+LRU58|gFU18a-5k(J609q3T0DH=|$jADv_9^mX=#(_4Djt~b6pw;^_&%{(Q^zQxD-70xRGw<j'
    'i&R^y0S3epF7}@E;SMcpiT3)DP>4rXU?r-HcmvdY7ffi&f_{J>fntM!!Tv$x;P6VA@$VGMRD+V+LS}6!rU;z$CsP3wen5DL)H--qizp&b*8w(#'
    'y|qg8n9{T1kX%XRn~>JdHTvQTRW2=og#)pklCWm7q8)zm5%hhL;}}H)Mw>#MF3jj^jmjUyKSvT{NK5IgtnR);Y!j|AL+y1e@TTZ99z4XA+yARf'
    'MGOJn#^s4OJhC+PaO8YWLqF4x0*9@MG6P!dYc*ld0lJzs5PBIi(`wpc>Sh`A7LYm9(1@YUJ0{Vuul&0q6@WIN*I{@y-a+lBheH4V*?Y4rN0ubX'
    'ax1xk>Lc^8|BW?YRzQRRu!ZM$Rm4|!)^l?=KvjeXqD!<9GSMm@yo5~K1=>GMo|*P}3^}OM1b3OTh=e@%C<^ej&%!E!eGU*{OYltOG=ewrOKrx9'
    'Ns{cmLN~iUnXa8#8`kd^<*9rh)IPzxV@G&0qyXHa(JCgoir+u$Cv7{C>($|wsUBHEN83gkPpCm1jCk{93}c1a{dSHM=~52uk$_WY65Hw^T=zKC'
    'vy*M?q>gSi3^w!fmYTEXiPXNI0GUTC&4@o%-Ak3X1ZbCHrJaByi6mLSzp_V74OwKc6SLx@w41^Lk!8d~pYG1baTwK!3E$qr>BcG*!A}W8Zd>>c'
    'VCV1^_HvI8y2PMZqf7K{l>`gAuTnt7f9DB|W-WN_H)`24LVD3nGN{-Uz-bzQq;}W&&ayj;y?3Hfy-&E6i#sSq{HaWoTV1ZDjN{@Mfn1(f0;VXM'
    'E>a+*=dl`Z+(Vp*xKc*2d2)fI{?R(DJecW*;30<JEeO-}cY`kWZ0mQy>oV_23h1yzS-zQ(GPu#}qV*5*B_RB@3fPCLK0{BcAD@w5Atmw&Cv*<6'
    'tjG&wFIlb)DC1#zTzPghm-9w+K&zh#GC9)tZ5DH=HnlM;GSEYEP5jS@7Pt0h+c7JuB)eGb_08>nVeO*D8$(y*v|hdLj~{mSsb4*@Rr@2)yQ8r~'
    'r+;wX9KsyLk~!`fQrVUTpxskjAiD#q?g-bIpZORJ_#v-L6+@F}cWt!uWspTHrf4=?2M2>_^&(TTXy?1gbgg3611VO9SWENMBnB5+Di8!k$iNO*'
    'tkV22WZ|iVDY(8)6a$Q1^)+{)HQ0kTXkEfoX$UM5*lnF#%Tp+E)M>5faH|J0@9Jxz6xoB`_LY%XEk?X$SmpUFPo9v<%=|QokN=&d)*NW=Nz%3_'
    'M0L=0ZH+anTMqp%U6MVQ=>mPB&6l2w$gG*1j-2uz34{5D7<3N?S~nXBFI<8W9-UTLek2`rcrN0+<tlg!@E|FgV0uVKk#Uy&c8=ygH54gJIZ|*?'
    'GE5L=G9E_SxP>r9mPsVIbJH&^!xVW7v*z0x@Oz+45B3qB5%2k)DqYJtYqR&XnPFCFYWC)jaxaz!MWRXi-zKWPrU<cbZv{8uopE81%uRyxXfgqJ'
    'MHVgldva6Lo@eNeTe8~!pWVqTS}PU;`tHu<iyZdw!WKT5&>Uv}WxdvBHaKo26Aqb+q)JpQrov$&-X?NOx6bugOEE?cKCpsBqj!l;vEQ((LYgYI'
    'w;K6E>_3ArCmS1-f>ox3lvWX{#p(8_tFz>;VEq|_C<RA4dnh)>2{v=rm{ZpDC7-!q!<8Hmx*aW0300Ge$^QTHeN{4QV`5!OP%m1a!HRn+!ou#B'
    'th^c&>nVJ=hi43nZGbW;Pn{h9p|D0rmO+qK(fX{K>eLt)yRc*HVk-=K>ZMi}iV<P-Y?p}EXzLBX%Kqx6$yenj8v)7KaH~?R*ej$2-H`(6BUyyc'
    'f0fs?xtQUW{u|lAsGOfbs=1<Z1jopnf^5$fq0|(*I52`$1DYWRMqTn3Pg<f`??tsimyZ%c>#d*6M+gOMW!Pc5^G%4v0Tx}aPUM`s&mrzXH0Ar$'
    '<Mzn`pNn{LzWKiF0Ed*?bFDutdDL_ereHUer`z9YMCF{6m7i6@m_1UshT-RGSjQe79w>5SSAylhut4jUl~%MmA$0S8)X<Gq-_bI*xhPR!&j^-E'
    'ZS8#4%+W>^Qiv%rvF?$)E`O|-?%*l!%Lu<HyH%Qe*rj_sw&Xjfco5=xW9wwg(%@jAyt+2;0327?-lHM!qNtwD(azU<IPw3C9N?5;iSL_M1$%q3'
    's+RNG?#me=k@HKmSFdiQF<9*BO_1y4TY-{{_I#|SrIM-MDd&Dz#f-}f(R9(q%-SeC|92EnkuE2-rrDeR3ibM4#_+0@ESt=pXxo(<L0D2Rab_sM'
    '=_R<eYDH+jR0XWr_PpgC5-)-FJQ~Tji6xU~Aw%JTYIsD0Yj4>POveN3tqfI1%<|$S41*y!CXFe0OIdV#BozGey2Ad^r(;vp%2tVFAy9whC4HI~'
    'r)NB8=^qiOib{0)<}dh`?X)MjI7yV+6UZsVx#}I#Y>^Chk!~W5tDR=;igEZy4-Zg@hHZBIAcQ`{2)n(r0$dmp_edehw{J&h?zd)P_mU5jWN?+R'
    'vi}3ZV&|1etgRuj53JnvQ{{iH2Ju;J9(zlYWN9_X&2(~+=(Or*#YpzVj9dtnw6S|h@#?br9p&zjZ`X^^oY5t<U36BNt_kEjm7$?S>^Bk-zLrv0'
    '_M*B9sZX4GdIvux@o0C8w(+dD7|G5a<942nNyN%k)GvP-V(e|0W$AhnT-aeJ`H+d}Y)EtNY}lVo-y_^B6Uv~N;(+{)XBGUNPEWpH$yJf;L_FOS'
    'n2qWjp`p&Rh|~MHAfp+v<q-Prvgb4kb;(?-a460*Y~OI6<@ET^+N7>f)y2cnrl6Z=w(sSnq~bxq-djfy`el!t6<wlmO4Q*~MABs3<QylLH#(*%'
    '`6!sySC?f)%kC457aRR8vw24{C(5OyN45QKv&E!d`*-g%x+dTR$+nF8FuR53Mb;WMWu2Kb>bp_6Ec)9D$AsJANpesjX;~i>`W_3Mh^Yfq(X!?{'
    'TlK%P{II%muCX_*+GinKvdl^I@kK$k?LCkvOnZ6cJKaw1ZCeafgf!Rz(!-2Z&g$#zY~3xEa=&0oZXtIzS3CxN2*NcZ625xvi2x`^c^{An%T}iq'
    'ez7s`pr0qTrxX-jR`}GRzA9)lt4m7+_l_4TT%0fP%S$kb<jiJ)#@Wj>+e|eq_v|dqQzDu163Fr!ru|&K{^7G2M@r>jyZ3Ee-uPVBopHXi{jJyD'
    '4en1RP6Z{rl_)Xdm*+LfmB631W^Tr@1)sT2q;?c313*{Kq_kXbRYC-cwOYt-Bxf8PNLxa@1#ntdFux{AEho|b67iMqkd=Jr%CIeJ7?BAmRR}g)'
    'ZsPxD&bG^pq`xc)B@_-U1%Yz&pvmmn>e0Q%v9*a3lA$|G<zDyxUxb$9z(3I`&z_vQg_O1=#FbZ5_|q^l^BCgDMIF?O+)gDY9=J$YUfG*VD)p@P'
    'X%)c&TCd1rEt-kiJKXlU5;#9tI|wJ;`iRvd-vB;eSZq^`M_rbT%||RS#KMBSl%gC&yT}Pd5iGhFGrv=i_ORTuoFDL!caq+G5hm4VJY>r(sHN|+'
    '&Y)b3Lk50mEy@h4`3xm=NH(Dpq3<jQKW@Pnnf^K5b9KCL;_tCVuty)ELxbVv_3i>s^D42u-430A%K6eFeaAucBrOOI9&|}UDJ=!=UtWP!fOF^B'
    'O1{Ru<d8QFKC-U7ys&ORW|;5%Z@Wwut}c<4XTV~SZDD65koyiU>~!Sd6L9r(9CD<R#x2oBEaN9=lFf+KeuPVelV_}RRV*D($)Mu|yC&;i(RI8A'
    'tP1gX`P8h)ul(_jeDd~4mAhU;Z=}<_(a^yAhLc#MskMEK^$H6rA|SNSg0cg~Qhz(#jlGz#vb(fz_WF!0QE>^?=+4pm<QgYAUU?pB5%s|2@k+7p'
    '064@QcoVEQ=BoKN2ZMae5=wsT4W%-a8AwBIr6y6r1V)*`oZcQ|{D${S=F%TSY30>PlPl&o`E*=M`c4NRWJGHhS^)gy$>S!Ix*GFx9}4<|;#x;!'
    'KOD@W4rY_j$J)I;(_J?5m6rf#<j1xBr+0NdAM|6pU){!m+*gF#BS*3;M{<#(w*-ggF0U}b%c_gnZ~PygBCb$Tdf8=%FF%5FKx>jDUe?bIM?n53'
    '9+k;M0wB0Gw0GRSxD&z?ilc~i<Hz-3gWA_L$ak%8?;=AoMlU_Yqjt{8D$3VEBybk}bKY#>qA4tCgsE#1i*-^^T_s#Hg*H)NGy+``24(j?RcBR`'
    '?BNl6auFxpEbs1K*}TN0`<Ewou{+>(zNIyFCm)2Y<R>VlKa45TBUe6bk6#m!N`uZD>EUy8;<&Ic`)7jaBp@QKXnBu|&fb9}L;Z|sr%KsHZ5<`?'
    'B~-F23t9rI;boaTKkN*#zqDQ6TT0T;9Ij=`|2(ca>l@<me3Q@7B=JPaUc(>PP#NZ?+$)^P#01@|anMfnQCkjEEeF7r>s*=u(5~K-^N8TGI5UyR'
    '(5%t!lYo?Q@73+H)G@>B4O;AwYqY|k7oy^_x5=9G(CUS@JZ0fpcpI~2y<qhL3c(d72Vq}3+bY)+)uK$O395aU(3{V7Kg6BI!<e71>OHhZB)Fo+'
    '&#r#<u)-7j0J9^;M+1^ZIJCwWCjGfJ09aY_X9wD&rqM23|1>`^PIZemguO=3ITAf#hmeG7m+`vj=iaO=HR|sqmHeg(h)Adhd`(rNvB^oAdqgr`'
    '@^<HY^XUpILe4(sIoVp99?hIGL!vF36l)UprJ@mCV2@9<M`p0j*RLT>_B+^ro@Lj^lf9xTJ<LR%_FC18%}#Xf;G<rL|4-aNX46&V5J`w%WN3|v'
    'Xa!AU1~AF}CC1B6B4w6@1rmpLcqz9uJbU7py`Vac4|QH6cQ4%5A+0W=MW0>}w~z7~V|X|%XQ*~J*28lWtsW2>tP7|@E<Gs+vOFiek4383h=BN<'
    'jMJ|Jh*<i!Y!&wo!>{MFe}vFXbMm8Os9y*LNGJD|XDG9*tx9Li#=BPgjD)O>CZ(q-SvF9d&IS#*L+P^y11n5ctG+Ta+^oJl9(!M>7+{FQqx)8A'
    '+526@e?U}J&v*Lu0`8Sy!H6bY)HbN|CMkmH{>AG-L4-vJx|J#K6TqKTgbLuLdJQ`YhRSQ7uf-WoWp3G3P#hew`+v$YFRWflc}Fxq(v9IEVc|T3'
    '1TYD9`Ofgd3uvWlIJF*dgPKgO8#x%-j5t6z5>2m+<wkKg_QS??<#IbUNZAoI!VW~c+!WblkM*q5tyZV6)?mO_-QSk(S-nW2P}G)_CU?o9(;dM_'
    'zq1g=>bvWxJ?$OYHCR9c!k%i=Znd_24t{IIhFt~G)X54M2E$TU!gkYWOah?ozX;gf&5MRE*{Q^{lp@|BtJG<+;+ApkwIyhEbhdX2@x(e72s4T='
    'J=d1<1=A2swQw>RdFD9D&!_rl&`f1LnofkRM(t5tntOJ)7^TKbnREwyR^j%&dhB7ePja;-k!KSp|DE6XNgZ+|dD`s__)r_!B`QKMqUw9>CZ0f-'
    'woE1Kqbk4l58}aEDS>?<VI1hv=5_~pi|KWvuZg0=zGY5G(u>{tg8Ut%#_WI%)II*To{6__{yGbRBGiuVlh|OwM<ikU7#wY0<UWMJV6STw9pr8n'
    'dyBiILh`t1%~hW&B~<Tuu0<_P1>OZU+r3jEAhsSbjX>QqXuWfRUR$|=dfh%j<flCh*tXZH=~S~|FKqSRly-8b9-Vu`ZH&E2?uTy}&*D6LE7w|{'
    '!H2!AQ$@Rbr%b6bY805}jM-!^PTs_}Ued$BdHK=oS5OkAg}Gzfj?*T<7%0(c-2cNMKz5`Zt~M41;&vl7+)gWXn0<5BK<%hz^U`{$C~AWJ(a7mS'
    'LQshiAC2J9%O~fjSy3`1R+Ds~o@!W1;J)u=3#OBGxv;UFe6x4t&<FLOK73!3pjr`t+n2#BIpWmCjj&1d0iza)+JseN)k2WffMmFm<SQG_ImlhF'
    'VjBf%9M0*wr+MXL(6z#Aq&g(RkpPM|Y8bi*uS@of$t8UE$(}H?qF_y5c7fKENR365x(J^?RIXo_Bcm|26R`iEN%lL1!em7A41BFZjzr&{AvN9y'
    '8!i$@K`bewKfj3wgE{b>krMxgr+r6<-|$t?MTbwf&WapxZ2yfz#}<i4m-d|<0N8H`TPA#I)W~we(2D0;C`u4O*H!f8XN5J|486hj)Lj@HMk7w8'
    'BHua_o;|~i+>YqwQtUyZ4qRJVNuISw%6bY>yDA=s<{Q)wPH!z?Q$m@^VLlf_u7h&zzE$XtaPz&5$8Zo<Zln?04Om5DwB`l1vp`4hHFLN?2ZX#-'
    'U;*+=H0RCP4B>;#j^Rp<9Qi;kChh*haI(T2Bdfz#mPPEuN3=BQ*DbFzpbNC>+Q0kvkn^0Y;8}q4=d!}=Ro;o-&WDq#qkklD7gKjDkt!rPdT1(='
    'sl$hp^N6Iy98IhepH&ldHq^qDbxC9Irajt0hC5u2GI}~k$g&F<3ePS2P~W^djNj-*Plq2e+6T!jtg%Lc6&NTVt+%Wca2&wgNgouQo=XZvcpg6>'
    '^oi<<0l^{_AC}B;PYE9LE+qOZthqFq3u7;9VVE)F^d=(US-OW7-pcn6pn{K=;jQZ=#rMh5a}K=rIY+|~;F!!rFEG#E`m)8=Z1?0K_~-YzMr+*X'
    '6s0BG_cKi5*kqougBh~@;m6k~C~nBNUCNq--!KP0P%DBtdvZEbDFoLg3$F*6R3AG`^kENKCG8s9#Xh;0<^OvE%|rMNWBqaA6*Q<2zhuea5iYOh'
    '@bkg88rY?&b;&i;p%;oL3#U}jgt0xv8zFrnJMg;2tY|MQ(5~nSv3qfjL$9z}2_Mb#4!p9fXv3pPiYw!JVeX>f@imGFl4y$+=co-V3s-tOoT*<g'
    'urWW(v*5V)kbCgFNl1zOf>r1rRz~Mha*oWt9uz&^zwu&qKi<D_Wj(g~G1XUg`yhU1Al!t~-rxC$c-fg+PkmhMp<N|nQgMb(IRPm7%Q+HY^i=Dl'
    'f3rmg|DM{107RFMjk-{l>^t1=3E&UL&mN9)0p|8GSwAM`WxGK7iY=XsuSl#OR@uOI2^sCy9KC&MXLCIEDw=dxQV$dZ6)mk@MpY*vv#cbi79Dqw'
    'Xin%wO8J1=TV0JuVw_@zF(ZnEsDH|=9t4`XJ{YN3=n{mxNc{IyTrz`1);otZx=U@ppA!j$$5ra9zOhCM>IeVV3@}0e$MckFzh&0G7l*uPo$P@h'
    'L9#C%a&)t@<$`8CN4=M#9n#p>BcDe{!m1)GyIt*Dn)NikNHmWk=c{<Bp8l9GtO1+C;RG<U;OL%F?_{q=9#w;D+5Da@v;(c`d@Gv1Itj3Qw|-~c'
    'hO1n+!*<W_8fSXp;oB<K0l<{55_ggkK_XcHEq&~th>BNNjEzCjz2~x9FharO_EpQg1nQphyy1o(6jRz7QOj5Dw3`=(HeZGCIJxMFz=_>cj8(8t'
    'Tc1os9oNbEcl;4TFvwUk(WKtmOO|z==da|*7Q<b<Z-DH;slfE6NbEM@&cEYO0$#~0lD`M(NxJA$PKkt7FPbbeShP~_^e=rO@exctX#E~p=BJx)'
    '`{5)04EVSZaTm@WPOY=YkV`s3a|GD^VLar4Eyk(p=wYjHpu2s+B3rQ1%lfbuKGYk;O4decbCPLWKOu_t$QDCt=onY?zpwz@btV2xtemZf4<?et'
    'eM9AO>FFA0y8|O=k6r#vj^X~(>xDRE_7tsgz*qDlF*?nM)<vnqVq*W?h#+&DGV#U{Ddq!@ba&QN)YA`KW?Nc#`#10NkM`=G?e>n&`MNxH{f-q8'
    'YM1MS=7h0NCF<Fbomz=m`=yj+_9r58?d(WuxCkB?W0ok<nc%i5^iuoEq1fecgbUPA=YWv2i|$CQ;uy0=@=)?KN|NcEM<=Ov0P`e9y$Ba#=OOjk'
    'mcMByboP30^j<?Fk@eC^`Ua9g?3I|B36=?2EF-QXirkFw1+CNH=?+1xqeBiA?PK7{q$<>ihb{JQrr3<43+lfz;0;e7U!IHrHd|P+MEY?Z2truD'
    'Fp66DH|EjoT|k&*4cAtqlW%(F<ov*Ur1sHE;r7TY7=nOg&Z`I+8JlJkYaR2oQD-=-B7X4rLY$z~^Od<itj84(>g_qKBA;QZ$nZ);z<@|*Byk|K'
    'HjywaDdoHbUGuh_$vL80dh!M}nKE<963Qt~bgOuK9ucf;5kTGdy%~ryYxkyQWrv`4hB!J*iW1X=)HL5OitT!1v6|xy0i@4W@F>K6d+pht(jp-)'
    '4BAB;$yLy(4~F(?|Glka(7r+kWdVfLm((!!KIJ90$o$trhJh~wsx8L|%FQeS_iT;Y2;y;ws+vqgum6YJZTGhX(iebs!~fQrfWUl?E<5a6LAsn!'
    'qfI>Z(=UFZgPqlWL!TjYFjsG~-UO^$b_>OSlWkSHp~m*lP?|V)+n(b*vHh$s?<$5{;e5{YxS1@akjF;e_K<`L^n7CoYQQtIcbsmHBq!g18!sH}'
    'pc6h52>H@}s}QnWkXd#K^%@KjWsT2<q>qsQYn4+SN93Hr9F;lbLbsv|J#7vloSxJ>MV>BxS~`)+A~8m#qS&Y+hhQ*OvJlH$Y#r~;321GKa|JTo'
    'MIq9x)JP+|lI>AI$=FnQ(t-K^6P4QRkVDcSlk&+(;I8U$1bVJ~q_!>$F5dxG9!k%gU7HIR0`{=9w$E`M3w4meIH@Dbj_N&{Dtpi5-pBk@)>RL<'
    ';VXzzt!xw5m=dw#vdLQS597uB%kkwiJpEVKJiuUe844i&D#56nwA`hNdITJ!OUevUWg$tkyCv4Dr=$9}&d^ST)I}j)VW3A6ht0v2g`E58_~M6='
    'Su-Ql@+8KrN_#k9Atr$iJ#R>k?qh%f{hgQo1npRzQ+2gM(CY{}9oFRHH<tOK^@e-Rlb(`ctmRcBqlg#r628vpa9j^0xwXtPf}v6N9GUTS(=NcH'
    'aNPT_xcs<nd)Ck{#%TK;n(;rZnXD>I<xWACQ)*KQ^XKj3TAG|{Wxy_%b@V6ADf{ZE&Mm3viG;l-N5F)wyJDHs0?jUj3OfVrA3AR7Lpf7|=-Fb='
    'L!PM)(aU7=!;1V<2$(LphX!m4_$efSfvQEat8u6F9fEH<eS8vWQY|X9c0M`Qm>K@zNw%!yVoZCvW)joH%Rd(B1TsSCGl85CBDdJak>&ci8iEB}'
    'lX|{o^dO#Ov?a&M>x$s)N{+IZqI^SEoU+Zv-@SsO+$({GCR5}^+wlmZk|H1_g#jiL`a`LrVzD~fzGT7aHhx~{NG2MiHgVP4!<DSalCcWN{v6hN'
    '_KybqyQS#rknq$wlISbL^SUliSh|lyT6TA&wZ#n0o!|HG5T+AW*LuEf%mimYZ_QSlf7oGh4Bb_BxrC8rP5ME@!q_vknV^Le1pgy+lIc(Ofk$xk'
    'u@OwLVFhqVbVU0IH(T+6?TRyGch{pb+{~E77szN})%NfGB3?hAUX*{Tt$jac)o_>Z#hy04`1{1w<;(ssr5td@(b|wx16F3r1f4lvTTyOQ?tWyq'
    '=VcD&*4Au~+=3d0dYD&LGsy@`KiNjv7xpuSC$^nwxHB#!hm5_xM7zy{Ir<;`Jz$boP79<KLZNi8{IhEyOl5ZA9w^b0ZaYSYOtFwG%oJK<+6Kwg'
    'PU%4MX759P7?<ygiffpwWy=ofQ|44%^08J(u7HM#746P0{1O)G`jaaiInQC&)Fet0GTWpYKu~PeL4+Y6NFOkXE}_@h|0T;#36L<Ly=aWf6Tfsm'
    'j*+5?-eKVP3Gc>?$4g1N_I=|qy_1mX)8}xbVkIpT7y%&-SjqJoi&J;iS+vTjy#Pl_GA?HLAV00Yd<4<j%r4Qs``sC#;@M5cU$e67m@7K3J9fHD'
    'l`T(PZ<ezV-<4f*C1@JK<7b(2(@vjA;s5boZl7b~dobzh^dbSO)`1q0$PS<@DXb+x!qy}wZ1NRK()<MHT7`+3Ol>bte?EN2E46431U}%Dx#W%n'
    '9RXnBf0|7*G_y4wvqkQ_#_M->?AgO8dub<TDA8iA{R1(6>YchohGKWoob?*7P{gP7i-={@+hAH$kq@KZ5dU{VY4R?azoI!Nkk^}CloKE+amO`c'
    'O?wSL=hB#DRf#-EdA~)1j;Uh27l!58iH?)?juD}^<&;FTQ&_W>P?9?BUzoNePD8ofx>5m<7m>z*TtDb-IbO+rstSV|-k<z42$~;;vT;hU&x9ui'
    'WFw1DTRz&4wV;p55=+Pu2O+IB1zcWygI!Y`K=Y~GgAS{pLOb*DC@WALa(<{m`7sd;4`q`{nf)m11Xpk?Vwv7o)U{l>w}!BcJ)m#S5PbvCsoq!%'
    '#1dlZTO1MgJFmz@KsSI#>u%i}HEO?Ko3FA~MC{@ls!de?Y;sHO>XmN{2S?;$bcE2n%Y7t*I#t06!pu4nI6)*@O$f<M7@%_AlY117iD1VFuMLO5'
    'ZI|fdZt}1F7gqUUy@|xX-A;^o<vt|LNjN-C2;#r$3YQvoSf<s^6xJgIRN#a{vc1KvUy{K#ak+N6Qpp{k;;VL_&4~2Q6UcLwuska^7>0=5wD}nm'
    'h)E`yVW(rHM}|vKkEt>7j`tBM?+9)nI97^iSV;ZI4k}45ZAZqb%!{*61H-PYrKFuyVwe*QGkJJ-%2$p+U`&w?mkdt^^bvmC=irhM%svl9*<mzK'
    'DzfoMoDfCu6*yXXZTI8ZX1GV%{%l@hVvLu;(wH+NPOIPcC37*iw4wkNs(Hx-_Ymy%K23<bT-lEDh!zNvqcxGDwy|O9^?Orcfer9guA`^e5>4lm'
    'g4B|P!Zy@MCFIdsHJVb3Q?8$P7`b*hPOt-joX*>@;bC4y?nHoHd+lIm)O{7O4ik1QkBU4VEE2Zb36%oGfx{sPlEm#v)$)EkFud}k%e|(${qDU)'
    'qQgypIz|=JJ(Gr(4{I_F=!xSuP@2jp&2`nBD1uo^9lmv=6!A^acc+Wi_Y!cLn=q;-6?JeiHAP>y75n5j?sI@m0^oaGci>|iO2CJsiBnFt?>G`?'
    '6lAjaGS%Xap7p}q@VihEgmkij@$9T&E0mtrZp}gw0}5{B5WQfH%aXu=XQuboXYiU9Us=UHBCLcVeM^HeUH&<%9-hoKyGo2&;b)0v#;TH5Mr2tc'
    'W_he7VanKIuic=yb7Ykgh2)_Oyf|1~p#^`7h{<8`rm%zup)$^K?u_x0k_=-r>ibkf)AdNC63d34Xx|fof&d4x#V^s-)58b4Kuu=IyiE5f@dh!K'
    '@!LEA6d8|Tue?oFfWG91A&e<eRvp$(Vh4uQzX%Ds7-RnO!MG*MYK5w<Yil;HH;nR$fa4MRyXd{g`=cI6nZWD_5+w1?)Di{{P?X7h_<^HdM=s>}'
    'o@1Liz4A%2a&41!4^4j9M@ZP`F=fSYnHoI6T40DglrXwp!z_=^G&ff{Es$Al_2K;y7vn*vj2y*vJ;)I*V(M$Io(xe+c0P*7(av~=GdG!n-Ju<A'
    'QCz}8`BmN?s+V7x)22<iUg=arEbDahvpgWy;w^F2G7Vbfmxt<kimjHG>Y(VDB$N?xQ?9YLDFHYT{U<7`J)|F5*fE3Fg^O#`y9-ZjTi=tDI)^~T'
    'rFi9CNlsm2bg)l*4}smFkZWs2PbU$z>mGFL(L|ru@gdwH?Ff(SZAW>!zcNa?yw&5DgyHvDK_n0<Sp$OYJ~{Wk`~0bzLRII$d!kGJ%>jPWDSS_f'
    '%Bn@!Uy6w=azVf7uQCBb`wUG2o><12QBEbF3-LIzF9)r(A~cX4YYAwh%sZ=RYTQ|~ql|NC@XGjrFSWo~=T$PoUCqCu1la62!>j=53~zjj3K*r5'
    '-4vm1fDpcuO8X$&S=GqRu`?8i5q#7HNL%NSWL#lyuFfT`o?+Dyoda+^`q8fBXXcWozzLFP6Gc+gihQ=noBS*mi_*8)0Y*%sVfA=cyac?|u2|P6'
    '-Yaq~S{F=>Ajn|)^SgBeNKB^4j6|J2<ZiNy7_RMV61rud!K@(^Z8N_V?JZo%cKaEIWrTwCnA(XP`2CNZcs}h>OG^g8+isnZ3}k=I=S-Tog2Lz;'
    '#TN*}_tEyPE^x-1#IRW>WG%XEP&|$?<hrtb+f!@<R?ODit0WqsqF#}00PQ2>)hYxvE{f}hX&ktK{%fSL=A_CWPnW$?f`_!~7H{47r(lnhzk^L_'
    'XJO~e*-;|f^mZPofCtw8ntoK8znY`9b>oq!P16@AAx-yR{iy%K*gCIRvLq;O5KdFyFapr$3fyq`jAVI3)+eELZ`(z75gsI(z!}C`FduI!Hye#)'
    'J9J3<;8yCR{Yw4pszqgcW^jGmF|)^&!to-_6e;M5k54C_4)+#O*8TSPj}Vn=JFpOY{bNPx!Qegt%B@K}mfN{k7%DnPwwqpxwhRL#?-VZ)!TD>I'
    'V6PL~0?D_t7Qd_@?})L{$^wq$tH%L!@3P+M+QYQUWFOG`<*@Go7Hw%7G{7%WLX{n6&y|!H+hywJiZ*f%!vHYA#$}2@XV`^1mSpH`chW>lL|l^+'
    'aOOYE3QIvYF_$bYMdlZ@eQAj;z{|BQgK7StJ=KBhrfw#WS^nbUtU7rWldtKc!8D4G{w7=V!!p7s2{R|WrNnA#4~@Tad*H-avELgLPC?d3J*w^I'
    'cEO0W;?<v}3##*PxJIqGm7ewsi)(~3=ktd_b(<pwB@hF6%Wz5%IyE`i9R(94TCr<DLpR0I<#QbAHZio6F`|b$Qx&Ckb^w)n>Vss+*mBk$I`AvD'
    '2ytYfr8!b+ZxG2w4&!&QbT#=(TeZ(;3AJSOxQr(3(a!rOgr)_+?3O;f#Smup3iE2My*v^ZSDLVoUZTvIfFXKNzvKTDao>K3tzBd*0C#2&K|mm$'
    '%UiPJ-6mVbd@&}C<c$EJ`1s@Z$+z&UXtSuibM8b@K1Z}|Aqrx5&9ETXW^@r(`z#{`#Vj4{yxzd&fVb6T+cFt!zlIV2jDJpb<g|2!$Vxu_6=Pao'
    '{-yU*4UJ7SzZJaEZ2f<+L*!MOdnskKLu-x>f!ZfJFZEL5Rk?NaOqnq7{D+uYg0Hqq&GQmt=HtGN;>_U+r<=g_FhoZYz{T)GKJ2A`;=7m#)oUt2'
    '+J&7kN*X{I)uzU{<ibHH33&u%t=|Z97ijVn&v~b&cA+RWVWfQrgI{uwM#h1}h}yBw5qd7gjBF=zqHf{>35H%W^ze4z3OFguNYM?3m+Z9e^4^I_'
    '?3d(+RT9T?Tz?P*a9y0wx^_2V2>j(K=e@O;Z$A9M*GrE=ehf`B=Dx}ix!O+B(eJKey4-#b`Dw+f|J7nX43Op_Cbsm-8j#v}LnP_}l0E0v?QM~$'
    '0Wxk&m!CGJ+59r@*`qY~$<AAz%6xvs=R;T}93Xi5AlPhK?rZ`B+Lp9fw|j?nWr$giOXRJd;AtIV8N{s|C5iqdE)vXznQm`!xLb|8cQmH%kHw%1'
    'sRe|Nq-qdk(POwDH7TZjp@Big=|%o2q=>9I>v7=96A`&nP{a@hEv)W1Omq@yeY`=BzAwp%MpEQP$q}eNn7ZTlx@TP$3H9^CfG*cKFm_Dr9%6_m'
    '7ouGizCYVJMN@=i-}Bx>#M@b|0{`ckeT<!VI5AEbdOm={BUh2(XU|V6E6f80i6A23hHYVs8@3bi($@}ei^VSfF7|=_t+SnE*#Y5dtlv(LnQFTi'
    'TK9XntGJ0=%=V_$d|_PC5asLYN`;tU(QeV%hfe%T2iWS_`(}jno~Nr&WbU!Znbo(aBzy!0thMi6-yiD{ZrI5J_`oD##Y|fe@0G)F#a4mkTXGq6'
    '(5lTM*|pO{Icg8_MOO56(P|)0VO%DW-wVxbbnMc(k9!}YCT0~_vk+&Xf%gbAZiV?Sw^{2i+@m;#m<IJAkitJ)J2i&5$?cHsx$85-58L6y2KG$f'
    '5+V;VOMn%uEP%k~_3xpceS%cQw~Rz-#3Fu$J{Bt%+UOS?ZCC5YP=29A%k{RQn7ctxt2bqYQ07^$d=BX64hCEfcLZT0_RIp^X?cHNpC=5W=5Ene'
    'ell1)#l6o<nJ5aC$RCiWy8jlxgGa0BBT|8;LV6*pmHtYh`Y{VIco{{~c-lrXr=oPS=ti*d8hpv6NN<1zv3KPPLQbvGGm^IhPVPBbkd3d_)r?Y3'
    'l;wALv_03C_gG(y8=3QwCK^H<G2qzMnMLNuB&KeGRQNz7dbF7VCC6#qgCpaMk>~>tE8R7kRFS;)@y&G9JMSEu|M6HuyD|6Au{#JIgV`u{8}u;;'
    'drp)zi1ZO`GeD8eb52s&*oh(2In-5|IImEVZEe9gnk`^|1TxE%ob6yuC5P_>|0WUU(xBuaO02&&i%`o5ZxMDF|Ki=6K$!>g<{e6^VX6q(4$V20'
    'vmAXNvY*eqfkYNum-(0zKE+SNj5SB8=0Al9NM{yhTR6J7jn)ns6`j4)`tzx#FGKmbBeZg3&vrMjv|Sllg~Z4C=sOhqW&mMzUws0BhbhZb4H#-i'
    'k}97Yx4K1sW%c=}5A@rgYHr^#OaCn<NO&(d6}JNrCwk>LwKve`8LEUorq#Yds00t*Pj8Yf)5B;7%+UP|#aT+)7hY4<wtyEdGQc#v9xAx?aIbR$'
    '6i?(RJv`_E^{yLF4aOU|y7-4G?R5G>RBZ*1-AQtxdz_-Y6HKE<0B(A6#70`8`+yo!^URV^Q5(S4f2JP2fU#prSd*PGegDB0rSsIh#Qpd|MJbd('
    'e#{igS7Fsou|2!>vfTsd8DE!33TmeS=UNu)Epx7=!AD@zLkC}9>_nGdDfTSpq;?Y4`OyN|8|l)!5D1w8$t9ir5p2=>WS0N}f_r#zzhFN&KMmJZ'
    'Z7B|_Ls-2;wr?G*%<u`GW}20In3j@}zltxxjs+{pMLXl@tl9!qB2tjL1((^`c$8hmqMx8-he}lD)%`{+F;`;8SChQX)XBf!GwNsl^bv0?9FSlt'
    '?~7tTc%?N%MdUyTQtFM;fSw{#5rB?ilm_z{got`G_>+Be>Y#<&WfJPLiyK}w#?Ow|DgCYw4pRy6i42#@ajkIHTdCh}GN_YWeODJvDB-w19esVt'
    'XC3i)7rTF%%zcQlfiT}`01ApBvG<)hWRPf_J-%x{=*9X}*U(ePx>=5F@1v+D$m}>H>tW;JEm?1IEPW(1*9hn_2u$L!zE=#d(kgc#OJlQAUQvj<'
    '9gsJz047^;nUD<9Nq>krznv9rTCyHT5!yAb^N$+DQ97*0>wOA=>aD52Mw+$Mu#SM6I?m3@nno*==5BXAG!fTFd8ACG>~Lsu$CQuxwWq^U@`fYX'
    'Y)_KCXc7a48*|EG?Lm&g-RO^2Nt(Lc`HHpTwZ{r!dk#a_?0d@fTIdk?b3>hfwF6P(+>zes8cVvz>i}WU#hsGuRE;99-KtIA<P9QU=qZPc1{hNW'
    '7-^T255{dGw}l@6lm{<<Btz;be31?LW_Wvx>xMX?i~k1H-bjIh9CTAbQ7l1?ED(q6wLr1FGX=h%?3WNE%U;uOH&HqcD|5>4FO5jt%B5p%@I@W{'
    'tBf{#_l<INB@fq|lqDG_NL}_l2$`jq2-MeHQe1DoBvV#y<aGgOo{65*4%-n%?NVKHh_EQVKh3w7;WI+4+^!-bg~O3*8q&=5a>84_?diRIL}Xh{'
    'pUV4deTY*NLC5|LMb1_`_p9JwC%+&n_O}S3GPSC0{#NFV7U$_q>#G;ZrC#o3eahYz&ad&CTrc{MXOXl{a4a(#!Zgp1+9GD9%*yrPMJDxiYSAc!'
    'nN!UY@(2JA*v9=~-O)E;2|@~=fcx!+B75cbpcRZ}PJtZNP6+O&r3@w435b)K6<FKB1gCmfVm?l~LCgW*gnKg%V$&j^B6-JlmKSmGJjoYjNf05L'
    'l)wSZt{Uv4fU@lI;KMJ{RONg6f6>h5NPV9KWalu?<8bo-!XD|vK>niT;kzrx48!A)(hbI92SM)onL60()@Sem;EjWnH;$|va|rSC2cb!zX4J`H'
    '!NDffgk4)1R<*-auqWOQvJt4v%XHSxbw2>rtxLkC^Q7Y%L7{A|vGvmL(~@EarBjuIiJY{a0&~3;llGUCd}=o<ukFy;p$?UVv{uWMP#JEUStRF{'
    'fk290LuP`#i+TqsTusWPW^*4`FNuHsy~~Tw{clhVr+!}ftOUPd9*g`kf#y1lOk8*zfF*F@`OGlkb`zx(mgOaRnNNux64?uktz*e@!Y+%r-6VW<'
    'E%I@~4QX|BwCsIyo|h6qDpDW2gdQiv*z0!09>fV7Ubx#pw<4^dr^c+h%I!-r@sw0`S=-!;8KTAsrccsrmXyU!58{Lqx9u<#f|;%Ad^vDO`gr6<'
    '9y6+bIs=5Rdk`oQRhD?o%JTjq3VnMA^7fifgg92=Wo2VnRl(sE(#EZTtv8$#Rp1zVEQy@O>J`bnoLt&rx!;F-4Yw=FdfQt6oubqu4?CGcuKX~v'
    'ctMFoXs%GBO@8<#tlQDfTyiN#;jiO5v+G%1)iby&VMvMwVzls(4Ib_WpuWgE*-LUi<1eBj`?l<K=ipRWD2PZqdr)FrNl>MpS%}AZu}7DJFSi`B'
    '{yXeN8oePFT*7Y0-BemL*K+!$Jv@;mV`P0IFi|&wi2(kYW#?Ft3N9;)jcDhl>Uy6W<X9pCF1nVj=9FSJZ4=@Jr02SVdx<);PodZM?aZ&M<Vl2_'
    '?HPSw_AC*pcNc9uiD^kdOwDXgN;C^WEzfLDD1nsMA7oS>qY}%1DHw;uL=LOlRFCsuBwHB&0W(NfG83j^5}iBV_;(!7PhcE@_30{=c_i9x%3vXp'
    'B{I_Qe891QH{+ob_=!npi@SCcW*yS&CicmF6$;5WR(agjrG0aItQl^hKwHWpr8|cI7W#H{Iv_t3mox~}OR9j>Pulc<f`VW~C-R`(N0V*FE>l_?'
    '{@N5P`o6T+3^uLut@6zgF$Fg{dZ)nM*QAtEavx-@Y^RyexTEKbnqaS6+Sl&!=J$ako%uNbe#R+jD^~Dn_13)stfLq$_o#8?tdm&vr+pLabAQs!'
    ';~Z#C4N2A-jvux*UkehP)gzkZ;GDC1R#=k5D$;Ahu<no7qWyM%zQQR)4Bz(FV{Ye^Td$%LZqZfah}5=Zj7*qB?l%g0*z>ic@(&}DcX9h|NdL+{'
    ';?b@-3T$$>hxJ%PbY)iTu6fp2|A9>g2oPmu644Kz-{sxV-;gR&!djc^ZC|xzcdre!UWtPQcRK=G0b_5n;_pSw{-{W1WwipK<T=8C(9K%!p0bs~'
    'D5b2xcq`;XYH&{s?WH(=M142G2O(OXt%cvy!_kmxjmlayL=T)SG{>N#8y}gd1z@p^!Z0?5L=)~2yIOaW5XMAs_L6-z;_x<tt)YEZ_Wz_NoQ}yS'
    'iOI!kLl{Akl;>gpZrS`**E(HIAKcR(&Y>5ZTun=|W5C{#eYLNI^41;AH9(><>=iu?Tb^`>W{F>ShC4^QSeXRMIawGONd{wsp7>&yeAn3=Yurn>'
    '5gCK*gZ)pC)ANDR*h|@CB~+?yO4VL~Nru%fVDT&*`)wuF=n^i1(_pR@`8bNFN=hHNrsn4Q+yF!3jx5o<E?wv4xU;FjV*kze{A#jb`=Gg6-O{d~'
    'Vmfd%MTv$b^KiCXtvz~>A%;d=tvyAHX$<JIOV5b4uWqSNHvX#=7Rwkl7|!2MyHb<M%})J0qAVjGNBdf~1Pk+Cblt_1Yn>y@4S~Gy_Ql*}#hiC;'
    '_ixUdm3WkPM@pWRLh?#lv$3c6<~L-K6%i#pg3BHu5r=W8^&t~Kaz1DKy=?sXVe}%@Icl!Unq7ol@BA#A%a29rAu3)N%ou|4vmZ9l#f+#B>R{Mz'
    'Mpu1h%*`R0$2}i(>Q^meRTJ3WTcMJ9Ii!*E4NVK@*eC#J)5N+|*;?MiVJ<s4XGoaFJe=#5e!|0PV&xu~D&lGv24tQ`ox;>B*`?XexDYdLn-TtE'
    'lFvKpkT8tRT4YKpGa-!8jBlcNC~vzTlhp=kNO=wtUj9|~H9pl^&-lD?SDe|>e-McP5=7*2N#(;Hm7VsrbT)dw2UhT#1#Mp^`eo_Vhx-|2XUUqi'
    '#OZj+gTtiXt4g6j**V!5tdIy<?6_X)HpjlN8tm<RwrG-6$k@t)lUy1b@I2HW;S*++gXJ1BIKuHdcAfE0`x1RRtyGBGNObzr$OfFKk!v0v7{Gq<'
    'prBc3k3YM-4)9DmVGrPH_EZCSE%Lpxe4Pgg#RhHe-_Fw)gQMFSaG^--8kU~8v~X6}M3&G=#K3+NkCzjI1o}k}?EI+cz$N@*-s>N-va?#&wE!+&'
    'i?EThPSYx$D-^RA$_}J1G%CB@AOc708T%L8<tU3n5{i2j$q2mG5>RlKhyIi$Dx%3~JT<j7&{p`#Uw=qP%i|L6y~@#?q}yhuoFe<}aDLjjPQ~(S'
    '9tC}vI$HZ0y%lz}ZnAHsyB=r~x`gghv4>o1Z?=~NV!)UO@hWyhu#mri19;;*zXnX@9j1}iBi-@MNP%FGeqC(bsotHb)gJ`y`+D;5F;aa3g$VLt'
    'J76>^p>y?jA>*=gY?0_;v&uXa1-dG5;$1**g;<9c`7o?KU8K0lbJ$qW9#N2ILDwFFa4UmjWk6P@eS%nnX+yet99w7YRvNdmcS%u|k#2v|phME3'
    '|5;&YSm$PEzsyMXhF}(kT`_VyA#IP1jnn516o1mz%i1XtFWK)0!A;f3fDk0Eld|h&#`v~_O?wQB&LP;*)+)}lGI~t^&8V^V%g0$>7SaYCUu>cs'
    'cWX-FfRlNz0M{4ezw6TaJBPprr@Rybd1j1;&s4G)?2Vfxx+tu?qjEaIemjv+)b(z(i+1;D=tjuXTuVx{$&hU%ISK8T+HFPDGCZzJ#O@n}ZI$l%'
    'uS7=bEHVYJo`)vZ3Xd@r9>%~fa)bFrN*<Er1GkK4`QB0013aR{Di4U*Q$o661>6^f2_9siRsWd>O$~&-d&yj=RmY7)YwcRB=1WpdyJ~t1Ue@!X'
    'SIBnB#AO#5iVSOJ%8>QEI~`ldW^4niSHA<4GCj0eh((y|)9)W9(RA?}&RcgdHG8@>FJ7H48I0>3%YcaWUNZKjz&SiV{L&#1MI1kpL9N4=``ccK'
    'c7qr^A)_8B@Ylc%s&!G|<t}nPE&D%opy$!ZKNqH4GF?M5u}?R~7L(MjzCyW|OXY=Q1Dp49+E^mvQjEiM8v%Dop9U)=R+4;@LM&&Cd3uDyr6HX7'
    'Mj-Mos<s1q5VaIQ9SO)4^zI-WNbxlGPaENwkR;)z((xy0Zb~(k2T;anoo_dgJzhl#i*6t-ak$dA)NDC7R6uqGd7aq!8%1Rp4>asdtXp*Xg&eE0'
    'yN16|<pixCn5BJ9?e0sg=|se`c5H8&=MVRIV_L7b<E+Hv-LzHY$(Sg6q}WPUc&#JEo21-TxDKbzr{E)zD{_J{l3H05Ib=oEc@d0gol~9B1RHvB'
    'mux~8S=@%zm;G>K<J1`YFz|CpJ`KICYz`5r^tvKYTdN*?SB@6Q3b~av+QhyHN|pLgRUc$7``Wfl+bRJp?<}vc&l-;C&1$Jn3_+Kvq17<>2g0p('
    't`N{gpq_R;Qtuc2B}I8i^hk}(sLak;`8TcHtYE;U7=2#WfdhQ+Z!NT)wsXHT#{UD~$>rO2THUys!UP)|$N95+%(d6AZ7vgRgM;LRA56Q~s4ZoW'
    'MTY4PpeczC@hzWaSY7O|9rHdM-BX8lgom@=1Py#TchmDEj5GPRlO%<s*ylpsm$iuN6u_1Iuu4$ztDgB$Ksi!XkAUfbX7R>zJ4aEU+40q{RP_3i'
    '5;+W8pcqy#r@o>pVy5C#8x@K4JMnX=43eL`{yOQWxT3sfP=^(%9Iy8OXYxIp+CfRl0RfeYr1kw#g1GYEj!n5z?7YQiB{>e8l&=)F_dH&?+{1bi'
    'XSW4vS_!dTEol9$G%k2a?;JPdJCsQ{wv$!E93{4mpJJ3rB!<R*4d6R%#Fcq(L(-l;k-#pBtegtbQE%l<o<-X?Jc>i2iYMLw3Jf#e%zNy~J-Z-|'
    'mMm2biLWyjVSh-5U#kLSwKDYbmLnwFK{m{}UdN1`WJOS{k5zNk6$z>B7F~CeOEha|Z!d#y9F!5ey}QPC6|vGka#bm}fjgHOMk^90#!^-45IZ*S'
    ';IF;R3Q>L?OqS7wL1GQq{DAv;*k26^bIIj+vit_S(k^g|=a&HD-_a)34Tbrt_i?iSG^vfu1<kRL=u;$;VJKS}B43k(RYs`uu(xN-Llys-^0D0I'
    'GFTy4(^|KEA>#CM_HLKU<|25sw_no?Um&3lSJBoV^8FAI_CSuf|5(Y8b9b**SpMMO<%#(#3<(->YB^UmzB3l?Nw6t!@<V$7iC+C?YItg%+keg%'
    '$0oeTB~0<$+I&kdPsP|WxdLmt<_>Ff2sa~3;jl^AlVbA`3C`%2dIC&j4Fz7KE@7)%`NG05dafG)h)ctH9LPJp>o`VNdA>$hU)m=$1*K&PQk<Nm'
    'GvAOjTnoQrz91JGsl>jYp669lT}0!0rWI#l%;96d{qEw6YE}^L9nJ+HC9SO<s<AlAZosUa#ybj5pt#=XLGKYT;lEx?o)2qL8|IL7jS;L0^~rr1'
    'W0ZfZRk=0ULS`K*Sk<xR*5N41IW#vdpn*%eMr+-8Pvju1_JT!-KGGn`C6WNgCzMf?Pn$2Q&S>&&N6V5=n`{OqqXPVU&gvZ78tg0Fv#Cg}ttQ*m'
    '3P*Q^vK9_1>nLvXQ7h7Qv8KYH-XkQ(Sqt2f0=ih|?31YWP`FBo=Az$RVAoOcJSK6zSo^kms33=oQ?@V)e)8);3-z-9plucs)9>6k0Yj6A-P57h'
    'h`OlEhJ2<e>ZN7x-xT{l?coc1|D%0k_f!R@Zj~F$78wsvtOh`IF0`cj26lP&G*(F&!J0}_7yZtF#On~o*6x$1S#BC8)jb+NStXL~=xi|_;U$!0'
    '?-%YmH_TYsI66k92siXmh=fJ<;C|dPjRYx)3Q$%i$qI}TkqaEEsv`M~7vgNG(V%hxwEbNtJ4Dzmw;BV>diOZ13YzyY`-M++J-fFz^r<m$TtaW*'
    'qt7|U>Z}XQ%Z~;=%axTd<j&9yW^WVB5aBf0-EW=Mg%msTyG5@=W?{cXC;{g3Bg`Rr6nSk?A(9we7GXVS4#_(Y`P&UfC=)n|eyvexKy#IuBfF_O'
    'OZ}rENk^h)5B2hDGpoclpA^uqs{(mj&sX#LSZDGbS2nu2I&;_*SR>S%jzeHov|F1$ugHwrD8@VXTuLdit!QiV%`vP$V)BoV3r$!}@6h4Su&uq~'
    '>dO=UWk*D;DtW{kd4I(1j(!skt2O*1pdDlfnL&^_pK~I*v|lA)apF=5aM#1irgrAuE;<63BAFv>99kIIWrfpN?fi=Y2tfr&m*vc}GQ%vlC(XPz'
    'A^A0)X}PTIa(a(yf3uMm+Xpdm@UidmQ4hhmn4v?74U@FVMEKR(WDvetzvt*?vl|tAWD%2eeZlzZ>w4jBXdxn$y6gVV?FDC&>X?Yfokz&L<UXB4'
    'n3s0NuG+Q&=VB$FXA+i@KO(%_76>MxmN++8$Ud?H;}T{#F^4nK$3H0xPDxyLEYCaS%8BHQEmf}FHKhl}0*)ovwNhSZtWHHq8Uv3=DBWgu(NAn0'
    'ZFBVtMI?lzKbKJX=+Pxr6odA(9%U#k!CyK)icmzRHH(ea%D*DAoWYISqS-ahG4zz>E(cplTy{m*(b$=Lpz`V^pQY_nU1PZx18pX{e%2hxaH{u{'
    '$9zhy!rVkK<!F3U4&gMckJI>=^4ZyapcaV);TiIHy7w{tHXm%eS`#Op=bb+t$2ZlYlT#!qjICqdpW<3+AhL59uGE<jpQIb4VD*$ebV*PU`#~YW'
    '#~3m54memp<0>faL>Ipwaeh=K>bV5FPF%bDWChyaQph-*rza!?r3MtF?s2Aj4$*hozrP=E2eYgU%(n#JHD0-KJc1Nj1O<y^jgXu^b1--Zn%{Q0'
    '|KbVGb>uicCX+tDk|0=#+bQx6HnNq|3PBjflN2TY4uMscrUKl{-l3y=5yLlx9b^xXou4A};F+x9Ii20bBC#ev=QL{DxpNpcjFak$X_?LywPybZ'
    '1z&aFd8uMNSmgXH*;j&>VhSEUhTU3~bU`6#7=?4msb$|({+%7$i{lMMU+0+S)V8liP>dKkp4a~;$>k#y+LfyIPu}^S`8ab^&Y>>M9@(C*PklLN'
    'e`}KW3e!|zGuX+0hp5!1v(~po_df~vKvTCqn;<kuQtnxdjznUw6b(TZ;HgiCU}F1n>C7GUN3Nxg$oOvke=kcgdL+$=mlF->EFn>}Pb<2r;FOti'
    '2}guA2SU{y)obwX=VfS6LC;Ur<U+ZVcPs-g_J>6evh1tFoFb<1b>)EH2af9<?>Dur&PiQHd+bT)3JT-O$?T~(_c7*i%S9$C1HMrTW_!(p#RMBd'
    'm(x$Q8ZpXDzd{PwmoT+@+)-Icb}_p$0I#9Al%4wf7F&+|o#m}M^wEo>n!~xxqK8~V_yoYW-e0oAsaaExXcJ@K(g!z$;-MA!VNm4uYt-<&IR_~#'
    '5M$7}xJK{cWtCq^BbbG5$aU8R3bdzLJ0p$tcHP+uN#FK-;NUt!ao)|EuhRqvRUhHJ2zg%XMe%AZyYoeE6!ceP21l2D2Ysui44~y4a;<(O;(g+y'
    'EA_{eq^v~Hyw+bRaYu-G!f(+_yICdtzPslW-^p0{1-2@mYnar11T`+?iOKhkr$MOVyNCJvC80h*ocH|MCpd1Fb$NW9)-jEU7t0o(9E{t|Gf0+U'
    'KY8+2uIHc)sGfu-OHuY!?CEZjb+yxE^pH8!9W#9gG;H>>q?MAR`VIj6Ax+|`IgGStf|rg2<3ZBcxzodMi5id)JDF`S2e^o^4S!y2ocJV7c2{=2'
    '1k~ki4WEbU|1H}g>$}R#^V1sRZV%*bD^@uHl$a6-M<V39$KX@Zq;M=Zt>e%+Cg#|6J9GbO01m^_iL&wrz}c5q;MwI{h@so5rG@?lO!ie2FmR}6'
    'PeRU3j5Weh<Y$(y`?lDplv9LI-mv-$;UC!Z-cy_3#kv1pwZ*-){ews8tLvl|oA3Jf;+&9e*6vlP=)@RLt!ZFXjux_km!~Awrkq&h-b0o%Xw*y4'
    'KTLYL_Z{+RLN*E0;t8UODLG!M;svjl6ASpg)-0PC-sT#S&#V{16(sp2RYJS`S4I!nW)vDo;I;ivth$8M^mMBb4|{B(KSti>7aF)ab@l^z4e+Oh'
    'I!${WSHHIm9{Z8XIZ6}x8D-{RkKF0HFD{WG2$7}oRxBVGX++Za)DjNhvp6rFPp3^)COxJ_KFYxe@zLwKILisDOWrVo>31Yg$AOM{RFsi+m)q0y'
    'UVgtG>rz)RxUtel*Cs5%>_07PVQBVUJcH4%$C?I-tmZjoEVjyPQ}C1)*63h*{duDvCFH>jJ{nDH364_-P`91_&z)SqqqvCLn{PoDvu~8zLzy8|'
    '%QK9^Y#H=um11$S8M(77ZpPk%*^9qUEtP!-gOd$J7jnG|w{yVYLJGG@#J$-rJcM-Bl2l?hRw{uOj{RMFlh%FQP>{#ATOE_uH~WRaq7e$0jcgFs'
    'p%^vrHshqxcmN+KbP5FWOndgD9B$7kUtJk@CK=YKVb9FLA-3(rYQh^^RMp3K(|Bw1tCapO?~%{VcdqIGa3yxK$tBazZllt25Pyb(rtNa|Ck_pN'
    '5`%swdfR>uvsRIgCAcD@nxUQx$~BbCSe2x^>#RPP=)+R_W87R^AvjY$LYIWUWNTD?cF9r+^#v>Kma9NV8Hxz7$v#$a@ayM28a{NZw_qa+#(u$7'
    'i;!su+slFQ1!UJteN+x~bgQTh)o&TUkr{+~4g|M1MFnj0nI&)<+A(taUmwKSL6UxTcpP9r32s&Zc?9FkaO+F>Q9~*y`Wf+2CiE7i=4-&R;;Q_b'
    '&%HNA$8^73yY`Wnh@y-ZKG0@8P)nkf)?MU~?;wIQ5*O`tY2oV0ZiTC!<xy#Dq(>-g^0k!EGZBEFIQQ2ww#?K61=)5z&yolMd-?6-$>LHc@S}%3'
    'U)Iz|sEdGpF$s{{+e6n|)Q&TToHq#p+7F|TJVo~Ad+!ylUM3LgED#s~P_kch?_#WKDs%@)7x~RTf>u={Refd{eKEui4$+?5=RRIt;-j_(Q5XNk'
    '8^}QtoXVoR*C;N8?e-Ez(0A0XN+`ly_6iuUuW55*jupDg>^)%v`7T)YawvjSsRX~39O;@G%e3ng>Rz#&u-yOLCj#HiYimF<9waeO6%75O=g8t!'
    'JsD|riRS4*bku{u9o%-5@rs;IbhaL)cCpLkU|XTJYFMMyj)P*~+Yb6zxhhLy1V;`18D2V<+&<#HntII<u%^AD(d(Fie5wrN?S0x;7*?OnOMq?n'
    'mmcSyfPL^-SfwC(PNG)l3>4DEIx`u{kqMXBnw}vr>6mku05CNDE-hjX1C)40gUp)IOZWUc3;YR>NkBx>nN!m~5q7^KuL9)joZsOm-ig9m6YCXI'
    'g$PpyUxQstzjIgXZm4H2GZw*f%_)YMZ#xb}ugl`6i5J@5B>TG~mJ=e#M)XyYQ;rTwiji!udFv_IO`$w%&l&J~sO{R#Nz$xf&D%v0Qq0jZpJ)s8'
    ';0(xx9flua^cfRG?f0oN?{%B`7O>|N)>KMRN~8NIT?Jd6*`PQ>p18-~hq|~nI;EQLFwELAQKI=60GeWOVj^Ch2bn^uh@A&-76)&2<)a@FCd|rH'
    'D?8hTC-4Q=I~@1!u-EqMKt{a{Y-$|dG8%|J-_4?rF9EOr?h?3%Ha3v-rjt>e+e@4!m52*9MayD)EK+NdmB9FIoGmT~Q*~x<<XqlFjZ7(?$+Ot4'
    'isG0qWTohdc1(gU13dG23H5Li%vWoW0lwj?HbHffYBp}c#q@Vz2bw{fMLpP#S`F)fb%*;y_DT$I7?w5s?2wU%S062%o*rv#4nP7A3!jQhHcNI)'
    '6Q*~M?FZqPqtpi$GDE=nn5(B;ub2*UXj^WaGWqh;?9e4f=Cp<^t=c<yfu5++NX8kU$Y~Cbt~<SNzF{+bFZLhQfUN8B#r(;eB7k&cpY*-r*;=Lj'
    'j;qnH4&)wEOs9Mk0_}W5Jv^UrcKw+p+8;j4;+Nj^s*A9Q!FOm;mJ@x#T-R<np<KV3q3@c*YP&X*YCBMICCM+@2+DJG;7k%-yKoDL@!C~t4toXr'
    'NoV}lBzH^#>j<@wc9f6MQ35b_R97|vk7g2uia&q*AdJA$zaX67+bK5V_L|(yU4kRH`z^yA^%*dmK$OyQ^?M(+0ynOIPz`&}ze8<sxru?wiMFu+'
    'Z!1abvF+Kr2}ve0Q=lJC&)E3oH9-ZFt}uk~_8h>({JE!RK6d~kQsR7sS1{00MZhLJ{YS>^F`e@x=gyvzS|X<n5^>lxGs|>>OQ4=f2i159?tJgD'
    '>?*1fB!T4cm3&m+u2gGZT91>DtbMQIC8BhhH48YG1#~SVxIjDU7qU&l)ZK4Z9RnMW5RD8%K&9*_XTc4@K^bCr$u-wr<D6hm_PrnNCtvTeAC0xb'
    'm?@5*CW4JYI>eU!>-PVJesU(PyGWS*aK52LqE?o<HUArH<3|!6=T{B=5K+tet%KP*ImYGwQL-@Bkv<W`YWKvP`}acmVvrRdNVF_KkKwwpQ11=c'
    'f%WRSG0FojyuDthw%Q?K9lgJb_gE0`be-0JVr}r~E>=+YA8i*|>rFm)ry{>*Hvct~|JO|Zf4!L;_0!!qINW=z`&eSh9n`WTP(3W>JNe2ysL%3{'
    '*O(QF9b8ckeL`e8gvE@ub!6GV*|TE@NQnXmP&%O!65^_lua_`eth0C@ieIYS1yn12EU_6SN$LVR(H_cy?YFDx14PV)ddG<iou(CXAr$nB)WjHi'
    'Aj%P#KNh7m0#6ZQR(lh>1cznoZN$03#Mu6e9m<FX6BvQLkzkXg4Qckq<W<AaS|~*qqROJd*Q!HkdHk?|)I_)(<QD}I+BGqfqT`smx1K`vWrHay'
    '>{t7PwMg5>pl-jkKtE_bE7oKaje6alkbBDzxnhTM5J|&$&=WPA=1PD^x+Mq<ySmp5T_ZSU#L?Y6J#bIqV&V1;pAocSN8R*L@$p<Kgp4Iw4&8~w'
    'vM1$pFXpR5P+&Pdv1Wm2i|f4fJ8Z<Eldo{rAyR3qUor$RPq0i=-bZ^5)6?{OSKIB$wQs#g;_I<<6FwA0bvsmhK$TSXyZlZ(s*mxsq_kg^pl^Mu'
    '?a5D(Q?3(`Fm6KTldUW@=D?UADMq1bv%M8$Y;Q=-`<ddoDe(_tELovj1lHD;aahT_nSl6@Fh!%<$JL5t(zQ%XOVd;wJX$>_ih^WM7t}N;gr|&s'
    'nF*vh!fVFPkF@UEt7o63O_989`>INm<pTz_Dn~xO{&&Y(b$jke^slz(l0!p~2Ed|+ECw8_7v{s1#av7B7a41n(v-$%<;)n-uwo{VdrdKCC#&%_'
    '6$CZ8g48y22}D_(cfLNWur8I86HOj#QTd5TNaUig>vpkow2yrSs0awgTZ8*qhgfM{AB|RZMZe?R)2;Cn!@igWUr?SDp{l@tuy6HrQyV8p%01Ba'
    'Ju`@L$J3a5MiFv%n(micVTuMWqAl}PY>I755+tz^x#W<)kw`Y0w-BkMKix)dU?c|UnnRU7v0Gw4L+nzrc*&i8RzgPj9)y4u1r;y?QSwJ2%qS$X'
    '^IGh-<eqQ|w`|L+n~zGosAY;J)@a7PeF;dubFG#Z9e%4cM|Iy_D-7tds_2;?$5k|Kf<!Ly(+Gj)6i4d9Qc}Xzv^~~r3l(_ir;=dN?XR6V)flrv'
    '1a(*1J&8kC6gW{B?mA)=e4QHKfr4F9T)jxL7d(fHIXA1S?VOO8RMfCcWN;1B2H13WooY{Bs`at=O&(JoeA*<!Hz~(`xS?(Zs<RU9j7hGw@;Q|x'
    'R&r`=;m8;0m8+$2Cz#1D6TnIN7=}PjXYwCL)A+8c^{K{vBDbf8y2ry1APCbAzj4QV8?M2Wl@Zp+oktHZIkui+Q7q_jwix++qZpijUQnc#B~+q2'
    'f%08@;m@$s|7}tEfo3zm%_8<#<v_(Bb*TML5m+6Q70U*ZnkP><6<u>iju_zti6}?qmL(3f5IdgaZ{~jy4ArUYNDhkPnyS^^YI&x8+~JhZdTR1|'
    'B_0s=$bC8+|8B-L(LkR-i8<0av>S1%Ws74EiSG=B1r-i-O?RnSs6EoSjsrXTdJV<cB!wk4BYg0co?^AsmLdoTVG*ke%(>X9YmagNC#=c-tjC6~'
    'TQLpqwIw#MVGfLNx!>auU;bJpxKsXL<(m@*t^_Xoq|>EHlW#srS_y&MD3l)ha`n7`<T+&R&fX^q=N#32&FvTcOff!Fj~?w;8F9e5l}2euM96~Y'
    'a`Hq`cyVIp;McX&(WTT9aE{@qbqxz}_E0?JqoDo}f7nx5w`S>}X>CD}DF}#ilQ+n6vA=h}OCW3RmfaC6wcz~toLl?&l<sB-p@sSKA6k1BG?zhK'
    'UIuW9soS0bI(JC8o19cKxTX}psq9}Z=+`v1VWX1o2|9)SJxh$ZuIUJSp;cWj`QF1J)Gr54MQS8$ieE7VtteJV=_sf<(lOLi5?TkfW`H?Jz9^SE'
    '$)0<wU+S$Q?AWocsz*gG_dz<E0>Dj(iklWgtXF=?48x05YhB*!6MU8E9_vxu(lYlbl9UIlC_9ThV7toOzB4|?BJ*Xd{z2I!XCEh-Z+RTGd`INs'
    '1n`Z_?_-_bTN=P{6(RYOoTY`HC1P)Q!t)w$ERgf{aig3ul>M*?O>sqtCo;)a6=&b4?39l_(Sy!loJzoiU8i??GcajR`Gx;honm%HE-M`O*%_~2'
    'f^$LB*rLggeJA#`i>cXt6-8!eZIURNf@vxPv{waLUWW&Uh9zGc4x?J4R@FtQhwp@*kBLi_NImOG`VNx-H%{}*5wAo!-B1!f>0F><L{nT`x8j5J'
    'yfiyKw<!Fc5U*&TI);0z%N>g`UA@PcKe;&ea&%A+fDhN6*O-y7eZ%X!Z$FRw=2JW7bJw#nbDklPJ=p`f$YaMzHi$poPxWMDY{%AC>?9vT328J|'
    'r}QSS%A%v@M-aw6_q%cP2q`4&LR7N$dW_tpl5wQ!2>V5L>$QtTkt<_$2@^wj)^nt%w9Y+MmB>QM+=rot26U!Z;L_KMF~LK)LnkhXNkN`v{X{hL'
    '2#iUzFZt;#9~XviC-?nX6}w)*$@Wo3vs1hs=Vzs)?t98Qn=$<it4k{_b8ixXYUj=^wlafh!Wg0pF>PW`wz!b&0NGXcmG%;ofiuE%0sAKj$`|=<'
    '!`G1i$!B7B<!hv-xhf>7Cz(YoOt+DUlcM$t<k?j_fKXYP#6a}Omelc!<<^BUMysgO0S~ZTAuxY21gT=mw03qiiE{Ab;}0`AIS^SSpNNx7lkNrg'
    'f_<Jjn0MQanx2#DSuk#EPe-}YU1>3v9WmtJT9@M<Lpn)!wy(T{Ua-!9&&2=Lj)pEW%<j@NMT3^ePhdv_zes;y`kohH9Yv~Yt>P7ldTM)>^6N@0'
    'uSGyxR;OIvEywf-DUpeudI@c#(<t|Gr;&DWv)2m4U>NJBB}iXZ|8t7Eu=awbJ*p6^^Q**x3Odc+%&t*yj0R%RBZA0wM%Gp#rSaTj?Uh%}VH_Xw'
    '-smZTv$zUH1X|r$U}3ZXcc9X$Svj^Qz{KwA>=owUU~nEXT59CAt|nat>)0~lK?0Mez4V~osY%uV+>i5he5vLky&Nf+gFXnwl+ox$$T?f21)EQ7'
    'MtfTV2^=Y{is=>ta*gdtRuH`?wa>VFSS?)ld0HRr3iiY>tqdWcL^^`dkcHVxf%?v%{-!<?^Hs&cVi!u@vO&W<UuF(_#j!mJMe~HS-N&*Ra;Wc2'
    '!QF<1FU`6%)11}ut!ENLjS?nSXUfR;Zt8#B>qf&ODLx*_v7*au5YBNOSoLKm7hU<JrGKjtzBueeE8n>k(k#2oK8i>EY2W(Qjeqa{E3I_Ny-JD('
    'y9R5JqnaKvWq(J`Q7-eJrkB+9fBpvAaKze;%prS(Ys0ah?<v*hBVlPDKtmjTT0X+#C#1C^Q$2<QyTyeyqu+zR@}0z#;aGE++S-BcbbFe)WX{;9'
    '(k_$=+8i69&?o>K>GYsplN8Ds$(^^}r_Hj{e@RXMB{ls&Lu#__2God*#o2I8Ns<}X*`<g$?Fp-x4!a;*c@$-a3icq49F1=le2ylY6RT*DAu9lO'
    'juQm-wO7rZk?WL6Xu7dPuXR#csYhEc>slyshCq0c+|8wyfgbAjs_UIw|3HNdlCWeC0|}vpiDN7-^BIhzuRuqCAQOMbnfrl_*(TpAW-Pv|dR<Z*'
    'FXfD+PStZW<b`6ln=TqWyYqC7-@kbJyO-u6RA#a_vFgccrFYUNLIf%!<&rKxhhnN<10$O#cqc1*vhDwANXj*WPHP)izhs{T<@$Bcd@mTmPv$JD'
    'knWS{HvhR`<XI#&qbnxCu18qCIt#HLdq?{aBr<k|-6I>SWZFu7w!^LWFgcaKX67bvD^>hIO0!LDeF;JAEeR?FY$+PgE^+cxX5inRG%NAyQdQbS'
    ')FTx;nNod*-&J7`Et*j0*k>)ET<AKZhK7A<nlkSJn=vk_6JH}J(l9a_RY<KjLwOg_A$3Whshv@6r!e)9&tl8=*0a&m?VYW?uWN?J)CC-+$SBgc'
    'F@0pQ(*2SAOzr{YLKk0Vu@h(r`x%It9W61j+J56wMCO>Bs$JGUSByUMvC3u17Mk#jRc~w1a9zwa|G8oWAz=SuAL21zSn)<w5;)^!DNXNVAHz%o'
    '+f#`yX5OCFB)Jgf!MG8E<yB&bQ$FT{_BZWQVtehIsyJN}?Q#l)9T|=A!7C85fS=_J2*aG$+3FwX4CLjWTC~hsKpx(Ii51wBMCnI8Ssn)&op}hX'
    'st7Af8<%}eJfvn+(BHjM|Gor)!(&1FahK60mo}Rib`Tkb&=Tfqb>1w~j31+^T#K7wt{gVb;9`;hSOM3Dhc#pU&F6RzYac{iPcYb95|hHPdLlo^'
    '(wqy>O<Rab*2*A$4;39&g<eVm3wNuYl58BB)g?WLfxw)^duRDqyAxN0P6Mv`2sIfKcuY@sf<a4NlEhJ@8Kd>E;8>OW$j@JE5&H+Gdr1J8;!JEi'
    'RBJb@i1lz~K61(XSjlo~<JS!YMT4@aUBgCVeO=c*JeW>n9|`v&oqc+JcdMKAOlpD^+W<$xfre-G-k>yg{|xSPzQ(JL>nu)>Xh<;fm4#7CNg^_@'
    'v7JpR(6AKOu&gc{lui?r#D&D9<(<X+-)w>ZW()k^oGrlQwa<N7`G})%7RhX6;;aokJ<h?)UG<sl7s*7CCmK#zIHT)}xuFvN3y)RK!a_on(!#<8'
    'g!dA`+Y@76q@9Aw6tV5_ZQ8%G$r1_p`CLK*DN3LG_5<Tk&Pk$KEAdZR>+E7*W;D9SO2lOHJybx?QqmO<{AD3<4QUZECc`rnaeLmZXV`u{Vl@nQ'
    '`9ovtI-D43+VHpr6e83OC?8&FasTt9*<<2OLD9lIZo3ZhCAGQr)VhLt09CA$+TO7-*Xryg&S^Wc#>n{s=lF-D7b)b&L9p!?-ZI%@cb7+z3W9B0'
    't|8suPI$=+0|Y{KCIUcfG3p_!{KN~*^yc;Oh2kzL{Su(q^_N~l!LpqyxeCj}O^i*DJc(H3*`mLA4j1l=vy7dr#;ZNScHwl-d(zUA=uo<SJVk`p'
    'n=d(xng<?Gr{t6+s~7qiWcZJ#sc1XcO>pzpdOjaeTovi>mJge>wlX7*AjL^8*FjKFq71Q`jVEz1Y*JlWeg%<<qIyekBL0>+<6O6CrK{?{DiE|$'
    '&i#@w97b6itO^w*K|x||c+Qe-gm3<><NtoLvu_AwPF+0>R9s08!3bs%&9Ab4JLzSo5Bhpx&G7)S0E`9#b~&Z9E9drge_X$HHSU+0<AVUS;J(HG'
    '17g22Cs)kZKr-kGT05FkmKa;CzG5YMR_DK+Vs(>|6-D<K#=QLkSG_UYIKxv^p^|)-@3<V%0}k7+7$=g{N90lC%YPZ%UV;#2uv=}_?J-(M)eB&$'
    'oVRoxd^plgF)rvel(O=8&70AVxS}vHmczdAUFPTnnnY^NoHSObjyq2NkD?GnbZbhkq@DDA)^dkcm9}lv_20t!zlHVx{h?ih+1i#=XQZ+D4n+nG'
    'i{4Yn$^dJ<lt@?`q0VC4u&0&^I@&=Gi1x=_>IS5lrDkQLovw)cODVlUd9^73rU$qrcz_00KeUR$$yV;!#*5jrJg2U39W6XSuy`cdG1?7{o%~Kg'
    '><F!6L<3k>KJ`X9WT!5z8%_HVLh9r7`H8+Dp}a5EI3&PPON<9#;lto9hexXEijah!&vE3Ckg=mWcX1@{o!;y7Z}$LS^Fr3Ub2uqqE;*C)6$<q}'
    'TSlw3=S;Z-b=St;K4qFmbORb3r3=kbWB-741v^)PgJ&rC>mzCfy72F>j8DhjiY3yY0pY52C7MlVE<|IaO;?bIki`&zboFl!Y-mvl<8j_aB3@rE'
    '7Fhz>|58-tLT<7XvL^M^78Y7NMvi=K2kg)(Noq^`4L$wyeh3{TZ7h4v6V*|=mKt8=_0CoJyJ_U%A;`DL-|NZncL2=MH^&Ur!&v`nlQf48rZ3BP'
    'ZqGd#%wgm*LxHzl=6~ET<kF5SFSp07ju44z#r}R>=E93T9CkJ<`(4KVATt0kC2=_{+KD82%WSPs_lLDLc=tQ71)=t2f8Sy|<#EzxK6R;xN2au2'
    'R+H~>H?m{`C4y*rSav0dUGh|8xX4N+;Fc|+pakLhN^O27xac=mkPoDACCg!)6}Yi@ce3aQ_G3H`B{){E<tgq3YjsH@7VS0>N^?X*erv99+UIjT'
    '&y?PrKd(yuvYi73#{4fxQ0zuP*v{gKjLIC5%(>ZClXR46H+oL8EK5$2TG31b*RuV}=>Wzlq%W&<gL@;JCk#JFigh#U$zs@C%J)rNs9pno1Do$n'
    '$b~RbH_Jw4;W38Gu{4!P;c_&vk1?)`Hl2CZ=riI-^U2n>k2w8Ns8P`laW6pgtP#M*RpMm{o5Tu}y$?=d)7-UD19`-v{O8=;GL@HN*i(*xqy&Om'
    '`@pPlI6DhJZW>-)LFsHw0{f)qR-(JkF^4s#0nNNUqn%~rK*7E()Ls2;L?cqi?K=-8eW?Wg1Nt31Y?2Urm|fhemqP$0_gv|pX)p+#?8C2OjiDB`'
    '+9$f%IBfdm1bZq%zTjZ?Nm!MGSs8i+wxC-AgBCG9OW<Ryy~_-=>CJkwRnNKzrE(>V;j`R}WvQ5(AVaKs$}PgAKfK8-nMdnk8$+#GZyHq|?k$vT'
    'r`#?;{p&IZXmsS-uF0_9@N^g(aX+@aNHlHF^jb=D`OfZQsJlmS+%BwiwcMBx*axl<+W*Y%qUYa8$R(r4m6z)-7=h4p>-?)-e9=HtmV2@QLP#S>'
    'EX9Lb?~P#fF6G<S34?=TLX2$g%<j^Dn-}bM&Lv6ePH~%V81oK=^rCLz%`iU5hiAQ5`bVELS0IWl!Kql=Wxez|{MLq}`-xI5-)Rn>r7o|-7-e1c'
    '&vXO`Pin;aMSsM!MnnKsCtrye^ak#r--QTn9(xUM5|Vv3I9bZLCy$*iD4yeRHfB?rb=W*f_qw<N%he#_;Gr0%u0dT@bD4c`#^^qUVn+|l%2`!1'
    'AT`_ll^+v94zk@Qd9`U?=p|^de)2Hf1V2&IMvE&#U2786Uw5tIRf-m>f)b!-NbInuUBK@%<6P}0Cy4QN`UQpk)YJ^dbNa}jvYQ28MCf{GkW(&t'
    'FS~9}&>n+lr<NJ!BH`a_V)XExUc17@g-dKklZay9u83=3?{EzZp&4b~enmd~Y!?Tf8?o<zw3t^*Pfi&Z@$VPRc*Oqh%k_df2e@$FO+)eSZ$=*%'
    '$<U*U)4)oP#6D4iRVGO5uv5fDw}ceqmcx(aP`t<~+%7h<K6jH+&yM7E<*4<g+Wup%=WbDv;993ta;X9t$<U5M{Vr+6J&RG*TH(^)`seu!oZ4?P'
    '+d$oe7AT9P0_1Z(QhfZbJ$Sls3jiP!PUoZwPF2XvB-5{ON0{szD2I4}01W`W24}hBy+%6x0YazuDtAePKzBnH;o-Cv@iB;>T>AFd_d{taICy(Q'
    '_ABK3TZ>x!v2OR_qN*$RIz@Sob|nlrvucYhru*L#!z51|_5tp_8FsjptgwEysC{w9?R#4hhPiu~J>J-+Uo7EwP3>oQp-7Kh&}4(9!?|!;U{8Wm'
    '`n#xpl#`B_5a};~#C`88D+KLkY+)ZkgjthGF#!&O?q|MYXHx&2h2rG5hu}Jp$X{mwAjBH(Q@iq0tO)h{=uW0Q;X@qMb&{W&cNp=ytC0JBR8Dr+'
    'J^dH>{TYS^B#hblBZ#xFG?HY*EuXMNO&h1p=rW9RW$P04^CF+4^d9ccai}E8V#{y_ZZs#OFndAwjAR)QnUR6r!Xc69F8%dJ|Byn%7p+8Xg5;Pu'
    'dxtPxhPSz)&e~2o{!vSKo$JyM{PQQwwD%mJ_s9{q)NKs+bmInlwTBJ@azAq4Lz7^v=aDK9@BFUe2FczRJQn@1s5=*25*eHiXm<V`gV@VVUfPXw'
    '$%E2$pRFnc%@r-KI>DKBKZx(=uW5YCcZ!r<Aa8|h==`)_cRc(3r&g-qBxKK5u)miu?8K@c`*npy?eD}|r*ma_)bg028s5V&KGqsLU6Dw4dOJbn'
    'QNw8GOE0C}v(BDG{9;qAGM&Y8Kul(gv71__$S&JestXW|DUC1=qVz0yK|HtlRzOy&<7*kAa|#jCm6N!2^W((s4qV#L{aAmNF$waOh^%=Ti`scy'
    'G!^<7t>xeI;ab<1;R_0_K<u2j9i;fMcbJ}cer70R_$v5JtTUUd5Cum_FF`Ul?3rZRDKzs&TVGN{HL)fw*LK!~_I}t0e-gTZ609qC{*H4jcHQr@'
    '$7D`Sb17<PTIe7u#VgB|mPF6N0-h%<=|4QvPjKv7`35{y>$M4*7gFl-(<$UQWmp#Cqr>gK(-6QC!@U*d;XvA~dnv~W579n}dSt%61Q0(Qrv^(%'
    'fNIu?mD%zX$CVwuD~G`2g_l9~%3fVBojIp6a|nT|1L(+D!g2B`kpvK^hy6?bPGI1auCGIR@8W7Ct%Hn}*c``p=y05>W3Jwy0>=*GjS4<#KnkVF'
    'nu&`)opr45?)%6ynpY2smA+*t=um5U9WeVpOTh2v5^~tx*x*~;)*(odineCw7Hog)i?>3s=fiQDB%YiZh(lCY8X?DtSGW~XaKVXw0m91LgtH<1'
    'h6YBfK8R?)>!0Ea{32N`#Zy6}rGoRASCv_9LN_lDIm_o7mYIlORuX4!V;y}d@r$yFT)^lLZ}Neb%=e#^=OU^usCO$xuy$)CSHF&(%_`AuP}$d)'
    'zse65>b`}$m>>JJtngVf!QN=&XQYn6fLN=5T&(2c3)oU?6Dts05u7cQsr&kHP)*J%{?SZmQVRLuJ_?yL-MpoZKO{@#NUJi!B)>fQYu0}#t?_I|'
    ';naAoFxyo@0x=j|A{wOgtGUO*zu`@`>=VbV>SM4M9R%|DuZ8iy7RLW~ER39uJJmFVSz{N--jb@4eP7pupgKbCW)hty2mYvn7_lCk=*1R3VBpK^'
    '*07eGRP2PV`UJs#jZ?^JpL0t&0oF;Fl0F#jHIk-gU_)D$Dprn>t<wHU;Pad`hD&nKjf8PRCL)j??C;*80jhZjy2Sf}8xx8<73mD%1gKVF-)5Qg'
    '(IhtMF&0C@G{S$iRk7nTd_nedipZE1$;t81eYF&J){Tq#^7mxqC1L|OAt{P4x2uF;sFuXD!TFWCWNc*gP9e)7d#0B>PXwCxEr=J=qf5#m0GLFy'
    'JVb)rdC*J0k&gDz&;kblVDux9GNjofGHaCEXs}S2;m`~?+0_gHKH3-s1>l$Q(2N|K3GyO}`<{O7gQ(E_3K;QGK-Ozb;*pyOEK0kh8AtlXhqje6'
    'i7~FQRR$AmL2IFm2{yCPzkuD<`d<0v<%G7wL|}`z4{m>WwoIA5hE#|N((y!vX7GE*zlx}oG^t>8$70nE*eUP{82#x?JluZ7dyspiIj4M;=hWVD'
    '^!ef*fx`IQOZB^Ax4$SK8J!22G;8n#$==w2hEk2~B%XGi6yW{at6~3c9IY1h?m$}BiHztraFb!|ib5*)uyS*#jR5QVTzd3FCLK|=3i!Srn*9I>'
    'YUj_$H`QQIAS2Pto_L7W>kx__4`Dg%Gy7Dnz8X8qL*lBwJBdj=mfgTpswWEepu`3}Qmn{Gmwk!Ap*LAfCG^OxMK5s&!c=9jw&-0>>J{1p;5TJP'
    'Br1CcC!%`E;;_p|SN?^OW-$4tXYk0@W32(YX8-Cb;&glek!!4QXqCvhQuXY~27VR$#AG}HNk0JL4KM2hP)L)0#FI!Q7(do9Yu7H1uIbSQwZT|L'
    'ObL1SS4R488S*&KWq&kr$crxC?k|q(P?J`M{kzmdg;+`^oAwKn3mX}*?Alw<21St%yEk3ud8Q7TyToV|UhE|aH`4GuOLV&zOGp!VY&^P4nWX$k'
    '9pZsV(P@nxzL2DI)un9i($!xsrnxmtK)}(CdqT>~WktwR3Bw0Z;Y*v!Q|~=PJyJX*^76y%rHv_mO!|5*Me^Vlg5#T7W2YNv;Rpk|v4C%BA`oq@'
    'cKFYg_)!uLqmDVzm#&W;YxV_|u&l5nj2>F$7U|br1jkH%B=?7y)^`K=U58Zm-zq8PJ|BH+Oa||F4^#_w`}Hye*JY(W%dGo=VPu(+NWG$?s4s@X'
    '%iRX^N01{UK`;p;sH`-1h}$3O=8%SwaqS+p>I2Y+Y*y@**rkcz#-Z)Op$#yFZAiWxzC_T~WnFL(ptbtlpwVh~F86VW_Ex;K@q&P`SYe5AIt*qR'
    'gdBzGZg3aV;VA%7Ma^<^(mk|phQ7)HP%CX=yJ0^0sCcb2&w();FHJ9UX0n(oB)?qyY({Dl&|4YJz(#C$VPOsY9ytr028Ti`D<q6Ke^;{Wljrly'
    'a2|r8D+%D<?r2)ZqGUaLse_b??FyZBbE=kS1=_18Spw+hqv8;C11`8S1m2tMsFHSUGs}vW*#o(@T|hG0>rGfSZ+gpKD}@que7WN`jm;%->3`8Q'
    'GTz@XZbb~j41ew<=}L>q>WDKm`~CoV0=U(vz`wZd@ahJ`I!V{SLEaVFjEQxQ=)84kYAsbluO~5kLVL59AVmTZynRIVCXKA4b(6iW-D@-fGbE8G'
    'ZuBgs4MHE%`lmeAv8haDbJoL}kdgazIskCY9*Oz*23uM8dtRik?=#?JDun;|w-6yqskD{GrLth|o^##yi`{ImJ32Pfg8w^Cki9R7%SYo(;};^('
    'fk9h)BD@=K8YPnffiC4;;Kp_e&60{bF=M=t-|Q>-L#zA>B%u14_xf(g)~g`=$oRJ6D_G8QotI<!_V7UftmgyxOLC2GaIn!_m&1gvbBtUhRNmf^'
    '2WR&s=Ck9%UY>F;2~=L?y)N_D3+8)G0n>6iW@sI7&ev((AGG#Ua+TkaXdP4f`VF<cTuiR#Ak|P9Qb1kSa_1IpF%T`{e#m@BxkOJik;q*qlVs1Y'
    '`zB^)fcOZ4&?##&toj{NH9?20rAv?=?x`8l=B;9=M=QjUz+)y>EWpPtRxb{#Tyiere(oQsj0~(0itH54oS56*k%`?X2{QNY@lX3y#Z8Kr+60D#'
    '%uBjg?zJ5(DjB>o&s#U3IWvN4*LdP`rx1<AQqKC5biQIlZu>o^cH%p|!3rYQlgm@{L04?&N63_)VBe8bur9PpevvnJ3MM=G>Jo-&Wxf_kfRPV2'
    '<}|W#Vh;MIM)7sOQjo$*g`sekwZ`(8=zBR3kTqVy_kUOjh|C5@^3h#ek+k*<-OTT8-m7q@4n1f!FHL=`9bSk?qE1d6Xl4ekUO(2iP=h4DC|qzP'
    '?_R{(K0Z1{GAdc~hL>sE<cGBn)C83JE$WI9hijk`w+k#~ZE)>j%2E_`?Ea|+6Ml?3{lWXaIxMR@W-SMV^AOu3Lr&w6xjBL)s!+GCqw@6p()9aS'
    '!lG_<7jd=}Pn{L<ALv9Yz(P^{0;saj#+#cy7f~){=(uG?BTyJGGJK^C0ApQF+0P?i-}|_)*=uzN^^`6T)0*h5?R5D6**mjrN0KASZY4KRd2Al`'
    'zp>V<4wwl5jU>OD+4a-KV!U_(xSQ!9a;3L0hLL3&D+5iQ#_qmky5C%`UxFj6L3rtPiEfP;7r7p+b!CnaVieK<sB?NEyd-O^EVC5*(t7CXeKNo_'
    'A>5lp0pGCgfC?BK($@B;eaRYRaV@5sBi2lPEmbEwmVJ>^*a)qBx*Ig&m+YI2x<r`6sh=P(yY3NYA6eWVvo&D<ri3F{F%c9-O<};GZ0aw+m%`wM'
    'GJ1gP)fHe(R`?Sh`3$0F7|t9Z6H&hqh=m#q#wtu#hYGgG^_KhYy5f32LjiD}rz`rpge&Pc0u{_|43%(o1}1OH%lE7rUA8^NB6->5-iT<S;|I>s'
    'K4DN~m#NMc0;skkZ!+gv4#t@5Nv$vthd1}~b^0@r*<^x8bE1Y$V2NG&$<`RR{bCK&`;Mjt_5yPYW<d%#vqkb^rkHF#Kkbc+HaBtdSCGx}Oj|u{'
    'FITVOxSBG29y&%tA2Q!#U$p8Om)=hzA;b8=L}`o1ggk=xjff!Hp32@v6zYS-WUEUI1dBXu*4|km{+s*w<S))TuY=D1p?1HDt_{2QPB|3BgUWsK'
    'lj@7-!DMINeG^me6i!{O?xqKmS>ECmnf>-KWO~26vfrbRoqY@-?1suUmahgVW8v%pvZi_ED8~mA34{@nwd?TUw4=ER>C1xa!p~#1Y868hX?AJ%'
    'C~TKZjeb{o`&9tTPJzwR>^(A?BIB%S-()=etafT#rP5;^>yi1|#XwEOtX8*YS)gyvNXfNFhG4hU%r4?^Up*HE7qWF~abgS+9P(fnN4U-V&1Vm4'
    '96L08qQ3eN3@vN0^L<JNXITn^Zcrg5)$J$P4O6!5V8Uh!RXZ5qh1bg}dV@IAnE1t7cgKNM_R&s3(Q!dv!7~^$_^K;@xY40kp-2~80XN}@hmicZ'
    '&`-F?U%n-&M>5-8C@kDzWoC=gHDE9k60CSMg8QnBkSFAf2qCjr)eRBD?QaV~_t3x`8MjC=?|`C;!T<<pz5l`dvxX1NLs3!|gZ`~9^v(b%x-1xC'
    ')NLoe<Sd|MPI8=PAA7}#Mx;S3fme3N2ud7@9qSD$O$4uTrqigh+q4p)h`@Fih!8tg^Y4X1OY)<zr|u?;Khj9X4wv0NTDb%$PVth`7Ma<D;EPuh'
    'c7BTLm%yjnU%S66xwk3WFnZp>O{zj>yQUa@IOiaMhQm1;!d`@XK2SwXsRW@8tw2@(6ZJ)Bz_RiN!6CgRlQBj0h+V1HV2N>{ex~Q*60*mAwH(@X'
    'r#04DzAJpIv#Ak^i=ecSF>fD8-}m_}oV8cR!cTJuM|9P-=GX#>_Ohd3YFLr)-&P3GqHY(T7_AgSr>qVVHUYnw_5bs%7&v^F<jL0g2KO}2!8+k6'
    'UwQTn%HAY+ru3tyHnzk`+a<bpMDw;=&E5mS%T8GS=RHBVkn(y5E`6_{t%19)F)S{dA*$eSz^aBN39*b;kCY-vT|nwpyD*m~=3t`9n6%x%Y=m`o'
    '2V5HiGzJ7~A-XHsZ7~n}FMG4Mx8$u3JRfUE<{F4I;>as6ZK}l3Aj<zkAPKSV{gPh6qJa%m_jwgO!&g=peRb*AS-9;fl<>Ui{X-6ppbR>H&ZFg>'
    'MEuwR^rn8fg}z;_k~;|OpA{r2AWp&L(nU{(q&w&Tv|!{ZtuT0QQRGzM6M@L^-%9I%`_b^fQchJdS9>DX7N}`QIuX?tgwd6EVGFvzy47{B-R^s-'
    '+=9xNmvVvI(9$c}Z_;WQ^2Ozmw<MfC9IpP~oYY{e)5YgbUY(J;)Fs{SkX%x9w}BwDZs<1FEO7GGSmKc`%Z}6%@>r~$*-6vJSueIb#^Q-@5qK1L'
    'IB_83-u-w1ri$<W0l;S$046|nF;^M2@zd<_ND-w7RYl{^3;Ns64*vGLFTjHL-eP^jc<CF~RY)@q=r?8FzmGcNtySAV$0a<9eIT9xY3nke);8iT'
    'yA@9d&Bhvs!PCBL9G?zEMo99=?J-?qO(7~|p71+_aJlYOqL6iDGS<)t9g@LokqvZq6l-MCoz60Z=pa%qBrJj|%c+1GNr8SZ^GoI%HbFT&ZbyiD'
    'i;r)G%44KgrJmcU*yq)r_q@EcvCC8GqY{}qRpEW^K0i_rkFv3jw}+si4LwaBO?!l~Lx}0F<Pg7yoD&-OQPEkw(KmApSs+<=Q&)~68Nn%+vF^^y'
    'ZZ+!Ed-g4qCi@p&`~)kOD|q&tAFIW)NIKRANfr&<a5KK1*=L;SBJFk;N)0K@4ofbE&5{X378?Y;c~Tnj`R^FXVz)&r3JFMP0jFHC(;mLOa{kT1'
    'cA5Tex6r}FP7sI8qOxZ^Mr|YE?QMyL03xil6jLBY6<tr9yJ=P66)ZYbZ%7p}JU$-UW&{jtn}O4iA$h_0Bp<>els6O^utP#vDXsAb00-h45V5dJ'
    'r+rGQMVk*nRsYTXPFTK$C>ta4l)M^775V<*iEl5uX0{jIb^?jhfoYvuvo{xDpEU$pQXI&Y)LAVLr3RGUa;@Ri&1q#*5_jDUBQR2uqfRsZwL8m%'
    '$cY1?Z-)ZhR?wR6k@rsFXp{g#`FxoF_p()*&O}^1=HI*MgnU^jZA6ULm)H0Elx9GcdeQhq#Hb>t_|xRNa1RlvDKIp<X~5daPL+wkiY}%_`uy#N'
    '?Fpnz8u_X`b7wCc&whYh7U32@!2v}@Gt-CL;f$~2P!P}uI+Y|&`R9~wk$5co_cQN{<Kl!S8>uo1&e2HA!3f!&g!dBF4RZ)UFL2rcNv-~#(>%*-'
    '57}(i;%IMy%aLFHD|W8F1=>v_jtgn0=>!0~{dLj_M9m)W+$m=fP}pz0hOj(|5_x}-UtTXTA9_S_S?qKz?8E<G%!4BOH7VsYlNKo`VLR%e5r#y`'
    'QgL4j6dt+v%p$rb>U`$fqAU{pZM8^_EY6Z~OJ%ZRYa|5eSeNYcX!Ew@9)YrSvEfpQI5LyrUY6++Rxb&RVF$Hq4aZIP*NE=aM;OiixHLe2nq0H&'
    '8{~iqrU<%XlYw2K3W~LJz{&fYJMAcehEVQkGyjC%gZ@CGOM1T#c62qLJs-COg<M&`<+yKK+AFl@k*Z~nr52(S<X4R8k?v)Z7be?zZe>Y@M+ZYI'
    '^($Jnf9ji(TwcM26NA&qa@!<djFtoOnU$4xjL)-d8l89RfJ@OQ+8Ox5n0-$^oU1weottF;Z;Lp(`;7iq;JkSDXS$HnWn~83pMEGT2pwOF?jNhe'
    'U?|r`UB^!8Pz(9Z(20yNRD)?vD<U{?=i)7q2_=E6)^a!vWCtiJ_Rgi2B_(1kiRk#I$q}Je)GOqZBf-nVGY*K<#XIhtbQ3e#!M(TWNZ6^Te}}$y'
    'uTz*_<IHsL^*T=_prVn>m;rPA35pSofZdMvla}C=2z*O5s_zPcwrAGKoD_+Eu2P{9K;5GQzGMo&CdVC<pLZU_Kc^4C1S?I1AW%REjS!{qAoiez'
    'YCS&)-~VNNhrRc8w13_b_)h2d^{10xGYjF6Ji)g}77@EM*;bg2mC(eNdB;)189jF^Y;EbvJ2IErYh>SkYF7qDudw7CT2>NS)fo-~Id<;S15~_+'
    '5j2U-b~;%JX1yJU!p&eOsR|;d00m&<*{vR&T12L_cP10U7iM|0os$S?%$4tWR5P3M8vGi|AIfqbXlqvN?#Sn1>4>P}Q3U!5#G4}J;20WbaZhek'
    'pT8MQKWQ2uMKamv=h(GIfpOOF#Cbz9bmYbx+D+Y^`Vh4?3Yeft9sFbQf!+twj!F9R&?&$$4)cS4nOOn{2tMyEM=D>0@A&sb^8xH?f5ZTh9vPV-'
    'e6eJe(aJKN%x`qo`|W_glB1~wOrgGHZ?Zqh2|h#|N38tyYp)6by^948FKAAR35hduI}>J_L1A*)OIZ56ymdR(2o;Vb0#<XA-&F)-?O4hx-kH=='
    'uVIgcQ+unJ+5Uq$#;t!S2yzK6;;wY-@)(0Y$VL9}$dsyeg;2B1vzUdQSy}1}-DHx{x}p$yV8Ep+A-Mz17#}1~tG!}E1~GXnY08P+`d4zZsC!p?'
    'F*JSzP5?vJ5LIGw&8vD-?J^Ctq|Pdt?JnZhqkh{vi>H1wagWx{R+ZXACi`1tL1eHCY!nIL4wc4eBGhi<mPLh0W{5W}WOEEgBun^)_R0Ejeal2Y'
    '{C`*N5E;@er3xvALA;X5fQFFx7qysn%cr`l@G}u_NJEmrd_2?SuT~ES%v2B}ShEuf^K8t>2>FHeWjLqp^&P*)%DylWNMo&DCx-X}U|R<PLccJq'
    'sU8;+SnPorkp%{!EjAX}INPVZ%8{b}!#Iyiq`*Xy9L36%ZRaJvOZ(#LinDS_2v1qS@(uf=a6A8$YVPmEl9|Kupzc*9@XAgfJLC3<lDWqKq$8XJ'
    'PSQJI8tz~`cF6BMjRHF8)&h?QMr=|HU2^K-v0K2m<tfPwp!j9VHfxL5FW!VF0ZH12+fiKhyMsbBT-J=1s5Cr%pS=b8ij;DG`;9#J8rM+yo!Vxm'
    '1nQ@-m#5cb>n&&!5^M5+^hfrQ##GI^2|G$sOxRU1LsbEf(s*ay{Pjb+=~$Uy_{}i%ny*xZKGr(f!*<v8|G<$1#yGYh1-o;<dY8u|5%n=c8VESI'
    'DSG?r)@=bi*jGMk5KseqW#jFT?Hq*8n_7h`!#?}KFYTseYUxpKbqK<B-2}$kVeZTorZ-wMarMRZZy)LHt^{L_iBbX^?qb`xyRZ`~Pe`dUiX<f_'
    'N4gWYthRmUV@OQ)Rm8cL#L?10Zw;((mV9<E%>H>j?{eRMyq(tb*Pexyv0QBVB;(8eYAH>tQ*u+FlV&eh9|`_*SX+ZfM)D09q+n#v6)pu^K_YZ*'
    'G#Dk-YHCM-edW>d0kv1OtKeV}qmSzZVn(u4^hOkVqpEA;#fvQkj2%O(zp@v83T!Lq^2Jm*tcPvn9hHL@*(JQq>tp_uPzNw@WeN1FkWm!wUWkzz'
    'NjI0w14mOmsw+DqdVzDtw($Ktw|Ps~$7Z?A9zqxFYiL)6PJwM_#!cn=#{!!Jb5tQV6e1_8?jhKfmmr;D``;|#HWb*5>zC33dtGtpB*~%qnyWe*'
    'Tq-X~M73j-$|RyxfNsvZxJhWxMI@2nZ<M^lkrPU*Dow>;&mfz3iuF}WC_u|Bq@_|tYqOGslT|Rgb%ZQ802f6B;hX%~Rb_8B$v|467uTUlc}R0?'
    'mz4cal~R?N-LG&C61TlXx8hlBFhS#rM;1qHbQU05U<jLTARmbpLoReibS>2gD@`X;FRVAe3<U3krTfCWrQbr;2SaS-MHrY6c70f|JvH}tyoWbq'
    '=b7q_)gX^>4hc^pDPsn6Hgrqm4ZNG>;h?NnY>4Zc28R7Z5oP6wp(1=-XeDN;rv@pww9mgDU^OLmjH|G2tSb$&S}(Cw7q@aJ5-no=0Pai5Gux;p'
    '%VmRL^t00}`ipr``nwGGfH0>l-a^BE3m}d&WYfAb`6CKs?mogz`SPe-jg008Z&^!P$+q_jBg&a4i$#p2$HtsV9254s%m_%$p_UL_b;&9~yxym@'
    '$&=$L6mfJ`AL$;E&t_4Ny;|%oFJS;93JtX$0AcEBiUHtH;Dq7fH~TU^hp#1O)#1t}&SSMu={Ja1)Kkt%+EG8aB#q5Pw69(SkqQvQY5F>*hpBji'
    'tG+N5b|-*8y0rtr5KG3XgJpH9Ate>oR@$0qVV8fEr^ZerzUMIOZW0*K^<ss9Cc4x=jzh?8?cD0B&Dov(;{9pXeT`S1;rGLh%nK5tq%NYx6`7`e'
    'kz}w%>=HHrV@r%jw*O!c*}gGpND-?Pq!mqtpkXT=+3oo82eE(-{XmmF-IVL>gH7|0J`!8`6>c^Y{A}4Gk?J^wR<rjXQ0*)Vn0yYAcDuVs%o?b}'
    'Mud`NWoqw6T}cBPfSF%0hQqoe^=<jvc`GwrKAAO%R@^Qrvbpwa7`%;WuPxD^+Wos$74jFNEGsyoM+}bc>JoDfl4vZc1lh{9J$ooP1|^r(8e?x{'
    '-WvBybWDoifbJ%{02dcD&^-!H3Z_<y5x@=2&fP>!S4gF+L0+EXkXw!J7t*Ka#SgI5X}3XsoE~kQ;LvZg27YVdq}t`mz)2`}_L`No;yRn1{M4M#'
    'q^@%oXwE^7kT=_9UFE~tR%Vx}fxjxTCe_MhZWICYEHEG+M!Df+o9NA8(R^@`k5<oKma#>|Z--ShqeNEf(Xrw<Bi0}(4#H}xJ#tJ&vkvb{*SI*V'
    '^)&mIRaR8(7QlA={UwmT%ABl0CP-b=Us8_3+_=pvoG39<+8a1}YAVfQ_#fuCU>xm|r~&Npg#11VVEg7uY(Af)pLVHt@Uo|i6vl6lj-uDd<_`ft'
    'EUyg}*I7FH6K1c*h&d2&vQBt)WpLET6k#swX|fkXT&YV|B@oCb&qcf~z*>5+IK=hkC}+*+gf`POrzeN`xhw(~PrV2_?VOxlFYHXr>+dOq5iJoh'
    'cZYio_jx}!1*<W7ir^Dy_pQ3${c=LV&wdU@oKX3jZ)nFNk&H*2I6aME%q~-H42r=l$F=vs0uMutESGteckCYQpAgah3c;sZbRrC)s4Ay9@CkP9'
    'kvoncaduUg)oWMTxHkvGD<o<=ME}|^WL;{KEJOTTS1XMcABRm3nOUy>7FbthW-Vu?T|ujl(AV;b+@q(#nV~5N%v}SC6~7Yy<|?n@nA$l$)Owdn'
    '1y3}Qyt7E_$m_6|G*2_dt)t6kr$=_)IT>J+w#914I53zLGHXHX^JeVY*MbG_@RTx^zm#$u6iTF8%1zW*Tsc*HbYjKJ_)qC|xPAGbBD#DU(XoOJ'
    '_P~nP-l9r(myI?=k+U#fwj7;0aD7V>stL(M^9*JPZ}N3!v|%L@bACjak(!&tjzjfbU){}5$~jDro>VB|OHr{;Fy>h?AdFD%QeTm*Zq0fy7`KRH'
    '7rJw#qMO`!hin3KuM#$1E>afD`b=@K7y<I~_xM((XO%y3g^Z|;Vp6x3xNn*RY86pEb3Ewt@lL+5M~_uvoGU(RUwcMFFY@P^X)PT^RN!vsDgTpx'
    '@Bbe6z^X;L?7TdxShVLf<r>{BUP`~?OnqpG1UTQYut6$Igb72jR(vz_e)DKy*VbSFh5%7uIF?joxH1YRmSdbR?jP)n)qV9zO=Jr%TN4a9vj_VC'
    'b2t<Ew8UV>A$4Yc1b5r_QNqD{g>)rN`6`dhL#rtAq2_-h5n&9mu5!a|mxOq>s4TaCbGh;2KQj?z4XiW6v1s?G$0INYpui?YisSUBa1CfMrJa<e'
    'hkY{{2H#($Hn=qLzj@jxS%ra6NZ;NP%*5NCBKjsSzPd%o2DNM^2_Wts18g5@*s$Vg)KMnH(RI-Tbod>UZ$!YGh2(aLSLKi94;xmk#5mMY<w3Sa'
    'zfJlePP5279ETFXz!1IFt=;_?B~1UWB$C@&iI*Pt$8=7N>8;Bf`fP;aMO8?+?;;A;U~DEG6qKLRdls%u>tq&&%IBn17fwp}A+n<C?%bQ9#XpgD'
    'u(gu@>Ka|oH^W$ZSef<4(!03`1ce+ef9e{+<8XJ%p-qPI9^)fh-u@dUGC&|sY6@d&&OQ}8C;44H_XwhAj~B!(nY`2Gt$nqVUO~1{j)O|I+*PR^'
    'NVO$drrT#pn$8F>SFEsC{SPE=^?CWnmAP2&5n$2o6)ZUOBXg@9^~LmhyFA8>`VGw>m*eK<I@%19V-r)g9fq$R-RE=N-0Rw!WXL9jEyoFsOm#D?'
    '|LuX$t&sOEqdt4<JxcM8vOpKrfUJ+@W4Z0|(yPetl|{UjFdkzOs`{wZZr$e{-qY)sm*Nd3QJAbzob{V^ZD_Cd`@Q|^)i3Veg8Px-<LWYb!VA&w'
    'cmrjfb*g&@D{Vq=tvtP(s7BCeKAb1m2IWn5Lq1_0QC+o1*R<1C_xZ1GcBn_j@#oD9QR15}KZ_uU^ut;&G;Rz&kY7Deg$Qu-NqtIOQ(jvH>kPDl'
    'JOX-8%94IftBm(@x+fwd%M$U+{ZwYge#x|nyw=w@_%Hb8U+~TU2>9mTrQ^R#$N!I)j(}V0NKF(u%cF?yR>w*UCU}uLh1Fe_nGqr9_8(+vCkf!}'
    'k=Q=jXFFGtX%66_E@D;2_p`1QmgiQAm)0ekENgz<@R+L+hk(09>$=_-nn`k0R_tW|_S>^h63;9GFoYu_pl_&_qf2$;^jMM}V6{grMS8f!f}^bz'
    '8ecPLY8gn{XI<}jlPNSYNg*Zxy=p7%<K*rKa^tPTfQ6A_POVihTa;JLBMz4|?_-62Zl{=M^vhEx3JEZ$05c2CX=K)i$N@^?t~<%oz8&+T>Y6EB'
    'I--qo{OVPILcLi@K&Cu2_8~?w2=*4yH^_K!Z4yGH^{@Dwcc}A6mw#MnzCKFr)eMxk0S2@}9DZ)ET+^_@r;bSSW>f~dQM2AXj6m2Mu$uc8Apnc~'
    '*+YH?VDPEY4}-9xkI+~ocfCYbtj)KWN|!*OUV(+(|G;6_;i_XcPpG&LH*n*Ux+b@hHMUkbon`L#E@4<UceWE($BF@L*2tF)pDdmC!d=+8U!DU+'
    '9N$K5;XK-o-UE8|bd?Hed%a~~c7B0Io`JTQ{=dz3O^%f6dx>$0^0_^A{g1G3xYBmtPZs;|+)at!h%zJT`htSJb^L`va>+a9UBcSoM#>7Obiiu-'
    'u!_*ViW#5$#2kHGsDIm#%XvJh7&oWBQV`aVhsQwq(99nEwa1%HIg=a~6*Z7^;2%}3C(GH~)iK2!==X2t&n8yGhclK)zwPrZsH}ki&9INVd@<15'
    'CHOTufS|+O=doQ}Vh4`Cs%wISsE^8erAh<2J9Xeb;TN8d)PCD5aIuIDWmHPk$?EG8rj}>Y#P0Hf9q#>yk)v7}IGdnU2Xp-ty_C3}*DPBN@U+L_'
    '5StXz_0FVvfuF4nZ_-_1^o*M+zqY#E{j9}ym+(=0qV33>kqq1FA0o7624vTHPkSmA3KeAy?(>BQzT~Do;8iO<qd@Q0dl=Md70(s*Z=a0Tn?<?*'
    'evbXH#;%OHw&cVt_Q*?oj*X7w4LSe90&~n~RXc!Xyr-TM@ENm4ud%z4F4nTU$g15SIBWzN$$gvxWmm$eA`>Wu{z~jkFqcMHp!bgLd)>)s^{IeJ'
    '|Jl}*5e9@1!fXd$7hDc#cO|wAud<`rdu@%^OO$4i#O_L)pVsMHcJy%50$QIeR<y<rTb@rAQuUoQ_Qj@S0DjIFZeXb4OZiSHo>QoO!f?K;=)2~V'
    '{82?xnEI(+zcZpM1G?K``xN_v9t&2hxg$Z%bZb#fg7L9m8%xI>`}=vl!?byVnn#kT=1!@bq{dzjA>BJ1T(P4OAQp0y`f*2jz*&#h(;PbMUJMRz'
    'uapz=ACN0P${)5;`CJw5Yp3{?=o<+2Eqbk(mm{x}iSSng>7YSk(4sA|eT<aY5)be7kn~c2Y#{Mdd9x#ysER|a>hb9yxlq`Lqg<*7CN&&dQyq+U'
    'i{y~k#1tkC`A1iP;`dma53(*AMr|yuywkG%Z*b?t1H)BUa-cI|6uNvoCtce|J`(2BiN`hSdzrVgX3>nZ+eW;mr;c!l1FD-<P%;Vk4=CMNWtZZN'
    'zo=;*{`rV873R{GaVMs$TEcqO{+^byGDu1$t3cR54cEtrjzFx`X|yi7gIrK*+{cF7>Yac6^*vj)HD{4^N9A^s&{zVlXWAp=rs5<M^B@qCRcjNf'
    '$!6&l?UojGa!>kiPIaksit`oKT+~4FORg!BIHZt8g9xFjwGk%WyScAR^l|}dyu~_&?K!6$-E|IEyX*!+(eeF?bG#yB?JfGQ?Bn2<lGn<$cmiZ&'
    'k=IHup=e~2jZ}mcSIB~sv&6TVVZ<y^FsnEDd=j)>`MCM!MhmZK`x5KY8O8J{a|!;0{k3;kme@;x+CYy>0tiwU)yU>7Pq7w~sjSp-c37qhbfk8-'
    'tFs3cq;&+K?5@g&|G<8cR5UQEv&D{vI@y}9eDys~Mqb&{jeXSIUmqA{`^B~4VY2J1*H1U&>=AYnn>vidaRpICQ`zTzkAp_YOCLcKX(tSUcTJ3r'
    '0L8_|)1MPI)HG3X&&$0oTrX2M-i<+4jz*<+(gLd<A`#ZrI?b!So)1i`eh+x9-Wx~q<_~A1U7w;KYjIw5cYR#S^Qu-iRy=JStO;Rk7;R4Z3Uiqk'
    '=;N`6$0+JSas-G)8(TYc!5&PB;n+y3WpGNWVvy|puv9<(Ui(7!s!7No6yVXymbGhUJJ1iWnyK;w*?dLmS~Hhh7*1mylQrYJ%H;r;EghH?1h)QE'
    'X}>g}hE`P&>Q<?Ltb5_Y*Owpjr`X2vJ?kA5kWICkde7OG*A^nWo7J5w$qwIM&)u64bF&o~IrIt)N8E~}#?A#zotuEVV68-_E)JaC^o4q1n`E&$'
    'DbvCT_U`+@dwga11X)2p?GG**uwI=do~(tKdggRHNtonzbA=B!O;sNnd+huejr8;eBLW0gz-{-NL8XKgWGUU5xku=e+zw6H>g)vVDVvO*`?mX{'
    'hZ}(Af@Yabke-y}Bp`!5a-%p3noWgl9+@9ZDQ2v{BYrX36IsCvjZ@O}@<U~o>vkgin2~H`F(S+&uv_54pVdx%2Jx8L_YQMgeUn{eLxQhn-T4E9'
    '_E7@f75S3W>1gjk(A?nNB&xuAz1j1x<aZKN`@Uvg;j27~^WYBK6RTZjbdKyrP_Zys8SJ#Xws@lDTTC}&4!Mga#C7{yde_jA=2NO)JrlFtd7fh&'
    'yaczEa~Rssu^T<laKpa1fo|23m}i_k2@Ez==qU(5rRXTfl!osBn1jSr(kJ9LqhFXDvUeQFUoSjx5=Yu1`E36_l<XTkZoD3X>?VS}^y|20v0g(Q'
    '0NRtAw$$7?I8(dmisdmSJ4A}G->yY7Jr@yQnnPiqTuLhd<#dlz7($=6)_Y#Z4$zUs0n-BjB|2tN6gMzc)>H_A7yAR7xu{d$7}c^AmX338B1hsZ'
    'U-6zeWnUNyfD$rLm%i#x>;)J)Bd0oLG>P46ilE1149D;Fgn<K;Nm6pFHAZV403|(DpDp0X(>InPY2t&_!lPjT!p9J}2h;QXh`EIU>umGf)!hUY'
    'Buw%egjB5a*VrT8+Z6V}fDBhkNFb=(+*<&YgZ62k^o>DXJ$mFTr8q4o43Aw@r|j&XE0Z+yihD*nw{#b)R&wG*6Ck4%+Gkz$@la3eB+AS;PKK#V'
    'A4?d8_I;N<<4`|_t|D)>*OuN^uocTb>>RGOfgzT(x{OjkhJDRv=zWG@ArA0<vLtvwC?KwpufO<zB`Pzm(zl8$sX~T5Uti)gupb^Rm-(AVldNWV'
    'V+9PgL?5+EJV%tx-DpiE(8@tDt6WJUC2R_FJDwS=*phg;W|d(+8JP1~N_<~xutZW8Ee%zA9;dQ);SWWBG|~lyo=drX2s)k7B)k5lg9=4`L`wA`'
    'so(`qsmQ6(-r3a@3ZbZKTCF<jtDFN8XtijcQnbJ;nA_XfLVB@Y|MuLde(h;3GyF$4GKC~9%%yQvW)k*MCs;+{D4IKUs~?TEf7cAkMqK&`x$bNo'
    '`w%n^d@afM?Vi~i_l#r%4=L9gZP1-bpvg+qKG7VTk;~pPmSwg40nf44-};TvX<)zhk|T^riQ?n@`#O6XW~h(x{I4Px+#(4zi%KQWjp8j-Hj*8k'
    's!;TOWVWV6N`QUbA-})c`?ket(D-EQkaj;IN)JO#Jx6qDDg=tGJvH@!+hXqjO66A&yQU)4Q5I7ve7HLT{VoGH;_22=0`QdJC0sqSlu$^wA#r0P'
    '<1dx$%-vY#2j}*D0nXO^(B(A+oB;>oQ;Sq8cII&7+&^spUQU;Lin$^-oWvlzZ9`^F_8G7R+e*$N#$-hQX59XzOErQ=G)a=HXBLdOZ%iA&P#1Jl'
    'hp>2m8jPD0JO0JGu=ahFl4V+F$9xsF4Qi*^Lz{SGp<1L&Bx5Y88qA?yZTtz4mZ~fX9ADMt@SQ{yMZTksvBMszDf6|4AXapGb(59cQ&f}C8MJ%j'
    'aB0jUZcoIIjXpATnP!53w!R2FQ}SKyX8_ux=hN!<a?Mt5%u2^p{ycL8I<)B6hasMtI32u-IQ|Tl9ksM)<W|0=cBl($Q4;N74WW!VB&;P81tFS$'
    'b|~ew&T;m+K;iKBM5O^7k>MwY&>Sm9quYs*<hI5bkv3;|0V63Z%jf4pX=No*B9hXxzq+{}sAPMQ&=;!{@^J@$tuB#79ybP^Sb>&yZMhlWzzDOE'
    '?H#w)cuH*HOlc;ijO9jjZ~)zEyt?Uax%Ae88nzE{KM`|X`@fN-|3;Gjk3^D^+YR4F1h)zQZk2F%YwP;izJ4tXpxqWxj5G|G+USf4o4!Yum<G)z'
    'PyM~oB~%%SyWurOr^sLjms|PM5W7%ukc$>Q*lTSzNFRR~{7(MQLu`UeII~w4-PLME`QX`#@i{^j*L;QrRdcygz4l_JL4`<yz}nSDo2|;;i}B<Q'
    'xj|cwfIiJYT{nw$H7$RTiSoQfI_ju~N+GHytfwVqvCC?@5UE!*j0j6ZL<64;&n}x4rE0}JoF#kn5oJepOxbpi$ZyG<y*KHnFcq|y1L6D8P9Otx'
    '^iIdvYZJ+w$L__6xek{Xr5|@zO)};wuy`WzJxFT2ZbgV+Kq<9}<RnPD$HSEgTxfnM#|0#e5!||k8B&O=t18s<WB-h?#PaqiyhDqA*;ZjTwQe`>'
    '`=%5LXkm9~V;;kfn0&8d_S#F?<j6?#NIR6CIZ0a}4$I1fso|t{%@a2iaUER@C9r7%RmTvwf_>?E#GZl0{;<kehi-uUm}{Is5f){tGDJr(Bvu?E'
    '2<mlTvr<Kwy`q+gPHx^`o2B>wceS)JVW}QAk;P9L5qz!1m6nreKpKXINQc$HQ%&U-r;So5$%G{FeAakonNk<f^&WJM=}L7I%({^yF%<bBA9CnE'
    'KME3#Dtn`&`5m8b12rZ}Dm$aaQZ!!3l2E#9ZAcZ+zB7;O=u?z_31L<ET~*Ke%%W;xRB)r=*o@O_^hloE*a0Y)fVpZ<!iSmg5Jnir$XI(`fg<W;'
    'J6oQxiI<e!YK&gS3VG&E6BdQXCuijGnVlN%YeGU-qu4xa!z)kSt)u>I6-$}bg7DQ)80J#;XV~KFsrfM-Fw?#5DlGNk(Q#+jlfW9R-#G<A#F)+{'
    'd6vmoJBl!B$xK#YJjzEZab@!Mv^h;yidaYwIYI_m=jjp_F)DeaV;9yy!78$Y``I&y@<(p3H4K;CwVz4i3^LBhdtz~SxOXgisQ}xOO2IME!Pdks'
    'G)Mw6h)#}yJ|2TYpzFM*_JQ7Lr&vCmua}5sv=L!>L&#s=nP}TY{&SNAy9Dqhzr(7hL$60r{x)RBL)K~B2Ajd>0VG?m)W<vI_j}ch4&q&#cx-yc'
    '+%KSo3Fj)<fielpU9!HiTb*+b_cX2(k&F+!p4`qTz$3Qpq0TL{wcAzIx{P8aIw-lkJfznwYNH~nICdqY3OvJ(yn=2edWVUoxfrYdlA20QM5hjj'
    'GqfCFnv~|zW#~&J7$0k&uMrffiY4iq6L6i_&h2;h1oCJs>?y=mI|2Kml1@eJmR_YqZ!vTqXi9flm=I+PjV^76Or}uM4c0_-fc{1Grpts>Q<S`k'
    'pw1q>o7@7Q4!_UYZ*#dJGFWkj<znnEvGShtB|#^I?$XQ#4W#{&5Pk3CnG#{!GP0`<yBhW1hx^}$`+w`<rd3K3iNqSw{%%>nkcu!@7%%S~kiZGZ'
    'Ft?{lF#lwpt`@mk5})REcmQj(Zoi7R#J*(W3!(u>CGqCY7staWyJH5HAQhTlxT>*FGGBV9J&rJXt1O8gAVW&w+5)0d7b+A+swOqN{r^i}^A8zs'
    '_5Dq}@^op%@|8!BbTFb=Y$hr0Efw^q_0UVjGqb@dpCe3mA-YizOH3<wl`asYWeTEv`K9cYB<cM&5-lSGz4BGYSUbv?^#ZHsECV11^evG!{=>sv'
    '$wa*`0uRZyzFAh^_AaM)80z=K-Fc!$D-wlWuxSV*&U?rj{oZUq%aFNFb=cij&w5909Rrl<Op0_*gjPQo^72C4mA!k<aVqybTgD%<<l`+Ve!NG='
    '{+dj#?u-$X*y|vAoni{+`QAek2cDEXHt<S_TnS-gFVPW*iErGq78Mk0<aK$G_Etfh9)XI?K36Nlo5UxN1sLYTNj3X}p!?oW6|{bjE6rhAo%^)('
    'E1Ux~ID;TB_D8@N-k%O)6r7(j@BnwVT3aF2lNp8PhfQE3^=^%J@hIDfDvms_Xog^izA$99Pm0iMQ2%W=*j5>^bi8ZsZk7I(@^^Mj)?EFzG0n{#'
    'tbD~qVvo|6GYE=V+c`S@tnsvVSj2v?5LXp9l0CfECWRTQKr+OW(q5EoH9@9L4r+90vw2#ok{uKGpWJ)JpX(GkLlXP8t8{f~&L&s#nsg^**l@s`'
    '=$M-jWyle-L*%v6WJ7NM`<(MP7S}Ey^7J%@s-bl|&}qhLtIfcsZtY@E3Tw8u4R>l6ECIPVVG!x|>TO*TTjElee}BbNQbr9!WH3CcMQJ&f>Pl|&'
    'yuzEWr@A><>VBaGyyezu*s_9n#ra{N5=yOGr|x~<MHejo!PGg-s3c@*3LMJr{)s(MUMQ;?M$6&awN@*Rt7bO?Ls$0W#CEbnEY;|uo}aIAwoLps'
    ')7c&VVJrS|^4?m6M7}jO2mN`fqbi*!^Hzk&WR>4eUHCA4+v^$;zNd4edTSEEH_*G*E(o#8`l^Sr5r!&aScGkfMpCE>oQ5g7Wjunq%dWE%k+8~z'
    'AM)DtJVRI9_S)lCL8+tqmzlm@F!N*ue=T~b=U4BNLWs+2=ue{>sU_sl6p}~;Va%J7KeQVNd1?rULb@*kQVc9Wtc1#tF#?WU!;K009f_{)j6IW7'
    'WDr$TR4aoUvkH96vuwvo?N!3@8WmZq5v2}1Ufo3^pZ)3nfJ0^U%QM`sBTtiwp;eK6R?$RNSNw%4TKlP1Mr-Gcb23Rdp8YhM#mL*+NlcrvgPS<a'
    'lHj{D<=CvzfzMt$R}Gu)-D<y~w70fTMC+GG2LNl4y^cMsJ0;;A<nC1IsB8o=Zatz@w;Xs$YR&=cG?&QoiEX`J2P0}AFi9f$@nH2RJjg{D<lSm+'
    'YGL;$(_XSPM(_U`*Z(!H{|_10S@w17@`T)s*jL2v)`%CvwM)RzZy@e(&+cqeS1Q~UQ<@a|(Pj)@&_0?V$Xjd9J*70T6Tgk{f-aCZG&Sc*ejGx7'
    'gW@@MWZgvcE^J#&qR1hnK635gZqXRcu1}I}tAt<Rm6NlWxPlr<X53uQB4j@HDY_!egIh39NKr$EEE9zG&P{(}R!H-z<07F5|H**^+j)MuT^+0a'
    'FCKp>3M-<oe$hkCO@gnHm-n8;ob{^F-yUX=M;O(M)<&H1B@;@2m4XPkdkIevEB)+o5Ky&U@e^K_{4jj-Gd!>x(dAj*#ms{4u&f>Mj@Gh<%5pZj'
    'bF119e`wEgxx&$&(fzmqKMy%#fr?tSEr*wNM7!k>c?8;dtb2>&G5?_X5la%jPUOfi;F3f;I{-+I7liYxg^cV6)?0LCZdcMoN);Ka-9xFT=-=F!'
    'J6(nr$a>JcN$CKTz$22NO<vPfWc`5WI+u?iFT1052wvpMZbevj)sDU{U9(kaxvm^KxM^&1*DMW2Jmr1}9>f|=z_^nNa|8k3*gsLix&E=#c0*y4'
    'd~`>H4Y8{|XUs^E6_U50P8n#pTy4m<ol_LI@9#XQKw?lR31!jAfPb)k9oBtxVAFQ5H{R9TZo7u31oah6$5A>Sp)L)3nzZ*`i^xRV3I@Rc(U^4d'
    'wcJC8q8#DeERx4nXNwPP`yK7YIFSKJ&KLlRi)`XO$SF8E_jdl%xY^>c_4*0twSNT8@0(nJ6J&&1@0Ng>v>Q#2@AHU-?Eh{re;dD-EMjdcUm?K`'
    'hZN*HWkHn{@c&^NGba%%kA2$=)`pl(*h>U6OOl5Z_VdHPw^n-{=HO%gK_;}L?whipKab>VHaZ!-j$n?R|9$~6t(?DBzD?|){p>7K7Yv+#DDU^A'
    '&${*aK#$V}*Ur8_1?g7KsIkAMekz^G6_%5fg+R2!+<&`jRSjaymQt0~4JCX{IRXkY+n-q|eox4wC<Vvz=+US^X8^jIFy!sn>s#S6uQ^eNfE`gN'
    'aDC<L{34v0Z)pvMDrDN1P(?;Ps;rDj7?4<D!SA4RjMYW<@{4&Z+kKt32o)>jv=**V9|cq@LT?}{XGjy@JVGQCE%VK8K@v9b`duPZ?Auzo&d7dc'
    '2Qp585uM73c}0^`*$Q92x+2GPW8)f3>FvdMn@HZaLxy{$ie~^7*$M@a3pU}qXed5R*;P=p_66cdj!xyo)^=#s?4qal0|->vs+H2CLwT?&UJ`7-'
    'Fl3Y|Os4&CR)5bWV#r<^uNUH&7UsjeVy{9Uzl#`1g4BKa%xfWfmUm$IUtdB#4%!7-Z`a_2<Hg^H?!EEHrq`48_}s|~Svz_ZtQ=x>BDL+dI%Y^l'
    'W+U}-tshn@y^6JC^G=+HT`BggT>Yy(&Gt*L$YEt)ZW$+3ie(y!Qd9n{eFL+X$XH?jJ)WVYBhght<P+`wxOpEBdc)pt6<@QSEfJ#maSfy_sYW<x'
    '_AXPXnD$uB8p=X|XH*qEx>Z1czW!fmv0{=Ftcj)S!{YL&POQ>Op#<96O$@+dm?4g@jIId-Da5RH#=g01AqP8N`$i(s8?<7zT9}(fM9^-MTW4*h'
    'eSn(XSE+e&I5QsHwUU9fFr$~NV#fJCdkXCktmBV(h0nI(u1Gfp&#s>%6Zwdvq&l#Uy}lPkU(}~fLfmY<M`b@Enmz@I1=eN1-?hJRRCVx5JIGUF'
    'Kw<BmB#yGbPeY^Wc(7Xt>Z3`rR#L;e?HW@K5x}w3Lc}U{W4n9Jfse^LL$%*z5`9BsCb;w}EQ|qrcnRoz<J(L%I;n>rav}+R94?|SA?v5EP!WRa'
    'bzT`PReyy)E*p0Ty@$)i3b(n-j3>X|pnF@Px_<A4zxUo&sGhGGDjyKU7wW)0F>c<i!OGmu53977NIEc*AL%)^bezB=hPF&#bg$DH4{=YgP10*)'
    'o_)(w0zobO4#lk{7AWGsKt*EbNfelaMX_FMB=$XPdk&?F7t@~l$j?6gp{tizLi`jq!JcHUg!HDZP;iZQ_Xhm&ykJ1q-E-@33!)CGPKT6SlnBd&'
    'UBxGeS#p=KDNH!f*9~v;V2LvB#t(#1D<RzvuUX3NcHnqhu4&hes_2{Bil^Ia-#cdF>~m?7SWfbaUAt)yPcxHkJH7f6Lf>W)`@837MrTc4l`@AK'
    'r<~*0uUU?tA0-Dtx&SjjWnC_tyT39qto659$R2b(C4GRvP~`ox<)TLkRhpcS^*yQQJDQ``wWMh1o+}&*7I&sv5ygnT0OKS&vQWqUP($}UsaYCX'
    'Qh_Wa8Ljbl<s1#BVT^6I9ip=w4my^wFFz1c<Z!kM2vFpbcyj>i()HWFne%gnsJf2yph=ka&^;<F1rP=?W`Xp&BC#VM@#c>i!XvFm?18VlNtV2o'
    'NHa|dID1Bde-PvLR;Ey7p>?;rIC0>0&Qa<MKC@?xaOZG97Zhkaneo+deL?&Qu?y!WtF_oMFL_-9OGqs{<Xo}XEBEX|);qgI58KHF*{H{at%UDN'
    '?%y@UwT3qVCP9lzw&ATq6{VyLU?rn05kwpmKc2su!35k+vVe(NGQ{iNc+K=Ld-7lQ<o^(Ra{rB_dK3wzwilG*qbl#kzV{h<8!?q_jc>^arMFgK'
    'h@n=FlI-SW!YK-a)m{DE6w5tEZ;`xf50A`~`^C>~5vk#=%CknxuEsiPHT@?NfRsd6=dVh*+#{VPH@WIMWX>rk)3(zWAXKz3uOM2gfR8Xn>h74c'
    'xy%yvVHu#<)u34uH~CMt5bYMJ$|;4e%H;OX`)k`rxGw)223?7f3H+O>-ivBjtQ_hhM6CE>d)S9<x2P&0r-6o1JMAP>J<bzHKDEl-VJ09EO`MYC'
    'ET&g3`j2tfzAOIO_w7d~r66+>66_j|E=s0^l27ZQkF@GC_#%~U*HBbjm;K+mYHw=kLu<fHT;+(aEB_+2NB*$dK1F^nbEQ9VEVsVD6`cH*oRa6^'
    '4g>hoVoN8+d;9m+dY4d(#5sHBLKzA$x7CZXN$uNGD&(JFl<sf!d`Q_XT8*2l{IrhQzK$Kj2FZ97F-UV|n6NP@jdL1N_!UqH64tNT29$Qu=v4by'
    'LIDh8c&s|o*-)9)qCf3VA?kpTi0YHtwR@CjzQqC`S9*yXzr`TgzSn-WNdRf_nWy`4+hb$B;0~5efJmmoL?(X9vIQa7fvJ|LjqDYwzJ!lY0!hb6'
    '*I;D7N!cUnBYhK_=lxn;c&#uIC?{u~{e43dIJ(4x&Pa93N^~i6=gW73S7r?jWlVMT4@Y|*J3e~>boMP?Fst;fID3yN$Wg{uu>wLfx%Tx8!TnmZ'
    'lXv%%ZEi;p&uafgv<?EB9_?Cr(Oo@jt4p39mtG6q><HS^`@ILi-q8}@f`k$CjB<X8hbHAQrSc&i>aB<J?J8!D6pqfA*q$7K$IWJ+)UL7}+|c~P'
    'dITyc`(d%<K$e|Sc(S`1p^63WWzFg#x7#(}{ga0UK)H>0F@*@GA@f4(!|l}*+$l_rU0b+fv!_d>M)SdtH4!LGewQ@D$gt`1Khf{Om)Nqce^!^F'
    'I2BexT_RP7kJ~gv)#?~4w`aKfwFe}1c_t7n_AExj;W6YP;<bPtZ_=G)7nq8o!_|DX<uLI9#DToI`(cw^Wy*p&n!*szMmTPlfhu9?IAL}2hl?k='
    '*RAQX|3kX_%k0XOarH|ZujUHDp(v&Mffu{-npN6I{53P`6wb3VGt?xCM^ECI!so@Kkw@^4_9&Dk;#6?FYY1nX=hGwUqb6Z_X4sW;G3m2AEs0f-'
    'u-6fN+C)U#8?9l&rHeKdD=QgyZHKut4phG?hS}J?Hc6fJMfzwV^9t(7m}lC_bi3y)5UlE*W;hJDhlH8ay;xwz%`9Lubk})@A)wYYjv?EmvRPy8'
    'yE9zSqf=PRG}k)a!?L=ARz$Jn7Fx|4a+4Xc>>(h_T9N!+NlD=dDxpk@;okmmgtDl;vD;7|?I9>5T24dzn=9!k$GKHW3gB7Zl)|4<o0#k^+Hx{K'
    'k3;#H+ZLeEB^2BwwXY5Lh;HRp+kd0460#^Obo5qr>9ILVmbN+_n{M=Y4&;_1D4Dul@uFaQ_Z!`GqoHjQstx=3q@wxd6TlYfV64|Az<%k@fUTs&'
    'RpTa&a#P2%9v<SMwF)YyE$)%FU1S<2w!#muIx?+@Fr}LVV)s(qf3IuoZdtvH4In!paq42g9-G-iT67yL-EQuuCgJ;lSTv%Vg#GNe@rq1uyAn(h'
    'ltYp)(f2$VR90z~C(67o^8z5@`(>f|1@|#rlrEhTwO0A&0pI3$xHBpML@|?N94@*~<uI*fhNE5r42nozh&BRDAPe@oc3#mxy0fxKB8L4aBVJZU'
    'J55Rn`$%FU^YyscE9&=`IF|m<M>@=mX<^eo(vIIQKx5)*BT+Aw(u4u&9z@6oYZcMFHQf$Ok1Udnuc3&?W#e90s<=mG6|VMqcT@RzKC@l2S+0R*'
    'HQ@}aI1lK)Z&Af=!&#mMXkeXvWZ}TwkpYl5P36SOd4Jl(+z@mLZi_?+P=rm(?6*sBB<k)r>G0>_wgy-hTX+;0I?GxR9DEJ*jvTEx9X0ixC3!y*'
    'A2$u=g3T**bb&I=w}6~T!-hgdQY)0NQQf;U_l=++J+MP5f(Sfoi9T+IRVj;lWD%+|1f5XQQ{{x|1ReCY44L~TUT+JOpfh-mMQw>3mOBBoTh5N1'
    't@FgrJfer&Upo(K5i)Myy71?O&8<apB+eo3P=Sg}D>d?{a^X|-v>H_wm@ZH+38<R?5!sOaBjyUC$vvudlsQI{O@zHhTl0Syd9Bcg0oVg!<o!1<'
    'JZuU;?pD>krUL(--2a~3|1&4|YRSV*K5`)?l;IzCH@HR`^?j0)n}qH)K8y`1N@L!Cd!l_>Q=FvsWRIqJ0jO;?+G^SXIJa8fTzjeyceBsSe(TVW'
    '(Fh_fS1n#qWEM#PPk%~UBDuqQpDIzUc&SPiBX3!xL;MDw@qcP?ek7=)5?-yT@Yt=^oKf-0)0dZM&x*$c$R&bku<kX&Wh~UyCZq6V4@HtI+ip>y'
    '@F9F_I+s#ba@M2dBw}FH(5r+iQP3S}3vbz8DaRX1*llI7)x)ik;|;};iP1~9?kpesi+}wslggxncX^fzGPoz^?ZE)@$k|z5es#ydYikxo3A%%R'
    '{{y^^zF`enD)2W>usHZ&JZ0kadj8%X4Iu4R_5JB9bL}C4U}1cI>Bl*qT1^33(zW2L+%Zcocq%2h_2Y+q+qGE<`PiVoGRdtW%<_hUDKGW6EyK-Y'
    'pX!<YBVb>Io*wZYFLG-k;)m^S-A6`>VI9ED`^n6HsMP*D>F{(hd(C&dQG5V91p;AD^e#1$q+7j(IBWkg6mKy4X`$}Vj#4*Vou`N-f%=YF{R_o_'
    'Rb#nD$44R+@2y1q3mNCw>k)&$Qt=Y(n>wTCc@l_mK{*fp-pUw>$>YTwD>5ByGMOZ*4Ly{-bVK$4py@`W>uj0uzI+sAoQ*k476XbuG*eN}&1Ced'
    'Zt|;gco&h_H&FQZ96XX}gh!Q)GY$5>c`o^H#}NlpZP-o~H}RF*){H4#Z2fb%ie~((LhTsL`ONV*Rm0r#l}Y=Zi1(KZ-QE+*_RW!EX8ImwfJOyL'
    '`$%=lLvYPx%0$yz`_NXsF`PiW3abdoZgMAB?!TskUK)skdG&*M(|eXxs1^M=jCENEMdpq71#Ej4H&3KR&3#7d8EVSet7_@&Yj{!IzD+cP?yboq'
    '$Y`*uwX{7!(*aSaKzX`uCSRXW+{vevU6U3qds&y26Jv?$muY=j0vWX9KZ+|W4ue=rM5}X;PcM3AMAhTcID5G9y-DpWZmkOp{WUvGA*LRWLB3Bk'
    'RHd4x01BCpmRP*;;YCL=jKas?Sofwc=!H2{Ur_f~1<!G{rKvG4we=IR{)r4=0e{*wo-JsEC&v&7D5U-!f+SIOKTvZwqZ4_Ph7#CIAG5V+Baa`k'
    'chNU;>y5+1#9+aaBxE__>E27c_f}m4;YhL-NA=uuyr4;T1k0IBF<6p@Qt`Xw8UPlOolpD91X+Dp1#_`F#aTl31EVei*y23XjOxODD>`nV$$q?2'
    '(aK;jQ|L0d{|XD^Ujf`Hg?ujGhN<Vqs)SJ_>Fp7Ehb#5^WbH_DDcPG%@igeZqrbU24W%6v2I%&d-iM8s9d_fT=b!cC8CW^LGR$9qh2j{J^YL-w'
    '<-lNp6?5XvsFIydQz_mB+~GSg9f3-xgG}q~h=8lwmC=I4=L9i+9O)I~eM%QoNhH)bWLoT`i5{Vbh2h#Kj)Av-LerVo&*s#xm)rK=?1F!@3;sv4'
    '3!pJ!MHWv6JFj~V_?QyD%P$0oT5i1)Kg;Vqn1~CTEL!&nhtD3CgFoc3I&n3l|Gh8O03y(1dl)BOx{}8IJd5H9><Iv5NOQFgqF3qW?b5*qe((3~'
    'UE&N+T+D7o60IR3rz~$|u|p{J-(WS~7?y7<3DX5e!kctT<}7mF8n}Sj-L{8D^^^_6BgWPp=?;qh^lQTzZcDT;^tz8HDQo5QT<qbSRk&fCFC##!'
    '*Uui_71?>I9&r*I8L9$_`b<rruY6Dv9Z!xDr0wW3??Bll-`(CFbV*BJCt%GrCYafKJiGdDt=YRl-5VP<`8+Kwa5>(^`HKy{uB6B#E7WID|F6G^'
    '`B+U^Fw8T$2vMuwV!MYYwug0!9cF#x@j^~v(bU3~$|68amN^i$(^5%P`$HbD+biw<Q>&Jh5%a)L{HN{IBJdjbBQfPtmnPcQdtqW;0FawQRbWL='
    'y!&~wF(2hig7pBDo%;?Uj=Q?}KqSJ*q{n4E_y&KcP(IbjZxtov&r}b?Wr=N~NG?ZFd0FpmWcg`-Jh8h&@Ou%wJV2p(OlJHvfUd6&<6rjVzwF8X'
    '5%y%q>qU^=X;V4$=Y>6GcRtuz9{tUkRxyn`nXt0Pj<q4fg-mZz!`%8zD7O6C(7(qGZLS{Iy=M-u3^D`v5C*QF+?oWy8$aU#p%v+A3u7+hBAw~}'
    'wBZ?eHJZf|5x9~-ZFsa#;HMWEI0}MHU4Cn;tIu4+3Frht_V<W{AAOP)zKrf9q~DD@8ctW>7U6;M^v)o|2|6DTf&1?*^j>22X9Joeidl8N*k1c|'
    'xZpxpxoKIXugo4sBxYn37t`h%RUdSm#HYeaVh2;4k)MdZb4evT?P*kmu1qjS560Hjecu;i%230;p+o1(4wkAVG^P<AIgVBqmxf(3CrrQ<1J-xe'
    '!1G-m?pGC7<q{1&GeGdkk+Wd|H9PvW{Guha`-NFE;LPXWKZ@hI_Y2Vi-nY8E&h9V^zgg={Vfx)~b}^tzhai>^+^rVO6d0#t-hF?*85Y2Jx1ZkP'
    'K=O*<hjkKW@$QkqJ93M^7nFz(?OhtbuK==zVkhhnPB==1lV!oCeL>ZVoFjCBM}09V5$$Y{_3M28Wco$WXLdF+n#Q?Zjy-5T?({mN$nZu|a2W?*'
    'V)b0V){TY?$#GB7sqDk%qqqUWL_sa9SOnbGa<-q*i>td=-vY`4$d-nbAofAoeY4jO`=nF8!3@3CVZmavWgv*BgusaCg&**aPM`C=NpzM$=BbJV'
    '$eq1Jl{l6<0VY=QcrR!0wJ4T(B5@H}r5$kKH|0M-ZCYCp{9UXDU3nhBRk4S+9v_K4veup)nAn$GoBYkY<MO$56641c)@t?j`VFRYjC&LHD=`RT'
    'uPopPtBs@WI;U>h?Mzvmbg@wsoj1_h$oAaoKOYr~`MQ=wXJm?+^RvLBma;=DeVhBGad@q>?Kw@?{N;JYb`1p#cW`l5N)u>wqJsRzLtdnWI-<^h'
    '&#V16L#lG7QV{om=)C_O`w9Bn!JEk6k|yy5=Z|iCyWV`&*Kg*5T+yuB4Vc`Q{r+)y-Sv(^hrJ`gWPiUlHL3$+>vOal@Xs<Fui%d^Zq-x0zxfHH'
    '-#_UC^FvazpL@T>uYW#3h+NKpb<tWD*%0Cm&OQ7kGMi_q+wDKGreVH+6!u00D!<1y@m<#Uoc|^x+11NdTD3D_zaP<!#Q|?*y4bXd7VLazF1*`0'
    '8b?}_LIHIi5;j5Y9Rq6rJC$A<kzkr&b1Q=CJhOctI<GUvLC-ens8&FFbDUO|@P|c~M^<G(VSU&lVG?5fQjTNJV?}+TFMS7pHddHZb=2|UfWVnj'
    'uDqeSoHd+6fYInGX03>FghkLE#9j*)8W08T?7W*btGRelq(<vdvwpRijMZB|71+}a@ZszS$Az*2lYXgT3WXJZdA^5uF0N##*KW%~e>eMY{ZWY;'
    'WEs{Z3figG_aCv{$+@GC8f{-azKj01D=qhsR%zM$k-~H*)Ok6HViv<-JJbi%Q25$Lyp{)%D9Dk`SbK>@+@4D-j-qf;)QiqvTbh|u?k<IDj$(6$'
    'UokrADIygdS%9uD9MO~HqTd>9)5|aPe6swkRs<!8I%KUh!}l1IiztqH49k;*S6_Mb0?GKQB-jr&iF_rjt);54yJH6Xz8))m)k@cWM6CO~qG38f'
    'W|J!rC2<svbG!LGmImik*tpOS;l?ph6WB?#Eo3dyb3KBA4YKBl+J<BGfb>SB2uN;S=XuV7lb3A)<nO<NR@_p&mFwQ;o;!RM6K*_Agb74MHG2ui'
    'v5&g;fF#8szX&3<92PWo%aEINKV`bzv3u0tG4Q`3_LDPk&6ST)kh8K!15EZjcBmZFB{Ew09q+vlibm@xTaPfw_I}nPIYkpF*NFh+M`f1uDcHew'
    '=2ADdD*N*%{oO7N9@_Z*qX@^@ZJ#rzbmD&MLEP^@;+i_a#^>7aPVU_2O7i?YV70;Cgt)J92-fy51*dxenSdfD{ZbCSY5dTY@9zuvtDeud+S8O@'
    '(%^6TR3qI{c9@06tDN7rE+|bo+Hu=WF^@XReqZ~F_gb<)ut$16Dz)#3vE>En`4We3d-Exz>6~_;2jZl#Yp5J$3BzkpgZW$!vyvBGMpSFCXNI(O'
    '#te2st*6&1FFq$NYiRRq*1oPP5Gh{ok%Z=(fn)twQ#ZOFt5f!t@Tj@&toAEH6Wi-$o#xj>S*Qw7m&wHn+R~c*QkE4WzjmMU)BdzB0mf%n)*jN9'
    'k~!AIRYFxT)^GP;9MEM)Sbcxi$&sXQMx_1Ur(F&uoPyJhtR|%fV(M~38fk#Idw&PnJ_dA%{FR3l3=M1E3$Z%VH+r_v9L}RFVKV6C&UIgVbTz-l'
    '^D9->4pQ-?6SH$}o$8&!3h4cA(X4MKNqrN$30&6FgIP%=+W?q}1hAc14$C#`L(98`VeZ;$i*%P(yO1H%?{`#qA@ZF%(zHg$1}V5!n76CCBw?rA'
    'OL$KL34Uj}YzkSGzgw#H$ld0VyWvOem~sPY%*0uAB3<o6UqEf&$J~p_*M2h%UrU8@)@}SzltJVE?`WX+FT;~>$Np=l65)VES<%i|%l<+uTt;ZJ'
    '#+EzpwZQ=35?ej-5av*zYVE|aG_n6L#wzczASxki*s0EbkLB9PK^o^Q-QEMvTUu2T83LyV!x8(#hqO?Ac6Mi4obXm>?HbLI2zT0V)dv*ZOzcu='
    'cjAc9qweSlZl$n$Kl*OOp=DpzcWV93(=5MXf*`dtU(zm!4;wqM=d_{;RCOBhIA<{DH@LZbqFZ%?Mm@|DETmu_u(5}*+@}~1GN{@zLSi?OTU(*>'
    '9&UTiN|AX4dbWw+^;T@8eQ6MCMlBaG#aIM1cEP8O*rcGOTjj7h$SH@ipTaO-lbG#RT9@ex24OLs*}($|D{p4CA|`d9Gu(s^Qo<BxY%{Z+%I0~N'
    'w<v}v%-Y()XN6{LD_*7(zEy53IoMl95J}h*R<_l3zt5aVqOM9KD(hW${}dV>4OtskNKe1k=P5Mma|VV=0skyy6Kh2a3N5kCZ?GXH6oz?pZ;{6X'
    'ti$cnyI1KaC*@1XVn_H|vffKxX~^n&1GjqYtKWr&^&m88!qEcnKBTvJ>hBWLJ7|s0sL;vPY0NYw){}ZE^(d{(KFTA)<h5Sh0(&1hF?^l)MMTXy'
    '75auxI9=sEKq59^Imp}@w2D`=i1UKXFR47nc5Qiy7UpOZ4yl9IiZrTmT2#<6IW9C%QQ9djqKfQ_|8r=&#C`M<wtH46c9D&+2}Q$tzcgpol(5jC'
    'Aa5sE0jc$*PP_Y@0_Hf+-v!6x6=mLbhd2lcM|(@cSgec)+Cf8QrS5a|)JIETSJ25pqFLBjK7Ma~-;$=2lJblLDK7jgdCfEMs|HhdfHLmpI^Hf$'
    '*)-3%aR2>O&0g`;AIKal*1K_Lmov_DM!UP|$%FO80LIbA*$QMkg^sY${Y&-*mJ}@(T~csB+4yZXo6z*0?Umt~&_RXcOYNINOlN1+Ta^;hwYR^s'
    'D*h<bq_<f4XOe;D4q5Tczkl@6yS8JtL7LcCzCX$_rs15bY){`go6iQdu%;{hD-713^pyJZ7+Yx}z*M#)_J%)s*Y!D^?OI#``38;0t~HI@wy|6+'
    'S!P^p{b~u%=F*+Fs3(TqDXqxaTyV0)u<yyVRr<J>%li(THw9LN9)ycq3FfY1D5_S3EX`ncRZ$6UQTvvx7#|Yp1ySo7{HYrd#FoH*=jif&tM2KI'
    '>^n1=@0>u3c_8bRA!!LgUUp6qG<cp#P(100r+)IxnUIb$<20wnl;Fx5Dv>9Wq|u#3^IIemF-X6%WnYNXrAfXVW%#Z+nhB;3=-?Ju+q1nzRobDJ'
    ')li;k2P<}xDyq`T>b-|YbK^Ubd}+Gg*4~#OXERw^!7S#Le$lriFInf_5bIn64k4&Jd=>eNS4p46-5#fTr(#R+iy}w`Y1varysiUnC$oV6#;_|U'
    '#6#j4nnlJ<RA@~N@eX(lvU>Oz(T<ER3kYj<g`H(5v3=e>0CONt-AtJ8$CF7zeF(1g=c)dlb3==hzouf&yb%TYZ{NS<Uy|{+plG5DCM>m`Pa5)u'
    'wP9g~g`NH>Umt8Dlke*0imU5O3r$175at!>v#PeAbk?a%8*5$f&+cnDZ8u5UX-R^GziLG>4{g7KH$EO~*0$I<=2<JIkvt>#^aA!K&a|W5GY+}c'
    'u2u$gayo!AuNmw}_?<p+!&du{_Z0;e3fr?5Bro00_+a4t4RT}@$!JYw&AgK7ce23{WY|e_n|7jp#ybPS2c$x*`PnAgOOnpY8fw;h+Y)KonKrLq'
    '6vp=OgmsRX?c<+xED>lzQQQW_#`+uD=s0W?&?qq1-oO8FK>7~6EBl2`A4=_H>-a9D339B&3e?{ZX;PNt`#ccnQ--=nS{~6h&s9No9`<pEBF#RM'
    'w=2W02-KD>Oq#UYEngKa6zPOVxLsLO2DhD}b9q!o#P7!rZKbuJUDP&A&DOgP{*o|HG;!kHV*7rxNIi;<aFB42$s@ZCW|yL8jn_`Nyw0rbmek+E'
    '7wGtwJDd63d|+zzB~(WN9*(LzI*W^lUk(elTCE+Z0oK;%!mHn*=#v(p++lntM*0RiQ<7(Y3!>GDmBmNK?{dkt`#LjGO#Ul{d+jv=TI_cIdG-$='
    '2mbC(hI7ONgJ9*8T2t-uHN6yi<@x8^`8UfNzlDCbQzo4h_X&yLYLfiHSf3GLn9n6|8R<izW@q`na*^nPB@G?^tAp!Zr{J>dqaIuC%LI&`#pfR1'
    'PcxBP=}p9?oFT|)mD_oewEM<xd@TUM!<?hV6Qo!M+_)LPpLUdOOZvm8<$R_@f#l=xWzc3{{7eB0p-t9G?fj)2$0P3>U&?BCu3urfXX%h8bU_zD'
    'B62Dp0?%mD%%=df_|huuIV>lo&6NOF6pj(7O4;3CT?mXb>o(Tinl39ka<7p@45SvOj-zpQIAS}3)WH%m_@iGd?I(=dZu=wN$@a*=viKcU*-7|7'
    '8uEF$$?-V|Cud-d^3MKcQsX}5$Ubg=4pK+Hdv<=?c6J2ZChHlHx{JLnhB7JO1q^^BFFWs#CZm#iS|p)yz~qu3oq{-UNBX^mGQA61bvpBZ%a=9m'
    'kFOjg+qrq9jBALqOTk|EDRYSmQ7)EMWcZpA4<PB`uc!Daxoc?5$Pi+0*Vq_I^(q|*B}-Hz^J1|&5^p%#?>1zC&SE<U4tFN2ony7#=a2RDxa$Fe'
    'k#~!hj;mV1@;Kn$niJa3b{QAQ0K?jie4%olddg2bW9wxwgN~Ufx_YQL+yEWY`)NkFx<CB5EN^<iFUr16n%B(b#%p5P&I3IW>8_wVb!TA`_%O?2'
    'R5p?fO-Gzw3a_6t8CCEB+;G}(50-HO`^n`$_4f)YWbIt;2$t<cy8NA_IxZl19fp5cy}gGuSQ+e^E26+<_p#%xSTE)6Xnc*<*?v5s+2xS}du+h`'
    'd6f`?LsqZ1q$bIrFWIHMC>_xxqfgw~Nu+Z$@=A}e4<487`hB!A;e#iah`)wx?sP>)3mZs27X*NUT*88KAe-I-O`<kyx8TUHP`KzhqeRM*cnC>i'
    'p3qFkH;fhldN@)Uu+1cX^X#w7Il!`guq#K0-)8H7&LF3m5N{+Hbl^>U27Ikq5n}JVNAdYilvDARDl=>-r^X*(D{rRd>%RovE7&z5o%Mj7`^V`n'
    '&Lhn%sFK(5djK#cvNra_s=5Dg<GwrHh9A74R2g0WO$SG+Tyf4eRXTfWVcyYb%bK;tK=b^~oF(~+c#*5s<Ak&y_w{>QtzT}M{qxVXOMRTwW)~k@'
    '@AMl5I6X6~&n2pCw@&kbma|xAR#46Wk^j=?Z$S~t5a2BJ__&buUZr%OzG}(QKjIls*n^?}b$rqHBV~}&UJ#7$GZ2T~g~jpPH!yb6p_MmP@n?Tr'
    'L3-VoZR7>(tPBxO&0g!T(t5$HvP<YN=Eyd;mtwV34Udg4ImAxc*<f$+ohS^q;g>s~L^{0sA6B^O*q!Q&AAg(CmH!~ItX{#CdaKJs9D2H}kOD(x'
    'q~Pph4v-F?6orYM?q`@Z|ArBC)*IhBqTs2@)!^mnZrCHHz}6FFaZm3Eaw=1T;5y{ayMBVu7)bB4f8Y63lTi2|6^MY!Ppg33)LiXW&&Z~_Ca<6T'
    '*6Hoh8Z|oBt&!?S%;Bs&;xcBs5vq2Xz63!(#v^;w$C1CX5FaR3@9-B1L>Vi3gxMVKn`X@#7kz;^S~exmsF4NM4m^ga{c0fW|5?c?Mc-pKjjA3y'
    '(3U{$zpO$JX1#$qj5<-PtdO*bqF@riE5;-D@jpw^pK9DLq^;d@rJt@E4MYM)fT}^3?AaHmtK#R>apU0`be}8r%e&`U)?IA(oDRL+a=K(Lrbul<'
    '!P|y}RH4(@MFKL(t{G8Y7Q%%@_Tx1PPAZj)+ZveBcNj5k&wlH=k;CTLk3I|upVii@1}a$Vyvs|6SKpn6CqD?UmhL>KKV9bC_Z`=_0!<?ICcYy('
    'G=mtD4VxtK>YY0=6HYOYrDS->{famZ8|TWpB?M2xm6H2~InIy%G!D6KQ%I|vKsEJ`i+uayU<bvAujIDwLI{wnW!--S*`zBN#V!=k{odF1eN*a*'
    '?Fv$1Ic6Lx^Hz7Lp5*;MQK@z>*>~I0^``odBbaRotnvFs#+DDyFr5#a@ovANI|y$kDJfAUMbmI7p_@5A<EZm70DHAwYKLU!jz}r%FPp7iWRCcU'
    'hiua19o!R42{6=MY^HSe!SM&1n6qG=PnG>jd9woka@qRsflmxtc&DCT=L^fsYEBpxYzi`WTCgQ>>A2d!87}Tq8L_x0|1$?4tbJ#@9aG8cwl6B1'
    'vCp{&0zK-d3P88qf2w_|2k^5z1<BsVd|y@joq||k9?jxuLV{Jc@A}Pl;71eB6l9W<zp^oA-1ETMBF7X%)rWsAIu8o2%(MIgwE`n?T_c`5<a5L@'
    'K^m~eTg6PsRjO*tbDeUIx(PaIp!s6jVaS&A`$sWNSS{MExDNFdnEqX_R;RSRD|T8lwt(wQyJ&$@<ai^5(kU{hP@oAP`L^t7vNZc$p8P0w9SjQ+'
    '<%m-e0`=5_5iA%Mt3Gxpv-J@MTEO{B6ezxTYiDK0u6YsybF@2-e0aD1KVhI0!S|1%fIiT7eqa07<P@k5tQg1~UT4|q5Xh45+`2o+IzqZtYNc}8'
    'c+`S3-8pcvucnt3cz80%m?hwmMpEzltN6=vqKE_?!N$e<q;ZCF<b84mt~(HbV^#AJOLe5BwFbMJO<LLUn1cFQX#E8<lJb>eTGKYY6mGv?9LY#D'
    'mSMkT^-@0gcf?(tLCX1CnW?sir1m{G0^@K)z%c^aD|BMc=x@A}E0GXmck6=lrJ}OvJk=-lqg6WlSNDcgCb4BdyOvj8$DN>}O0ick_@mS>d0vmx'
    'h8X-7QN8wADOT$GwYx>*$CBpQpPAIvHiFnpx3KK94B_*9m=I@~ZEsYqIppEk>0sa5U3mcHsO2`LbRZ>g4o(EUpkS^tfmzc$AoSu2XM1xZTPeX#'
    'XE}mV$>CoDBFbJiuU~#K3+Hk`L9URu<N6wbd~6y?U1`+;?u-j-M!=9S*b{B@e1(1eD9!;hj%x&RS4$Csj$QnDKjXY_ao>78)mJ}e4{&2X_8wp2'
    'IIaPS26zVR|Lu=Udu5DUJ*@V&+EAF~{p-)3U|jkebnV^Or*EQ@b2uU`V|6I*&aw`MM>@~M)!&HNI5Bh4h%`0^STSom;Cg!p-^TDSo0nB#cH!@Q'
    'zY@;CyU=oCT%NJb3uc8d&`SY6zWX&mEUPgh{@~X}3!7I}j~kaWNRV0r#V8SPr+6N?XRS=i!dk-95u7sI>Tn}+mC%8Of8ZK(HdFl)3ZdACzUwxh'
    '6$}?KU%fbc^B4c<M`XYxx7_g;wwAu`^?`2nHJvO!$w*Ety~nY=KRMsyI0Mp1Gcesc`uCYhzNo|WL?=ALQH~xa8l9;is>?NW-zv7eJ5)3SQ8@wM'
    'aNX<JO>{e-nb1>OhtIKFL6@$LQChBYqYUgz*duVokSx<zI<g+}Irdt28t$~B86uM#y~o|oa085Svo#6Kez^>OAe#1=EeUL4%Cipg)nh_n`UG#g'
    'wH?{oiECKG#C>-D-I;0YOfLDQHnmGY&az4f9_4)-NFpyyvN0H3KFIj5DB}#5bl$-q9)vxj9Tw9rviz>J6M>M`>^JX$t<MNEviNtQ5IElrr}JAp'
    '<PIwik<z1on_bQ~wsSFa!#H{*S3NU+jMj{+Vn=M}TlFILLOgvugDou>scbukOS|)B<YM_S_sdJ?l_DLyZ|8EoOf2`WT{BBBfbM#r@1C6}XVq&g'
    '_Kx`&i#HiWDMhKj$ZA@?!!khhZOLD1k6P_35C^bIO1WLmt(}hoN46h@5-#|o&6gL~EZ~YZ?C)0%u*sQ9*8&WrE69yQ%fLkb{Y#)DSWOU#jSdfl'
    'dHKD&oEbCat;wjMcZ!7t3%+!^>I?psqmTyRg1|3O-=e@_VVflb3W3=c&Ddg-Tt?6h+P}^ImU%Vc_QI8_h|=z>)!1`mPt%Pk@oY~=!PjfVv~PRY'
    'kh|YpO_GZUj@kIie}ISWPY4r~wMAIW_HRgs-RElqd$_I5TK`ES;{ql!EH3f9+c93QSfJ7lw_vM@Q3ix<**6`6OuNh6v-XnHBbxO?hF_XQKk!9m'
    'iK!@lWB1^}f+<;#ii(1A*t0T4=SC`rHL%ekz@T=nYlm~a!wp#~+&G@z7mD-HNeR0U!9#gXjejzghMgJbl4jA&*D2xE_GvM#Ua`1(ovPs>JtiUm'
    '1*-04{ftKYiK2E(+PAejDXl8MHl1{v5$7%;)NDJQAxHLA>KNDIZ!LOhq2j(-yO8cslx5XMCt4&%dz4UHz;o8sJ4a^=q^VvrDM{<Qx1h<a{{QBg'
    '6jb0|=NG~D%!;za`O(BzCE4n;_&ODlr#t2J+B<wp5Y;j9l-N8?yK1RlXU{!Eq49`b|0_hF6WQ#hPdgxbj({cB?~^EgYwEs96-e^4czmK|b?R8<'
    '&qjjnR4(}jtAQ0D%sq$4G+)s{!O~oA+P>DW{<&fn%weHyZCbw$T?E$iz+HG<?Cc1japdal^0J9Nk0f?vyReS#g3B24cF<8wJ3+=3Gd#MlbXC2@'
    '%F;1fi98DIkLSEg9wLla{f|z0;sI8@BaUWI6q2!Z{`U&eeyyl3vU{o>-}XeDVyD0bm;g@-6PvMKV<IZ&7{_Zk+7<97LJP|8$O})p)h9=L!At76'
    '_51rnHA`g%I01XIiGR4mNbc+dhawn(ZE*VCx^~~t&o&OAlC+jHE_b+_k4tix&;$G9N~Z=N%DirOdsfaQ{%76stUSoO8(qM~uf5v7a*u+DD45>c'
    'C1t<qbqP8|uaqSM8MG{IM}HKz+TG^aVdezeOQqxS+>y4<iUDhy@8F-rwbkynVY8&#(N3+cFAs%Pn|2;v6Tlw!?T!^W4Mua@+xwZfuA&`K!~2u$'
    'kfvL-Muhk?>fP|z)!WW{%7fQ;dvWQ$|KbF3eOwYWk7PRVFBW7FV1FALhVq%`YW>0Du*tY3d$az|hX!-M!>Zbj3j%%eU-0|jw;z<EKVKx12t}|D'
    'k1+Vt25wC}kJpgv|Mhe?aMoTTeG1kjWVhK_lJ8oo16;j~sj;?ohaq2)za^!6I`Ul}QUDt-HgG*)+O92;F3+el5!7_|fE&)^iXKPEk@a37t9|4r'
    'q2+?ToPrG;Q678Tzr<cVyVGx%LnvQfSAT{x+aAppeQ3b21>ed3z+TPsd@=7zqXbi4tE%n1xvSZZWcDs2LB!X3QX#%Jh#Q9RnM!Y@|Hr(?hHe7O'
    'jw&k#v1{M8+2+W5FYoy+Hr|gOt<fB!n<Z5g(EKr!5JZ6$FZ!yGHI!GG_ye6B%+#+PfrPYn5ljD|x529qHbe}d-3t0NE=(DS@N35YT}FH@F)-@z'
    'bZZ{vAndON&kyS4-Nsc?ekffZ`mXhKe}MBLMVMQ{>x!L)I@bN@{(g4dY`3yFbH)}6HN9l!SPD#wz?Nx3&)9ZBdR*tWv)*mIT;IQ<7w*L&(<TrJ'
    '_4amT(UmevG@XQ%T!)-M3@g(e&U$H8XE5}_kC_h4)ZMUrp;8qS*S&vDJ+u(OIqN&h&NCVOcb;cd0k^t^o-L=t?_MPPW+I`!+N?!${|t7^_Y5t;'
    '+ASBhT_FeLaksnp6=sT1C)UW@x6m$m*&3{NG}e^z_F9i<F-#KFI)fyK$Jw78*A$VPzxjctkQ3kz*2^<4*kXx*=bQpBJUbI&Ir9}RgMpoQ5jb5I'
    '5nEX|C#pb~2RT~p{3yGVeJLKyWA`T9<-*q;ZMVyHS=#v*85+oux0B}BCRh}-n~MII$BmB>bXfrf%F3Cx%ebH*RH#9!qNn`Y;a5-bV@EKyMZwn<'
    'IL14wZ6UF=OQ_$2kg0vgh3I#pfG^xP=?x`#-W>w0-gIcnWb6FatnpSE#pAlt2FAxP*UUe|*U@wKat45p!UARDkbTNpDX^n@@CMQad1JG43<VlQ'
    'EPaWmV0M6{_n7Lh`sl~*CIh%}VpSdjvk>35J;j;>L+dj7y+zvr)X!lM{i?;B#)Mu(PV-FEdEEhxsSS2v;|v?@kxyqJ@|XLuArc(t3LY4e<wGtx'
    '>}&C3!!|qbFd>LOimGv%{l;8;O)pGan^+3$;dK$*@1wm~woHjbWe?J7xd~9r6YC7_G8#4Ce_vzH&X{tSL}+E}q{6QZv-C|@^jHN?_zbsl_GYr$'
    'O;_mx>!Ax1hJ>Op5y$EY!i(-85}&*H-b>2d15vPO<Ivwt;+iVU-+7ZV7ttY+!DSMYxsTK_`7v6A<buX@mSCv?_;$Wmv3`K`)!q^Fe?8ct^aolh'
    '?dL)HLq|<iCP|(~O5I45PWQgj{d~a)yrX<wKTbJRqYKYY6_8q2<j<cx<9a+*bgMNGur;WRRT3YYUm(|~&nufoo8f!sEb2dqB#_P7@f=ve$*=Ta'
    '!+Rl52M~@sw-_<eaywYdYNY<IKN@WCezzL_dB5;r-@IR6YP4&@OukGh`3ANpMu<@e2gg8LRUo*7jdr>vGI0Yb(C(8W-i~7)yDuu_c6qdR=A;yO'
    'b_t!TS3m8)-`D5>0RdF+z(0<1zu$YSL)@=d_xmIzmZ+ql_g(WMq8Yr^e)+_--DQoyc}qxiK2f6&#kKW*p6KmL%O3H5*>IF`aSPeHwz_Z4%Id9-'
    '-%$+TjKjlaQ304gw*&JaljVfmf4y<YP4;bAtp!Ckbyor$u4xC^jG6!ZFf$4NrCD5uSz+HwEn~<2eEXN~@<NbzBIEcL@Alob@ooHPWdw(RBUANG'
    '7ZT%z0O93lSZ?`Vi;UItli%^CX@NTp7R!lj!`yG0D%+m?-a*q~SUXhe=)uxFI~p;CC4eg<fR&9D*|^4ncyL?{0QG09M;-N1ntbh);24T$j@qxb'
    'l5UVD*%{lv*Y>qpMl`W5^%F_Dm3-yOXY9NTPEt-6j>CDKi*^xHrS&7{qVUKyC~Z+7^!r|8aUkNU$hgXf2EhlsDIE57hAm$$=ZjL94rKiuQPcC%'
    'EYx#J6nr1+-P!#E8`bH4m=g1K)Z6c{CE$)BY<ozgHL_9IK`_f59}X6tQ-(HB?q)41@<Retx|2QHtWvIQ8gpdXIm*Y_tm{vq$ov8NbGcC{cGTwe'
    'd%<oWib+bXGd6JP-uY+sA$Roql53gn(9$<<7c*+_p@)>rKp~Fjqt+?^R02--IbQ9zb;tglId@v|KxPkay6YZb`sPC17n}L!3RKpf>wfImSVEXm'
    'wySU_;&ZeS{nLR>7a@X>a@wcNHzB9Z5?bc>n<`bF2PDm#DTW&~VW;f>alXbIVkgz(diNhy!Mdi4GK?Ni0HgWl{E#7Mou#Z4kky3aU$}#9$m2O7'
    '6;&ntz*MP!^r`S8z>xIwZ=J4X+>!~it(p$h()?|{iN8<usy)dQ@(+&R*0Zxi@5AIT`TfL^aR#*S<#?l;g5(~lRp$LC{}H<66Mp+Z?!=Wxt<O7y'
    'Ak8kPXpFl2+Lv5lL_5~Zh+}zDOa~KV9^gD~cGU3r*zu9BzEdEf@!xlq<7?pa?Kr^dfAj$jV!AMV?e}A7$JD%HYM&Sr#TDNA_KI2avTj-mBH|&2'
    'pv)Me)ftRBvz7Pb(q_;iSyOQWpI9%VuOGo5ek>OA+`%-A_GsS)qdzv*fbA7#C~?|$0XsWTLQ~wRq0;Qp4Kqk1nyHD1h{=!;4nKa{m#Y!)PZ6T5'
    '4?IqAqa(&pQM5q%Z672??=zCMet}%nCWQ8yN{*nERJ(Y@Us`C0`J`0l(;*eixARx{^aJ^afQ0-eONK0D`ykyHxCHIe<woMjn>WIgEHylPJBo4O'
    'sIk9EKB-CSvd92jAsNOn!^P@nMeFil6zA!x2D5|q0>L`8uEEv&V!1AWo}3#O6YIo`6y*_<MZD70a5{5Q9OG^^p3$5YB-PCQDkM=ftrLfUMwPXE'
    '4-Sn4`oG_xJc5NjHf~f(tF%D)>Z_PH-G%=h$9TOHfeflNz$ZPxWqX5KzYw}ir+9oSES+kQ=DR39C$V;CRqRTbOUBE-(+jM*(^X<0Ag@7`<Gv(3'
    'k}Cp7%SJ=T?V2Qqn_TFd;zBr$eHsCLpdmLo!Hf2zh~c|AS0E$uI{~<S1-@{?HA&@!0#KS7>DEX)I-;=o!*e@x2Qmc@2$s*6RJ14$3lA0!{0pce'
    'DW#k!MYa%joY$E~ur-5sogO{eJHi9}iujO$cITCLaH4<4ye-7U)cDs6F+u3m8LT&~L$yZknBxOA8J3e$@(8g$_HIm>pP?q0XM6ShG-pZ$Q_9UB'
    'K8`P+#cYPMT_T<{y7XWg2Sm8OH$A=%0U1%NNR=R^M&}!|@LgLLDyY@6M~U|8dQ%JjRuX9DmO&$pRXI_@g*959p$x|k5w^kU+95wkHQpA(m5Zh?'
    'RvYg6$h}^~0irHjJsoKC=eb3=dnToJBTgZyBkGjjnRMn<Rv7SRM_Q}t670};-Na^YMrejFWZBo2#QF*A*RI5o?kMklDi&`MMwqklwXZ-tKidcM'
    'DmwydN2Gk9b_ZPhdYqLzVs*QH6)8*#gKL;X9ND2w&&x-D0ZL1>uCR{<OqFEJqAQlC8N*`^@PABP?Vfu_G`ItV^J-tlsy!N{TSXO>-KJ8h^0pEx'
    'J3grUb>!%2hBR_cIzBn(=y^(I2l=i>0%gv~Q!oEa$%e1bN{X8GLA+PW9|CN$&h8Qq3sDUgDuBqX7TG=BNsOIT``yPm@Uz3;tA9c@BtIvvXP=^B'
    'O5<vEj|Xu^ajT5zukZ$HQmS`cfz6x<rbxpxYaJpfo0H6rs}oz#z&$`Uke*wH;IZvkXSaKaM;>h#WxU#d3}xWncnr66#(<H(rxF1ooMnBnu?)Fe'
    'TH;J-Br)HaNrE0;LtUE$#~!W~HQl(BVmkyUNdmz?|68}iQ`fT&PRJ3+fyB&;Uwg=Iq-}SdvSdJzdODtHq_2h!1No2!bY)m(yWKLxPQg4wiP^+A'
    'oz6p18L+=4KR(^?okkcuF_;q6OVCU_6i122!A;Gz!+@X+{`iLckjZ2Z=eS~HW-hOP!;k3KS6*+2B?|88j)=><!Ard(h^`M7=X-dQ{(kVV_NnI!'
    '9d|eU4Ojk>Obr)<sKlmzhI->dwZkHZ?hF)&a=zMQ3)1GSec)2{K&yoe$WGjG4OJC1*{F)?fmIedN~}7d=@i%#&F;QY@e~gw+QK6x-{-yJ8h3f9'
    '<g}Jmjyw4i!pl(PX0&scIC84oZRsS!EA|>78Z5Cx^WoZcAj^S<a2EQvEnml^swzD)xLH3*;N|HNyonrFo-pz{@owMGpgrt8-AFcw=O;we*G)`X'
    'SJ3EoBBaj(+8vs>7_Tte+EG%mgH6x!ZA};YZPOEyPAic=7RU?i*%<2_{`~KxtCxCzl#_GAO`4m*Z#tUwv@k-C#u`9Q4sy#T^z>f$RjBY+jw&?u'
    'LA=S1fDcH5cfZv!-(fil7CV9m<bV_z+?f8~6f1jrd|tlv)v*t}kH|o1C)$x8T4UYu(Fw@BR1XsjHkC}Kp)lVFE2_au`|+5;NQH?8a5F5ukIn*b'
    'bG?0)Jg*fYOR%Jn8Qkx}E>mq>13j>_HiE&&;!;SX-G!*#F6%BFVN|eYj^GQdj;;j{U5C{|Y2R5!LIXT=up`3nkdFAddZ4y7iLwixC7r0lFRC{k'
    '-Ay-O05R7MvaGHi4(9NrC*^*}97z2a+pi07E|AqnFfdtGnw}HDHCD(DYmtEjSdtWlbfO3s+Q~+|H%7ZO4oZ^WKf3-V&q+L89o;k9pZhWple}k8'
    'mS<93&hQEl&L)&xlEoL{p3xILX<!a?wNues({=L;5C-h{QY2KPy&rA{4!JR9x*kn-$UQ>rdAG-)vR?=Y=F{H)V|wxYn?Q&gi)pvQ0w<EsK}yyo'
    'iKGEV1-a|#UYHhnJ$ubG{9J}SQ6E90exs<%_g=#ua4P|RUpX**_KYPNnHA3Odj|WDioztEsvs|G^QGB3b7cEbo)=aEHG`G9WV#F45wyHQI44}T'
    '2wsj@eWshmIbptvS^(u<3%kRw^kNtOqJci|y;bK{50oG#v!DY__YD3LNIrbNv}P0l!}Bn%=M>%InD=n>EQu9PSXjxobj?0&dk{I_c{vwtFMkB;'
    'k##r)3rEW{B+JT5`4xQ1`>~IgDHOcIutOlTi#sUX1lucm!&9C;&KBjrJ;Ep}LMIH)cnvXvrh4>VzJ#SM2uu!ZyR-4yZ(Qy!Y94EL>`RD@KOUs8'
    'ZZBlthLxgM$n>)bX8f|Eu#)C&R=-XQKpLf9zlwXaYcv99p1Jqs=V8?^*C5T0D#HrX2x1HaXIz<z-Jp5qCQcrfy`dT@xmLZMO|X5XcEYC*fMI{4'
    'OT*ZlvYFzwv;AerF|SdBKeua;L;U<?FO}Mz-7oimFG&c*^_ypK7kO&;0h!q?K-6HmS1GxIR$;X0m#(#scb(+{E!A_!&fRHsZcQV&iafBVi|qsL'
    '*&RhL3iF9z<oEX-?ssqX&c45Xl$A9(YIYX=wY`YFUO;0Z`cSBThz<H}m3Q?#(ZE}3N5R!RqPyFhShq{{1M*fz)W2Y7&G#>H=6K%MF+obe=#is!'
    'CMSya`_&NdDmGd|A?u8A9o1Xb7mWU|N2oi`{_OuTcBbiWB)O8^%3DBJem)QT-&o@yNB|_ERKEUI)g|d(GQu5$*)jg}LhDttUscYp6LMdNreHdh'
    'ZoyGV)QIYJ#h!Un@UFrmo3FGA2@uEAMwG>O^YMV9X5R0b!la4#9{w|es=tZ!KrMH>)?PyX%Ou+|1i~v@_SBr*Kw!r96+AkzTCXiCimt6&OI8a2'
    'XUbbF`XB`eS14}c(3sZxi=d)g!bM~GeJaNxM=Qv-6|TB75<Fsgb80c<COx9LrFyY-T_dG}Zm!dS*{xFfAq4dYU;d)`immxV29b)k-SNgWV9e#%'
    'EGA`rmy%(#3Ij>&y)gncds@|SMJ(;gh)xBgQq;4<4sLc9GKcEp^fasu8vy;G&s-+dwh-^fAgQJD2jqtGIas>1BI}B)g;iNkoW{)_j$4_;j1ke%'
    'Qoq(f{AF0ySC#-=kxuw{YwGE<TQG2KthQOcCG!;AXFH^tJXvbfO=qk%TKYm@ZR(b@sBZ|u>=5j$`rS<a08LS$M*gi1vN6{UnVuK-NCtJ5s^XGk'
    '`hKPf5SM;$g;em?lAwa8mw|NX+b57$+v5uv)JmhYeuXt$Q+Twh`#-cd_hlm>7%8lJYW8Ic;ISY4WeNtd_wq1X-7T9d0v5kEkzq`Sx-I-&s_3~l'
    'neah8>19Srxs<^x{K#rw>wEk?A)bA2g;e3Bfs3R=d!g~2afgj?_)dk_u6Cb-5x2PKZ==doi3hMuw+3|-c;VLv+urKQ6xz~6Ny|RwJ55GueGzhY'
    '6ycbLIpL;kr`AA!8(6S%faP~+4;l8yGs$YFL24zguCRohOpvZ|fPpezF;XpuFi0t6_uE6XY_nu=MskAob@vMe*f6*j;>IOyK+9?bw6gVj)&yZG'
    '!`$VJ7I@ZYHkDvlfmg<-JVY|skAb#l6P0+=OO;BXPWNXJa?)$ahFNfelVLx&o>9W_Lv%%1?N4Pg#5cR?EU}|k5O3bv6;haM)>PqRFdp6ek8f=A'
    'zSW_9TCKFKCxe?=LrXy4EPtOHhg8$a@HGQSkk@H)g4Z7>Lqyt8Z`J^;+idh(vf0^=nD<kay2bLAbqk1Ev1O)ze$Ewgr5@G>`_|B&vK*T=yJZ?_'
    'Fg6B!5+!SBGD!$pNqV#XOe6#OyU#V*ZQ=Intg1#&Wb4We;3Vkr(D_KY7qpRbo~>-XirYupravxjxAf+%_)1b9?>d9}mXUyi55K<)d7sr3>y(R8'
    's?S{Y%V|5uT*#xagPap+<(aO1elQEEd=|mF|ByUxq;1XcX|TP$H=ihs$k>-3VYp3{CqzE$l};209J`J@r<zzdd<Fahw%$WyI(_x>H4(wAdcz}!'
    'ho<@A!2>u(#xeg@;u7A|x^wmyrR~VH<*M@-zHJ5UW;1y$f^l}vj2+wXnpOX5P)&q%1uTL+v5Rt%DVwdBAe>?SIsxVn1tl8^y!0e;6ndTlqu%z!'
    '!Mt!>s$BeCS8#9b>v=(s@Rp0LhyGz_>}npbB~QRA-loV~KGqEZ=$)Lxda_*Ui;-Y@N+65Z>6~3wE^KXAW>(_P>wCLDkYLJt-AMNFmwl^bb~VAm'
    '6Y8zvdmqn}Dp`6@CcbiS$yZ-}tS0a9dJcvDxPhQzZEKdnS)IfQ$C4WQ=JqI62vci?i7?ppnJk!ab=7ix6^pV93&W@((q;E71nVbforGTKz#+O1'
    't?8~cbrWAIIA8h7Cyry=ZLr)nr_ap8%}7E{D*wuGrt~)cG=k-I1pw<RW!i-kUT7dgB@=uy&Oh(<4>K1=Gi48#d#JU!WiX;A9?Q-uA{grQuD0XN'
    'OGdE!iHnSI(n}#~@;YAV15RpZ#24vj-r?(t2q!-A)=*x_8JUFaFn5GY>G>tn>7(v;?;wcykil@2PO`^;7{QJbIPcH@pu*|OU`iK)g!?%foAdBj'
    'kte=N7T*Ac2tkqk>V%euYm$3@Ca<E_7%D7u2hHGPbY+!v?l4u8UsWEZyH8+V7m`1ZUyIU%E;34zPibp7WisPhiUN5lzjWx<Xm3Zm4nHTP`q#6}'
    '^b75s#gH$JexA0}f#b+Cy%JI90%Ja#i^q%HRHL{xkTKxgJM;I;`C3ec+MhaN#Z;c5eaFoPy7C>_I)xQ?;|R4i^;SPwy7{^!x=bJF(l~;5h->Ov'
    'O4271fi;R}gki<l$BRsU&9cf)H;Z)4+ZWnMl}9k+uBK5Ii2|JM_84x%VUu!~`=l2S|31lyP}qFGx8_;|GiC@TtD({0DNh%Lwxz_(M^SHI)%i#1'
    'HhpIUPxjmH6j1^Mmg=wumP6__r~3gn$|F1a^7Eak?XLDQ0TJ7I;Dkb)j{}ww1{~8#D#&r&gqg(j3Y6^i_Ww$`FZa7`j0wo8NtzSmIe-+tg8xW~'
    '(3zuDQS+pMDey;{Lum&wX)Ug}iO)-Aw&1RotxYvR1sJxAx0R)^llXM>C${_MA0dXRggwrsyhv8^?3OEOD<)+s?6>C*a2_mRSs)FEakm5_C1%Je'
    'Yp5t42<;4*3Y`me?%2-OIP%5~pNC1m!-h5qP{h49?Odv4Z1a5BBuSNz^`_9@L(X!w@ka%pJhpX!um#S^(W93L?NU7yT>hRhL)e}2vyuTev%HRq'
    'Br?JZY2@$AANPt>XzxMQzvlbFSz%Cq{iEVhy*1M?N?|Uw-z=Y5(f*f_bb)y|oicW*irfan*ol_IO|}0zk_A*Xoi0rXHp1HG-8WYJnHZjw-%>3B'
    'my75SH{fuRKMKZ4!H`E|hHW|xV%cv&Hs)B4-Hh&mk_P->s5pt&Ev^_R2yg9?NG=zrikZ1^X!Wv3fJ;k-vd^B$QQnApo;FoPWkGV#JLL(BRz+5n'
    ';?=YRY*7wIbl7`*&zK5*zdobwXUV`pa#jh!hY%Fw?^_f}mimZr`Hm8pOA94Wm}e_1e%(xC1>ojYN4)R~M?KZm%L6tqwPvRk7XIb1_482gjG<Tm'
    'InVc)DeMQ)@;m&>_hMoiKCu7Das4_b<t<q)EF|ipYqm~tq9v9t=C_n$Q$QAtsV`X&OK;rYKQyRmO$5s5R`MB+dp(w$?=7@hKvEDRW=2W%SXzuC'
    '>l|PjV|vIrUW!Z`$%j=KwVms^WK#<)lg#Xedn_X~5!g@-+p1wT=Xkcbn!V&<nOTv+mbmT_2u%c7n6yYy3k1hyGF`yp#6%X?UCgm=^8_!sjpXP|'
    'L+fda<mWc2^OB-assO8QI{m_%s`c>|Edo*vHN)<fG|rG|GVw1btDK3RTF$_3lM1bFXGj**LOBtNJlfiFoL?+2KCCiBR6H*F6xk$2kQ|iWY=cf+'
    'V_i=|X9%ASuyBqYdxI#PRq^WpoTKWc3^hgyPdCROohL|L7m+e6DG8B*zZ5~3!8VrG1}&zm!ABn%@P@FjKAloCl${lUlw?JZo!mmVUuFE7mKg}R'
    '0ou-kd~Y+qg{1GOJ*I!HTLO-+Vki1zfI7p^WmDPn&#SpB9}=c)35(F=oL6SS%NV0Yi~Z2}3R~uk#_CVxn_K-y#SGf5^;Uys5}S0SrKl17NX59z'
    'DUK!}Cj>%b<^fs^Xj>5exUTV$NcLrVq|ZMpB~<S|_-BHCY;<+_`LB)RA7aTUBZHkVBVOjwVm{G`WN;M3&?9NJ46V@6f&0qmdo~Q$qMecT`{$`B'
    'ZQ#sGDR5K%IG@mnI4Q2)NH7Q=P)B^U!TkB|kr|fbl=!LBw3)JRU0O6b=jY9mR^|FTb*58dDfMQ6Z&ea^Edin_$Yu3p+bJgsd8pUrZFMR3)lS4d'
    'shoGl86yzqDPUfb8Mr^kH%rop6HlFxfCylh($-qE2R9nxF!7L#k!=dOi&}RVbkqFoYRB#?NT_A`RR_KGv7e9Tsa1#JN*i{~68`Dqp9<0Tu)DMA'
    '1l>X@oo>&RC?LNsaijKm%D=}qQ`~(@K6KajynlHd&~>*^K2$o{<GgNOs8=V}zhZ6~&CmTpel5+Syk^r;TscJX3_J&XZ9HtPhZY8OvA@^<I>2iC'
    '<_Z+X!24y`)elq--Mx%7-8zwhIsocF?9%d_m8KAf9$g)7C3h36^qrJDfDwnSzIdrF8FRzF=%t6m&CYAJ)KAan)`Bbs-ODjlvgrnr8C6W(^vQc<'
    'B5Rno{;Phk8cR{%P(c*21o@`PgkNGCMq_JxXGe}VZ>=Q}655{pS?Hn{%8`u)H?tS3ex{-=>pe5|!!5p#8kP$x#<G8vt6IK<g!@~RHdejpCAwkR'
    '-dDM`s-UUASlR(r<@)rL*+aP=&WX{k%#sq*p(vlX;aJlT6(sG0SEN=Y?Ytombf3{_Ld>x&LAc?IzbxtN*zOzx?Yocs<-cey7SN{dCJtn+j^v^)'
    'u9jm960p)}q3%41wv#m5a;=L!1j-&)oegjodRqLRZOpQLHlhOKJwIeNMB%lQ9E%#|V}f#WO{8Tt^WS;2>-nngXn5vwK<h#`?ZDPbzF}fprZ3uz'
    'xlC^gdxN!^ecqKJ$;r;AL7>BMN~v^3Q=b3nus}J(6A0YqT|aVrIBSFVVNCyg<y(d%os1)qCA^}CU(Q82tlpq^m30B3c!Mf%Ky3)s2~B1&c|V?<'
    'z>hy2S=`7=rckZFayJ58R`YL49wRA<cPi*D$_A>g6zj^3a=g#RN<3@LzE-(Hr&Hq<nl!g1oKOJMirsAZ+G(SoOZD}ZP32nnS$qg(X*&sQta2Z('
    'Hb>AlFZImFm;jK0EqU$eNya@vr6$etYbWy(PClN?k{!qLYQs_t$*71Cg63^QKuOHt9l<r05&NWxzh;PUEw6UGI|cPL)bl&y$fP&y@3GZZpNqUf'
    '^Y>X;YP$?!X;MuwRtx`yltqAa6GyBDa?KusQRU|iUlF0;r6txrj70LeJi3_vz)m|+>($7*Kq%Ku3<@--8s_OSeV1`fx7<EN7V--}zg7F5N?<^c'
    'qS20ij>7ilFl$Y@rU8;ZpN#_9vv~M|RH^T`_4&`p)RM#=rnN+*7f)<suivi+cLQ5oC39`^Of}20Wd_9qc!Q~WW7?`&5Sm<5$J{jywysp1E(NTB'
    '7+1~Yv}&oX?jhsGGv?rf6D=bdXvnRsp-*l#N1Er6H0w#@PJUcAtEC666%12OcQ?SF0)6|ozou<Z4pR1V#nZN#`K1}S#KFCDNj2%Alw)c_C3S42'
    'JbxF~_{`C4IkN^#qkJ1HbdktnN}fIn@6<xB2gy9|4GaI7uIMFn`6()fJlw_<&`?SotFHCHO#T8(jc-;CL!ljp;ZVt{LB3!pz$7b~o9U}QR=xge'
    'o`S-PNrM8C=_d`*3`xjeXn(#S(x_+lSg?(Gg=4;i@Lj!+%k~H>R4L#3T{T|>l3&ptaoSkE;uRWw{Z_~rxVx!*2462A{QttrksA@hl<(NtL(a#o'
    '(H~ba**>FY?@w*jxTRWOZtbmuVo3Vsl3XE;$V{94fW}e3jI$i%fF(*jc?)zWVhIl@pyXE@Z`Fl7QfsWe(%^V(8O)zu6fE_-!O3x3Y*uFJVO$k@'
    '?)mgw`w+mbq-Ua86Vd>*B}MVN@SJJNaMH$<Hhrv<+*^D(eFZHAc4^1GA?ZQ1G?J=|eD%KkSgBeJgc0$<;L%Sq{cM=rZ5@kYMIQOFox;+qryhUx'
    'v~XK3X>C5cuXiY^ep$D!nZ;fcabH58vo+7OI*y??XV8CX?-<>WZZ`lI^s-iSk-u5aNki8V2DKfwLX(ima(tvsLgQ~$PV_U3K1Z->fQ;=qsQR!_'
    'j-4#AFHwB&xwFb}<Up|$&@Tb{Pnwf_iyk4VYKY+n?B?aIJKaPNf-3ERfs`P{lPR(EI;o>1n;AJjAVnIH7)T2quI*_wOxuzb=hA4^v%9Co%T0tf'
    'aCs2A+A^|*z1k83#PQomAq~@5$>+p%3@~*$JB-hzw((bF8BP~DH?2A3ubKp%PcMI@%xO0Xuigj@EcY6~2%VsBDhBS!GP1H5oBrOeO~4T|yEEu#'
    'L(lOG6j=oJNiB9g5~a0^tkKS;foFL31W~|LTPlqk%AMASr<qCh15d)!;-`Ffb0zNrqe1DI`X8cI6nI(7J&8$glI^*+Xa8V2+cWL(7U56+*H&I6'
    '`FEC&rRf#dN_{Dl$PJ&~fEPtk7a*Ki*-<p7_}516PC&S#N^s5+5+S+g6JUnc0AN>xfk4UTCP(jl@VzAJZFOYIU1pkVUE>m59l9WPrft;hse*L!'
    '@}xcQ9@FVt0~n~ozgja3R3-vJ{v}5uUTHT=StAU;>)18($3C#S7F?-S9-V&Nw~DH3WG!NSVqN!}`B}-l0W2JRP3yF^cBbG)3%R>9a+s5h@t#kT'
    '(7=8vZPl3KG`lbPu*w1dVKY3cir4n+yO}1MaNqM9TkGeF<Z=4rAr(M>=QRFk+&GqtTa_k=$QB^Xe^LmM542F>xlkN{#x%c}=145uO<Am&E+DQc'
    '<*(dHt%Uj$CwG>GsgUJWiYalwY-tbCS<x=pNOtScW5qg`W%ewoX71~7?$Q|5u>uTL3VkQe1K-@~@ERT!+q&Z|5ZLVHK#se>o7~|tN-o2dJ!Xr*'
    'o)8F&pW8!j=7YV@puwu%MDD)55mzIG^IpLW-g>gs80Zi)yhK|bcZcF2@0_d8mgR=Y@k>yIO&<}|d!VA?QyRnktVWW1ceb)Ec^hf<3sIY|p_w>#'
    '&xAcl@&kl@^?MPNB)dYofOpoj8lcFcs)3E9w%C*>$#vOWCXaBmj*tlplr}NspMT0jhzgk%?6rzpd~|WKG9MHWa|%ZL`E(5(EJ_8pUQ)DPX11iS'
    'gntC}<T8sA9L+f$HzJLFZEcopcTd|ge5BS0x3R$|GWmIGEs4`+%c)a#`*UW+AkAQ9Yuye3(fGh>nFX2QYkwexvt@*TIqc3Vb2#B2@l>FOLXWz$'
    'W`;O<$t-VGN1$?pKj8HCm<=f2GQw)(xjC80SWgBitB<?xiE2fQW1Y7&7eu`5v|fbM8RXcc%kCSH1#1qC?M^|p6`PcI&|o2l0fsdCP?c-hU`D>1'
    'yIeHzVfc7UK0#oPtY^BU39at`P*I1DwL)mk=vqs5-z^@Rx<6uGSjqWNOS=BK659<c#6l6`L#wJl#br`U-GjVq;jiwe4sBWkw(Q&&Nc;6!1DhHr'
    '31_OBWV?L=#KMU0-9LeiDe$Dl&=aZYqbJS5+))uo=;~ORstsCjn_=LGO=$Y8b3ankY~Nbs@Ef)J2Q5z<#zg<iCQ9it(_BqfG<SPJKcQ`%wuAhi'
    'YrJMZ^gwms9@mREp}2Bj5DZ(hGZE@K6`Iw4o9VH|-!YGszpIzTIYQ6dEIs`^lQ$2+V&})i;#aTGAG+F`U#%+ZPN8Q4AuUhxmENdGf2WPg5kAPl'
    'Xf1VRf|U{(i@Y;+IgyiPCPRw9jx1oQ7P*o98Nh?ROuGy%PL8pJ{PVNYBNzKfo$i7|M6j7j9xVdOOGgdLKga+)D8i-YF*ELG@kE(<z2PVM@Jtu<'
    '{V|A;-OH%%6d_k-09y@z6=|9KPpUR}2<_c9b6o=iS@s@ZjnG=HWY_alGK}9zxKN5iwRLXYPERASRP;OCx+PUifEKei?TKmo^`w17U_m<CA;3l+'
    '!;y$sv2u0Z3={CF;n{L=HAzH4vl*@o<7-f@;fj%d{r<`E3?y|Vs??;MkaL}7#|Ypvpdm>)0gj&lf<L?NFk!FkUqyd@P2~wCqOgVX4-Hqx85yHa'
    '?za#h$ESFg1&ruNBTxD1nTm#+riD1!i39lE(mvirdFE3=94mrXdHLh`*to@d7~UX7))VnrMj)lACGoRTO`r+xLDJ3^z#c@}C-x;t7%QWng@Rx@'
    '@dOk0oZV->a%V~DeKQvvV9tHR1YYm|Xdsc^Xw1<vb^YXD-Zi!pir%V!51Fszx5|^!y=n&9CZ$IUuTaNrlK8C-al$S9)KaIz<S>Q&g@Q6y;%H;@'
    'K1%bnMnEO<*VQ0$glG192zNy&_ErLV8mhM4>HH2_#-*S1K>T(B?U><GI}t8(<!YJP2Q5Vl+7(x|6X8_5Ghn4IDJA)t`J4$dj9(I`up1p;-uO3e'
    '^SemKEmf|gOW3JOTjWXw`42erN@}2>A&!PG!dQI?+8M?%G|GPLuUuI8o%jGs^b^UyT+#;<RqdV29gdm|d(R2HaUi)Tg9$+KyFRIqf=aHgYnCy7'
    'w)qS`GT0Jj-VClM1~J8?Lpt{%SGOjU*-O8?w#N{{zODP6EXw*!nIzc>QadqEp%i~Aqkw(83MM$o7M`@UdPcyHX@JMP!C}LHP6vi)bI!Z1M+&9p'
    '?BRkT`M-K@rtvS6Yp#DBAGcIDRj{hqi4BpA_C1!*Phcky>p48`o;F%8&7Ju^5M`C)u;roKv}V%Ui@P8b>mSWJgoJF<@`BvBET3X=QT~bt>JzwO'
    '(WyEUQZ|Bnt%19inT&uy^bd20)MuSs0vdR9rOzea>P5}bJR+;XaOwe&R01I9TeP#8ja!|Ie{30hcOhwDSeE>w{W7s~pNHJVjQ>nl)tQqe(IN`8'
    'WM(`)P!`YdD7IPL1sT%H@a#tom+q@sI<fNP_eG#-O?@9w>q0tlV}|7E1gzjW<=XsB*J3bVE!A7I>alLQf(+XHy{SqFqT7>3sZwRto|O?`*Ia)+'
    'PuG$TZcPsm(h;HUnstDH4>B1F;<uPHaOZG2UQ}))0=KrAS-xicG~>pb{hz9TaCV1?ydwu-yV>Cg(B_^Px1hY(5&|+h**~d$k+7QHuLo0*D73vP'
    'rc2)W=mPDzoKd5=lU07m-ewrVFfKo3C6ZIhpJB%`C)y?A$A;|}e@#m$9x9wBUDXE$Qh4mwRP}8hq1YJbS5L3m87&0Hkgr}wUt#DCrN5@ZUX_~G'
    'HzJe=-LDoDhi%R*m=dLrMzw{Fz`jiS-BAm~;18PB+vR*5u07tE!!4*><<zhY??^X&q-4YZr#5L#OUTG<c$Vf$c>@RtyZgSh0PoF}+;VJouw2O>'
    'KAo{;3!#u;V0YLH$!T{mR;5A&rp|$<pgFR`cSvmRY7%S-#eo9)1Y2lh3ubCG1&7gB)<jUy<nDYWv-H<y0LyxqImw%@{MN*UNJPpB0W4IqsKm{M'
    'fXuz80W73Z57V123g$IF!Z;5PSC&YosJ0bOiwUCtbeB$TL}%k<Mns}jcTfv5EcYU~;$QAPgZ^OGhH7p5Q>|$5Aep!YylOwaWUxOpDmyHUrCBJx'
    'GNjbHM5E1o%ZG)~DJW}$v%5Czn)0$ivq;vda(NMGZ9k$Ek!6^=LDd2DxFcbOw9X95y-p2Iiak`Cr`lU09Y=hsi{^ZS_J+Q;_x$wZj}pT>tDlRV'
    'QDF^kLPEmxM0%DS%`!^)rq3O=oas-8ax>A=!2Mtr3$|sX_j)~c6b5oKb{E~H(~jsN(f2v{it4A2QnScX`N}OZM9I*5?-^G62ina|3RgtkOrTwX'
    'iCz2nTpSl-dEka$;yHP7<U4DIqiCdfF1Kz-;VgI9$XJ)Oq$PYc(oJq-mFqeZi%<O*<REO3`sY0Q)A;Ywz;E`Rj($9F1I*NL)eh7W>bRze%Okwp'
    'KX%a_giv<OW>VcIW15c~???2;Y$1>f8<K$x*(Fbav>cvSz~MY=KwvdiUMHJ7q9k{=IX^F>-fpjYsl55;Bd^{Jh+h*bK9b(T1dWJRhqrIq=w!r<'
    'Uju8WFA$*{7%b~?0|T_k+_n6L9>d9@!7?vQ)c#~K<(}5PNV5qVuA$&gJ$P3~HUT7M(hY|XmK9<A4kOKT`ya9zj<n3F0$FbEVGR)vJkgvxRu+tP'
    'XU#kkm8IdVp18S~dk%klc+~HniQ9+ho&ER;o~60eFGyp9JP_IQU5Nr;%6U2Eu31AhLHR>~jCQ%D7X8tPHG5~?KB=g!n}}FTV|lLE$-+tH4()NP'
    '*VF{McW?&EYgRKMN{4{o+*c%}pFASN$5On&yQk%v)DS4MyyD`KzT5m2fY<9938?AWSKEMwX(^D0Q{69GaK!Y?1u3UUy{-}c6ju=Zc|aOw6_ZqD'
    '#WGvvj}jF3UJk2qP+pKs`|?!o{oLU2EVOSfgJw5=9<}7plUzwx3mV47q0z4IeJ(e07Eez!V>fvrfyCoIGJnU|keZdWJZs}D3;6;lKbFnUXc<qM'
    'Gr^<COW@dh&D!)BL;SDY<vW?`0x<6<fb6B2mS2ARijGr#XX|*WA?08jUVF%s!;NfrCFri!l4!Pe>E>qfG%dA**w+k=pzkzFjty^_uC+$;EXTC1'
    'xSpDFj=!R?7|WnOIVRuSJkRH*DSzBh@sysGsEoj<pqKgWf*Z5kPV%6AzDs)d>&ftiFsPPr>!g>$ojKp>vYjvS_)1>ou>=j@G|h7&`{_<NlWJ*1'
    '*Bu{mE6G_r)u7eS3F~`kbk&XxO%&O$_u(hJVstL^9U=yxjyAU-cHA*zXII*yteo%H`+tx8{R4)D`|T!(2%Wh@T3R#2DnLwWK6B9w+DwY9LO?RY'
    'f-7;*_C%Jna@tpBH<jZ}4rNnomaFdI5n?r~K?J_U$QzAl_b&EgAGtLI7RcxPQjdKbQ4q^}tIdy$e07EN=>8<1B{PFCLL2Fs3`_g;bh$BDl(pjD'
    'D6+NNoZ|T4u+~nAW%4AAV4h!(d)s5WC$7ZT#s4!yR*J<kC6v5_m^+3~<OVAC%%w-B;a(2>xd>yX+jdZGowbu1K}p4@+<>!^4|e%@RNc_YWc3-g'
    '*X2LozF+z`MBcsDL<KNt(k(|)RRL1{496ttxFdyZqT3kN3xa_~Z$~4GRE{?@hH!w_cB`Lra^)KpXKS|XSvwm-RXJYT_Bg!FV0~@34zW<5E6Ei_'
    'DkOKj6TS**py4({{4n1b-p^8~qt_aCn4-_<)p-l@9@5U=cSQz8d!5m<mKFhIv%l%8W24uA0+0sbRm*W%*9kr-x!Y<}b4R)Zkpq0j5GwwbIXt3P'
    'rRjHN4rHWA>+Rj(c|bx{+#P@UxX(=0SiAT)gYm!Mb#-<D;eG4YBxe?_Z8ObRd*s3~X->Mza@$NkPz|&RvBc8VpKEgB_P9)Q_7yxH=}$acSnjHq'
    'Zug}g21U1f#_EzY%Txv`e{Aiz+Xs0C)cEAGJt*>Qr$j-nW>>4SDQ|BQKl!5)dYp;(JtF?Ur+Xc}Bv3*nn9TYIW78yhOk*V9&o3UplB_XN8!Yy('
    'CpDQqGhwBZCpkAL=8BPlkObiIZ9P4J2?RSc91slfXYo_%Fl@*uyn`y!Fv}XvM5JzUf`%N=rQQDkZ&NvC^#g)2%S>cRjGHitI_5<mx~LfHsGdhg'
    'rsX;)#h<!awB)g~MHe48%?Bjy;Y<(Xm4}Wj=kA%%nzOUAAr~QauEH)Dx@;}589xH?$o#OCKFFRWwV-id*jqmE`plA~)Lk}2HqzBa7ubJyHe8^J'
    'CCm3bT?J;o#FG=gNAe6W{q0#psVh5EU=8&V-{w)FDi~Tb-S9{ww0pPLq@u6dK17svzP-)x*<@`&Na-HRN#=p)$HL@<W-$_J^VRFnwO<!%TyYgW'
    'S5?gt4+!emC}_esOd6HCLz)0dhuO?UhnKNx*zjU{;HDildnW`o#oJVu<*CrfYZ+t~(#4gLMbI=AJI;{4arSTy96Un9kZEAfMp~0u4LEq52a3Y%'
    'vfK*Dc=>;8W&w)OExM1xo<O6pB<9JKH-Bc6K`13Pa3-Z8vYuY&eXEX}xyUq%+6Fnwv6F|`GfU)v?8vZ^@fgg*5>zX!3mkDzs;HbJGYhXlL5$lm'
    '?dA@E?dhgJc_P-bX&EIb)Uurk$ctW=&6holV|*^FV>xp4TDHQ`5!QMe#Mo4H&|f74D9U?$R?Z@qo219xV2nJ};SQ6{#EXZ<TS_}SfwXd>Mp=%^'
    'z}A=TopEJW%EjZ`lSMK~_x)E@BEw7y%B=x{9!nxstp1PfkOF|7`4-VWPF7T5+r71xbFPMtg|nTaV&)AdrJbur>$zHZNa0;?N}@v61n=nS<TxCF'
    '4&CYgVc?`U+wQAX(83~>WG27C_RgU<>WGs`jpYZDlRY%dq^^&(L-t%_JaN+vo&3YBq<tepz}r411}YOPb7JvC{(P7XZzoETjxdfF#ubH3Qb%x_'
    'RiA4Bdur^-9i}BkM%^pv#-vVR$nS_gt%O6;JulR}Er}S%lC^5{e}tsqKqD*<Eg*HBNUeQ`Ta%C61f_T+I+P7o=t=y$dehx+4z3&6vo&&3B%fKc'
    'WDOiczIp{nOtR&*|6Bp^)Y1N0N1$6_cd@Mwl5%g2J(5Oat)_G`yIj~`d8QF_kt4;z&U8d50W4B{;P6!uhX?;kA>4b)e?u=YJmCO-U3=%QKRxU0'
    'uV|C+q<c7EXS%@uF_C#LfIQ8R?6XQ3F;%}F_l_e|T6AYRNt?9#%38U`Z__>LeyP(n5Rpq&f<y+IkY~am)vx}`^FGv@fwUnROjCc%6IxHIis@)h'
    'jpds;(21<Ka=lBvaCC}4(Q@<GauXyVtG8(<f>y$uVzG6WbGc=eFipw5UT%}&I^N-G+KRF`uH&zx2;?kC6t^EXVB_v@5v^dmHIFmqRC#D158?yl'
    'Fdc878->?`-;_MiU0$$ms2A;aa6tK&F6Rz=y^?UdNGM&d3!TteZWfk6zr`ImtaT&Y`!!2z??BdlGg&oJ!CM;95JHP{RQ|RWrlrKjmrN!_4)W=K'
    'K>^yKBzNC^Tj?l{pXn&y-#4q;h54pKIkq~hNtx%;<+_$@Xawk~;oen$>X;cys<AcJaa(_MZ=I<575t}aKb;A(K~5OdlXl`Sow5a2jvOO4$o=|#'
    'Z)sL<=)ibrlAG;VkT{iR4Aeho3k+bX8th!GcfHa&-Xg+QjTiHIQFd^YE_zU<rb&cR$;gB9nxlM|n-`jfv+u!SRr%buZ-wO<HPpzL)(E=djNn_T'
    'gXRq$bYx0$j^$nuBGR49N|`rC*d{=9De`vrER?ksG$2m@=VKn-zcbuS^W5x~#!NF?3Fd|dmwyGfBXjEnI`ohfXMkHg6AHtn=mUM>G~bK>#EE`d'
    '`H-G5H4dJAp67S+!nZX7n?W!Na1D*<cU(CA)=k0^0RCP>al#st2ly0^@ZeoIeR_NwUXhv!ayhXKSjvct8J&^G`h%!Q1_-!j(F(N|!s{*rwCx`}'
    '{{J~=G8h88drI(oLFcIEf@0}~kTFRtSDE`IllcVpR_yx74<c7%r!}|y$kPBs`m}UFLur}3>ACkcKPOZ0`Eg*ugI{YT+gY6tdsBgw0oi~XYdL~i'
    'Td7IlQ&OQ?9>PLe^vysly}vU}<e(%X7zYH~#s|1ts=#}9HiRuWyq}p`d&T_8qkopU#JWoG`~HzsD0fn^5*|S?lwIN$?Sh;tajeVdOvM;JX976x'
    'n;EPrRZ^Ue5mGUglRSP<dYb=4c4u{9Cmr&(8j}vPB}qUUsHuLVV2Wz`J5}ps7jTwm9ooOxPh<nsu(O?=e#&*4Q|`pON_J!~x{BlS|Dn(GaQQIk'
    'XnC<(0`NO%q0I(Yl^?xN63VP9TOj7$3Axyn<u7ozo;)fnH4}+sRiJgq{5<j*uMf<PTr)40;LivI=fB01>0_lMC!e37F5S0eB`mn*Wv1B7`Y$yj'
    'R{6}^Ep;2)VYf9waIB-P{BQegrwHPFPQpiNEcZOI@4kH@mZ8k7QP3L56tFttk!N-tmuk=w9fx+Unfu~-D$zz3tM<uF+sFHFIh=_e%RV}J7S^^*'
    '84cN55Z{6UpaSC>Nj*)NW4F~9&xX85XZN5GBbOp0E+D!oa!>?SDXI`F#-GB1B&O_uCe5Pn?N2{gc)2T5o{$`&<THm#31oh+F2|Kdw#<^Tj3-Wm'
    'FhJ?7(aoF4O(Atb0(yf}&MxKpTUYCnw!H_jLf3s^V*GED@YC+z9!A1mKqE~fojQS!cO4Iv@zy0PgAxGJeXGZ4{!6YEB8Y8RuCS(V9AKx84Mm>w'
    'qWib+Rk&O!0i;bZtjCkDe>Bl>o=jiSm0sVh)Gm~D4>q?y#JNRER+8P*(UqCJjp9-22o^$i7&C@C<#*;f|DjNn^U24x3tDfDAhBL{|1Ahrhr(gi'
    'JQ%q1Ii0uQ?V=P<l(2eAcd(9jNgXuvEijwUp&A?`LvAu~-EZ=OBFCkdwe>Bp1#^a7dIAnVl(5~!W=r1a)*aadE%gQS|9%N(LuO(c%BhrYDPg@@'
    'kSZMXbiyngOQ**c0sBRzpQr<%m(Dbn4SjlWK{lR)7NUOl%)wZ^L5G!L@I*<muy+rh7ev#^o6Ekm5pv^Fp-PChkA6N1jjyklvu0m+`N)rPjzRUX'
    'XL1OXo}Hy~&j#M-tqlBw0}NEVWfFaEsE;NHU-uIrnlw-EA-_;g)|T8s^p>m5kPqG&kU0~g^W>b~Q@C$zk#(gCrn9>xLN4rjpX(qcV_nr;SqFmN'
    'W#53Zz!7t2&md2TFzfY25x#G$9vE}V!!vfqLS7u`#!c6N=Hl65@DiG;{}M7*{Yn*+a~UbJ2O4*cw#x#Vh%*RiGChSOeHQ&#O``g)B9K;diTfaY'
    ')f{X`?v*Cr{_^!9{Z(?(DhNyNtJ%VikJrzI`3pOJ5=svAtg*y%oS5p`N>~@lA#FjgAO9-Qo!)(2-cV1Nrr<B?c@z;21p~?8=@J)F7m~R0*Bu{U'
    '4ZDg6*(&tAE6XM)IJy6czp{AsTd~O>skwF)KRc!c!YWBBRuVz=m2lBe*ScIdK&I5ppdRW<neG;Wv{T-h#{AXt%dz}OzPZ~!Ezt*QoKbG%L+tts'
    'CC~f4^5NT;PD&kjQ}c&*uwPm|S%)L2D>g1an!vkEQTw}?mPedz;(xMyFa_Ep+yZcgvUy;k=CHy(duTPqu<a~WO!bj1;M|5Q#IK6<<yvg%v<R-='
    '+fmC`GvRlc<)f?MEfX$$Sw0ByZcG=h{Z*tvls?l}8%^cNwjk)rqTP%&z^u;dKT>QWyv(*B#JjX&yGChsA<j0xq`-WPgsr69a95#`wT`x;c_2Br'
    'N=o8nz&%zkAE{!sx*kG0t~FcB{q=X2U`2pAe7rAu{7>l~57x_Eo;?e_V6+elOU^?=++oqLaWqC`>Tn}1DxPa?>tUZ<p+mlhH9cOs+@e}2tI8lY'
    'C3uPRDi^v7UO@{ue_I3K%S@u-mBBBcQnzxUVcIt3gf0qDdHt4=nD}8+&!FAY<@@$WOUln=YXgN54A*$2&1l}wcqe~6;?1EUqbcaPvdIdlUBt56'
    '0#|_55=oMwx({T1ImL!UfOc7{9Z$x?CfROXe^pj!DSUy~`2>fo6JlqwM0-?s^*B&g@l`btVp@c>eNCt^oLK;?j+{62-TQY>Kfe*@cCheQh;!Bf'
    'Kd$as5-|_!Y`CW$+`))()v#QP;)*A+=hmJo>@a%v;$F(Ck*6-XCnx#r)Hn`$nCPzH%I{2|S6~sfRuYJ(D*dl~uF>%VK-jrWL;{uDv+##51??;8'
    '{*^IbsV`b=3Z-D}*&0ZxU`A8Zuxez)If@WHgi)*X;94k$y`^Nw*^@4vg9y~VZ+5lwwys?@dQ<M+@n6hKOq-AUbg+Q!d6R*0moKhc@YH<L6MSHd'
    'dFgR|B`qMqlo3T(nXVuEc^|cXiZ_N`eS*fZO0CtFyP+X&?1ZR+9GPUVzCz>y|8E9(H3OrJ1lp;y-ld4TpY>+2U!CO2F<L?f)@@$jVL??@_~7nm'
    'Y0_E}!PYbbtQ5a9<fegIyTcNppo-5RMGG9bM;;V%TugiOFXkmg4s3vnDaR&<O!Wz5<B%9PHIlN_sJ350n_apq*#G57h68!<!J2)o%gNvE0>yAa'
    'Ig>$qqzb@qS&gu8s#4ijn3;mI$!rL0<1N4MYYwJV9|E=RFOP}23)Nn2(S+a3YV|dq`^qYn6R$O2!jZ~-jD^>E_@&AJooU0LJ!_e^0zLv3oEbc6'
    '@~G93HRdRnmQkGe%|P)NTh|?`ioKH3OUm@!+wvtD85t}gJwy)b-ATmlFmU_z4*cs0zQ{7#iy!_qoT3_)TI)I_0M&@IlxfF})A@yDnF^jX5ZEDl'
    'tox%>nvQ*1zPOv9OWc!-qTY1e|B|-L4h)3QwRT2|nKzzPhpt4vBj!mWX!SL=azw!}6>YOabRf2$B*LCFVjo=e2OHr83M_!Y25u76OWAU5xJjsa'
    '3B}@9aPN0rJZKk-Pb?u1?h-UJQj7lblsF+eV*Ki*Lz61Fd9gd78jc(sgP+?CxwE1$I2Hn`s;+kbp20*51A(PA|JAE}Q#V66Rq$JOx_s~J*zFLx'
    'H<G#-?DQ*{zCBzBIlkIE<wap*xa~3nsZxOU+daAE;qO|HzcomWDfE*d^kac$4Ofe!dg0bwv+T0<sv0=gv+s2)ir{13&1TPiSv=4u#dNc&+x56i'
    'rIibV!~fw0;_$RAPmcjkP?gG~w?HHt8325#wNqcdRe8OEmbkK65sYu>*1x*>4@#3#y~~m3_RhY%KBpfo<tCD?Ek|^jPWxqlw2zw$Ax@VBF|0Ft'
    '25QZXb7V!Hb!}O&c!5SezZ(i<^emtPT~WGmD`Ir_tf=)w{82GkRY#^lsFhc$hp<TCL|Le}B8-OhE$BQiF?$wrOqhvWCt2=|d{AQtCvcZHPrjq&'
    'tAh^9Q-^*1d%e-sS}~kZ1H$)B98hDXj%(64_e^=xJ=nW$4JUSOz?oRKGUF()Q450jpN2}xdAyyl170(UcSNGt$q#qbTzl#OCwdXsv`?be1h%kD'
    'h)OUy-l$tXJg9$&cLbdAj#PhiU%OhmD!H?K^*teFmuH_G)SB`c9UnIQQE#kt{Pv1_eQs#x@@si{eSk;Y1U^WM>!~8z6;?13_W2@OfYk7NlryVa'
    'GZ2?LFo2K`4);f{r^M|o)g->*9b$clhgcmAYu|fkdSiy?W%=$EVSceVO;+htCrL%|!gK4!quTX|ByrE?vW9~rqK{;TH>Eac!w2I`o9g!8pW&dq'
    '(^=Yw)LL@jPMa~ZZaC%%W=6^y<|K2ZeuFOfb*GT~={pTM4djT<d^9uw&DFC!{xi<$=<>>LiQ8yug!6L8O#@pz0(bNv^|Ki@4t#q{KgYvlNtG42'
    'fmj^gR5<-qQIWNoc|uVfIEb9+w7tVuCC}O}W1sA+x`6myx_=eyw*QLfRyG^j9ovY?`zp_ie-wz`%FfJEUpo!HJCZbBgcIE-EZ5(Bj)$d?c?)5G'
    '8aW#G9-?vOiCC_>19}TRG%X@}(~MYOvsT@+oUp8tr$2#qsgLENx(IacrTLgapS*aJvMhQ=v+|~V#E7oPJ@le0PPQ5~GbT#ijwd^a1{x!7r{^&v'
    'UNz0O9vta=`!a$ippBsNI8bm;b|y^*HPdq+h;lggj8F2M<$sdqd@vbW#s+(+&DTlHjGE+#O43{BgSlEkZ7NSGk+{{KqK&io{!V+f%ErTOExm4k'
    '>Iqd|S8ZALBg^|dx^svbtuetX)6qWcKa(`Au2W^RasQtpoaIfl5O8@SJ6YGy`0~4oz>(zcu*Lf2;=pmygcRp1SpmR_WAp8E2{}hVG}tXNk=B%+'
    'iG4b<)ZnT)i0I6<0y20mDNU21+}_MI-2L2jVYge1aFL5U`=fAJ3@FRhG`LHHK4gT|-?!$b@!L!vxv-R-_N!k-MXbcWPdsR`QXg$FYkarwb9k9Y'
    '|7$J*>dYpy;~#~a7<^GDm*igWzhZ$<Ya;}?uHY=ps%p}3^TU<x@ms7No3Wp?E1lgk4!GYTS&xkUbhuke@YNq|5htkV&pTP6{nmT4yRV)fpuT#m'
    'pI}|o(YtuShqQPGWdFBd-NVt%59D61g>KO&94HeJm?qhmDB_%{izADJtlNxBI_P3%&(8-!&i83&BCW^mU;LC>vv>!7o*5<rQ$w<550D-mGpGEC'
    'MP|S`%_HXIge*SZgBM>DlhuMb26D%~MmolNou5KJ^-dmrjX*^ET(?7IP}da$f6MdCf+s1no^A2*vBUve(EYE1V{}d_S#Ys@@~f>smcfOR|2dYw'
    'J4k>4{f3E)oDT*C(!dw_$ZO9@1thzkchk3|TstkyJms@}i?e=A9_TIYJ=f%4>42x}g5g5X_a^KsV_*Xn3v_<&s$L_HS3bGBW~LY;o^tl;mZ!#E'
    'bC(_fywWP(nHBy3;Oh>sZw<vQ;o3ox*-xUP<yO5Yxr5$}vd4`Y<Rh;?4T&}J=Y~H#IapeVo-A^_$t9S`#t5LTHWaMdahH4ce=tKcy_Z6~`F4}C'
    '@46tFQ|r}5^nb-#$seU<sv(@7HStP})N0zm!Mr>N_`qKZxG7~2<*BvnlFL|1KyU+SUG|6rUOXAxqyh+(-zUa9p;2)yKm1B`PNn5#a^Xa2p?oXB'
    '%FqdNCb+NtzJ@%OL}%Cs-8JzuSQ#38%bqB#v9rAjyH{MaJVR{;pcux(wom3d=Zl|V;i}mjF)(^+e9BBY>1cn-4%<x5S3^zoS3W{2zLdXK>Zaj!'
    'V(piHdxkb)ubvE5I8q16Qr9QbTsv;fsEd}sM{y7$T1~0gL>lzv0j6?hB0X&$+i<|guH+(RkyVTVr8;78Hu@Fz7tCL2vi-MKK_3AMAXwZ8s~#mF'
    '?r0xdCc)+k>!$pc>I!o44w9=bpY1p&Q?TeDdDUr?RB2vMtu+_L!vG8m$5z?;d!l&(SF&1W?!Q2zmQ$AqRYuY05T1kCAR&eqa>siZ&J&n(Yxgp='
    '!{g2};@dra+Kc~V?q^RSP2sG`J2c4vYCddZ#(zve_OIst$&hY_=qkC4?r^9PsgC80mGxa@ylt14%ROAKQ>2hC&P4o=n1#~iJ$P3D6{xlAJ4BB7'
    'o~krIDOX$3tH4>U;@&9aYqH>nKE1XJ-k?57865N`m;L_uu{FULgO)JtdFYoTs!ac@<WR0NH|a^CRgTT?<*5O}>^V1;(MV4v6h#sK(`8n?>}F)C'
    '|7eczNi1xe^<0VYFGf-FFBjLQ?Yh$KMW^pDF)rQnUYbm3b!P=uoXM^<vd40Ne;U0(VY=JNns^qXQdNgTuxOsNIfp1N=FR+S(BV&YQUs9`!`!+G'
    '&5&xXC73IsaF1nrZP4HdeVpUB+RkEtHL5@=!@j!1>prz!U<c3mRjV>Co#&3_5>13U)0VU-<_KnMJp=S>WP;-<<o3EdxOS;+)*YGRkZ<wLl2X~v'
    'E_XUg_JWwGy{9gF#4L3=759?4>YNDB&13$Wp;51#{$jedho=FGBOET&UMJxuqxnAfWe#{L5H`j-Pb50JeowjyxenwIo*rwvSY`!=C0)v{8Qfnw'
    'eeh-O0zTgwcKPaBZ^<aXYw*zxjgr(7;R@jd9(SjSWo(R$W=Hq~EDBfl5G{D?OyoF-5lKV#q|7V-<GTJDwxo#xYOu2Ia$|yV`O_uoCkX~U|LC~Q'
    'iVofd?W=uP$ZDb_Xco9~at0Yzo`fXw7eo4(CLz|}Ej>#ssWI1?E~Z>_8x7>pe2{OR4eGXBXLEZGJ2fECc`!@OPWPmI6)i6uMXLF5fumfA;poZ6'
    '0^={C>Dz=bQWCeZN_Yh?-`7b-6-&yJ(S6p|(ZwvkALZvIUNSjH@kn~~dV5rC@d-w4woe9~nw^-4=1(k9w!@7*I`zmlcBrT*X@*anaA<6&g5nh1'
    'cP4TtIbsp{1Yo4PfbGg#82b_&9R(}HON*|fNRyZ7mY@n_H&e5X9aV%0k5ufk>j*@nRdxEDt=$fz@9m=sro(n5GX!3Isad@Q7Mk<h1dwWT@hR9J'
    'H+BGpIVT-Ox&Yejz}`pgwEvJoBVSClk?%$U$9~>xLqrmHrh#lj{?DcwG>8qHO85=roh&uk#$ab?)hJ|#C`5S>yVoN~40q2YoN%A)H91cg7E@>@'
    '2Bn&P!hOhbS6v!4(O6CQ5&c^V4Il8T_6cj&<#pm39`TV-XtWZbv31BqYpmxynnlx;l0_L{D#opjC0dN-40uB2f>S3egQ==o4-BzMP1#*?dQsHO'
    'mG~9uTWWDj+9^kkW?_AexV9%MHUyAmI-HisoMmS+5!Re}2~u6NkBG{)P>Rjj(!C5OxrhEC-FMKZmu*UqMW*?COZ@Nq4kSu-R1ZIrg2UNswD^TF'
    'PeN%UcE}<#AF?zs^*&uWI^hT;kbsMdt(^YHeTV#1b0-~rSF%EDE`XchB9~|VHC=e5<Kp%tJ{*@AcCt!c*VwojPNu`$g*BJ+$1vK2dV~Uxb$b@`'
    'T5%UupWG8!`Zm%tLTbb$tfUlR6Mao|pbiL4L-Wb_=~7?M#CUU=fHkj=K7Gx6XzWAoeY>kCJA(xKqB~TCSvTOd<?t!dps%UF?Ke<Cx@N*G?4l)4'
    ';k;y-Kht8At~NwRs{j(jtjZVk^>~T$zZR4iTH3R1e$f}C`(5_%>z)ht^~ms*Qa0{p(LDO#>3!;pi$OwJDH&Oj?43GBOy}u)M8=f@>DQ_0R6(vB'
    'hkH_#%V9TWh)+I$IE#@+f_kVv#Jlj>*6m>=T7cCg!(sqQ^)1p@Stz?<Pd~f7dvBN!gL|l7GYLf)>XIGy0ueC>yRbB;QeLS9c1lE&=1-2}0HK^5'
    'VPOu-a_lDFFsw<a3HFMR2+)H5Uo`Sn*GyybSk9d>!cME79pVX#GPY+b+aUP27teekxOKIXYU6?1RWs(RfEh0D{SV|!BEVda##oEr*oRjjF~-WM'
    '1V7|VV0Sw}7YlZznQ6Bnxlyax&!>~(8{$aiwc6z*1qr402$&_}-BxpV^%;MNrs=A8lHb*SKE*ipFbu>-o8u%W>2_hq4_<D0af`#Xm6f?Od9LaW'
    '*le_o#vz<q)%CLzEC59X%kO~rW1xI(ILHsNXK9(KN<R2D)*q@0G%1`Ft!8A>%K02pq}h;PE1;z3kQaJp?rC`j<)<5=fCAEf9r8xX(YPynf~A}$'
    'zuHuK4=bUpz-a<Bl10#0F3e#9lcgR4`G=f=01LO)OjrLEqWb{8mZb>m;vY^-$+mxoSjZU|jOQ5=7aNE}1Zb^f3P&JYR-HYk6fMX;IrQy&2vne%'
    'h6Hx$L5p#$HCh<ac}Wa+6Q68#XE8F}W!#6aXmh=lkqPy*=aZMT4=)nr%Eg_y-3NkewDxpYqJRuPlLwxs84udY05hxc<@qcA_HoX3I^Nsk?g)lU'
    'mNhwG@@~)Lt>d!s581_9IGmd{erV5T{NMJK2SYQuzr;b+pbG-M=KdO`=IomImE;<e|4$dkuo~f4^y~=fOvXucYCO-!;5&4Bjni?5GZ{u-O&CD8'
    'FeY*}v~Ht!*w!8Z;mMkXr|%HU=L-b+*>~~I>0=N0btcRIw7*mMbbVaVaaE5CJmtQXXrJw+9759X_y~&OVC~_IMmf9FR0QZvZG;%Xdirs364}^2'
    'X<>eJK|l;^bB5M4Yq@<TXEN~*kcI!?S5l-DzS;BJ>A_&E!4oT|xtbRKnm0=LjdrFU7D7<5MagSqHsK6uuu`EN9@D1MLlanbcio~q;eBTNOp?vJ'
    'tv=Zzf~y_Y{@?{#eI&vl6}CBk;SdhH9kVxBv(%H!D)<1AhgJ`!(W*8qJSOqL?Y7KJ1~bj4g9qJ8Nc{aH&3Gu-<6dvW7vHj30W`-~PO_COJmhpg'
    '%|I|(ilJ#YlUG>}^dVb`F6e~51u8tj*R`Ht9rpe`{(q)zTPb(`NQ0A*cjJ^~RxzO)+SW?-*Ay>)9Ks==^CUPZ<_(A)Y*tmJ(zl<2zp76Y^?g1{'
    'i<`1}T$A33)5l;F`oSv@@q1o8$hk0)mOQ~qcyb;$v*Vo+^kJjOGZ5+^_D6NPdUmiz?k%yxC>qf6yQ$kIube9Yr<Mh?X2~feEiQY*bWB%2O!!!m'
    '_sq~_&v4m?rW-Ml|5dMsF>6y;IyaxdSA6B6&RmMqSy}vYYwU*IBTA^ibj0g=4J*=lJQ@_&lFD&@4BYobMUkv`{<?IgB^WBA^rf?0yfd@&E}+GC'
    'c@VZ{g0B!mSmm}!auKE~DDQU1&jw|?XZ;WJV*W-y(s)@(tgE)yDu{w^0(<3IV!AL&B=|HV;!JgEr=iUUv@^kCJ8*;efdhp@u_x)`7~{UluGxuh'
    '5W4nPZIC|6ABDaPUD>yh_4@%Ha+A$D)4(Cb`EJ5JQ^!cY(<Sxs&g=+=yAu=Y(7~<cn`07c?ktJUS8B9XyAyllypRX!*leg&?)RjCN6UzIxLohC'
    '8mdDVGytGU7-)Qn`m2SVGkH#?XhjQ463$7`FP(3!c|K54$k+6_9fZ&O#Fe(MgI|iFWh+Mh5oT+xJJCs=!*etHWyJ<GZgY>3ykOmdE>z>|#a#ZG'
    'w(-kOk9eEf4)-wEpd4>GjhfsIBRMF?9X>;K)gLbTz4uC5JDtEkr)+=PLeTT;MuizC^T;HF6^+iBCr=4|bWw=7Z>Fj0n)Vd9_r0m3_CJl=<j6i_'
    'g&>}ekLCn57puLQC>^`z6Uw=L0=>WP<J?FZ!j&br;Fe%MG<*UxI}7O(qMfm3Jeq~?MT8=cLZ;7H{DeVZ$qHj7`Nk^blq0S-&!rq_nw>5pcdpbc'
    'qb8p-GJ@myKBmC%tB#$HTx%;S#(AU{Cw+Jr8ux0HS7Oy-q?+-87~N0vK%RdT99vrPX$4}{WD(7iwWp;RKl^&%_uWCu;E&hMn9LP|t1)>r#8H}v'
    '?ip?J%aZb$&4AJE$`f0S-pvDj-lD)KGf1BAJ@an85J(#Ex3n%N27mK^39*>hVQsnmZqV*ZeQ_Q7FjV_p%Fc!rEa9w9M2-twc6%RDdADx$S^lCX'
    '&7KyN4#zCJsv4XtHhn08R+^?H58wo(>L*91R3Fn5;)&Y2S)kqM8qer&FUPFOTJ>YU(*K@p%p-repm*@ne5fPJ0hk@h?9G&qN_`!kS<fdkdzL_|'
    '8RB>_t*{?cG0W{gB#?Ge$EtVXg_z6Xp=5Ft)LK&l&BU)1@}oJd<xe+Xbjh9a#>jnAS+vG*VB1du=^!Ed>_ix}p&8ZXvGen4+KjzYoXj<qRdA#u'
    'O|XGOU!=2L|EmaS*-4|<l^WxY3KIY1+G~D)^39Gcc!@Pl+wNi;aKcR_rwEC4Z~42d;;2e`&us+@Q-s;GTDRlQI{x#01vo-X#yofvVbH27b=|Hf'
    'W>r7#UmYM)dkyfWFYD%`U<HTl10G;euc7w$r~<NCx58eNtn0PLq(Vg?6k=p`91}~kMdELl^2_El48TjX@a>Ox3`tzj_irS@V8H_n21oMEf`K$s'
    'zPtZ^eUV;^RFgXyoeuKJIF9Usl+)yFTl%V16DwT?NK^Oj@L?eD$n2gu!-hQrq+u`Iy7ux*YGHM8q0q%*(=rpDJi%vQ9}4U6c{tpN&Unv}WkBnZ'
    '9EDd)rKN6iTzFAd$u{?ecociqQN$C6>1YdS{QLPGwgFojkY)Xf8LU5+?|KebP_h7l#43eyeZ%>c+f+*PD~8ub9*_DbD}TolS<RV^YWTE(&s|fa'
    'qEmnPft%@g(^YY<*nd_!ah4YYKmefNb>{8eA*<)oH{#E(8MszwUru`6QeVvBRDksqk%^UmOBvK}#G9gK3nw_SZe|ky#QRRN^ICELI=8mhV)7Ca'
    'Zd%;RuR-()6hI!5;%+tleC4Y)zyKs+PVSqKUgepVW7)0z8;8X_j{mKsY$bxttLk)L{A{Wr$SMB&sIPEkc*sF^dTY)U27ota_@e|GjGKLEkk(dc'
    '>E#tH4|SnEP+l70n?aoEqF%d{G&dI|og<RvYeRYKp9FKsS$6U_A7tBenmR!LI62=o1|oKsonjJ5b17iwFp>JzuK(CO{*{pXEmrK9=8wnhl9%Z?'
    'g^Q<cW>23U)Q)dq_|zHNwfIZOks;k@OZEqs+)d+Pk&nx~+p*#wskoo-DQ=Y0ophj1)$2ZE2CSAk@yT&eU}o_zpY%yT;s^jn09F!ZTV?)S{e$|z'
    '5bKnw92Gf#pQD^309?Y$#2CDK)G^c3o@q&|KAfkV<-~O6A$9m^hUBoNQV0gk(}t~#fEUd85u%kECExfJ5m7RP`$mN$ZB$RX-+IiJ@n4*vEKlSa'
    'b7a*)oicB7qqDnRC!h-+dr<MnGh}gf<&mp$Sk$#|=uPDCYjH*rM;Bj>4XdjbOm~%2!T&aD9L6SD9fBHZn=-WHSs3*?QB?lF*3RfYxCJ(X1Sg~d'
    'SC~dGAj1q*O1AZ@8eSOjF{;nmz8+GBLN)*+OjMpM3nuxJ_JH?+|G%@a>Z^E<8J(*D(ao*Q%{S&rbII3)P3;x%K1f@Kn=`L=fHzCvU~}Nnm<zg*'
    '{3z{ubUl-XbVYPvOAeRB$you{zk1We^fF75A@>=+iPyE<^nH521vPsA4sq4fEHzq)ab)RmgEx_o@^yau6~Mnzvy5UtA4>CLmmBd~O5RNCTE_U@'
    '|LD|-R5!cQZKKTOe47l(5i$7miN*T$m`yolP)DU2S`8t&z6}Kc^(Me*N_0nIlN_oj)vh&?|5IYBQUq<vF^E{0nH>z{+xGnOJ{4=Sqp}Y-cKtow'
    'P=rDG^_Av<|KI=Wnxo{!W#s8?@;{S`-N}8I(C1eq@lU9;uEPUc6VZ}s5t%L1nyqZYjwZPTfjcUdyF6rSV9dI|ry{5yykm9-O7F{*xw^J{OFtu*'
    'LZW|?OR4tm0o{cG>I?yrSI(-tvJ;V%(J}o5Bmr_*M##VWKD+u)y5pD+>FvvhRaFBRl8|Jr*UNE?d{vR}Uu6*NB}pn8&S2>{fT~+$Iv-*tO;g62'
    'wXet^>gHVXm6p6n<?fP&kjdp#ePxIadiQlh5~!SquM|<0E6Is*cH)pB)nJ7M*CTiL?y#|v*zc4i?SAFD%VK)T|0^TkUwpehAwZWkw=-`5ohKn_'
    'ENjkfF_v{Od~(<~+6p;l_h04i2esH9cB09vfHdP~l3SE5yT_mJD~>}BDNVXh_I_*P*O;Dsehzb9Q}3$2Esb=>`7OoKc5*4gUyq=MR=H+og>0H0'
    '<2WCt8+|&i^Isr=av{re*H7>VGST+G3gzmoGYi&{)AbtI@+e|*TBthFuWbOZOC}4L8RzsMW)#f6?bq2b6?4n-WaTHSqq-}ETea^r)A?)FL6C$+'
    'pUYoGOWRw>H^0iEPAg)%#Ix-pm2Zl2(&;EmnK-K_1tNvGs|a{{<e%L$<T3;LeaT<ix7k`g<@1cMA2Hfxt9+)1f+@ErnVSQ&LYGo|@c=Q6f*ISg'
    ';joADm1nV7l&q_q5LStpEd;s#u_z{rg@~Eupzf+g@^E<}l-sp3L?=gEDb3-<h)1i7RiwR8nr0d)DUGxR!1E)5LmM|Jbr2vt>#eU196i~Xp~=A*'
    'Ickjw*o7SBo>3JWf2q+{<($FAyu$-62USh3txE70OS16Ez{Il6J4PaE@tHV!q|%t@f$zOQb{)A>z~rHZD=&G}-%Lnywqu_ye?66<$&zzWLQ{u+'
    'Xf#iw$5!|I<$}7QmaR^*BTXgZ>k(zj0iQ^x8ey5Cm_^yJ$6kSgiTrsdg7<=5aC)T7g}`iMkAFRt)ku4tqAD*VPN<M*i-9s@R>>pG7}++d`%&ZC'
    '=S}@oa@r2qpt_vjEw%Ti#`B|fYqmx*Y?u&;o5vYkWJwm0)1STaO}`qdHe2U@R~5P|p73&d8fSo|wY#wdGt_K<6<4x<m2=@btiE^hHUmyfc_=#-'
    'yic4zrr5@q$OY;-c&xZezbBIi7h+BvU(YOi9Cz<&#N+-Y^@H-lB50F_!*GwOhr4#sS0n^d&HS6i-mowC=BtF2IzL0Hbg2$p={hvO{xrYxDN#~V'
    '6_ptXBv-X<M*g?BJeaJBq*KD*Tc;5Tbel9+Hxo4fC>d}!>+H@$yz|Kh$Zzv$b(Q(A1ZqP5MuD~K@a?O{;N?6QIhHdqJ?tDZyQ_<Bf%f+h79c%R'
    'ec(a;A;DCser{41GgR%z)leOM!MV;dWb5A_G3$3Qf;={JxKq3mpYtl2vxNl<n$m2`ve=|r@)(20gWsbjvtNArxA`;>Kwi840&g0<m1Oaf$Ho-S'
    '0k(~=r-Z>GO6z?RwbxPWXV|i_Zr=7JlKnQQ8?scmq3EvblCrDmk-^CiG-o8c8EHk|8Md#V4>ZCET$8C6tWD5?9gUidp+3}TPo5lt&{6}2nJ+Nu'
    '*gE{E>x0~%neb9@UpDf_{WY(r29i*`-ad1kS-0~C;#=Y1pYPg&nQ!G*4__4^)0O<H70!UE`Pl;?xEL>)gg!7*PAB563tCiT4}VWxl3+L}caKP8'
    '=CWQn9R+B&%!9nuV?4r-#4@!<on)&kU)?Z9hpJ=CY-Gvz{p%>E8*$#vlFT}a=VS7d`O2YfeyYgRK9iqW-@uC@)n7?02-3ckPvuT2$r4rii&3m`'
    '&{d9H7P~sInx;;^g$$h6$Xuu;xU!(!k3cW&Md2MY8}b&)Lyx%Jg;+(SAkiHh7I4|TmEo$>FQD*9Ww62)l@fg?Ky=(PBrCTTxN&$KjW<oPASY>}'
    '%zQR#IQhcN30gxtp^#pK6y!{8-pSHj#|aHX(ZAND@oY)8{F1VNKGj%tpfOJu;OUsbjY^92l3uK33adHa3Bs>0nVp9lPn9Fk%4n%1=0Nxi6QoM('
    '6yfttfm_-1N!~(vNq1mQj^+D0ootcN*RpeEk?>`adYA+bD1qyk#s>=95=9oX>#*X`E%j`k{dVvz?tM=ZsH*6tX*xfgSqmU?aT*tktC8iEFU4U5'
    '47OgbK~o3gVk8N3(2F+3jKyZ1PJUqV@lcm$bR_UuWPSinP4>oHhegRv_N>QVvgQ754W2~pg?KUKh4P?Yr+^REF()bcP;EZ*l>Yx-HsiM3D}u0z'
    '<k`{EQ<_z}{mi_wkOE_t3P*`_;K@aYL2p0M<o=w}JcIT_KY6aZjAY_GpRuq(3YVPlgZ^mQDNVIm?oFRb`IPTEF2z;@-J3C`&SiO%ZkkoeKT@qR'
    '(*uvoc>5{cUO3lmd$?E<`T0ir+P3W}e_3Pkm6Fe^BC5j6E_W^|l3|^QB|nY4D})a3R>G=Y>Us<5P-#^yjBm5GC8=4%F`J!uh8aso!js3!6kCLg'
    'of#$t8-H{2=rUcRXHw3Kx(5tbP#sEr66T{-r?i$w@x3g9)L+84Bz4i7y{7}N)xmMqOfwAfZ}qU3B@9m<nV_mW@<pzZ?8ZO}&l5IA8WGrs|Jg}p'
    'Rc6+ZUuI$3Ayn9>$a40-jJIK@gR}#M>;|53_3*^!53|A_b^S1Vxug*WIPFTl7v2&7tiS%YueIrxC-!uYT!r+u9S{r9ANO%h7MNU0K9|u2CgC3M'
    '1s4AUnyH<E_2H*@>4DFSLDKT?cwke1I)U%7T6vM~e&sl5h!_uf7-2HUSxMrbf~pRFv$*}Fs}Fo*SoLFMV;-iWfg_^cso@Q)uKs{tX45eFZO=8M'
    'CQl0n)^S}^&VsSM_WYmEG^G@v#o~pU)B+6Fy$B*kZzkR`Jz3TmyX|mrp*sDj`Ssq*+)Xf?F)jufHLl$ns&kAY_w@9*Eda4vIQ6EQovmghQ>5>c'
    '{#jJ=@u$AJB63W_{6j$v-UM>UiNEo!&EISOGHT@8Qidohx9y*f^t>D3dQku8fgUgWYKtEZcdyO&ILZ#^N6p}x2PN^X!Q~IOyoWEH!10Vobamng'
    'YASMtdNGRzi^7JB<`18=;$Xs+K&C-+Ud|;+p|_-UkzYqz=x_&K;&EE|=nv+%*IxRk{@|t^p}yJ}Mduq9_r!`3U6(lUX<OH}WEi6PZVaSi;Eq%v'
    '#naLf?dcmgc2ui|hF@U~Wn^O!bOD~`#BifAqj2IsKG}Xs`a*0_<HJ84*bLPdycl-*+s?`IIk<^>s6RFs!64yd-L#TqaFyH;tx`hgvX$=aRn(s='
    '>ei(@aLp*q2F}nCUTgB8lHB(Hy_nIZddA$u8dX9YRIAHnzEW(<p>-WT;eKddPsvFLfP8W%|B|Fs*&gA6QGP3A(MH@@j!Kr!x0Nk^{BxdkwH9Be'
    '9R<kHo#MA~4XhsKRjgK~P4qDrAqP1<;cbw<vfB-9KV^=gB9zz&?C)k=_Zc&ybONgETYoy_R!7B|dW7}*bjiv`YN&dKNuo=EwHm==4jqadCOti%'
    'A)gb!CSe0bEojSsL=C(+BecZwD(suZfx-hR<#=t%4B##g<c0q;Q;rRmeGm&#76fKA<WtFUH%8_v(;QtR>&dTxLd8rC6iZF>cJhF~tG3;DAn4cb'
    '1T^Z@F<DAep-G?4fNG->4iM<Bat`;C;d5|H?-D8vo}%vKW9mgh5B5p-Csd|>9YP90rMFL*5mD;@uC*C}<8lZi00>N;{HR>FzKx{!Wx|6PU~{Yk'
    'pHFJ>8{W&csB*K9YmzKH`gXe-tw(?DwB<*h$op^B^pNToj#*Ah(5J)pRu7T;^>=-q9E$J{eHn>$-NKCYNt1_Tv}6>gsGaSRR8ktA*W$1J#v{ms'
    'D5rm&X>)xCk@MC1ki$H+Y(ta43X-2PF>TFfCGUy_Ec(Z$sjSLbmHjGd7WHin4*&g+&NO;+DJNxAEi*$w%g5Hs_Bb1+9cE*Fm~2cyOZHZgvMgs^'
    'NOU0kWc0K*gwEPVFhw6$X+EA;9j?66D=9bt6R_^~0~PAH0T_8niM|+lL<!;Bw(tSFH96UQN_8<v&#_Xn84$s9Lu5QPLJQ<aW?U<SdUOWq#xB)a'
    '(}rH@XWfVH?(>{%7a^n1@Y_KI+x*^Vcd$zq7<+b+@@thyKfyK7-_l0js?uiC{mVzgz2G;BeC+VC9g`tHhMWxd*HT2L^ED5=2Eni*Hy|(2Yx$#d'
    'r3UFbz&E>!3<i&>5Xh|R==p21J>58`Pn`7wLw-kjDJg?ZPJ%&~Dt{7%8n{@E9BwlWsBxK~tueC-nqp(DKI8hPSnj-*x(GIsf>Qr!_@^xv?;RIX'
    '?70vb^qq1=AkH6mdhMabN$4uEASR3Ir<Wtg-p9y~5Z^_;@b+esd-${~)CB%IJZr%k=5}hyzw9SU{ugwqz`_s8uW9@|YPQKDeP-?uDFBWXEj}I5'
    'n|XMcF}b4hnymD<MyYr|KcQkZ+cT_{Lo-WQS)_`YM)1`RuRX(;w=~r$Gff=TyLo-bw13(@`T3sZXSw+gy`<R<3{N@YPMaPv4-2`sannEBdNt6}'
    '(+-dE%0&Erlu<=?r{eQbvpKCJW&`RnHQR#WG-~lZ;NiRR^Ex0hlkE!iP_2PcK>b&^aBh4t5Ot-$qkN;Sg}Z7iWYZtzq0XI>z6Jp2>WyQGKCfq@'
    'fpX-qU&#P%vteI1j-#cFyJf~1hqM{BrN1#fd5_`6oLHehTylWdl?uu(8`kh=<qJM%hfYK40CqxI`W)?`$5V1>6du}m;w=0f=_f5cYq)-J3|D?<'
    'D|9#+`X3GrDew0#@f($zcm@u~<?cQ;@mg$H$t|pBbs)Oi`1t78i*x(^Sc6!f{*_`v$4eWw%$P#=%A9vV`ld++^=pzce}}Q(1){lsg&&j+oZO~@'
    'eVxxMj(Pa75?TKv(?$90KjYT9#0ZI`Kmc3TdIMG5vFMj8ib+i%aWE3TDTaa>mIEePBV42x{@HicJ=W|3KXMMq5pqL*#3*i`;MHtw%YOYSKr0U7'
    'KL3Z?BPj9$M$WA}&dOGQtPQ0ovEq)Squ=mCkgl@I$=pNIb?-ff14SuOEzCyGYH69y%$N`=%XhTtQRq`M>xuqkkIF7lJIZgoo()`k2{e#4uNeZ6'
    'P*`W?zo{V^@(@Qr#$I6sp(vhu5<BTDa=wK+T0R{b1%EyYxjHQ+Btec@-SfFYkewQCGI?c+efHl~8m8$mAXKN5v4*H2z&YP+%rQVXtm!v(MZOXz'
    'd5QyQXC~O5I`E?T*F-9C>p^e~mS!h@V%4!N!@+ZzkjdI!h~4ig$1{KiwVd5@NF?N1b7^dcwAh#ZRT`6C606SU66!(7j`C;QiFg(LKkG~;0KBW!'
    'pah!mO#Vf9C(#7l49z{X@o|t8K^Ho-^BoGKgFaK%9^5n!NK^R@dCVD?1r_cl)-+6F<U5m+FvuLhRE?WR5zz+)@C;>3S*KK=k5tVeAK7T8Vy`vL'
    'ag{%-!KOh)Bp06CwIlN#l9pA%)Jm5tCi?0sTHCv4XLKtpq=ygQ$<L}MrcdfteI)DXN?5(;j5Ao|ab6BUffR;LJXse|2pNt5x$si?t?7Q`2<hvr'
    '#96Nc-5vwPdYK+uHWipXo}#^f{j1Wp7ZU=~jY|zvZkX|`c1ZEgx^JK_-P{2%TRt{#i~w~<kzK(Z6`Wbgb)KoY>dhfs8`gYgQy{OTnIek&oVB?7'
    '7Qn=^e$St1MiYBgAIHcOP6l5KNM-h4DH*mLiu52y<NIyRv+)$aQ<FRn!f_^vgRbEPTx67pM}G%hE*rfy4OY0WRC*9A+vuwo>_Q@A)T)D9H{6Bi'
    's1&e=97&ul*=B_wu&x!V4*6uI+wx!dlA@h#{aK&mc<k49&m_f2>Fh_xuB=Pi`W;ss-RNw;atGzQp$`A23hq0S{8+s+_-o&B5egiZ(m7bIIz-~+'
    '?$_}s1nRLIJwoyS&V+u=&|a&^Z`_~sJDuD%7b{%`{%$m6(s53|DM*eW?Yl#l!4hkDJqscC&diGfr-}g8ASoeE;gm2%_>=ms>YPNXzmHC|%o~^_'
    '9gz0!nK(nfqurawCvW@`{bQjHJt~DQfty<X7`_ZqH$)P$Kxrdwy&ule4BNs|*BIdf)Mt7nQZ}+zcKzso`w4vrno^8kCrNLo<miq@<i9A8NS%u='
    '1r4e3NkWrU5!?5SG#QKaq03V-^^JCC7AZB6?hN-f%IogvdgaK#mCaU#a^c7)<6NpSu|Ege;YEa<uL^<FoN#?DzOmu}@pee<&M>#qo8j6ED`&>`'
    'Yx>9(jNpD1UEiId>U3Kict8@c!~n{WiD%|!0}lxD9<ck2aLB2SU24HLmfcI_jtr-n>U4?W?#JLyFNn029vYnFRz*FfkJB)Fb)(PEKxl{M;PV=?'
    '6!H^K|F)gH!b-l1tsu2;HPG(UmvcI0C!Q2hjBjgbEJ2*jpmBgHu(Bf_#M7}575#TqQX77f239UjxkUc&QREFXwY(7@pY-<JkFQS)^GK~hxyp}h'
    '0A^Iuq$;<3;30juRY&|6BviG)mQyUusY#WF`K$Rs`}~Y;o-pA7ivF4l|A1&QdC%qxLmrk=Z`goC(-;}CLHa<chr$abH-<cHF@S^HHKA#!Ru601'
    'Hm;q5xo-Ff<S#$Hp7d)X$+4Y3kYhO$HS9YXqXSMy=vczWP|n$A<Ty$I4*iuS$Roe4HFVwFX>H9J{xX$zWr1sm5=L+?$;rBBGavp771vxdt&5YA'
    'M)Dm#E1T&yND<+{M~y$v%?2j`_Ij=1%-WDR{lMqkog-_EFKE+M!ww$+?<w9Kf)E-#eX%-Y_~ZCM@Ems)|I4Hfn`bBAHsN7NUz+b+z<bJBQnRvS'
    'tGK7Nu+?qQ*S$)+*bPd?7ii$f<IIgSi)^)E6Ty7kvpsIna%b8A?^5jC5zir{IXNMZe6$v61o{km*+FSMZb?@h2QGGrG88;~r^oy>rVdG{o|Fmv'
    'J0+GKEL&pPLD@Q!M@inmV#)M;@|;g+I5YH55@|dyxJM%lZIHgwdzjZCimh3UKv({aDJhUFoF1-asm@SAW<@w6|8_lX@BvI6^Wfqwl$JL;+_AfO'
    '4l2dJbgzy%RcFO`4OPC`st}}W4=S<-h<}aHNRZjsOUhJ++vK!EJs0|EF|>-+bTWTiwEFG#pxD8&CKq~O9E_s>n`{R@Hl}D}>eR~!G0a?He)U8V'
    't|E+~L-^-^TmC^1a-K0^4c5A?INoenLj2t*=MNE5OuPq*knDX}+6^wB{lQ6*-rAEQ5ww+-64oGf%`@F=BWG|Mj?R$wuk@74!)$WfuSc#8qKNG%'
    'EKrCEGKmisGHeL(ymCQxV*}~pp88;pUv2duTWY!H4$R@S0S!;f=6X7*suC*Yad0o@%G(^1j@(oJbT(zdS5#WKMhGo=XH<Sa!B;cj&I<ELTx&A~'
    'YD>v(A<myJ4(Sl5OI$c9NE^8Uu|uqIZJE|&-Rqpq$7!-=3$jt-SLmfPF{6eRx3DkQm5R4JRC_lG%~AX5)@0!@=x4k+eD<en!X&f-dNOIn!SaH>'
    'eL8mnO0qO5<0fjirabY8omh|zyiO=uarLa3lUs7k6()>@jFM1cfut<SC4$`_9fvD1D)TjWCFvNAeD^G(f{DkAllY39qVJEzCmE8XWpPy<aNCU{'
    'J87{TQLHYWCS6jJ2%tjEM-9<mlDgS?Ej|Ul5LjMWW8WnPEpzecfukVwQ$v9wr<x|sC2i3_U0Ua-y5GP_m<PxeEjv@^5_&@Ds9RRRLDtxkPI5!D'
    'hmtw)x?zK=LL$>}EPe^OS5E4~lBaS<$$3}RR-b>J)~$aYA5n?Iht}E*N(exIyJTxV;?txAbD<8mmkZ8$;>%shCPr<wqy_lLx;4`%h;!Aml%)Pb'
    '6*|nHjgbKAZW;=ME9P*<CawTAFw5WRWkG4mF1DGdngmX2;O9NMD_9|ktw4@o%0T&cr4UT&I5}R+nZd|_ZcN%%e?+VNy!W+W`F@@oE(M=OK3)Cv'
    '84GU`{_mcsb322{xQQjTds@6XR|81zvNAwn)vQ!cxsz6u5Qwle0N~MYrithPcKY%I>R~F@5SojMG{37+z`7KFYvk~%VefTMKG#riYCm%XfdD&O'
    'uk&FFD6&1_j1-i6<$4}QV;yD_LWK_^YRGIs(krbyE%|d2>_?oLvzx9KAip|}`j4&(k24v8^Om!{bVvGGPA-M44ZftEIXnpW)VOOK5PC9vALjkq'
    '!i4OiM(H8gcxZA{)}~*ExR0Fi8FIr1?kDb@;S0~)Dht->-mrcGP&C~#LOVXlVa+st5Jp~##rLsV?`csoBL}JV@?RSjW{fB1m16ekapQQfexJ0L'
    'yTDoY$tT7N^udjA#~*w__rpbnB-wj?<dr+U!kL+o@f$@#2cK8_Q0=kNt=vGVD``u4TjmMAM0lRXQ063-+()U~et+w7V=&OmoH#VSz<~uQe)jl-'
    'Qm^(80hngc`Y4XIyV1nIBGQ48o&KSZ@-q1OX~^T*APy7lEH(4)(da|g!3cAbpAiK-Ge>%<QumZKT~$K(>NqvI5=oZ6>tl*KH6hav;Ef%$<<m{+'
    '+27n5B*RD3CgYDz+-@o?)MCM{6W<a}WnHqsI=Oxnwq=k7$w5{v;!(rX6`Yhsi?TzReSbWfAO0r!;`gW}vD;5pDi-bWKs_#QLrZ#qw_H}zlu6ey'
    '+@C`I02OHl2Nqts+1JWqraL31893r>yCeJk?488vsXM+?;TWfsIk;TM%Yi`B{)~~m1@a28y$CD;0&3bh&NM(DJQDDueDd5TEqtc>9C!=5q*pI5'
    'G3R%K$CmGukp5bUuC;lj+S&=|&bX2C@t0;>B-LZg+Di-l4$-nJ2aOSmYi?H{welIXXuWT8s?}uK{}67511jq)3G&!Zs`0S&R5J%hSx%E+gp~Tf'
    '8RS@JGJmRvr5xl%@?TgwX&o#0_F9{Vn@o}bYId`%KOAgnX{X}?q%W7=GlMJ3?^Z2d1WieU12V!kwxSU4S;pgG8sUMr3cZV)K1e#Z?MxKZK}o-t'
    'JZ$z*few!^q~1x@HWXI0e%r0?a)eXV9@iIugUI|0z*7TpOSj%<8B9u98gfUxd_zAzS2?ea=d^*)l?QOQkT=R1n{YO$LW$3aqfKPLzK%NdcQnj1'
    'dHJRClOOYXJ&b5ze~6Vfv0mB8j0LKPs{BvsN1D}U``IIZa;KL!gUm95<c;OxnI$x|de~>1>&}dP2aeFjh4i6JWDK5xu;8Lo5uOkm(K+m@+YyN+'
    'XGfiQ`#@uy(BhEry?qjlP=p;zPoo=^mw&A-liv-bb_sfv3YKK0ni+Y!TkzNu&*?*R_#Pyg>TmVKjzJS-fy!Ftz(EVG2v)e>T0%1=jkrGC?DjMo'
    'ERg5+r+gPr%K52?@uz!Ht7B=%k^zr^YNG%0`&e8eAebLhxK-VlB@I-^#ZuQ_hZb8sJui4(e;L1<R;I}jfC%!m0JzaJI!KI=Yww$0hjRNB7xwE&'
    'i^}eXYTysisXB%l{YfV4FM^32(k)z&l@Q3$XC6Iq0y_=9g{F(UjP9Ak-9&B6Sc-otpi{_W3Za1bNV1_8Q$^l~8}cQ0*ZGM*ZKSG{L?*~L2-7%b'
    '{<5Gu&57@NpCu8soUZajpBY+U`QAb%9G9#9Uvd#6{bZBB!U|Q_Xz|5{=KqqY|I{9cWJh?4@q);&xdTs49^Z13l!Nt&HPd;pv~|Y?PlHj=T-kaA'
    '$KzVQ(3B5AO;SE2>hyWP<UqZADB)V1axxHZUa1wRCcUJ2H{5~&%Y{bMQ(-uc?gPS0ro34AYG@Iqi0Na7+#343k_2{;`^hwJjqnhoI`ioiR;ZJF'
    'Ry5<9?GbioaK((Mt`^CI#VM1ZeuzQ2+0k-^O(72canjnOpk#Wlqcu`(GcUG|lIWe9uqhddK_e|8$dgq<H^!v?-N+8pL?V(K`rA9HodQVG<&PCU'
    'j`KowezRjx@!z<E#;03)2o>lcTW7NIfUzPE(hUxj3w(A4G$DOSYdeGO$>%87%bCt$r{C8Xe!K`47*f*1mdX2>&1Htq&bfJL0?Ctgrn))OyGF1^'
    '(Je_C<i{4;@R67zDS(fh{~xwG`O_=3no<STnGIYy4G{(ex9w3Je+xVl3`;Me!RJ3r-g2KaoE;Y%$XP+7EnoU*>E`W%SsBN7g+%O(5r1e3`Hb+h'
    'Sh`!($^YI8=QBLt;eds#`4XErGykC*CrJ$k|IQjdfxzi5sh%370&+xOHz@p1W7h_W#<{JE6s|C3k%AJnRB-=D%~@;DXeb3%;*M(2BS+whZVuoI'
    'iGAgNPXq0QT4pQKzU*~(fY$+!ohc?1<l_pe?q_l5UT)P3j+y&biIa$3|3+uqf?iOonU#6DD-o|XvV_o=hfaY0ruUx7Wj=*QS{Prn8S}5kDiQ)P'
    'M#r4F!LDY-!UyW}&i>(NVpr@*grGT`#&#9DC@*$R*HcJ~0;v*Z(R#B)RMW)zb!P3$(a7B3N@Sc5&($?;hh5-JK543+dLk82e&$od#+jU>@zR%m'
    'Px63mq6+ouOdj((1AvZf>63^YP&FOGDuOXXy3RcV$&HksRsIlJN*vzpo(zU)Js2jX$dlh<WTwtmS;(~TGq%_%viyL9m*yI;2Nxc8yS&yr<N$sA'
    'WnT@OTEV@x<&>AhXNDvP5;sb+!Pt!)j~k3h3U}~CiAs0#hd|{G-XN@rH!J7vdOxb-olK|D#um8}T+*~AVw@g!J#<0qWZ~djUYu%eV%5u~*-{O+'
    'Nn_?-f4W2jzmHVe_cI6u4WIL9Tx*7|V5A8!`K>$QFN2TULJU1bYbLds8gy%$IBICP<oDfyYQ`8ve!2ua#lB&$FnyC!1OK-Va5`S3mY(sG7ud{V'
    'Qq?kGnn6A+-@Y8aJv-t7r(L^x>r;D}81!gZUoGh*Z>a3Lg>q63R;O8i36DOh>4)Dgh@*+^9z8gIn3+-*BybWboaKb_o(GE!sJ^#aNFHxO=(02|'
    '(B?DHyyynxxSQeom42x2{X0R}I>~GHA)|4*E)<rx`LHKj$(vB1oL%owRAOf{vWk6$Is8qhT36okct}bk-HiE*2#r$WWZTEm8_5<idy@E@fZLD{'
    's_vH|@w+)(Orl)0hjFaXwE19~tkr#H5`G1P5f*uvy|F0>)gs2XnIu=zbIK(mkMoTokkgHHDpHhwqGiiz7PEZ?vCvkHeKp+9f}vAeR!Xz?xorhQ'
    '!+{LAywg(LK|aW1Rl7ogr-PL`0stC=L(3%S@|*Npf`}ZX!a5!S^CzPq80|&j7^q^G%-x`3lEPSii<(p`M<3`C;_KLegho@*P^XX7UOYpH%FD=o'
    'ykQy2VYzAeyfjcjTe1`h04UG@t^)ykWe8T@uNsWW^2P4H!|ntR@#v}WXcZdCGXl$y3blu;{2-vn765t%uD~P(s3>1|;2}X(n>P)wk@B;cy}n6L'
    'ivBXFK8kKlM|L`MVXWEy#LDIRl`6XRZ2q@4+wL2PwN37xUi~lAgzQA9!R7&7QJDX2q7F;=sG&fd>=vC$0Gdx=Q+t^paNxIq6{Gw69kR%4LnrK+'
    '*HFmiN*U8xV1#^^ldY18yBuYP+$gpT!Mt<j8wc;13J(}}-0MN6q!C6es7AB4%)QzPvR?r)M>i&gfKy}sl!A$2a>FzaizKFy4!=WPa&$7Cx~k;K'
    'x+mZ#DhH*jMW+T8w4MCelvD`&4O;dzhO2~{nsOWzTLv|Uv1uy+J^$o?T;pE`o+33y$<1C)4Xm5i!~UA_5Q5#E>ooS!*+F;_H$dq9*XRkN{G<u`'
    '5ISsC8LwD#QCwsSBB@ViQ0~BUC5t*3^#44UuB1)gNT6ZZc|#PVyu|E5GBlw4(JF7aIa|m&ZKK9@!X>V`rMF>~*L2<J6FI}QOjJsZ2%wcF;&9UR'
    'Fie*&c)4S*DiN(GQXImJNaqSGk68$o42@z{=~7KMh-1$hhCzhIP~Ha007jfr($I^E8EIHPpgxFbhCb%h4PI(;+75hC&L@MQUD_*v$Nn}sXLedX'
    'XcFOeGth(A_X+9(`8j6}R#H}E_)=LJuNG_E+H#HzF+i<FGbGc|6_nb%ikZfP-hOrRQk+CyxliK;03DRPjmaIsC4J(IhMDK}#izzm%yN60ZGJg#'
    'UE0a0=$`e#>7GUl{(P*F0JXZqmO`pfIz*3G<7-^0SFo?P{LeBWmY&LIp6%HomP{6l4!@N!&cU!%qH2uU;(a-de(GsWRM|eXJNzp*{2Ij5wB8al'
    '%2~*G;$(?WCtk-Fn_?C!jn{~rmt!BblmYU{d@!ukwwVn~DS85kR9^ZSj@SQFTEMZi#;?iJUHw?DI0YpPup8K8Ju(kIQ>he)AE#gPzONFJa295~'
    'vJ*U~^myxz>&rNd*4Ne`%4MF$7yXd(kc8c$CR}<L`A$Q44>QZ2z|@ywQ;G?lp)gOf)<T&#7QqHtiFg!c)J#6Zd0wnNhUK}y@%L=;aOnnvD@jUm'
    'wM6|bgBv602BH+^QLLI>360yQu*#{q@EK9Vj-tgv(D@C0%=MM(=Zcs)ayv28rxQFEVqx~`Cj)kM0@Y@JNAS|PjHQ&<h}p)sm^oACQ4x^8?d(ro'
    '$d5?AIP&D%nWB8I*dYCX#@;l|nH)K~-O9Ist9~#K``_60<3R#O(zD;zKb2HjnPddugV`}Vc!#W}_ahQbQ9uw|MzK+og4LD)N3EEE1sb$H1hMA('
    '4Eo%hA@Xe~3UAg3_KvAc%&t?I6ZYo~MR4^rBUk{Ll|YCu>fNeb@<7zY=N{t-PlgL+j+=){;8o4C8`_>F+hzdWsI&ItEAlGW3Q2oq&kyqhI=Id<'
    '89zxH9FBs-awtnI@w7{3%uwV(-1yz+qS_>CcCCLx84H1*R!*M~Cs&qMR>`-Rh8<EuT2)5*c{1H`TiA29O4T2R8$K+ixkZNRCHzpmVF+LX2@Qz}'
    '1@<_LnhrykT^6$L@krSfs6$uTV-G!`i6tLeOasmdE`QA0jmG^JH-3;GR}1x{H7zwlp(l#DrQ0k_9!p|3D28iE!OK9q(+|nZBdS>q%g@R%!BQls'
    '=Qn;Or@q{9p8USwhrNBXmkae1SCMNHz|H`NeD;PNvG+Pd$<N#=GUs;EL=Cbn)}1Qhu4%6k+z%J2Qhf?TW-fOg-e(2Bat^-U?A!lZ>xXFH9{zqm'
    'tcDY(#qhc6`A_3Y1^7BYRN~gCt`t?#G&eQv+ju{sr7j1w-GnLh08x3_x8KzyJAuDmXSnZIdSvEC8(cn8Q>c7|qA^fzuHS+K?$g3LBJL=wjQ6n}'
    '`(O#wdN4P1PvumQDUaE?+?Y^1kv#>%GW!NxIxM5j-ZhkLQLyF_Rt>ArkJSt3LQN<!_rr~^H5`*#ZE_uoFd^)gI6e70LFIRv4ND6)Mp9jPOF|>&'
    'N~JW&F+&EO;1DsW{hM}rtttz6f=qItq8aGxW|kv$U<H+aqgERH7P_VQa^D@J<7kl*ex#G2)7?AF*+idE1)$(^jB-vAdw50(?7O}`PUO`}(NIV#'
    'o>y7=AakRzJQ#1Ii7J8xVq1zEL4Tiei0bPeAs|C22s|xqNuU^{|3oQ$g9*HhO6^?FxoZRkSx}KaE5F3j^J0;<vm3hpFnioa)cM=ZzogyB?#t>&'
    'cuDZ^qCk&6X+SIUj#-gAvfUBlH`L@5h$I;AvY8dl{o)xs+}~&K%jM9+sWh!@5#okbXe9m#U)?^;GJok687Jj#MZ3a^2T-!(7GvC&BJ;2n!W3W^'
    '`Hpl!w=7obSwmAIJ1onbc^Ar!t@tf`>E@x2{-HO34>@&npMU}=7Rj+;(IJf(KwNvAaP+QOu{Tr$0TzkLc_-akYEaljX<t#EJ-98J@#5X;1P92Q'
    'NcX+9phds5QiQL1IM7%2f=OePrN~@s?^bBj-cVe}Ok%Xa?{L31au|ACvrKIBsR~1`!SJZ}p<Z@R&A{P-@6I0nDMn4FzN{?|wS6Ex^)ML7YPqWz'
    '@i9|!bLN)SKT+f92b9C<;MtQ)#Tb1p!Ml{@S&Y%OW_M+^^R>KVXFAJ-kE#V=X=y!_T5oV|$KOR?I3Fg7=cw4LP<FHq<FkNUTkcNF`Zdzf1@F>P'
    'O}w&8KP%eu#M{VV!xLevP?1(YG25RpS+F_gP0@BZ>j?{F5d)^;`bCl_7N$tog45KVmD6M)0_bNSj~&47j~tEF?deUW3*&p@D>_;6*{5-}=Ky8H'
    'inyus$4;xk1XrlHge0X%t6^mVxP!9@N12max$9}B1aJ?dc?;{ik)<#k?okZ#z3E0!$LuU-=}8*2_#Jpo9?8@<4;k3pmMJ>I`n*IxV_(_WWk>t_'
    'nn|&$r;J3@IF>B*AkvhYdN|kx){)Lg)W>!w=IZ}EU~IG%AfuoHL243|YhuEZu3J%b9pCwBURBn5Y!>8Vizd#VW-E81qzsT0_Go5Vx7XSS=lSyn'
    'Hg0&bEP{Z5MexhLMR{C7e)ogMpuyUG1YlC8cE_Fnq+VIUavmc@6QkpI2@c85f~@2i#OPl(3XBwH)v{;0c_i=^2kC!DJCgxt&lJrIXdlbj-1C;u'
    'ZR94m5iY<K@4*|~<+1MCn==!p%$+Ym@@l_=a!MO}<eHE~fvfB;Hff}(_U}f2vb7fgqz<#0`*@`{lH6I!<<7*uWOsOerxc(MV?YeFbFUZUtNci_'
    'iT@Mb8G6of-8|I0B24{!u=uIDYU=CYTV294xBoqbarG>s#`8(PT=>v-p6bLSW9`whdgeEPY55d`(_D7<iE;8#bP}t8F(zaz<cpqqELZl3=R21r'
    'mt>_Kss541%zh(dQO=bl6Jh&(>zM+6{kis9*CZ%a94~#%vj@i`7TMAM&&Nq+jViSmR1=jdF%E7oTFDP2iU04cUrJ_cDnTWMeDc6Fj3uZzEnS*9'
    't3R8#cOk;|vZHU96vH~lyozRwKT>b3;l*tvRf2YJ9mP^qq-538blI0w!!`IsEO)!gMV+`v*}e^$zaK5|glBHW{K0&4u>O*v5?I*ZN(Uz+UfrP_'
    '9I!aM{cF;JqYR_kkgYCCg<CW+il6gnYCad}IHC_90lFHN*?rr!yc`d|!xR=fR!TpG4AH{^{H0A~)g78TP&t)0Wqi(#^5phVz~Ii0PD;jdV~gJq'
    'J8skbquD(uJr;H_rqeN}otprf);~ub?p(KzBTt!i_~kvDiY6$&qr0D75|QNhqh4=S&HINTX9^jvgowezmT-%wQg}ZdOo{COWyJBx_8o4`N_E=8'
    'JpeeQqvZ;keS74FMhl^SZvrq`L`ny!BPXz=owG2$?JD(U{N3tQ>pucj2vKMD0xOm8xp!r&W4B{Fi@eYOu%~Mt)j0>ztRFkEU0gjMD#?kkybU-^'
    'wC+L8XG+@LD;~xt7qSCBGD+NiCrXxTA7F~;@zp98s~bgFT=4lyV_YWN#~5l}p!lbCH-_lM>}sC>gW#VY2_lggLh$MfQ_>rSAaBwW*HBnqfsy@w'
    'c@NcPPg<oFy-oJg!GXBNf*VizMSqdqnM;XBrS|n->;*X0nI)E`ytXwh!hG5%owwjch^4KXfP+}oetr)2V8{rb8Qd$6GD9D!2OR`4b&V8gm552&'
    'NlcAAYNZONzYZpvGf-EP>(cw%K*#oN;FxS8A}q{UA|q>ofOCkT9)5!*;r#FH-B$|@p?1+%L_QLt2!VLtc`c)^nIdLwC6#1I$|D=iS;SEeP~-KD'
    '{UmG>VNiyE91;9qM?M7};Q5Fy3MliXk$fwD5Z{U18aH_x!C#C<pP^$pqUO4%9j#4GfQ)Z<q;XnyUv-ti_D9ovW+vFKcHc3_)oSVcJL37XGGVPd'
    'tlSSCJ^#*i;IgzSqq^qpdKLqr^%5y?g}WLRk@U6vqD{k&cdFstxu~PTe`_cUL~6G)`ognogo09c4FvxnYkRuPX+#w*%OHEeDhtBQ{?GYoD<5p?'
    'cajR*1}>!e8?edIjc`>Iji_ZW*aF;wJFNqmGGVy;BGUX{O(F$zJt)OfIEt>gs!yDW5843Ch&Z)-d|U6KnB0m20a>7SAYt?FJvi)pK%gQmm*LsE'
    'Mj^ZV*OD`6L6+`44glUa`NL0CSIVu+jpvZueFcr-?%vD^Pl8WpNH@D&ARRS^M9Nn^CzJOLNdQ8NOQr?+mVkEp5@eSH+4prj+3!ylMr0)KqaTjf'
    'BUxNNUcTL4=T!O)d(Zs2$kxwH1Vdnd+zj;<&e!tsN_-v9*2boJ0liKH(}*3?TXr8&Z-)o8FG1M4Pxxe+6i+j<A}iARB<5SGCJkU%*3XN(%MQ`}'
    '!fSq)(a-rT`*XWM*8Cv2Gu4hF_I$KKP-GG9XTu;N#Sc01JJd&m$Eht<nTuGflyw2QD7!*pC`9ggVD{f@KDsQs8FP%<=J2g)YiWajMl+oL5J(d#'
    '4B=Y2e*_q*OA96Fj1nueU;brEv`EG%td1_;HjdEWla`CHWDRg2E2lN-Zz(}m3%w{HP)j?kX!a7LsYGO=K1em>Bxa{9zK>-)JK2J_cLjsfJooVN'
    '*rbRYn(9MRW24&Mk`RidwCecfPTOln7@h2o3Gq%_31SJWWc*O$c}ZDhl~(JBhky&BIXLOJ3C}E4^N?=N&Y3!7BVQ=1qg6KLPDKHqFD}~}h4g_B'
    'fch22`&1^1YpeODdDg&LO%SCS1H(4*k0CCHUflH5)to^HYYR3c_Q%fa0olG#TUo{B%e`dLBaMYCZP6*&$BsUz>+Hb(_Y?>_I;kN|zGM53qQ%RS'
    'H(vpe?f)U~igNZSDjs1AV)7OJyH@Yy%lYyKw-U>e-#^SMtTNk(T0<-D5JiZ>(ayJYQ&~WjgA@1-_#)8=MQ&2EAh--k{9?yaMy&ZS@Y0K*HVRp4'
    'Q`X>C6O0;VsnP#WPFvOdr_!6eo0E9uU<Jj%qp@_5Olu}zWi%OhSX&ZdfB3PR3uWjp+0N>kt`#xyinFGzdYt1!+m)A8J^h6*jU@krtlJN)Ba@O('
    'aK~^}Qgl#5346qK<%OSO6CrD19M7}t$ODVfHj5%<ClAoOtFQ`gWUWBMZl9x7D*K$%U>0&;3OMjw*65tJd%M>63(g**IR|gP&*_p7n}?az?&+!W'
    '(b%2MnDqcA(Rl6$!1x_l<3RYVW+N13Vzsd2z#g5QLirtGU_*w|B+kz%cHD(#sHPyN1DtUC06W^3bfSQ~mG(1XL^FJ>qveDWkr!^o<%d+0fW#f<'
    '`(Vo`EnP9j<2xw_goji4kFffMANX$07%VZ0>^Tl*`4rDXW-RjW+OfmxMW$2<%U}=NVNTX5q;tUixr)P~+Wlkg&HQakS;u}ng;t6XJtz8`>Gq?@'
    '=xC?3{a4C4r-{{0w+7lwBG^ktbLr`x-X$&tD>O)feSTPR5$uSDw98fODBmsv7d`xSweYg_Tpq3Z4+xJqtNHOHwN!+BYRwX^_!^zlD6?;Fofl?}'
    'bj+k=c>au_WomYI%-Os0mG;zJVJl+TzJ}JU4ge-!!~st9tY`k<HJ>r30Qa8Yl^?@lX_rUES}N36BUDRD{qimC&CnPRek=fC0r@4wO%Q1hmrU<*'
    'y4uyEt_iQWJ%)lRAYvekCDqsA0lp6qO|pkw9_mR-dF?U<nb;<Cq=RX$SDN!Cd;<fOdU_O>ch-<Pnu$}tZ~P<(XWso#;guO7n~lZd*rEkR23L}_'
    'wjCct>mh2yeyJwo2P=V?X^G(wNZM;}`-_yGRwD@&t}asC_Z?m!qfM>PC~OmHxcb&6<Ra0x<tOPYlCaomXvOwAV_8~_q<;oWE}U^8$`0UwX!U*B'
    'a_okH6EhWdq~F^aH<38^>@H>L(<MxuRZ9t0fU7d)*m`Z}D#>QP*h4AQ!swgCadY`9zR2vCgz;DDffgRtko<QSEyG$`e08=5{kQFTM8&(m)8c^n'
    'TJNUfUmEHban9^t1%=sS5)H41Kc~?#y`7{av&RlVtjcl?sq}~n0LVzIZ)t!NPK~VkyV>3?z)y=fxzefvjTlYWduTC}sWq2SS7o=-?CL&&_$9}*'
    '{l#Qgt!NxqH@dNTSqB|6<<cMam$!Q{ez;-V{pCKn)ii)7oLzAQB<Sm*<ls3qy87`g{{F=c^EDIK&&<oZbbW{hbNz0n-N#&Ze&;aykb|2ypH(%M'
    'p&Wqso0}kV)l}132)LBwqlO?cLMv4Vj<0<kjT}O<Jq2}OO>>HOhxYuw;d=#71VilOJ+v34&QU_`F7qI@aWgI1XTke7=>TaW*bChplD*U9sY)b`'
    'zL;g-f>QJsEsx93M_5&M6{j|4!xd$?Bl%84EEWxo%gxIpWM#LxcO*z}0C*1=n$s`wMvHd0lB^tp3L!YnYy5EttmG0REVTD(g%@eX6m(tdj0D(?'
    'iZ$KWHaoIr6`-!IikO{Dg|1mocEOFgB1@>x8P}as0GJ(noS@A~Ay#TYD6=!C92ogiWo^Q3su)Boj`13#gMV1FJm^Vh2X?j-#+7FXzd~@huWpwB'
    '*R*dE8^FR8bfo?iRa&DVxhB2e%rC;jnbY-1LP|_>zc$6{V%#!BgNV~~wiceyiA=K#h32h8iY`_q`_Xo8)Ii{sD*hTVMc@^jss@5ulv52sCn#!D'
    'I&h@|c2en?R^L0&JZd}a(Eo@Mb^Er&x?yFidC#y7x8E_#TL^o4=DFe@_9^>E<Jv?L&gOrrzQZ5yJ|Yn<^Gth`rc55OaS0TI4$!{$z}VGbsoYv0'
    'e|Smj(pEc?dbA-?SBj)Eop}xzMeFB=v`)h%d?`FS5&@x7k$w(5$o}CL4cFllUsx8#8O(x}^gN)De(?QY6OV?|>=9bEXEwyOdKFL;egYj{jZSI2'
    '38Tmai+s#JHQ9l8X3aoKgCJ=rDsSLc)Bpo=dB^AxBV~95)w-qcej2MgF?N1dkCdgrX!aH}wXc@zg~?42S~Kbmfkv(&TZ&BDMwVOYC0UAQOKW3R'
    'l2O3|Hd5~6fnT7vkV65pqt_(rpH=!fdcC4SVx1#4J6g~!sA1RJ3%MNgR!b45%mquwn%QwQr#c0RV-E(y(&(c~s1y4OeE2Jsh5+9D{s`^S4Q{nC'
    'oLN+z1Cac|nMWim0W^;JuH8q{Zd@b_5@7A!ZDh_uqFH%)?&gx}R^q7KPu@!qADcagG*=PM6-STs`_T`Yom^Wu9NM%QXXWn|_p@m%Z_QRepZ(YS'
    'kN0%zL<T$6jW`E9vip2e?dy+|#h(8>?ukdPX+B51hdq`atq&<$82F`j!T-qj%aCA9J3rfuctc;ovahs)JgYR}gR80j>hQSlCQF`js_@;d(aJ)D'
    '#j&;{)Nu1Wnj~o!*$wsUmh^b&Qm;`|???=f#k+>fA9cn(@rbUCLM7eTGiw%H4sw`Y=dA%)wpgRK!?=RURV=;=HYiKJIaq%0kcEGz%61%{!a4an'
    'RhEH8*V*)<%(?%7_Om!PJQO|RE541(#E`DiO)C97M)Y7a#7+ln;=xF|Z5dvhGGB<6`SwHi609?N1z7OH#7*JEV<NRDN=>GAP$s(!<C&PB$^Msr'
    'Aa(H%aJ&*ElWzJvy012Z;2`D^j27;Hh17)0;z8=d!ZpHCG|Kon&L?9fU6qfHU0iBA_*eNLHuKwj`oq;daQi6nP#^7FqSt{Et|)G!;K;8R-LbNR'
    'S%VP01({^ki?P8uP)8T?Zbm~DPw=q$wKVSGVdYY_Mn%G&JfNin{UP3uvYWCUoew1LXY0z#44>x(>hClkoQ``PZPrt&a&3lEYf!><ONi7n!MmOo'
    '#yEeWcN0hzQQ~<r)~V|~9(0F0@#zAPll=5})Im8&Gco`q2W&^=p^LhGiH(eD!5xqNWohYPnDcIu-cuUh__S~31$42dU*!@4etT+epMf&+WKsBu'
    'gPf*?WjG1nM_W=*c<xK?aM9Nq`I(!R@2i1jaF@7%K`cSq#~$M*tre%C+wr=l*l?x;0tI_$cQ8R&A-jWZ`e+i824KP}m7#4+*@1EiGu(CrjEtm_'
    'qDXvMPHZxF`e(`^K2TyFimGX*6>}|2Nv=dQrt59oAC-s4U9D0|oRW+|M}*{nKbQLOK0qL!T8o;Q^(#Bh%_XYMvBR*;8i7viwYlqI!n`w@jx&>Y'
    '@@p_gA6O$s>PoY)?|fc>Fd|>O`O6Fx+6_J3X$Cnci(R*JZVcb;1U$D_)O@}Q$3%2#zg>9szbO2a5hIi(i^EC-b2=8ss{#xTEbs^pu1>-ivI|nd'
    'wU>j2Z!h!i59@!R&EFC#V1?$GtqRnN)nY^q+0_4<Al-q5-DEer?&#5?Y?a~&80+tZeQy_aD>4it?xW=tebS5K?#6!Smk@8>ff1iUjskQhW^4HV'
    'Vl08+)+99X!^8NMWC=L&Ir?8BjYce#XjKPQc_2&i)NE{ITV|ukqi=$-BOO-_7A4T`5F6j;1JzBr2|{+wqQwom!^SzQ#N6g7nElqqT<A5R{XNm-'
    '_$;nbormK>Fx=P=AG3xLSdGlVDeYmga%bG^K4|z+t;2Oq-I1>s7AWqxSQxC&ldzQVq2B05hSHeq61`-n*BZVDZb>JT@k=Q%7dN-~Jvw;mHKQ#H'
    'Uf6x=ET^&TT$?7IEZ|mQC7j6ySQ?kh|J<lU601pW2*st1Dochm&QoZzqV+>M8sY&UZDlK_wDmsI#}Bh6umv}o6-rYA*#~KU>qg6Df=GyM2vR=_'
    'E|OlhHW&mc3@Jlg#1Rmslb6x&LyK&3k{A`eYtLkxHEdSwsxOB|{T*dLAG7-oKV`dAny@{Y*EPFETL0G}?HU2Qyyd317h#QKQTLtF9Q8F0{Q>XR'
    'I;GcVu*^NRt_KhG<eYIDC3AzJ04?jp>MEK&oOY-=pswn~8s5R#gsV6{f{T+^MB!73Bf0JPA@psTU9Y*#IA@4exaNM_19Kzs8c*doNr+#;@EjIN'
    'f5!r4peqxvk2E9IGB>fc6q;I9t@zb-HsH;@g4dzA$Xk_jMA8h^4b%;tH#%yexcBm|Y4d+PEK9pv`=#x~nqd?*XPFW{LC^FvHbAaVbZ3fV#t|Os'
    '?`ZdMJ}v+{7S9`qEbVqPctV=~!R)Wnhw37|W@+D#HGt!_)TN0|E0?CcwjS+wod(RPapcc1sk%O%Ona_VMDOqJhnI9gyDR_#d`45PUSlS7g-2G!'
    '*&(!k*I(OLTth$;kVGr-KcB?5#joLP>K&o~oIY8%28K<0UPM8O(WA`~F#sG3rpIk(X@B>=l`3fBRSn)JJ8nyFANB(~t~5E-oBSFXR34o46&@t_'
    '@~H6J&cU<0{`$1+jpy^ar%$&B{Vkw(V}#WZdB#^FZHj97?V`$e4VWJ;O>co|3Y7ckx_cvmIxASty378GYyY0%Ik(`r4=bWaD3ax-E)Zpwaxhu~'
    'JCLthV)75h%^-F6I(mZNo_J$RM9H7tGDT-1PG!aI0)9EfyK%E^k)g~7FPhKqoO~RZq9|NG*L|Zpy&<f3`{*ya>!nLzs$=;$+i}FQ`fX$_FA=&f'
    'jNdy-IoqVv3Cxzquv{Nivww5!GXo;Gt>^aZ?XyN@Kfb`Ts3EW`y`NebR0XqYXp&>re7j9LSD;Gu{2GabnTHVawQI_J+c!e9(79SW)-GP!D!%o2'
    'sq6RpwD7rK{Vy)Blt&i!x^-eB6ceKqXiqC$UqQCefJ$6UmsM+%{nxOwZFi9aRj{JD2C?uM?+#7uby0zplo6Lb+$k_E(`unROgGrHc?G4{#jvv='
    'g=;M)Z1VuP81gK^^rIKq3=_l9zD+1X6(9y$AA37Qb?CIxsF+UHl8vO520ee4nI4bA*=l^lugH&(DeYo3Hzj~5@h0RRXr(A<>?Pb9Sl-9=Uqb05'
    'HEBeXW$M4fnfIq1=&$a4?O)tGwBcJ%2ZV3@k+96YLcCm9dY~2Jd%ArZ{-M$C4|9X~Y7u00GvS>}afP67g|X=7qw$B#qwp<=EjDpEw10a5P`hp%'
    'm-0|=E(5vJI4yRR{r&6JvU>(;uHtPyt~TVHZ<3(e5+5)R^&K$TTgZ7A%BAfWMq(l}v5m0wRxwX*XC&0JE*?z5M=id35zUC*D_{*ql<h*C*pPsB'
    'gvd#FNq;!R8Tc`I`;gPvvytVIMufO?jc#zCOOMOiCtRgSPrLEU2m4_kvM-PLJUf~PH+~iF)@s_5+hhDsYDKy_Fvo3QS|5~^93<ea$3~UqLta){'
    '&!xeVkgoHN_bgVtjwP`qL{_vL+rdJ_>Yc!x4zXtY8e1x*3LvTqYXSCQ5Rkw&1McVOS4Yk^eJgF8<iRj51ZVba0^NALhj-$!-4hYx)!|Z2v3Ylx'
    '*-J9Ce1;c}rOAy*I+YnDa#f|w+7B;dMRD{_7#>6ZJ79NaKiXlaH94@R^ZX`q>ts^vNUte9EjGWo`|qZO!)zZ~Qg$WDxC?GLg4WOmkf!G%CARa-'
    '3fWKCt8=LhjqzusNO!=ngHB~ClX$d_Ezg5t%ZN5Dt>7#>w?F#;6pu;xh45<Y3Lj}JU6LJUs$xlW#9dA-#B@8Fv@VdS+A4V}l*FdtJZ`H-n-&Or'
    'BQ<fsqv#NGnYCTO&}#@BwW?zY!|xGg+!%{Sq;nu0eg+M`z`b`;UV!^W1?ft>=#efyFX>AuO)fJKKM=2FaD^j<w!5yjDdrID^zCYKoL2tW9lq56'
    'Xc6r0(?lt-JN=@ANQfAtMFo#5$DX?5k^fuE(Yk2q?VToUB+9a{9tpc;rHy#_$p)@IT*5Iw>yFVuqH29LD78uY2q|gv356`*k5&OYB^m<+&IbU<'
    '{eg+;#XS3AJaQCv*|jFe;0lrVzmS$Vz{k?;&W21Si_9$%6d&bDWoMSAQ<d=yrZrVcKhklFe3?A$d&xI&w7lYh+PT%(ACwWgVbIc~Ei@?fQ@PD6'
    'Xlbq%iHd2u<gMguyDsod$`31aslUbp${R-FH679tEq$t&J(SVpv=O{Z1JU_Nnl(fq^19PCDuh}i@+eHovabmxyWLA5!WEB24Lvw^@Z0~4vv%*@'
    '?zpqT_jro@qEUYXQ;-~_OW~aOs_p-~pPw}V&;*Hry!>q4MwNvYyEEaXSV>N`7kg<Ny1lHikPz-#`yVLj3C32E3LYfaiJpDDVelhE^nY=U^2Jhh'
    'ELC^9;|{H!_lgF1WV+-r)+eT*T{8ih#WR&2IxPTL$&|4bAXjz#lX-y9VAzK(&I-E<5c~;xiB5WzKsaRg9{Zto3|iPU>Jbm~IcEo#{7xC_uKps)'
    '3L9H~k(xZA(<y;FguuW4Y$(ekEzJ~%!kOKPWqM?!ML96u-aK-d?Q==bu-HwxooJpEuEP_4yDrLTd#4=ubw)YFuclu(l((BnI=kh}fqea%{2q28'
    '^*^#`+hBPsCk!8a{n*OkDa|>G452F+8Z|WNNhN!je9gD~`HRH^#;5k9ZeZMKU(@*Lxkdz6r+mfvD;(MK;y~Zv56?$xe?+sa@ews-HR?rr)t8?o'
    'npPH6<EiTWx!9E?jW97eGYHV7wD)beqzyCXRO@dA2S0+L(ncXy^rNXXF!lB;-JY{OYjC9uk2kz@6W&x7al%c6B0MA4;B(}(LzNyF#ZStKw-sb-'
    'm5@ha)oc}+AGcsCH`awY#|mJ@EX`qp%3q6iwA`wYp$S6gGD1D6{kbWLhrGM~aNqKbT-rTGv}Hw@-^HrCut_&X5c1STG&|VUvzCK%VgJyaXd^$2'
    'kyQwlm}EV4iEndzmJIKv`;lu1;Pp#I7C9atbb}iH?hC<66pvSthXLPI*K|~<fhmv;Br-d8&S#FjQ7c{WFXn(jpV=dL2fv^ks)?H-i}oq=h|g%g'
    '@(J%~WDilfFKHqe{j?A%3-)sg-K<ZiEWRJg&HAyd--2Vg<)CMYWG6PRotMK+Hjv|jSQf|}b&4L^{YQ5;!+?c3{+`hTopHctK?WocD;GJO|3mUf'
    '-8mX`&?6_+-}%r_B!?B$(I`=`k~hxX_aQqmdm$;y8hAzyW@J^iufU4lEE_|f!Th4$JA)Y)@Tl=fi9j#$vRH{wy0|1C|7_mhUz11M_oUlK`;X#t'
    '7PYm#qiK!cfvyB)U!y+j$5swqfKRj7&C#Nq&qlg&Sqd0RXwk1P#)>UE)`<GufpxvV5~v>l_<EP<TJj2O<l66`Hy&@M!)LJC)FYpzecmn&jHG+h'
    'p8Fg5N-@q(tRA5B^a?scl`et=)WS`?Xup$pvhn5svKTgm!%I(@X>QPvXS?E@#IXWCBXoR(dD46Hh+g_8*Nk;Xb<{|mwcTyJ4sH{RVA9im6R}*w'
    '1A0F$M#&=f!NgrgZ3fe^KpOjG(WO7mPHV2Ddhr$b2-sa^IK@>vV25uDj+=GbY{9XEetT%Z<|n7P_&ZqZC{bh=<*~d*1#yu^LD8kAQUhK955hj+'
    'X$H}BG#VYo&5=gb<8(GDd)%}-`;6<`((GSZs^Mk@Wm3kfyOfo02h48&LOWvk!vEBU#>mmr3=wA90sCspe)Z8S*E{b=nPdl07qwx+*S1SEwcE@Y'
    'WJh5I^_)%Sh1LPgv-JFiS=vi>#DxIxxd~+iI5M4Xu<>)jyVZo{4%$3?>Aef_P=~~bA{ua@L)l&3Ak=}EaaLc|OC$(|q*kU7zu}CuA_zfHv^~0z'
    'GRw1`Zw4hJ9yY*Bh@<v#?8gcHt=qQ;_Gr|firJ3UCf`!!UJMv#o=7zP*M3`Gt(~;XPzGQHn`bc5d8DRmU*RqaqV|?r_z&|mf*y$eVTswTo{~BQ'
    'i>GY}w0pn{`Cag%3A2w%9Rsj}+DalQwWip9?{EihWt+V#OAy}+Ieu4BZL{Ne1d8&EF6kpc9cEY-7!@po{fFw`-lvphbidH1<^J)*>@JPQ9QMJM'
    'UL>CQNjOZ$yhylfZ6jV6+E=O|JG)uRNczM6oJCvt1~qqoBzEp53PkR$(^P+zXq(6g5!>;V)Nw|Q==$?s5!liFQMG@99hX?7$;RtzMG3U_46b8x'
    'B0+hQHXxmV)i2glhdyPBo5NQGO=@NS83-+hSfpsM%K668xU-xc<k#Nh(Cu~7?@|X^_JB%0GzHZzf01ca8j>0bq=7R-Zr(#oiuPb+(foNveEJnC'
    '`Z~n<FTylm%&bKdju<@xJ5>I_Nt(!&tuNfAQlWyAi%133ssJv;G4N<QZlZE#e=x%Jw~sW21~A%hVqg0GVUMO`Mk{POimnn4dv@#fp@G5fkkz7J'
    '8WEutR^#-Ac5)yn6-Dje#w`Cg<v&c}Uo_sa5ffi#w0vWnzmH3|z|ykzPACU$S)aYe3bRHCf&Vmx5SH5CT_%RG-FC$8CbLQs+~`*-;4Kw(+9R~e'
    '@4sAK5NthiyWuPu;;?o<+lRto?Xie3yJx)P_gAg5jY9{Vf<VWPU!*Lp@wS~h*UaNC*NlmvhMHko!C7vU%^q6~NO)T7`vd*_ysRA~HB9`0UVD@y'
    '`D4GvXBhQ{NP3#q9z-M8EYe+9lm4F4c0AZAJ*phaxB!<jW|4%?l>=qQa@_7=X8S^DuOlI#$EVKy6=n=~POmHrk;dtMAs84Jizk)+EBGFBsr?B{'
    'Dph6<HMks*3lsRz%aQ7XNzW$+p7_q<<ie0}IMOY{DMUs=D|22guR3Z^VOr;|cN+o~-ipO8It>6Ptw(8{%lCU}N&ZvC3VshQtW12;BjVAsJ9=O$'
    '<<0HHwy(1+`{6V1(9mrq%Q1g1pNenw#E|c7ZB4jb!RcKDKJiMRO}C3d{~A^-5Nx+5U^PS!MU<p}nZZHF0hARM4L%->EmCIT?FpU^`U;*o$q60y'
    '2DS?tL<7aTbVQpO>=acLT$NClOP8d*$Y3EtX$E$~x46~ua=Qztgzh*I!B!%j5~7WpREYa_(!pL^Oap-T3p#&DGu!$<86sE5OG!}UA7K_gV4E%X'
    'b>ac7QL+v|fYR6$=4%J@b4c5rwVK>v_C|1?T*F}_S0n$viuP$yEsc-VX%3z=Bz+6IT8kXvWE<>*{jqv6r}6PobHUXpxeEmisG}x_bW^{fPa#LN'
    'BynXSc@s^1O&um|xeX%R*OSKeX@R$fZeLHd?+wbD5~*X~F?01G^oIfa-nkw1O`{5b1i??pp}nOwTvS3w%GW=~Fc8a2Jve@*rmOx6y7Ozn(yqL('
    'QRB*V%>GRlcM+w4yFEWwp)1(Ati`#ACC2_q-ccUKvZJC^;mMTlA?nw60L)>8agb;stYjC_4!=kpcN$2B%6#eae+cR5c69ShtZL8e4roafI<cT%'
    'DsE!8mO6$OxV`il9bTe4%>@JY(UvY(XWn;1^|pGY9y(DtrP}I$h)J0)QZwiEhutt|VFohZ9$jL4llwLz?;g<~?P`AghYg`iIwFy%l*p$*SO=*h'
    'a?+s1m|Og5Y+a(g12Dk-Va{5ZdDYn25!BQWSA_9$t!p)4Pni10=+*kCoI=~6Jz&~Qeui1J$JNq9kUd)TQe;9zXWdhiu{@^pk%F}ny8W)JAiBPy'
    'A+eMFz{?}^V+>Dt6WSK)Jv6(0-x5UsB^cz<9&{3di{={EaRlbB0`54W@;{B?)sL;MPC#@rqlb$Y^g;D*mGcX9P}bcXrZLlun(>N{e#z>eNmQ9-'
    '$5@-s4Ib-e-PR-Wns=T?>9%P-YM=@QTHEG6#WZvk<V!~i@CODDPzgSNp$0g}G0D<q;0j48B9%hrtXH<0Jt0lCnbTb%A;{jktW_?Frn0iRwO3{V'
    'KYFCwqc>taC=|TxAt`q<-Rus}X-7VVhRWxL(J~#fv-L3IymEymky>nIX~ZdrL#80*lUx1oRf``|m($~-C>X(-uS`o6nAw@7G&?%3y#n-9A0kn4'
    'aAaH2-Nat1r<5fln3FZ%ZY0@T{lXcvAB`j{f+m|N07+9|G~FA-7OAs|UR}N3`XM#qiHr6@oQfflj%mnzQSSer_m)t)(<wc8@=Ck4m_1MHEuHzM'
    '2%v0g=i~(^(GIQcbutIQH8ukL-|9XRKsQ3yo3oYdM0;KH<!f2=S$R~Dq1Ls@v1_$oj?9A|y_-N@u8>9Jmm@Qr(n^>C0%~(-k?k+$AE83p&cJf9'
    'L_T}5u4Rq8<H8?(An2dYT^`=8*D%fWkid|E|0<8r@yFRQ%S}@O*|nxP2s$ZP!P)U}>yiRTsuv#bs}=eh9b^stpxM#tvM))*I0&qy?${Lmq<+*a'
    'mmebKyu{>*V|wP_C68~wnBJr-?!eR`E*{wVWZecwv^`%f$yB;Rd6;`@7H=AXt=maO$2;xj5vxsxddaK<DQ)E_0b_$xIbO2wyl{LQsHpl3GhOU#'
    'x@K2^mu%ec&-!cL!MG{I?I<#2WMR8hX-yV+PVp|hR6+hU(ok2ul{6S&ez{*v;pAGel1dg9m}!u*%E7rj%p`)}ReO$x++XaV>~@gJ3n^&5M+Ue8'
    '$71L*{DY3(eP)p#=+RHm;U+Fb*0kWD+DDP)2Xqv4Xzz`EUhi(nl>;zOBnA^{Ux}`yP95Y4W5)ry5WFAB(PTf|Ut5hYQ3+*qS)yfYpe@=xkpY6$'
    '0IuLcBc$6(giCXw?IlF8S}rrBGLnXtPrd=EiQrL{R60Q(<Y?{aDOOn!*lw&iQ9Q3Ix{l~gn&6O&fG3g_V_Ub<X>sw77H(P~Ku^J%TK7`AdVZ;G'
    '&)@NBG-MoJmBGlBJgb_1I{w-SgAF}ty*~|pm|eMBwG-AI*y$bu9i4s9#M1nvA>YZXy=XMF(YS-)=v+no0~&DOQB!{55+Z#E9mdL<Xu@RZ;sbqY'
    '!RHyQr3f|s*jXU_TW7Ft*Ko5cB9B10>$aj*WxGppSLU^$A{TBc5=mM;`Ho?KQYSDTHVGO}_*qjJ=z^&hLfVlm6fdMjCT7RGb;dPVh6Amh*9_tg'
    '$7V<TxswZvk3C!DZ@}C(uPqDiUK%12rP3=eVbS*kO-}Z@K68HS#HLk%o$mjQhSHEt>k#SR#>Ct59hI<W+EL(7Aqs`}>^#uh<<&wC1tXeziIid7'
    'R{e-W=nH*&ouq?d!n2}+B_W_~dq~4Vz0+^$?`L#4>_FUaU!ag=3&C_ERrYr@K#f7xVkMhyLPxT<zdvS|^52M#5C|~jOx-fDz7s`l&B;E^U*+xs'
    '=~PpB3o@#wUD-LYWvNzDn?cYR<tDfLJEE@;yM>rYrZXM?+l*?;vNNt}0YWo>KjIiDl+Zfb4s*sBlPj{GWS=X2fF+s`Z02ZyM-tjhsY|?x-PV2G'
    '9|x1U*XYyQ3`xq7{cg9rO!Ygj#^I78;tz2AeRTVvFD6@PiNM6;kV1k@Cf=+{{;pRYC-DRH@~}I-=V16>Iomb<JtOFvbARYldl*HFv<Vl3gx#ab'
    '>Y5h5Ne>IBojKvW70D6^!OOA2GHU&I-r?N<ROEDXgSpA9Sw4Z}2D#k+7v#U)jRX%}&z@a*B}z%QmSVATkmL>0?{X2;>Jv;iq7>{7EmC>T8$8nW'
    'll&dr>C;Z0S_ugUWxS?f&$q{7uTWXa5Gf_uv=(}T>x_ZMpGGY?f^X7CZ53LwC}_e<$fV2H9bIeF1Ga0spiVe{N@`;G7g>g3S6YqNM^gNJz2lBX'
    'MvQ$=EeydKehP@L$|{GE#Mk;vJ4UeaXvvQN@%Y*w(O1`1Bn!ddO9o@cxVj4k&uZzKC61Q!W0g^(cm;Gg%8n>_L4@nR$zYB$)*%>?JvYvW1DbJV'
    'G32;fWbTlIhk4`X9C+D|q3jlnm7f+S!X?_Hi_mpwH-q@D!Re4i0~f0um;6fi0NKRPc4;^5X|et(hsZFZSgJaNv0FABt^lL#EfV6{e}-tS&nZRk'
    'T0xMKkV<q`gC)9=h{g4UZKz(PPpeQ`4#TX|5(Z5B&+DkZ{@L<m`$Z!U>dB>P*4f_cmaoe5i8+k6hm8+<bd8X)w@KB53@2(Zr{Nu4AUAIYy)`XG'
    'bQ*uFORXYgD3ipD&hNXg2M1BG1BW8zGESJ1<}tL`Tg~rse`kh8d4~>KN9dO8&;lhzM8#36F7J9na~ID-6@i!?Go+jxN&B?7W2ayH4nEznW_d@z'
    'P~>^!uh^7H8*VQX!#^<x6_CTtwjrE#5y&0j_Zk&bQ<<VY8ub(be%0G=oqO1Qkj+sqe0Du1LJ?3j%%~wi$_^x}iG`>7`9!;oBd@wcx8f#{UQ|1P'
    'JW`ctTc?18dlu%kYfDE=Y)ZWpVX`Uf9&udB*USUjM)Q5bQKqo5dE_<q^S?WAw5YK$;ZVBq0k~l(xl=%+v^UWH5BbD=!+F_F5(MoDw-;+f%M!Gt'
    'mlzB=j?w5lPoq3|-drP!ja>xxJL;@p`xUe$VWvRYZQp+`qZPOdbqQ#vHqlBDzDR!H{-f~NX_h2}ggFq!->0Q{o>Y#^G{6uvAQ&e}>P_=PSrDf1'
    'iME%_F4`k5g}uXxugmqag3Lcx-d8B$BUbG*@~}Ukl(6uyN65}mfqs$VR<2yUzVOu1$fL9F?_Ck;Jq-=ssHENR7QhA=EZb)zng^!Pqn7_ghSy+v'
    'YpT=0dUb0}`#nlszo9OicTf)I5jZF7V7J~XlmF|C$I_LE(whkfr^(LyazADJ^`qq<=YB_|Elw%B7RQnZ+XU3D2dv=^7b7iwh^h~RM6zrbMHwOT'
    'pH#w;N|QIqD!wdOw!aih2xsw)%?|)Hkq%Wnv<bvp#8#r4&h3_W4P4+YyBwX#QZn|?GN4O0ljO<lY^OiOa!LysKF{vCs~QQnZyc^KJ1L}pzhN#M'
    'bQVe*c5IcH;EARxLD#`#p_0ifqa}dYhg9q-wJgz7(1%Z!S5n)-W6Mo4JP78jH3iaNQYY-OSy$ZTE7$dbwCNG>o6v4L+Ib{*#Y0)rafx>FcOoqd'
    'C{Ms!X)0tVjnf)V&vO~62Q+C=KDCLB6Y)Vmk^!2AiMVCQ$bS$sH}47>t{t#fWDSR>Se4m>n6e2Y=8Cy#y0D{)e-X4kGtuCAMv<ORdqNO$OM8jm'
    '%S(`*B8wK8oqg(-#XYq1q(c$uYwcOIHUtyd&WzSa(tQYg8**s9Q1GufID1<a95)EcNuumv=a6=^V@21uu6-PoLIO&2G#5QPbtf48LpIGRi3yLW'
    'g((#O3VfkO$r+zA>4xnUmyeFOorD@HJ6N*Jc~*V)zERDe-*|ZCSEFwT|LopssA4~J6!aOFg!}3OZno8brO1|$`THm$x4>;Mcg$28PfWO24#`4X'
    'k}QdeItOwTKgjk+AbZbNf!u!5{|fRUqnE$19zXLL6qk0#l&OqGt6pjtaKNE6t_%*0EX3XSjoer}rJllC#ZED-Y!dL{6KMg3HGiISkeq49GfWn_'
    '3a#0e1KVnQZs58DXJ{t4dG0AnrDeZnIlutVFwI`t*3bVe;wpit!=Zo-J}#DFaC7t)X<(2G_VHFoRTM9FAx#lj+xN{*BU<e)J2&1XACH=+D$mOt'
    'ByTMRXC0%tj!I=$8O89cj0;RatCUt$i#qS)_U%tJge>pTs&y+T`wB<W=**Uk%o5JdgF0q9PG6V$h>XsIKrT`KFU;B^Q1`6g6y}@aW$W*xNUn3M'
    'l=d9+{xEyl(jGm;lQL!PTO{P4*bYjaJY&D%L=`M0^dKrZ&HKYVHmil=J1<sL*#WUHi(=5KB$(|X7Y!N24H_Y-0mfPu)t?v9JGdK|f0nIazzr%)'
    'wmTbd_sS7Z!l@1E&KLZJUE)`^R~%GLiB?kVMK2$OkT$y;-UB2mDIUxnt5NNU%1OlfjI~x5&7kjD=C9<Sd$5wgH%JD_ls7U#OCHk{&hC&ZqR~+A'
    'Dx)lUzv#Yqix`D;(ZKCL(h&OYnlvy;ZjDkq?H@zTxWLPfSwyA3z-V&+BgJAvSF4(TIWJ0@xuSh>Bcz29w9$%&bZw^clEnL<np&Xm;M%P<>+eW5'
    '_{ZnhnRbPe!$h+xr4`-3G%<`)I$ZzYKg@>VoNDj?&Q1yf8Xyl#WQB83`n*oC<~pPF#}E5vtJ&9d0zX}0?6G^}CWt#hG=z3D=)7zq2B)<N!xB9J'
    '=gDt-h%2y?=%Wl~@Up7zE?ncShx&gQX<JGN=6M7cUe)D7)Dr5-ft`%Hv_C9}E#&XOeHQr^P-I)%a{ZzCKORd|KZ8%Kmy0<Z82p!T>pt>Se7$V7'
    'TMm88Bw(Z@c{>ypW$*S+m^85jiM8fO+)$}nk!%E%dGv4>N1&<Z$Ki4BqL=bL%Zz->g@j<GJHggyQ?7}LJ7$H~ME9h_tLa=7rMd33h+Lkn#x6|~'
    'rKJ_`?ub42V&nE|J5)R_cBWY|oyqg(?<Ri{SKkBuD^=a6O9QaN()+4=P)UqDV4+^>J*g1q>>C8ix*CAr4#03kdPoeUcZrl}Nu<LIKTb6Du;O#J'
    '<6#A<SygYwCeFur(Nc>GWMm=XJ!kdR|FhV9&w0LAkJg;pGcUR}N73)~PcyBz#cg)Ej0ysnED~>WOH+ihWRc=Y#Ovskfp}bII%jI$FFbYm+|oRu'
    'y%KVQ7i49r!GM{zvFEQ-%kzekRN#12?)31<r?zUZwJ=|qu-taaYHz5s(6`;B9aS&5hk?V1i`5C!>;5p08>%BJtL=-o4s>OygRKynV$&X=T6zza'
    '9+fSdik@jZckS0d*dQIJsLDi}Qc_mBV+L|1BPTmLY7mpgN!^P_qFg%KM#!dDIqBAGhGjJu+f*y6c&nWHpH>_B59)twRVcrMEW68X>o59lWVR$9'
    'yca!}HwIHJs~1q&9VvnAD<Ns+sx?Tc&-o4|7BQmFrmB2S0hi}bE;W}iauJa%8Ea|$w&a)v;Lz3ePO~Qg61RZp7&0Tq=<KlT4q@xrwGfwO|J5_*'
    '=IhNtxZC4f-e-UNhSAqkt^N&3#|y@sqw3a;u};7JxWmU^o#f{OjX)6U$CkiNua~5hp?jJ8mnhHMjnG2!1qjsaLy|lw<x|UHSfM?DWxu_oKu`!9'
    '=t%<QC*!Xu5vp|PLE|blUqHtC?(fS=3_>G=SGO}Jcdf0!Luk;U3;W*5MW52845mc9vo4puLH2~9GNIf-=Tp3g?&anh+D&qd>3k8%o0}srN*SV3'
    'G&N!%(N59`icx#f&T^u^RR1|eEV0DS?L+2<FZDjM)uZc!ml==@QH6RjQkN%DUfEr~M}&$4ZC2|wJ3HHPmnL|S`i7kRI5s)EO1(ES-2ED!&z%yT'
    '9M(HDSuUp&0U@7p+Y#9nDd+_SxxP7JhL->wK;-ZE#;-DbGKap)KClv7$tz5kXFz4AX<Z(?BOv<bi^KLlgsi})AX>3Clm({{+#yt6VOoELs=?zt'
    'Sr@PlP6{|f5tu~QN9CEK%ul;}Y8GcT{4cU8PDxr2C5-)h4-QnBEz5=5x6c#=6uC&7E@4PHf#7$;F}QqX%AG4%f^hxUdBkeHB<hC0<2@kikE(gj'
    '6l9DMS)1ni05a}2ZV)ZF_q<lr{o4(1w}P)w>+6vuX;kEZcE_Z**YUfEu68q&R_@-sb&oq6Vpu{rlq1fPy$HQ)o*`O71*ck}f`unKW6?UHJH0eK'
    '+z}cTh&`sHNUqeW7%z~)iBHlQ4y^QcSLW#BT2|M!D}P1$L#FK#CmkhuAeJ5;VLylTRJ_4^D_3F{D*qZmWz))67(%&5?(v)=@U~7mGApxn+_Q=('
    'oeF8m=+RDk^bj*Lvu{RxX8@kqsyF%b?VJZc9toc3kU$=;r|v;v=ZeAiS9^`fAGmxzrs@0|2M&%pjgBF(S3mTXRwR!Bo##?|SLK5Y#Y+7@KIvFB'
    'Ni7O?p;SrnUM1%!mDPS@4EVR6w7&s`Mwe9!$8(P2D*<k$H@cPk)d7@JHW8e*0)I6Jv|V5JS;+Ly!C`scF@E6z57+2<U~iJhU)Tiu1B<(E7RF~&'
    'IFVQgk&U{2458X*wjw!#%34)DS`{i38vK*6>AW(#HeRTwFFhJ3bh-l-E1J;D-64Xnhbma%-**rS{hn`+0WB~0;u7C;nSc)>R$)!v|6JcI2@Ew='
    'QUBu|3FYjdYT6C4XK~)`udDk`qhkIp#}fmC_BI<k#2;MW7K=H8p;sXAZwXB%tBiMj@pee5UKx!ZtxGB85+Me%@LRBrWC1`fqSYnQ=nt*ZnpfQ5'
    'dG1q&CbRfLwT8Ra>^luHEcOo`4U45+Llj>1J3|tkHh8PUw(zUgNqFVAX8?iJmd@YoRme+Fcu+&qiXJV%P^C)hlwH`VooxbwW4D#ghzg1HRw!^J'
    'L4&WmKPxkFjR5KBWBMUzkn0D#ZwSqn#KtK63KRudW_?}{9nzgkYhvBqgL>Qj8OWrToRn68?_(MM%Io|d82W&<pR5nwg?)r=o^_N@P|}Ype4rQ`'
    ';yD6lmE~^qe3yXFn7u3C%0#m6G?!U)5jhHk<*4VkccC3ZI-q@a=PFkkcC)OIqMJw&S%Tns<VrW%5KqYp;uw86KS#Zjeulyj@HOM$j|dA|nGIA?'
    '<afm$!7k_$l0SEVX3sFg>nmXD`I-`%vU5pce8}{ZB?4J7<`;hjH=!g_0hGdSF{0EByUy}qBtjRf?!Fh%|3G6J;>IT9kDTKqIFfsDKFt#Q_E)|a'
    'AJ7Vj;p50VHfx>$HG*z{0l36`0<@QRaJhA-hZMZwVYORn@_j8YA@8!m2}!>e%jLAe`e0aILpy6#MiC_-BKBEBoct%{AX8MW-ZdwYVZqoDU5Prd'
    'G)Y18?Ml1RhRJLB9?tv;ep>3{AR?~_#&YUu|9Mg})lCXHLiV=PhcJ%+ryUY-?EZRKUG?#?K>nVRR^}GVN22x}GL(>FN|=+~Pw*5aLypR^|N2QD'
    'pmcwVj7Rdn34N0cNjsm*VPt*{gvba!6Mt7@`9k~J7xA2iGDquvlrD~nE~BzKlrv;72<m^)O0+6@bWL)^)TANFR=xGts`5wO!4{?ori*0hw9_?('
    '0eNdtCOVeV=IA6*WbD>cR!uzWfZjsfpxu#qtJnu_e|opKe&7RZ4Wm_<cDHgjLJz5Qp7pHNcDu%T%vhZ47FK95F5;EM>)b-ozgw!I^fbLkK$S|4'
    '7EIR}0#%rQ%%#e8m@sQe#4tBc@Px3$%!r4iY0ir@yDE702Z|GwcTRR}XCj!OPpH1b4%AxGh&piR#jSRu>S5<xXSk<XZ@3}RNXaTRS+ybXDj2M8'
    'tpwS#x5&(HZ4zOwSxLutp!l>|eBmQpgah8<n$?}?F4x;Oo~gvx5%tS%llGZq!2-W(I-Fetx$Q8d)T)*=q$%rwQI|TOHg?!B4bDy)2OZt1s*jMz'
    'Sx1fHgvvNQi-zYzqcKvy@c;J0HT{-|Ub2$sr^<-R3_v9=10qdyH?btJnnAd582Fc@!q`>ZN6DdWfQRcNwu?esfbI5)ka83WM6Elt5AwRQi+unH'
    '%=Cvb{|hfR;6xWH?}i(KB~DjAQN7k!a=<2h?xbOX<(hXK;(`r)M+XRigOANWGutAYtW7j#Nns?MoyGm8B0cf1ZCrOf-!b3ZjEH62sEFd!&oWBU'
    '&kMTK3aJGM^o3~y>xu*$MJ3uZ9x^=q|HLFm&N&~vrF6x|;Ll>}pnW`TeFlCbgcRgKm4vx3<Z&)ZSB`8t%V&}uHCc{Q!>a#}>_tZ=??6;4m8x^q'
    'K(<5^Fzy5KibhC+(xiy$QOZ*L&jUQK>B0sQ>d%h@q|FjWQQMo+x}MhdI;GpYkS4-87?xbn@bX8B6ywMnE%SF-?TTU80v|^v3c`;c#-6T3r!|s{'
    'J!O5spSLe-WnhUad#}iNY=<o_cGc`K*%EXCK`x_)JZ*$RyG!bIC+PgKGn|sfUiH@F0W;R3CF2@u;R|5#*0!%0TZ)HM^EWB1F92}7NFEXQpgHd*'
    'tF)^2q($69>^?(oVdfO?ns8<%ecJWzwQv}<RCY9;MqHRbjK-tfy?|%TPph?~>W5^de7ad2k7$tfipEaeR@Crq6vkKV$rQ$J-$(B5J6~&zG4e;R'
    'guO~oU{t4N|77Pouwu79M~m<@c7ZO{cv%%O_m(5GjF$6j-Tot6axXv?ZTNz@A_nnM>8!(nEisZ=HKx^P7}BR^)HW%H3b94D+V$Q6-W`IMvl7*M'
    '?;2Wx>=3ZGO{Tpb)pE@(74u6DS-g-H)?CWe!N{MlY{`coVc;Pc!lZ94Nn}<o%eaTey!B{gJCmL2k816K;uYNB>u#fWCNkzXG@S4acJx^iH4uZ{'
    'wUNEWFr)MD%j9yH`a*_1%7X@(uU#5>#<d-Zh#!4%J3fh5S>12CPRGhvYZPnIAuVLV+B1G#M>_Ohya1U!Y?8zjbV${ArZP1Jf*Rv#3JIHfML-N1'
    'ODJl*#2~vOZn4kRKT01^mYPP?W`ENbz7J|Ea?%wG0VMR(R*OBLi6JAx>a%gU)z9%k{c+a=zU+z>li1ybYwI0fDM~`A4yef`LM}N$vwwur3eM15'
    'To)%rB=8DvLCi4Q_h{`_QH|rHL~bwg*6Ifl!tTIYG(#y{(b>ZN);C5BhhIrmKY=mk8qutYXJ=ocMt*qG?dk{D6MQ!Q;7j)|6(w)=JgO%r2;0M%'
    'M-bJB;jZ3;#;fQtiwvJ8E9kyc^jJ^lrc1W!?2GnLp!0?3!WjXdhW)wd4zv)axm+LE^pwMWAb2`4;G={@`Kh-epV*8GHaPS5OZ`^kL3`c6glb$?'
    'Th~@T2I7SB$)@>_5Te4CdRH9^vbCxT7Ddp_mlX~{uvLm9pP8@qCHGrqinhk@YOWGQ-GzMY|6#V1_p>LDZ1ivl;&7I9cij9(swSW_^WQTJx9JGb'
    '5C4l*EQo1;RIAfa?a$58-sXYmqly$q`X+;+e&8M-o9_1<b2O(hD>GnJEea+>(U%PJyZW#nT5lWS`#<~Rl3)cIP?6kv1jYfOFLS)*I2yUxiyQ5G'
    'WOJKLJdQ~P_WLCGcWQR?+&wU<I6{YMqvUG&N_(6XGSB1LJFGqi<8^^d+4DQo{VV>?)k1*i_DI>DvixsSCWC9iE)zIL?Qkd~)uWjP(rIt;`G%di'
    '6j)T4>bWRX3{V8L^`-lS4Rt#ZFrgnFPebL!2SPmroWe3m%t#~e0uyfgB~QU%<wPReMI?oogCo61YpfxYn$%$N&-}0Lc;*OJ$@8StN+pSTk`F4U'
    '5eud(;f^x2#3?_9-0H9N@MwF=n&Blocq^tUALPigMuzoUm`#lj@y!_KlSgU1>(C!oN6LTfPbuW@ENrN7-xpHV%bwUrv_yXEDS{j{_rUiM#T`M!'
    'tXr5DSMdCfbX!wPx0C{Qk(E|NDVzu~H^`Q?Pvyn}6zk*~zeiT%mQ;fTj5+(XG#86t(cgn(zk(0PQPp9ho8!}dG(u53gh4#V>o7D?j->pagSaqo'
    'M`~K1#2{EwLKHzh^~$v}6v}{!LsIGV+h0wW9@Q-A=?vpmSaqa15zWd#o6qJlVt_va*a>eWdp}X_XK~gt2$V8pLDcD?vz9BeUq}p|%|=ALEOM6B'
    'ZGDHbI?Bc(VJ*)?*lws6uBgBKi&WiWW&>tXZuik%sGHHE$*J9R{lh{;Ab~s|R1)@!ME_yX9i_gZ|K2_C{7%t%3NLG!qs#>Euf6Q_*L)TGPVQcw'
    'iw4z3c<_+M_@adSK5>A?Na|#;qqgiFMq85-b*PEW3G>t_=LI3Nv(|9vu@Gm=M6J@}NgZx)G~uo5_Z4QCJ&KZ^iJ@4swk1M6z2@#e9by3!B{)7Z'
    '#3Y;oaWqq&706@Cc7bA%rx5k7L~3`s+*)!dWw%r5&0}F<YUjIEMpYl_+Dh29@Hw_Dq{qiybzO1Lkk1zQCx0mIB5N0`QJFquQO5Iv<S1YsJ$26?'
    'F+fUErfUHL(IYi`N?urAAr6tacW4UT8(~&kztSm;i};#knaFcmBo~ZwH{h2ts>fMhk9G)DqW~Z_dCpc1g|-%Uo8zrO{eLdd+*iZI?yNEDu79Z0'
    '_r-B97&?7y@f*0K_Sc&_913lPg&2<-RG}sda&+W9a*DUdnb`yjZ$%c2`P;GqKiPr66?6i=Rk3yto|XIdI}ZhdK@@%9SwZJ*zz1M;^$?@0Jve!m'
    '8m4g(#&dFinfijnEA@J0i0W)*goMb(Q7O7XyMp><s8wf9wZI1e{saeHmAAhVqbLtnvR9fDK8Q%6R5*zX-3O8L8M{^217(h(o^B^V?q!%62}8IX'
    '=IKl9nFRL3LmWsNd<$45U)ND<<4%hFLCmB%2JITS!!_F0979C&w=rNsm3tVjZNJhDR`R0(^hNtXuQ|Fhh^29q+#O28h$c>?hHW7`zF7shTmj(E'
    'Dspv-jvlGZnp3_-ow@x676ema)*f)cJwT(dC_X*Mp3Y3GA7dpz;E_DW-u>L_RSiF~=_|?WSln`7@mQ&}o7>5932C4p)BLaxkQ`QDTYt4be1{B`'
    'mrNZSw@NF)YTJ&ZbRRk|)`@w&EaGR+vI2Q(qw3eK_@o9sGXqak8oX-(S(YR%tCNywJ7;gQowuuq)ppd|QLUa7HCJ5Qb#fHfG(O`EV*MfFp}Cq8'
    'wM=MHKH1Ay3h<>x;gE#1*x^03Xr84=(qwb%ZwrZd#I%$uqMGGp+?21ExkoF3iESgH9Xfzs_#R};LH_%W=1DjO?L$2FD7JEqJRl7`Btf7?9RH1!'
    '=Sn@xGmt(@7C$Cul(XiK;2^bE^j<7Rt`@4i^nJfI^n5aGs}7_)$bU6$Pe;8)feCmq?ye7M9H`)a!|{+&1->+P44rUEv4?q(tnBkhX|eAp0Ajdg'
    '`hd$kK7ckL3t8KG^S-0(@Lc`>c>31Q6PjkjN`@B8o)FJpBsKUP+^u)2g)EAFXmDtd-dmRjrH)PV>XmupJb3Zae(DW-QqxI;@tm{_z*jr~+zlfF'
    'cw14&lu8)9#AaXOiIcYN11ZsdwmB>PCOmOU^r?f(IrC9sd<BQIot;_IKl{TJx+X>h0wt8EZZdo76yITS^?(j~-aqWC-Y9m3^eIyu8jGenO1l_K'
    'R6sM*pXpEm9(~N5_7t8f!M?N`(f08T^Dyt*KLfBV#1P-HC4{#U><sCav9-MaUA9Ng;}#PXLcfS9UFdO8ldlW0a`n&=J-RSIl4=g;dX#G(>JTOT'
    'Z8(+>j7gEg!qZJ4+SDU1$MRlA`^e!ySr)=OHiP6Pi7xaSMRjTRxU-ImTJORGE3}^t%;i=f7z#jW?8tWr*#TSIQDcxQRpen=))Vg)QmyRm9M%WR'
    '=-3AnGKaSRHTN(XZV@+El`4;QunBwdBeD6@^Mo4r1B>G5RCzp>6@e#+{i`GFT@AYky*Eig_1zP*w43EOpLOF4MrHS*wyc8l&W|)PP<DJ*98S;6'
    'udtYRU>Pf}8lmi2P>g=ff^e@ct+*qIP3zGQe{aD_(HhsIH2*_@$$C%dAK0}OrxDI-aw3!}6kVv;vs8_;CgUpx;#D4X*<d*&s0sA3V__~exCy~j'
    'q^Ev-X`E-tQ={TL(C`TG>S%r|AJe(cJSP6lQ+RH3d{!+acT5k0zoTQXp!==QfC_|*dn%4frkM0V%XvFB{$ZcR9iGl11%};kYp4@vz<66Dh&s$n'
    'Ql7C)eMO9PtC`c?Z&v%x@~S~Ewm}uiY!#8|`AQ!1pq-_Gk#;^4jt8|^o<xL2V57_#QPoDxGteKq+J%CUd|)?<OwO6;?vM;?z0cJ+`BZUQS$x(='
    'U_<S^58!_UfF3T_GbghSK8tn$ZDx95oqlsH>=wNyyzM!!_5w@%(T1frqIItQ(=sj_(O0s=K503!+1rfZVtHuRY01OiHkN=QV4DT0vgA080_m{k'
    '^{uPJV7g={R+}jnl};Arx?1SOqUJd3<umAjN#keJK)<2@KPUu!Md~}M6MGURaG)U~DmM+w!5RS@8ZYhiHO7Zrw=Y=-1Jg_+Si1uWWBh%G^qsVg'
    'T|Y*T2@FRl>Zo>{38&PD0B^M&!wPF5=ISj*yODG(4;R;eOLu$*jb;V!KZDcl`xO0_20kFsh;8ESSL6@J;N0&O&hx>xQ6iSn6O|S*DKPPV*!akh'
    'ExmpS5IvG2+BJ4(W#)?=q;HZbwv|JyGD0ESZaOji%K7F}v&$pxFxsVOFD12Q$#x-hF~qx+FY>hzD|PHL53{`y6g|bOVtr#g=&+Z#+`t|5A*@UG'
    'DgNPD^=EVY&HbUsN<)%P8XuJyMyq{8>#gkjFQMVcp%}wH=i|$Jy^t~ns;=(1<XSK0Yk0uCipmS@=@A;2ntcaFs^2w`!Cu0W?+aUtv=Ida*2wFY'
    '(xj8$S%-ZjKwDod{GYr-@SmDjPDuFf<9-G-2rH0B6abC(;%S;i2fH)shxXuTFTTjM{-Vu%5gz=5{lUeu;xi0c9>|||xi9=P>*xpJv4HaL5r-^S'
    'WHn+S0t$6ECfU>BS}?nFlG{SrD0W)Mm^TSdVY%`<K?aRNKih@cZNFsOck5Bf6lycD#JcZDLqe#F^3~#TTU*UHRtq<cb|$?S@AmN95mZ#f#`B(o'
    'RRhBHRRNvN_rGfg0X(LQcB;Acwj#-|eQ`tHLJp2%rPVy&5X7b(^c?K_t$bS&ipm{hWw=!3U<+S{+wbSub8Iewc6h>gzV?4AJRb4L5QlW3Y;{F6'
    'l41od&y&MYBf>O*?_cYbWq<<g!;(84dwmYmEQltqH+(AluIzFh5&hcF3a%CJILZAIHH24IcWa-0ycG$gmjl6)>Y=TcNTQcSx?Q~k3Q}deC%N2n'
    '0?mU}jc1a;PU}EZzxbK<{w)ckLRDFo1j-OsTXTx$QMjEBj*7n)>4!!a;qZ$W=aWbMjCMd9Z3H@J+=@lJ22@v<6)~l*m$8KMGruKCPq;^_{kNWl'
    '6VOrAUeO$8EPFfd<eg$~zS5R{0+tiluO^^%dwsz?-p`%&QL8x$NMmOs&|vf!RigyPChsyVD?jsSJ*gWb#vKG&X}fa@nPGNjSO*|L>L(ybth9ht'
    '<2>4hgY4R2J?V)I@jD3gG>g#1G7uFH@s8l`u3b<mNS1P-o=3%Q2lUM=Db7x)XBLg61Bd(!o@W$jISJ{<o%0`M(}Hc?*Kf%Ix?-HQDzZ7!WO_Lj'
    '2N|Ge-T4*JkVrMZ`hKeIHTL}z8u0~fM~#0ec;L!i^)rAwAwS-eV7-_TecMw4is*XT(C#IXD{Q#)Ywr0yjb?tb?qO%O9k{?Mm>H`nz^=?+V|vG1'
    'IAznwe8z6RTYx*3Gmg<k%)L)gfStP7D7HK;Oo&rtrpG!lE-TUpj69@|O!y%QRT#FthLdavdE)8XfX}Mr(T<0))P5fiA}6NC=VfaP^SzQ-+x32q'
    '`>z%rDC<>Z0^Hwg7a=jC%0EY#-N@j_e?E2u3iM`Y<N}(_4$}R$v^DrV3(ijD??PbQt{Tf)T`R`tF^*d)-p;z{^P1R6T;V(@iFBE9kClPIr1of)'
    'hC3-A9u2RNb%?8UNr%Nf8Lll*2?*N(l@w~$sNk3+EYz*xThnUqmF4Z2$n9H?f_qtj*?II4c?Aoget5K26i40v;|X<HNKVvw`O`nk*e_0`u&Zt_'
    '0Tj57`AnlZ{%~2<SbuIGWc>)ivXes-@ai8WG-nj}xu$<HaqC<CD;O_Y3zKL8_K2~_Eg5%NgN4|>j}cMu`L3((o?uT&(kRCcV$m>Qxx4v)p34v*'
    '{?JG~JN+0dhDxx6vs+wtlN#h{n_TH{HKw;hvC^M~LQ;l1krLNG>r(&mJAVge2bFn!Nd42&*^W!dUetik$C~9k;kq(m03tAkwDT#GTNC;sCR@W^'
    ')KmdhL4zV7q(^LJ>(^;2%DT32tWmoUb;AtnW@kKCrfHN5e|T``o#0Wjo-NGJ1P{o)R_t3R81pp_ED>75E}~LCA8V&4su_le?;d3p+l!ma2T9Lf'
    'f+ZGw$V)8<@Un}E<1=g9Y)98DGJ19HSl?knJCE_(U+=&#umL(9bBcB7Hb%ISo8-7sPuvYA<4&lAJ3f30NN+WrP<?dyVRoRjTqDD<EB@v0+f?mY'
    'Ce@Vr`xV6Uy*adcb(E`NAIwHE<YobQ@>k8EmA2|H*Q|MNP>XIX9j31i_@(_5`;QKrkFTrFyN2;$5A(gG<H34M?_?DYWVJ>ZK399v;|SrAHO3Sj'
    '--k}*Q)2;NeNIij#H=i%w1)7h{63f2=eN5`mB(RI4i^7!VrlmSbLg^@u-$nf*cB6+eHANWNE1JQf?-$;S46yBbeV|Q1Bo9Uyd4droeFp(m^%9{'
    '`_yS+TCivtEDn;CumjT=TAMSlwGSPvdlWZ{)}~0mWA~R`Yx0QVBw$b<N|h$fuyh(AZdw-7oSg-!jmXhSg+KR4HfsSs(PhO$f_P8=B=CCZhy9Z3'
    'oh<~J@6dWbH!Hh8pt)|-$hnqYLhF6zp%bZ+of0c6D8<Ly5j~7vKTSc>pzK&lNK5wLwEyx9T1zyRsYr_RGi2>&c!2h5=HK6$Q-NS9aVS9YIAj4='
    'L)b610|cji%+~tW0~Q$08%G;11`kj3j8YkJ3E6ox1$w;?k?Emb&KB^qpT4=&Atzgh^m<MWV2FhiCj>g!n=nMkGF>a#Z+DDF0w`rjl(uS)S<sa#'
    'p~R~KJA{Ns)?C0hU@O|?%C++p$nbrWXef3Gsr`0!5rA+e6p&og$GXO@#+-w*eG)T2#;#Y*{E)*Wmc_Gu>kMOAD#HRfpTVtM2YY>;&n=R&5a=_+'
    '1!@1Fk25xOfENhAX!J+1(;xC>c@cQ+ygEaSzf)NnX>4}gN<?UeL|x|HFcGdwj^@8uc;!DHq;sKKxT>rwuPXv@R-8T~Ux&RtXQFeU!L`?=?Pv)`'
    'uH(9y_1tY_X|H>@L;YVRM4jbK?%x$~7ck{R>cTP@B#s@sR}zpCDdDk!WEtKg?68wFG=f(s`1Qk`obHn80i9#<>i@=u@pz;(*RCI{@H#^725B(k'
    'qQ4O8mXAW(-qd|0IX}p;HZ@e%%4kM{BldTFZls-z{DB#uXHmzpH5S}@VcZxclq82DaF?tyx>_rxO9>8mcR{A6ssqkwxg3~Mz!wbz79gsZ(;sdU'
    'KIZsrfLiSK0S(e>y^}a(CNC1)69quhUDP@Fx}L{NGHZyh%o66!^;n++^{YQ5rNd6cqPu^M5#)!#yq-H;g`;h{sb7sGJnLhseAL_SmLBCKFtjK-'
    'GGTU>FRaOax}AYLvSr$H$94aPs67TB;&4B1#8g|bHzfXu#qz#;mdCw)owNMLd|=Ve5p_cq+X?a3W3DZ7w)RJaCo8y*Kx-P9s$s%d+*aBx;(oQ-'
    'v*dOacIf2|$SoY)cqW3T)XKM!u8JF`4W{vSPfR+JK2{Vb_d)Dv81&jlGL0KVqTpa~w+xcfdjE!`6Yo4FifeawM+K8cP%`Zj(V%eJ2O!{~=@(NJ'
    '-VG*eJ;J~GlxV~Lh+VrxTQYl0L1#!`v=sHfp>ofvrX}VdHGoL4+V36)Eu7J99lZXu0mV3tsMuos)8A3`v|W6?X{i?+T4d-2XZqQ1fAc->u-7sp'
    'GMV8f9F$U1H8z&aOM{t=>Mv98i!y+pGb#0M+ZivWCi{BRiugA;ZEzmtyU=BcUeXfH0ddMHY!1=6mJf0a8m6K=aL|}x6HbU#I{&}Jt-`yPj|?!>'
    'AtU^j2ul!SHaVZKD9iHyIrWJJK^UNAk9#UNjFPz}LoGRKsi)f`AN_(!Su0d<PJdX|%jL^4Epk}Q44-Q2JnWbjLpb8NzBSN3drnPM_}Du>cB}dz'
    'rm0S2lFZ78Ws5FRm`C1VbxZ3{G}-xKsGV~h;)||&N&B(x%8$y|epcEoCL91b5YG&)%<PiA@FanjV8X*1s%ot4c2%DY#Ar{(%+XWrg6yq`15bRB'
    'z%eX3M-lk7L2UKI&9dFoDMRDD&@9HvgKYgr9Y<PW4+|HSt^rV5hn^IBD(&hK>72c^n9tIC)8n>@7GpFg>@BH9lYg#DP`sL?_1S*#6)m0OO#m4B'
    '!wPQX(ZZ*?dX!Grry7lC`=6fbN({vZYC|5x*HhhUgjfrnYTqD^j~`ZY3HeCl8K8@f#U~XvWz7u$arpRlMW}o{P{8*VRe}GkVCCXZwNrUMp6a>P'
    '5<Xe!mWcaj-Dyij#WxjIfMoLR42<S?$%GIs15{ATsFe^|J4bjcdq&7G`}RgV<a2T#_@Am<08}k?{Pk2fp^WlS*FzEHsb)(Y2M14kMXS8>fQIdc'
    '$^X++ZS{xZBkUiLy)`7S5Lr|SXL;MRKiAfBg|<?XEh^<(l+_NPw|1$5C()l5=to!KW0NSd59Q*hcUMaFHa<XS@HhI3n#@Gx<oyrlml35LWB}8P'
    'vco{^h6ZFzIpp9COx<Ye5BqfB!;8^+qf_ZN4O^SVd%fR*91H3EtVqgJxmW&(e9$iT!|k0(^*4No;EmRWfR&l$%_ehhI}TF*{f=N<z^QqAE07~)'
    'js%(fLXGSrhiXYCv-$F$?7x}z?ss$J5!@|^hr>iShg8(?6w?93u_9pO(SoR@L`wLhka*JJ8u-X1jC_{UJ;!8eGP~vwQYX}&d}qX9zB&GKz;&|2'
    '957MRRE#D`*qxNNQX{Yfx<$7O9EpLwHJn&77D#CGb_Au~<V3D?d8&?SKhh;8-|Ubc1=_cNs6j`&%KPbJ<aKw<Ut9LC4be<K8+$#K^a0E(lUj=0'
    'vK|=^8^vea=ZVH*+}cfF`U|fFP#D?ryNu;4wJxtWiQu+lcSk1NUsx$b%{k_E^=+%eN6OIx!HqVjC(i8kzynFhEK;5|SI{JOSc)3e0P!}TkLrcr'
    'jc{-oxi3||47M@@R~bIAsIp2foaVvZFGYf{?II;3<@VSc(M@l`EFaUtco;WtZxwsWrO7_ltp!v!+U<rd<{Gtz=ZzHkVW_-QSWc6kTrcWiJkRvm'
    'x4@<$Pb&<F#G_n4;y3KJm{Vf3pGP_xd*FDRS?UZPtU07>90b@ziyllXzMQ@)N%)uhrZ;4yzP-0!Y0MLQl)C(fsbI@hta)f2Zuw}J$4WOpRvRC`'
    'CAJw&su*hlDyha%av?*VGhq3Cs2P9T2hvCMLEA?GCRgHGtU&3fIdx6&PQCTWJG`BdraI4Fs`|bvRNWGK4_OZOEquVr0O4R3+)3Xsd+jJhyCdeh'
    '#gd1WUk0d?65u_A9$=`PZQVc6&iAbt5fsICP)7<=a@z9oN4P|bEy-0(TX$g(X8skk(LE&l&x(?TvvXWU;uovw(;-rc7n%2WB$Ku#S&0|If_wM}'
    'BQNTE-@1dq^?cGLkb_yxvPytgT5bqwlC(NNHurV8M^=^wTKw-r+VyBeGRwvkpdd+b(GHP~X%mVg^o}iQBE+uND4PbED0$m?D`CW5n|zLZW^=Wy'
    'Xm`2hw)1Yt%HK}e*2UCf#}`Fu2>)k~|Dn`C#<MOalIN$|XON??<UF4^HC5Eu^C7c5y8#F<OTlw#_dThY*TxFx{KlC*jV^+OFlQZL&ldcaC$*e?'
    'o)2{GR0|WiNGSL)-?{UC%JQ@Bc<jOY9Un^mXj|-Fk|S>|cc4TJo?@Tv`mknCqF+v{`gf)U)llcT``{dR(aJA3H}<tc6*BYpU4$fq3Zs5%@_G$*'
    'b=jV+*4JHD02gj2yL4b&vL_}ys0q_9Ls#$}EvZOdhP|u@fArb4pjydlT?YJ=WCq&^9}a`H7Aq>dPFEMlpby6-13VM?J_AQRCRnRWK14MhMlIYh'
    'HdJ9ACmQm61{Dv(#sReVih!RIn_KL~v$9o2)X#Q&G(|mFKbFJu$0~+&$GwaFFe)6ykHnc4E`R_fFG^hNa-c{`#I$`&yQ~q5fbFgC5-HC$Ipth{'
    '9{Zv_f34OEbfI2vm6&{RDnakV4wxJ`R6Xl7;j6#fl7i;yx)N#SEAZnI+n-aN<#y242rTsBr>h`znY^c;i;?3dJ1p;hwr#6X=Or0RziLxfcFrr<'
    'H>TDO!n~KXx7mpme{9JIO8#-{20(=EqNZUJlk-_eBxT6ZO(Z~L!8M!r9Gy?(VAM*G5iu^978~a%>*L(?=*}FKiqrKL!Y@jrh%nE=XE~j0;;+dG'
    'w(23_>Gl`4o%=mFi^mX^NDIMJyzZzKeoD_b>e8p&gPefSo@1r(c&t3-6;*8~*gN+Sg-3PW=g&ldo3&3#N`jC9TRxRbM8P2WTMvAXKllw3qR4!o'
    'E|`g{r>Ok_d~<N9zvf#0e$vKr6e&2!zrLIr+Em8n{S-Ac%4@W+&Sdv?K@Au|=1Nxr$>Eh*goZoC@|XjJuUIM4_FZi7uGbz+B8KmEt>{y}G_;AK'
    'PJ@ZTE5fb~mmBXh(JIipo*TWM^r-N_e)!TU3)kYc1-$9xXlQ5buH8OomCg$c%0-IKrwnN;2z-LAkgCRFa9%VS^OL{5BQpe8AG^AAuVwE$x@QrN'
    'dH!4u=G4jeoI{hQ>~^a%rv?jZM@Ymycf|BjQFaEe`Q#M5ei&+~X!}AaOiL0kZ{1T#$In0rZ6j?yhQu!FeGx)77s)QcJ|GMDO)tTpwkYKDhgy9W'
    'F<}&Y%}0yyrO@!ml$EcP@WysUNf;|LJeEM-Q;^scosi=OYhtm5L2vhaR9u67Na;!etV{)1SU=K@Lwz6fL(0@Wgc+<jG2i5%^9n7tNDftng~q7d'
    'F0gxETnjUZJmFAbJoYQwmHNTjTaI9lT9H%}`vWmg5z#G4;bXW@XaB17>@EO5J;dj3kMI%BP(7}B!_B{1bNgIht+`(ao7XFex|cYHnS)NmPGSse'
    'r^`bsFlTv4xc;*m#mri<uOHoaUjkmNjB?k9DY5+qTh#w};K$7$1-G>bnYllU*RZ|*{+}R#;YhJ6U+8AhE2TCPo;(6pMTX40#DER`eU2=z0~ows'
    'QV4z^<Z&Gef_>1$GmHZj2e4FXjS2{yPvgEz-w{MiQ0<~?vjj5``h=pTp4BO}cL~`1_+g*8X4Xg#HT?}fM++?22e@BhqDE$#2GjFWF?L~M+OkBa'
    'J<`)a3RQi6Hbpg6QO?PR0c%;n*K=YTbKZ24lM6W+<p^LAM<Ja}IZOq$O1}#YmuEWfH*<q3+v%HO9FHA5`~Jb6`;#&CiN1e=uUo&`53si-&x;=^'
    '<kzltOBsk_rKgaaUOnA*Oj-PRuJ%0v;KIxx2&lNmH)Ty-Czbr(ufyjUyX>fGvrpQ-{BmkS-r*0?D$YfA-BGK#2t=2U`Z-#!mKrnsY+(vBGhPDi'
    '?ipbJYrRvmzXy9fIaX-g!3}BQ)*SY7YXHc);8!F2hn*{Xq#<JJ(vJqwmvGkYr4{Yiwmk|sdL?-cIMTk2=k_uo=YO8l*%O>w&r4%A@I%fx>b8t)'
    'Ev_iO+lzeZKa8rj<r+~?+1rjP&tV7kb``!4`|%~T{jxN>ioQt@r`^13T_OuBWBU)k{I@?3H3EX(WDi(~!s_>-E)TFKkWv<p0~4~y=B(N4zMJ+C'
    '6vhkDNp1b-m^Y)^t~91`Q%-6ZdDZZwrj5?yBj4>NwrrnX149f?>SnL)5*ecXPhF}14`Fd0Ue`{d2Pd_Cuvvs?Rt4IVx_5X|C*nE;G_mcr8S~0}'
    'QWC)U$0zML;UOtJh3|M(J*gd-AF*SAh@E$Z@#J8#x9>?k06MJ{6)w}$9#WHv2?TsT54H8o)<DF;Y2-CsYci$*D==^K_uo$H%CFedoqj;+$Xn*s'
    'R(V*_oWp2wdZrs<@X$qEJT3TjI+7g>Z>^n5BWshyrFdr$llR)5gAtV<xNG)_141f<+p}u5%D%5eMEnJIW(mXp*_2-h)1a`eA;QhRl2TUE=mkRj'
    '#@(_%QE@9!eymoAkq~vtn$UCQhqVf9F-lJyc-NOibTctNJh=ZR(SzAuZw3d#Osh^)msHd0hVtk=Y^YfF0rWbM?4&<Eu0*>_R}PIy9i;3PVsqrr'
    'buU*Ru8znzx`=Jgq7lmvZT*OY3mM4)7?iHo>c*?oqIyP>Ax~AJ`RsoswluEZ`{E>4PIm_UcOU>h0wxvGYAi5$cbV&P1NN7d{7>ZvcU&PyN0n}f'
    'BS$ncDR|0@%w3<5m9=g<L}C*ZO8XcUuW`>+UpQUUTlv)beFE#Kjh2jc26E_{w2T4x6+O@PhV5}=K+Y6ifT!+S@DFo9b3#|T9OU%~YVIn68qhj|'
    'KI^{%TVO$T{x%`?57ZZ>g}M(Mg8lM8w)Gif9?TAK?YHUV`x}B3b9;Du29rziPWNPoK%X=EpsELWFNuZA==~MK)O?Q+>P?})lqq~pAE#Y}^5ns@'
    '9wuaa+TGpXRF#)PwD(UTIUz#%-(x(7Tag+0A)<!lK0^5G&=9iu3QE3}G*_9us4KPxSWP7eR(jAf=l$qiL!BZKwLSL5>O0NkMG`AZuUS0eI)|!H'
    '(iQiWn{q_8Edpg;(Hfu-xNmq~m7;2=d8<iECwytUMk4CuhL<Y=ECdIs9vPC(s5vTFQX&^>@0*njz_z@Gt$3D+wpwv!YO<PisZs;dUu*vad_XeH'
    'tRkB(6Wg9TiJ(rL(T*Jd4#>%JcNg|CV|@~#TcbtZJCdsxZqls4<w{$24P3}C+KvtY#3VEa=Q1R9Mf3fJM=-W<0!ZNC5-zCIN1v1DQK0c#L>Z#n'
    'b>BS;?#a1vYh5jaQ}_C><&}CY!5-3`?*c-p2?MJ}YsmR5#!7yduN>MWDy;Lk{2fl)YejccL?KFtwl5n6@8_O1N?h#ZLhqaOwrJty@a*I`v=P#-'
    '?C39VWfivw;hi@a8yO7WcgPf>p(=NA$IjcwZv9_jo@_eV$F7sP?x_H5Uu0F>(C);k0ATJ#wctBM6h?Q&P1^emYNg_@-JF8SNKRc^m!p6|>0so)'
    'jLFR6XE|pFl*hf_F{*w$z>WyW5zg2oYW-G`k0r55TW&{o%RStRG2hE#Ik8s(27$~DVCA(EpLmH)k90T9znCItZ>xvA_i43<M*pbG%PW4Qza>=V'
    'zv)ex9s&~YGCQX?J`Wq3GW#Q*WBlu1#phD)BaA+ITN9q{mt-FmFHC&1pl-O40?6a*HBi`pm$`LBaDxm|#=cxZ&k?m3wy%FB#c`-K*p5RH(O_?7'
    'EZ6ONSg-WL@5%$YwLP#B4h6{jMz{ED%*Hj06MB?JTdD<6lma-NjyznE=3}f(EqUrRT*@hztUt^i+!{H9>=|SvGxQeE%71k2>L0Em@&$DeXZ1f?'
    'Ve!iPiajblGN|jZwe+w8U?fSX1=G<+7+`FN1w;Yqn_EW{BL#6{v>y;4iMpUTRk#@B+#-P$;QhlV(_R#-AkuNVSZB0v(f4fUW?DTeJt}tAApzC&'
    ';A!>aN_2f*iu2khLnpb^nD>yIB0-zhdOT_HU-ORPqHtwpSt-&q-aA78e(eu_PU`J=v8sq#yFi;NFnu;14e{(OOiPX}R&(p-P1&8u>Zb-s1TI0k'
    'vUFqwFgeleiZp=ah#u<#`K%={?t%15@?yq6Jk|;awSf5YoithlFy|Dm$y1#NQfN>dOA2@XfgRNpqe8L5VVVNn7c2GDuW-WJKCbd>rzZX*YNn|I'
    'D37M#M6G-#m{R^=7;tI+PPKiDn^XsopiCc<I>4ZgT{cBo(SsP?GvVkCg6yIxwsv#Ar;A@hOMH`CgZQTN>d(mo2J*$g_@N5|(!4isMv`r3dQXyq'
    '#@W--{}AEA>02F;D>=k=&jsiyB5LsCEa{)I6y04u;8Q?=E26-H2+ilCW~4>e^8tb3DGi@2tER#FTq*FQU&`lkDTzYAMASO6>&JyDOr+587>A&G'
    'V*5D#bHkx3^@EePgn|VtI|ou%_Q&>>JdMlqB(Xe4DE=Pt{uP4A(U6{c=xS3pP!DM+m{irYgFP>%LOW`Y#{|Adxu+$NKKsghZ?Tzi+0!-N)~_#g'
    'O?h0)c}4Oi7pPqze}}wn58=&kaAV{M%#5x+u#f9x)Ppq|?zFw@waqI$!(Gd^tE8UPa`)*E`}Atlxt{yZ0*(viLspbv!I3Bl*W^O>S|VK?JnQuK'
    'IsQNIM6JeAd$b+3JEAzExypEBWdl$?b3GXfv?xy$o5)t!_GbBr$}@k6WFxqk{m$9`zK=P)HAqEOMX5064`Z0@NnbnH8|w<xfe_%s>Lrb*tsWqe'
    'n+G1>PIQta^P!w<$TxBX=^LI>mETo6jmw#=`dqrTHgk^Fhu}qz7K<~C-Sio@R6Z%1iO8PIEnMgR5SZ0cu6>I=%g9llH#EcEgbkQo5+XC6!X5oe'
    '==g~xvNo!mMvYp$f{vT_oaNi+oe5rlE3>zzmn>YuQ*J{_w31qM&f0I}M9b5BRAR^H7NHA-m>$hbNS7>^GsEgkZR1Xf@-UrfV8Y!wtN2Dz!RwL!'
    'hYm3#U0~jQ*HVojs{4NUC_eA>N2E!jBEFCGzz8_T*;V^2C-t!M*r*HSmvo=7mLfp@YvA8{bRK1vvpobv)q8sX9$>`m0g6_hkP{#{jB?iLWWQad'
    'dS!dah%5{ju=CvdR)Nisirep)=TA=}QF)B=<~pAhF4=X%ws`I<h?r2k#BVs4Oo_^4tGVlDShOk7^1!qtmqRtB%nWN0Vt)1YQV1Bd8o!Krvri|N'
    'x>Aw-!(S02C}-CZ-<#McdHedt2lM9GfngVIsaX~zI22P)5|W_2kk28pr*x+eP3e@xtL6WI^xL2?B?l)DX$a?ZPTFCj(LkjhIAt4V!;b(`4-%=j'
    'Yq4gr*rgXpQ$DFRtb;awntQRJSRZRe;XtSZcYD`2(-p}`5|7gztxn8g9s@*&WD<?#P5ruo!X9xJ<m4duE4?f4%C6))wLokI{I$2JJ0zd09W9Ax'
    'YJvq=FYU7L(sBw4yauv$t(@-xg{Jtrlbq!@!}6a)$G+XeU;#T#wwE{q$Zm`7hG#jD@wOjlpIcnEPVrb4uN2wUS@iGH1bKxbCXGAgeK<@t`hu;N'
    'S*?uwA9AnAjjLszMk&=JJ{VixL$|@<!hB<&^kuMg>3~ZqLjM2S&TZGRWVx>I{)&D#x{>#b{>3m1*@grdl405Ye#eMKuAC%e*RHCTk6^U>oGNZ^'
    'GFN6~#GGT+{U~6{d*_yHt2>$8t-onJ=Z+o*);$3*YLM|b&MnptcS%1BeBzqzW^k?dx7z15tk%Jk*~KyEE(G}T6Xf6-ab(=LdlqhYKlJb-cUg>K'
    'M=;XE-)6^ozrSsM)0W~n3+rOt89N@e%-OZ?>kC0&yGJ8hDv@=Z_{?n3U5#U0AU!SWRyY@&N%vXJ#ObN2*B&tL{*hK1fBZh4p~w0=@w%1p1bbA%'
    '1-Sv%F4>DSW2U>9cRG9z9RNOZqK%qFCe%@(ZILW)jAqfG)h^eytdlfv8#CzHdpOMN=5%1^9cFWHIanKuJJ-zVtPT26J4yQzJyRq=O2|x;mY9sy'
    '>r)$u`G$V~Xu)cJ0jPVCV;@^Cx930G-^;b8;2i@nnYX0{f$Ej)d3ranTx$uA7lc^#45*n$b#9|EKb>1iZjus{)*MaRIv9ZRXSu-MymC9zED~o3'
    's~GoS$V!}(8TJN!9B})3I*uJXc(ynpar)J~LwY*dhCRvC>Q?c!q`JhD6E=mR>#x4f1p(jPL#dZPU+rygW-XVlk4>p3fyN0xgdIfLdUhc6K2A&u'
    'X2Ba+;O%(NkiDHENU|S|%lfgJ2f~P>ASyt1j}CxD<Ft2t_pl0Jh%=rn>2Yhnu?*U+r+qr>8UjC{Qo{6Okn!w*U-2{hVfXo1&CrwQVG%o_Cik;9'
    '(>%VFoUaf_K8I(1?`43*pEb(&t2x6JL0)iLOO3FdF#-T0nJ%br&uXUdYl41Ymx~}`l!?=ukaq(I`kzkEA<u@9zNA?;>XP;KY9_C~&QKm@dlNyS'
    'vi>UW>i$!!xm{_((%u91c95VQMtV2{r%)d7OO8+nLW1O|zg8M-n$z8HL8U>0^BDrV4i^FSy;oa-YX#Ma2z^APj@*{SpP{{~w>5>RgjMDW-8AzU'
    'v#1V8o*6@wlQr43vj?l$s?|mm#+Q5Zr1q#YlHpVOw0utDZd>zuDCa-*{!SS`Q<cy3AurV!NR|;Npm=upiv!ON0x3;5O0?i6>6T_%sR~dADz!Dl'
    'cw_Sp$-K^FyekYUs=`tDBEKofnZx26Q$rQ;iPNRW-A>w`o)P11><(*Rg-#5seb2(V9k6=KtYH=b3`%KQ!%t2(gdvl_Wkpvk{Mq~zjU&qL)sgWd'
    '-PWM>j;xzBs+SMeuDdVL?84g^QQMxpt$P1iKcralFE3UAmPT*7b}2}F_>8c-o_d=FR6<hi_7RwjVz)biMvTJ4jkTozft9J<)<4gAgY6rk@K`qj'
    '!ADQDz&V`tLE7$`Guk3%$Jv%{7Oz#VxZ>*z@pO+7d$|ZWmEzSK&v!cskwjx}Y9XR2f!I3oTKh5ZLuVW~_bP~XaJPu{7>Cv^sA|M~DF~)#_H4i5'
    '>f#}T)py@y9p%4EfJ!TPpjOM$AIJ&ak2K41c0D$v@=@Ka-cH!ya!0D1rWkcnlxK}8GL6G>aVW}SN_GD`4x-s2vCTqG)#R!<Nf~;M#+-yg_;;A%'
    '$1E#V=_7JWPj?b2ufF)TQm9QTp5dE|FhW4ei2H^ytv&7mH;iDB*2n2k;l<21+|7^t!R;=E>Ckq_-SsM;JnJm0Pr3kKh^@_RkAl8S?ZQjn{ABLi'
    'X{3N9G3hd$3f(?|#N@f37VhdoY<gMH9>aJE56h#r8u*cqShvwFToJdt*Sc~&fL1q8-tA!i_Y@1ji7dkC+fNr3DROzD+&n@K2W={!>lqt=?@7Rw'
    'q`MEf23q-$_-`V=OQu`<cv(j%*j48?_;~iW6(%{-aM364_bva*skHC2_7h`bhkc-<@D*5HwRM2~R-xx3fN?Ww+7El+$AV@P&2m|toc@RHhR$h#'
    '7ms*Dn}J3aaZDyGG@m|u_vqEuzg^J$m<p5XxIYqEgps}wuM8c1rHp7+&>x;#3JPsT3wmepe3AlENl%beF0VR5i9cTAk)9T4$IpS{-(6@{qujZE'
    'uS*LkdzJ+aWA?V57Qd0@8cvQW`$pHPBCz6lGJPvV>GhdA`d~`uRV86P!(BgL5#){(4V?%<r+v52f(B66Pd%2o*+#w4%XeBzJ|8OD*>VT3qSCU3'
    '7UiAT!_eS8$Yq9c9jueHpcw$x7Gm)1&`fCyAYfr(X(A79o67gIxS<berT-Lw7GkKLhZwr7Ixb~I`VGW@^_e;0xf~?(NPB;haXZ8D9^vL=NxsQN'
    '4C}WPl^$T2Du-s<7VHv<8sIIvS4;oNbCy4QZ3A2ua!(O}Cxmg5%vn7Fq6a&)eVL3>kKucfyWWpyr6If@Il6RPb-C4cLFCgIvP`WwgB%xzVL(?r'
    'J%x={r4&ty9DXE)y5#1G#^<_9WYcroU~Z0nr$-bQbaCi!&Xt=fia7p>jo87=2)k+`pL@v>N~-caYBlt^?F+2qxKcMxi_A)u+>Z_8Yoq}es5VNK'
    ';PAyZ(#0)fPZgP_(crg2^|=KJ-G4#52ujF2DG8C5$sZGrr4p;+kXDb`b0A4(3Gx*cfwn;SY*o#E{%}M|ham9RmaHGE`E4($=TyN-UIV4epBH?$'
    '*Op}*P*6Fy=bNUtnD$B$)-gMKTSNt1x$8aaD}<6?G2P|eux^j$e$r%{=Q{3lW9I5&`({fe40c!;M^4u7#LJL`wm6$7%|@KB;v%*)lwjm*rTP(8'
    'k2+Rfk%Fob$X4aX0EW3_(N(j~@(dASYd8x|(J{6<UZAcM3xA8WeF$hdm6RSoW`4;q#?dsxXSia<t;d5@Jd|hkm+r)9HS91D9F$TPZ?6KSiwBG3'
    'FG8BpIi)0Bkcsel9r5d>1)0${o7`5vU$RYleSy=b6h=#1eO<UGNIZjGngGB{!aE44y@P@UJ4-4rAC4<@<U-E0wA38(Gi`l5i#NZ_kvsH~$U!lY'
    'LsN0hXq`o_QtiSwPQJpx5SLQStzu4`b(7)*SHi(S`jSQ%X9aAtyu3#%UzffZL$DFn9ai6z)y~b(32Ldl@OZ~9$#}Cfqbk%)s^YoI=WGQU-6OGE'
    '^YgAeJqzABZ?@G4Zk+B_ip4IOVCp7w0dE^`;iOGqZexs)q_8}pOY`*l9jx{N^L>l(A?@a-embv@U@En-kQ6b4>(T}OFPDJX^*zKkl{+crfu312'
    'xU$%N%r4D4-ID>7z2i>Eq*H4JWcj@9?7#A9Wj?6Ci*zFqGa+ADQKB6{Pr@&~1gkDs?|tVswgBZj4_hR$0RvF4;nMRy%YCmM+%TZBC<ViHj)>g_'
    'Utpzk9ydj}@&V-aVB&nNX~J+%seu6=TgFb``oE#gZ!Z6`rtM=cN8W_Kw_pSc3CnhD9~v`C@$4Q3!4+=O=OWP=w48oNt@^ZtIkl{5f@f|0)o4v`'
    'Vuoa6cB$~1UvO;RE<KE`yR2#Z6=>i}zoR4w-_by*GJ<IjkA|;d+d*$iAUWN3txP2nZDc|MuF3Butg!=u_Sk)t1KGCHG>8v%M#oh_ln1RG+xhKG'
    'M#W(lo}7ZZHakft-R~G>h(tTkcd)G}PT?ZfWL>hc>&Hxdr=zZKS&rzYxe1LWI-6oeW0LzZ2~Ll0$p_N_ZBU(O2AR0=tu6LZFi`<7hbehS{~~@U'
    'yL{AJr-CK|8*mSWw(n^cgVf$8azCq6vu$H>E*;V^(f4ov&lJT>23)dM%qGK_b}E-*$5?Ikr`9wmEqkB@2-G;pg^FfZg3+E*w6|}$c~Fw<(`Z#r'
    '`pH+|85aUr(NyPg+fqHGCm5ab#8ya7pK53@@F1FQqps6X7HX;^%k(&~&!0;p`EnxPYx<Rw&YC81H%EYE`;O+Uw*UjlDxLr1C;v(g%|4B!whO1u'
    'E@ie3`+3i6);nkY-CdaMga_RySw|Uc;K3wx+ghwB4%f65NEdMR2AhwUIB{IRv-jv&o8a~CaR)g+m6n1GhgLor!>^@UWq4Yox3aH3MH1lP+O@=%'
    'H@Z5lODv<6x?3bxmkf0VW!}WmplKZpw0J@U+`-wC!|%!Vq}h6wQ;n@cvT*DuC1WW_ru7my$(yE!39RU}Jl(|^0+HYc=9z9H3U6c#`K0aJ_IDDE'
    'JRGs#mgKbKyi2#0!fx^08+UXYCkJVYi7|PIglM`6hfe`c#Z5V$!lf}G6Dh_~y!uuY%b|CHy|$G$XFqz0WfogTS}lZL5ra~@l2I{rWQlLyb`Q-v'
    '42N6fNT{0tJw%`4nYA@E=ZKc|@^Jbr2@f78JMIc|J83qD>2UT<4Rr}y<z4n2t>J8Gp`E@?m&#<bQHq2WXIp~SP*{e?F@d_hodVi9PIOphQUmvB'
    '?5NP=Ijd76M0qrmq;Y1py%M5eGa;#~j3e}%_$ih)2iDqQ%z~AKq%$8COoK>2(k%t0-i(4}d`e&N<_Q-OS6`M~s{SC~MQH~KZlrumn^R4|<un4n'
    'qB#!xAoEbjZiiGS2yjN3BlSQ=GMRx5!NyP&<S9@&NcKcEeXH`%Flk`2G)XP8@1o^MY+&DSqrH2Mr>;hDINs|ZeO`}6-1>P-)CyLUdXfhbqgXu`'
    'P=-7lq&<cv<%tB_qo*nhbs=w$h+DX+ussW*fbawz=6RPYf|J?V0-@y8#pqS-&=D0qefQBq)EDK@z0Gmo$Wy>Mi+JjM-}m3CyLR`v1=v1tM6z@W'
    'WA9k^(8)=&ubs}26XOP>-}7Sj9oJr?17^|N?ROnO4+0^SL}aDrZt<{74Rr&tf^^|+xojD;*|%mV<EKM`d!PeNMY{uSy$IdVGb!fI-v+hNO0o03'
    'Y0NHxQ8h(Zu8y?73c@05qU;~nHFV_?k(H<*iYCKzuV{z%_4Z<~8;IR(>e8Lr-f^!Qr0%!99qF>ivNhb5VNvnxv>`D-2!VDK2F}c8$VSBl&KB|c'
    '+7kOq4Fh5rjZ=l(j&mpfi8+M8s5M)}83mm+hUDiZdqU{x-7m(VEy{Qy&>PdF7<+OC0MwV5su~E?5fI@?xybqKACQsu5KK8*9IKVXmL4&FX-i|8'
    ';HS9+u1o|YK=PnRUI$%q4Gy<k)`9lAA<IgI72=VQobH1+utf@-{=Ard06WZ0o}T$2W4ZAb<cZC?5akjTd$x+K=BnbMDxT1%+Ms=!-me~1tJKpL'
    '(lC<SIVA|q&KltC?fcASyJhlxbh&pCQ&UFmk)L~{Qk2=3K)TH;<e~xtC{iIXzHawfUi?ymSmdmaGCff~s$H_rf<>|Wg21yE<H|KsbaA3n+J;!N'
    'ahB&Sru)Wc9}uD3D4zl5odgcT+4j~)q}dj;bC^DK(L6OQ-FB_g6@upz#U9OHXhXaN#WB=N8oKPQ`AC%K=2$9WF)2d$J$ed%wHkSx{B=Ffr6@3G'
    '6q&!!l?VZv*PCop=Aia7kfoG3nEbVmx#@OjUhyWdP=g_$TE;OxTrk%8nt8(5oyQhfQ^%SR`Cu~S>ZKxHtE9AG>qwPG7!BjPk8;#CJ(ZkaF1<=G'
    'eEqaKeC(qEpa%+7KCLq*r`zy_4ww0xEQeY0hNmDGiRveG6hcUT-JUPN!K9g$lU(IhXSA&C=G<S(S>V*i%67@9eJWGN+C7fQ#JMTsp6*DmsjrZ6'
    '+p&#>y4o`Rc@X_BNZX=PZbunoc>7e_ag#t<+Fk-Zp7V<5Z&i;W@ohdqgSbh~Od?oNXKo8)?+UH_7}WB5=}p*wCn=o_z#-Aw4i~+1CY-$~D28=?'
    'Yn1%@Neg2KHd7*<>UGB}4_YyNSIsqP?zTGKv$wOy?s8*E(T8?)ANi5rNhxj87cW4>Jg20*z$}ti#tZP<#4Tm8-hk6j^Qoi>3hZ1a;j9IrSKT#K'
    'wxV@=vFBqc_Sx=&D^gkE{bTMk4uWwK)B2*ew*?rXZrO|zp=YOY<xbaIZ|#enB|~SiL}Mu{74Xs>WwQs@@53{BzCgPbMsF`r;U)Ilzmy0?+Tt|g'
    '268)jU#@4EmF^nm4{v@ktv<gvI;&i2q&s#B?>Mt1zfj%AipSd~>nKI3)cG%0wBKxgY9!m^*K@n{gql)YploBtV~6_P#CxPlt|w{&_m@3kScE^D'
    '+KdL$Nt1&Z#+%3zipTEw;IO=|&`zEP_=q!m^PuVDw5O(6CGum!6ZLC@Xj?j_(cY9FZ@>SF#s?kWy0m2MdTOV%WK!8_VUJpk)GffW_DQ6~CR-0n'
    '+*@m;MAT2N8J`BGt~RB~01m0iNd4-|g?LiBhtMrg{`JC+BxY<>XZG?lAe&%-TIcdl-0!-_k+AKMkk5K}E34l(l6OxyV+x6PCS|^y2KiTpKRUi7'
    '9p8{Y)V0J$o}cQjw)J^eW4=@1g$X{^-f=l(7P=-pm3s%1NH~J^j-<Cr!LAysE2pXA-AWN!t-9@CohcqQ2;t}9f+qjqIGMAzKkq1#ht|&lWWlA7'
    '3gu+d`c|0dd6#exFy56C{Du%D_c-wp+$U-GNm&ZLH+EJnK()d^zR2W~$2n0?=Mm*Ht}7PxEJ1JxD2l&d(A+L)8ut_Ez8b!0aj<Sm=enbj7?8%>'
    '0l?B>e%{Z5M&<hSDRLMyU5TWsT;eL)<J)X!^~xT1HGH)g?x>1aQZca!^G2~2_a%@taQA&W0IUf3?-RTt+#T%l<;d2NfwlgCJidS3R#kE%EH~8j'
    'O0LR+q9w6=hC&3Yn3>Lu+ke}V!x7#fE@#U;hp#f9com;YKs?zYdI8+-d<p;_XV{>(eotGST}h+*(=j-SN2D}7{=|nE7=xyaGCcAx9IBBKy-v!`'
    ';{_%t^_T3f%Xbn;f^m%3G~4?NSfM@Y6n-A}iy5tPDA%@t1R3a{fOOOC(cqMzukLjcN`-y$*r@$VS8sWD+WGwa?Ch#+OHAh<n-y1+;L3r&=cpKQ'
    'CC}3wh{}VpOV;^^t*sFiGW6RPVMM~b8?+Rj7Z|!k$xiMSg5gnNX5n75eA;7H{c*LzC|4y>0QVLBap75$NE1rTuqMz5eeufgEn&2bjCaaWiWDp2'
    '&Rm4g+wqH61^9YkYLIO}Bwp8CurV(zxUMkw0&S~1X9z}Q(j(@MSH<V^MroJXd?SJ0ZdECbdxHHCAHwkyYeVrU`5r#hogCk)P-td2=Mu!x(PIHN'
    'zkDs_>{Ii~deC5&LGo;5-hlXJC()E_H+v&frHlnk+=yBpd-V2#aYfgVPJenJ0Jf4h#H;#vVk!5>_Wg4DfS~R+e%cm>S)l~pDY7|v5n7X&e43R&'
    'Qw%C!!HA$G;uw;9G!*YPs#^5{4$L0n!~hlL{BpbCJnCC4M|NAWohf2fUGaVM7S2zUPHQUKa8Wsht`cM}>zvG2#P+@+QXTbI92@~rfig)n*{Gt9'
    '?IO81vjwEuMO^(&#QLlc9GDwOQJkPeg?^}|3$4*@5k&(7yff17GJX^VEvShJ_UA4w>}&%3A0&I?{+e4B>`HXNgdf|7y|F$+sHZ&^)CgY`ywxkO'
    '5k;9|pBg}A#QsT2&5jFR-v{q&#Z~YY8D7zC#@zCGk)UQJzFMZl)t<<ClqhH>rw3?9n*ovW`{|$<506yeqQbEAIG*kkfo?^}LAG3>!IC9zD`oAJ'
    '!-99Y@9UJPtmRdu3H0)6Z+A!ApKxLymVlK9wj{cY-Lc>8K(=CaF>C!sD_gCI3ht09*b7sMzEX^m;|2^p_=b5N(i8b-IcNldRYdTa?;F~nFVKZq'
    'SdoIX83_`Ae?1LApDVf|%(fRR(Ewx@sPp}Re}+eRt=p#vR(IzVCB%|S1Ex_0Id&)W2c<uzHVsEjyrStW-6^M6N-h#%xx#He&d7TPbhj4Kw_&<2'
    'SsEOWQvsJoE4o_=TArqe;iH$8`sqo)>Dw(#0qBW~(r#a^XpqR5e8&z}^kfa57B!YC<dNy@G(M^ey1V`h6_2JnJw-n<a#0r<*uns&Gw7z?noS^5'
    '7%a-UchRznI9Tj`G-0ORt3l6<Tf@m#KSl|(_7GUdZA9SfBqdjTT*(2xBVbHXf8<-f)t$(m70Cc*8r)v=j;lvgALS@9oc+Ei+vDL~wg8;c!Tk|>'
    'e~$ynKu%TE^8609=P+Gr62jNI1Y%l!Cbp=r%VemLZ2fYH7@Su#pKaFm;S&n<mZ#-=v~W`F#d?-;7pb;=24pCDAD!M-lg`%mLDM~%N3b4zzdnT~'
    'P(wJOgtV?HT<NdI#SHvwVZJU&JTM(44loUedqn3-1?`;#q>pFLZae3{Aj7l;KwHT2_Hb4?g1pl|CDgNXWY-!s+zHcKPq?Swo0&S^7jGu93m;tz'
    'p|p53mH@5!%bE%VwtB39%Sdf&r{C#~`yMrBQ*qN5%0NdX4oZh32^iCvy-9it3}+%#9C@h8B}Zb%vJ3Cc9;G(KMwCkn!PItp6MwNCXmwY~A0uFT'
    '#WGA<B}KS$R<CoRmGGuInRY~yS9`qYQf!4IFYeo4p}A<6jS;FK=z1Kk)1nMzLjJ{zz2(y-?u;S}qIW(;onYk<9izx`OaG!DcZ~z>o|$2f_D8LG'
    ')hMZzXu8xyfJINDoX8aFhvl&4Z^gyn6c)y}A(`4JME4gCBUC^_o0vtr1#CyiKTEjZ67;q;djOHYzH8JqEykL3BxZ*bz=>EX3r@Sqj-R<7aF&lO'
    'z~)o!C7O9#TL*mMO)(HOQQHQY-MZS?^)Vi_yJr)<ab<(6VHn`H%5&7yBu6ImAX1d4puo=9Fp&qTz|Qg}n`<Z2O>QFPVeBy6a+6Tmg)T+9@6vh7'
    '*XrU0wq<l_a<xnmaYn9yF;)vxrlA1XEv`!_1c@X=pfV+eu?YlOETSPrLvE$o?vgCnfK{bXM+GV%*MkMJvUcn(O2ctU$0Cm*E)Xl{1v2M2-+k+-'
    'Rv{@nN4IA_=9xrYoQV(_^PDk2YFkTzoVwySX%aU;h)_GNy3M^2uyweqJy)()>Fe7KYaEXO!R!pjbCB1<Ty*t%Vy5f_uBFScj<}95nq`P>QRWJl'
    '*2=&A1on8I!9EckZ@aa}s`ax#E<=)B!Lw(nr>G5DdV{iaCZiJv=gfM>#2LhYI?TOb7CL+Y`-SI{PB68Um4Tt3yxG^UZL#53MKKN_>Z(z!CshV#'
    'Xg~c)%C52nJijuFrti=KhdM#pVq~%RD%V@+*{k&&J{7e4+f-yb;HgvNza)&lkD*#?t;!y2o7YOpqHHd1i*8;#^z|;U{nbX13UXd>iw6$%V{K3O'
    'Q;TKT89GVU_TXd==+w;Kyx}RqfGm=pr8hg_*oS)o5y|Nk)LyRF+G`c(&hkK}s0-H@i2)LYy0sMHWVY+*?BNcEhNq^Y%rI`I;)K0Y<5R>8s~Mx4'
    '<eo2Fen~P$5F51!I>aN0?@uUj81V9xFE8OrY$eR+!5Vtv%;3v%FA%SWTj^s7yf>=177)z+XBB`7krBOxRknSrk4fi4hRsPj){=1pJhU+L5>+7b'
    '=W`$ueX_DB#689}<&fc($5>j9v^w?;wBu1hD3MdB%H$auS;`T%{9RupiUl%H2q+%>b*&({;iRmn1}i&zif(}Ox;~}@k|b*28O3_NSe}s;*IyR6'
    '&HN&p7{Yd?^At-3$t2pE|4t1h_Jymr^0{v#qL>_3=_Mj~sGrBKODv)sw!8{jbNOUx*&TWH*S-)Z&6bwb;JDH2N#Lkm#8V`5^KKaZ>v2b8bnwTe'
    'zPuRy-xS6nIo9Yr6s+ocD|i-$(Y03mNfS)nMGZ&b`{)u*ygk+C^I%_0+~|0zh_G}??%~p3>a&^rxvG(6N=7zpw1WZrLEyGG2s^N73CX53n&M1_'
    'CrvnRw5qS%%i|}nSm1Mhd9hYYNYvs>y(bplFPAn3s~Q66`1FQ79+sN32C)EF)r#F!Og-x3GIh`dWna~b=i<eB4Yd_^q|{MlSZ`NtyLO3ghk(E+'
    'Ylqm&4t?Wg>P(|jSM_A_8j~p(x44_V*N~SF!I%>5uy+4qRd3nZgH;W`Kd1G@s%|f4mO??W&e$>$V30l^W2W6>-#4uL5y^7Dp}ZQ8Ch_|)kzj<L'
    'Wqdbm5%wxmA0T43?L|elU87p5Sq=rPA`(YAac$*3LkmtKpZ4CW@dJ)i5s_eShgA*5WZ4R<mZkgZYg%h;D>bYVjsuna_p3#7U%<*#4H=KaAezIV'
    '9%b59>300XONNV%drFb)5r}QSAXJ`gHk=!{?qnyD++iaIH3*Kvyl%oefl+B&l_fVia9(u;;l?Dn-j^i0EB^cMq8wusHvvPiL(6VY7{xVAUqe*w'
    'R;<;B0PsUTX9ZVNex?<bR)+2T9h1u^wy^8wQAjZ-Uw2lS>z?fDJ7<Ui9mG~0Ylf$448RfRt=J#gB?r!*juozoP=T&6%U>1;5o<M^KUH82<wjY;'
    'Ih}`eG2LR-R>085#*XSw7U5+H(Iws4@OE^0V*&QLbP?LoRO^TzkhdjmOL(nzK%i}V(BFAA$fspSNv`%BdRp@h-(7@vJ>m6?F8%ikr_D?rhb(DZ'
    '518?*UcPF}yL>KbICChK@41bAXC;aL+g=Ghc_5giIW_Y$C&#xtzz<a$v8g>jj`zrIyXy3uWgw1)ca&q>v3J#K3gt5nA+Pt6b|z3Q@)c1$Jf;^P'
    'M(d6TS2^}psg_o!d`A_}b97yFul&qfV7i`MIdg#7M|k5SBjKF9m;r6EX|oJ--!p*rgEN;rKut1%sj1i9cjEVL=?j8BNjO57*kjsvD)+FL7kleF'
    'F(Z=hI@;^>g4DOm&>gw<CYVM*pT)3YYm(-M*nX_u90BZ!AH3lXYSl9muzqRzkwB88)?(uZdKVCk6iF;r<Hz=V>lUId@Jo!LZq8C}f-6QH#xA|b'
    'f<d@wRY|eDX;FxK*hvcPzjlYP4ejNVMvN<?FZit`?Gl?(bCH|QAdlq;m##)7o$Vf|WD4m|Lz=1JBTFB8e_O-ux8dvy2FzW$H|k6KujI;QYXc~A'
    'nB8L!O7AtRqD*=qt_jOI@Zb4|F>}#gi+A<U@V`DBu2GmPS#ETV&ewz?>Qu?J_ZX=}NPP%1`G&=^mC+J@el|ad-0mPd$z)fGZUL7diaI3bs|JG{'
    '?|v*v&9?8XPECc}++{*g^c<LVjzOSoG{Ehi)!SK_X!=KhIuvb6l1TJqXTP=Dt^p4$=|}{yE0@zXfi@-fC%RaPUv`CanBL0CLUXJKmpq+(+cj_p'
    'cYs2Kv>ldnORHi+^X`K6Naxi1@eR?DF>SseD(!L-KK^*ydX`Vi3y*t~CT9i&#BN_Dh7`Gb5YF71+}y4}`W||5+@SsP9%~&!4&cQ~3YA>jA0rv}'
    'WKSR2TW(*qKMve?5QqJYu{GudjnN!3yq-RI7@^3SXwK)dq?0uP(Y$nTJlQ4GCrs|5N4(Kp%zs1Yo;V&3k%;kd#pjLti@Ef9Pwe{n`)RLG!n>Md'
    '*+Z0zQ-71(<+5M0%7KmM;a_A3>mczlt<$rsa7ss0N{Ag2Ek5>e;M|pO28f?`mkT_<^i9)zA8c7sHD8P*%4Vj#LyF&{?gnRdJ9CHSSqrz8_o7p$'
    'QNq#qO2}xZ@MdtJjM?SI$RdpLlL`;@X&*ZkCEr$tKYm1VDP*0ivIdpeKAG(}w)m?fNbKMycdI%6@Wv2<Uqv__R)Oyt<_P0c82J|4+MNcaNxj=7'
    'Wcx<?Yf@H|-GOwj;n~m6Es+hPLXQZ3TdM_{JeG@O6vW1|ue%hqhf)taSnT4V`9rmOX1mO1>?GSY)Ll`n_w-tN5rrbJ2G}BpY44E9k8Z*iKO4qH'
    '*t)lNhzViY%f5EX-ua*t9pqMr@=)7~cjMBM7_2wL>4V%_uVE>#E1HzcmpFUa`CD_hT1T}NJ;HBR0MdWy;ioMs69QjTz`(ct5`@0~U=Gb8&%Zq`'
    'DInmw6rLX27)LSC^E|SETcU3NeW?*h3V7;#HmLq^X)TJe^=92k2<*d3a(S9bJ4p$-Ss9wNAa=7;2#2<d#k^7=KNO|*t<BaXP#x2l{qZbG2)5}2'
    '$z_1Y@K(qq!%9Cs<t<xk2D2Jv*-gm3`P5k%YFQU)W~SngG}76*1$znZ-rEi>NpE3d!7U*bkd*TxsBFd)_H~};=&`qdcAH!(lIsu;pxtj3#6;V#'
    'l3@`QBHr|W?nt(ES3*Rdl^EI_TnYIm4BE!G>k2|5`J;4Q_T6h+6%NBVS32E<-x^Q$8w|0kX2dH^qGPi|L26*yViRonW%%UR-Q#BG3YuFBy~Xxa'
    't`l)?d)Jt0ESUy7?mbe7<AiCsZV54KK9GvKa)9#l!5;X2JZqUpxKB`d>#N7s>n;QbsvKs7`!#OvPB(e072PN(_y-_v^JO+cT=mXzg^T>M%g9xs'
    '74)9$?CyKUvno4MN7Zfso5eW4a&eVXq&B3e>?h^WUu>`Pl7mR1`HH4ZIXNw)y{0ZA{7-lK2K%bVHtvXk0Yf6|NkU5?Y3L$1*VTflKu?3P37rH&'
    '7H|7415$V1w+Qe|>Tv%m@zK98UlHMkJ{&iH7l6#b%iD|DMp5JMEB6(5;F7yiYnDWCyX~Ex>(}^j!ll`fMXHcfV@@Kx3cKj+9BzyCiu=9wjQ)5g'
    'nUmP_X$yQtC@K^ZYf5wpN6}H(A#Nz_WGiYa+f4=joG8AjapxQw|JT2e>Ehu?8hVex`F7b|wc92Uu|yI8dye<B@#`}wTp_7H3Mgv3fuXnrE0??f'
    'Bc$(YD<dMDhve<tcl*gJ;FnA!Iako4gYEhKQT0pEGX9Z>rfqBf`APd#%q@nxxn*#a(eMvx%l6*f`7vuE+C4@a*d$T%7@hFx-@@r4^DS)&hX&-='
    'AB2GSMa>i~WyQ;Pyf(5hJIwc40u?2P^*YKQq<mmUg)6~$FN%<LaoL7sczYt4Y!rpUL5v<d6!-$$D{G#XnOgz1Le9iIhe^eQI4upNS(9LfEoKaF'
    'SVUn>)3(la$JlvsgQiGi%BUK~bE&;*$9rWSahL_3)0oF0<P#>|chG-Z5HrVpx9$KW=kCH`LL5E=3KbAd-p>;jPFyk``X-7>1Q25fja_JwkGn(^'
    '@{lnQGy06Mj#hf+S%k1^C&{k&L74w?;ZYq?O;L(>UbfH`IkqGsDv{CNY8Yv`F{rh9rr8@4?cL|}6<)y02j|wNcDW-Cz1*}C`_T3IyKCn;BbGM9'
    '+0j!u(`<NEeJI6l&uDrFFc7tn7=o@QhSvTb-4?hwurQ)!7OFCz)kyMqToHgF_;aL6*vJ-bd}*~v-by0OYz5sYgq2p=UpY~ieQa|rP0x1L4^keM'
    'CsFV|?_C->**<x(*~f!?^MPNEkiuN<u))+3&JWe{VPKBTquf&a?DtdW;Fu|(6^!L0rEow?SlbJMx57R>0RZ(%O&g)#&3=~ybg3SMG<Q2o3nxxb'
    'Eesw8-?OJA&C~U>15iNz_LJL)tVuj~GJ>{s*ej{0=dfYi(npV6`}P2EbGU;$Z42WpISrXiNX!i9+4xcIi>VT8%R*e|W{r3|MUZ!QF(w(<lYwa^'
    ')Da%Qx{PB0-TMFpa){xBIEmMBfL-wF4(&zskkc$TeFTZzL#O3S#%`H*ijNk8X(pnUb`~`)RF&HD8}~I=l;u+w#+7?fXY%631h&D41oE)rNnivn'
    '-ZOV@xoZdx%T4!T_$Zx<o`rj6n`KGkI3ii{YnQMS?!iTAiz#RKeWXMU^bn<1*q$R0vGuKMsbah;0!pTAE&-1g9(gGW7RhdgIG5If9kYhXwu8jf'
    'eNx>QQD<FndE}s$5@vvO_RA@W3D*evm@>v9-7BP&u!9R+3Hjh47^n$83_QZ;nnvYG_+61Cutwm}P%R?Lw6N<>u3^fe*@lN8k}n6ia+90^n+_G#'
    ')6(u;RU(~-#LL@XwEJ&Jh)Wf9lW=q@DET#(fXIo}Y0*awcM3MLMCj6)>Zi45>lFHLr1ze1P>Sf3lG8=>28IG`<q!)yfrpEv7U6SkRUI&pT6zfQ'
    'u<F}G($C)9lESrl)cHt1Pc{#NqT4270lLQhCH**~0*u5*&1>pD4<$7-06#rH5XP;}kiCe*e933~g5UCL(2C9xCVDxJZwd=~IR9EI#jB*w0bj0q'
    'qMC$;5dVyuNSF$VDPVv~wr#W@So%e$m}!o?E-3c6%>dEdZ4txqd0||ljqmra*HE;iXtQ;@?LEElD<P-;9kkh5S@5HxifNH3myYDIV#Tb_#~Pst'
    '3Z8u3Ki`3J5OXB>caYw9!=CN6{68~7;4ojy$DZsN>UO*|&Gte*HCni!2aJ={<kp{In0J*+k_;5kW;($FDhjvT2)7k!e1O9bWUVw<7SN+-=ca?M'
    'ux(E+oQ-_T63VSFYMbO>J%%Xdsl8-sYu|X(waXB%lHjai@z{k%B4@6)N06OLXIQdFE^v6>0;Tskeb7C3%tvcShI&Dh+V!xPi>^p>eFWH-MrBaM'
    'tB(8X=vlCw(Q0eKB$Eh^r1@DBPGh_2bfAo3DDJ4pb9>Y%Cyf3jB_0JI-pKUXVQlccI5v(Y#~v#>QK#QcUD`u6EI8iON*0OoBA4P^0Sd_gW0tTH'
    'P4$pufAU!OvkS5E$Z(F(eKb9C)v#xO2+J+ALUT++Q7{{5PNk7HDEw4v@w-r^PF-ciIpUseaP9zpMLeNt)yuv+KIQdku*E^7tw#4P9dA-1vfqb~'
    'VYOY!N*bB0@Zpap9(wG5bqDvYRhzdj?|s!y$C3Iy>CP_F*!^U`GyDuYMs`-`aSJ~9qZoVEOZ;^$HTG8973D-AA!FWtJ2~IPgqj$odqcQLWhknM'
    ';E*oD{GC#dl4jT^97H&8M5NtX(v-NqNhDGVFCr*<?9-6O9<7{*b@p2#SSKP&Gxmz}XXc7%lnnUlu}9c<iP-PF%aT#xg&!?>19n5Y<rXj<=Y>vu'
    '(EP22yW10Q->B<3#mwfiqto`VW65sUvjp=#SZZ?$QD_kPQu9<FL;nIlvo3W?+tT)+{LPYTzWYqK+wiTI4>Lv4M&uViV~kL=Rs!trB|Rwfa}E&+'
    'RB|`gD|3$aI*%!P(=zIz`E=9e(kMTxjcrLwm6@$@Oo2?mA2N^zvg310ZSw$jtEWHrm}-{Vd@^tUcqBU<Nd^!x+h|e?;}<hYeiG0N5n?~uTJQHm'
    '8prJ<rXfV0beBa-Z2Mu|Be*V=>bi7|9z!hJk^pEfq(LM{>T|7nG>g-RcoJQJackGue%Oh=O=n=RUd&dwe5v{J4wItc-U#8;J{ja2ML00+Fv(xB'
    'Z&+BWz-#QqaEN4;BiS)NQkY&0n?}%W{7XMAMP%)W2_vE(y3%ye+a5DxpY>Q$c+nOK&oprE{k}Qit3^qEQ!b~&B>GNKq-G_4BMW}mDu;H)aXuX-'
    '2QcE#z^t&Z@LV!zW8FRB3n6D;aFnw|soXJ>q=Vw**Hy)%U19hw!e}Yx@Z-o!soZQ!PZCLr?LM_n_Am4Zp1q`z8CJEQi8}QIVQV>zARXjyjGKgn'
    '_^WowhrpHGM;f8Wz7DGri!_PYS`SufAVZ#EF-;02D0g*fTL~=A;)_31f|(k;l6~jtxoQ6DSDzPy-n>K81M>cI7<rKIP+iS)lH*k?feDDtHvQvK'
    '?(Z0N+s<+r52m#}Oy00JY_(@UoIoz6`9!JqzcyyZJVR4lfseZ-Uk(JrS#Iw0vdY)1$ogL9p`5H;^^2~Y5`W$W>iC?S(v5A{z|NQ@Ls6R|$3ezF'
    'A1NyzsS)<z5({@Ka^QDIf&>@_GD8aPJ+>_EPCX?JfI4~WJQUAV={bHJxq{r7V9#PMH|?Gw4(IO*_Yl|S$ZqW(sV(^xjey*eK6NwT8LZqqvM6Vz'
    '_k1e!_L#MokmyMKQ&Vyjb>R}7I7<H&hzaOVB>;a-=6@MRdplp9tzBqKIe6O$P-?kbaNRx2<#63lqXNL(QN=brQL3H(ot2GqpGZTNpgru70jjc3'
    'rn03{z&Q4h<vgG&edGsr<^8s3g|Pj`MG`G(!`C9>p@n8MN4Bj(TjktfxLDhEcv9HxLFojl%y4NBs+r@&MNj}wTP{Rb9Ptd4EQCQ!O`=KQkhTO}'
    'z(JkS^aXo95s3Y$0PJ0sDfo;*%RD0Cjla@>kG{{(xn|_YqcSt>&X$7NoEvt}kQmqEupK|07rQvVE$GaXI=c6rUXF>kA^#B=IgS5Z!`X}7q{{iU'
    'e8Rx*cfKc@$90U4{fpflxnw?DHcCQx_Ci@CayuM1IWeqABrXJ+$9S$hzotBEp!UVb+?nx6AJR72he5PF>NXDFQi3g;-*+B+7$hC{*h~9(F;fbl'
    'vfQQ@9tz=N+eh;<3AJrurW8@6Ck_ha)N(STINOqsM>~e`Xdq78E=?@zu|N2fOPCIeXoN79#HFY8BWNb;;zivB@itWB+4;|qWPOp{I{vzPEMb9-'
    '4yY!HdxPqNI3|9vSMOCu?9wy~9N>;L`Wn$-DMkHP?}7}O9C!E3%+r6ct;a4_C2BaKR!U8kyQQQ#j<5=H&DbkGDZE|XZ<#)kmR)7QjE3usiav03'
    '`{ofL*CN^Ji9aTZ(UYK~JmQ+k!F!p;St<UMFYrFdj=w~}<vp;)X|~cowTEHU+(b)H0%Dx9Z6AXb3K}rB7!q$$EfO6QQNoMM=vr^Orb}sC?Uzgw'
    'Y8anL8E&PQV780%=<3UY`3n9pfoWDbzn2ojGMNRd))Ipw@bqX>HXVcAbL@zD)dpBmTlR~nwl^vCc`-PQB`Th}J1CWgnVB<6W-M^Kx<`)GP`3O}'
    '3J*Wn{%?zkipW<xlEbcvd%q9=m!5ddMFpN@Hij#6E>Q3CB)-z68MD~_8PU-%o=sN8@uFUC<7^+*n;|1d6mHnTS%tv%`ztXPdv1DI>_4nol?#M;'
    'PpoRP$973?1jA$NQK^~NBDPVNHU9pp$$$vwLwgm;;Pudj=-K7^2m^nTXYAi*>+!&Q3IcmPI&z0RV;?{v5hafF09hyl@8$Z+6~EYP_Xpk8A!qvG'
    'RC*FYGGO{zchukVl*%OD9SAk4OWH4!f;%GZl`Cl*pRHL<t;gOf79`ubh+B-x2u}lOCr$`4S$VO)Q+h}95IYsYstp-X^-ElRpNcT?cKYH}q%@H+'
    'ar5<}NwDZRgJdqS6W(ypwbkkBN{hlCOxu7$$DuumcCzIrBs7Amnq+knb=o2`TeCv2snK>!MEw%cO~lQLSxlr{l{<KaUyz{fa_m{Q*Ahq&$`XhV'
    'SOPY@z#8k0EhIdy*YInCi+uw;{49gqo#l%C6^r)@h#l80q&_&lR_FAWiVMrj4jXOo>P8kBculBn<Syo5mocJGI$@tU4+OZ3HRrhxxhy>Lh2oW9'
    'OueK#408PBAXpi}9WYv%5GiSetRv3D(v<F3!vW4vWA&KQT6dS_0ISii6X$?-32G~md=-OsRp%@7ot!1E*AqH&3wLX}scp+?j)-n-otFV+4)Pc)'
    '4bPRPWPfJBE={gp1TL%{DnY9wPek{H`f?sdJ#rhWs})iq?H$U{+biCFJ5$)Kw?mS`>`UC#$C)x=T%bta@??L}3c==NA4e7jBtb<}tH()8bk;DD'
    '5%K0t<#&G9#Q!YXH;%>t23M~1zN7pG*9@zsohD!1WWDUjQslnF)<q`|mHteSxb$`K#-X~g5gU_^BY{YhmfCr>VMor*tbXeXU=A+pEY6-CGt^Z%'
    'OBDMG<$iV!?0)LuLMpuGGTDuJKW9)Mb;uz0zAdT6D>_99fMy^QbVA&6<yEc{S*(B*4PPPzbcM8e`;KXuN|FKB<`>H-gTz2wz1+!k?jBCEWRHD*'
    '*jz?9wqq~~us6lXFbXdk<MOimzd{C9hEJ_EF2X2nE+!Ovp}>K;;3%SbFDNpeuwn#()po)WDNu+bD@|Z7bU(L3L_G(htQ-=Ane2~;f|a%|G-JWv'
    '4Mv}1?~rgvyh**aAXL_bt=u&T85ku+w|eGWxF60&wb-VPE&!294)P)>5x@t_Jqs*&p8e<sOFM{wkCY4IOSd>D!Mfg)j4N#dc3_UK^)womV)n>B'
    'o>c@&q`|I;5{+V&&a+7TTlQw5jPU60M0ySnHOGy-lm=Crgl2F`<Q4R-$>Qi%HEXuDU-_1%9l9you0}h20Z9BxhqsznCoNaL!wljpc^ctBG&w3r'
    '6kqqK7No^=ok#S8Z9mRiE>ZZn`U!1zTO9zsNHL5fB8e*-r*csSe9hA*+fmJDmzaIuu``&rxvPk8?P>xWH0Clj95_>`hQ#ge6AR`P>F~zMxwVE|'
    '0ce}_8KaaMy{#&}zwW9%p{Hj%H}?7zn*ahv4k7k^@K3wfv$4~y>N8X^*x5xJ=}rJ3yJ438!uez}6*ZX}i=jwWb~P(bJFlfo=PieyPR25QHVq`r'
    '`$QaGZIdWN)<N<p5xWX)S3Xk^jA9uiQjbzGh#pMo*j{!TQe88`1%Wr#d%w3ZsmYRB;PlOAXTo{U;9OdS&qjIySiapb#!=kKm_&zVC4CL!r;+8a'
    'n|@Ew5jVsw*eT|BQ3zrqB+YFz=j!d7&2e*mX-~wf#-K~^><r_2mr@#qPhSxv-BGo<YvM8zI*Gsida6iF6m?7ydm>$sCGG?ylI?kPyoG_;1~zWV'
    'Jf9B8^eQPTnINeaJ`YGw_mtohZ&l3m*)LLdO6V^rsIht$*BIdZCT<-sIHGPI^n`R(tJmlSVRbsZ6E_0P8IUXS=g|geXJ(Dy!)nM|xwe2B^v`+&'
    '2Rp`SC4FGGsS2OKMAX_xG0SDT0L08sk0JKDu_brGR<vtGqQ2H(K3&TR)OoRTF@bIjphH2R$K|!7hh+aglVFrs*apL)aa*{&MSEJCkGSM9WLx?|'
    '|A9><<suOcmvrJhTy|V=2@K>AjK)x=Y6)%Mxg>5AMjKXk%=_@$rF8TksDavR37w->FWMnu;D&Tf9nxR!7{v*nDUTVK&>dlkP=*@KN?}HL=*wrL'
    '#uZ;0qkIMZ?ksh`{R&od)fT#hNdYcrG;NQ9RkGz~81c^1B}Ff<y_SNW=U)6?_1dbO@}7~B@4#(?<f1c+&(8|lL)5zT*u6nkLoLHVDt>E$7&P1d'
    '-ugYhK}T78I2S{Uf?DR38C2tI->*el*GM~JN)f-ZrQ`(`-A~ujlYeX5JFK8FM3*i1-b8e^hI`S&xuZZ{y5i{XR~E4gTODq5>8LY(&nplzQ0Jx6'
    'MB$Hv`jm%!oEs}v*a0kQrDNrukH*_Pd*wu^7}3}Xm;!K@j$Vbno|QaH;;QJK<P!D|C%wfY*{+m%lF>5Am>o6Zz!%=;np>?zzU`l)uONu20rz$^'
    'duT85WqD{G)@rp@QAHDf5<)e)I|zG{&&w`ZNAFWS<nU3Pe9#z?oHW@!3N2z4kdHmE14kXiYhbgs^p-L~d?H7GC_f@OCeau@t0ovh=8%Y%$&yPU'
    '%y9rbSu!l*&QO~Myx`_3Inpu^o7uLcZ%i8WN6*^b@yYiWg=!&MI#W>4^3_fSwK!M8HbZ}7qEB2U&-~J$mZd)#(Qb6QZ9I0ViV3qj%}id+MW`bL'
    'I5peoPFgm^Kr7aUTz+hS<1@>lNS}_$OGGTLw0uOF3>;INp#v6CrcfQ2q@*ss?n-B3<>&`Dzctp=jmutTR{5svhmNQU@dslQuqr?wM4I~s%YuYR'
    '*U?cwgvP)ahKRr*Rt<p{c&j|O9R~3fY@AahiZerhm9lHXX5S(A!`*#LBxHNSu92OEO?!NP&D$9gc>ct*hudn-a~%MZ9Wb2<g|_!y1x}JV(2g*>'
    '`%q}~6b0H;xCSbXxj3aVowNtc>h>9N-V*BZJG+>Ih)Z}{LJ@HbOpkap1jv`s<^0EKS7mTx4=23K(0*&rU%T9+&aQE&=AD&=I}%iNG+wg`qrBuk'
    'T2TsR@>Y{GhFJlF@EwrOyD0YQo%tA79FO;$k=PCr!|jNEtmz^15Wd8n!^Swl<#r_#@e|MCO?E+BTZym@lTfWFg>@j<I$&A146Av|t&TXcHWO@l'
    'qB>CXSRHpX89Z>nPd__qNEHeOI++k#rA<Q7DJIxJS{SrcOS08)g5B*N%_b@s+d;C+h)yWli*^P{M5NQjj$r@<V-7&}t^%-1B1z4M_-psNRD?@^'
    'EpWh2p1LW&@WAnWu0))E)xP1Nr$ulB;bL;E+Lez`u_CMoG?!(3*4Dj}Pb4?o;xNl~;Clck%IJY^p9rscAUbq4Cc{`Me<});5q(R10r_lOWI!bH'
    'XU55xpj}<VlF!z(x9`vHfx_T0u)PVCZvElgBaz63lRXYvLt9BBYzr!`um`f;>MtErD1j|Y+vf*97>HYG^_Tg4UlQ<Sk4FtV)Pw5WKHtL~tH`W-'
    'QMFL^(DTR>4!xtx(B*1d%sXqidGbjQbvDyh$cYwq6jYNwM*qt}B0HRtqn*zh@HD5x4&=oWX2z&THm&NmGFJxM3yNvKy8@WCXW1WHre{)D`h^l8'
    'x)ACKZ@s`POHtS@r{}YB_WKw?!=m^@4>$XaW-g8Rt~7_(^!G7laA1I&4nAgk&LGen4>;jf9YDnmm(K2|slm*}wuX5~1MbT}mAg)1Y)4$%k733~'
    '3*-t?m+-=Z;doU6jDq~{fiZ)9)M|ano*2OVgh+!oZL}Pw=^?V;$6yS0X?2Z(r8|~qGy<?CxmN39O>EMKqXn|zQLb=7+sR?2!Rg!B*6B_2a~5Av'
    '-V~Mq5JjZt!xbi}KoY)n2@VL*hsjGy(QOZ3!3qks%V_GKV?+QbbIORxgCEzFs+b8u3Uk=`hF5*;;m$}W`b6}8cvx&(jG7?yN6LL$c?aord|spG'
    '5={h(r!5cZp4!f~?^BMXW4a(M()U_fQK#_o3t-+XrWB&zpf<*Xar?hAkJ<#CzC@0(Ts)Dh22GH*j+^lMPH@|*vlR`<8qxIw<zvk)pMsFRO)z@)'
    'Z$trwvzJbjv+OT2w3rijQU@c^SZ4=rLgqy&&IRP#2rUS}!$)9eF%U98*Qr(Q+fUdEqZvp=Y$8;t1`%NhXK|1<Qfffq1pKwySdtGy@O4Ro5ok5U'
    'S=k|uL?AtPR#g9FK$LTN!^fKH7koFe>qMCI7NdpH0a{JSm!<<gmNXqQDxV_St%`u5r3hp0Hpdg4>1J=JLtTwWbje3lQCBaig<;1YXFp6@`^&z_'
    'V5hzw<C>caMbnyDLKl#G<Y;NqfTbyX222DSm93FOG_Rvrxl5ukPcv>fckJF=o%W2Rq35t!jXh5E<J>o2lwfnm{PfOEw$wL=z<M%O9xrBFS)@Iu'
    '7Dabc1bt0jQR%0@hSxRc6qNFIfj}B7&pa70=Q2a$FV@c%9TD_F_w8l!NV>MO950K`e%?_Ijy)&<F(nMnD*H#U-enOWPomd|(Z-4v&&NWam(~zV'
    'M9N6QS1_z{txuQO(&2DDX~<$*DG&L`M?Gx??@9ilG_{N0J%GP9RjYExj`U_!wU11Ua&J|A2}ua&Z_Qy;wN+yK+(5YMZEl95;GE3oOSB7Q%C!>~'
    ')~}|Xf_&=&0F|OkrhU}RAG4$Y&?S*RyC)d+2t#dec#nJE^ZQ6?W633baGTj-Ywaz@v$%%-WKjI+;(gLwqfC*EgS`kcb>hD`LG*lpw9><HIZt(c'
    '2DmttA!Tv=5h=dMN@0@<rLFCXr#VLO<~%6=;$HJPcMc);B-ltU8@ziWt-cNNbowM|#Y2tOr3%;1M?h8{)Nr<Fr_8bkKZ^mC6e_iXScSWHz=*>5'
    'urO;J<6dpNlPS%o#b53-^v{7W-8U2hZQjU=UxVFNThy2%*Dh%>e5Gk6kwiGIF)TE3y|%R!R!_AozDGCV2JZ6h*&#dMPGK%4+{60osG;W(eO!ct'
    'r9<?tOc#l$Xy0zw9@Ay_xipR;eBz?j-uD7-HfL0b*glT_`cBcbPbR(i!I*Kb7&M)2r^;2HN8Fj~Find?)oNa$L7`qwSTUVgeZ7>LV5etUC0f};'
    'GH*>Jurn)<qXS_E#MxMP+}Z<UE~kYp;x623E9HRE<|tl-xvhOQk^qjbN0VH@x$TH=MV}mDbHAHYot8(PlHxoEy2j{;!`a)nOhaDnFU>B=Xf9ff'
    'dmy8A5uQSO$uAnawnG+(LpO2cRK3=+3yWN>5emISlXfSPSB>dxh8ic{?Ho!oM^?oJ2S>){Jhk<}-n_2sE3&`5bM@$KAnj3=AxNSgUpE<qta?}='
    'xIFF8-l?R#=T=ixE^<QER8AAs4pwv#;^fQ3GnL@d*xhO>f<&S&caniNha0_&*#5E)-RH7dXGEPzVkT*kMHIq}5^z;fM!=*wDF;waZPUy-_J-TW'
    'bkoy-B3qSvjKCW{%84R*YwUH<jw1_eIi))SfSS*4Nd&V+i<%zMQlUZbp#Q2MCn(fgaA~YA2cELF>H-udFJ_Cn&J<bT7&+}Oj1^ndyEah@euV;I'
    ')Pb5StIg4B`Xy8zyT;TJ=e1dB-ODB8u3lY3Q`KaninllNMGV(u_Pd$?erI$N9gOu5U})QNJ_88EAXfU^tiAGs$Sp5G&$V`Y71oYvnpJ8Sst0)C'
    'g91WeTGiGC<KM$28dqq>jtm0!S98+1se!&=wX+r3>H&{AzAddiF1OEJ*S$}B8eQKtOzWzAT;k~R1$_WewvR*Bik5WaKHP+^O|V2s+9h*9@aG)0'
    'ddjl+Hb|)g8!Al|Txxx+Ksl51v}60(U~~2<gjT^JIqCmNp?RBiG58eSECR}ztJE_2!a4<WGOCZ8SXhu`+h->OemeCdl~CP;^Kl$hob5T>|L?&d'
    '=Uq=NO!>1L`__X5Mpt^?Xgz%cOL@c)0+mvhj88}i$S2^-N(X!OnsXTXLiOFV41Ni4tWywmyjp6X(iMA@&|}rEjXk@Ajrk&@V}Fk<XebC*daA_H'
    'Zo8aOW{TMpWukkgeVs50Pj|^%)%O`rg>9c@51km7sz5D@3ta1{f-3k*t_RCyYoH)k<*7YX#C$o>31Xi@yw!_{s^O%Ua^SBq`GYMbkq06)YshC^'
    'Fmpu2MV<m@HTAW|xns6WQC1H-zEUhk(ta*ybCx??FaJf~X`Swv{SI`2SIyzAXJWS#0fCztp4Xs_ID~@x<2_>xE{wT~+#YO+rYbim9{jyx<wr?Y'
    '--z(rjxVb=1zUt(dzVNxU%WXgPJ1gb4yeOBoyA)*^V`Iii@9S}NNMN(g$kmF%76qw`<x@~VdBa{;Y^f2D>Sj&?yyf_5<_>pB&Mkx!X0XwoNF~6'
    'o{8&C(wLM%kgJ<p4!Y#@5pYenv!?AAV774|>8OMsSiQIJ*e~i;f-dQM?0Mvz_6^#x-1_H|wo;W(v7NDY^Qtr{;-4&*XwHpLr9MB!Pn!U=ON{t6'
    'vL_((TSpc-oNZI;v$UqHOr^B|jqA;;l%_ng^eslwLwTVDA7SHT7R2E5bOG3RFe%B^%ZiJ!1ECCwswt_>9^ZJ?@pHjYm!lCmNt{-r-6WgDe$q5b'
    ';OpyQrbH=nw7fz;@Y7A8wV`Jpic1j`g=e01d9j!F;G001Ls-+<W&=$!<;4gFFHu7eA5=ccrXU+}!H(0X#=K$|naFcv<y=|vbGT{llK{b7ncH?a'
    'etGTZbx&;0FS*mUp{BAUUo=(~y0kI|ZSZ&UlLfwthcf2yJhV^xXzmvp$0qpSahYG*r_`l83UgJNDY`o0xRn>W4`6{I%LX_?)wqaW(G5$WrTvcp'
    'exkV#Ce5XGH+Zomn2N&-S!WMb#wFXMJJcJ*H#3LKa_p<j;eFa43GEVjz0giVMzOaAOfhEJFv#el!dmVIjAuEo)ie>0ZR_}j;a5=xtnFK+gsm-l'
    '4QXh^J??8c6agW&G=iKgJ5T$pYHVg2tm0h^em#;j(yQS%y^dOxfuxQ$nC+DPg)4)h@>C{ewWl-Rt_-VzTGaZyKAe`Sm_QrtTa}Jovh>-b8!=4m'
    '7hG4hm9D!{rtQDOWj!;^;wDDB0=rtIRiu5F^;y|cPufORs;onxzGaIT0zrH0;VytnFC7=w<qZ%3DpH{qNaxwBJ@jNrh2?(RvQFsl8jxPXy&5op'
    'rkMl9A2NV)mtu(Z#TT$O_JtIvuHMR?07H!&QY_Gkl%hlwx(!FGY;PHFc#t`lpwMptg<%XxDrYaA&XB)A)$>{IThK%||8OaM<EN3DqFT*1xOwO0'
    '279&r3-{a_hM56B%>s@fPUVPT**@$Nby1(+_UdJ2TsovwUaV<{*<K8EeU4FjiE=clKG;6|+O@OA9ebt#rI&51Nn3_<8;r!(GZrv(PBnz$qji#D'
    'z)ZFQ57o|f=q&Ly(Y}`|Qi`f+_moEVFKLIG=0#g8<t%DZ`v_7$3efK2>=uOFP!*P|1@y#zx<q(N%)4KA$4~S<(qQ8ffx}n##OlxY+DQ}}BpfRB'
    'dEwkkGdxi3sI4XKr&IAIkk7ja#P03TKTB?0dNv5Jd|VJ*qIY9-RDv8{zZTYY#|pY;r87f4VtcL$QDj6W^_<*Yp>q%uUH2K?Hk`L#q-5aO-c^iU'
    'KFUv}D+`RLB<ST83a%s$vSv^4Y}!X5XRMF(?L~2;`FqZn>%$CVJH!Y}J-*;P*U>cXM?~#0I@nX0R{KJWotnOD>WMmJQD1z>;6%;t)xT39t}w5J'
    'kW08p*|qwCbGLY(<mm|TTUJdcE|U!gS^N{e_zLq!I?^S}%XXqBcM#{?0FNt?oB@^Fn0IoV8-rRsY3*DxOOPz;l4$SF-n&J%_bjLoU*_)Z4()f<'
    '84{8OJi)iD=bX)}j}lNINGc&Bqzfz&PSg+^O}6K}JP07~s~TDEUW#Y~Ynhth-4Y=FRzjK;zV4Um%+~3=f3yc%H>?&6QAZf+(FrJhfx5wW?fMG!'
    '!J&@QCwj?*=Ij@cU@k9Qoo=;$i_&_w|L2rIh?`R?FE7^97piQAaarOR?AM?{)nDkC_Nzo47_407?zysQaT&~tY+R-86&i>;;ar+RZ9jB1$&$q*'
    'l}h+`v9k^31aX15uaW*@oQ@HSi#E#0QW0z<Mq|HdkNt6p=Ej$ST=@a4l$W&nF-Hz-bAUZP5-+X4+zO!3O^j}9jV3DYv0H?_qfJ72;73J$2FT~4'
    'F&U$yudLDLxEm3A^PaFQs@8FD75-uvic(mZZN_{_Ey0qCK`sv7j8x-wxAut(@EI%MHa*00D9LO1a_vf!^(&mI94Eo1^q@ct<MeE!No}dbfY{WY'
    'zEzqsj<R}JXIb_`9Eu~#WLxh$<j^wOsJC|5>ZL|3R8psYTP=WWxi~8!;FvOiCRD4QNM4*e;|mZZJtfT_0aLfCO3fMfe3s7uiKu^G*ts2N&Z))8'
    '7<%fhzUkk=P_#SYNpI`UXUIMBkfX+{RW28c!Rfg{(4<Q<ydvS2#_6_hodDM9J>_b<ipT0iy+3O7Rm4x^vL!*+3{1~E48`E$RZ7(<ORjmU8F7mr'
    '{Z({O>CZ2fPZMGwH|oatx9%vx97nrO2`8-g=Q8}`8T2v(nL0S$)#Y)(OgM)GV{uhzt+prytkfeJgyJ$8_BJh^iQbMdA0QeTI^69e%!s|*Wrv-|'
    '2;#z_aOa{?&cuNrD%2eJYb|=+J9vQ6HO(8f!-}-?sooc@KOYUqzQQyvzR6bV68G`kV~cYY&V30&$cyJ%?Neprj-jnNvm|*TTMsjriwl49-9m;B'
    'nooBY;OMXqEbOqtPMB($ZZAwRA6KG%4k824oB=%PMdV4uJT~D2?AQ-*`Eu;h_L{6dgk=LvY8JQM@<CG&n~b3+9}0{!sIw$&a_1>2YBhK(>4N^Z'
    '%StdH`po;DR;w#JW>6bIaJ-Sz__RSm=OmmWpusZ=c?7F-D`2_pa5EAE=<q1g88HP0x9%Y#Iv8eW(#d2e**BKLlmarV4(KssNW=PfbSwoG)#Ix4'
    'AuZ(qA38KZW4NX6Xe5f29$t;Qw>Y%;A$(1`kG6B<JFV(X;hE4=q#?TSzQ+q!kvON;N?1GQl9GgPKHA4O1*a0Z7BLw7O>zO2Uzu!#B}!<I)CdX>'
    'WD)ibvzCh6qhaL522p6CmZjR4U$j;%bb61b6*MdJiYT~<P}AM#%Y}r<fK4$oM9NEOQdsrkX<6D8<+_wa5Hv7E<dB->^Qt?cnO+Qm16#i;bUHh#'
    'dhHR%9U`QDuQs0dSV&)16f{`~A1`19*W+^b!N#+Xs~uh_4LcgT`HUBfCw|wczi$P$tpxc*(4v*{s!K}I+?vAH6}GpBHetrmDFweYkrhFYt#R}f'
    ';a^<chZC;K*glu=)3gh$eV#Iiv;$}D6Nqi^#QDA@pbpSC-c1<9sBfkg82-eKsPWTUpY7Cv<e}R_yAzKyvP9G}^|LLwsU}`*zZv$M_?)PiBSv-p'
    '%jq;XB!<Tt!c39?3M=04EUPyQ9r2y>T-EHLry4EK@*8bLzB&f)2qLTv_YCa<^;6fi6CLP7l-#47949Jtby4(V#Gxo2wxk}8#E;ov8ohP(y_Z&P'
    'w@TlV)4BAQl1^9@q$7UatxUbE%@y_6?2eueKgY^%dunWNldIkE^sX?J-QHtYp3W;eop+5)FNLX$v%Vyu<MGMftx!TkiN+|7rbQ7U9ak7kOPBO='
    'RBaE6S3=^?b<90147=KC=UU-2QltRu2zYXl>=L3nj$Q_#a)-kam=#g@+LtnJ<+)Yv<FfgdM<bY=t`UP$8l3052+B^*N$1Co(Fp7=VKHW(<4orZ'
    '9Y}BPo7@j_#%WH`$rK~^pksDgCX<Ax)q8BB1zK(QDM@`Jk?`PM8&lbQ!!lkrkLBJ>Ii;Y0Bi=t1S%u-&qCpwOc;BZeApMf4MM;5U_SK_C5AX`B'
    '$fP_`f<O0DZcQufPSIeh$1(zpq0M&0qxq5X98Cs^BI!k>=O9(=X^<C7bS8V0zJlj_NmdOtXcsmh8B2Y9-cIn$@O0~2CUNR!uJYP&23H`&bl=MC'
    'MW3TANmrilUPdHIU_}##HhRoCxB?pU{RiGE-$87Pj%tW$xd&em>)L$wi{GIkWTcgGbV-P8`<;s9;xh73a*3PGaCE11jL?#vvDw$D%%~MubW$l;'
    '&6S~Wo+2)@C63QjF=E(p7KkFy(@vU{keSYZZZk~#Y;_&zYI*C{QTY1sCJ;oEr(?h4?M`yLC7<6Ew{jDg`8#)rhJv(x?9Q>Ke%f~09?Q`im6r!V'
    'fc*BI!gqaK;d#m{m*W6vh4D70DcOrwo98>$r1@Z|tQ1^7?exRNoJ9{sqLL)Bu!4G&e+{^fZ)vuJ3WJ*}5D|#fb}b6qBK8pXT$=_~ualkE92Fgg'
    'CU~SMGS_oAFEGJ}1i#HnbVLm&DD223$?_QRO4Ryaka6EPw2wl)02V*aFZ=U@AzOX8B;J?r`}$%ADTNNx?^>kuz-4em_HHLPpY8Oa-OM@3f`f^W'
    '%L<u%RxZ>m5$kWSMpS&}l0Cx28OjfVv-J$Pb(b@^A5d1KK1JsGn;!vye>_whHzb$&XeG%v<{z%Q?%+!w06~2b2&`S@c1VXP)rlsuRdvV3ob9?H'
    'O<JF=Wcuz2d%iY*$rRes5X%hRLiQm%taLkl-oT~T!3IC9L|-s<Sk(|9{8#NU#Qx}MkS(LLIFhPM7t(^=KnI!SvG^+Q_WV+=hIc2tR4NeC<4XG0'
    'Rp|H!v{x?zySZoCvA}L1o@ucYkOD0OdJhLN#8338&2p!D3W<|k(-J`NDkSezUuZe@9#?tI0pKK!Qnh@RGINq)I1Yz?#;zGbcJ6`Zqua*4)7)SE'
    'VEnW$(hk822>kB`H^LmukyBd|$Wp%`Br%CuM=wP(jg?C6c{L<CapxHe?B0^}s?!JM9Au!s^9}?u73ILr`bl>zh+-Qf!*|Vaf3`a)!FhTR4se$9'
    'TY*0{8F?_hpGaOv_PuDLC`-5X9^yaiC`;;SimaTn2Kl5sYUBdP34}xcIG}wY%C)NjA_xSDY^KAGvttg#(lxWdt?MI-IRzChF@~)_B@m8JwYq#O'
    'Zw)_Oi`WSCH6Kyt<vvK;3HDhY?y%ba5C_w!POJ1T;b#dmm;@TLxbEfmBQ+BJ4lwv44Q3|3N{Fx^+KqgwyCZ{->Ll$kBqAO!#3sY)%PJ{kVI3rS'
    'c7Ss#qlZA;Nf@mpI48PWBL7BuuyzzZG*~)AR3FXF#^qd|n#Wz_Rw8&;CLDOJBy58@_knUFvjP(wO6(w(T2XM$BYT&viSpc*ye;|?_8qM9%}M<+'
    'UP?E)dZ7l*!Kz*!z=}aeP`uE(_Q_ap-lxeyA8yb*Jcc<O5$M)s;)t(v<j)!lcHaQDQ|?u5&&`@@F0gXMzR8t8S2gJt<!rB1i#d&{tBY9pKpewo'
    'QwS7Y&pHPMWt*aGDxJy6iU>;=_;T!1P9fB3F9}<Ak!aD~5<u@>3N?b*eoxKvU%pYdGQ*svLb2N9#=<0bomM+#cyBqJ6JnPW&b(ESb`M#n_5XFp'
    '<4OZzsCpK)J@5>c1nR<LK2vla+olAH_x4^A9KD+)d(B1`We>A4z`l*rW2vWa)nIK_$UC(%r{2}MZ`EU0Nl&PcHr@yt)41~6`HiQ?YwjfBWefdk'
    '-~qp`>ASbHr~$t9Az72mY{n&5dt1~*C*I_Ni&M)rp`F5joSs^G5Hg1zvCk(k0!ttPNRoeOrOYZ(;{s`4CcgM?G=5-cy46KFngsX1u2#A9WmB;&'
    'y_7q{$UX}u0+C3~K8HGw2NeEw@u-ROSrMrENG<=kA0kIO(K@&<X_$^vkl4meGb2LpP^yfpgnTS&=r>D}7i>k-A&zvmFS<Jx_4ae@7@bK;?Lj-<'
    'FMfr5qatbXQ&T+(pt&)lZQipv@r#JmlWE_aJx1%jf(?jSx~MvjMU4wGi(Ap^4=d?D|1Q>v0oWj&`h#iPez6ufIW&DPYP;Wgw3{S6(IJvh!ggs;'
    'Q-mx;uG8kds1q&FB3BaT=+V^OclnXh!9izx$7QB-K;U@KYnHGJIMbXGH%KeKZK-9x-l?(Hw<abqz&=*KNC-i0uFMVxF1Xs(l8aMyAN=~3gvag9'
    'PVzd2#+YOTKceVrl|EV2tbs1Ui2Bt2ppf>?a#wVaX?Ry$gf~mW!>@|>sxBm*f{Ob>4u(}rM%Hd7%F>MUR(I6vW$C0W96i+XcCroSm0(^9nNB3D'
    'c7Yb#7T~=s$uTMn6$5#!JrIOqsbL0gngskoZM4n0hht=TWcFBewS&6frmBf=!BC1SHnc=*pLPp<nVE#}K#^W{m<^0D0fviQ9iRtEST=C_9_Pp7'
    '<m-~<!*FLldjIcB8S`Q~2@eitB>K0iO<+Y77F^b-P0C9mkL_1efy)c~@eWnMxc_@PiZK#-r8j(C#@A?J>tZth(ww9>G4qDQmuvWUVb%Ag9RwOA'
    'd#LlY#!DI7LSOpua^oMWkg$rjM3M2sOQn`2D3Hmh0J+Csu8YnqNGu%e&s#SM8dgO*OpambakFw=UY1*+ghpCvIj+{7Df+a?%vwvP+FVuPTi;#8'
    '#W@-WrtayP7Uw-~tRG7iI7G#@z%Fn14iu>UMXlSGiX}F*w-`NRPx%x<5`?x43BG(Q@hHCiLeiI*650ilhJg_knqJhoAWb~WM{AW9c+$f)X!g>5'
    'j;KttPXex)^k&>GD>a_$#8+FPfoqXNBylvhBE&h?9$76t@-l4NgtwmC`D@9qNH4K5CBmXzSHzZYbb~*92ubU=HZ0G(M5a?DT)S*c@@MLq*xq;K'
    'm+g*Z&-wnr-j8@{5-%b;D9|zDq-oaI!MS@<mCzP3a@RaZ2xv`%-CXK+j6~0=$M%`gPJuYm>QZj~r^CWTlFo;uYj%L>E0*59X_p#a0vt?YHY65j'
    '?NE42P=jRQcF@Z^$x~u0(=LBeOOD=!9SlWUav8wMxqp#3YNu8h7_~9I&J-gyu4XD4g&AEE6Plb`^Xtr>%|uS50bgkTeOlRQ=7>$+Gw~#k5T{Yy'
    '@qON%mrW2Z__C4&cEYc*>)-lZ*o^(s*3@`3IOA5RWbaDsz`g$G6&q&@>_l5+=X=zCFKkq@tE3QEN+#x0rT)jo;ce-@MUmi1N_r1<q3Hs>3mEmv'
    'i+T!AihUclUPrjFnG;Kv8PrtLlI2{b*AugqnAqa_kv+pz{-XMyu5k!%Ac%if6Hf$LX0(=syVdMjZ=vb_g~R0k4lwBj)Liad#dmoOQ9k=T6fTk$'
    's3Y3^QA(K6H;Ein%7E3eOr_&MqgSY--%Wa#u44s0c9a-|@U0q-jJmCkH12dyI`b|-7Y|)UFjw+tyeF%=3?XgWhq_Z2Q9~ufFIt$BH;G_zJZJEe'
    'sa6wIlO2nrMnQSikw3s@p6l?{$>O0iE^!R|u8nzX(W-;C^}dDig%GMzQh=f%$vKUDgs22K_p+NvJ^&Z?7@5xTqp7p7r|ka;e{5m@Pmu&4mu9`}'
    'xHo6xXn6da90ZPsA=Su1dRj-cfVSppAK&*xV!p!#vJe1@EzYZud$+a++wK>s&iTcpt2g1g&zyG|T_V2e*966N^N@w1VHV7*r^c_(f&eGXKCxmD'
    'cSF%+I;u1=ECAHnIdHSskIjNt_sgx7r7~qsB*e}V7Hv>ho=Otn50hlkB`?+itgaaoAy;X#W%{f<A5`h)2HR`2e=G))S@1DfS4$V*v#rx|YxQ2)'
    'Z1$}T)wJ32$CQ?&Y~5)0!c`3o(f-Ht>DV?#=S4yZh@T6bquH6mk8#5DO1>$3mjp8tVKJg!0Ht)6T>E_;2ny|3o<<WvJNQ_O<nneWES+dA=UBGb'
    'gE?toDI>Vj50Mtv)%+L%kAL}?yuCu8dKNBj!<e*cG(&adr6v<kgaNso5fTw*_k8!2H$+Bz7yXbQiA97>-iDeh#Wh6Ucdugjr0Nc6x*8|0rC5=k'
    'f%wHKDehM6);pJuiFVE&NEZeHK(b9+l0=+ze5xjH8oNdOUAtrfG!SS<ChVvRV95@q{&07HupQmY-KU;x8U4Dz9{rREGT;jl94BfXxgMsL)<w%n'
    'DH9G*5_zxp^pQaq&^4jFtikpXjxwzbuv<*u@eB}~L`E1kb9k}O_8)O<gF+D?Wxd)Uy^g5hs?Z=b({7TIAo1#?9M$km+j9b&Xy3IgAHD59=gL<e'
    'wDic}1&2mDPVT`yz9q~o5O7Ka{!KJY#<~rE_7~C3ij!eDJTVfScQ1>}5$RwvCHeAxu}K6;iIB4PeXgw+7?|946F5%@bMl%}kI>Hev{sj4!o<-z'
    'g>ld9KXwMtaJ2nF4DPOroG-GiJc8z_6O>!MlIF^z@oGo!;U;4r%#M7I4n@9~XCgN3D>}SM6VW#9;~nwvD^*|@G-PvV|GXzT+l>}ATGZqt%A5vs'
    'O0^2Q%-Ch{+NQ=>UB&5QYjjH+W5eCp7@opJFe<Eh^2P0;vzkH{%vW9)yq4HJiz;I_J$!LOIQy+ei@ISqTI~dCf`eJ(w8x43v2!kJ8A0XBr5?Gu'
    'oxP>s<RaQlh{#hUhswyPj{}2v1UDX^er~y%dp$DfFd~)E5a_OX2rk8giXI6OA1x3hCAo^|deiun$d--(wR~<|U>4~d?<R}dK6zC_RN&Y(?qVR='
    'z5}0`j3zFb0!-NM^P6GpE3JZZuD4f@d9<^rH+hpiUesI6aLYj>`;2q@qo!_=5E`a>lcT8uI4JU8VAM`K6c*8{twW=_Js0NQ=CO@tzNTQ~8Uu@N'
    'l?ZiM0S7O_qV81HIU3Iqf_U4y5kJAS-NIe8eY^G=L4O9}qg~1m1Y7HwRKn=QJ~^u~qD4(CS8rktfH9tzKzu7#uGJ#f8Cy@&MQtBWSMLlCJ=^u-'
    '93p#N;IcxNbZ&doAP*~EMqnb&fQRQCFY~pRyoh4Y>c~B2lUOD9v}mA}A*%k?6JIYz^h>lbZQsi1PDFl91LT14aPajK{$290Ru^9@lUsX_MK`@r'
    'eA=mGxo*7B9l0wjRSy$Sje57-x#pBnck#ZZH0H>GECG$BGkdf}T{Plhkm@4el7qie<k$8g%&kh|&d{+8N)a{fvrIe8_on%A_lWIuVF-Mu)d1HW'
    '2Q~Y4o39oB@esr!oRc=s04I@$6tLkp<z$ngFN4KV<aiy_T94qH&yJpby=@=OxJwC#BEwK41O<fEvE-EYBBvOanO2V4&3+}j3x&pGyJ6xj$~Rw#'
    'SyC<H|7?>lB9i74aW_v2d%JAi_mcLeOM=uNIY9kDT*98BYeKtTd}nmNM|8TRo%o|CfT&TN>V`t|*;Zy`s;Bg5rJlo=tV@LRr@_t%dx3JwyC}(i'
    'P!&6*9onVet=NJE9eI94YMzcmpY>=?Y=vZ1X4=~tyOv0eU55MK$Fot*rk`3lVd3ON+5IEBpxV0XZOXtZ6G409?S43cu{$L=iYbz{49-AlR<!+g'
    '370h>oF<4c5KQfWMF{=MCyD#@-c)f3Xy29-wsyhl;ni$2Y6d{t!N&CL!WG-L?&#C2qYMirG-54^z%3cwSFBb)CoGrF7G7V-N07`9x>kU+BqnNW'
    'f&NM$VE36r?Zp{Uxm;gR(yXz=nyEt|pJxh!#ObL)`r}&D>*~tA5z|X%yd>02y9Y6Bxf}DeW%dOlc>Al{!2^5UKehX0qMt2+ee6z;ds~+I;&t8;'
    'v$eb;@dq8u85&Qgi`hg2p6S5Va&j;2)m9B%jdsUF(j-p%iNrT;J>{$#89A<~QA?!5Kgbn)TRwy6Ns>#+5!?z|=@z!Y1V+`wBfvh^qQL6r2>gTK'
    'p(sv{U6k%I&EuRxV<O18a%$0+dV$jmt<w(C(!`Ig!*}(iMz)*0m~Z<LamluY_99c^#c(MJE*jp~o}q(ubAuT7;<A!(^;@GyhCL|NG(Oj*hin&7'
    '0v10G<k~Zs_qSA{VI;6l8I;5QX_r>^ks6VaSA_G#PGOrw)$f<YB%oBQd;18TKEnal=|x%(+V8$Y!Img9C|o=zw+aZ>QJ}4Bu{H3XJOor9$rjyC'
    'rbFx{q8Htdns=E9uFw02vrWgX@3<~pJWKNo%B}RO_m%U_++_dTxLvCJQQOdUI7w=`9Rg9?1RsZ$#aaBKT796XZ=To@b(u)58lu=BPIMe37m5e^'
    'c_mN=*uB&0{(6=$Q2VBhBRenXZ$;%6`b8Aa_u_)dQzZY9!Gc{}(&w7S_qUGFoQHa)d0$t#o$Mz%r~D{;$m2A)$Zzq4a5z30W$(#{9*Q`>aJ|sY'
    '%O30zZPTQjv~eyygtW1;4gtdLSv`)`Oe;&};a~WcOOO4<^rQe>OZvb5w}1TSfBWZu{g?mz-~Z3Q{Nw+&|Ifeu!*ivd7yq~a`=9^uU;od){O^DG'
    '!}i|uU;nRv`qzJWuzw-`&{ZuH-O7JV<@~At@<0CPKmC_~;IXkQXaf@T_~(!Q@%S_H`*}t9`U12U1|6hH7n3}T&X0ckTaKRr>)QrtVpc->ANt~L'
    '{L~Nr`CsJqb>83q&%fO3>%fG!l3`&D04qQFmGU>g(jUL_o4E?jVGF}b%8bfYKeTVYmv8?0>mSYE{e++Ytlr+sSewXG!Z!T5V7~c!;*WqaFRbzD'
    '+Q7kd_dh;vT6UK%+vRhM9!>qhC;1EO>xXaWhkx`J7TS+j=$}9J$^6or_@$Tl(K<j%lu;8Qy(f&|^`ZasCw}bDeTcuYxPRqC{e>6$mF4>Oho~z!'
    'SLd|r<HV_cZ79F~UVi7h`n3=E_utI>Cdzo9G)DJvSVeLzk)PQ$3q;n2?2EvrjC||szx&58uZ-V+P~n=SzNv>e5Yc<-8~d3x@pliWpF7=t=CScn'
    '3n>av`scDsXLxe|t?h>!B>Oe&xMkO4dVZfr!q0pbzq3yNT{cu~aLmT+5(Tt}Fs)!Q{_xBGnXl~a1w1p%u2w_}ft&R$qx|gez+f3<1ti{98{rJ!'
    'zGo|}t)K<zWpy{P3h_ht^E=LOJbXX(DbfNS7?F1&H<`=WuO9ioRfGKcCh|Kw!QVKyv5ErQT^kni!lv-KJN<4&^!Bq6FEyGD{6vHqkDYKj45L*r'
    'rK@x7>3jIClj(~Bgxm?NQYcjzL+FVbe_WTtsXN!9BJ23+6R9!rMVVhPbvu*)`)A|-&$s^fi&rqHBw0`RT!8++T_t{}UY-6ZBy`k)zkJDiSS9Ue'
    'u2ibi&wlU{@f+WXp0XRFJZs~JTjE!x1xBof@gzgIZS{wT-WRX2nb%u!LJ>fg)az=%X$HB9ff+kgn?JY){mOReYBI2biOKA^zd87_P@T)9zXU7c'
    '``hH>AJh4Jr6eM30c=OBA$&WCfeTWGnV9&K{`~VNR7U+~M-(xL4PDZ+kn*72)X=t}i>f91=&F-E`t&$G`=DF;K^gWtmBM8BUFb<A=Yv02$xm#d'
    'gq`COM!VH`h4S-tZ~iM`iso*0^{{x%>MySF_(K5l-%1}`lI<a$=?A}&+YGj6UZ*}8kNt4F`*=FYVYU5|%zH<%zOIXQsd>?MVxI(p@?}NH*ocRn'
    '@v(0mz5K$SO&HI}sRAi!;HQ7Py#0;N(npsJsb{0w-l(s}*N@pxoY$bNvwa}AgJwel5k7k#J1|OkSRoB5W=X&Bt<ou-D0v_XPwZshU&(%4t}JB8'
    '1lmo!iVW8571K8*545%oEG17n^cym}AD#g}QxjIwQOc@FkDO+)qJ{F&kU}q-Iv}+++2<dB$w~z}BdC80isiRu_cIkP;(xm1^uTQ=z@YQ(8~Qu#'
    'pTFGOo{6qQ6u0-UKPVN8;U#0RiV(r$vm7MUb23D9Sk?aZa`1Ob^})Ae6==I;jlHDbC>4I;R{HukiFvdy572PIZZ~?=?umaWj>%MpyV&q~?AP<y'
    '8E`lb7LM{+M|hw9)ALmkLy~*Bamf9!Uj5+-37xi(!b854?HgnL!nSelY&;S$$2NX76n@`tSi@;7rQ82Tp~g`5`&#Xbv%r=>;Y^<hG14eM>Q{bz'
    'n2F$Gcrza7vg?nkjh}zaKbq&A^lmG#7t&sg0l(KN`5WbrjD7~Gun6XWnWXTL4%+{A?mOS3R$>6>%iQGs%eA_A;IG_$CHHE4dms4mxdNmE0#)<O'
    'JjYu7%o;;11nPeSoO<t+{5!SsL4b79<+-pDx^aCne`gvH|1vyVWHs{OjJDreRr6wvlZ||3zU?2CUa$Wg1Tuqs&s2ncy<Y5pySebgFUJXvo_!2^'
    '8(hHmU(OfT4QQ4S3c3FikNu<S<<H;lDF6SOA^rLH`};rtN`G?-i?DRt#f`moSjpSpTbF|ifPEJUOJ{?b{{8j%x0`Ja8*TN<huUq7^b_06?@Xz`'
    'S;(V+E*i4~WU;68@h;#gXTV>xG%R|f|8lkH=?b-DRJVPv4kBv4ehoX!pqje;&<@tN_75JZzxyD3)HWoMFQfw5@3`*`oJ!(fa|A{$sN(AP`&0Z6'
    '`z;Y;xxIJTX7&Bl;qebhe0JTeq@w_Qn)Cb4`Olofe|J{(8}mNL=D=>T-M)Kf-%B&$-+A6gk{Z@>yFz5Im)=AZbj3DZ`|J>KoNGT(Qv9w^;m=?A'
    'PJUd$Lg_|$1I*A?#ZL3ruM%Gm$3P0(Rya|Idr<c8%Rqkd(f#8;{fGYts>Jqj'
)

_BRISQUE_RANGE_DATA = (
    'c-n20%Z}7A5JmU>6`2JqI##`_+&lL00SK{Z4GkIzrqM70KA(z{IMZRgaw+zy$Gy&*kMBObKVgBxIF5&>^W(S6X}G(){=7Wi{~mvwpPwH79)@9f'
    'y8iif8fe<yU%%&;ZQ6f%J`G>n=KOp<4W9-sB9IH|G6Eh4ER~EALo=r1K!veRi`)0<>a<2Iqy~X*>pc}W6PeD*anQnM?%*-URKy#{Uf~#7NatAQ'
    'r?AeUs1~(zOeReH$6*I#C0#jIGYnboafwi>;k9$3)gyb3C;&nAmE%J0!b@LL5SdIXtTEJSeo${XnO<2qsE-ln_LuTTDge$Q*y{zRnzNt#UOAEX'
    '!WBi=vME|RSU9OyR1M?KK@FBFxZ~v1h=MjbtYN%!H2X%`^dyiRsNHh7aA-I!_`R%cet|6MgkFy^Apz_;!sP4=2caCZNN{pdg07+9R3u;C0`$8%'
    'I?9oFt09khK}kgrVvZ%Hg>-QAR*Y?~@g*jeO*v9X=onH+qr--wY=rCPaCr>_2nG_%H7j8XoQ2PE8W>$&d^5N-$Iu=F=26)?#s%1GbPT&|8f1>S'
    'I5!1vtH}`}g4RvOEUpREj!jJ~k`u?;v{FlR`<&9%>F9{sm4KW}E?qHd$Q9N!6DHrOvFtIHJ`2T;dD;6=OFme4Za)p0nx@P`#CU_5MwpLPw|^9*'
    'N49Db<Jo5ZwK_&2O$xNfgh8d*^TX%EKZ3Fiiv'
)


_BRISQUE_CACHE = None


def _brisque_resources():
    global _BRISQUE_CACHE
    if _BRISQUE_CACHE is None:
        import base64, zlib
        files = []
        for data in (_BRISQUE_SVM_DATA, _BRISQUE_RANGE_DATA):
            text = zlib.decompress(base64.b85decode(data)).decode("utf-8")
            storage = cv2.FileStorage(text, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
            if not storage.isOpened():
                raise RuntimeError("Could not read embedded quality-scoring data.")
            files.append(storage)
        model = cv2.ml.SVM_create()
        model.read(files[0].getFirstTopLevelNode())
        limits = files[1].getNode("range").mat().copy()
        for storage in files:
            storage.release()
        shapes = np.arange(200, 10000, dtype=np.float64) / 1000
        ratios = np.array([math.gamma(2/x)**2 / (math.gamma(1/x)*math.gamma(3/x))
                           for x in shapes])
        _BRISQUE_CACHE = model, limits, shapes, ratios
    return _BRISQUE_CACHE


def _brisque_distribution(array, shapes, ratios):
    values = array.astype(np.float64, copy=False)
    negative = values[values < 0]
    positive = values[values > 0]
    if not negative.size or not positive.size:
        raise ValueError("BRISQUE cannot reliably rank a constant or degenerate image.")
    left = float(np.sqrt(np.mean(negative ** 2)))
    right = float(np.sqrt(np.mean(positive ** 2)))
    asymmetry = left / right
    moment_ratio = float(np.mean(np.abs(values)) ** 2 / np.mean(values ** 2))
    normalized = moment_ratio * (asymmetry**3 + 1) * (asymmetry + 1) / (asymmetry**2 + 1)**2
    shape = float(shapes[np.argmin(np.abs(ratios - normalized))])
    return shape, left, right


def brisque_score(image):
    """Offline two-scale BRISQUE using standard OpenCV, NumPy and embedded SVM data."""
    model, limits, shapes, ratios = _brisque_resources()
    original = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    if min(original.shape) < 8:
        raise ValueError("BRISQUE requires an image at least 8 x 8 pixels.")
    features = []
    for scale in (1, 2):
        height, width = original.shape
        frame = cv2.resize(original, (width // scale, height // scale), interpolation=cv2.INTER_CUBIC)
        mean = cv2.GaussianBlur(frame, (7, 7), 7/6, borderType=cv2.BORDER_REPLICATE)
        second = cv2.GaussianBlur(frame * frame, (7, 7), 7/6, borderType=cv2.BORDER_REPLICATE)
        deviation = cv2.sqrt(cv2.absdiff(second, mean * mean))
        coefficients = (frame - mean) / (deviation + np.float32(1/255))
        shape, left, right = _brisque_distribution(coefficients, shapes, ratios)
        features.extend((shape, (left*left + right*right)/2))
        for dy, dx in ((0, 1), (1, 0), (1, 1), (-1, 1)):
            shifted = np.roll(coefficients, (-dy, -dx), axis=(0, 1))
            if dy > 0: shifted[-dy:, :] = 0
            if dy < 0: shifted[:-dy, :] = 0
            if dx > 0: shifted[:, -dx:] = 0
            shape, left, right = _brisque_distribution(coefficients * shifted, shapes, ratios)
            bias = (right-left) * math.gamma(2/shape) / math.sqrt(math.gamma(1/shape)*math.gamma(3/shape))
            features.extend((shape, bias, left*left, right*right))
    features = np.asarray(features, dtype=np.float32).reshape(1, -1)
    features = -1 + 2 * (features - limits[0]) / (limits[1] - limits[0])
    _, prediction = model.predict(features.astype(np.float32))
    score = float(np.clip(prediction[0, 0], 0, 100))
    if not math.isfinite(score):
        raise ValueError("BRISQUE returned an invalid quality score.")
    return score


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

APP_TITLE = "Inspectra AI"

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
if getattr(sys, "frozen", False):
    # Bundled weights are read from ROOT; reports must survive one-file extraction cleanup.
    OUTPUTS_DIR = Path(sys.executable).resolve().parent / "outputs"
    try:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        probe = OUTPUTS_DIR / (".write_test_" + uuid4().hex)
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError:
        OUTPUTS_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "InspectraAI" / "outputs"
else:
    OUTPUTS_DIR = ROOT / "outputs"

# Internal captures, images, and session records stay outside the PDF-only output folder.
APP_DATA_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "InspectraAI" / "data"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SUPPORTED_ARCHITECTURES = {
    "mirnet_v2", "mprnet", "nafnet", "esrgan", "rcan", "srcnn", "swinir",
    "restormer",
    "edsr",
    "hrnet_custom", "smp_deeplabv3plus", "smp_unetplusplus",
    "deeplabv3plus_resnet18",
    "segformer_b0"
}

DEFECT_COLORS = {
    "Blowhole": (255, 60, 60),
    "Crack": (255, 180, 30),
    "Fray": (30, 220, 255),
    "Break": (190, 90, 255)
}

DEFAULT_MODELS = [
    {
        "id": "mirnet_v2",
        "name": "MIRNet v2 - Image Restoration",
        "task": "restoration",
        "architecture": "mirnet_v2",
        "file": "models/restoration/MIRNet_v2_weights.pth"
    },
    {
        "id": "mprnet",
        "name": "MPRNet - Image Restoration",
        "task": "restoration",
        "architecture": "mprnet",
        "file": "models/restoration/MPRNet_weights.pth"
    },
    {
        "id": "nafnet",
        "name": "NAFNet - Image Restoration",
        "task": "restoration",
        "architecture": "nafnet",
        "file": "models/restoration/NAFNet_weights.pth"
    },
    {
        "id": "esrgan",
        "name": "ESRGAN - 2x Super Resolution",
        "task": "super_resolution",
        "architecture": "esrgan",
        "file": "models/super_resolution/ESRGAN_weights.pth"
    },
    {
        "id": "rcan",
        "name": "RCAN - 2x Super Resolution",
        "task": "super_resolution",
        "architecture": "rcan",
        "file": "models/super_resolution/RCAN_weights.pth"
    },
    {
        "id": "srcnn",
        "name": "SRCNN - 2x Super Resolution",
        "task": "super_resolution",
        "architecture": "srcnn",
        "file": "models/super_resolution/SRCNN_weights.pth"
    },
    {
        "id": "swinir",
        "name": "SwinIR - 2x Super Resolution",
        "task": "super_resolution",
        "architecture": "swinir",
        "file": "models/super_resolution/SwinIR_weights.pth"
    },
    {
        "id": "blowhole",
        "name": "SegFormer - Blowhole",
        "task": "defect",
        "architecture": "segformer_b0",
        "defect": "Blowhole",
        "file": "models/Blowhole/segformer_0coco_model.pth"
    },
    {
        "id": "crack",
        "name": "SegFormer - Crack",
        "task": "defect",
        "architecture": "segformer_b0",
        "defect": "Crack",
        "file": "models/crack/segformer_1coco_model.pth"
    },
    {
        "id": "fray",
        "name": "SegFormer - Fray",
        "task": "defect",
        "architecture": "segformer_b0",
        "defect": "Fray",
        "file": "models/fray/segformer_2coco_model.pth"
    },
    {
        "id": "break",
        "name": "SegFormer - Break",
        "task": "defect",
        "architecture": "segformer_b0",
        "defect": "Break",
        "file": "models/break/segformer_3coco_model.pth"
    }
]


# ============================================================
# 2. RESTORMER ARCHITECTURE
# ============================================================

class RestormerNorm(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.w = nn.Parameter(torch.ones(channels))
        self.b = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        z = x.permute(0, 2, 3, 1)

        z = F.layer_norm(
            z,
            (x.shape[1],),
            self.w,
            self.b
        )

        return z.permute(0, 3, 1, 2)


class RestormerBlock(nn.Module):

    def __init__(self, channels, heads=4):
        super().__init__()

        self.heads = heads

        self.n1 = RestormerNorm(channels)

        self.q = nn.Conv2d(
            channels, channels * 3, 1
        )

        self.dw = nn.Conv2d(
            channels * 3,
            channels * 3,
            3, 1, 1,
            groups=channels * 3
        )

        self.p = nn.Conv2d(
            channels, channels, 1
        )

        self.t = nn.Parameter(
            torch.ones(heads, 1, 1)
        )

        self.n2 = RestormerNorm(channels)

        self.f1 = nn.Conv2d(
            channels, channels * 4, 1
        )

        self.fd = nn.Conv2d(
            channels * 4,
            channels * 4,
            3, 1, 1,
            groups=channels * 4
        )

        self.f2 = nn.Conv2d(
            channels * 2, channels, 1
        )

    def forward(self, x):

        z = self.n1(x)

        b, c, h, w = z.shape

        q, k, v = self.dw(
            self.q(z)
        ).chunk(3, 1)

        depth = c // self.heads

        q, k, v = [
            value.reshape(
                b,
                self.heads,
                depth,
                h * w
            )
            for value in (q, k, v)
        ]

        attention = (
            F.normalize(q, dim=-1)
            @ F.normalize(k, dim=-1).transpose(-2, -1)
            * self.t
        ).softmax(-1)

        x = x + self.p(
            (attention @ v).reshape(b, c, h, w)
        )

        left, right = self.fd(
            self.f1(self.n2(x))
        ).chunk(2, 1)

        return x + self.f2(
            F.gelu(left) * right
        )


class Restormer(nn.Module):

    def __init__(self, channels=48, blocks=12):
        super().__init__()

        self.h = nn.Conv2d(
            1, channels, 3, 1, 1
        )

        self.b = nn.Sequential(
            *[
                RestormerBlock(channels)
                for _ in range(blocks)
            ]
        )

        self.t = nn.Conv2d(
            channels, 1, 3, 1, 1
        )

    def forward(self, x):
        return x + self.t(
            self.b(self.h(x))
        )


# ============================================================
# 3. EDSR ARCHITECTURE
# ============================================================

class ResBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                channels, channels, 3, 1, 1
            ),
            nn.ReLU(True),
            nn.Conv2d(
                channels, channels, 3, 1, 1
            )
        )

    def forward(self, x):
        return x + self.body(x) * 0.1


class EDSR(nn.Module):

    def __init__(self, channels=64, blocks=16):
        super().__init__()

        self.h = nn.Conv2d(
            1, channels, 3, 1, 1
        )

        self.b = nn.Sequential(
            *[
                ResBlock(channels)
                for _ in range(blocks)
            ]
        )

        self.t = nn.Conv2d(
            channels, channels, 3, 1, 1
        )

        self.up = nn.Sequential(
            nn.Conv2d(
                channels,
                channels * 4,
                3, 1, 1
            ),
            nn.PixelShuffle(2),
            nn.Conv2d(
                channels, 1, 3, 1, 1
            )
        )

    def forward(self, x):

        features = self.h(x)

        return self.up(
            features + self.t(self.b(features))
        )


# Exact inference architectures from the original local training notebooks.

class MIRNet_v2_CALayer(nn.Module):

    def __init__(self, c, r=8):
        super().__init__()
        self.m = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, max(4, c // r), 1), nn.ReLU(True), nn.Conv2d(max(4, c // r), c, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.m(x)

class MIRNet_v2_CAB(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.m = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1), nn.GELU(), nn.Conv2d(c, c, 3, 1, 1), MIRNet_v2_CALayer(c))

    def forward(self, x):
        return x + self.m(x)

class MIRNet_v2_SKFF(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.a = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, max(8, c // 8), 1), nn.ReLU(), nn.Conv2d(max(8, c // 8), 2 * c, 1))
        self.c = c

    def forward(self, a, b):
        x = a + b
        w = self.a(x).reshape(x.shape[0], 2, self.c, 1, 1).softmax(1)
        return a * w[:, 0] + b * w[:, 1]

class MIRNet_v2_MRB(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.hi = nn.Sequential(MIRNet_v2_CAB(c), MIRNet_v2_CAB(c))
        self.lo = nn.Sequential(nn.Conv2d(c, c, 3, 2, 1), MIRNet_v2_CAB(c), nn.ConvTranspose2d(c, c, 4, 2, 1))
        self.f = MIRNet_v2_SKFF(c)

    def forward(self, x):
        return x + self.f(self.hi(x), self.lo(x))

class MIRNet_v2_Model(nn.Module):

    def __init__(self, c=48, n=8):
        super().__init__()
        self.h = nn.Conv2d(1, c, 3, 1, 1)
        self.b = nn.Sequential(*[MIRNet_v2_MRB(c) for _ in range(n)])
        self.o = nn.Conv2d(c, 1, 3, 1, 1)

    def forward(self, x):
        return self.o(self.b(self.h(x))) + x

class MPRNet_CALayer(nn.Module):

    def __init__(self, c, r=8):
        super().__init__()
        self.m = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, max(4, c // r), 1), nn.ReLU(True), nn.Conv2d(max(4, c // r), c, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.m(x)

class MPRNet_CAB(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.m = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1), nn.GELU(), nn.Conv2d(c, c, 3, 1, 1), MPRNet_CALayer(c))

    def forward(self, x):
        return x + self.m(x)

class MPRNet_SAM(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.i = nn.Conv2d(c, 1, 3, 1, 1)
        self.f = nn.Conv2d(1, c, 3, 1, 1)
        self.a = nn.Conv2d(1, c, 3, 1, 1)

    def forward(self, x, img):
        out = img + self.i(x)
        att = torch.sigmoid(self.a(out))
        return (x * att + self.f(img), out)

class MPRNet_Stage(nn.Module):

    def __init__(self, c, n=5):
        super().__init__()
        self.h = nn.Conv2d(1, c, 3, 1, 1)
        self.b = nn.Sequential(*[MPRNet_CAB(c) for _ in range(n)])
        self.sam = MPRNet_SAM(c)

    def forward(self, img, extra=None):
        x = self.h(img)
        x = x if extra is None else x + extra
        return self.sam(self.b(x), img)

class MPRNet_Model(nn.Module):

    def __init__(self, c=32):
        super().__init__()
        self.s1 = MPRNet_Stage(c)
        self.s2 = MPRNet_Stage(c)
        self.h = nn.Conv2d(1, c, 3, 1, 1)
        self.b = nn.Sequential(*[MPRNet_CAB(c) for _ in range(7)])
        self.o = nn.Conv2d(c, 1, 3, 1, 1)

    def forward(self, x):
        (f1, o1) = self.s1(x)
        (f2, o2) = self.s2(x, f1)
        return x + self.o(self.b(self.h(x) + f2))

class NAFNet_LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm used by the official NAFNet design."""

    def __init__(self, channels, eps=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias

class NAFNet_SimpleGate(nn.Module):

    def forward(self, x):
        (x1, x2) = x.chunk(2, dim=1)
        return x1 * x2

class NAFNet_NAFBlock(nn.Module):

    def __init__(self, channels, dw_expand=2, ffn_expand=2, drop_rate=0.0):
        super().__init__()
        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand
        self.norm1 = NAFNet_LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, 1)
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, 3, 1, 1, groups=dw_channels)
        self.sg = NAFNet_SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw_channels // 2, dw_channels // 2, 1))
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, 1)
        self.dropout1 = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.norm2 = NAFNet_LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_channels, 1)
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, 1)
        self.dropout2 = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inp):
        x = self.conv1(self.norm1(inp))
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.dropout1(self.conv3(x))
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.dropout2(self.conv5(x))
        return y + x * self.gamma

class NAFNet_NAFNet(nn.Module):

    def __init__(self, width=32, middle_blocks=12, encoder_blocks=(2, 2, 4, 8), decoder_blocks=(2, 2, 2, 2)):
        super().__init__()
        self.intro = nn.Conv2d(1, width, 3, 1, 1)
        self.ending = nn.Conv2d(width, 1, 3, 1, 1)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        channels = width
        for count in encoder_blocks:
            self.encoders.append(nn.Sequential(*[NAFNet_NAFBlock(channels) for _ in range(count)]))
            self.downs.append(nn.Conv2d(channels, channels * 2, 2, 2))
            channels *= 2
        self.middle = nn.Sequential(*[NAFNet_NAFBlock(channels) for _ in range(middle_blocks)])
        for count in decoder_blocks:
            self.ups.append(nn.Sequential(nn.Conv2d(channels, channels * 2, 1), nn.PixelShuffle(2)))
            channels //= 2
            self.decoders.append(nn.Sequential(*[NAFNet_NAFBlock(channels) for _ in range(count)]))
        self.padder_size = 2 ** len(self.encoders)

    def check_image_size(self, x):
        (_, _, h, w) = x.size()
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

    def forward(self, inp):
        (h, w) = inp.shape[-2:]
        x0 = self.check_image_size(inp)
        x = self.intro(x0)
        skips = []
        for (encoder, down) in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)
        x = self.middle(x)
        for (decoder, up, skip) in zip(self.decoders, self.ups, reversed(skips)):
            x = up(x) + skip
            x = decoder(x)
        return (self.ending(x) + x0)[:, :, :h, :w]

class NAFNet_Model(NAFNet_NAFNet):
    pass

class ESRGAN_Dense(nn.Module):

    def __init__(self, c=64, g=32):
        super().__init__()
        self.cs = nn.ModuleList([nn.Conv2d(c + i * g, g, 3, 1, 1) for i in range(4)])
        self.o = nn.Conv2d(c + 4 * g, c, 3, 1, 1)

    def forward(self, x):
        f = [x]
        for c in self.cs:
            f.append(F.leaky_relu(c(torch.cat(f, 1)), 0.2))
        return x + 0.2 * self.o(torch.cat(f, 1))

class ESRGAN_RRDB(nn.Module):

    def __init__(self, c=64):
        super().__init__()
        self.b = nn.Sequential(ESRGAN_Dense(c), ESRGAN_Dense(c), ESRGAN_Dense(c))

    def forward(self, x):
        return x + 0.2 * self.b(x)

class ESRGAN_Generator(nn.Module):

    def __init__(self, c=64, n=8):
        super().__init__()
        self.h = nn.Conv2d(1, c, 3, 1, 1)
        self.b = nn.Sequential(*[ESRGAN_RRDB(c) for _ in range(n)])
        self.bt = nn.Conv2d(c, c, 3, 1, 1)
        self.u = nn.Sequential(nn.Conv2d(c, 4 * c, 3, 1, 1), nn.PixelShuffle(2), nn.LeakyReLU(0.2, True), nn.Conv2d(c, c, 3, 1, 1), nn.LeakyReLU(0.2, True), nn.Conv2d(c, 1, 3, 1, 1))

    def forward(self, x):
        f = self.h(x)
        return self.u(f + self.bt(self.b(f)))

class RCAN_CA(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.m = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c // 16, 1), nn.ReLU(True), nn.Conv2d(c // 16, c, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.m(x)

class RCAN_RCAB(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.m = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(True), nn.Conv2d(c, c, 3, 1, 1), RCAN_CA(c))

    def forward(self, x):
        return x + 0.1 * self.m(x)

class RCAN_Group(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.m = nn.Sequential(*[RCAN_RCAB(c) for _ in range(10)], nn.Conv2d(c, c, 3, 1, 1))

    def forward(self, x):
        return x + self.m(x)

class RCAN_RCAN(nn.Module):

    def __init__(self, c=64):
        super().__init__()
        self.h = nn.Conv2d(1, c, 3, 1, 1)
        self.b = nn.Sequential(*[RCAN_Group(c) for _ in range(5)], nn.Conv2d(c, c, 3, 1, 1))
        self.u = nn.Sequential(nn.Conv2d(c, 4 * c, 3, 1, 1), nn.PixelShuffle(2), nn.Conv2d(c, 1, 3, 1, 1))

    def forward(self, x):
        f = self.h(x)
        return self.u(f + self.b(f))

class SRCNN_Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 64, 9, 1, 4), nn.ReLU(True), nn.Conv2d(64, 32, 5, 1, 2), nn.ReLU(True), nn.Conv2d(32, 1, 5, 1, 2))

    def forward(self, x):
        return self.net(F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False))

class SwinIR_CALayer(nn.Module):

    def __init__(self, c, r=8):
        super().__init__()
        self.m = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, max(4, c // r), 1), nn.ReLU(True), nn.Conv2d(max(4, c // r), c, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.m(x)

class SwinIR_LayerNorm2d(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1, c, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        v = x.var(1, keepdim=True, unbiased=False)
        m = x.mean(1, keepdim=True)
        return (x - m) / torch.sqrt(v + 1e-05) * self.w + self.b

def SwinIR_window_partition(x, ws):
    (b, c, h, w) = x.shape
    return x.view(b, c, h // ws, ws, w // ws, ws).permute(0, 2, 4, 3, 5, 1).reshape(-1, ws * ws, c)

def SwinIR_window_reverse(z, ws, h, w, b):
    return z.view(b, h // ws, w // ws, ws, ws, -1).permute(0, 5, 1, 3, 2, 4).reshape(b, -1, h, w)

class SwinIR_WinBlock(nn.Module):

    def __init__(self, c, heads=6, ws=8, shift=0, channel=False):
        super().__init__()
        self.n1 = SwinIR_LayerNorm2d(c)
        self.att = nn.MultiheadAttention(c, heads, batch_first=True)
        self.n2 = SwinIR_LayerNorm2d(c)
        self.ff = nn.Sequential(nn.Conv2d(c, 4 * c, 1), nn.GELU(), nn.Conv2d(4 * c, c, 1))
        self.ws = ws
        self.shift = shift
        self.ca = SwinIR_CALayer(c) if channel else nn.Identity()

    def forward(self, x):
        (b, c, h, w) = x.shape
        padh = -h % self.ws
        padw = -w % self.ws
        y = F.pad(self.n1(x), (0, padw, 0, padh), mode='reflect')
        (hp, wp) = y.shape[-2:]
        if self.shift:
            y = torch.roll(y, (-self.shift, -self.shift), (-2, -1))
        z = SwinIR_window_partition(y, self.ws)
        z = self.att(z, z, z, need_weights=False)[0]
        y = SwinIR_window_reverse(z, self.ws, hp, wp, b)
        if self.shift:
            y = torch.roll(y, (self.shift, self.shift), (-2, -1))
        x = x + self.ca(y[:, :, :h, :w])
        return x + self.ff(self.n2(x))

class SwinIR_Model(nn.Module):

    def __init__(self, c=60, n=8):
        super().__init__()
        self.h = nn.Conv2d(1, c, 3, 1, 1)
        self.b = nn.Sequential(*[SwinIR_WinBlock(c, 6, 8, 0 if i % 2 == 0 else 4) for i in range(n)])
        self.t = nn.Conv2d(c, c, 3, 1, 1)
        self.up = nn.Sequential(nn.Conv2d(c, 4 * c, 3, 1, 1), nn.PixelShuffle(2), nn.Conv2d(c, 1, 3, 1, 1))

    def forward(self, x):
        h = self.h(x)
        return self.up(h + self.t(self.b(h)))

CHECKPOINT_MODELS = {'mirnet_v2': MIRNet_v2_Model, 'mprnet': MPRNet_Model, 'nafnet': NAFNet_Model, 'esrgan': ESRGAN_Generator, 'rcan': RCAN_RCAN, 'srcnn': SRCNN_Model, 'swinir': SwinIR_Model}


# ============================================================
# 4. MODEL DISCOVERY
# ============================================================

def identify_model(path):

    parts = [
        p.lower()
        for p in path.parts
    ]

    name = path.stem.lower()
    joined = "/".join(parts)

    for architecture in ("mirnet_v2", "mprnet", "nafnet", "restormer"):
        if architecture in name:
            return "restoration", architecture, ""
    for architecture in ("esrgan", "rcan", "srcnn", "swinir", "edsr"):
        if architecture in name:
            return "super_resolution", architecture, ""
    # A directory identifies a task, but cannot identify a network.
    if "restoration" in joined:
        return "restoration", "", ""
    if "super_resolution" in joined:
        return "super_resolution", "", ""

    for defect in (
        "blowhole", "crack", "fray", "break"
    ):

        if defect in joined:

            if "segformer" in name:
                architecture = "segformer_b0"
            elif "deeplab" in name:
                architecture = "smp_deeplabv3plus"
            elif "unetpp" in name or "unetplusplus" in name:
                architecture = "smp_unetplusplus"
            elif "hrnet" in name:
                architecture = "hrnet_custom"
            else:
                architecture = ""

            return (
                "defect",
                architecture,
                defect.title()
            )

    return "unregistered", "", ""


def discover_models():

    models = []
    known_paths = set()

    specifications = list(DEFAULT_MODELS)

    registry_path = ROOT / "model_registry.json"

    if registry_path.exists():
        with open(
            registry_path, "r", encoding="utf-8"
        ) as file:

            registry = json.load(file)

        specifications.extend(
            registry.get("models", [])
        )

    for item in specifications:

        spec = dict(item)
        path = ROOT / spec["file"]

        resolved = path.resolve()

        if resolved in known_paths:
            continue

        known_paths.add(resolved)

        spec["exists"] = path.is_file()

        spec["supported"] = (
            spec.get("architecture")
            in SUPPORTED_ARCHITECTURES
        )

        models.append(spec)

    if MODELS_DIR.exists():

        for path in sorted(
            MODELS_DIR.rglob("*")
        ):

            if not path.is_file():
                continue

            if path.suffix.lower() not in (
                ".pth", ".pt"
            ):
                continue

            if path.resolve() in known_paths:
                continue

            task, architecture, defect = (
                identify_model(path)
            )

            models.append({
                "id": path.stem,
                "name": ({"segformer_b0": "SegFormer", "smp_deeplabv3plus": "DeepLabV3+",
                          "smp_unetplusplus": "U-Net++", "hrnet_custom": "HRNet"}.get(architecture, path.stem)
                         + " - " + defect) if task == "defect" else path.stem,
                "task": task,
                "architecture": architecture,
                "defect": defect,
                "file": path.relative_to(
                    ROOT
                ).as_posix(),
                "exists": True,
                "supported": (
                    architecture
                    in SUPPORTED_ARCHITECTURES
                )
            })

            known_paths.add(path.resolve())

    return models


# ============================================================
# 5. MODEL WEIGHT LOADING
# ============================================================

def extract_state(payload):

    if isinstance(payload, dict):

        for key in (
            "model",
            "generator",
            "state_dict",
            "model_state_dict",
            "params",
            "params_ema"
        ):

            value = payload.get(key)

            if isinstance(value, dict):
                return value

    return payload


def clean_state(state):

    cleaned = {}

    for key, value in state.items():

        key = str(key)

        for prefix in (
            "module.",
            "_orig_mod."
        ):

            while key.startswith(prefix):
                key = key[len(prefix):]

        cleaned[key] = value

    return cleaned


def remap_segformer_keys(state):

    result = {}

    for key, value in state.items():

        new = key

        match = re.match(
            r"^segformer\.stages\.(\d+)\."
            r"patch_embeddings?\.(.+)$",
            key
        )

        if match:

            stage, suffix = match.groups()

            new = (
                f"segformer.encoder."
                f"patch_embeddings.{stage}.{suffix}"
            )

        match = re.match(
            r"^segformer\.stages\.(\d+)\."
            r"blocks\.(\d+)\.(.+)$",
            key
        )

        if match:

            stage, block, suffix = (
                match.groups()
            )

            prefix = (
                f"segformer.encoder.block."
                f"{stage}.{block}."
            )

            mappings = {
                "layernorm_before.":
                    "layer_norm_1.",
                "layernorm_after.":
                    "layer_norm_2.",
                "attention.q_proj.":
                    "attention.self.query.",
                "attention.k_proj.":
                    "attention.self.key.",
                "attention.v_proj.":
                    "attention.self.value.",
                "attention.o_proj.":
                    "attention.output.dense.",
                "attention.sequence_reduction.sequence_reduction.":
                    "attention.self.sr.",
                "attention.sequence_reduction.layer_norm.":
                    "attention.self.layer_norm.",
                "mlp.fc1.":
                    "mlp.dense1.",
                "mlp.fc2.":
                    "mlp.dense2."
            }

            for old, replacement in mappings.items():

                if suffix.startswith(old):

                    suffix = (
                        replacement
                        + suffix[len(old):]
                    )
                    break

            new = prefix + suffix

        match = re.match(
            r"^segformer\.stages\.(\d+)\."
            r"layer_norm\.(.+)$",
            key
        )

        if match:

            stage, suffix = match.groups()

            new = (
                f"segformer.encoder.layer_norm."
                f"{stage}.{suffix}"
            )

        match = re.match(
            r"^decode_head\.linear_projections\."
            r"(\d+)\.proj\.(.+)$",
            key
        )

        if match:

            index, suffix = match.groups()

            new = (
                f"decode_head.linear_c."
                f"{index}.proj.{suffix}"
            )

        result[new] = value

    return result


# HRNet inference definition from the four supplied training notebooks.
class HRNet_ConvBNReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

class HRNet_BasicBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv1 = HRNet_ConvBNReLU(channels, channels)
        self.conv2 = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.conv2(self.conv1(x)))

def HRNet_make_branch(channels, blocks=2):
    return nn.Sequential(*[HRNet_BasicBlock(channels) for _ in range(blocks)])

class HRNet_HighResolutionModule(nn.Module):

    def __init__(self, channels, blocks=2):
        super().__init__()
        self.channels = channels
        self.branches = nn.ModuleList([HRNet_make_branch(ch, blocks) for ch in channels])
        self.fuse_layers = nn.ModuleList()
        for (output_index, output_ch) in enumerate(channels):
            row = nn.ModuleList()
            for (input_index, input_ch) in enumerate(channels):
                if input_index == output_index:
                    row.append(nn.Identity())
                elif input_index > output_index:
                    row.append(nn.Sequential(nn.Conv2d(input_ch, output_ch, 1, bias=False), nn.BatchNorm2d(output_ch)))
                else:
                    operations = []
                    current_ch = input_ch
                    for step in range(output_index - input_index):
                        final_step = step == output_index - input_index - 1
                        next_ch = output_ch if final_step else current_ch
                        operations.extend([nn.Conv2d(current_ch, next_ch, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(next_ch)])
                        if not final_step:
                            operations.append(nn.ReLU(inplace=True))
                        current_ch = next_ch
                    row.append(nn.Sequential(*operations))
            self.fuse_layers.append(row)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        branch_outputs = [branch(x) for (branch, x) in zip(self.branches, inputs)]
        fused = []
        for output_index in range(len(self.channels)):
            target_size = branch_outputs[output_index].shape[-2:]
            value = None
            for input_index in range(len(self.channels)):
                contribution = self.fuse_layers[output_index][input_index](branch_outputs[input_index])
                if input_index > output_index:
                    contribution = F.interpolate(contribution, size=target_size, mode='bilinear', align_corners=False)
                value = contribution if value is None else value + contribution
            fused.append(self.relu(value))
        return fused

class HRNet_HRNetSegmentation(nn.Module):

    def __init__(self, width=18):
        super().__init__()
        self.stem = nn.Sequential(HRNet_ConvBNReLU(3, width, stride=2), HRNet_BasicBlock(width), HRNet_BasicBlock(width))
        self.transition1 = nn.ModuleList([nn.Identity(), HRNet_ConvBNReLU(width, width * 2, stride=2)])
        self.stage2 = HRNet_HighResolutionModule([width, width * 2], blocks=2)
        self.transition2 = HRNet_ConvBNReLU(width * 2, width * 4, stride=2)
        self.stage3 = nn.Sequential(HRNet_HighResolutionModule([width, width * 2, width * 4], blocks=2), HRNet_HighResolutionModule([width, width * 2, width * 4], blocks=2))
        self.transition3 = HRNet_ConvBNReLU(width * 4, width * 8, stride=2)
        self.stage4 = nn.Sequential(HRNet_HighResolutionModule([width, width * 2, width * 4, width * 8], blocks=2), HRNet_HighResolutionModule([width, width * 2, width * 4, width * 8], blocks=2))
        combined_channels = width * (1 + 2 + 4 + 8)
        self.head = nn.Sequential(HRNet_ConvBNReLU(combined_channels, width * 4, kernel_size=1, padding=0), nn.Dropout2d(0.15), nn.Conv2d(width * 4, 1, 1))

    def forward(self, x):
        input_size = x.shape[-2:]
        x = self.stem(x)
        streams = [self.transition1[0](x), self.transition1[1](x)]
        streams = self.stage2(streams)
        streams = [streams[0], streams[1], self.transition2(streams[1])]
        streams = self.stage3(streams)
        streams = [streams[0], streams[1], streams[2], self.transition3(streams[2])]
        streams = self.stage4(streams)
        high_size = streams[0].shape[-2:]
        combined = torch.cat([streams[0], *[F.interpolate(stream, size=high_size, mode='bilinear', align_corners=False) for stream in streams[1:]]], dim=1)
        return F.interpolate(self.head(combined), size=input_size, mode='bilinear', align_corners=False)

def build_model(spec, state=None):

    architecture = spec["architecture"]

    if architecture in CHECKPOINT_MODELS:
        return CHECKPOINT_MODELS[architecture]()

    if architecture == "restormer":
        return Restormer()

    if architecture == "edsr":
        return EDSR()

    if architecture == "hrnet_custom":
        if state is None:
            raise ValueError("HRNet checkpoint tensors are required to determine the trained width.")
        width = int(state["stem.0.0.weight"].shape[0])
        spec["hrnet_width"] = width
        spec["input_size"] = 160 if width == 12 else 256
        spec["mask_threshold"] = 0.5
        spec["preprocessing"] = "RGB [0,1]; no ImageNet normalization"
        return HRNet_HRNetSegmentation(width=width)

    if architecture in ("smp_deeplabv3plus", "smp_unetplusplus"):
        import segmentation_models_pytorch as smp
        if state is None:
            raise ValueError("Checkpoint tensors are required to identify this model's encoder.")
        encoder = "resnet34" if "encoder.layer3.5.conv1.weight" in state else "resnet18"
        # Infer output channels and feature sizes from the actual trained tensors.
        classes = int(state["segmentation_head.0.weight"].shape[0])
        constructor = smp.DeepLabV3Plus if architecture == "smp_deeplabv3plus" else smp.UnetPlusPlus
        spec["encoder_name"] = encoder
        spec["output_channels"] = classes
        return constructor(encoder_name=encoder, encoder_weights=None, classes=classes)

    if architecture == "deeplabv3plus_resnet18":

        import segmentation_models_pytorch as smp

        return smp.DeepLabV3Plus(
            encoder_name="resnet18",
            encoder_weights=None,
            classes=2
        )

    if architecture == "segformer_b0":

        from transformers import (
            SegformerConfig,
            SegformerForSemanticSegmentation
        )

        config = SegformerConfig(
            num_labels=(int(state["decode_head.classifier.weight"].shape[0])
                        if state is not None else 3)
        )

        return SegformerForSemanticSegmentation(
            config
        )

    raise ValueError(
        f"Unsupported architecture: {architecture}"
    )


def load_model(spec):

    path = ROOT / spec["file"]

    if not path.is_file():

        raise FileNotFoundError(
            f"Model file missing:\n{path}"
        )

    if (
        spec["architecture"]
        not in SUPPORTED_ARCHITECTURES
    ):

        raise ValueError(
            "Unsupported model architecture"
        )

    # Permit only NumPy scalar metadata used by these training checkpoints.
    # Keep restricted unpickling enabled; never fall back to arbitrary pickle.
    import importlib
    numpy_scalar = importlib.import_module(
        "numpy._core.multiarray" if np.lib.NumpyVersion(np.__version__) >= "2.0.0"
        else "numpy.core.multiarray"
    ).scalar
    allowed = [numpy_scalar, np.dtype]
    if tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 6):
        allowed += [(numpy_scalar, "numpy.core.multiarray.scalar"),
                    (numpy_scalar, "numpy._core.multiarray.scalar")]
    allowed += [type(np.dtype(t)) for t in
                (np.float16, np.float32, np.float64, np.int32, np.int64)]
    with torch.serialization.safe_globals(allowed):
        payload = torch.load(path, map_location="cpu", weights_only=True)

    state = extract_state(payload)

    if not isinstance(state, dict):
        raise ValueError(
            "The checkpoint does not contain "
            "a valid model state dictionary."
        )

    state = clean_state(state)

    if spec["architecture"] == "segformer_b0":
        state = remap_segformer_keys(state)

    model = build_model(spec, state)

    model.load_state_dict(
        state,
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    return model


def release_model(model):

    del model

    gc.collect()

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# 6. IMAGE CONVERSION
# ============================================================

def image_to_tensor(image):

    array = np.asarray(
        image.convert("L"),
        dtype=np.float32
    ) / 255.0

    array = np.ascontiguousarray(array)

    return torch.from_numpy(
        array
    )[None, None].to(DEVICE)


def tensor_to_image(tensor):

    array = (
        tensor.detach()
        .float()
        .clamp(0, 1)[0, 0]
        .cpu()
        .numpy()
    )

    array = np.rint(
        array * 255
    ).astype(np.uint8)

    return Image.fromarray(array)


# ============================================================
# 7. TILED RESTORATION / SUPER RESOLUTION
# ============================================================

def tiled_inference(
    model,
    image,
    scale=1,
    tile=256,
    halo=32,
    progress=None,
    progress_value=None
):

    source = image_to_tensor(image)

    original_height, original_width = (
        source.shape[-2:]
    )

    pad_height = (
        -original_height
    ) % 8

    pad_width = (
        -original_width
    ) % 8

    if pad_height or pad_width:

        mode = (
            "reflect"
            if (
                original_height > pad_height
                and original_width > pad_width
            )
            else "replicate"
        )

        source = F.pad(
            source,
            (0, pad_width, 0, pad_height),
            mode=mode
        )

    height, width = source.shape[-2:]

    output = torch.empty(
        (
            1, 1,
            height * scale,
            width * scale
        ),
        device=DEVICE
    )

    total_tiles = ((height + tile - 1) // tile) * ((width + tile - 1) // tile)
    completed_tiles = 0
    if progress_value:
        progress_value(0)
    stage_started = time.perf_counter()
    if progress:
        progress(f"0/{total_tiles} tiles (0%) — processing on {DEVICE}")

    with torch.inference_mode():

        for y0 in range(
            0, height, tile
        ):

            for x0 in range(
                0, width, tile
            ):

                y1 = min(
                    height, y0 + tile
                )

                x1 = min(
                    width, x0 + tile
                )

                top = max(
                    0, y0 - halo
                )

                left = max(
                    0, x0 - halo
                )

                bottom = min(
                    height, y1 + halo
                )

                right = min(
                    width, x1 + halo
                )

                patch = source[
                    :, :,
                    top:bottom,
                    left:right
                ]

                context = (
                    torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16
                    )
                    if DEVICE.type == "cuda"
                    else nullcontext()
                )

                with context:
                    prediction = model(patch)

                cy = (
                    y0 - top
                ) * scale

                cx = (
                    x0 - left
                ) * scale

                ch = (
                    y1 - y0
                ) * scale

                cw = (
                    x1 - x0
                ) * scale

                output[
                    :, :,
                    y0 * scale:y1 * scale,
                    x0 * scale:x1 * scale
                ] = prediction[
                    :, :,
                    cy:cy + ch,
                    cx:cx + cw
                ].float()

                completed_tiles += 1
                if progress_value:
                    progress_value(completed_tiles * 100 // total_tiles)
                if progress:
                    elapsed = time.perf_counter() - stage_started
                    remaining = elapsed / completed_tiles * (total_tiles - completed_tiles)
                    progress(
                        f"{completed_tiles}/{total_tiles} tiles "
                        f"({completed_tiles * 100 // total_tiles}%) — "
                        f"{elapsed:.0f}s elapsed, about {remaining:.0f}s remaining"
                    )

    output = output[
        :, :,
        :original_height * scale,
        :original_width * scale
    ]

    return tensor_to_image(output)


# ============================================================
# 8. DEFECT IMAGE PREPROCESSING
# ============================================================

def preprocess_defect(image, spec=None):

    if spec and spec.get("architecture") == "hrnet_custom":
        size = spec.get("input_size", 256)
        rgb = cv2.resize(np.asarray(image.convert("RGB")), (size, size), interpolation=cv2.INTER_LINEAR)
        array = np.ascontiguousarray((rgb.astype(np.float32) / 255.0).transpose(2, 0, 1))
        return torch.from_numpy(array).unsqueeze(0).to(DEVICE)

    rgb = image.convert("RGB").resize(
        (512, 512)
    )

    array = (
        np.asarray(rgb, dtype=np.float32)
        / 255.0
    )

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    array = (
        array - mean
    ) / std

    array = np.ascontiguousarray(
        array.transpose(2, 0, 1)
    )

    tensor = torch.from_numpy(
        array
    ).float().unsqueeze(0)

    return tensor.to(DEVICE)


# ============================================================
# 9. DEFECT DETECTION / SEGMENTATION
# ============================================================

def run_defect_model(
    model,
    image,
    spec
):

    width, height = image.size

    tensor = preprocess_defect(image, spec)

    with torch.inference_mode():

        output = model(tensor)
        logits = output.logits if hasattr(output, "logits") else output
        if isinstance(logits, dict):
            logits = logits["out"]
        logits = F.interpolate(logits, size=tensor.shape[-2:], mode="bilinear", align_corners=False)
        if logits.shape[1] == 1:
            foreground = torch.sigmoid(logits)
            probabilities = torch.cat((1 - foreground, foreground), dim=1)
        else:
            probabilities = torch.softmax(logits, dim=1)
        probs = probabilities[0].cpu().numpy()
    if spec.get("architecture") == "hrnet_custom":
        predicted = (probs[1] >= spec.get("mask_threshold", 0.5)).astype(np.uint8)
    else:
        predicted = probs.argmax(axis=0).astype(np.uint8)

    predicted = cv2.resize(
        predicted,
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )

    original_rgb = np.asarray(
        image.convert("RGB")
    ).copy()

    annotated = original_rgb.copy()

    mask_rgb = np.zeros_like(
        original_rgb
    )

    detections = []

    defect_name = spec.get(
        "defect", "Defect"
    )

    color = DEFECT_COLORS.get(
        defect_name,
        (255, 60, 60)
    )

    min_area = max(
        50,
        int(width * height * 0.00002)
    )

    for class_id in np.unique(
        predicted
    ):

        class_id = int(class_id)

        if class_id == 0:
            continue

        if class_id >= len(probs):
            continue

        binary_mask = (
            predicted == class_id
        ).astype(np.uint8)

        probability_map = cv2.resize(
            probs[class_id],
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            area = cv2.contourArea(
                contour
            )

            if area < min_area:
                continue

            x, y, w, h = (
                cv2.boundingRect(contour)
            )

            local_contour = (
                contour
                - np.array(
                    [[[x, y]]],
                    dtype=np.int32
                )
            )

            local_mask = np.zeros(
                (h, w),
                dtype=np.uint8
            )

            cv2.drawContours(
                local_mask,
                [local_contour],
                -1,
                1,
                -1
            )

            scores = probability_map[
                y:y+h,
                x:x+w
            ]

            valid = local_mask.astype(
                bool
            )

            confidence = (
                float(
                    scores[valid].mean()
                )
                if np.any(valid)
                else 0.0
            )

            detection = {
                "model": spec["name"],
                "defect": defect_name,
                "class_id": class_id,
                "confidence": confidence,
                "box": [
                    x, y,
                    x+w, y+h
                ],
                "area_pixels": float(area)
            }

            detections.append(detection)

            cv2.drawContours(
                mask_rgb,
                [contour],
                -1,
                color,
                -1
            )

            cv2.rectangle(
                annotated,
                (x, y),
                (x+w, y+h),
                color,
                2
            )

            label = (
                f"{defect_name} "
                f"{confidence*100:.1f}%"
            )

            cv2.putText(
                annotated,
                label,
                (x, max(18, y-8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

    active = mask_rgb.any(
        axis=2
    )

    annotated[active] = (
        annotated[active] * 0.7
        + mask_rgb[active] * 0.3
    ).astype(np.uint8)

    return {
        "spec": spec,
        "detections": detections,
        "annotated": Image.fromarray(
            annotated
        ),
        "mask": Image.fromarray(
            mask_rgb
        ),
        "mask_array": mask_rgb
    }


# ============================================================
# 10. COMBINED AI PROCESSING PIPELINE
# ============================================================

def image_quality_metrics(image, reference=None, elapsed=None):
    """Grayscale metrics on [0, 1]; references must be aligned and equal size."""
    array = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
    hist = np.bincount(np.asarray(image.convert("L")).ravel(), minlength=256)
    probabilities = hist[hist > 0] / hist.sum()
    values = {
        "psnr_db": None, "ssim": None, "mse": None, "mae": None,
        "sharpness_laplacian_variance": float(cv2.Laplacian(array, cv2.CV_64F).var()),
        "entropy_bits": float(-(probabilities * np.log2(probabilities)).sum()),
        "contrast_std": float(array.std()),
        "processing_seconds": elapsed,
        "reference_status": "Load an aligned clean reference for PSNR / SSIM / MSE / MAE",
        "protocol": "Grayscale; intensity range [0,1]; SSIM window 7 (smaller for tiny images)"
    }
    if reference is None:
        return values
    if reference.size != image.size:
        values["reference_status"] = (f"Reference size mismatch: {reference.width} x {reference.height}; "
                                      f"expected {image.width} x {image.height}")
        return values
    from skimage.metrics import structural_similarity
    target = np.asarray(reference.convert("L"), dtype=np.float64) / 255.0
    diff = array - target
    mse = float(np.mean(diff ** 2))
    values.update(mse=mse, mae=float(np.mean(np.abs(diff))),
                  psnr_db="Infinity" if mse == 0 else float(10 * np.log10(1.0 / mse)),
                  reference_status="Compared with loaded clean reference")
    window = min(7, min(array.shape))
    if window % 2 == 0:
        window -= 1
    if window >= 3:
        values["ssim"] = float(structural_similarity(target, array, data_range=1.0, win_size=window))
    else:
        values["reference_status"] += "; SSIM needs at least 3 x 3 pixels"
    return values


def format_quality_metrics(values):
    def number(key, digits=4):
        value = values.get(key)
        return "N/A" if value is None else (value if isinstance(value, str) else f"{value:.{digits}f}")
    return (
        f"PSNR: {number('psnr_db', 2)} dB | SSIM: {number('ssim')} | "
        f"MSE: {number('mse', 6)} | MAE: {number('mae', 6)}\n"
        f"Sharpness: {number('sharpness_laplacian_variance', 5)} | "
        f"Entropy: {number('entropy_bits', 2)} bits | Contrast: {number('contrast_std')} | "
        f"Time: {number('processing_seconds', 2)} s\n"
        + values["reference_status"]
        + ("\nSelected model: " + values["selected_model"] if values.get("selected_model") else "")
        + (f" | {values['selection_metric']}: {values['selection_score']:.4f} "
           f"({values['candidate_count']} models evaluated)" if values.get("selection_metric") else "")
        + (f" | {values['tied_candidates']} tied scores; first matching model selected"
           if values.get("tied_candidates", 0) > 1 else "")
    )


def run_pipeline(
    original,
    selection,
    progress=None,
    stage_result=None,
    progress_value=None
):

    def update(message):
        if progress:
            progress(message)

    start = time.perf_counter()

    warnings = []
    per_model = []
    quality_metrics = {}
    references = selection.get("references", {})

    original = original.convert("RGB")

    restored = None
    super_resolved = None

    restoration_spec = selection.get(
        "restoration"
    )

    sr_spec = selection.get(
        "super_resolution"
    )

    defect_specs = selection.get(
        "defects", []
    )

    available_models = discover_models()
    def candidates(spec, task):
        if not spec:
            return []
        if spec.get("auto_best"):
            return [m for m in available_models if m["task"] == task and m["exists"] and m["supported"]]
        return [spec]
    restoration_candidates = candidates(restoration_spec, "restoration")
    sr_candidates = candidates(sr_spec, "super_resolution")
    selected_models = {}
    model_comparisons = {}
    total_stages = len(restoration_candidates) + len(sr_candidates) + len(defect_specs)
    completed_stages = 0

    def report_stage(percent):
        if progress_value:
            progress_value(min(99, int((completed_stages + percent / 100) * 100 / max(1, total_stages))))

    def stage_done():
        nonlocal completed_stages
        completed_stages += 1
        report_stage(0)

    report_stage(0)

    def run_enhancement(spec, models, source, stage, scale):
        automatic = bool(spec.get("auto_best"))
        reference = references.get(stage)
        if automatic and reference is not None and reference.size != (source.width * scale, source.height * scale):
            raise ValueError(f"Clean reference for {stage} must match output dimensions "
                             f"{source.width * scale} x {source.height * scale}; clear it to use BRISQUE.")
        ranking = "SSIM" if reference is not None else "BRISQUE"
        best_image = best_metrics = best_spec = best_score = None
        comparisons = []
        stage_started = time.perf_counter()
        for candidate in models:
            model = None
            entry = {"model": candidate["name"], "checkpoint": candidate["file"]}
            try:
                update("Evaluating " + candidate["name"] if automatic else "Running " + candidate["name"])
                model = load_model(candidate)
                started = time.perf_counter()
                output = tiled_inference(model, source, scale=scale,
                    halo=32 if scale == 1 else 24, progress_value=report_stage)
                metrics = image_quality_metrics(output, reference, time.perf_counter() - started)
                score = None
                if automatic:
                    if ranking == "SSIM":
                        score = metrics["ssim"]
                        if score is None or not np.isfinite(score):
                            raise ValueError("SSIM is unavailable for this reference/image pair.")
                    else:
                        score = brisque_score(output)
                        metrics["brisque_score"] = score
                entry.update(status="success", metric=ranking if automatic else None, score=score,
                             metrics=metrics.copy())
                better = best_image is None or (automatic and (
                    score > best_score if ranking == "SSIM" else score < best_score))
                if better:
                    best_image, best_metrics, best_spec, best_score = output, metrics, candidate, score
            except Exception as exc:
                entry.update(status="failed", error=str(exc))
                warnings.append(f"{candidate['name']} failed: {exc}")
            finally:
                comparisons.append(entry)
                stage_done()
                if model is not None:
                    release_model(model)
                    model = None
        model_comparisons[stage] = comparisons
        if best_image is None:
            raise ValueError("No model produced a usable result" + (" and quality score." if automatic else "."))
        best_metrics["selected_model"] = best_spec["name"]
        if automatic:
            tied = sum(c.get("status") == "success" and np.isclose(c.get("score", float("nan")), best_score,
                       rtol=0, atol=1e-9) for c in comparisons)
            best_metrics.update(selection_metric=ranking, selection_score=best_score,
                                candidate_count=len(models), tied_candidates=int(tied),
                                processing_seconds=time.perf_counter()-stage_started)
        selected_models[stage] = best_spec
        quality_metrics[stage] = best_metrics
        if stage_result:
            stage_result(stage, best_image)
            stage_result(stage + "_metrics", best_metrics)
        return best_image

    if restoration_spec:
        try:
            restored = run_enhancement(restoration_spec, restoration_candidates, original, "restored", 1)
        except Exception as exc:
            warnings.append("Restoration: " + str(exc))
    if sr_spec:
        try:
            super_resolved = run_enhancement(sr_spec, sr_candidates,
                restored if restored is not None else original, "super_resolution", 2)
        except Exception as exc:
            warnings.append("Super-resolution: " + str(exc))

    # SELECT DETECTION INPUT
    available = {
        "original": original,
        "restored": restored,
        "super_resolution": super_resolved
    }

    source_name = (
        "super_resolution" if super_resolved is not None
        else "restored" if restored is not None
        else "original"
    )

    detection_image = available[
        source_name
    ]

    base_rgb = np.asarray(
        detection_image.convert("RGB")
    ).copy()

    combined = base_rgb.copy()

    combined_mask = np.zeros_like(
        base_rgb
    )

    def choose_defect_outcome(request):
        best = None
        best_score = -1.0
        comparison = []
        models = request["candidates"]
        for candidate_index, candidate in enumerate(models):
            candidate_model = None
            row = {"model": candidate["name"], "checkpoint": candidate["file"]}
            try:
                candidate_model = load_model(candidate)
                outcome = run_defect_model(candidate_model, detection_image, candidate)
                regions = outcome["detections"]
                area = sum(d["area_pixels"] for d in regions)
                score = sum(d["confidence"] * d["area_pixels"] for d in regions) / area if area > 0 else 0.0
                row.update(status="success", score=score, regions=len(regions),
                           metric="area_weighted_detected_region_confidence")
                if score > best_score:
                    best, best_score = outcome, score
            except Exception as exc:
                row.update(status="failed", error=str(exc))
                warnings.append(candidate["name"] + " failed: " + str(exc))
            finally:
                comparison.append(row)
                if candidate_model is not None:
                    release_model(candidate_model)
                    candidate_model = None
                report_stage((candidate_index + 1) * 100 / max(1, len(models)))
        model_comparisons["defect_" + request["defect"].lower()] = comparison
        if best is None:
            raise ValueError("No detector produced a usable result for " + request["defect"])
        best["selection"] = {"metric": "area_weighted_detected_region_confidence", "score": best_score,
                             "estimated": True, "candidate_count": len(models)}
        return best

    # MULTIPLE DEFECT MODELS
    for index, spec in enumerate(
        defect_specs,
        start=1
    ):

        update(
            f"Running defect model "
            f"{index}/{len(defect_specs)}: "
            f"{spec['name']}"
        )

        model = None

        try:

            if spec.get("auto_best_defect"):
                outcome = choose_defect_outcome(spec)
            else:
                model = load_model(spec)
                outcome = run_defect_model(model, detection_image, spec)

            mask = outcome[
                "mask_array"
            ]

            active = mask.any(
                axis=2
            )

            combined[active] = (
                combined[active] * 0.7
                + mask[active] * 0.3
            ).astype(np.uint8)

            combined_mask[active] = (
                mask[active]
            )

            for detection in outcome[
                "detections"
            ]:

                x1, y1, x2, y2 = (
                    detection["box"]
                )

                color = DEFECT_COLORS.get(
                    detection["defect"],
                    (255, 60, 60)
                )

                cv2.rectangle(
                    combined,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                label = (
                    f"{detection['defect']} "
                    f"{detection['confidence']*100:.1f}%"
                )

                cv2.putText(
                    combined,
                    label,
                    (x1, max(18, y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )

            per_model.append(
                outcome
            )
            if stage_result:
                stage_result("combined_detection", Image.fromarray(combined.copy()))
                stage_result("combined_mask", Image.fromarray(combined_mask.copy()))

        except Exception as exc:

            warnings.append(
                f"Defect model failed "
                f"({spec['name']}): {exc}"
            )

        finally:
            stage_done()

            if model is not None:
                release_model(model)

    if progress_value:
        progress_value(100)
    update("Processing completed")

    return {
        "images": {
            "original": original,
            "restored": restored,
            "super_resolution": super_resolved,
            "combined_detection":
                Image.fromarray(combined),
            "combined_mask":
                Image.fromarray(combined_mask)
        },
        "per_model": per_model,
        "quality_metrics": quality_metrics,
        "selected_models": selected_models,
        "model_comparisons": model_comparisons,
        "warnings": warnings,
        "source": source_name,
        "elapsed_seconds": round(
            time.perf_counter() - start,
            3
        ),
        "device": str(DEVICE)
    }


# ============================================================
# 11. IMAGE DISPLAY PANEL
# ============================================================

class DefectModelList(QListWidget):
    """Toggle a model by clicking anywhere in its row."""
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.position().toPoint())
            if item is not None and item.flags() & Qt.ItemFlag.ItemIsEnabled:
                self.setCurrentItem(item)
                item.setCheckState(Qt.CheckState.Unchecked
                                   if item.checkState() == Qt.CheckState.Checked
                                   else Qt.CheckState.Checked)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            event.accept()
            return
        super().mouseReleaseEvent(event)


class ImagePanel(QFrame):
    def __init__(self, title):
        super().__init__()
        self.setObjectName("imagePanel")
        self.original_pixmap = None
        self.zoom_factor = 1.0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(3)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("panelTitle")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setFixedHeight(28)
        heading = QHBoxLayout()
        heading.addWidget(self.title_label, 1)
        layout.addLayout(heading)
        self.view = QLabel("NO IMAGE")
        self.view.setObjectName("imageView")
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll = QScrollArea()
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(self.view)
        self.scroll.setWidgetResizable(False)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("QScrollArea { background: #111827; border: none; }")
        self.view.installEventFilter(self)
        self.drag_position = None
        self.scroll.setMinimumSize(0, 100)
        self.scroll.viewport().installEventFilter(self)
        layout.addWidget(self.scroll, 1)
        toolbar = QHBoxLayout()
        self.zoom_out = QPushButton("− Zoom out")
        self.zoom_in = QPushButton("+ Zoom in")
        self.fit_button = QPushButton("Fit")
        self.zoom_label = QLabel("Fit")
        for button in (self.zoom_out, self.zoom_in, self.fit_button):
            button.setStyleSheet("padding: 3px 9px; min-height: 20px;")
            toolbar.addWidget(button)
        toolbar.addWidget(self.zoom_label)
        toolbar.addStretch()
        self.zoom_out.clicked.connect(lambda: self.change_zoom(1 / 1.25))
        self.zoom_in.clicked.connect(lambda: self.change_zoom(1.25))
        self.fit_button.clicked.connect(self.fit_image)
        heading.addLayout(toolbar)
        self.info = QLabel("—")
        self.info.setObjectName("panelInfo")
        self.info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info.setFixedHeight(18)
        layout.addWidget(self.info)
        self.set_image(None)

    def eventFilter(self, obj, event):
        if obj is self.scroll.viewport() and event.type() == QEvent.Type.Resize:
            if self.original_pixmap is None:
                self.view.resize(self.scroll.viewport().size())
            else:
                self.refresh()
        if obj is self.view and self.original_pixmap is not None:
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self.drag_position = event.globalPosition().toPoint()
                self.view.setCursor(Qt.CursorShape.ClosedHandCursor)
                return True
            if event.type() == QEvent.Type.MouseMove and self.drag_position is not None:
                current = event.globalPosition().toPoint()
                delta = current - self.drag_position
                self.drag_position = current
                for bar, distance in ((self.scroll.horizontalScrollBar(), delta.x()),
                                      (self.scroll.verticalScrollBar(), delta.y())):
                    bar.setValue(bar.value() - distance)
                return True
            if event.type() == QEvent.Type.MouseButtonRelease:
                self.drag_position = None
                self.view.setCursor(Qt.CursorShape.OpenHandCursor)
                return True
        return super().eventFilter(obj, event)

    def set_image(self, image):
        self.zoom_factor = 1.0
        for button in (self.zoom_in, self.zoom_out, self.fit_button):
            button.setEnabled(image is not None)
        if image is None:
            self.original_pixmap = None
            self.view.clear()
            self.view.setText("NO IMAGE")
            self.zoom_label.setText("Fit")
            self.view.resize(self.scroll.viewport().size())
            return
        array = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
        height, width = array.shape[:2]
        qimage = QImage(array.data, width, height, array.strides[0],
                        QImage.Format.Format_RGB888).copy()
        self.original_pixmap = QPixmap.fromImage(qimage)
        self.refresh()

    def change_zoom(self, factor):
        self.zoom_factor = max(0.25, min(16.0, self.zoom_factor * factor))
        self.refresh()

    def fit_image(self):
        self.zoom_factor = 1.0
        self.refresh()
        self.scroll.horizontalScrollBar().setValue(0)
        self.scroll.verticalScrollBar().setValue(0)

    def refresh(self):
        if self.original_pixmap is None:
            return
        width = max(1, self.scroll.viewport().width())
        height = max(1, self.scroll.viewport().height())
        fit = min(width / self.original_pixmap.width(),
                  height / self.original_pixmap.height())
        scale = fit * self.zoom_factor
        pixmap = self.original_pixmap.scaled(
            max(1, round(self.original_pixmap.width() * scale)),
            max(1, round(self.original_pixmap.height() * scale)),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self.view.setPixmap(pixmap)
        self.view.resize(pixmap.size())
        self.view.setCursor(Qt.CursorShape.OpenHandCursor)
        self.zoom_label.setText(f"{scale * 100:.0f}%")


# ============================================================
# 12. CAMERA CAPTURE DIALOG
# ============================================================

class CameraDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle(
            "CAPTURE INSPECTION IMAGE"
        )

        self.resize(
            850, 620
        )

        self.frame = None

        backend = (
            cv2.CAP_DSHOW
            if sys.platform == "win32"
            else cv2.CAP_ANY
        )

        self.camera = cv2.VideoCapture(
            0, backend
        )

        if not self.camera.isOpened():

            self.camera.release()

            self.camera = cv2.VideoCapture(
                0
            )

        layout = QVBoxLayout(self)

        self.preview = QLabel(
            "OPENING CAMERA..."
        )

        self.preview.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.preview.setStyleSheet(
            "background:#111827;"
            "color:white;"
        )

        layout.addWidget(
            self.preview, 1
        )

        buttons = QHBoxLayout()

        self.capture_button = QPushButton(
            "CAPTURE FRAME"
        )

        self.capture_button.setEnabled(
            False
        )

        self.capture_button.clicked.connect(
            self.accept
        )

        cancel_button = QPushButton(
            "CANCEL"
        )

        cancel_button.clicked.connect(
            self.reject
        )

        buttons.addWidget(
            self.capture_button
        )

        buttons.addWidget(
            cancel_button
        )

        layout.addLayout(
            buttons
        )

        self.timer = QTimer(self)

        self.timer.timeout.connect(
            self.update_frame
        )

        if self.camera.isOpened():

            self.timer.start(35)

        else:

            self.preview.setText(
                "CAMERA NOT AVAILABLE"
            )

    def update_frame(self):

        ok, frame = self.camera.read()

        if not ok:
            return

        self.frame = frame.copy()

        self.capture_button.setEnabled(
            True
        )

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        height, width = rgb.shape[:2]

        qimage = QImage(
            rgb.data,
            width,
            height,
            rgb.strides[0],
            QImage.Format.Format_RGB888
        ).copy()

        pixmap = QPixmap.fromImage(
            qimage
        )

        self.preview.setPixmap(
            pixmap.scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

    def stop_camera(self):

        self.timer.stop()

        if self.camera.isOpened():
            self.camera.release()

    def accept(self):

        if self.frame is None:
            return

        self.stop_camera()

        super().accept()

    def reject(self):

        self.stop_camera()

        super().reject()

    def closeEvent(self, event):

        self.stop_camera()

        super().closeEvent(event)


# ============================================================
# 13. BACKGROUND AI WORKER
# ============================================================

class InferenceWorker(QObject):

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)
    progress_value = pyqtSignal(int)
    stage_result = pyqtSignal(str, object)

    def __init__(
        self,
        image,
        selection
    ):

        super().__init__()

        self.image = image.copy()

        self.selection = selection

    def run(self):

        try:

            result = run_pipeline(
                self.image,
                self.selection,
                progress=self.progress.emit,
                stage_result=self.stage_result.emit,
                progress_value=self.progress_value.emit
            )

            self.finished.emit(
                result
            )

        except Exception:

            self.failed.emit(
                traceback.format_exc()
            )


# ============================================================
# 14. MAIN APPLICATION WINDOW
# ============================================================

class SampleCountField(QLineEdit):
    """A plain text field that accepts positive whole-number batch sizes."""
    def __init__(self):
        super().__init__()
        self.setValidator(QIntValidator(1, 1000000, self))
        self.setPlaceholderText("Enter sample count")

    def value(self):
        return int(self.text()) if self.hasAcceptableInput() else 0

    def setValue(self, value):
        self.setText(str(value) if value > 0 else "")


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            APP_TITLE
        )

        self.resize(
            1650, 930
        )

        self.setMinimumSize(
            1100, 700
        )

        self.original = None
        self.result = None
        self.batch_store = BatchStore(APP_DATA_DIR)
        self.current_sample = None
        self.references = {}
        self.reference_paths = {}

        self.thread = None
        self.worker = None

        self.busy = False

        self.models = []

        self.selection_used = None
        self.metadata_used = None

        self.build_interface()
        self.refresh_models()

    # --------------------------------------------------------
    # 14A. BUILD USER INTERFACE
    # --------------------------------------------------------

    def build_interface(self):

        root = QWidget()

        outer = QVBoxLayout(root)

        outer.setContentsMargins(
            8, 8, 8, 8
        )

        outer.setSpacing(7)

        # HEADER
        header = QLabel(
            APP_TITLE
        )

        header.setObjectName(
            "mainTitle"
        )

        header.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        header.setFixedHeight(32)

        outer.addWidget(
            header
        )

        self.header = header

        # ACTION BUTTONS
        actions = QHBoxLayout()

        self.upload_button = QPushButton(
            "UPLOAD IMAGE"
        )

        self.upload_button.setObjectName(
            "uploadButton"
        )

        self.upload_button.clicked.connect(
            self.upload_image
        )

        self.capture_button = QPushButton(
            "CAPTURE IMAGE"
        )

        self.capture_button.setObjectName(
            "captureButton"
        )

        self.capture_button.clicked.connect(
            self.capture_image
        )

        self.run_button = QPushButton(
            "RUN INSPECTION"
        )

        self.run_button.setObjectName(
            "runButton"
        )

        self.run_button.clicked.connect(
            self.run_inspection
        )

        self.save_button = QPushButton(
            "SAVE BATCH REPORT"
        )

        self.save_button.setObjectName(
            "saveButton"
        )

        self.save_button.clicked.connect(
            self.save_report
        )

        self.save_button.setEnabled(
            False
        )

        self.refresh_button = QPushButton(
            "REFRESH MODELS"
        )

        self.refresh_button.clicked.connect(
            self.refresh_models
        )

        for button in (
            self.upload_button,
            self.capture_button,
            self.run_button,
            self.save_button,
            self.refresh_button
        ):

            button.setMinimumHeight(
                45
            )

            actions.addWidget(
                button
            )



        # IDENTIFICATION FIELDS
        metadata = QGridLayout()

        self.worker_id = QLineEdit()
        self.batch_number = QLineEdit()
        self.product_id = QLineEdit()
        self.sample_number = QLineEdit()
        self.sample_number.setReadOnly(True)
        self.inspector_decision = QComboBox()
        self.inspector_decision.addItems(["Select decision after inspection", "Accepted", "Rejected", "Needs Review"])
        self.inspector_decision.setEnabled(False)
        self.inspector_decision.setToolTip("Your decision for the displayed sample. It is saved with the inspection and can be changed after reviewing the result.")
        self.inspector_decision.currentTextChanged.connect(self.save_inspector_decision)
        self.batch_capacity = SampleCountField()
        self.batch_capacity.editingFinished.connect(self.batch_metadata_changed)

        fields = [
            ("WORKER ID", self.worker_id),
            ("BATCH NUMBER", self.batch_number),
            ("PRODUCT ID", self.product_id),
            ("SAMPLE NUMBER", self.sample_number)
        ]

        for column, (title, field) in enumerate(
            fields
        ):

            metadata.addWidget(
                QLabel(title),
                column,
                0
            )

            field.setPlaceholderText(
                "Enter " + title.lower()
            )

            metadata.addWidget(
                field,
                column,
                1
            )



        # MAIN SPLITTER
        splitter = QSplitter(Qt.Orientation.Horizontal)

        sidebar = QWidget()
        sidebar.setMinimumWidth(310)

        side = QVBoxLayout(sidebar)
        side.setContentsMargins(6, 6, 6, 6)
        side.setSpacing(10)
        selectors = QVBoxLayout()
        selectors.setSpacing(4)
        selectors.setAlignment(Qt.AlignmentFlag.AlignTop)
        defects = QVBoxLayout()
        defects.setSpacing(4)
        defects.setAlignment(Qt.AlignmentFlag.AlignTop)
        metadata.setHorizontalSpacing(10)
        metadata.setVerticalSpacing(8)
        metadata.setColumnStretch(0, 0)
        metadata.setColumnStretch(1, 1)
        side.addLayout(selectors)
        side.addLayout(defects)
        details_title = QLabel("SAMPLE & BATCH DETAILS")
        details_title.setStyleSheet("color: #064c80; font-weight: bold; padding-top: 3px;")
        side.addWidget(details_title)
        side.addLayout(metadata, 1)
        count_label = QLabel("SAMPLES IN BATCH")
        count_label.setToolTip("Number of samples in this batch")
        metadata.addWidget(count_label, 4, 0)
        metadata.addWidget(self.batch_capacity, 4, 1)
        metadata.addWidget(QLabel("SAMPLE DECISION"), 5, 0)
        metadata.addWidget(self.inspector_decision, 5, 1)
        self.batch_count_label = QLabel("Batch samples: 0 | Inspected: 0 | Pending: 0")
        self.batch_count_label.setWordWrap(True)
        side.addWidget(self.batch_count_label)
        for field in (self.worker_id, self.batch_number, self.product_id):
            field.editingFinished.connect(self.batch_metadata_changed)

        title = QLabel(
            "AI MODEL MANAGER"
        )

        title.setObjectName(
            "sectionTitle"
        )

        selectors.addWidget(
            title
        )

        # RESTORATION SELECTOR
        selectors.addWidget(
            QLabel("RESTORATION MODEL")
        )

        self.restoration_combo = QComboBox()

        selectors.addWidget(
            self.restoration_combo
        )

        # SR SELECTOR
        selectors.addWidget(
            QLabel("SUPER-RESOLUTION MODEL")
        )

        self.sr_combo = QComboBox()

        selectors.addWidget(
            self.sr_combo
        )

        # MULTI MODEL SELECTOR
        defects.addWidget(
            QLabel("DEFECT TYPES TO FIND")
        )

        defects.addWidget(
            QLabel(
                "Select one or more defect types."
            )
        )

        self.defect_list = DefectModelList()

        self.defect_list.setFixedHeight(150)
        self.defect_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.defect_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.defect_list.setSpacing(2)

        defects.addWidget(
            self.defect_list
        )

        # SELECT ALL / CLEAR
        selector_buttons = QHBoxLayout()
        selector_buttons.setSpacing(6)

        select_all_button = QPushButton(
            "CHOOSE ALL"
        )

        select_all_button.clicked.connect(
            lambda: self.set_all_defects(True)
        )

        clear_button = QPushButton(
            "CLEAR ALL"
        )

        clear_button.clicked.connect(
            lambda: self.set_all_defects(False)
        )

        selector_buttons.addWidget(
            select_all_button
        )

        selector_buttons.addWidget(
            clear_button
        )

        defects.addLayout(
            selector_buttons
        )

        defects.addWidget(QLabel("DETECTION MODEL"))
        self.defect_model_combo = QComboBox()
        self.defect_model_combo.setToolTip(
            "Choose one model family for the selected defect types. Choose Best runs all available "
            "models for each type and selects the highest area-weighted mean confidence of detected regions. "
            "Confidence is an estimate, not validated detection accuracy. No-region results score zero."
        )
        self.defect_model_combo.currentIndexChanged.connect(self.update_defect_availability)
        defects.addWidget(self.defect_model_combo)





        for field in (self.worker_id, self.batch_number, self.product_id,
                      self.sample_number, self.batch_capacity):
            field.setFixedHeight(34)
        for combo in (self.restoration_combo, self.sr_combo, self.defect_model_combo, self.inspector_decision):
            combo.setFixedHeight(30)
            combo.setMinimumWidth(0)
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(1)
        self.batch_count_label.setStyleSheet(
            "background: #e0eaf3; border-radius: 5px; padding: 8px; color: #173553;"
        )

        splitter.addWidget(sidebar)

        # FOUR IMAGE PANELS
        workspace = QWidget()

        grid = QGridLayout(
            workspace
        )

        grid.setSpacing(6)

        self.panels = {
            "original": ImagePanel(
                "1. ORIGINAL IMAGE"
            ),
            "restored": ImagePanel(
                "2. RESTORED IMAGE"
            ),
            "super_resolution": ImagePanel(
                "3. SUPER-RESOLUTION IMAGE"
            ),
            "combined_detection": ImagePanel(
                "4. COMBINED DEFECT DETECTION"
            )
        }

        self.metrics_dialog = QDialog(self)
        self.metrics_dialog.setWindowTitle("Image quality and detection confidence")
        self.metrics_dialog.resize(760, 480)
        metrics_layout = QVBoxLayout(self.metrics_dialog)
        self.metric_labels = {}
        self.reference_buttons = []
        for key, heading in (("restored", "RESTORATION"),
                             ("super_resolution", "SUPER-RESOLUTION")):
            title = QLabel(heading)
            title.setStyleSheet("font-size: 15px; font-weight: bold; padding-top: 12px;")
            metrics_layout.addWidget(title)
            row = QHBoxLayout()
            load = QPushButton("Load clean reference")
            clear = QPushButton("Clear reference")
            for button in (load, clear):
                button.setStyleSheet("padding: 5px 10px;")
                self.reference_buttons.append(button)
                row.addWidget(button)
            load.clicked.connect(lambda checked=False, stage=key: self.load_reference(stage))
            clear.clicked.connect(lambda checked=False, stage=key: self.clear_reference(stage))
            metrics_layout.addLayout(row)
            label = QLabel("Run inspection to display image metrics.")
            label.setWordWrap(True)
            label.setMinimumHeight(75)
            label.setToolTip("Sharpness and entropy describe content; higher values do not guarantee better quality.")
            metrics_layout.addWidget(label)
            self.metric_labels[key] = label
        self.confidence_label = QLabel("Model confidence: run defect detection.")
        self.confidence_label.setWordWrap(True)
        self.confidence_label.setToolTip("Model probability scores, not statistical confidence intervals.")
        metrics_layout.addWidget(self.confidence_label)
        metrics_layout.addStretch()
        close_metrics = QPushButton("Close")
        close_metrics.clicked.connect(self.metrics_dialog.close)
        metrics_layout.addWidget(close_metrics)
        self.metrics_button = QPushButton("VIEW METRICS")
        self.metrics_button.clicked.connect(self.open_metrics)
        actions.addWidget(self.metrics_button)

        for index, key in enumerate(
            self.panels
        ):

            grid.addWidget(
                self.panels[key],
                index // 2,
                index % 2
            )

        for index in range(2):
            grid.setRowStretch(index, 1)
            grid.setColumnStretch(index, 1)

        splitter.addWidget(
            workspace
        )

        legend = QHBoxLayout()
        legend.addWidget(QLabel("Show:"))
        self.defect_filters = {}
        for name, color in DEFECT_COLORS.items():
            chip = QPushButton(name)
            chip.setCheckable(True)
            chip.setChecked(True)
            chip.setToolTip("Show or hide " + name + " in the combined image")
            chip.setStyleSheet(
                f"QPushButton {{ padding: 3px 6px; background: #edf1f5; color: #657589; "
                f"border: 1px solid #bbc7d3; }} "
                f"QPushButton:checked {{ background: #12263e; color: rgb{color}; }}"
            )
            chip.toggled.connect(self.refresh_detection_filter)
            self.defect_filters[name] = chip
            legend.addWidget(chip)
        self.panels["combined_detection"].layout().addLayout(legend)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addLayout(actions)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([330, 1270])

        outer.addWidget(
            splitter, 1
        )
        outer.addWidget(controls)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #bbc7d3; border-radius: 5px; "
            "background: #edf1f5; text-align: center; } "
            "QProgressBar::chunk { background: #169357; border-radius: 4px; }"
        )
        self.progress_bar.hide()
        outer.addWidget(self.progress_bar)

        # STATUS
        self.status_label = QLabel(
            "READY — UPLOAD OR CAPTURE AN IMAGE"
        )

        self.status_label.setObjectName(
            "statusLabel"
        )

        self.status_label.setMinimumHeight(
            30
        )

        outer.addWidget(
            self.status_label
        )

        self.setCentralWidget(
            root
        )

        self.apply_styles()

    # --------------------------------------------------------
    # 14B. APPLICATION STYLES
    # --------------------------------------------------------

    def apply_styles(self):

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #edf1f5;
                color: #152338;
                font-family: 'Segoe UI';
                font-size: 12px;
            }

            #mainTitle {
                background: #12263e;
                color: #ffffff;
                font-size: 22px;
                font-weight: 900;
                border-radius: 7px;
            }

            #sectionTitle {
                color: #064c80;
                font-size: 17px;
                font-weight: 900;
                padding: 8px;
            }

            QPushButton {
                background: #176b98;
                color: white;
                font-size: 12px;
                font-weight: 800;
                padding: 8px;
                border-radius: 6px;
                border: none;
            }

            QPushButton:hover {
                background: #0c5077;
            }

            QPushButton:disabled {
                background: #aab3bc;
                color: #ebedf0;
            }

            #uploadButton {
                background: #1177bd;
            }

            #captureButton {
                background: #6f46c7;
            }

            #runButton {
                background: #e47713;
            }

            #saveButton {
                background: #169357;
            }

            QLineEdit, QComboBox {
                background: white;
                border: 1px solid #a4b2c1;
                border-radius: 5px;
                padding: 7px;
                font-size: 12px;
            }

            QListWidget, QTableWidget {
                background: white;
                color: #172b42;
                border: 1px solid #bbc7d3;
            }

            QScrollArea {
                background: #edf1f5;
                border: none;
            }

            #imagePanel {
                background: white;
                border: 2px solid #cbd5e1;
                border-radius: 7px;
            }

            #panelTitle {
                background: #f1f5f9;
                color: #173553;
                font-weight: 900;
                font-size: 12px;
            }

            #imageView {
                background: #111827;
                color: #8b98a8;
                font-weight: 700;
            }

            #panelInfo {
                background: #eff3f7;
                color: #31445a;
                font-size: 10px;
            }

            #statusLabel {
                background: #143047;
                color: white;
                border-radius: 5px;
                padding: 5px 12px;
                font-weight: 700;
            }
        """)

    # --------------------------------------------------------
    # 14C. REFRESH / SELECT MODELS
    # --------------------------------------------------------

    def refresh_models(self):

        if self.busy:
            return

        try:
            self.models = discover_models()

        except Exception as exc:

            QMessageBox.critical(
                self,
                "Model Registry Error",
                str(exc)
            )

            return

        self.restoration_combo.clear()
        self.sr_combo.clear()
        self.defect_list.clear()

        self.restoration_combo.addItem(
            "Skip Restoration", None
        )

        self.sr_combo.addItem(
            "Skip Super-Resolution", None
        )

        for combo, task in ((self.restoration_combo, "restoration"),
                            (self.sr_combo, "super_resolution")):
            combo.addItem("Choose Best Model (Automatic)", {
                "auto_best": True, "name": "Choose Best Model (Automatic)", "task": task,
                "architecture": "auto_best", "file": ""
            })
            combo.setToolTip("Choose Best runs every available model: highest SSIM with a clean reference; "
                             "lowest BRISQUE without one. BRISQUE is an estimated quality ranking.")

        for spec in sorted(self.models, key=lambda m: (m.get("defect", ""), m["name"])):
            if (
                not spec["exists"]
                or not spec["supported"]
            ):
                continue

            task = spec.get(
                "task"
            )

            if task == "restoration":

                self.restoration_combo.addItem(
                    spec["name"],
                    spec
                )

            elif task == "super_resolution":

                self.sr_combo.addItem(
                    spec["name"],
                    spec
                )

        self.defect_model_combo.blockSignals(True)
        self.defect_model_combo.clear()
        self.defect_model_combo.addItem("Choose Best Model (Automatic)", "auto")
        for architecture, label in (("segformer_b0", "SegFormer"),
                ("smp_deeplabv3plus", "DeepLabV3+"), ("smp_unetplusplus", "U-Net++"),
                ("hrnet_custom", "HRNet")):
            if any(m["task"] == "defect" and m["exists"] and m["supported"]
                   and m["architecture"] == architecture for m in self.models):
                self.defect_model_combo.addItem(label, architecture)
        self.defect_model_combo.blockSignals(False)
        for defect in DEFECTS:
            item = QListWidgetItem(defect)
            item.setData(Qt.ItemDataRole.UserRole, defect)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.defect_list.addItem(item)
        self.update_defect_availability()
        QTimer.singleShot(0, self.fit_defect_list)

        # Start with automatic model selection in all three dropdowns.
        for combo in (self.restoration_combo, self.sr_combo):
            index = next((i for i in range(combo.count())
                          if combo.itemData(i) and combo.itemData(i).get("auto_best")), 0)
            combo.setCurrentIndex(index)
        self.defect_model_combo.setCurrentIndex(self.defect_model_combo.findData("auto"))

        found = sum(
            m["exists"] and m["supported"]
            for m in self.models
        )

        self.status_label.setText(
            f"MODEL SCAN COMPLETE — {found} "
            f"CANDIDATE MODELS FOUND"
        )

    def set_all_defects(
        self, selected
    ):

        state = (
            Qt.CheckState.Checked
            if selected
            else Qt.CheckState.Unchecked
        )

        for index in range(
            self.defect_list.count()
        ):

            item = self.defect_list.item(index)
            if item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                item.setCheckState(state)

    def defect_candidates(self, defect):
        architecture = self.defect_model_combo.currentData()
        return [m for m in self.models if m["task"] == "defect" and m["exists"] and m["supported"]
                and m.get("defect") == defect
                and (architecture == "auto" or m["architecture"] == architecture)]

    def fit_defect_list(self):
        rows = self.defect_list.count()
        if not rows:
            return
        self.defect_list.doItemsLayout()
        height = sum(max(28, self.defect_list.sizeHintForRow(i)) for i in range(rows))
        height += 2 * self.defect_list.frameWidth() + self.defect_list.spacing() * rows * 2 + 10
        self.defect_list.setFixedHeight(height)

    def update_defect_availability(self, *_):
        for index in range(self.defect_list.count()):
            item = self.defect_list.item(index)
            available = bool(self.defect_candidates(item.data(Qt.ItemDataRole.UserRole)))
            if available:
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable)
                item.setToolTip("Check to inspect this defect type.")
            else:
                item.setCheckState(Qt.CheckState.Unchecked)
                item.setFlags(Qt.ItemFlag.NoItemFlags)
                item.setToolTip("No trained weights for this defect type are available for the selected model.")

    def selected_defect_models(self):
        selected = []
        automatic = self.defect_model_combo.currentData() == "auto"
        for index in range(self.defect_list.count()):
            item = self.defect_list.item(index)
            if item.checkState() != Qt.CheckState.Checked:
                continue
            defect = item.data(Qt.ItemDataRole.UserRole)
            models = self.defect_candidates(defect)
            if models:
                selected.append({"auto_best_defect": True, "name": "Choose Best - " + defect,
                                 "defect": defect, "task": "defect", "candidates": models}
                                if automatic else models[0])
        return selected

    # --------------------------------------------------------
    # 14D. IMAGE LOADING
    # --------------------------------------------------------

    def accept_image(self, image):
        batch = self.batch_number.text().strip()
        capacity = self.batch_store.capacity(batch) if batch else None
        samples = self.batch_store.samples(batch) if batch else []
        if capacity and len(samples) >= capacity:
            if not self.rollover_completed_batch():
                QMessageBox.warning(self, "Batch sample limit reached",
                    "This batch already contains its allotted samples. Finish inspecting its pending samples before collecting the next batch.")
                return
        self.inspector_decision.blockSignals(True)
        self.inspector_decision.setCurrentIndex(0)
        self.inspector_decision.setEnabled(False)
        self.inspector_decision.blockSignals(False)
        self.progress_bar.hide()

        for label in self.metric_labels.values():
            label.setText("Run inspection to display image metrics.")
        self.confidence_label.setText("Model confidence: run defect detection.")
        self.references.clear()
        self.reference_paths.clear()
        self.original = ImageOps.exif_transpose(
            image
        ).convert("RGB")

        self.result = None

        self.save_button.setEnabled(
            False
        )

        for panel in self.panels.values():

            panel.set_image(None)
            panel.info.setText("—")

        self.panels["original"].set_image(
            self.original
        )

        width, height = (
            self.original.size
        )

        self.panels["original"].info.setText(
            f"Resolution: {width} x {height}"
        )

        self.status_label.show()
        self.status_label.setText(
            "IMAGE READY — SELECT MODELS "
            "AND RUN INSPECTION"
        )

        self.current_sample = None
        self.register_current_sample()
        self.update_batch_counts()

    def register_current_sample(self):
        metadata = self.get_metadata()
        if self.original is None or not all(metadata[k] for k in
                ("worker_id", "batch_number", "product_id")):
            return
        if self.batch_capacity.value() < 1:
            return
        if self.current_sample is None:
            self.batch_store.set_capacity(metadata["batch_number"], self.batch_capacity.value())
            self.current_sample = self.batch_store.capture(self.original, metadata)
        self.sample_number.setText(f"{self.current_sample['number']:03d}")

    def batch_metadata_changed(self):
        if self.busy:
            return
        capacity = self.batch_store.capacity(self.batch_number.text().strip())
        if capacity:
            self.batch_capacity.setValue(capacity)
        if self.current_sample is None:
            self.register_current_sample()
        self.update_batch_counts()

    def update_batch_counts(self):
        batch = self.batch_number.text().strip()
        samples = self.batch_store.samples(batch) if batch else []
        inspected = sum(bool(s['report_path']) for s in samples)
        failed = sum(s['status'] == 'Failed' for s in samples)
        capacity = self.batch_store.capacity(batch) if batch else None
        if capacity:
            self.batch_capacity.setValue(capacity)
        self.batch_capacity.setEnabled(not self.busy and not samples)
        target = capacity or self.batch_capacity.value()
        self.batch_count_label.setText(
            f"Inspected: {inspected} / {target or '-'} | Captured: {len(samples)} | "
            f"Pending: {len(samples)-inspected}" + (f" | Failed: {failed}" if failed else "")
        )
        if not self.busy:
            self.save_button.setEnabled(bool(samples))

    def upload_image(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Inspection Image",
            "",
            "Images (*.png *.jpg *.jpeg "
            "*.tif *.tiff *.bmp)"
        )

        if path:

            try:

                with Image.open(path) as image:

                    self.accept_image(
                        image.copy()
                    )

            except Exception as exc:

                QMessageBox.critical(
                    self,
                    "Image Error",
                    str(exc)
                )

    def capture_image(self):

        dialog = CameraDialog(
            self
        )

        if (
            dialog.exec()
            == QDialog.DialogCode.Accepted
            and dialog.frame is not None
        ):

            rgb = cv2.cvtColor(
                dialog.frame,
                cv2.COLOR_BGR2RGB
            )

            self.accept_image(
                Image.fromarray(rgb)
            )

    # --------------------------------------------------------
    # 14E. WORKER / BATCH / PRODUCT / SAMPLE
    # --------------------------------------------------------

    def save_inspector_decision(self, decision):
        if (self.busy or self.result is None or self.current_sample is None
                or not self.inspector_decision.isEnabled()
                or decision not in ("Accepted", "Rejected", "Needs Review")):
            return
        if self.metadata_used is not None:
            self.metadata_used["inspector_decision"] = decision
        sample = self.batch_store.get(self.current_sample["id"])
        if not sample.get("report_path"):
            return
        try:
            path = Path(sample["report_path"])
            report = json.loads(path.read_text(encoding="utf-8"))
            report["inspector_decision"] = decision
            report["decision_updated_at"] = datetime.now().isoformat()
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
            temporary.replace(path)
            # Keep the already-generated completed batch summary current, including the last sample.
            batch = sample["batch"]
            samples = self.batch_store.samples(batch)
            capacity = self.batch_store.capacity(batch)
            if capacity and len(samples) >= capacity and all(s["report_path"] for s in samples):
                folder = OUTPUTS_DIR
                reports = list(folder.glob(self.safe_name(batch) + "_completed_*.pdf"))
                if reports:
                    latest = max(reports, key=lambda p: p.stat().st_mtime_ns)
                    write_batch_pdf(latest, samples, batch, capacity=capacity)
        except Exception as exc:
            QMessageBox.critical(self, "Sample decision could not be saved", str(exc))

    def get_metadata(self):

        return {
            "worker_id":
                self.worker_id.text().strip(),
            "batch_number":
                self.batch_number.text().strip(),
            "product_id":
                self.product_id.text().strip(),
            "sample_number":
                self.sample_number.text().strip(),
            "inspector_decision": (self.inspector_decision.currentText()
                if self.result is not None and self.inspector_decision.isEnabled()
                and self.inspector_decision.currentIndex() > 0 else None)
        }

    # --------------------------------------------------------
    # 14F. RUN MULTIPLE AI MODELS
    # --------------------------------------------------------

    def run_inspection(self):

        if self.busy:
            return

        if self.original is None:

            QMessageBox.warning(
                self,
                "No Image",
                "Please upload or capture "
                "an image first."
            )

            return

        if self.batch_capacity.value() < 1:
            QMessageBox.warning(self, "Batch Size Required", "Enter Number of samples in this batch before running an inspection.")
            self.batch_capacity.setFocus()
            return
        self.register_current_sample()
        metadata = self.get_metadata()

        required_fields = [
            ("worker_id", "Worker ID", self.worker_id),
            ("batch_number", "Batch Number", self.batch_number),
            ("product_id", "Product ID", self.product_id),
            ("sample_number", "Sample Number", self.sample_number),
        ]
        missing = [(label, field) for key, label, field in required_fields
                   if not metadata[key]]
        if missing:
            QMessageBox.warning(
                self, "Required Identification",
                "Enter the following before running an inspection:\n\n"
                + "\n".join(label for label, _ in missing)
            )
            missing[0][1].setFocus()
            return

        if self.current_sample and (
            self.current_sample['batch'] != metadata['batch_number']
            or self.current_sample['product'] != metadata['product_id']
        ):
            QMessageBox.warning(self, "Captured sample belongs to another batch",
                "Capture or upload the next image after entering the new Batch Number and Product ID. "
                "The current captured sample remains in its original batch.")
            return
        if self.current_sample:
            with self.batch_store.connect() as db:
                db.execute("UPDATE samples SET worker=? WHERE id=?",
                           (metadata['worker_id'], self.current_sample['id']))

        selection = {
            "references": dict(self.references),
            "restoration":
                self.restoration_combo.currentData(),
            "super_resolution":
                self.sr_combo.currentData(),
            "defects":
                self.selected_defect_models(),
            "source": "auto"
        }

        if not (
            selection["restoration"]
            or selection["super_resolution"]
            or selection["defects"]
        ):

            QMessageBox.warning(
                self,
                "No Models Selected",
                "Please select at least "
                "one AI model."
            )

            return

        self.inspector_decision.blockSignals(True)
        self.inspector_decision.setCurrentIndex(0)
        self.inspector_decision.setEnabled(False)
        self.inspector_decision.blockSignals(False)
        metadata["inspector_decision"] = None
        self.selection_used = selection
        self.metadata_used = metadata

        self.result = None
        for name, panel in self.panels.items():
            if name != "original":
                panel.set_image(None)
                panel.info.setText("Waiting for processing")
        self.status_label.hide()
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        for label in self.metric_labels.values():
            label.setText("Processing…")
        self.confidence_label.setText("Model confidence: processing…")

        self.busy = True

        self.set_processing(
            True
        )

        self.thread = QThread(
            self
        )

        self.worker = InferenceWorker(
            self.original,
            selection
        )

        self.worker.moveToThread(
            self.thread
        )

        self.thread.started.connect(
            self.worker.run
        )

        self.worker.finished.connect(
            self.processing_complete
        )

        self.worker.failed.connect(
            self.processing_failed
        )

        self.worker.progress_value.connect(self.progress_bar.setValue)
        self.worker.stage_result.connect(self.show_stage_result)

        self.worker.finished.connect(
            self.thread.quit
        )

        self.worker.failed.connect(
            self.thread.quit
        )

        self.thread.finished.connect(
            self.worker.deleteLater
        )

        self.thread.finished.connect(
            self.thread.deleteLater
        )

        self.thread.finished.connect(
            self.thread_stopped
        )

        self.thread.start()

    def set_processing(self, processing):

        enabled = not processing

        controls = [
            self.upload_button,
            self.capture_button,
            self.run_button,
            self.refresh_button,
            self.worker_id,
            self.batch_number,
            self.product_id,
            self.sample_number,
            self.inspector_decision,
            self.batch_capacity,
            self.restoration_combo,
            self.sr_combo,
            self.defect_list,
            self.defect_model_combo
        ]

        controls.extend(self.reference_buttons)
        controls.extend(self.defect_filters.values())
        for widget in controls:
            widget.setEnabled(enabled)
        self.inspector_decision.setEnabled(not processing and self.result is not None)

        if processing:
            self.save_button.setEnabled(False)

    def show_stage_result(self, name, image):
        if name.endswith("_metrics"):
            stage = name[:-8]
            self.metric_labels[stage].setText(format_quality_metrics(image))
            if image.get("selected_model"):
                self.panels[stage].info.setText("Selected: " + image["selected_model"])
            return
        panel = self.panels.get(name)
        if panel is not None:
            panel.set_image(image)
            panel.info.setText(f"{image.width} x {image.height} pixels — completed")

    def processing_complete(self, result):
        self.progress_bar.setValue(100)

        self.result = result
        self.inspection_id = self.current_sample["id"] if self.current_sample else uuid4().hex

        self.show_results()
        try:
            self._save_report(automatic=True)
        except Exception as exc:
            QMessageBox.critical(self, "Sample could not be stored",
                                 "The inspection finished, but automatic saving failed: " + str(exc))
        self.update_batch_counts()

        self.status_label.setText("")
        self.status_label.hide()

        if result["warnings"]:

            QMessageBox.warning(
                self,
                "Processing Warnings",
                "\n\n".join(
                    result["warnings"]
                )[:5000]
            )

    def processing_failed(self, details):
        if self.current_sample:
            self.batch_store.failed(self.current_sample["id"])
        self.update_batch_counts()
        self.progress_bar.hide()
        self.status_label.show()

        self.status_label.setText(
            "PROCESSING FAILED"
        )

        QMessageBox.critical(
            self,
            "Inference Error",
            details
        )

    def thread_stopped(self):

        self.thread = None
        self.worker = None

        self.busy = False

        self.set_processing(
            False
        )

        self.update_batch_counts()
        self.rollover_completed_batch()

    def rollover_completed_batch(self):
        batch = self.batch_number.text().strip()
        capacity = self.batch_store.capacity(batch) if batch else None
        samples = self.batch_store.samples(batch) if batch else []
        if not capacity or len(samples) < capacity or not all(s["report_path"] for s in samples):
            return False
        folder = OUTPUTS_DIR
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (self.safe_name(batch) + "_completed_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".pdf")
        try:
            write_batch_pdf(path, samples, batch, capacity=self.batch_store.capacity(batch))
        except Exception as exc:
            QMessageBox.critical(self, "Completed batch report could not be saved", str(exc))
            return False
        match = re.match(r"^(.*?)(\d+)$", batch)
        prefix = match.group(1) if match else batch + "-"
        number = int(match.group(2)) + 1 if match else 2
        digits = len(match.group(2)) if match else 3
        while True:
            next_batch = prefix + str(number).zfill(digits)
            if not self.batch_store.samples(next_batch) and self.batch_store.capacity(next_batch) is None:
                break
            number += 1
        self.batch_number.setText(next_batch)
        self.batch_capacity.setValue(capacity)
        self.sample_number.clear()
        self.update_batch_counts()
        self.status_label.setText(f"Batch {batch} complete ({capacity}/{capacity}). PDF saved. Next batch: {next_batch}")
        self.status_label.setToolTip(str(path))
        self.status_label.show()
        return True

    # --------------------------------------------------------
    # 14G. DISPLAY RESULTS
    # --------------------------------------------------------

    def open_metrics(self):
        self.metrics_dialog.show()
        self.metrics_dialog.raise_()
        self.metrics_dialog.activateWindow()

    def load_reference(self, stage):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select aligned clean reference for " + stage.replace("_", " "), "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not path:
            return
        try:
            with Image.open(path) as source:
                reference = ImageOps.exif_transpose(source).convert("L").copy()
        except Exception as exc:
            QMessageBox.warning(self, "Reference image", str(exc))
            return
        self.references[stage] = reference
        self.reference_paths[stage] = path
        self.update_reference_metrics(stage)

    def clear_reference(self, stage):
        self.references.pop(stage, None)
        self.reference_paths.pop(stage, None)
        self.update_reference_metrics(stage)

    def update_reference_metrics(self, stage):
        if self.result and self.result["images"].get(stage) is not None:
            previous = self.result.get("quality_metrics", {}).get(stage, {})
            metrics = image_quality_metrics(self.result["images"][stage], self.references.get(stage),
                                            previous.get("processing_seconds"))
            for key in ("selected_model", "selection_metric", "selection_score", "candidate_count", "tied_candidates", "brisque_score"):
                if key in previous:
                    metrics[key] = previous[key]
            if previous.get("selection_metric"):
                metrics["reference_status"] += "; rerun automatic selection after changing references"
            self.result.setdefault("quality_metrics", {})[stage] = metrics
            self.metric_labels[stage].setText(format_quality_metrics(metrics))
        else:
            ref = self.references.get(stage)
            self.metric_labels[stage].setText(
                f"Clean reference loaded: {ref.width} x {ref.height}. Run inspection."
                if ref else "Run inspection to display image metrics.")

    def refresh_detection_filter(self, *_):
        if not self.result:
            return
        source = self.result["images"].get(self.result["source"])
        if source is None:
            return
        combined = np.asarray(source.convert("RGB")).copy()
        for outcome in self.result["per_model"]:
            defect = outcome["spec"].get("defect", "")
            button = self.defect_filters.get(defect)
            if button is None or not button.isChecked():
                continue
            mask = outcome["mask_array"]
            active = mask.any(axis=2)
            combined[active] = (combined[active]*0.7 + mask[active]*0.3).astype(np.uint8)
            for detection in outcome["detections"]:
                x1,y1,x2,y2 = detection["box"]
                color = DEFECT_COLORS[defect]
                cv2.rectangle(combined,(x1,y1),(x2,y2),color,2)
                cv2.putText(combined,f"{defect} {detection['confidence']*100:.1f}%",
                            (x1,max(18,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
        self.panels["combined_detection"].set_image(Image.fromarray(combined))

    def show_results(self):

        if self.result is None:
            return

        images = self.result["images"]

        for name, panel in self.panels.items():
            image = images.get(name)

            panel.set_image(
                image
            )

            if image is not None:

                width, height = image.size

                self.panels[name].info.setText(
                    f"{width} x {height} pixels"
                )

            else:

                self.panels[name].info.setText(
                    "Not selected"
                )

        for stage, label in self.metric_labels.items():
            metrics = self.result.get("quality_metrics", {}).get(stage)
            label.setText(format_quality_metrics(metrics) if metrics else "Stage not selected or unavailable.")
            if metrics and metrics.get("selected_model"):
                self.panels[stage].info.setText("Selected: " + metrics["selected_model"])
        summaries = []
        for outcome in self.result["per_model"]:
            scores = [d["confidence"] for d in outcome["detections"]]
            name = outcome["spec"]["name"]
            summaries.append(
                f"{name}: mean {np.mean(scores) * 100:.1f}% "
                f"(region scores {min(scores) * 100:.1f}–{max(scores) * 100:.1f}%)"
                if scores else f"{name}: no detected regions")
        self.confidence_label.setText("Model confidence — " + "; ".join(summaries)
                                      if summaries else "No defect models selected.")

        self.refresh_detection_filter()

    # --------------------------------------------------------
    # 14H. SAVE INSPECTION REPORT
    # --------------------------------------------------------

    @staticmethod
    def safe_name(value):

        value = re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            str(value)
        ).strip("_")

        return value[:45] or "NA"

    def save_report(self):
        if self.busy:
            return
        try:
            batch = self.batch_number.text().strip()
            samples = self.batch_store.samples(batch) if batch else []
            if not samples:
                QMessageBox.warning(self, "No batch samples", "Capture or upload samples for this batch first.")
                return
            # Include reference metrics changed after the latest inspection.
            if self.result is not None and self.metadata_used["batch_number"] == batch:
                self._save_report(automatic=True)
                samples = self.batch_store.samples(batch)
            folder = OUTPUTS_DIR
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / (self.safe_name(batch) + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".pdf")
            write_batch_pdf(path, samples, batch, capacity=self.batch_store.capacity(batch))
            QMessageBox.information(self, "Batch Report Saved",
                f"Collective batch PDF saved:\n\n{path}\n\n"
                f"Captured: {len(samples)} | Inspected: {sum(bool(s['report_path']) for s in samples)}")
            self.update_batch_counts()
        except Exception as exc:
            QMessageBox.critical(self, "Report could not be saved", str(exc))

    def _save_report(self, automatic=False):
        if self.result is None or (self.busy and not automatic):
            return

        metadata = self.metadata_used

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        folder_name = "_".join([
            timestamp,
            self.safe_name(
                metadata["batch_number"]
            ),
            self.safe_name(
                metadata["product_id"]
            ),
            self.safe_name(
                metadata["sample_number"]
            )
        ])

        destination = (
            APP_DATA_DIR / "inspections" / folder_name
        )

        destination.mkdir(
            parents=True,
            exist_ok=False
        )

        saved_images = {}

        for name, image in self.result[
            "images"
        ].items():

            if image is None:
                continue

            filename = (
                name + ".png"
            )

            image.save(
                destination / filename
            )

            saved_images[name] = filename

        model_results = []

        for index, outcome in enumerate(
            self.result["per_model"],
            start=1
        ):

            spec = outcome["spec"]

            stem = (
                f"model_{index:02d}_"
                + self.safe_name(
                    spec["name"]
                )
            )

            overlay_file = (
                stem + "_detection.png"
            )

            mask_file = (
                stem + "_mask.png"
            )

            outcome["annotated"].save(
                destination / overlay_file
            )

            outcome["mask"].save(
                destination / mask_file
            )

            model_results.append({
                "name": spec["name"],
                "architecture":
                    spec["architecture"],
                "checkpoint":
                    spec["file"],
                "defect":
                    spec.get("defect", ""),
                "detection_count":
                    len(outcome["detections"]),
                "detections":
                    outcome["detections"],
                "detection_image":
                    overlay_file,
                "mask_image":
                    mask_file
            })

        def model_info(spec):

            if spec is None:
                return None

            return {
                "name":
                    spec["name"],
                "architecture":
                    spec["architecture"],
                "checkpoint":
                    spec["file"]
            }

        report = {
            "inspection_id": getattr(self, "inspection_id", uuid4().hex),
            "session_id": self.batch_store.session_id,
            "batch_capacity": self.batch_store.capacity(metadata["batch_number"]),
            "application": APP_TITLE,
            "timestamp":
                datetime.now().isoformat(),
            "worker_id":
                metadata["worker_id"],
            "batch_number":
                metadata["batch_number"],
            "product_id":
                metadata["product_id"],
            "sample_number":
                metadata["sample_number"],
            "inspector_decision": metadata.get("inspector_decision"),
            "device":
                self.result["device"],
            "quality_metrics": self.result.get("quality_metrics", {}),
            "model_comparisons": self.result.get("model_comparisons", {}),
            "metric_reference_files": dict(self.reference_paths),
            "processing_time_seconds":
                self.result["elapsed_seconds"],
            "detection_source":
                self.result["source"],
            "restoration_model":
                model_info(
                    self.result.get("selected_models", {}).get("restored")
                ),
            "super_resolution_model":
                model_info(
                    self.result.get("selected_models", {}).get("super_resolution")
                ),
            "defect_models":
                model_results,
            "total_model_detections":
                sum(
                    item["detection_count"]
                    for item in model_results
                ),
            "warnings":
                self.result["warnings"],
            "saved_images":
                saved_images
        }

        enrich_report(report, self.result["images"], APP_DATA_DIR / "inspections")

        with open(
            destination / "inspection_report.json",
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                report,
                file,
                indent=2
            )

        lines = [
            APP_TITLE,
            "INDUSTRIAL AI INSPECTION REPORT",
            "=" * 50,
            f"Worker ID: {metadata['worker_id']}",
            f"Batch Number: {metadata['batch_number']}",
            f"Product ID: {metadata['product_id']}",
            f"Sample Number: {metadata['sample_number']}",
            f"Detection Source: {self.result['source']}",
            f"Processing Device: {self.result['device']}",
            f"Processing Time: {self.result['elapsed_seconds']} s",
            "",
            "MODEL-WISE DETECTION RESULTS",
            "-" * 50
        ]

        for stage, metrics in self.result.get("quality_metrics", {}).items():
            lines.extend([stage.upper() + " METRICS", format_quality_metrics(metrics), ""])

        for item in model_results:

            lines.append(
                f"{item['name']}: "
                f"{item['detection_count']} detections"
            )

        lines.extend([
            "",
            f"Total Model Detections: "
            f"{report['total_model_detections']}",
            "",
            "WARNINGS"
        ])

        lines.extend(
            self.result["warnings"]
            or ["None"]
        )

        (
            destination / "inspection_report.txt"
        ).write_text(
            "\n".join(lines),
            encoding="utf-8"
        )

        report_path = destination / "inspection_report.json"
        if self.current_sample:
            self.batch_store.inspected(self.current_sample["id"], report_path, self.result["warnings"])
            self.current_sample = self.batch_store.get(self.current_sample["id"])
        if not automatic:
            QMessageBox.information(self, "Sample Report Saved", str(destination / "inspection_report.pdf"))
        return report_path

    # --------------------------------------------------------
    # 14I. SAFE EXIT
    # --------------------------------------------------------

    def closeEvent(self, event):

        if (
            self.thread is not None
            and self.thread.isRunning()
        ):

            QMessageBox.warning(
                self,
                "Processing Active",
                "Please wait until inspection "
                "processing is completed."
            )

            event.ignore()

        else:

            event.accept()


# ============================================================
# 15. MAIN ENTRY POINT
# ============================================================

def packaged_self_test(log_path):
    """Verify the distributed binary and model assets without an installed Python environment."""
    global OUTPUTS_DIR, APP_DATA_DIR
    import tempfile
    import importlib.metadata
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    results = {"frozen": bool(getattr(sys, "frozen", False)), "models": [], "passed": False}
    log = Path(log_path)
    def checkpoint():
        log.write_text(json.dumps(results, indent=2), encoding="utf-8")
    checkpoint()
    try:
        torch.set_num_threads(2)
        application = QApplication.instance() or QApplication([])
        test_root = Path(tempfile.mkdtemp(prefix="inspectra_exe_test_"))
        OUTPUTS_DIR = test_root / "outputs"
        APP_DATA_DIR = test_root / "data"
        available = [m for m in discover_models() if m["exists"] and m["supported"]]
        for spec in available:
            model = load_model(spec)
            with torch.inference_mode():
                tensor = torch.zeros(1, 3 if spec["task"] == "defect" else 1, 64, 64, device=DEVICE)
                output = model(tensor)
                if hasattr(output, "logits"):
                    output = output.logits
                if not torch.isfinite(output).all():
                    raise RuntimeError("Non-finite output: " + spec["name"])
            results["models"].append(spec["name"])
            del model, output
            checkpoint()
        test_image = Image.fromarray(np.random.default_rng(42).integers(0,256,(96,96),dtype=np.uint8))
        results["brisque"] = brisque_score(test_image)
        results["ssim"] = image_quality_metrics(test_image,test_image)["ssim"]
        assert results["ssim"] == 1
        window = MainWindow()
        window.worker_id.setText("EXE-TEST")
        window.batch_number.setText("EXE-001")
        window.product_id.setText("TEST-PRODUCT")
        window.batch_capacity.setValue(1)
        window.restoration_combo.setCurrentIndex(0)
        window.sr_combo.setCurrentIndex(next(i for i in range(window.sr_combo.count())
            if window.sr_combo.itemData(i) and window.sr_combo.itemData(i).get("architecture") == "srcnn"))
        window.accept_image(test_image)
        errors = []
        QMessageBox.warning = lambda *args: errors.append(str(args[-1]))
        QMessageBox.critical = lambda *args: errors.append(str(args[-1]))
        window.run_inspection()
        deadline = time.monotonic() + 120
        while window.busy and time.monotonic() < deadline:
            application.processEvents()
            time.sleep(0.01)
        assert not window.busy and window.result is not None, "Inspection did not complete"
        assert not errors, errors
        reports = list((OUTPUTS_DIR).glob("*.pdf"))
        assert reports, "Batch PDF was not created"
        from pypdf import PdfReader
        assert len(PdfReader(str(reports[0])).pages) == 2
        results["pdf_pages"] = 2
        assert window.batch_number.text() == "EXE-002"
        results["batch_rollover"] = True
        window.inspector_decision.setCurrentText("Accepted")
        application.processEvents()
        assert not errors, errors
        assert all(p.is_file() and p.suffix.lower() == ".pdf" for p in OUTPUTS_DIR.iterdir()), "Output folder contains non-PDF files"
        results["pdf_only_outputs"] = True
        window.close()
        results["passed"] = True
        checkpoint()
        return 0
    except Exception:
        results["error"] = traceback.format_exc()
        checkpoint()
        return 1


def main():

    QApplication.setAttribute(
        Qt.ApplicationAttribute.AA_Use96Dpi
    )

    app = QApplication(
        sys.argv
    )

    # Embedded icon keeps the source application self-contained.
    import base64
    icon_pixmap = QPixmap()
    icon_pixmap.loadFromData(base64.b64decode('AAABAAcAEBAAAAAAIABxAwAAdgAAABgYAAAAACAAZAYAAOcDAAAgIAAAAAAgABMKAABLCgAAMDAAAAAAIACUEwAAXhQAAEBAAAAAACAA5B8AAPInAACAgAAAAAAgAB9iAADWRwAAAAAAAAAAIAAiNwEA9akAAIlQTkcNChoKAAAADUlIRFIAAAAQAAAAEAgGAAAAH/P/YQAAAzhJREFUeJw9k1toXFUUhr91LjNzJpNMnDFJY7QxtakEamslbSlRLKUoDQWpDxIsXiqKRREKglIvULxQihZF6EPpk/hQ0AcfFAUvrQhGahst9dYmbWJMpi0x5ySTuWTObW85k+r/tDesf+21/v/fAlpANLc/1UtrcS2mKWY+T1xrwKRLE2u6MPNp4nINYqXxyxP88cHfCVdWCp7YL07xMHa6nbSDXgqwb2oht64bQ6Ayfp2g3EDaHAh9iBqLetl9iYvHTwg9ezaKUzhPuhVMK6KyLB1776N793pS3jxWyiYuFLn68Rizn/yE5HOaOLZolLWuuRssTOtexACiQM8vWoXnduFs78M98iEtxS6qcwukrID8oScJ4oC5k6NiFFoDLUYKyx5KmIKI1rWG2P1d2MPrWX7lBA/se4TevSOUVm9gemIO78D75Ea2YhczqEaYrK7RWgwUYBhCpYa1826C787S0t3Nhb+qfPX8O2xcmOTtV0dk8cosamKazOa1UKkiCQewmiJqDSoizmWwq2Wmfptg19YBHnxmC/fftUYOfz2qw9WdKLeMbstCHK24k7wNMWiFmAbReIlwXR8t7hyffn+WLZv6OXjqnD7dV6RgC/XbbsG/UoKUjVbxfw0UxDG0taK//JHqqlvJ7B5iYexPHnr3Iy7kTXq/HSMc3EStHhOeOgdZB1Syu8JIBkguYgKhEB08TvnxETIvP0ZnTwftl+fxdmxncWgbuWIHdx54GLPi0hRAxSsraB2jfB/JORgzC6inj7BUCvEGtzF/z2Yqv1xFXjiKV/Px39hHfngQ5S1gpG2sRDxbYtKOSdVzkzA1J+HoSVQibi6DlUoRKYH9bzFzfg/O5AyprEFQr2Ph+zhtsT526FFK11ws2yZSinTWQcWKSrXGqpvzeG6Zy1P/0BuUGXh2J68f+4JLP0xLMsGZxnJdPvvmZzsM/OiOvm7p7enk198nm86Ul2pcGhfCSOmsk5bRqWt6thHYnuuBqc6sfKbijhcJ5E0My0ksJZuBWpgEDMwkrDeQKC8awqBOSr2Gd/q9pEEzloX+4YHlhvRjagl8cNIp4qZFN/D/UXS7o8avj39+UYP8C17aaHf60bb/AAAAAElFTkSuQmCCiVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAGK0lEQVR4nJ2WC2xbVx3Gf+c+/I4TP9o4TdykSUXWwkqitoIxSLWp3SambhpTGaia2k1bpYGAFtSB2MRDbOuQ0Cil0wRbRaVVSExTu26DrVO3NsDYilzWqh2lTVqSOu/EdmLHsa997z3oXnshIKRJHOnIsuX7/875vu///S/UluBjlnC2qO+P+/OSmu5zgCR2VwOx1j3C49uC5gmhaKBoAl0HoSGNKlSqIBTw6Ai/XnuwWgXbllgW2GZBmKW37InUfqb75+tn+qFC23tNwps8STDei6I4hUH1IDxe7KIBlTLhZIzQyjiKgPl0htnrWfD6UYK+Goi0wXbAgOJESpZGtzDck69do+Mrz+Jf/jWwSiia7gAIXcfOzRNd30Hn3nsIfDKBbZVRhMCr+ylfGOPK00eZPD+CEgkjTQukcwurCoqfhakD8urhbwkaNzWJaNsl9OByFE2iqEJoOnZ2lsSOW1n24/sonHgH7di76NNFpKJgNDeifvkLJG75PMO7D5E+lkKJRZGmCZZpIy2FSmFczl5Zo+HxxpGyASkVbGmhCuxMlujWDfie2M74nicInRig/Om1mN3tLIxlCFwaRew+xOX7LpDc/3WKQ+Nkz4+iNDhlbIG0FSCMtiym/Lfu0jDQwiqBpx8iv/8QvlcvEtq1lUd/+Qitd97OXO/NTLR3ct2nEjrcz9grb7D8qQdQKkWkbdbqSLlYsg5QN54jcD6Pf+vnKJpFfC+8gbl5Izu33cHRo1c4c/g06un30KdyNCfj5DwS8dxxzFVxQj0dyMI8wnGBC1L70Ba/yfo2DcTNN2KnPqQyVySwqo2Dz7zO0DsplKDGQi7NiV/vFfmqwbZv/0yG0tMY6XG867vhr9egIbSEEVUqS8ipAQiwGoMwncPj07nen6IvUuHA924XTF7mzecfFclEhEf2PS/VG1ZhGVXsfBEZCTl9gPzo6O65LadrXM7qv0uchrFm8ljxGLJioJQK/O3MGe7tu5H0X14Q8bCPjdsfkzPdHeidCaRlYkWaqEzl3CYU8n9qsIQmVcc8fRZrfQ922EuguMDFsQK37PyBPPmnD2Tfd34hF267ieaHNhN8/+9Y3Ukqra2U3j0Hfh/Stuq1nI4r1wEcaznLthDhMNbrf8QsWhR37yB0bRB/yM+VaIIdb6coP/xFWu7uQX/mONq5Aco/3UP+5dNUz11FaQi6DCyysURkWbuWBF2FOTC/uQ/50j7XVU37X6RhchyrqxXx4jD2h2kcQxZ/+yRlAsTO/oP5thgLs3Mo4TDSthddJFh222rhazyHFggiFOd+Ak2H7AxsWoty4Lto2Qy+106iDw4hVZ1K7xqMezZjXbyOdeIsyccfZPXwIKn7n6QwU5UiFFQxigVZnllXB2hwARShWTayFsgOSC4DMT/cvxX6NkAkXKNxdAp5/BQceQ0e3ob8/i6SHuiYHCd17+OynLdUqBZkObNOAwOkYzELq2qiBXy1SHaCKxarRfTPfwcHX4KA1726mS2genyIti448gf3SOm9u5gTISoqrvuEVyzRwDZRVJPkyihDl0ecBsHNZVWrcal7nMyHuYrb7eGVLeRz85DLguqFX71MYGQEUSgSzhcwI0HmZ/N4HQCPlKJiWni8VX70jTvF4PAk05mCi1sqG6xIxBkYGqc92czYRI6grrCxt4vpbIGr/5ygORHl+JspuioLPLBzizh19hIDwxne+v0ITSv8QvMapWzF5y9ZpuJfHm8k4FWpVKp8YnVSeHSd8ckMX7pjA+GQX87l58UHFwaltAxu6u0U+dms3PXVW1GFTTQSpq2zhbsiAfmTA6/YYC4k/NFcbeDENx8R3tD2nhsSpdm5Bd2Zc80hv1i7rkt2drSIg88ek+3d7SI9OiU721tEsVjC69PlwLVx1q3pEOnRadfks3MFej+1qvr++bQ/n5n6jci8/WBtJkc/u0Jo4X5pal0u947Ijrj+OveOVJaNO58NR4dadOH1ODyCx9FK1nKhXAGPNQBGH9P9U/8e+sHPNOsNTY/ZQtkkEQFFEcKp6eaLO72FW3PpG4UTYQ6WXWtSp1mLquCU184/NT/x5+n60F8cCB8llIB2xwD/xxo2/rMO8l/INsIgs7KKpAAAAABJRU5ErkJggolQTkcNChoKAAAADUlIRFIAAAAgAAAAIAgGAAAAc3p69AAACdpJREFUeJy9l3mQFNUdxz/vdffMzrGzyy7LAAvLsaxcQiRRBBQJqKjEKyhBRaJiVIxXjOWRhGhJCkqxMFKJSoxUJJQYDgmpipSI3JEoBFiE5VzkEFh2Z4/Z3Zndmenjpbpn9sD/k1fV86Zfd7/f9/1+39/3/R5c0mZo/M/bpTZE199XJLzqUDIpTNFlU9D8FUjdAImUBo6UoEmy9xL0XI8Ex8FxHK/37snd2x1jdgY7dZz6I1up3ZTstEUnABfVGpuB980UeT1exx8egOYje+kgdJAaaBpC03BsBakMWDYIAboOfgPN0FGO412eYeX2Njimh4FU6ynVXvc81cs/7rApOo33/8lPRbB4ObofhDTRNIXQOgEIw8BJm9CawBcxKCgrJhQtQApor2+l6UwDqXgG8vORfh/KskCp3OUCcG1hYLWj2uru45sPP2LGDC3rgdKb+wl/yTGMUB5Cs9F0d6ngulhqCF3HibcSjIYZPGsifadfiy8aRuUcrrtOjiW4uO4LjqzYQWsshSyIoCwz66EOT9iWjbI1MsmkStcP5ezfL7rfIrXQQ0oLBHHsNJowvJe9UCrve6e2gegtYxi0aA5WQR4n9uyhfeUeON/oGZD9SwhNHkfvx25i/N3XcPiXf+bcjmPIoiKU5WQD7c7pGndMEy0vX0r/bAcWeQAU2oScr1yz2aayEXBiMXrdN5FebzzMmdOnSD66iNJvTfqEipH+PGzHIbPvJE0rv+TgyB6ULJ5H+fJnsR9cTM2WI1kQLhnJcUJ5rFVKuDbx1ulaC6KU8HyaI5C38sZGIuPLKXrjYc7urSQ9/RcMixUR71vOgdGD2De6jL19I1T7TfTCXpQdF8TueJxTNTUM+P1jhPoEcBItWeNOjg8d2adEuBuA7zaFMjNI3aZw/oM0NMXhmQUMCA+jso+PCc/dynsL5nLZtFsxxk1FXTGJaqeNGruefpko8adepalHmNJnp0Nrszeft3q394B0WZI5QN30QHj5rppbCE4ejTWiP8k/LKM0HuBggcMj8x/lyeEjWLr+EN9sPIC+9yj+8w0o0ZNYSwt1qoUeVY3E/rGBvLsmESgrRCUTCDdd3Lk9S6rzR3YGvNvqPbR2Gt+PxpNMZQhu3kdc0yidejVDzQB3LtzAzo93kdj7NckTh4jt38mca/px94+nEkvUY+h5qI83kzYk+VcPg0Qia9ijWTc7dIXAzZUuDKaFDOhweQXmidP46xK0hCS2CLFwyXqat2xDxM5ipBuxzlUzd+YE3n95jpgwvB8qEialOehnYqQsG9+ocrDNnCJ2s3FpCDowqFycHPBr2JEwNDRBxkLz65zbtofeyRpW/G4GRUYL6WMHeOqBG3j3N4+I99Z+qp578wPEsMGkNYFsS2O1pVAF+R4JlZcBHQBy4SCrIR5JO43nXlKmhd2WRgWDKKmQ7WlESwNmxub+W8aJQdGIWrdpF4tffFgsXb2BxxcsQxtahj2kN+I/p1D5Osrvx0q2e9MK4SZZdw9cEgLV1bmr1yUkU1gnzsDgQWQK8tCbW1CaoKrqFOPvelqNHVUhFr/4M/HO3z5Rj7++XInhFRi3jccX0DAaE5hlJSi/Tvvhb3Jq2GEjpwcozwWyC1DOA56HBMoWWJt2IXqGSV09kry2NHqsFv+I0Xz51WFumjNPzfvjSvXEklVw1fcIzbyW6PgK8jbs9bTfunUyGRPad+6HQADlpZ+3AXaQsXsWON0IorxtVOQX4HzyBdbZOPbTj9AWUURb05gNF5BTrmdrWmPBvpMYc26n6LEb6TOiD4nffkDkcD2JK/vDrDtIfLgRu/oiIhTMSXG3CHg2yQHoXH2HmxTk+VGxNqz5f8IaWkZywXOkVSt9axoJHa3CH/HjGxlFkoD122h6YAmB7SdJVATJvLuAzMl67Lc+QrmS7qVghxx32hFdJOwE0TGc9QLFJah1O7EH90P9ehaJ4hKs15YQqfoWueEC6lMdbHcrtLGLQzTfeQWZhb/CrmvFWvABY/76Kom3V3Fi5XZkn77dJMDV+exfvYt9uceuq4TM7WAKiqOoxSux6xpRLz+FvW4Z7dv3oO+vRK9pAJ+ONagf5oRxqFEDsbccRL29BvngDGqHDGDMuy+g94hwZOk/kT2jKE8N3fndqoYcgE5muh7QuqWj+6KEot6wfCPOrkqcWbchpk7CGnuVW15kP08AVcdRc9+ENZ/B4DIYPpCLDRl2YzHmtbnISJCqt9YjC3pkNcHJ+lp4P31v24ERmehqoNR1zXFX7qF0KaJy+4MGyQQkW6AwCL2LoCCcLctiTYjaOCqjoGdPaG6CiijivYWoUJieIYNe8RhHJ8/FMfJthGaQaflcnVt7o+xKgqwX7NaE1+uaROKgSeHVopo75pZbvUuRej56TRvi63Pox2OIZoUT7IHWtxRN6mjFUfQTdfDoS4i6GLF4gqNvfoTdnETZlksatzjpxgFlKW/ANrnqB4PZd+g0Zm0thEM5cVKga9lCNBj0PGNn3Pov5JVq5PspKgjTePZ8toxzEbshPFYD9zxJxehyTu6oZPDIQcRb0zQ1pzuFSO+kpbJRZhuvv3SPWL7qMxUtLmb3oWoCeX5Kigs4VxNjwthRbPh8tyer14y9nJVrNzP75zdj+AU3/fD7/GXFRoxQkKbmVkp7l7Bs1RakmeKrpS8y7aFXeODu68XWfx9l9ap/YRT6MbsAOOlsDaaob4gz594buVjXwJSJI4TfZ5BOp5XP5xOVh06o9xfNFYUFYS5cqFVPzL5BbNq2W1V+fZzmeBMzp08QZ7+tVW2JEPffe7s4WHVU9SwuAJ8U9995nZJuKeYJkutxPd0lRLZViXI8wW5Pp4nVx8WHazdz7kK9OlZ9VkUiEWEYBhVDBorde4/w5d4jNMYTrFj9qRpQVipKehV7YUmlTFau28qg8nLO18QIB3xMm3Ili99Zq6K9isTQIf1FvLnVVSaBsvZ1ZgFF144QRuggQnMiIZ/7XJp2loCFkRDJWAMvvTBb7DlQrVav2YYeDjKgLMrJI6cpHzqAeEsCKQV+n4+LsWYKIwH8Pp3GeJJgwE/D6QsEehW6PHFiTW2YpqmU0zaC2PbqzoOJLJ70vPLlL1K2KxDCROpZIrmKKMAfNEi3ZRC+QLawTaeQwQBOeyqborlqR/iM7HnA/c4lruOg+31Y6XZ3zIehg5l4lvotb7m2Lz2aFV33jDQC89H8EU+QPBF3lUvgaoN0j2cdJYU7dunhMlvsetrfbU/xMtx2n4CTimO1z3Ni297udjQj13IHxqKx/ZChaUhtiFBZknpSJF000t1F3ZRxUXmmsjA8OqmugkMJ4Q27CIRQQmVwnBMkmzbQvv9C98Ppd9r//3j+X7QbrUtmeaJSAAAAAElFTkSuQmCCiVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAATW0lEQVR4nM2ad3Qc1b3HP3dmm7S7klbdtuQmV9nGBTA2zeCAIXAglEcPhBziJO89sHnBgcB54JdgDiXmkRdaCCWYQOi9mA4OzQbbYOPebdmS1XdX23dm7jt3ZlZaGSfn5K/kHq1mdmZX8/v+6vf3uxL83XWBzpxmQe0kyT9jdWwUrNgk4XnzH/zmYg0E/zpLCkem7y7Pdy9doMOvHcQjLjtJlJSditDGoXmDCAHC/gWaR6IpnEIg9IGvq9tIELpEUHgj7IMF9IvhKsgyQUrn5ZxL+xwrgZXfKrPx99gu/ur8HSXbYGscomb3A0PPmS2C1XfhCx2PtwSEBrrHOWqFo+7Ipc6VVA44hHpvXxZFgBz5pWXZgipkzo90rxdAqPsWWOrccM7zKcgmVsi+A4toeXH1oSCKALg3hp13uSitfhRf0AuWgaZbCF30C1142QK7R/WHbGuAlctDJg/5PCiB1VL3vD4IeNH8Ptsw0lJA5ID2i89tEAqUKbGkhhAesn0Z2df2I1qef64YhAvAvTDkB6eJYNXb6D4LzWOi6botpC2c+7KFV1p3AWia/dbqS0Amja+ylKrRdZQPr6a0Kmg/IBdNE2vppmvXQVJdKfD50MJljiWUtm03dwW3z00XhFU4N7AMD0YGmWiby/5XVhRkdu0LRE4pE+U1G/CFh4FmoCvhXYFtny9yGReA8OpYqQykktTMGMWYc2dSM3s8nmGV4PPaLq8XXoaBPNhL96odbHnuc1pW74JgGK2kFGmaA1YohE2/O7lgTEOdeMnGd8vEgSm0v5uyZWDOYg8rfm1oDef9UgYq7wKZQ/d4HEFdbQ96OX4tdA2rO0qwoYLmX55L5PQZ5Pxesir8AMO0MJP2M/AEQ3g0J2MElK7yBj1vrmblnS+R7EyiRSqQhop1N+H0AygcDVAgLdNAaD6R6rzG2vfsfUp2FV1Oahl+wVp84SOQKL/XHY0fovWCv+sCq7ObmtOmM/6OK0nXlpNSgnf30vfmWyTeX4GxowWZyNjf08pL8U8YRfj7pxA+bR56aYAIUHagi1VXP0T72t1oVVVIU0F3n2dnJfVeWcENaNMyEdJDNrpK7nl6tkqtjkSVcxpEaMh2vIEAQjftFKKyjfL9gvvYvulgsjq6qL34BBrvnk9U12zlxB94kMTvH6ciH2LYhKOpbmzCHwxhGJJkLEbPvm207V5LqkpQc/MvCJ93Nl5gaDLDmsuX0rF6F1okgrScbOY8zxrsSpaKfEMnn47Lrj1jSHza6Xyy+vTpIhhZi8fvCG9r3g3egkZszYPV1UNk3jSG/2kRXUpB8Shdl/+Uki92M+ns+VTVDSF5cA+dvW0ks0lMw6DEW0JFVQOh2hF079rKlg+fxD//LGqW3m7Hx9CeGKvOvJlEZwoRDLtuVKhhbiDbdcR1JyMrZLqnmbY3trgA5h0pglWr0X0D2neFdrSv3EdCOoWvOsDY5XfQEynDSMTpOe186jtKGH/+fLrWf866vlaYNhlGj3DCtzMGe/bArm0EY1HGTJhN2dCxbHj2d+QvOo76P9xDEPB+9DVrLrsTKipBeAeeX2wF2xIqHnKaTLZPpv2djU4l9h5a24qoj0pzwtGATMSpXfpj4tVltjKi8xdQ1+Zl9FmXsuGtZbQfN51J/76E+eMakTmLW79oQfQmYfhE0iOnkNz6DevWfczwtl00X3AN3z51N72T/4i8+qdUnTydujNncvD1r9Cqa5yiV/DbQoKyxSpYx+fo1/6ds5yi77IEx23cDFsopPE4JUc2oZ0xmwzI5NPPEVi+huGnXsjm956h/dx5LFx6I19Mb+TITIZvNrdjtfQgDvSitcYIpgSR8DD8U2axr2svLes+ounUH5K49T6yO3cTA+qvOh2hW0gj5+aygiwukGLFkisC4PM7OczOxQXgA+fCdZ/whSeRKfFgJjIis/QBGmeeSdfGlXTOmMCVC65gsc9i8cr9nPbcBp5YvgFj014yW3eT2d1Crm0/+e5esh0xxKiJ7DuwhVwuSXVoFLF7H7LTr5g+jrKpo2xliWKBizVfkMsWqgBAlf0C2kOJs7SQ2QxaZSnacUdgAJmP/0rwQIpAeYTd8VaGXnQ+l1gGC9/Zzj0vriO/dhv6lh0kd+wguXcnmY7dJFp3Esh3cfbscQRUIA5rpG3bKiKjJ2O88xlGV4yc30PFiVNsZdkWKKYYg6xhA7HPBiiqwwCLELsBpHw/ncE3uh5raC0qGvIffky4fBixtt0YTSNpjlTy/Oc7eH75N5Tu3YfZsh8zHoN0HN1Kke9tJyySvHTnfPHqPdeKn555LPhLSZLDQuKPmeTWrbedwj91DKjyZGccF0QhrQ4Ie4gFvhPARahtMpNFDK/DLPE6rrdtN/5wJfF4J9QPZcu6PfKNd9eS2bie1M4tiHwCkelFNxIYvR1UBiyWP/grccKMZpLpDN9s2oaoLCdfHiabjOH3hslt32FbVwypQivxIo384Ircr9nBYA7bJAwEjBvMqlJVBLE8YGVMRCyB8PrJm1kwBe0rV4uujd/yw1PGsejKEzB7WtBzMYxYJ1UlFsv/cKOYdcQ4EskU5y1cIld8uQG9rgarsoJ8JoXu8UNv1La/LC1B83mK+FGxSxSL7fsbDY1tMorQO+fqsl3pXVDSNJxUl8pgRPsokXEeuO5CEQ4FaawqlQt/9Xuqhlbx9h9v4ajJY4knkpy7YIn88KvN+IbVYdSWQ4cfIbP9lNvmUAWafdiGsD8A+rPQYAC20osDWYFQVEIRtxhSWdWnISvLsKJpvKo0x6Lomka6q5PF9z8t//eGn4gFl54hpJGTU8ePEkr4WF+Cc65ZIj9euw1/0yjyQYGc2AhffYnHU0Iqn4baGvuJRl8KK51DlJQMdpd+tlpwkMMFsR0XhzYX0ubv1t42zEQWywta8wTSsS4CpWXQ1oLlK8Xj0bjnvqflLb9bZj9l4RXniJOOmUo0nuCsq2+VH3+zE/+4MeSVj09rwjeiGn1PK57SMFmZxDOx2U4Q2X1tyFTWoTH284uV6SSVYlxFMaC4RnGxGGg0RMCPufMA1v42+yOe780lmWonUF6Dt20f0uNHlIYpKS/n1ruXySX3P2n/oVi8T561YIn8ZMNe/BPHkh/XgDU6Qt3FJ0uxZj1lcYmZy5JvrEA0T7IBJFducKt/sbsUyWW3m4UiV+xCbk87YI0CF1FUw4fs7sT8+Cu0CSPxHH0MuakjyLZ3Ul01jLZvP8eaOQ9t45fCV1HBzfc9y+bWbra2drBm+0F8Rx5BblQNsjbAxLkzsCxNHPjtMhonnEj3t5/h+c1PMMu8mAfjpN//CkJBF0SRd/RjcL3iO2m0uKUrRu/GhAiGMJ9+F/oy5DUd702/pHvfWkJDxlLeF8PY8AX5o+dgjR6Lp24If/lgLWt6snhnTceYMITSyUOYdv7J+HUf28++hsbq8eS6O0mNK0e/5BJUeKVfX4G5vRVRGhwQdlAhK45XJwYcCygS3t+PFpnLFt5FHwojN+7DePI1tJ9fiDjmWLTrr6LjtsepOf5MtC0r6f3oZZj7fcTxR+Gr8COrShBDyqhoqKDcr9P20gd0LnmEhoomPP4wbXs/wffEM2RLA8h93WTuWoYWKXdEsZ9dmMMcQjCLzh0AwjwE4qE+6LZ6lVVYS59EHD9D5JrGSP+CReRjMTruf4maGd8jnO4l+sYLpOqryY0fCUOrEB5B/GAHia93EOo2GDNmNpmOVto6P8PzzEPkxk2wGTKLH8eTMTAyFlog5KRoe950mLikP4u6XlZ98pGipHo1mte0+8XiyUNxo60O8R5oKBf6i3dLGapSCQr9uSeQS35PWFYRrB9pd09mJmG3sEgN3RvAW1qGYRrE9n9Lako9+t13kBs9VrXp5H92G6GqoRz1i3PY9qPFtK7ajl5fj2UObmWdRKM6s6wm0912P+DGgHcwyv54cNNWAYQ6lEdgT7c0L74BOlvJWZD9tyvg7Zfpu+xY2tlJT9dmUn0d5LNJcrkEicQB2mPraR8SJfPbaxHPPk969FjMXB7zqsUQzWD87DyZq43Ik169ixFnz8ZsbUUIZRq3Gyt2byXbdy1QNWCBfu0X9QX9bqUKm4SeDqgMIG67GnHqHNvDPKoL7Umg7dqC2LsHGY0hfB6oq8NqGoccPhqz1FEiW/dgLboLIo2Iay7DN6qKMp+HUWEvI4Ia66+/ny0PvoaorgHNNxCPSqlmVpPx3sn0FjqyQcHrTgYGNRKF5V5XKS5SDck+5PzbkKe/h7jyXIzJk6AihJh1FOLYo/q/YhtUCa2mLN/uxnrxPfjLcmjvgivGw6Th5BJpGc0YYpeyaE5nyl3/ib+ukvW3PwXllSALg4UCufsOlSga8dkpVn2w0NQXxUF/QRFQEoZAKby7Dvn+Wpg0EjFzCnLiaGRtNTIQcIhgTy9iVwtyzSbk6s0QTUEkAg0j4elXkQE1fF4gjGhKxrKmUJwrb0hGXncJ+z9dT/cnW5yJhbpR0KcKvkEACulSyak0rCYuCq1thaLp86As5U4squtUow1bO5Dr33TAe9UkS3d5l2GPOfH6IRiE+ojbekioa4BHX3H4//9cK4yeFPFcTpppTRgVXoxwEMz8AI0oNDr5vDwMGx1AqLvRbxaGS/0k7xCaqK7bKc8DZeUgKopH5c79sJNNlE4sBcSepRW1ikMa4U+v2BRa3HodVh6RrYCWT7eS+vBLPBVhLDv7FLs5h1rA7YDU0sCIRu2Ril4WcviQyge2oJrNPqU0XfLqzE6laj1NaU+pNSGw0JBCbSGo4bKGYRiY2RxawO/kCKm6XsudI2hY9Q3IJ99Edvcizz2VzOY98JfXEXmJkU6Cmmr7/UUjluJKrAq5PQVWD1MjhyRnnnEM6XSGD99bA4ra5k2E4iiK8sZS4PGAV03glcOa4Peil5RgplNYmQyEStG9XsxkQkUlgUiQ4U1VbNt+wHFJZR1dV0Nb28UIlaDV1CPfXknph59z4sxmPkqnyXk0Tps7lU07Wtm/vwfh89nKKiy3qR/oiVVbIeO9/GbRheLaq75PQ4XGxDFVzJ7VhMzFkNkYJ82ZyKTmIdRGdJqGh5g9axT1VYqM7WNorZ9TT5mCjxRm+wHGN0WYNb2B6WMjrH3zdmZNG8qIep99fUi1lykT65k7txmvzKAZOdu7zrloHsvfuo95c6chezt5Y9l/izNOOgJp9x6FROOQuUOykDKNHcXE4im6unvEw4/+tzxy6gSCpQFx78MvyNqaSnH6yUfz1jufyINdUXH9wh/Jzq4eUpk8C677LUtvXyiymTStHafLx554ncfuv4lsNstTzy1XQ29xw8/PlmOaRoimUY3ynvufZMYR4xk7dgSbd7Ry9mW3gMhz+blzxKbN27nq4tN4/YW3ZKwvRT6bVWNFKS3T4W2DGhppRe26L+WgLsIy8rIsHBS/ufNhblh8rzzzlFmi42CbPNjZK3ujUSJlQb7dtFMMG3Y0lmmKPz+yWJSFSvls5To5YmiVeOz+X4nHnnxV1NU189JrHxCL9XHRj2+kMlLONdcvZdmfX2V/aztfr9vCcUc3CzXZHzu+gROPncbnK79h3txjaBrfSF9f0t16s+wyimVloC8+ACAebUEaLW7VsRvgSEWIsnCISHmItCJZJkQqwsSifbS2tYtF/zVfjB45hIZhdfzpiXupq6nkqWeWy0h5mKH11Xyx8mv51DNvcdF583js8Qc49eRZ1NVEWLTgCsoV8M27uOjSc7j0wjPwejT7OWp8c9OCS0mlMlSWBWQul+PGhZeLYKmf0tKAM51W2rWMXfSublXZwwNzPLAih1XzOrBAWqZFSan28BNvya6ubj75aotcv3W/3eHd89DLcn9rnOZ4lh9ecaOcOGUM06dOFsIS4tqbH5SPP/o8G3cclHOOn8YHq7bz6qsfyHjaEqNHDePlh18jkZEyUhHh9v97WkaTFo88sZzaqoiIxvpYte4xqYdC7GppF+dduVh+8uabzDv/B3Lm9PHi3kde4cuvt0FJqWVZ0iNl7hWHJM1REyTbCpLymSOFr3wDutdnTx3ifU7iVtVPVVTlVpncQKyko1x383+Io6eN4+Lzr5aUNaCFQ1jRmDPpU5VSZS31XlVjVcByObc46s49KaGvz7mmexDlZchYHwR8eEIhjFgMsmqTRECgxBIlAYGZS8h4RzPpr9uKWi9nw0yLHPtz6at4UG0z6R6vJoVK6crl3O1gBQxp53mVyiwzb58L3YsUHkxL2jWiwLssy0LXHS9VhrV3Mt1yX/AGXd23t14VThNd020vttQGaYFLqr1Ky5SWlF6ZjV1JzyfLijf53OVeiJzwa3yhW1xsOXdDTEk/uPoOmrMWMdf+4VlxQ1TcURWxW7UGtYvFoxO7YBUaYJ89o80lbqJnxe2H2WYtLPdG5LhLhafkDnRf48CupLv6gRRT7UO6N/u2m9C+s4ofKQcGaf09cOFWocW16fNuaaSup+fTF/7ORvchIBgeITLyAuHxnAJilLPBWGyFos0D54JLcgrnNklyIqy/cNrXXSpbrPXDjD4labB2SDP3Pj2bnoPuvsP9q8HfWOqD/2rrgn9YJuGkWPuL/4R/XVHWVM9WMvzt5/8/khNz5J1n5RcAAAAASUVORK5CYIKJUE5HDQoaCgAAAA1JSERSAAAAQAAAAEAIBgAAAKppcd4AAB+rSURBVHiczZsHlF1V9f8/575e5s1MpqZNKgmkkkCoaUIAAwQQCEgRKQoiAaWIjSKiYFARFBAIIkVRjBQDgQCCCYEkBEggIQLpbUqmz5t59b57z3+dW97cmSSW/1q/tTxZk/tuOeeevc9u57v3Ffz3TfC/3eT/ATG3a8xCYzYmd9xh8r/cbr9dYzkaKzDh389V/Nv78+drLF5seK75YFwEDAE+aR+Ltxzue6/9u9Z/jP+muX3d9/4zDYpwp82f72PxYvNfSYU4+ODzfeAQXnvqYVqs5mwZCE9HaCPQ/CVoPruv0JyRNBA++6gJ57ozvOj3KqHue1/tPCCl3H9WxXtY961H1G8TTIdW07DvYSQxjR2ikHnbTDY9T8OLm/ej5T9jwHy7Q/nxdaJ89J0Eo+cTTITQNPvFms/+U0SoI4p4h2iLMIcJ1vDiAAR7GOGZhlDnDoGy39TUmVTvthnh/KlzhwkWQwz7vaaEXDJDrusZ2fbZrbSubDwYE8T+s5rlhxUFak87VcRrfk8oUW2xWvgK1ow1TSA0YTNA61151RSDLIK1fkS6kmB3tS65BLjMsYjXkKZpL6Z61mGQLRhWJ3vVLTnxEF5kiGH/kKbENAPWy7Kd9TK5+xLqX3rrQEwQB1z52rnniHjtX/BHFHV5NL8PTc3I5yHSu/run7quGOMdWtpEIzFyWcjpkNcd4l3tlPZY/gCEgmjhkD2OIsedZlH03WEdRgh39Qs2U9Tqq6PimlFQxAbJ9+gyVX8Ge19a1p8Jopf42zXLalaeOEWUDFpNIBIAYeAL+IpEWwSL/ZlQ1HeHAS7hfg0KecxkNxQKBKtKqBxVS8WoWhJDyomURKwhCqkcPfu6aN3SSPOWRlLN3dY4oiSGCISQys7ZYuMO7TDDPVdHo1cabJtgM6KgGwj85FM9sn3PNNpe2wy3C9dD+PsxIiAiA35HIBLCNHV8fp/NVUe8LX1TOuYaP+n8diairknTWnFkAbM5CSEfQ2eM5ZB5R1J91CH4B5ZbXXR7ysqlWJMIACE1RmsXzR9u57MlH7B5+UbMThOtvBQp1VOOanl131IdD0Os60UpUPd9GLpOIFoiEpWPyzZmHEAFZtl6X3P6ZaKk+nGkVGLv79Vz18C5Rs6VBo+OO/oqfAIzlYZshiFzj2DcVadQMnk4PUC3XqBgSKQvgAjY5LjrauommpISv6AkEKAKyH+yi/cffJUty9ZDPI4WCttq7rUx7ggu4a49sBbOAKPgMkzpSFB2N55Pw4t/cWkWDkstCsTQc98nVDIFKQuW3vfRa4cBRdH3ujlbR9VjZnsn4UFlHP6jC6k4ZSqtQDqVRYuFCTo99OYW9MYm9G4l6gJ/WSmhgQMJDii37ufUvDMZBsQjDAWalrzH8tv/SD6ZRysrRRoeCZReJnjshHQIt4ymxQwDIfxkO96Vu/48w1V5UTQKZSdMEqXV6/GHnMEdgr06XrTYnnNHJC3iW9oonzWRyfdfRaa6lNZkBn8iYhGe2bCRjsUv0vPWSvSte5DdGYTlBAQoW1EaJXzYKBJzT6Ts3LOI1Q0lr5uY+Tx1sTDhbY28eeVv6NrahDag3LYLfdys5ui+6x0UAzzxguKaNAV6Rpft9eNI/mObYoIoin/1aVeKeOUjSvLQfP6iO/O6O9VcQ1g0SMrKS8zmFirOPJaxDy2gHY1MOkMwESG76VPa77iLzCsriMgSKkZOoGr0BBIDhxCMlWAaJplkkmTjLtq2baJ9z6fkozqlF59D7W3fx19eRi6ZpjIRpbqli7cu+BldW5rQysosevoshjc+oD/xhi0FyIBMt15Aw5I/K9r9zJoNK1aA3zfKdjdOZ1e3LUPiWlplAB1DqAyfS3xLO2WnTGX0w9dRn1N6rhNKRGi5/0G6fngXpeHBjJ5zJUMnH0NiQBlmuotseyvZVCfSMIlFYpRPPp7hR59ET2c7DRvfZc+il9j8t1cZ9PgDlM2eSVMyTb6qlOlP3Mib824j15VERGNI6SyQtfCeeKEYZ7hu1KLH0lPNFxhpPT5rtjLAy201ljLWh/Civ3GItqRAnbvW3yZediWJjB3I6IcW0KQb5AsGvpIQTVdeS37Rnxlx3DmMnD6PcChEyyfvsWHzR7TqGdLhIAT8kM8j0hlCep4B8TIGjT2KusmzGTBiItvffo49c+aTe/LXVFx0Pq1daQJDKjn6vqtYceHdiJDy1EpY/c6ieKbtkdBeuhxyJHHXhPq9LqHYwZICRbB70d1f2MGJzWEDmc+DKDDyV1fTGY+Q6czgK4vQcuV1yEWLGXPa1QyZcDzt2zewdfUyOocPhflnwGFjIBBCy+ShoxuzsZVsfT0N2z6lYc3LDFj/JocccyZj51xMsLSS3RcvgEiU8rPn0ZhMM3zWZIZ/9WR2PvoKWnUV0rLhamGUerqG0LP/sS71CaTEARggvLsVj547UuC6FkdNhE+JfgfVC87EmDqarrY0wYoonfc+gLnoT4w5/WpqRk5h6/LF7Gjajvz2N6n84izmloaZHvHz+u4kr2zrIJYIYxJBD5eQjZZRGDSC9s8+4sNlj3Ho0aczbMqJSD3H3q9cQ/CwsYTHjKE+pzPshnNoWLoavTsLoWhxXh4R8LjEvl7fY8HwOFSH0KKv9zLCK1s2I2Q6hW9gGWVXn0lnuoAvESX78Sekv/dThhxzNgOGjmfr28+zvbsJ+dv7OP/iuayoi/PL2hhHxyNETB9mUkdLFfDnBBEZoSxcQdQfQ6sbjTH2cDatfpG27RsZdPgJlMfraP7G9Si7l8sbZCtLqbvkJGRnJ0K5E+UBrAXqT/CBmeA2bT8JKEZS/ToUjaKK9EAmuyk9Zya52jLyGUOqUK7ru7dRlqijetwxNG1axZ7Oevj5Xdw0dQRPlIeIhCP8dnMHl7+6mVfW7iTa2oVR305hTzP67gaM1nbCMkzMDKGVlMHoSWz54BVyHS0MOu40zLfXkXz2r/hLwrTnClTOn42vMo6ZzTiRn3furpvuG0a7jtvDgNluh14goQ984BnQvaLnEdEAJefMIp0DrTQkMu+shjfWUDttDkYqyZ5/rkZ+82rOGTeU75SEWdVV4OIlm7j99S18vLGe3PYmcjsayOyutwIjo7MLvaOVQkczeiqDSGfRKirQB1Sx++M3icTKGTBiCl2/ehizYJLNmZh11VQcNx66k7YU7Kf7HpVw7ID658VMNNcLFC/ut2/36pN6xkSmUgQnDMc8dJilg8oIZx5/mnhJLfGqoez75F2ykyZRc9ThXOaTrGrJcPXi9axaV098VxPa5ztIb9tJZvceMg17SDXuJNO2G5luo6t5L7Vxk/KQhtHZgW/EIbR3NtHduI0BY6ZgfryZ7IfrEbEgat0HnDAF9LytApYaeACTIjP609Qr+Np+xBa55jbHuDh/FqczaSJHHUY+qGFqPgpdaQor1lA6fBx6qpvW1j0wfSbH+AUNnXkWvvQxWzY1EN1TT8/WnRS6kpDJQD4DBXVMIfLd9OzawpdPnMiGP90hVj1+KyMqE0ifBrWDaN/9KaGSCkK+ErKvv4XPB8qJhKeOQZREkIVCv71Av3DZiWLt3XMfCSg2D5u8207nVhGBsU99E0ZQUOFBOEBh82a0xjZClYPobtpJpiSGVlNLZl+Sv63bwcZ1W4i3NZPb14SmiM6mINcD+W7QU/jRye9r4qJzZok//eybIhwMMHrYIDFl3AjMXBbfsOF0dzbZQVPFYLLvr7Omkc/piEFVBKsStku25vwvjJ50sSZ3+0qvG7ToKiIv7hVXIpwgQ+mQiiYDPuSQakwVBoTA2L4Df0EQiJbSU78NKivx5Q02bNgljcY2YTY3k9rXhFo2fyyOoURW6ghDxy8K5Jsa+cp5J/DU3deRzeYIh0O8sXodr6/8AF91DbKslLw00NNJQmVVdO7Zg1Q2QArMeJRAZSm5hiRCqacbqPWnxY1orZ2sFRZarSgB9r3+NrKfvVR/DgMoiVr7DRV7mC2t+LQgQvOh51IQjWM0tdKxYaPo3rUDLd3GpWdMYfqkgRRa9uA31eqn8Mss+cZ6Lj3/REW8yLjEr/pQnnXtT2SPep+K9iorMQN+9EwPvnAUutPIdNZaS8Ov4SuJ2tveA0qAGwT9RzZANdEvhLROekEG9Vq15/f7MFyNsPTP1i3T0FE3ZCqDkWwlu28v379oJr+/9TKx/KEbxbknH47eVE9Y08k37uWyL8/h93ddaxEfCYdYtvJ9ecY1d5IJhPHHI5jxMLKipNcXWVCZwg2VBNigig3AeIgshvPONRfA7RPTOLfYrx0ghPTqv+UGC5hpJf/2rlOUlGCaBQtz8KmNZDplxfiakYd8moDjYXWjwLP3XMdZXzyazOefc8XFc3n8ruuKxC9dvkaeee1PyIVj+EoTmNEgYnAVlETQdAVRBDDyWQiHEMGQhX0qO6RnlC924fkD0eFukZ2F9BhB/8FoPyhirsxvNo/Z2gHjsO3A4EEYmoGRzxAIx6GjDWGYFPQ8vqDGj+97isMPGcrJM45E1wvi6YXflk9OO5RrvnqmyOXzFvF/e3MV829YSCFeii9RgpmIQFTAuGHIhnb8+QK+SBy9+zPEsEpkJIDQdcsVFNqT9rwcNLl49BJ9kKYVfxW5Ivc/evsrThdMzG17MANQSINv9FjMkjC5rnaCiQpEewsynbZidGEapHWDL33jdrnivY8IBPwEA35xzVe/JLI5nVAwyItvrpLn3rBQGolyfOXlmBVlMLACMbyW0OTR8NkWIv4wvmCYdFs9wUmHYSjb4/NJvbUDfV8HIhBwaO1P7AG2xtqBrYEsAonFIMIl3hMbqOs+H/kPPhPKlspsFm3YYMT40aTqt1pWOmwYyL3bENVDEYZBMBiRmYKQZ1xxi1z14ScEgwF6Umlri/z8G+9y7o0/xyyrQFjElyJGDcKsilJ6wmQZjgQlb62idPBo8tk02UwbwZNOopADIppIfbIVs7UTgoF+xrvfyjvMUZ7CK95a8b7p+rp+xqKPO7F5IWJRjLWbJK3dCkhBIYvB886mp2m75YjKBo+BTe8jFc5fWwd6VoSicZK6lKdfeZtc9eEmGY9F5V+WvS3P++69UpZVoFVVIgdVok0cjjEwSumkYdTOmCZSTy8VwYZWYkPH0L3zU8zhFfimzyDfpVtQcurvH1hghy3xB7L+fUESK7vmWU/N/bEf0Fr0nf0ZISASQe5uwXj7A8SAAHpLQQbOO49CXYLuHZ+SGDKWmJSYH/wDY+R4ZO1QzEJeBBOloqMgOPnau5h+xS2cf+cizMoatCGDkCMHIyaPoFATZcDIGo6cN4euVZ+Q/8Vj1Iw7ElP46dj8HpGrr6BQGkb6hdQbO8m8tgZRErfhsf3slivB/f26lT3ppwIupOSNofsQ790RaohAkMITL9sIi14Q+bISwrfcTPvmNeh6jppxxxHY9SnmuhUYE6ZhjhmP4Qvgi5WQCsV4d2cr2ogRaIeNxhw7FGPYAGSJn+GHDeGok7/AzvVbqD//eiorhlAydBztG1dhjBtI8OuXk9+XRwzwi/SfX0PuaYZw2IG8vIT3s12e1RV24tJqflb0F5kDcMyi27MtVtxWoOTKDRgvrxTitBkyvzdL9JKL0Ja+QvPSVxk4+0sMnDiTxo0rrABGm3MWjB8P7Y34shm0UAAzEYXqOP6aEhKjaqgdX0c8Xs4Hzyyl9fqfUhkqp2riTLr2bKareSPxxUtIq/RJsEC+sYPcw88hEgkX1fcANy6t/c77xDMuA4rN3U/3D4S8A3s2FioEjEQp3LlI+mZOtXxzqlUn/shvyZw8l32rX6Hq6FMYesQptHy6hu4/3g/HzELMOBZG1mJWRhADwvjLIwTjYXyZDHvfXEfnw4vxrfiAISMnkBg5ha767exbv5T4Mw+hT5uKUZ/GNzRK5pJfI3c1og2pQxaUYbYm3XeO+3kEK8/ex6/5mQVFKSimmbx2oBgne9BgB2UtKUVubsC85UGh/fZmKXdkSSUSxJYuIXP2uTS9/QJV005h0LS5pBu3klzzDulVy9EH1cLgGihPUMAk29pJcsdeQs3t1FYMpnT6mYhQgtZNa+io/5DYU79GP/9c8rtSaCNi5H7zMuG1u9FHDqbQ1Ik2oAKptsLerbybebY3OUVm2E6ulzF+D/GyVxI84t4nrOybBrOQ4epqzKeWSUbXCe3GL0tza4aeRAXxZcvQb76BxiefI1EzgcSoiQwcdAhmupO8Aj22tWLqDZb1DYSjBKvGEDx0INIXJdWwnfbPlqAPixN7/Tlyxx5LYWcKRsbQn3kDecdvGbj0EeriJh9d+EM6d7fiq6zCVPsBV12LVt2bUXbS7lqRWHpVQEGrRcPh6Ik38egkPvvkBxUSopigXv6jRZJoCO3qLyG35+jOaoR/8yCRs84SqZ//XPa8/xzhYCXhquEEy6uIVNahBYJ2Mjevk092kNy4lnTHLoyqAP7vXER4wQ2kAlHknjRiZAzjj68jFyyEe2+he/AAWTokyrzXfiWWXXQHLWs346upsfci3tjfosGLDPWNlvx9NcS19u6ie4OJfh6hqFR+axQxoBLzO7+G5na0716BVJjHjjT5Y0+S4SUnoa15m/zLS8iuXYfY8zmk8hYIJ9UEgxqysgRxzGj8J12A74vzyFVXUGjWkX4D6qKY9/wBeeci+OH1ludQNG7v1KmrquSkJfew8uv3sOfFlWgDax0A2434+rtCJ6NtteUePECowE4R6UWEvMR7cQEPfKZmolRGMaGyGrnwaYz1nyF+ci3aIXWYLSapjI6YNJPAcTPx5cDf1oRsbUVm0oiAH1FWjllVi1ESIVuAQqeJaCkgqgJQ34L5lfstvJEf3wajBuOrSpAUfkFO4RN5BsVDzHjmdtbd+Bs+e/hviKoqm/hi0OfJFQjLDO6fFxBWttGbVHDE3a3/KRqTYg/HO7i/VfVICAYOhuUbkHMXYFw2D+2ieYghtSgAL9+i0r4SArWIobW96X6VscuYkDXQYj5EhQb7OjHuexV+9yKyoRlRWW6hR2L6oQqCk0YmL5KqfEH6KXQbViJ16i+vJTKokvU/fQotUabCld6CDU9xVW81Fl4b4JpIFyDurfTodX3ug2L/aNFNUirmVlTbeN+9z2L+YRmcOE2Ik4+TTBiLqKxURSt9XbbqrhCt9g7kx1uRf18Dr62C+jZIJBC1g+0t9i8eQNaVwqmzoLFb5pEiiYFh+izLrut5DrnxApLNnWx7eAlaVZVt8Q9UoNWfAVIBHcV9syM2vdCZRxsOwojiPSd7HIzBwChk0/CXf0i5+B+gQM6h1TC4FgYkIBQCvQAdSWjYB7v3QUuXtdukJAa1A+25qGlFE5D1wdU/hgduFfL02dDYI/MmwlRZKtNHvgBmDMrOmAGPvWS7Rmt6HpjMNo4HUAFcujwokOKeFTW628mDYKj7JSM8YqcYUa3yrtKGr//ZAB/vtKE1q4ujYgFVIBWC8krHrrg81pyUnwnhmP3sgjvt/meeKGR9D7pfyB7TQoeEMCDfZtck2XNwvVmvCjjFhlb84/UCvctdJNapD/IUQhRtgpUx7s249kpNP3Fzqr2s/wIRCEZ6X+eNMaRXDb21hu6i+O3rVh5QwnUOE849GdmQEYWcQSYeodWE9O9e7EWIijlNjwR47LzfQ71lH52KqCLHVEGopmkYqrbHO2DRXvQPlb11f/02UUV43Y0rvF7G9kFq8U3TyeC4TPUyzNRsJsbKkdffjUilEfPnWYCsshM9NzyC+Y8PrOhQCY563LRUwSPJB9wLmIYsZlbc4EEIjGQSI5uDsoSFySnY2Z6qbTesnaFHe+wTRZxNUi/TbEJVSYzdrzfmEEKz7LKK5MyeHISDCGXhHcmwQxKVlLGZKS0mhEGUIX9wPzy71ILHjY8+h2QaUVGhSmuk2d0liEYQEWe3aL1TzdcoWjftYPCRhoHs6WLeqdNY+LOvU5HwY2ZT+Mhj6hnMQs56RqOAVLWA+az1WyiU1MijqTI5PYswdeuakIZ1NPMZ+xxVvmfYz+XTCD1DPGQy7YhhlMU1yKUR1jh5zEIWn2YiDR0zl7bfY2VngtZK8+HnTG5t48r5c4jWVlhwXFV5WFz9zTOYOH6IlcnWLKm1iyll39yg2zz1daZigAk9Sc459Shx84L5ojIRgEwHhc5m/FJNLoXR3YHR1QZ6N7FgAaOjGTOThHwSo62RkMhg9rRZ11RZjJlsJRZQhZNtmKkuzEw3RmczMX8es2EnhwyOsXbpPcw5ahiypR6tkLTu+4wUhX31yEwHUfWeZAdmJmXDW6q0Nqhx3/038civb+Lsk45EtDVx6MhqHlp4jfjSF6dZyVNN89gB8wCZIWHHB45+OvqOQTqdkYWCIarLw/z+9QfoTmWZOvlQGhpbxHmX3yoTiTiP/upmUTekRq5au4lHn3yBe370TXbsamTm8VNZ88EnnH/Zj4iXRHniwTvF5PGjWLdxs7z4qp8qhJinnryDL0yfwkcbN4snn3lZ5vO6eOxXN8sphw3jhFlH0a0SIFLy0ivv8N3rL6EkHubdtZ+IK2+4Xza2dFvlt+MnDmf6tAmW8n3j0tPlHxY9i57LUCgYZFQOUq28Wz2qyoQ8lkUrCoDATa5ZUmBpr5IEIYTf7yMc0DjqyAni8Amj+MMfn2fShNHcvOBcfvL9S8XhEw/h4UXPsPqdd9HMDIeOGS4GVpeJRx97hhNnHiHuuuVSHv3lt8TUCSO49ba75YyjxvPwL64T9/3kKnHqSceIR3+/mJbmVqnCWmUjXnvzHXbs3MkxR00UR089VLz1xkoMPc1zzy/l5/c9zqlzjuFbV56J2dUBmR6uvPg02to7uPVH93L8MZPFsPEj6enpRs3b9uK9yV0r7BRW0WRfBkjT2ONUW/fywfMjn89ak1v0xAtc/62baGvroLamikcf+5Osb9gnv375eeKwcWMoiUUtD7ngOwvlDd/6Gh9v+IyzTp0hpkwaQ08qw1nzTmb7znoGVpVx9umz+OuLb/DtaxZw5rz5rFq7wYLNf3bv47zyxmprnGtvWigX3r0Qofk5YupEZh53BLquU1EWs6LNWEUJXz57Dt3dPUXXd/kFc8mmuu3Z9ymftQ2yNM0GjwpU23dM/WPMgqoIVzXr9i7NKgi1B9V8Aet3KBzG76vG51M11EKEggF5/wNPi1Gj6rjqivOEZRw1+N4Nl/PCiCFMnDCGF5b8nbohtQwfNpjHn3ye2V84nuUr18pvXHG+mDP7GC752rVMmjiGf3621XrXheedxlvL37PG6U7nLazgwXu/x9/fWsWix5/l5DnHW/4Eo0eefsrpVFeVC8MocO1VF8l0OiMuPG+uWPHOWosun0qv93odnzLQ6JkPbfKrVQWy9UkJdHW9RzDSgD8yUFW2qy8DlK/O5Qsyl80J3VBlARlyukHB9CvbQC5XYPrxx3LV1861srqv/321fPPt9Vx04TniiMljxelfnMn2HXu56ZaHZG1thfjrU3fz9BP3imwuL996ZwPXfOde+epz94knF90penrS8itX/IDV723gpuu+qlLkUs/rMhQKo3eleOnVlcyZfaSorq4k2dUtdSvS07j0grmioaGZicd/RXY27OPiy87id7+5hRNnTRP5fN6KX+wNkbPBMfNbSHZ9bDvgxWafYmlRNv1nRMq+a2HYmuZXRQdlpRFKYiEaFVxVVU4ylaOrs4fBgyrI53Va9jYyYuxwykvjrHtnPafMm8Gy5+4TZ11wo9z08TaaswWSHSlrBeJlUUaNGMzuvc10NHWATxArizGibiA7dzfSs6+dWPUA6gZX0bCvnfLSBC0d3aR7siqlzaRxw6mv34c/EEAtSHtbkrqh1WQyOZqb1Xg+q4CjblAF2axOJBKivStNt4otNL9KLgZltvMHtK+4u1+xNPYxNqFaRKo34Q+rOlRVCalZlRcqDayKGtX3Bz6/FaTYNYICLRjAVJbWNhTMPGEKj/ziBnHlDb+UK5e9h6ipQaiJqUyuyuXl8lYWxxcMWgGRqTZDuRwEQ/hCKvnpPuO3U7+BgNVfKuOVztipeWePoLAEqZ7VNLSg3wlGTStbVTT1qr+mfJzmw8g2yOaO8fBh0jVwougF3KLpxNFfFpHyP9kl88InNL9lDOyo18qf2Skmp/xE7ZcsNVNe2TSsDLEd6wl8/pAVOVqOx807OVFf8fso69MY21jZPHfKWJzg2MYv7TP1tU5xh1eE+Jyo1FMfbGXL3eDXNKxvR8BUqz+PztUve78a0XoZoC7M95F8788y2/VjhM9aImkahvuxkvVyBwAphgzOvsh0V0ULIK1iiaB1rdeZ2H2sON+TZbJCYGsse/+gjnY/FYp5BNR6j4riFHvtsWzW9DLTvei+wzSUBbAcf1DmOr/Xn3jVDoAU2A9oZcd/Xwajd6GpbzlM3f2kwF4iayn6FlR6fvfug1wIrW+9Ql+k2ZN79D53oGq1A2V+iz8dvFsBvupof90WwMiZUk/dSPu79/0HH025zXmw9Mg5IpD4Bb7Q5GLFlT2x3mB6v+//PJC6F0j1MqKI1h4wd+UZq19Wx8Org6TB7QJnSyyV7cq+L/Xum+h8/+3/4rM5txU7BCg/9hyhBc9D808FUaNEqqh89F8xh8Le7W7/TwC9VHgyLcWRDsY4F8rr96wHUDBlFmk0Is0PZSH/LF2rXrAX67/+cNJt+3WMEJpcQ0BEkYF+fR0igs4xX/ySqpfAoHNdPeME3vY17zh9cDhXnLxMOkgT6jO5HrLrm9RXN/t9DXewXvz7pj6rcYzlgbn4v9fUwqn2r78b5j9kQP/nnT63///O7v+w3aH+82Zz/m37fzqR3y7qyxc4AAAAAElFTkSuQmCCiVBORw0KGgoAAAANSUhEUgAAAIAAAACACAYAAADDPmHLAABh5klEQVR4nO29B5gkVbk//DtVnbtnpifP7mzOgV3CLkh0SUtGkZxEAa94VUQxwfUigjl7vQbMigoKogISRSQLArukZZdlc5oNk6dzV9X5nvc9dapP1/TCev/e7/mCpcN2V1edOnXeHA/wr+Nfx7+Ofx3/Ov5/eoj/jz7r/wuH/H8xUK63gEcsLDsaOBoebrjR+7/pff55hxCAlI1XTIZWT/4vPf/s39rYvUrg0UcAHO0BN3j/9Mf8c8daZgOPOXtZkSiAiP9Z7mVZ32hJG10Tnv8bgaYR6BA6R58tAN7/wvrIN3kujO8ugMr4EaSAONoGHnX/WWj3z3hBAZxjAbfTpNSRect8NHcdIqzkQYjG58CK9EDYaVh2DEIISOFzBAuweArCX3MBaUlYQkLwDxKgz7Ag6DfhQQj6TOdrS0rXSo++SQU/vhqQNKal31IAwuPn0Yh6+SzhCcCSnkfjAJZlwfM8SFdC2BbPS/rP47l7UnEH/Wx/PP4v/y7gef4ZmrtU96gxLL5Rz5WupXtoonyKgWxBelV4cgii2odKaa10Sisxuu0ZjD69Lljjc86xcfvt3v8pIvwfIsA5dg3wkyei58DzRaLlPMTTByKejgorCmnSom2rNdCPpXfn7/5U6Dt/tGrn+bO/4HrG5jnzDYKxwi9Hqy+MZxg3Ch+u0sAbQhx/LoQp/BO9iPp/3aGm5V8rPf87AZkHqsFHGv/q8+Hfg2vcYIqCENCtAuXRkizlnpO5oVux8+nbgL7+8TD4vw8BBHC9UDJpUq+YuPSjSGTfiWS2AxZxeUJn4ajV45UjIhOwmbL9lzOAz4utAeQDVwMrYJD6u3+NgopeJWPMOozwfwhhSt01oXMGognpA99HjBoiNZJE9ItmLD4wgzka/2rEYO5kXlMbh3/zXDUBPTAQFbAhnQpQHNqJ/MBP5Lb7vgXk+n0k+B9xg/8BAhgY13HCe0Wm60aRbu/2l6oCSezaErAUb69RnYkAmrsrYBNXl8L2KUAjgTlLvt+YcRiYmjR9CtYKXENFTnNi4zzPgbiAD+4AIRWw1KxqmCHfQEkMcItxwKunbs19mD7M8ybC6POOoYr4uEdyisa0rKigj7ldW+Xo9muw+8Fb/AFN/eV/AwE08Hs6MeHQH4vmnrfBjhK2ViCEDWGLOipV0PAZgM/6tQjQgKTzBPx6iBtcwUCKcTP2rzMJMhAXvkj2gcGLbACVF1KYYiG40EcAnxPRQhucpsZsNLX6CFdHzf54pFdopNbja7bv38NiwwQ8/0v3MQeojcn/EmfgP8IiF5YVE04ZcmTbz+TWO64EkP9HRcI/gADLIsCjDtJLFonW6b9DpnsOpEuaqs2rQUBkYDaQ8SZF6e/jFl8jgqFc1RS4+jFZAZT1AGf88BdbU1MgQsIs1pC9MKHqz4NvtWrsvG6pfAQJRIavjzZivhrR9FzC+oA5F1NcBPqAG9IXNHIQI3BJPyAslkLYUTm89Vm5+8VzUFq9+R9Bgn1EAH/A1NIDRfuMB5Fu74DrVGCJCESkHqgBhYdluvHIgCPQn8n2bYM6fGCMk+sm+zUhZ0AgoPA6aNR+840JPvzragDUSGyKCXMOhm7CBklI1KtB9ZVKrNRhR1g30Ke1XqCvUYAO5hhwMc1Z9B8rjFVYVhwjOzbJXS8fj/Kq9coX8+Z+g31AAH+g9P4LRevsh5Fq64LnVGHZtqQFIKUvrLihETKYT9UAN5TAACCyxgn4N7IcGkw3QAJ6R/1Z60wui0utevDlbNm58BxXyeCAPdOhdFT6ExEbViSiLBbfYmPFL+BIam5KvIS40psBmU+ZFB/WVfWc3AaKpHFfwFlIHDj6PkfYVkyO7V4n97x0FAqrd+6LTqAdM3s7/LeamkXz1N/7wCd5T1BXP/ossP4F696qXoYH12mlVSOPD0j9W4AE9HLjNe6658lqMB6DhV/bhVcqA+UK4HhAxAbSMaApiWRzBtF0DHZE6ate1UWlWEElV4KTK8IdGwPKjqK4WAxWMgERi/j4Rk8wxFWdXmMshCmignlq5S/ENgIuIQzgh/BHK45h5KlBKiIdtyIyXbPgzPmdLLx6DHCOB9weWrCGAH4T1t9x4u0iO+lsQFZAPF9TrJb5gdw3lLs61m+YcAF+0P2hBQymZI5D//qiQSOzIcNZzWOEUXqRly8BBPhEFJHeNnQvmIyJ+01F55wJ6JzcgUxbBslUDNF4BBHL5uE9z4WsuqiWqiiOFDC4cwQ71/dhx0sbsf2lzRjdPADkSkAsCiuTZC5BVlqNQ/n+DWkqvyaumkCTjZGEP/oyn+/T3nOFHCRKSGGsDexzC/ojLsDWBf3uVQWsuBze+F9y570ffjN94A0QwL8xu+ydom3qzRCWonytsbMct0PszwB8mOK1FRCwTR8Jgpk00BX04tRp0ZpN+hRPQ1Sr8EbzfFl83mTMPH4x5r51ISbM60Umk+C7qwCK+s9xUfU9bxbRsxCICYG4JZCwLKQAJAEkeGgH29fvwuonV2P1gy+i/6XNQLEMQYgQj7LjTq2J4biidQm4lgl300tucghfvJA81zqQfs86x5FGEFNpkYBL5zXnYE7qolqMypGtp6H/oXveCAnEGzt6ft4seg9ahWS2RwlVm3imYb4ZMjzAflMPCA05zvmjzo+Xp/r6eqvKXAhBFE+AH8kDzUl0HXcADjjnMMw8ZBZSsSgDesB1MVx1USTi4CnqMX1rFTWDoyZu1YKTbEwJIGvbaIvYaPWnsP6FjXj+9qfw2r3PQfaPQmSbYMViiiNo4AdiLXQEbFy/i14PQ9MPrBaN8KblYFoCBmLQw3kCvk+BWBopZ4WB1+SW2/YHrneAG8JC5Y0QwMeY9uO/YGUnXyvJ3KMBTXavP+uV1FisgdjIbtfatWWsOCleSnIb4+jF0Q4dzeo8f70ceAMjQEsKvWctw0HvPApTZ/Vw9GR31cWw46LEIQfyPY1/SY1TIiR4Gimb7AXwJGJSos0SmBSLohPA9m39ePTnf8Wrv3kMGCvBam1SzixeA702eslNf4UGonHeBHadiRpCgjqEcEPWgL5GfwdRSVQObLwK/Q98OzDj9wEB1LTT+3WL7JzVSDS3+AEQwbKYKd9wlATUH7bdDbmvhzUtBE0B2mFoioY6+em/nOewn8kbywOOg853HIGDPngqpszqIe8HdpYqGKW4Czsca34f3zfor60aS60VuywD81z5q9TclJMHsOmzb8oRfTFzlRItUmJ6IoYuAJvX78RDX/8jtt/zLAQpi8mEz4U1IhjmcMPDfO+9IMG4mIJhBTSKLajfyAyyURjYJre+vBB4LWfg/xshgI8pHcuvFs0Tvw7p+bLfsNXDzh797zhtOBzsCcn0OmeKqSQaL+TLPQqKeP2DiMyfigM/fRH2O2Yhs/ktxQpyELBt8pTXrCTFDT1Y/tgiEoEVrdntpmoswgp3xWGTkQ4KDtL9piRS4UOJFk9idiKGZgBP3fMcnrzxVjh9g7BamznQV+OSWleiz8ZIerLjrAeD0usUSX1eU7wMxRXqkUOQt0gi6g1tfCcGHv5VIy7QwAx8hHyQtogkLvLJwydpDRzfXKuzzIwXCuwXzQJ9zT0Arj7vL0SwHr6gDhQZw5lTKcMbK6Dr0hNx5KfORTKTwOvFCobJD2HbrMixae+PK12PAR4lE86fcaVYhLN1J6rbdqC6axfc0RFUK1UWPdFEHNH2dtgTuhGbOBGJCT2wYpHgXq9UgUfjWhYrjCyRhIUhW+Lv5SomQuKQU5di2pKZ+NPHf4qBB1fA6sxCEkpK4pr0knbNpNVrUeehDCvApt1vwidE8Y1ckP45/seypIg3vVMCv1JJJY/iDTiA7/RJvWWJaJ/8rIgm/KFMN22I/e/N5KsDusEhGsjZ+pfTbI10TkDm8pCxGOZ+4VIcdNahGHFcbKq48GybWb1SJZSJRFRpJ+MMuIrrofTyyxj762MoPPEMyi+9iur2PqBInJCoWwdbDHue9P5sKxLTJyOxdH+0HLcMTUcejljvRB7TqbosfthJ5N/J5wWQcjzMT8bYcnjg87fh9e/cBSubgbSJxgzOGXBPwzQc5/Y2nD1+QFXRoj7nh4sDKDfgBIoLqFyK8lhB7nltLoovbg8lvISh4LOI9mM+Lpp6vwIBn/0b7tEAyCbbDxI7avpAoMWbPgDN7kK+dA18jQCkxNqANzQCa3I3DvjRhzF90RRsL1awR1iBYsdvQZTpeYim4gyQ8pYtGP7NbRi+7S5UV7xMGMSAtaxmRDItiCRTsGNJWJEoLJvMOOVX91wHbqkIpzCGam4IEnSfB9Hag9SxR6DtkgvQctIJiMeicCsuXJfmaCmVT5CAkrA8iRkCmBiL4tGbH8aL//FzWKk4ZDTmrxGFTeyaxdNojfRSBHa9lmemwygcG9DWgBlZdPVv9HIUK7gIQ3+9JSwGQiKgS+mmVuwwZcu69Tp0WFjWvUQjrdY0/QzgmxQfIIHv19bA7x9GZME0LP3lJ9DR24a1+QoKERs2KWSB08yFFY0ikoiiuG4dBr/1PYzecjsw1Acgi3i2F4lsB6LJZsQSTYjGEogmkojGE7CjMVhEnYxzLpyqg0qlCKeUR7VcQKWYQyU/jOLATuTvuBP5O+7ArgMPQsdV/47Wi85n8eIUy/DY7CPwCn7N1yWQL1Vw5CXHIpqO47mrboKVAmSEoqYqSahmJTQQBWYASxMSL09o4etsV1OxCSuQhJkRiGjsCAnc0gB648AbET2nrxKp1jmQniM5amI6dzR7N0ydcUqhpnZtF/uOjeAphkvUZFkE/AjgDQwhsmgGltz6H2jqaMKmfAVV2wqwlUdyXdjpOCpjYxj56jcw8l8/AEb7Ycd7ke6ZimRTO5LpFiSb25DJdqKpvROpphbEiAPE47Bs0hxU9hbFByrlKqrlMpxSEaXcMAojQ8iP7EYxN8Tf8wPbUNi1EcAo4oceKbq//Flk33qkdKukMJJ5qnQRclC4Uoou18WCZBzP3PY4nr/yu4Y40LETw6E2jjBMGtGxjlC+R2Ax+BwgoHzDUcZr6rmw7SjyA4/JHX9c5lOlfAMlsKcVlt3lR7HUrLTLVqOaadKod66xNdPWDV7AwK8A6Ka3y2dXxHSGRmHPnoQlt1yLdEcTNuQrcEgTl0rWsksUQCwdx8ijj6L/A1fDW/UCrGgvmqYdjkzrRAZ+c+cktHb3oqm9C7F0EtFoBHYkApvYsE+1nvTgOsTOJaJJwHUcjnOlWzvR0lNFpZhHfmQQucGdyLVNQL5jCnIDW1F4+nm5ZdmpGPvklei+8XpEEjF4xTJcO0Jil63IPtJRimUsPfcoDO8ZwfpP/wJ2Vys8iluwPqDXkd7GCH4FnsCQL8C0kOoArUWxvtSHr/ZEsiVLqZKRXpWYK8gpGpCfiQDqZKwnCyHSNZ5jOGPqvHu1SSo8McyZQMM3WPw4z5ih6TOSeEC+CHS2YPHNn0C6s5kp37FsBronhBSeK2wK6sRs7PrSVzD6H59lLTvVsxTNXVORznajfeIMtPRMQSLTjFgyjhhRuwW4lTJyg3tQHBlEfqSfKdupUAKTgB2JIhbPIJFpQ6q5A9FUMyKROETKgh1LINPWjdLYVAzv3MS/57I9GN72Goa+/HkUnnwKk3/1CySnTka1UIak+UnyIQDbbRuiVMHRHzhNFNZtl32/eABWd7vKCQ1oaC9JyHUWAkJWgb92IddJjROYSKBtWKsZmJIBtgyZUDAQ4HoANwDROLnBo6xZsQEt/ZStkMPGlFkGotaoPyycjHPByxkxbbfCj5z7o4+ibXo3NuaI7SvLntHDdYQdj8LzHAxc8l4UfvkzWPHJyE6eh0xbL9omzkTbpNlINLWwbI+mEnArRezY8Ap2rHsR/dvWYWx4EE6hCDhlP45hG9k3RIgWook0mlo70T5pDrqnLUamYxJsK4pYqgkd0xYg1dKJWF8zovE0hne2oPjE37DxqOWY8Ptb0Lz0IDiFMlsJxNPoCVuEQKrqyGO/eCn+8PI6FFZtgZUl35qf8cPcyNcL6hay8dLVFMIGYlVfFNbVlJkdB1JkpNQd40WA9Kzx1Oq7a+sUDcP+93El8Aqav5kvYQLfSGgQwoU3MIyJ3/wgphwyG5tyZVRYXnJyNrFqYZP2XS1h4OwLULnvbsSyC9HWOxvp1onomL4AzV2TmKVakRjKuSGsf/Y+bFvzHMb6dykW29wFTJwPq60FyGYh0s2sIKNagiiWISkEPDyM6nA/Bgf7Mbh9A17/+/2MWFMWHIW2aQvZokqkmtE1fRGi8RQi8SRGEhmMbl6J7ctPhXPn75B96xFwFSfglE6SB+tcDwckYjjimx/An0+7FqhWIEhcsIfTJxqGv86J0GzdzGgy19LguEFKmelPaJBxzPImbphx40VA48O3s2uJF3qImn8oeMg4a6CRg0ibN34AgzLvB0aQfscyzL30BOwqVpH3FSrKzPdIqNoWXDgYPPN8VB+4B/H2hWibOB8tXZPRMeMAxFIZluvEQdY9cy82vvAwymM5oHkirAOOhXXAQoiFsyC7W+FlUpBWBF6hBEFh44oLOVYEcnlgrACrfwjYvQPYtgVy53YM7tmBwQd+hGzPDMw8+FS09MyCV8kz8kUTGRIgDOSRTS9i19vOgfjzn9B88EGMBLAj7DomHWZ1sYIFi6Zh5tXnYP1nfsGiAC4leJruxZAyaOoCdXqVSYCkH3na4a2JuAELoAfkx2mab44ANSww5makcpkPqcPE0ARM716Q9eJy7F50ZzHn85ei6Hro9yjHTYA0ac6yJ9aTiGD44ssV8DsWo6N3Lpo6JqFj+iLY0SjL8NFdW7H6sd9iZPd2iOxkRJYdDyw/DHLpAjjtWYAAQiFjyhcYHeaQLhwPkr4Xi4wIqDpAOgkxfSbQOxVizx6IrRuB7Rsx3L8NK+75PiYvPArTl5zEyjWJA7LrSS8hih5e/yx2nXEuon/7KxKTJsMrV+BZFmwJ5CwLW6sOFr7vdOy4628ovboJaMrUtHxeI8NHovWqgIBCwA+UwXqf0ji4afg38hi+MQKEMC4wV3zNNYjJm1m+BvCNBJ8a9WvM9H3VQsIbzaHrhkuRnNiGrbkKuVuFpIOR1oPIxDHy2S+h/OufIpZdgI5Jc9HSNRWtk+fzAyKRGLa+/DjWPnM3PBGHtXAZrLNOFPKkw6QbjQJDOfQMDOHgTBIHTuvE7kIVt64bgN0chXA8dvOiUoU7loczOgpnZATVsTHIckkFd+bMh+zohrVlI+SOddjy0sMY27MJ84++GFY0jmQT6cxzuZjIKReQ27YCuy64DL0P/wlWTYHlpdjleGhNxrDgPy7C8+ffyJ5LVTBkUK7pKaR0tGBpx9n39YQZJniNTDpuwPkYyX1FgJrvXyGjWYljyhrDl6/llp5s4Bk0UpnYsNByH5CjOUQPXoCpFx2H/pKDMnn5VJq1kJ4DOxNH4c9/QfHTn4GdnoG2SXPQ1N6L1t65PDQ5dtY9ey82v/AQrNYZsA4/BtYVZ8CZ2SuxcwRvtct417Qs5rdkkI7FUHKBHVXgr2MRDIwWYVGcKxaFbbmw7ASQaYVsr6A6Moziju0o9e+ErJQhYjFg+gyIdAoi2Yyhneuw8t6bsN+xlyCWIoUwhZbu6fxelDFXeOohDHzqc+j62meVPuBnHlEsYXu5iinHHYB1y5di5M/PsU7CSSUBMdUDWi23qYCbVG3Ke1PUGjDcm3/BP/YSo9QavpHyFXZYhF26JhYGsiuUAKE1bk5gcCAdF90fPRde1MZIleQYF0LxdRSMcYeGMPreKwGRRHbibDS1TULLhNlwPQ+2HcPrBPyVf4HVOR84+zxY118GZ0IHpu8awa+ntuLupdNw8oQ2xG2bKX97sYpS1YEzVkRxOA8nX1aUPzwGdzQPd2SMv9uuhUx7L9onzkEikYGXHwNKRaC1DZg8HdbEOSgWx/DSgz9BKTfC7xvPZJHpmIzmrmmIZGZh5JvfRO6Jv8FOxeFRMqoPzBGKUQCY/cG3K/FNsYUgsdMM75qwCFF/AOeQ2bgXRNCuAKAk9wEBKK2iBjzf0h+PIHVmh5EYwv82SBQ1fdhUbzE6hujh+6Fj+RIMFBx4frUOowkpiTEbI9d9Dt6mNchMmI+m1olId0zlKD5p+ptffgTbXnkMVvd84PzzYF15FpxCBed6Lp48cDLOmJjF7qqHHYUqKIaTjESRjkaxO+9gdGAMdr7MiRwiV2R9wMuX4VHe32gJ3kgB3vAoi4im1klozvZCksevUACSSaB3CqyeWSiXC1jz2C3aVEE8lWXEISSl7wNXXQO3XOEcwmAFCRnLVfQcvhCZIxZBjoyp7Kaa797wjZgOoBAXCESuaQCEHbvqx8bSf68IENHqvc6iCCZmlkfV/+t/1jV7gfxphMG+GKg46LjsVDgWkK9q+iCj32MXb+n5FSj94IewU9OQbu3mhSUbnWz8wW2vYfNLj8DKTgHOPBPWe06Bs3sYn2xN45eLJrEXZlOhirJHicBR5KsS928axBef2ogv/fV1FIcLsAnY+SKQJxOwBDlcgBwpMBfwyILIU3ZwDs7oCOJ2Es3NvRzVJd2AMoUxoRdW9wzkR/Zgw7N3BtHSdHYiUtkuJDvnorLiCQz//Few4hGVku6TwIjjsQo8453LVT4fV/z4XCCkJ41T3kxHnMl9zTQ0EzmCW0wzwoD2+FPkLRwvN+rq5ca5g42Yf2Cn6t+MuLUOb+YLsBZMR/bEJRgpu3DJflevLLjS2wJyN36J4q9onjId6ewExJs6ISwL1eIY1j93DxBvBpadAOuSE+DsHsV/dGRw3ewubCk5qJBrN0JiReK3r+3G714fwKY9OdhVF0lPIuK4kBVHaf6U/FEuQxArrlYhyUZ3HLiuw8+XFCV0qzy/dLIDudEdEpWyYCTongirWkb/pheY/XfPXArIPFItPWjqHEZpYCtGv/gNNF94LqxECrLqwfV1qkHHQ/fxSxCZNQnOjn6IJEWM/NIvqkoPTGhfMdT1jDrGohXt2kLX/uoQx0SKfeIAjlI9Q3fUELHeravrZg1UCYV4tXKglT8KlxWRfceRiCRjKBYp/544Jnl6iWSjKD/3PKr33ItoegrSLUT9kxgBbTuCLS8/jGohBzF/Kex3n8ps/+KmBD4yswfr81XkKh6idhQrd+Zx2X1r8LWnNmH7rjE0VV00OS5iFQcW5fwXq5DFCqRfOyDILVyuwC2V4BaL8IoFuMUCvFKBEcQp5hgxkpGU4AQxp6rMuI4eiHQbtq96FKX8CGv9iXQrzzvZMRXO5tXI3f5HzkZiXYASmS1LDpersJqS6DzhYCBXUOuifJ6NAdhQpoccciYMDKQI3MT7hgC6IlFnIdQArwJEtYH9s8ZDQ15Ac+I+9soqLVwKLaceysqQRzI9UBnUtYXv/5hr4jPdkxBLZ9kfT4GckV0bsGfzKojuWbDOOBHVlibs53n42PRObMqXMVp22CH0hzW7cNkfX8LLmwfRXHGQqlThUURxIIfitn7kN+5AfuNW5LdsRX77NhT6dqA40A8nn+PsI+ICbrkIr1KCVy7ALY0BlTxkJYeIlIh51MPBUXX7bVmIjsmolovoW/OkqpSiYFUqi1T7BAjRjNEf/hwOZymRLsBJpqIKwe8/6eSDgahNkaj6cq+6Qpl6xVDnNo5T+gJdwbDQ+E9fwKlJ++4ICiyNMJX7Lk6lsoXMv+BG/d3AaOJiY3lEls5DdN5UjOVJ+atVGIlkHO6OXSj/4U+wYz1INnXwn9I9JLav/TtgJyCWHAJ5wGzEdg/h6gOmI192MVpx0BSP4/cv9+FzD69FzBJoITu84sAZzaM8MAQvV4QkyiUrhGSvq9g8fa+6VQhZhW1ZiFL+IAHBqcKj88JDpVJCuVRir3IyJpQNT2KDlMKuLojh3RjY8iI6pi5CjPMPMkim2xDN9KDy7DOorngBcfIQjpGvQ5XE5xwPLQfMQmRqF5wdgxCptL9Wui6RECFUXFungJvOuLAzoBE4OWluX62A2vPq/vXHMvBx/L9msEenLlOWMnv+KsgcfZDwLOK8dE5dztU5MaD0wAPA0A4k23sQTzbBjiTZ2za6ezP/iZ7pEG9dwg6c45uSmBCJYMtwEa4r8Pj6AXz53lVIVFzESx7kSBHlzX0obNoGd2SUAUo1ggR4QZ9dB4LEEmUcUz2+58Ct5FDK96NSHIFXLcBCFfnRYSQiHo5ZMgcLe1tQHB3j7B8VxHKAbAtEthNupYTBbWtUBj2Zhqksvwe8AvJ33aeyh3x9id65WHYQySTRumQeUCzV0r4COW4WuBpWVB34DDFgJuOMMwjClSp7RYDaAzkg2MgGDSaiNX8jTGmGgINr/ZJmorpYBMnDFlBUHC6lcfv+Id0Go/Kn+4mBItHSgXiqNXAo7dmyWrXtmbcA3pSJSOaKOLytGRsH8xgpVbGuP4fvPPAK+/hj5QpkrsiAL+/eDVElgJPCV4GslNjzx5zAqTAVS6cEuGWOSDJbdx1USqNw3QJye3bj4HkT8OzN1+PhH3wSL9z2FfzouvdCkk7AXES9Ezo7gVQbhne8BrdK+coCkVgS8XQrBFrYocUBQDsSBNornspMbD1sob8IZnavhpzp6zUBqxTuoGMJE5o+byh+OjVed7/YJwQwWbn/3DoPlIkIfM7MExhv8we/VKqwuluRmD+VYFGr1KHXiMXhDo7A+dtzsO0ORGNpCIrJc0bvGEZ2b+KInpg3i+v45sdiKJddbBoqYLDo4HdPvo4d2/YgQ/l9uRIKm7ewCWe5HnvzSJEj2c5JnfSvr/HDKVI5HaRLSKERoIpYxEZ5104csmgS7v/uNWL2lB5RLlcEHe85+zjxrrcdAzdfgG35NXtdnRDNHSjlBlEY3qUik3YUsWQTIol2VF9eA7l1K0Amod8TyvX1gNTiGUAq7puEJkdtkAquRa4P5JpC2MjxF1Ygx+sAjRFA25X65oaD7u0cUbVR2GheUiojMq0HVmcLKiXS/hUH4BqGFOC+tgaybzuiTS1MPWRb0++54V2oFoYhunogJ08EBkYxIR5D32gBA4UyVm0dwIuvbEKaAJkvobR7N5zcKCwCapUifmWA/iWqZ6ATtZdVratHOoDDYoDbsnhVRGyJyuAgDj1kIe6/6TrR1pJBpVJBJGLDcVxW6E4+akmtVo+OlmagqYU57SghKyGAFWFdINbSClkYQHHVata6iJNpnZ+MoNSUHtjtzfAq1VCQxzShQ0QXLKqRa9no2qDP0XjgvwECBM5909EfgvZeHsoOIyPdy58I31R1EJ0xScUBqWQ7mKf63Vm9hss3o2mS/XHfCSaR698KiChEVze8ZALRqouNO4axvm8Eo4UKXnxxA8qDQ4i4LqpDA6gM7ITl18sJpmqfsqmynT5zObmidAV8+ld9phyUyp49OGLpXHHfTdeJ1pYmBr5NZZE8VYmIbWHrjj41XjSilOGoDdHdxUGX4ujOQE+KxFKIpal0pILSqlX+qvnVSUKgTGZpaxNi3a3MlerW1AS0ufyN3MV1yroPr71y7jd2BOnGOn5Jja7d04OyJVvzNtXhiSGzgiRFJds4G5bk1PQertTVXf0UJ1Ozc1e/rmRnIo1I1E9e8SRKY/2U0wZ09/IiJcoO9hQKcrA0jCbLEns27kSENPZcHpWRIQimbq++e5ffTIHzi3QXLo8o2FGJIUz5QHXPAI46bAHu/u6n0NKU4uKRiK1cuY7rIpGIo39oGN+59R4ODrHbmpJX6OaWDJBoRjk/Cqecg2XHmYvZMao3jqK6dv241Xbo/kgE8e42FFeu86dsavlmtNC808yyboAwdb+zftbQCmiAAL4L2KMmhr5/flzmqlZYGuSyB+yIQVovw8iT35lVl/vz5VH0EFu3sAJI7J+UJfrBdUqolPJALAE0Z7hxgzM4AhCHcFyM5aqojBLQPXjkpnULyuLgmnkBj9wZAUdSBZWqpwB1EfEDMZ6jgL+rH0cfuRh3fe8/RVNGAZ+yh/lNXBexWAx7BoZwyhXXYf22PYi0t7JzhxVVSvtubQGSaTgjO+GUi4imYhwSphR0ei+5Y6dPJ7rNHSnC6mO0q83QAQxLwDTvVIzcDxGbSaKm3qbzATUs/IqkfecA1ANC1/ObCOdjYsMUZkNEhAtATIWQssRbM3V4qpUaRpE9gxCgfH1yR1MI1YLrljgVDOlWSEqyLFdRGhiEMzQikqksJPns8yOIRmLI53KQY4MQiThc8vBFE7BTKU77Vj3/FBfyQ05Q/neHox/V/gEce9Ri3PX9TyOdSqBa1cCXLPMJ+Lv6B+VJl1+DF17bgmhXp6pPINd1lFol2SpaGKfon4NqpYRoikK9Ur0PovAGh1T6oX5ftoAkWwI2cQ8fOevzQUM+lppvv/Z7nedV6w4aOXzxwLGK1L4gQFRD0qhJMoEZ8vQF5mAYJxp4qmhBM0nWAfR5sgS5TwuJ5RyVeZINrQIrXNTCzhhHJU/SH2fwkLu2gPxYkd2zUSlRyo1iQnMUn/3opVgybxo27OjHdTf9AavWbEG0OQPHob6VGh+1t02yE66ypx/HH30Q/vjdT4l0MsGUTzKfXTE+8Pt298uTLrsGL63bhlhXB0iYsEOH3on8+CnSWQQVYCgt3yn7XW1JH6QUtwhksUByRGlEPo0wEhACkBVgAjEAtSFjVZvb8XQX9Bho5Ax6Q+fQm6WENfLzq0FrbdTMnHTTgWQAP3gPwcoSe7x1HqSvM7KY8ytyNfvjsmwKyJCiRveS4kiaMi0iKUyOB0FVPa6DiFfFHV+8EocunsWT2n/eNBx5wGyc+L4vypWvbAiQIEBkz0U0IlDZPYDlRy/Bnd+/DsloJKB8usT1mO3L7Tv34MTLr8Wq9dsQ7WiHQ5OnbCNy7RICJOJAOsG9hlSmseEjISTQmVSscyjIc6aQVpUYhoYsD3L6DIo26ywDxzmZoCHuG8CphjgK8em/JB73KRbAw4+P8xg4oZTCWg/lOtkVaKX6ZWpIISkNy4d78Ip+AhFnyjILrNm8ylXgvxA5cggJyD1LZp1bZvdtdWwUh8yfxMAvl8vCqVZRLlfQ2daC+2+6FovnTkJ1aBAR3XfXrSBiSVR278byow/UwBdVh3ov+jLfU5S/ZfsuHHfJx7BqQx9XEDMfIXkfjXDOAskPkUkA6bg6R9yKAVrz0nGDT9JLolGl29A7Gq56Xg9Gfu25aRBLCQMgoDvNHYyM4mAM//kNx9krApR15+TQJEyPoKnha4+h/inc28bgBq7kJk6BkeErL3S/R2KfomsgiveTI0hp1JU8ZL9XKgrjXVLgdAKF79olM47Xnexvi212YuVdbS3igR9eh/mzejnVKyI8Bn61fw+WH30Q7vz+9UjGoqJKdYZ1Cl8Um7b1yeXv+hhe29aPaHurAj4BkIEfgSAESMWAtiagOc3igJxOyglEVULExTwej1r5IZ3iYgFuPWdYzvSmZUpQUW9grLmBEGGN36R85gQGkMNmYqBU7JMjiHQAfphP5LVa1trRoJ5Ns/iw3DLdkrQgQ2N6dB8RfEFIk+lq54WiAAz3NqBuqHaEu8zLCmXuFFQqNb2wr8B5lTLsmI1nV76GF1a9jmg0CgImj2dbqJQr6Olsxf0/+gxmTOlmblEdHMAJxy7FnT+4oQZ8Tfm+zF+/eTuOf+fHsHb7AKLtbQx8ku+cHxiPAQn/j5Q3UmybksrhVC7BisaUGctijLrF03yqsDspp4HZS81y48pioDI0ZmjFIT0g0PK1UheOyvpcU9vWer3riLBxu8C9iwAWGvqj8XOAtg24hOEMqs3CkE+ehNPXX+eo8sx8kklUvlaFWy2rlv1U1xiJcvInyIc/OqJStyNUuKSideTTp4BOoVjE+R/5sty2cw/isRgDkg7iBJVKFVMmdooHf3ojJrSmcNwykvmfUcB3XG4FUwN+FGs3bsUJl14j1+8aRrSjg122BHzE4kreJ5XMRyou0dHGHECSGCDztJhHhMLXsQRXMalEUXL4lmFPnaLel0WekrC2H+twdg8oJdmwoERDhdtf17CUqPP3mKFh02zY52hgCH6+M8gYLAT7kBliRqlCT5MbqXTbqIz2GzGxJjyPsn2pYLPMFoASLYJLu1nuD+xRvnxyExOLpc+uC6rQpS4fr63fipPfdY3cuXuAAalYL6XhWYLEwcypveLZu76HP3z30+TQETW2L9jJw8DfsAXL3/UJuWHnECLt7ZythGgciCcU5aeTAAE7k5QsBqZ0A61piLgNuW0XBS54vlYkzkjqumVUKZ8AHuy5s3QbSF/XIRNRcL+Bat+gUiwNm17Fb7SCZHJUwzQ0WX9AsiYn1ggj/tGs4BA3Cf00vsGDcZj2qy4rp8tiUXgbqUkF1ctbmpuzY4Q8tJH5lOufQjU3BqdaCuRnoqVLKW/D/UA+r5SweEohA1skUsnt5gxeeW0DTr30WjkwOMKsnD11vjggDb+3p0s0ZdKCLALLZ/uECMQ11qzbjOWXXiO3DOZY5jPwieop5p9McCILs/ymNLF/gZaMiMyYCKsprlLMNmyhfjJIZScECjC1VK7mRtkRlFq8SDkgKTCuuJ+k7OdK/zCc3YMQURX7GEf141y/xvqaMBvnQazr1yT/sYwgA8CKPut4TD2Fj2t0EFYafaMiHoe7sQ9O/6iKiqk4sPL4FYDInHmweyehMrxHIQAph66DZHO76uYx0g/s6VN2dKZV2dncQdOSlKLhVj0Za23Dipdew2mXXSNHx/KSdAJKNVO+LUICJ/AJCGK9jot4PIY167fI5Zd+Um4ZGFMyn1zPJHoS9BejLBDuCUit6ZBNwYtZiM6ehNSkThmhcxu2A327mI2n2yZyLiGtB3kxy0O7YU+cgtj8BfCKKgqqOr5L7rZfWLcNcs+Q4gBhItPrWsdkDQBrJNHngh7ItbX33U7/CAcwGxKEFA4TG33xUJuZLkkePyxPIhaF3DUkKqs3cQE6e+e0GVmpwGpvQuSoQ+B6A6iUxph90vY58VQLZwbJkV3Azi0qqSPdBJFu4Zg81y77VWRu1UWsrVU+/dyrePvl1yKXL0gq3dI6Ae1jodvBVX3gv7puE46/9BpsGyki0kbavgVBwCeqJwcPyfz2LCQpe83E/mOMBC1L5iBOiEzjPfkK5HAfUs1dSDS1cyKp41ZRKVBewR7E33o4nNYmuBSWVj0Mgxzg/LOvqva2ft+Cet3J/GrAIHRZiDoNmBg9mfcdAfTNGhn0s00zUJ8bj3GBFRC0MfHzCSnUVqrA+dsrNcSWInCJknxMnX46K4KlkX44pPn7R1PXFBXVo2rfMYoFRCAp546ARYUkrBeR582SriMRa2+Vjzy+Ame851Mok2eP+/uydcDaLYmMeDyGVa9vxAmXf0puH84j2tYGl/a1ShDLp/4APhJQbWFXM9CahGhJwKNi4/2mIT21W8rmOLzXd0D+bQXg5pHtmQ1hk+hxUa0UUR7ZzXpV+oy3qbZUhosDESHoXOGxF5RpWedpNYHZCBkM8VDP6hvoCPq1E/uiBJoPDVG3OQPzpzplo8EYQVoziPVL5y/PsNViJaii15+kZaEyAsSPXw7RPQ2l/q2oVvKq5s6tINM5BbFUK+TIHmDjWmUBpFKQXRNViCJiczMbVlcFxe0lop0d+MvjK3D2+2+QjuPKaCwmSQRUqg57+Fat2yyXX/4fcvtIAZG2NjgiAiuVAjIp1vJlMgZ0tUFObIVsIR0gBjcqEJvYhklH7s/TdiO28G57FNi9DZFECunOaUqJ9TxUCiMo7N4Au3sGosceh+oY4NmWSu2g9PR0FKXNu1BZ+ToEPbNO0TISQPQam2LALB6pE8f62hrnVvRJ/6kR1BsggDb/tNJgAt1XB+v0ABMxxo1VExMEHQpKkTPkhXXAa5thNyu3acAJqM1KVysyF50L19nJvXkoEMR1gpEo2qcuAgrDwNY1wK5tQDwJr2syvGw7RwaFbQnqGyhsW1jC5pSzeHcH7n3wCZz7wRtRLJY5nEt/5DNY/p5PoS/vMPBdO8pz4x4KqSTQ3AxrSg8wqZUp38omITMR5gDzjz0EyXgMbktKFP/0LNwHnwDcHNomH8BVSx61mnGrKI7thuv0oenCs4HOLAeo2APqc0ZSM/J//jvQP6wsjBoEQw6gRkTpfzddirzkYd0rqBMw4sr7Whyq5UddH8C9VKSaekOw+YM5nl/sQI6UoRFU734CsU9M5T4A7Cxh96gly3mI5BX/hrHv/xD5vvVIZtqQjNAGJVVkuqYhsX0NSiO7YK1fA9naDZlMwp08m8PBIj/MwOdXtSLsXq9SC7nuLtz512dx8HkfxjtOPAoDw6O45YEnOJcw0toKl7yNcbLxY5CpBES2GWJiGyTJ+7gNKxmBF2N7Awcdsj/aO9uwEw6Ka/tQ/MbvgJHNSKTb0No7L3BiFXODyPWt55qBxL9fgTIrf9yYStIE7ZjFdSn52x+BIOAHpeFGE+6AiEJsPUzxYS5cc9b4pxoE6/YpFhAA3HTvmghiDhHqcyNCHcX1Obo/lYBz+0OqiUJadcrgoUkxGytDzpmJpsvehWpxIwoju1RAiK9x0TP3UOVl2LkOWPUsdZDgaJw7cwFktkPlq5CuwX8Upo3BteNs06/augef+/av8P1b78OIZ8NuaYcbiUGk0xAtzRBdHRCTuiAnt8NrSUKmaH+AGBxyU8csHPqWAzG5twcFz0M+V8HQNT+FXEdZPhI9s98SpMKR7KeuYk5hA1reczkwewYq9F6UQEg2lesK0RxB4amX4D39kjIvg/7CDeIBdX4WUw/bW+lYWB8IYCX/ARFg+gL2onUykI25Bc9uoDPoPy0GXt0E996/QTTXvF9s0QkLpTEPqWs/Aat9InI7X+dES7KnSbGKZdrQOftgyPwQxNY1kK+uVPcmk/BmLYTsmcxJINQFRJJZFYsxdXuJJCJt7YhNm47Y1CnslpXtrbAmdkP09gCTu4GJbUBHE/ckEOkokIrAEVU0ZZN46xEHYcKETvR7DnZVHWz76I/gPvoYUNyN7hkHc0d9bjbplFHKDyC3czXszl6R+eRHUc6rTvt+8E6ICCfIo/SDO1UE1NxLoQ6A4TXVUb2Gjpl6/AhEb3DNvnoCG+YfNWAxe3EAmYpLXZZKqHkyeeq++ztYjiusFFXa6GlaqOaqcHonIPuFz8It78Bo31pu4EgXOOU8WibMRPv0A+GN9gPrVkKufIqf5VHyx4y5kHMXwWui9u0W9+bz2IWbhJdOwcmk4VAD9Nas0u7pry0DZJMQxPLJqZOMcEsaSqaZNnsKjjxyKZKZNHbYHrbl8tj0ge+gcse9QH4H2icfgNZJC9hiIQW0XBrDyI7X4FV2ovXLX5LFCd2oFhyeC6eikMXSGkPp76vh3vMYBGURBQkShgkdNqq0j69O9IYUcvOGRsG4BkcDHcBP1udMMEk73xqD+G5Js3ZQAzqYhOGuNNufBEqNonjR0gTvmVdRueMxWOcdA+QpE8kvo7Zs5HdXkX3vZUjfdTfy99yFsUwHmmhnj2QTt3xrn76Yfe1Dm1+EeJ0qfFxgyWEKsMmJACVoDvdDkvuYctAp7Yf8+Unt3CEKJ4AnIBMRyJgFSXGwqIVIJo6u3lZMnjUJ2Wwz8p6LkVgEu17diN3X/gzeY48Due3omLwYnbPfgio1mpJAqTTKxSul/heROfsi2JdejNF+2mqBFGA/dB7l3BeUvnQzuSCV7e+F9lbUuoBe97r1DK+zEXJvRJsBbBojwRtkBBm+6FBjyJoiaE4qLLt0th8zPf8F/H2C2PSz2MnifPFnMnLSIUx9cpSybDQXlBgb85D98fdRfsvLGNnyHHsDScuOxpLczrVt2mKOFg5sehFY9xwwOggsXgrMX8CFm6KbHEVTgVJBNXggJwwZ4pTBSwpfmoI6UfbyRbIJxDub0DyxDa3drUimE1zJ2xfxkM9XsedXDyP37T8A61dR6A7dM9+CtmkHsLxnk69c4uzlsc1/Q2z+/mj50XcwMkZ+X9L61VJQ8opoiaH0u0fh3fcUREerjxjm7ishCcDrXddvx6D6MM9v5I8xe02Nrw5qgADVmggwLYe6rh8G9ZtRHdM8rEWu/D1Z1a4gKqjkp0RRx641G+F+7TewP3s5QD2Cgp5DNqpjVYx19qDtN7/CnmOPx8jWF2BHYlx/b0fjrBcQElB3jt1rn4Gz9SWIsUGgbyuweH/IOdMgOjtYk4dFjgePssuZCil4g4TN1B5tSiCSiiOWisGORpCjjiW2RHmshOL9r6L4y4fhPvoMMLIFsVgEExcvR7KNCkILsISFcqWE0f4tYmTD32SkI4uOO36DsWQW7jDVAdY2vLKaInD2jMD5z+9CUB5BAMxGW8z4XtU6f8tesq/q4R0su8qGrvUOA2xvH81AQ+kLMKvBBPXvpuJSx6aUem/s3R0AlzZv5Frp9lbI7/wG3slvgTh8P2Cb6qxFP5NDp7yrDPuwQ9H5m1ux+6yzMbjhaXjTlnJ/QOrNQ42ZUm29mHLQyRjY+KKqIHphJ8TGNRDTZgOL94OcOw2Y2A60ZyBIxlMiR4Jy+aJwYjZ/r0QtlISENZyD3Lob7orNcB5aAffp54HBbRCiirYJs9AxYwkXqFZLOU5UqZQLGO3fipENT0tqCt115x0ozp2L8u6y2n/QB5SwJWTCQvW93wI2boHo6eZ+AWqXCyN5U3PYANAGtRvrHHhqTKKr6xLqe18DcDIS7gsHMM3AkLFv2pNB3pvpFDDMD8NfHbgfAiwnUUBdoSVVTkCiCO9DX4P98PcEWhMSgx5nzjAR2Dby28toevvp6L79t9h9/oUYWvc0vOkHc9eQWCLNmjdxhO4FR3AvoeFtr2JscAe8lTuAVc9BtHZBTJkBTOsFJrRxO1q0NnFAivoFkwMKo0Vg625g/VZ467cCOzYD5WGu62/unIDOGUsQaepBlZRRSkcTgvsBjOzZhNFNf4edTaDzzjtROPxQ5PsU8Mm9rTikA681Dvem+4G/UHu4NniDw7Ba2/yGrLrhg264aTbfNomrBp6AmExMMJ1Bjdz2DQ4xfrOIA/YX2SkvIJrgNn31Gx2EbFX9sKCLmOH5C4sFM2IVFEH65dnEnvsHIM47SVg/vQbor0qUfBOTNxKlPsEO0hPjIvKXhzF4wcVw9wwiM/FANHVOQyJD/a1VzR2JCEojcwrDGNu9AbmB7Sjmx1QfH1IwKPwWz6gwrzR8GyXqDVBQ28rGbCSTCW5H1zJxIex0J5xKieV9xLLgOBV29FAhaHHXChGfNVNmb7sNpf33Z8pnE1RlfXBUUPTE4d6/Et67vwhE0phw8XKU1r2Cwbsfh9XTASkjxjrrfQhNp1DIJ2B6CAOOYabgG9FbCjsKz0Y5l5fbX5kLvF63aYTJEtTJ1IIDRHbWSkQSipeb+93UySv9t5dagTrZZSiMwSTduo5hhARy5x5Yn3mfENdeBGtnRUrH76od3O4iNSGOpg3rMHTZ5Sg88Rii6Zlo6Z3LMXjOHKJQK5ladgSRKG0i4cEpjaGc60dpbJBZNjVzYBezpDbvlHJm8b0UdUykO7hpdDTTzn0HqS8AFY5SBg+JMmL9Y4NbMbrlVTilTWg+7e2i+Uc/lENtXaj2l3mrGo1ThLTuhDjkinWQp18NOVBC04WnY7/vXImMcPHK+7+EvlsfhNXV4bfiMxBBi0rTqjI5cSNSNtO/ashASriNylhebnv5jRBAc4D9DxDZKSsRiatykmCnkND+wI02iKqDeGh2dS5L/4/DgFQQoFulVSlnUFj//XFYl50isaMM6db2DmbHoufJWDYmmq0Kip/7DIa+/C0eJ9UxH5muqRyKJXHAAONmlMIHcgQR8vdzjR9tqMTFako55dRtkteUGVRVnkc/m0j3RqLAVHGsH6M71qI0uAZ2OoO2G6+DdeWHMVKkUACZsbUd07jqqDsOb+0WyFOvgty4CZF3nI6eb10L8n1l0xHMaY1ized/jqc//3OIbIsyU3mdVVVUDQHCbfoa7bgSsAij04gfA7Bgo5LLy23jOUBjP4AxloKfyc+1vDEVFRMHtG1qWA/6mgAJVIcMyZ5RHT/wr2/OwLvqa6pl63nHCGtHRXoORfz9Wg7bRmmogmrMRtMXviAmvP3tyH/6Mxh98H5Z6F+HZNsMpDsns1ggJZFTxzg1y0FVFoOJCGFRPpZfmaKLE2pp6GS7U1FKqZRDKTeA/K6NKA2t41B19uyzkLnxM8jNn4fcLmo6QdKF7Hl/dz1KBpkQh/faFninXwVsXA9x8kmwr/t35AsFSXpNZ5MlqPXcIZ96N2IT2vHYR/+bcxtEMkn7QNf2EAj8AYaTyCQ6k/IbmIDK/cjWQEMtoKEVoMQJ94lX9YF1g4bYj5mXFt4iLuCFdS5JP8tIX+pjPB3Ec6IxKTMS3ns/B7tSkd47TwR2ViHKqjLGc6mPKKWQeRje6sj4fm8RzX+6D5n77sLot76N3F8fR3FwFSJ2N5JdkxFrbuctYyhLlyKKpB8QN1CxY+FrygqhKfRMCalOpcCh3OLgLpT2bIOHnRBWCtkzTkbiQx+Cu2wZBopAdbtK4lD5s/67ug683jjk82shz7oa2LwBYvlJEJ94L6r5EsaqVbRO6SDnk+ynwtdiRSy67HQkejvxl8u+wK3p7OYmv+bQWHLeNMPvJqowriZ+9drWGXnaGmCf3j/iCKLSJg1cXyv1MY/tynpf5HhkMLhCbR8RvQM4uRF1AoCBAPyS2jKgWVFrewn3378Aa9eQEB87X4oRDxjmwjrlO2HctFEeqMg9EYHkSW9D5uS3Ifu3x1G49TcYu/dBjG15FuijATOIRLKIpGnTqAzseBIWBYsEVQCpziW0t0A1PwaHQtDekF9FE0di3lxk3v4uJM67AJX9F2OE2g3srPh7BEVUapfP8cjH4HXHIe96EvKyTwMDOyGOOwH44GXct4Bg1jJnEqLpBIYrHryIYJPXKVYw5cRDcdJdX8HDF9+I4o5+WG1ZlVbGXEovubnvoNm9XROasbR8iQH4fbcCFh8oWqaugB1Tq80bK4QUPzPRM2z+BVZAAxOyHjvqvYZ671vuu6P0AtqzEv1DsN59BsTXP8SFlyCqI9nol1LVjAzab8hGtCXCOZyxgVFYLzyH0hNPovzs3+G+th7Ojt0qplDXB0n4qB2HnU4h0jsBkYVzkTj8MMQPPwLewgNQbImDEnvJMUXZnDqTmGfOeowHZGOQ1Pz7a7dAXvffnMYuTjgeuPQCfzscgcTCyUj3tMPyPKQiQrbYEBlyiEaAtOeiLRXHyJZdeOiC6zHywuuwKBmF+0T5yqHWw/Q+TDUXXz0RBu/G60pflA6w49V9VAJbQkpgoPSF9gccZ/9r34C5hXpoN9AwstTpFb5FoJVCbR3sHoA4cC7Etz8OHDwPot+DyKkAixqexJWvu/h9jaxEHJEmldcZdYEE7wWwG+6unfBGhrhbmaCGkLEoIi1NsLLt8Dq7IDq6UG6KM4qQVejmPL9zB9GCrofUEk5CRgTQE4Hc3A/5sa9D/u4hDnSJ5ccAF54LWcizOI8eOBPR3jb2/9v+LuXNtkCGiotsIBMhpuehKRWDM5zDo5d9Dn33PQ2rm0Lc2nmmLQTilnrjKTUj3kspoDlz4yi1Kw7KYw0RYO/FoXVuXe2U0B4rIwZcpxaYAQrTbDHwo7Z6DTRabQcb8yAkIDNp9UbIUz4EcdXFEFedB9kRh9jpQBbZkULpgP5CKOSTZdr0yUOZHkOJoLEU7I5psHqmjdv9Fv4G3Nw8hBI3Rmn7Gq0CURMg2gHIiG3RTmOkwHbGVK3DLx+E/MxNXP9PaWPi5NOASy4Etm/nW63F03mzikqxwpnJNKWq56LA7eA4RYzn4EYsbnzZ1JzBCbd/AU9d+Q28/tO7VcUU0w9v/mRQfk3o12lmdQp3HYHuiydQ3aTq6cMANEY2PVSmph9cEu4tZHKDsOno+wiMaCG/qL+/HpeBNWdZNMjP/QjenX+F+PglkGcco/THftpwijiHQib1ePov1RX6j6hQGZnkos86lzm0dFLIqJaopvhKamTom6ysMCYsiM6oqvF7/EXgazdDPvysKg5tb4OgJhUvvQhsOgLo7YGY1sbAl+Uypz7pZptUCaj89LQzigC5PLhpHiWvUiu9qIW3fu9jSE/uwguf/wVnKWmrJbSoNVjUhYDDHkG+ZpwmsFczUBotYtR9WgOtIUmdHNLaaPCIkHsy+KB1B9O7pSpja0qj/xwz9EzXECX3dALrt8F79/UQh/0euOJMiJMOB7rikFR/MeqAu0TzMAYgNQJS44kQR5Lm97owOnEBcksLgDaYoOYmZcB7bAXkD/8A+eenVR+izjbf1e4B8TSwqx/4wlcgbv4a5MIOiK0lAp7CS08Kl5CJ05glBBXMEvUHloTgxFFZpcYUVSy59hLEu1rxzEe/DSuTqVmDdaI4nNYR1vp9h9A/1h+griC90e/Gw8yuIP45fXstEFD/Wz1o/CtMDDKQjbgBs0vSCzygqRki4wErVkNedgPk/BnAO46BOOUogXnTJG3gS6xckL5H3cgq1EfA9FPUK9AIwh/+u0QERCIKSYm6VL9AfqMtu6W8/WnI3/0F+Psq5cOn3EG2XHx/PndjcYBMM0Du5yuuhbj1q5ALZwF9hAScws77oaiCcSWiqBsPN5uQqi8BDccWAhWu5MuYd/npKBfLeOGa78Oi4JlO9KwjwBB4AiIzHG9vAEXDFbz/AaJ50kqOBehLtDtYm2uBFzA8jDmhcK6Yufi+DdswD96IjGmg6BIK3dxJe7t8BxV1FkGhCJHNCuw/U4plBwFv2R+YMxnobCVrTmXd0DDcobKmJwklKbhJOrdwo1elvR2H8pAbd0A+9wrEoyuFfG41ZN8eiXgUgkrDGCn1O4d8JYQENDfaiaw5Bdz2NWDhHGBnSaqd6rgVDHdG5/ZClkDUshATAglbIEXKISmFtkDckkhSRXrUwp+PuRK5VZsgMmk1ab31XaOk0WDdONHORiWflztWv5ES6O8bqBkiswxfBIxzABl7BQVANiYQJIqE0pWCy81t5sLeqxAnCRCmPqOo1qOP9hVqAigFrFqVeOYV4PEX2GSUHS3AjIkQM6cC0yYI9HZBtDVLSsKkHckliYhqFaJQhBweA3YNABt2QK7fAm/ddoi+fm5HI+2I5Gzhrg7/9Vmz9AFgIq+eM2UAVTm1nIpY5Hkfg7j168Ci2ULuZE7AHaPZI02NT/zCGGqf61eMcf2gIyUjAZWyZRIRdB5/MHLPrYZoztQ4uqmXmUfgiNtbX6c31AHMG8PapBFz1ny0Ti8ITSqw982x9CRNJPCTAuvwzQw/m70G6Fre3Nr/8xOPyIHUQq3ZeKdJCNoI4u+rIZ94iZRayZ5EsuEpPYzqByzFvskly21nyBVL3kHenZTq/ylljIx7i9POAw7I4tzMcTTejatUfA5GjpymFoiREeC8qyF++3XIRXMg+kpCWpQgoMQB10czTfm7sHO1FBXPCpC+4JA7hoamDmt1FpRm7ZogDTg0ihHsGwJQbFYL5JCZNy5VWcey9UPDiBa+PjROOJ4QpvzA1xBGCO3m1IURptjwLRhijYmoKvPitdKNJRTv91OkwO47avIUuFVJThucSedQWJRKpEWbfmfTN6+VYF9ckXli+0jQnIWkcrbzPgpx69eAA+aCAl2cus6+D5IKyspQf7zbGNc38AbX/usXRnO+G10HfLS/xeSoOrvWALyf49kIAfaSFo7xGvy4+j8DiAFwwxsfGQAOFDAjbm2Oa3ICU1nTN/vArQ1GASJiwb6XLJCHEY7HKzOS/nzTiZVIX9hT11ECqB1X+QHsULFZCVM+du2P8Mfhf+kZUfVdP4sp1vfKBT4Nc6MtQgKq+RcQTS3cwNo77yPAilcpWCQkkbiK+NFmifz6tJF1xfNQ9uhfoOABY1JgwAWGSbwR95J7icXUwaGBJfDGCHCDXlwteBsD22Q7tbKjBiyn0S5YpoLSoORZO5rGHaGOF8EMfcRhQISTKrSM1oDUwDP/oooD6M91wDW9bsZnMy/C91OMR1YjiYbHUjEO1lUKZUYCQcrlhIRSAJSbl6NvRPEubShBpeXkL6h4KDdF0f/0WpQeXaF6E3Cls36m2WHcDLOE0//etFfw9SGAm+nEXsNc8/ERRtOzF5b5eoLmhE2XpQFkvpT5Yj03qUE+ZLCL8UAJXKc+BevvPkVLPmcF1B9cy9q1eY8Ri68Lw5qduMzzWiyEXbcq34CQQJQdyAuuBp55CeilzCTKWyC3Nl9LnahYB6g6Hoe9846L3PXf4YYYXPVk6nR1JXvmmoYsAs5xfkMECABVs5CDhQ/dq+WpScLh/WobmXYayHWmSh0bqLu+1kgujN3mpSaihkVUuMOWT61kjwcpV2IvThUTmfX38LuZ0Tf93URUPYbPUZgTWNzbQFY8yIs+BvHkCmBaghtLcBdxZpzKSHBb4qikBIrv/yK8v73A5WvjQjjB8pnVwvXr2FgpfLP9AsI5fHVUaALeaNfm7wyie/LWREWITdV91r8ZL6EbKgbPNjpg1b1k6OW0IhlWgsyjLp0Ntc8sRnwtXAOzTic1EdgUb9pHYchfc151gTSNBAIileGsYO/iTwA/vRtoEcCkBEBioScBdMThrX4N7jkfgfzDAxBt5ArXxMjKrwzW3nzHRnDeuxXYyBG0+EDRNGEFIglXtYrVbMywx8ODqF2vyXjhWs2aF8+nKP451L58HEU1OsyEEl/jr3M+hYNT+jkNgB44nQyLRNScWzXDxFCq9ByM6dSQw1BoTafVXt/D2BCKLAPtLKrSvoU5iMWzgSMOAncdo53MX1kLPLmSdzSj7e5rxaO2362dnWlBK1/OaTL3HdbP9PyUsGohL/temwes2/Ym0UDaYoP3+tG7OYXEgMEBfJNEFil+T9dTaRVp2Ere1e43U5ZDxMdOkwaafxhI/EJhRNFQMxtVmj+FLRVDzwicVQj6+dUDTIVXdRs3dtSYGU515XEGBxqneZu6kE8QQUTP5Q5koi3KBTJYuUZvnqvyAzMpiEQCkkBCVgoPwdEpNWeeCpmMfsIN+Uc4YGRyRv+3vSDnG+QEasBpM81c6BqG2XDRPSHLu21RIuTOwRJcSuRgj51OXzIhrhcx5Cr2HY/jFi5AnvBC6mvGYVRIQ9Zj+EgWlDuYJW2hLdr89/VKZd6DiL9TlTHVEgYcyae2OnOrQeXOOB1Hi5h6QClvZka1jKXdyaiopFSCpBZ1RFTBHC141GzAH49fI57wW9NSXqIhEsztZ719RgDKcPD5pN7JKaCcGsUQYdCeOQsOmIEn7/oGJevKWDSCk8+7Fo8+uhKRbJabLdc0dOVbknUAqe8nqDo71PxP9ZlFddEbpcQF26+bcr1eOVPw0M4TX+mr26TAUwotI6xq7Upje6USpk1uw9GHv5WbWLyybjuefW4Nl5ormBq1fIFnzndQ+WMpPNJ5efolTR2jNmc+U6kik7Dw6x/fiJ6uNu5tdPlHvoHX127l1jUkaml7+1NPfQtu+ORlvL4bN+3CJR/+FqqcPhYmHpP7ja8K2gsCBD7PGsmZxGYqYm4VScrObUpTnz5h27ZMUskVb7VKKkQtHsAtYKiLh6A+vhG1UYLf75eDIvRyVOHLbcQFX0M3Uncvzt8zLAYCkLrWg83UQdeZugLtLk6RNQmXt2Ehf0wMQlB2sLnRhVQsnuLxtNMXF5ZEELUFSiMDOP8Dp+GL//lvfP+9f34Kp5/9KUQSETi0exln5OjaV9rYQjWJJp1J/USuXNdP7lS9i+k5rkYWkyXTK1uAM5bDiScfg7edclTw0+UXnIBrPvnfsKhBJUVDKyXMmz0VS8ibCGDWtF4qcEKlQkTp5zoYeZwBElD8YZ/yAar+jiEGJim461i9qcmrfv6MNtRy1SYnuZDU2LkySN2xqEsHvZlahEx7C/KlEqr9BQ6UKCtMJUDQdi+iJY1MKoZCsQxngDZppKS5NFxyozLgSMZFOIBjZxJIJJPIDw2pfntUbs5IYPG4zmCOLZJkS5L2oJHFwUG1J1A2q7ihR7l9NqdfwSkh1pKGHYuiODTALd6o81iVtpL1JQdtRuGN7kGFGCRvKumvAcnqStWP0CnEcMf8jaoySaTTCW5KVR0YY8+j3dKsmldq3cYPbDGxCBfvPv9EXs98vohkMo7zzzwen/3aL1AoFRChNrUUQi4VKbGFXYfUBo9rEHTGMIPI1/GUbq4RrqEMGI8AUTJUA7D7nKqWqlGHtswu68+Wi3m0diXxy598Hol4HKvXbsEf7/4rrrn6XZg9Y5IYzRVw571PyM999eeoeja8qoPOjmZc85nLcPyypcgk4yiUKnhlzUb85Nf34+E/P4F3v+t0XHT2CbxoN/3095jQ3YH3XfYOpFIJrF23Fd/+0e9xz5+e4HRqbkdfKOKcc07ApRccL2ZNm8jZw2vWbsKPf3UP7rzrSVgZkpk23IEBvPXYg/GBd5+G/RfN4h1DN27egVt+92f89Ls3c0UQ1ZFQZ7FDl+wn73vwJjzw1xV48eXXcM2H38mU/5NbHkA8IvDxq96JBx7+Oz72wRtxyHGH4fKLT8dBi2eL1mxG0sZQr63bIn548/3y/vufgtXS4qsgfrMo6nSaz2Pmgqk4btlSQVwjSbueeJ6cOqUHJx17MO747YOwkzHe4ZQ5lZ/sYnH0SjsQdLKo2ZdBM6p91QEUQYcAbVhMdb4BQ/v2EYHq5lrbszj1RMXGjj7iIFz+ztO41k4fixbOEgvmTpNnXfBxZNua8efffx37L5pTZ3Xvt3Amzj/reMze/0ws3X+OOO7og/m3Q5csQJpaqvnHtCkTcMKxh+CKq76IH/7wD/xG3/ry1bjq/eebqClmzejFaScdIT/12Zvwha/ezE9537+dge9/8xN1bztjei8/a+GcSXL9xq18jkrNJk+ZKCZPmcht5miPghOOO5THPWDRbLRmm2UkGsHKF1fh4svPxM9u+rQvwmrHgnnT8Y7TlonLPvgl+bOb70Mk28LtbYltcb/gfAEXnHEsrxO1tKUu59FYjN/hPZecLu+4/UG1cwplPxt6U5AcEvhqjMwtLTYbW+9v0io2gEXtTys0AesyvYGB0uUXWLgeb7RIQQ56qe3bd6JULHKDRtrM4cy3HS32WzgBy49ZzMCnc2vXbsDnPv9tcdvv7uPFeebZF7Fxzd8hZZVldz5XEOlMStD4619fT2JH0Hm69ttf+Sjae1pxxtuOZuBTh/BCoYSbf/1H/PLXd8pcrsANoT9/3fuw//7TMWfORPH9b35CMBAoeahQxJZNZCKrt1u3bqPaEFJRGUZHc9i6rQ9rVq9FqZjn9yvki+jsbAMBnw5qC9vXt1XpOFUHTz75vPzDH+7Htq07uEUtjful669AS0cL3FKRu5wT4JxKGbFsChedvZzHsW0b/3Hdl+XI8Ah/P+aoJWLuftNQpcQXjhaY1OxDVuvtdSav+vMbfZj9AeTeEcCqjFFOLWtmwe7Qjf03Pgr655W5wO2PJOkDtHEDKXM2vvz1H2LhwmNxyjvei0KhyL/RsXjBbPRO6BEEGEq5zuXyeOTRZ+TVH/88DjvqHTjr4o9zJ+2obfM98UQMw8OjOOUd78OiJafjlDMup3sICUQ8HhPnvu0wccl5y5kBUufvz37h2/JdF39QXnLxVfjqN38kLZXdKy4883hx+cWn6emLVa++Jg8+7EwsXHImPnrNV3HNdd/C9//7OwGnoX7Djzz2NOYtOgXXfvKLSMajPB/9Hj//1R9wzPJL8ZX/ugV/ufdRXPCuj+PI5RfiyCPPwplnvgcnvP29okwmpeeJrs5WzJrawfsHkWlHjcMoo2nZkQdi3tzpvIgbN23D9276NVa+SPsoUm5LDBeffQJk0Tf/DM5LYOKbWK/wkSPwgRhmqUddInmAusPkUwqS5dFd8Lw9kHKSciKEgg/1CGT4DfxvHMBxA0wuFkv49vd/g5GcxF8fegavrFqLww49iH9vaWnB4088J2lfPjoOOmiReOjBX9Mu3eLZ51/BTT+5Xf567Srd4pUp6657/ooH7roPkfZePHjPQ/LOux8Sl1z8Dv592REHYs7MKYzVRNnv+7cLxfuvuEjSPHxOyFJsyf6zAxYdidj4+rd/idUvvQrROgHf+OYvOeRqWdk6Cee4riyM5Sl3m0rL+J3jiThefmWtvPTSTwJUxMqbR7Uwkr7/ikvw7a9dh1QqydTPXMJftgQ1qND7FtJIbhWXX3hK8Kzf3nYXnNIWccttd+OE5UfyuYvOOxVf+a9fYqxvR8g7yUdAjHXOUm22MXE6w8DmsTdDAAHsysOd3ccIoHYlqg3ZwEUadrqaqQeUA0/pTGyqJVMQsqLknn9ksy14/qlH8L4P/Cc+c92H0NPTxXfGEwkcecRS+hM7+3ZgZHQ4uGeEdgFHjMXKGOIYHatthJRIxCSbW/6EJ3R3Isaaszr6+wfR2kr+CYmEj3R0DAyOCdvOyFgigSI1lPKXgtrSBYekzaATKnuICkv9Z6x9nURFBIkJXSju3o0bbvwQPn3t+2r3+XcT9ZPoskjl8/u2kPJXLZQwadZknHLi4UzF1MD6zDNOwvHHHoZEgvYuJKeawPRpveLEY5fijl//UmrOwwObn8ZZav4X+ui6VNnqBgVAjUXAMn9VvJf9+LHvE1Bsh+vofGUj6PVTlxCJUARN2eysGQQNMGq2KG0KSQkaK19eiyVHnIcjjr4QV155HVa+8Crv8kELdt5ZJwWbVRDynHziW9HeOwFj2zahbWIPTlx+eIBUL69aj527+vk5RPXX3fB1ZNv2R8eUw9Ez/ShcefVnsd/S0+VZZ71PjowVlNMMwAXnniypXqDY14cDDpyPU04+Cp43yFvE6tUkhHOKo3CdkTobnquFqGI5X0HXlF58+P0X8ryJ813xgf/EtOlHYu5+x2CgfzCkGKr9kJAv4Jy3HY2mpgzKtPFFLIp582bhkEOWYPHi+dTfOHjcuy98GyR1mPa31fNXO0SMhpKuJLhq0+NWnlV3PGK9aVq4dItPCOlcVjvhe9/q8c2fQJ2rkCt3dSUNoxJvnKRDr9QGroYc+UJVLDxoKZ557Db55N9WiP/+zs/x92dfxjHHL8PiRfOYRkpltX0MHcRKZ82ciif+cjOefOIZHPnWwzB71jSUimXYSRu//M09yBUdLHvroXzuox+5ArYdx4aNG3Hc8ctw4bmn4YmnnsP7P3ANfvv7B3HKSctQKpVx/tknifaHWrBl4zZ55lknobW5Caef42C0UOXJkkZ+1BFL5U0//hLuu+cRVIx9j5WH1WZ/fXNrK29SwemHlhAbt2yThAjvvvxc0dqWZaQm3UQTCXPHdALvPP8kXjviXhs3bpX9/UOqPN3zkM6kMX/udEby4487Au2T5oAUVvVwk8hq5l4tDMPmm9qNwyk9rq5WXHYvCPCov9Ijf0G1OY8Ytc4OSoRrR33gRrkNfU5AGzxxRyQd6qHYO28Dq3LkLNIMDeT4yFWX8ucjDjuI/0JPkbfcdh/eftox+v34mDd3hpg3dwY/g3cUScblT2/+I9a+ugM/GrwTF51zMhYtmkvn8YXPfbTOvDxx+VFizn5L5c0/uV2+97LzxBGHLubdxZYfexg/wyU3nwVc8/ErxFVXqyypWJz6GEFccfn5mNDTJe9/8NFgkrafQUQ7lG5etwmvb9iKRQtns3Xy4N0/Z+uGlDiNwMGbWWT6FbHsmCU4cP/5PqJV5XFv/yA2vrZJNayuOkgmLPny07eLmTOmwo7bOP/s43wxqHNfzKwkvzIqWH2GXUQ6xe3I73ha3XR7XapQmH97LCNKG7bALT/iq02k1o8PuvgpVxVqdkiBAJ9K6Ttp9R5lOXqeJA+hy/0BqXZakAkYYGA0npY33vBf8ne/fwAjI/X6yY6+PfLfP/Q5PP3Ys2hr71TXx0gbfwaPP87vElDnd394G973oS/DzrZicDiPk8+6Wv7x7kdQKpKIUQc1iHh+5WocffIV8v57n4TVOhFvO+dq+evf3KuifBpLpBR/vPthXHjpp+Tzz66XN3zhx7JULMl4PMYX7ekfRtWtiQYGKhlYkQiqXhT/duWX8fo6aiendv8g4P/61rvlCy+8KqPRCN9DreyVu7IamH503PPAE9j42lbYbZ0QkSSiTVkUB4u49fd/DiB1wdmnorkpEyA1cREuAOZG1Ga9Bj9KuQOd8l2s22GZ2rQwRGWhgy561EFy/ttF06Q/wrLJNaRMgSBEoG1M2nRZYNa0Ht55y3MrWLd5D9ffzZ7Wg0g0ikrFweubdqnbnSp6J7Shva2VvGxia9+QHN61m7XhaXNnYsGcSazcDQ6OYMVL6zCyZwiiXMB3f/Bp8e/vOZdn952bfiWv+uD1YvkZb5fJZBIvrXodG17dyHUBRA2UMkVRPJTymLtwJubPn8WyfNOWHXj+hbWQJRcW1QXQ/yoVoFwVcxZMlfstmIVINIbVa9bh5RWruXuZlUrCGxnB9LmTsWi/eXCqZTy/ci1IMsya1s12fF//KPp2DnNpOrPt0VFk2ppwxCEL0NKcwfqN2/D8My9j0vRedHa2o1Kt4vVN/ajQPoiehym97Wo9vAq27BjCEG2rx32GfDjRljmJCObMnMTwJH/GwGgF0yZ1CCldmStUec0DTuBzYr9amIIhQuZ2HoLCy88D59jA7YZmu1f/EEsRW7Qc/jxSrYu4Ab5GAp8RMy74Hbw5fKmeqvre0zUEBF0mTjtu6WggLTopbXRtNAIrHlOWEO0JWKLyXPL3q+1Y48k4yrt24zvfu0a873KFAD/71V3y3y6/QeXsU4whkeS6fp0+yBW9fsUQt38r+2NS4+hUEjb1DuYUbE+JJ1ojenZZUTLtBGbRvgHcMUTt6sUhYS4Rp87kCTU/up7zHyKq1Ty3YaExBe9lSJVKrCCQ2GtKwyur3gJ8JOJqM0yCULnsb37Fzgu1CbVJaISoOkSsmTCZlH7JOndEZy9rjTv7l5FzJYrCngfk8NMnhbX/N6kNPJfQyJGV0etFLPMH1YzBCH3qbeQYH2yIdCZ4LCt9hC20r44vl4jFag3VSiSDrWYVIBS2Wok0rFTGD9d6HEnjgkkrgkwmEzhdUskEjxpjc47W2OLu4Jzn52Nm8Cxq904xGj/lmvfrpZn6eOlRLINgRHsPZWoRcKXYUYdvpURb6Qws5rrKNGSfAvUfMsZVWUWq44gVS/B76nA1RUEt2nQqWA81P/Xeej18BS4UzeUOo8RdeD3Vm7E5SUTl92sIElX891YtEyi8WvJkeew6df7VfQ0H00Fs4hwbxdv/iGjyPqS7TobnVKihnRGS9+dJAAgi2ipXXy2fj43qfBD80gEqPQAn+iikUYktfpGkv8U6dXh48u9r5KTeCbxAj9J+Q/EkHM8HvJm5a6Rn6T7HjET6QeFKJ6GaT6l0w5COY1YWU5dy6hfA9yjK5WfX6UQKWfSm2sH1fqyX5LPfMdfv96Ois3yVAbxGpct+3pX6zMnjlDVsJr+wA97EG/IwxWR59Acornq2Eet/ExHgQ5amFp82VaSnrkAsRXFMmi93fGZAmWwnQIiABSng+/X65iPN2KJaFG1MahNGsdPAlqVGz/zRVSHiJJN1LffffIJGBJ3PEuSThBa27vXlXlKmzAyf0IYY+ll1bxcauy6R1sxLrF8RE/V0wK0WglecIjC8AkQ1HXJmwwiKFtkRVEa3yIHNBwKXjvg1H94/un28B5xro7x5k5TJ94jmiXfAilLrDCJRJWbNlGr/u0IMjcyhBdL2arBORh9ZH1KKypQ2q0byYKWb6hYr4BQ6e6gOyWpz0guppmGkltXhgqx3XtVEbz1oDOdLPQA0wtN/FfWLRvil7fVw/DxwqtWWoQ5PwkgzrjxPL3YgiwSqBSnz/e8EtgwBr3KB2t6A3DgtPDiIbSyLoLLm96I0eL2QkjQ82tPVf6VaurfJkPZ+hCNKfl8982V8LVY5ERSLDypmeSNmtQi1DNiQW6IOeL5oCtLMGlE5fPZtZh77LD2EwGqKoUaZJvjMtkdmEM2fb9Ay2/TUmojnPyTMV4I1CjjV+CIdxSYs2kc3IvNDV6K0+nFl0TVm/caU9+XwTcPUAV+3mnquJgOKW3hyPriRAm6kJWtOpdyWAR/20wzlXojJ5CbGIppTrQ1s/GQ21AsPGrpXc4NgLBmqZKpRVa29vTmXENcLxUaCtairVQgtZ7h4xNQB/N9rXEW3g9Oqbcjfr1iiysyRbkTmB65F7vkvBTB7k2MfEYCuO8cibBJNB98gkq2fltw7hWqwdVenUElYwFp1UYRxVid/hh+xt/qAoMAinN4cYtv82QROmD2Y1xhU1EiWN0K0uueGJXd4IrVkUDUFndZuzr/RfRoBTKd5SEepS4NnGESpH4EsDn4EYyu+ta/Ab7Ay+4YEVmb/98pYy38hSnnSlAPOPgKlcdUBwkAKk+qCdWvQZSSgIBN4jcjIVMIMgI0TM+a15nljUevmZsrYGvHVT9OU+42pu16yawQw38dYizAu1aWaG8pfnVLJrIEoP4ZKblAWxt6L4so7/hHgN1ilfTl8kyI69UCR7v0OYs2HqzoTr6KrG4OhA/lmNlEIaa/8e5gCzcPoOVg3a+NeUx42fCvzWn1Os9SaSFIzMptdyRBehLhEMLb52Uiq1mOEGUXd88Jj1FN47ZqgRE8B3rJi/L08+pDM73w/Khte/0eB33Cp9u0I7MoImg76iIhmrkY01eNP2U9RDTS1ek0nJCHq3tS8tO4nI8hRd425PLWGzwEV1QEt1K5G82c+avTccEFCXKFG343EQPjzXjhLYKHUycba9YpDmTXz9KeqTGmJq4X1opL7qpdb+QN1095t/f8FBOAjqC8DurqtzJTLZTR+ESLxBdx4oSavfdQNWo77QApwoqbT1HhyaF3Hl2Yr2zgEMkOTbjxOGHn0YYwTUL45rkItw8lYY8W6ZxFfF/w4DgfqD7OqePwUFH5xe1BtOqhDhXVXykrhJ8iv+AUAahmiF6fx/vD/iwhQpxf436NILz5KWPHjEYkeBhGZA8vq5HYcdZ3EQ8qcYaYFvgODaupYpeE2VStVX6FTT1VhGWuMHSipGC9Gxokr/WyM19KDe7WoC5ex7x0T6hU9Q9ng6XHKGHUl3AnXWQu3+pSsFv6M0qq/1YD9P6P6fyYCGONQNlFY/rQ1I9Y5AZbdCUuk/D3SqXMt9Svw1WJ25HvgzdUpZ4DzyKhHq+v7lQUiFEXRSSZ1nynKI7mvChel6JoGc98jCixw4N6HDv3517mCai79Q1pwfGhF/LnxuJS0749p+2yL76MKELW5ZO33WiKMeh//N5s6PrG/WyjXG7UF9ccOWCHlhutokUUVN8OQpd0or+vzW5cbBwM+3Azg/xEHcQRbKSPj+9L+6/ifHhTJozWltf2nES0f/9tA8nnuOf9Chv/RcXud3fev41/Hv45/Hf86/nXgn3f8X6NeOAuMJm/zAAAAAElFTkSuQmCCiVBORw0KGgoAAAANSUhEUgAAAQAAAAEACAYAAABccqhmAAEAAElEQVR4nOy9B9wkWVU+/Nzq3P3mMO/ksDO7szmSdgEREIQF+VCSoiIgqH9EBRUREAmCgGSBjywIisASJIeVHISFZSObw8xOfnPq3FX3+51z06nqnmWAwfD/tnZ73g4Vbt2655znZOCe7Z7tnu2e7Z7tnu2e7Z7tnu2e7Z7tnu2e7Z7tnu2e7Z7tnu1/xqb+uwdwz3bPds92z3bPds92z3bPds92z3bPds92z3bPds92z/a/efu/2bjyf/O93bP992z6/7aJV/+Lx62AJyhg1t7DBg2cqYGXakCJB0U/6xO4VXeIOs7n7L4nOnXHO89Pc47/7u2/Y6yS1n6Wa8t5v7vx659wDTkOrYCXKuAGu+5ozdF2WfK/lTn8b1mBtEXAgyIz6ZfFJ7B/btATHEDl6m4o362e7PfHO6ecz0Hn+UncILsf7mZ/+dug88pxZ8eWPZfbN7qbe7q78f+ituwzOZHteM9v0HnU3fyW3ecnEDnt9ni75i7Tdv//8Zv630H0X4/TUp22U7ajNL4b5dIe5KqnIMpvQRSNq6hQg4pqgK6a/egwei4Jve1pKK2gInfnGlopRApKKU376kQrpegvEYPSZg0lSquE9kEUKSBHp9GgQxJtCEMnOUBH2pyM9k3M9OZyWiU5pXQMDWJcCqALKGi7ByKlEakIWtM1E2id0P/0E49FRZGmUdM98LKm6/Jgzbn4hyzTi8y4It4zBygaH703DIH+o99V1NPQfHdQugAkChpdc16Vg6K54tMmuteN0etolc9HiKKIz0XnoZG4eeNz831r6FghoUfH1+F7pvk116Xh9PhmEPFU04RYZkXTTtMIfk66l2ilNd0LkCS0EDSiSCsd2TlIaCx0CpoHnhk+1synmQ76TScJX4I+milSULkcVJRAx4lOYn6sUDnewx7Lc6Ch2yrpremku4QER6F7d6HXugvrq7ejd+gOYPZYen2+hNZtBHwj/p+MDv4nMgCC9pGB8y8LXLR4zl5Uhh+IfOX+qli5APnKLuQrI8hXgEIZKp+3sxwBKk9EBV5YvBhpbTOV8qMwa4UXXbikk338keg7DIeZANOD+ctrhH6y5yJ+Yq7lzwZar7zI+Fg3LsXshk/AJ6H3hshpGZv3YZxu7LyvO487nSaytdcz/MSMMwNqDFGa65nf3Sb2i4hrGXpGQvxH23vK+Wu5c/i1TL/zKcKc0u+GBZjz9gFv+p4I0H1pCN/yPM+R+LNKDGvmU9F74pbi3O7mzTF2bG6fvmXNzD/cOb8njid293jJzIGfI3Nj5hyWeUDHgO5CxS2gvQ7daSyi175Td5tXoLX2TayufBu47mC44hNyfev5f8im/ucRvoD3xdNPQ3ni8ao8cilKoxep6kQZxQoJVcP4WaLTqmNqEPdihDovUnvqLJY3hOEWql1oKQq2vxBx2lMC+TBUQ+VWmEvukTqBH44cixuFFZ6ZzZ7T78tSM4xjgFbLOEBe0Qw4nMeNVT5ucRJ3756B+H0j+xsJT0f44R7CnQTWQ4Ph90J5MMxUXNtAH60F50vNBY3JMgo+t2AwaaXA/eC/8IwhpU357+Q0O2bm7tX+dczZXEtwTdYAAgc0oNE92Bz/FHeB5gp0c3FZN1f+E+31L2H50CeA2w+EJ/XE9Br/b97+hzAA4pB+UhSqFz0SlfGnq8roI1R1uoZSDZpgMlSPVyLBc559LfQCQcws0bOEmeYP4W9aPfbMwUlqtzj4O4LCgjj9aQwq6Nvc9f0h4lqDtFGHABxFW4SQtVg4oT6A/OxXYWxhnLRgzQmyt5UeikAlfWLTHZC9ntFKdGYu+TvBl831A+0yYbv7dL94GveyPf3OoxE5rqw2n1H7U/sKFKMFs2Wid/tLJuyOiw0/YCRlVQjeElIq6TZJLWFOrTQKiHtAcxV6ZXZRN5f+Hc2lf0Lryu8cV9j9/5QBEI50Vvsyhu/7myiP/x9UJ++jqhNArkDT3VU6NvjZYV37cKzWLojYg27BADLU448PhNFvCxQE5OB7an8MOF7a2uRv4n1KeomtD4tKBiU2rxZIiZ89mTMNpE/uEITna5DMS45/wFg9ghhwgszI3bu7v1UzFlaTJBKxIldOiZTP4bqOc0hCzhwg9xeE7RGFvSeVmjrBHCTK8yjCIPig3oU7NLvafUguRSpRZAeJk6KOO0D9GLA+/1XdXH4z1r776WAneJm86P9vGIACHpQDvkFWIKB2n99S1YkXqOGZc1AehVa5LhBrJImxxpAdykllKUV5c9w6QGwG7sSUo9yApSgZQPb2gx7vP6cIeNAqy6yVPubiiM8gk8FEK1UUIX2kePaSymBqfy4vTgXppQjCnc/ZC7OMw8F+cw0JyFOkSOdhu5yVx04Hd7YFS8npGbW/yaln4jsOe3BogeWBTjklAoGFUTvi9Nfgk8v7CuNIS/g0U9PeduHuyQ3FMEfLriyfN2igf/zZMVqbBJ8gSpSKIqXjnF6bR7Jy+BtoHHkl6ldfPgAB/1/PAIylm7bC3nNQnX5VNLrlURiaop+6ZDm25mQm/AC2jES3luaMnJHSkpUzcxgvWEcUx4PnkkgtxPfSwZLBcRgG75b6SjInu9rtd4aWs9JTSmoMlPxeQDrDmtTlDRX00XLYMtKsD1xLorNsyPOZcB13GWnEtKPJagL9Vg2p0oST/0Shl75vSbR2dPavZ1QeEWSZmb2Os2Fk1BrtrycJ13x2Zw+81JibjA2DiTujvkhJQKoBe0c8M1CIYvYixd28Xj4AvXbkA1g68iLgtoP/XWjgv5gBeE6Xw+h9/xLVmb9Rw5uHVL7Q1UmPnWxGOjiCkxDeEm3KsJUZvidUsmDbndzC69vX/GMsCVayeR0/LLCwCN1LMAQpKwMD6ifEQUTu3meguhybP7+7rFjSZrxGEpvvHbHJa2XhbPguQ952+sJ+kjTCMCXKct9ZaG3tAOEKdlQSqWc+9/GrFHQPxkTjmrPX9AxAIBC7o5+dPnXFSefsVCT2HBZHyHvP8NVwn2QLcHcjmDGfSyCYrOfEMQ7HCGiyO828XjlwRK8deT7Wrvjgfwca+C9kAA/KM+SvnXs2SlNvV6NbHoDyKPl/YiAmP3VG0vZDacMA6H/rsvJPMuxnXEfC+OcMtWylcVDPHeMWtyF+Z8TKmL+C7m31Zc8QBIwcZFBMz65cLJkx9JNCijk4uO8WXlA1nKS6O208fV2zfxZFZBCCX/yOYoSXhL+ybnfexUi4DKq2h9k5l0DBSmaPF1Lz5w4VLs7UDFoid+6+gZZXe27BdBzTCL7dILUVE2VGeGfRkUcRVprz124M7salC9I6ppzr0COTwAgodENH+Z4CilifQ7K475+xdPVzgJXl/0om8F/BAILFs3LO41Hb/G6MbR9ThXwbSZJ32NZb772eH1nJbLYgpcVDSdkD5G8CSst9hLT1RjHLNNJxNJKgBfy0UtLovGEY7pxmEWXHMGBGhF7P95WFxF6fFupP+mL2Hpwky1CLOL+3B3j9VTAHL2Izj8ufJCAex/SMy1SoVBbupsfpRyhsdc6TIhx+xwW7/Z4NyQR1Sp8fBLUycF/Or9MunUSHI8is8uKDPOyxFvLzIByszw5bfiEMh+44isNy6oFDCQRgcoUE3UYhmbvlx1g+8Hvo3nSlF5j/yxlAeCLVi16kRje/AsObaCV0oZJ8kLiBUJ0KYAjN+qH7/OzuzEElcL58AYj7mYJH54JBpDChdNsZUGJPnjmd04UHwPas/uklqziZJFC3eRiaIUgh7YPRiywdRgr7awffdfoY/5tdgBkAL5HG4CcY7lM+i3TMQMal5wklzLPhNeLZSMYgjKMpFNG3r5yXIFVdQKKfe3sSf4g0zNn7DYFhWvzmVAmJqtz3aU9Aivn4RxsCrvyq9EzGqRDEbAwacUyI0UMu30OSFPXc7Wt68cDT0bn6YxYJ/ELzDH6RDMA9tjxG7v9uNbrlqahMdDnUFNpFmNhHHqUXidP1HZF7ypPQMO0V0H3QUui0EuJmvQjeuOa+JeRBXzv7Q0Yyk3uRF4kMEArXTUl+KfmkBi8RgIfS5h8f/ZaVMANRghyfIEQv7cJP3tKdXUt+2IOWQr/x0M1hGE7gjh5VpL52DD3cr5kSga7E5byRU/IoKfn9eCQKYNdbuJ71LAdmP0Ba96EHeD9/3/mlKuEfVmzhveMmzh7jFlOaaRvp71idi0wMv7FhVUUUYpjXyweVXj70fKx/7x9+0UxA/YL9+3k1cslHMHHKr6M81EbcywdTcyBglvRZIvXSQzIACXedy26Qa9D5mG3Iv7PAp1SCwbDanE94A6TGLG0UTsIJkeaFr2VkgegC9A065nGmXkqTviEKd9ZAorUWbE9I7msTIt/n5rT6rzmXCe0NI5OxEGG+nH0lqB4ZBuj9+e5rqxrJsWZiGYJrUXg+w0QGQhHjMpI1fR9pGO9IXxwniDpclzY5V9KdZyNBs5CEzFaeeN26tLaQVAxRBj3IQKQ+I6E5nnMeyIe9NpfXS/tfhbX/fKFVB34hOQUutvVkbkHxHL3kXzG269dRrLUQd4qOwAKPddJBSk43oQK+9kevpCTTwEEI4s2ugeP68510c5Zn/s3FjIc8GwpGNALMwmOHTtwedtxGGpsIPGnYCptYII7geD3IBRmkjnFD2S85Hr0/vNifjt8EF5TZYoNgUpe251R0PkmUaQhvnpPh6ynCc8/U6fbSz28gkoXnAZVQOpbzs2dRSVqNFsTvj7fzKrmElKZOFcxA/GwmhHm+6GNGadWEnrWbDDM/qXmVIM8hMD+s9MM2y86OVaIMO0Zv4KTZo6Sk4ckOIvUCyqjC2jde/IuyCeR/MQY/pTB08T+r0R1PUKVqWxPxc+KH32kw8aaCaOQ+EhZmJbgLXU0to8HnZ5cRGd0yUDLDTHzWHQMZuQCcFyILcYM70PEsJ2GMscfmwDjY6/fO6KCSx3kpJBYrn1hIHx+QIrG+HJwYDP1DxM/5iO58duGmpK04padvO9/+GQqis8+EiDqV7yCeFRO8fAzWrhBiEWzkZmboKY4k9k1Z6h1he0u+eeMPS81QVhXQ/kB56445hXk5vvBNGXAFs07Nv7y+1yNoUWTHZ95xHEwS51R1oqug/ybRUQvr33jlL4IJnGQGQNF9l/VQu/d71PiOJ6NMxN/Nm4XBTzkgAA8LM/A9Q/wpo14KuYuJHBDn048a7B8bfpreX15fxsEJa6+XatmlZaC1W8wSbLhh+mELKRjCb8MC4XfMKXSaGXlDV9A1A8R1V5Uqh/tOMEJnI2EhblFNH9HJhSvcXGS24TEHlcoDPUgUZLKdvVznuSJjqljeltI4i8NnBhqzkKDi9HAkQBrg+nPG1oAnJLJK7ShUgYy+IecgwyTs7Pon4gS8uW1B/PJ4JwmOt/X9JoSBs6YmcaRqU12V4BUa3WNY/8Z7TjYTOD6G/qk3O7Dy+X+JiVNeq4Ym24g7hZDySVtIzzUfiXCc0U9a4DNMIaseZN1//pDAtemcjiCC285iDwqg8dZtFztgPQ9ZNCGkecrr4OIBUuNNB9H2jW8QMBGuOSNpkr7FSb7qQOhBn3SKhx+flT6e2ZDfOjaZfDq2BMffxxnm6JCXfeWMtFe5yKTlU/Yl51aY+fGMSrI7YbhNu1bJ3kvPVkZe2iCmRLhV/RwNcOfKifM6s5t2wzgCUMgGfwtwLcFWKuUb9touJkCoJh7uC4Hjbz2Fa1LjS4sJIe1VxiNgg4PcfiZuwgkKjiPQKlJaLx2Anr39EYhv/MrJjBM4SQzADqh09sPU6NYvYmQLmUhD3K4kVrnY+CtL/NngncDy+4/3OflZQ1V28Q24VbF/IBwZ8y/29+4it+UyxCz1AScJ7ecUFJAnHsT56S8RpUMcYqWKQBIviYiwPbGZhau7MXSnB9CLiL0QAeUiUCsBw2UUR4ZQHq+hMFxGvlxAvpg3tU3olWjE3R66zQ66jTa6ay101jrorqwC9Q7QbAPtnlnE+Qgo5BAV8+a9hV8+EzBlaLUxBDamw9xAlAZoMsNyECIUzyItrdOqmz/f3UjdEPM/IKtQGZ0nmAbcPpIByLdphGVQgTDqpRaOgwtpz4U5huuUhHwDv39wOyqzQ07P7z+m5w/eB/jxgVRI/X8zA7AW/9M2Y3zrD9T4to1sUSJHtQjl9VBeLgj/ACl0V1jXcTzfv4vqE2hAPAQXB5CSICm9TCy0FJOQIcZSmxdiO3VNd0xqcOJ9ZhH6xWQlbyopx+7v/MGeAcjIsaBH8FUSQ+y61TYEXy4B4yMobhpHbc8mTOzdjPEdUxjfPI7R6VEMDZVRKxdQLuZRsCsnZ19OOSNM6XwFcZKg1elhrUFMoIGV2RUsHFrE3L55LN9yCMt3HEH38Cyw2jAHlEuIKmWoQsF4P3zcfTrc2qkHuo/YnRBw8zgA3UnpO2jVSuaQMiRLtUYefByPisf3PNE+YzGoHwOI2as8LqjIfS/2Tz3xzPNldGKrEaXGZtEEI0O20hb07C1f1UtffzjwhJNSi1CdNOk/cvHn1MTuSxGpLtV7Ugz9M1V3UlLfSh/erG3AC+HMsAagg7CQBtySI/BBU5NSQdyiCmHBMjkmLKYMg8mqCbyPZFjJcaGh/8qjG6dV2qo3UopwVRxKJoug4hhJswW0ukChCMyMonb6VkxfuAunnLcT06duxtjUMGqFHGpUS0UQN4WkJjFlptoFbYC5mXVn1efHocA1gJjPKuSiXOqOugBaAJabbSwfmMPBGw9g/w9ux5Gr7kBr/4JhCPkIqlpBVKAKTc4tZlS9vmeUEgiGCZhpd3/Ns3fv06pOv7R3PvxwpXBU4AXuGDvX9nzmnQ3+SiEy6dq0btO+4KGgoBkiFgzdqzViTGlXh61n48bjzimEgWMsUdRDp1lM5m5+Bda+9+KToQr8nAzADqB64TPVxCnvQnm4g6SbZ9ki8/GdXuVhd5RO2BkE+RyVOKmfZQDexidUBX+NDALI3q1UMfxiyrgW+9SIrDTK/JY1HGT1Q2nJF5LAR4pluD6fkgi22YFutIFqCaVTt2D6AWdi6/1Px9aztmF6aghDltj5VnqkeSWsNZDdLlYk2RUTbpcK/WmgR7YlwYPkK2etwkVoFKGQ1+YzVWXIW6HHc5yLEJF9wJr9Fjs9HN43h9u+fwvu/Mb1mLvqNuDYClAsIBquAfmcsUEc1+5gyo85xu6myYcdyySwLNJy0th9ZK+DM5ra/H0RVh08QO4YOrebDenFsPaY1Hjl8xMIwqlnHlCE4J7UOu57/u42ksEIUDIYn1qcS/Ta4QgrRx6G+hVf+3mZwM/DAIyAKZ11ihrecqUa2zqkE1pqLo3XGYXEEnNuMK/3S9iVfu9kY9qMIJjKAPFuLpWRNlno7s8z6PazuqhQHzJowI4u7RTIgAdp3Okjbm/NTye28LR0u4hJmtJ2yiZMPfxC7HnYeTj93O2YLBVQ4mskSDox2fhAGSUdSi7jv0ZK06tDBG+JlJmCX55GxrqZdKqZk2Pu+0hr5JRCAQpFpVHWQJUKNsG8LxNjoP2IGeRzfK01AIcOzOOWb96AGz/7I6xefTvQ6EDVSmw3SIip8eUE/Ce06I3FIsTaPyv33NNqIT1nug/PQAUUV07dkgIifTBS0J+/E+qBf5Cm9quZFReDIDwkkhnYJ9+HBDLLK6h1gik4Y6DT+/2ZBNOg32kCFfJYuutOPbdwAXAlTfkArvILZwCW8wxf/K/R1J4nI4o6GglV4zRjZuiW8Q3zHxnm20+A6aQgu89AyZxhAo5AB0kKEWZsfpGW+vRvaZVlEKu5OyQgtAhJ8Dw+I5HYnuMYIUPyxJcw0M0W9HoLGB/G0IPOxt7H3g+7Lj4Nm4bKoBLHSRyzlE+UQlsprANY1UDDErzxVodyCFLzDPBXDZptz9u8gHTrjmyEIp2e7Yak9lOoB4ARaIwyM6A6wAmPLU+Sn5ABgNuvvQvXfvYK7P/0D5EcOMaGydxQ2chnQsiMFI2r0DwXoRYOEiDBeihSrx3UNvcY7iHDmV0wlXi+vKWkc3bLficNecfZxzN2YWz0h4hgLj/moPpxcqxnIA5YCjWBqyzneipuF5O5216A1e+8+udBAT8jA7AXzJ9xXzW5+1tqaEqFQh4ZaCcvk/IASCnrkIGUsBL2ZRNJBkP7tAaYZiBOdwzFKYLP2UtyF7RiS4159OHpJntuEQrqx2/hpodzDhI6PU/abWxMwnqToT62bcDk4y7G6b9xMfbumcE4XSmJkVA1bkv0a1BYARG9gfR0prQkD8NJzY2++4c/0HEiOEK4c1HXyD42un5RA6MKmAQwSupEQiXyFNsCmgCOzK/h+s/+ENd/+Fto/Xif8SQM1zyxsg3AJ4G5NSKMg95O46C6y+ATdpXsFjifNR67qDu3HlT6eXrENmjCnBTORIKJ2I4ghzMCOYsUBtoQghoQvBWZfYKNKKHS7Hr54Jw+csvZwC0Ldu/kvwoBGDw2cv9vRlO7L6a6faxC+tJXknu7y2SMgPzXfW+J0MOi8L1ZhG7epbEus9hTXgF3DXF9W84qHX1nfjPDCa5FD+szFBXQiTs27e4Lh1l4b8tJpcpWMwc3hj1db0I3esDZu7H5N++Pcx5zL+yYGmIjXtLp8iHtSGEZYKJvasVwn4OZJIIcgAElsxwUPCNZbQpgZThIBieJPQwodvsQOcb2O0IGEwAzsCFSI8grWcohQoT5dhc//OJVuPI9/4HWlTcDlTKioUqw5/h14Uqti+dID4YNt1ZvlzTmIy0FZpOSVm7evoB0BmgGsfUfJIhdnl+shWBnyCJUc+6g+mW8PU5IeKEhmJePGXDMmOg8FyPpFZK5W9+N5W/+ga0o9F/BAKz0L57/hGhq50dRGe3qJGbiN6OzxC8t61mCFsE3qVXsRxWYQJoOpasoO3pBtX3SWlzfG27SyCOdSJL276cTQiQMFYvFnUtCf8cARDAJH9pqQ682oU7dik3PeCTOeeL9sKNSQAEJolYPvSjCqgKWoBjikwGPFCq2q4uxODuJuy8p9AfJLzlThoE4Y6o7k3PVZnIo3K/2Onxc5sySJtzdUtuOUaWxSSlMaaBIBsp8jsu6z3YTXPHx7+H6d34WnZsOQY3WoEpFo+L6uIwoVdAl/Z17jm4+zHNxhmbXtSRADBl1aI9DOnk8PEN7JynpnplRyYCy+w88Jh0DkCV6Y1sI9qEwoRkm4S7KAUIR9OqRRM/vvwDda274WVDAz4IAzDFjv/TdaGLXfbVOyNCcG+jfzaoBKWaQtZyncOYAFUHul4Xi2dvJMogMo/F+9fT+3s3Xd8fiu6zxJsXlpSHHXsxJfe4XFCNZqgObpjD5zEfi3N9+IE4ZKaNAJaR7CRM+SfoFDdRtD43cgFKi0raFAb+HoQoLeRiRn1nDnzPMzVdNctV8JDsJZM5GRFdmXCAhX9HcXshV0x9SwEbq4Gi9CprcjPk8jtTb+M4/fwM3vfOzwPwKovFRExXMKrQjeBFBKoVL9pmn1o2YJMHNAvwXE2moLo0kszYcP/FyXWVRiGX6KTQq4wXcMRmClvUK/VQ7xhrQpNlSruJYJXEhWbjj/Vj+5tN+FluA+tncfhddGo1t+xwqoz2dxFG6YozU10SEnfg9beQTw5APoE8VcNORhrfOem0EtgWnsoRVavGGB+OXehSab/j9Uggiu8kKO27Qzj8crLghlt4oyXqlDuQLGP/th+DcP34kdmwaQynpIekm6KkISwpY1MaYZ0EubyaJSFxKrK0whWnYLt9l71hCev80fP3+NLPomwHrznVn4ygByQvdtFh6MkjBSDdySWqrHmyGwgw08mQELZDLOIdb98/jm6/7GOY+dQVQLCKiCEYKY+bHI2JKnJqZ7flg1YY0YpSM2y+S1HrTkhFmMZSXvvIkqRWYYRDic1bPl++9y8+dMh034g2FqWCjbICYcQtSmLZeOdrWi/svQOfaW37aCMGfMhmI2hsBqlj7U5SGoZOesLcOsK4OIL7UopLS1O3vFpn/Kc2Z3dUcjOs7p4NQjqvLc4udQrWeAJ+968ZWIuo3MNnzZn36qaANx6FJ6pNvvodkuYncfc/Cnhc/EeffZzeG6fm0OujkIiyoCHMA2mxFN8E4DsfxUhDCzK0FycqyTE7mIYRYiXAvkkWY0XNji34U4Wwudl54CFHIBQhCLs18PMoVU+YQAaUYkMfiNmgcBrAtF2GaUqu7HezdMYUtb/kjfPvRF+Pql/8bkjuPIJoasf5Lspuwy8Bk+rk8D48KzDNzHZQ8i5JQeiDNarHs7gbup7G+sORn1Qb0r48slPfvxXkz/Du9xtPzmNYyOB+9q6oTVb2+9Ex08DzTS/MbyS8CAVjOsnub2nDajRieriLuJbxiA6i8G0t/JhGobwRZydsfaivbY6XUg9TDyJw8BfX9mTz/8Cpin9CQUJI2EZopjYmSCfjPFLUFJEurQLmCiec8Dvd+1iOxkdhtq8MMbi1SoG6Sdduj0i1BHxzpIXV6gckgYtNuU1QaSgk6V8LaQFNflMNG3qVDZJ2p0o6fiNISPC8zfhRuWVrVwE6WrO8nkpwHPmJGDVT62cTU8WsMwHZrMKQ+q7pYwL6FNXzjpf+GuU98B2qIej9G1LuTe5WmVIKsajBo3Q0ciWTsWqyVcIyRIUE3DwwvK9n71Yy+/VKqhH2WLr7Avk/1NBD7+7B0VzVIPCcb8EQ9B/J68a7b9cLlZ1uPcD+c+/kZgIP/9/lLNXnKa1EoUA3/YPwTDMBB/LQ6IB9QdhSSUbh/st+Jyc7+LmFZFg7QRqG0MjvYizCxELKMIjV9jgBdg0oB970Rx1SANfA3gV5YQfF+Z+O0V/4ezjx3G8pd09WsSYSvgVV7Ih+uK1i7uUXJlsJfbqJp/fy2327I8qP/yfWWL7BrXfpBbFti3joc+2+zBG3lpFwUcUQhLLSXS5Y1ml7PECF9b5OIUgY0EVLcZy8TWI3GHkJ6NAcP0ffboLGDvqSIwWIObUT4zke/g6tf8s/AegtqpGauTxGDIoEsuA1F5ags/BcsKNBqRhoP9oMKUpLrJcyOWUqyDqJcPxno7oWGOL2MBB2IEALDYLQmUCjHlZhw5QSttbxevPPRaFz5uZ/GFnCiKoCyJyyiWHm6yhdMx2z7aLMGQH/vgkYN2pdQ9TjcJ2XgEfNo1QFTeEIO7DiZ+M51aM8VKsmKE0vaEupGmqFnObl474w+1vjHkrLZhm72UPmT38AlL3g8tpH0anbQyUeYVYoNfLToqTaadsE7Ql/2wxTqkSM0Jr/IzAE3xmaCUIhKlOITRtpo1BEfOYLeXQfQ2n8A7TsPoHPoAPTKCpK1JXQbTeiuze6jq+YUcqUi8kPDiCamUdo4A2zbhOK2LSht247Sju2oTk4xg3CMhLIPEcc2qcuCUY9k0gIoLAPnKuM4PZOUZHHtAbKBJMDOSGGsGyOfxHjIE++PqbO34+vPeRd619yGaHIUCVXLYXWAOJy7low3sS9h9edLhGWR9sVrWXFKMg/rIxDaZsogatdGcP7IcGOxtgYZE1OKWHYbJNTsutTHQxtIUBoCisNPQQOfw0+xqZ8u8Of8S9TUlm+ryjiJD2bnoZEHbZmYbTnz0vcvOLTZxTISixr6Kvv6P+7BSgQgdTd3/jR66DfwZBCSuERaeg0ywrjvDdT3en9OQS/XgfEJbP+Hp+G8S8/HSLfHXW7rSuEusuyzK88yLe9xkAsqC3bM734vguY6QZTLIyqa1t0kzesHD6Jx3fVoXnk1ej+4Au2b7kDvMPWiI58ChWg4mSubm5rnZQA5n9zmBMrqNxGQryE3vQHlvdtQOvtMFO53bwzf+yJU9uxB2bYPT8hY1+1aBiVqBrpb8enbDtZmXlaNp/wFOnozgK08pBiqVMLRtTa+/vz3YfaybyCaHDauUFlHwoZ/B7eyNQbK6EKhSvWhANqsd8EJjbR5Ko10zNHpluNSZQuKqms1LiW8tOhmkUIg9jQiMHYas2oIttGXQiUw7cdyWD58VM/feAawf/lE1QD1UxX7GL74ZWp8x98iynWAJC9777mJdvX7A2E7aSxq/4mJ5QVkIZz5OZvFZ2WH8z1LBjDoFlPXyKgFlsuz/1Q+hH7Kt88iA+FSVuFg6Vc5hWRuCbkLz8bet/4RTt8zjSLp+rkI8xo4zEVrKddM1A10pG0fuCA5bwfw2MpK+lypwOTZ6fXQuO5qrH3lm6hf/lV0fngtsDgfCn+iDKgSVCGfjrN3QSiyCIV0tVld1Fzepl3HMdBtQ3M8H7GbBKo2jeLZp6L2kF/C+MN/BeX73AvFKmUJkDWza+L9czayz14jTK8Lpwqhyla8+qqL5FcehcZupVCOY0SFHFajHL76+k/hrtddBjVUBijbkGsz2v4RbB9wgsaiglQSUebhZjflZn9AoFnqmIxEz8RfuHtJr5WsAMmOx2JYmUrs1pjDIr6ggmUA2cQhFSVoLOf1wr5HoP2jL52oGnCCDMDuN/bA76ixbRdD6y4ik/JrDBTSPeOIXrrYMqG1x3UNuiCiQb4vZ+111BGgktOLw37hwYRkIgnzXHip89uKZ+OZgSVRE3kZJLGbeA5kp/8TJAvryD/hYTj/dU/B9moButVBnIvYyLeUGBmbNZD1AQ0xTJLLORoXBcXk84gKkUm0ue4aLP77Z9D89BcQX3Mj0CVLQgkUla9KZYbynF8QuzHa+7Dx9lEux8yP3/PfNFMgwiWIzUQvUmEdkzAp3pRxROYfsucTUyghf8apKF/6Kxh63P+DiYvvx8lKHNLeadtovoDs3J1LBOAvIx+BzXLcBY0x4oH0XbGAb1/2XdzwvHdzRiJlG9qEggHxArYSkXAV9xO+Sq+NvpiV9DMKD0uoD06tcX57ieC9fSEQqnQMSP0+fSHXUSiopoY3O2OhiDUJzKiHuFvUS/v+ESvf+bMTLR12IgzAPo+9mzG55UY1vHGEMxaiiFrwGgLOBP/0BQU5DucZhJsF1/1HErUYUkqFSA/X8QdfcNIxG0/Dspy36NeXlQh9k5+a1H4XH7+3ySdxD3qliZHnPgnnveDXMU7E0004fPcwhfHaOHmvCopbSAsYq29aCzkRYL5YYsTeabaw+qXPY+m970fn8m8DbSL6KlRu2EhBkyFmoWYClcsjX6qZV7GCQqmCQrmGXKEElSuwgTCXIyOhMf9QcpKJPzd/k0Qj6XbQ7TTQa9fRba6j21pHt11H3CXgl9hyYYQucmYO2nWTB6gqKD3wvhh9+u9g6Ncfi6GREZbqCeU55EIjk6xwTZczM6snVCJQ2ArNakGXrl0q4keXX4Mrn/UWKLJjVIo2WEIygQHBQ5K9eGavUiu8L7jIqauSjv1acHeSXT8Sltp9swgzZY8a5CUYEBCkNXtJQl8BEdlhDNCcJaiXD/4Ai1+974nAf3GnJ1Lu69xfVeM7vqgqI2TOjjjbL5W4ISYvVQdwQElvL9SzNoOsl0BW9JGsVfzupNPd3rEYn0URYfLEkVlrrZtcwQC4cgsxrV4XyVoTY3/3R7jwDx+CWqfLZ11VCrNEw3aBuGfpH3FGuAh8w0U/okKBJX5jZQnL//JvWH3X+5Bcew3r71E0Dl0sGqnd6xjdMF9AoTyEUnUMxaFRlIbGUSwPW2t5uIqOY8TelZRWQ71Lj+7LWtm5CIkNO6Zgz7jXRrexhtbaEtr1JXQa64g7hAC4qw1UlDdzwqikjejUPRh+2lMw8/SnoTqzwVQx73Sg8pFxWri5kFzRjtUxQhMXYYDwjLULkL0hKhdxzXdvwvef9no2uqJaNkyAEKmX/LLM3N0lqCG1cEKOgHyfEusWmUsdXkr57AlF8peEmWz0CL+bIzPn9HUNaHOBQ/IZygA0qw+tH13Xs7efBdx+QmXDToABWCgxdK+XYGz3S1Wh0IHmnn4DJtjdnPjNz0WoBiw5rNfHvbFOct1+l46vr88/udJioaBnP4PIfuWeriNJleG6br5cWK97H5sCHXR8twPdSjD9pj/G2U+6BGWr7y9pYEnUl2YQbR+6XwYZFZOHwz5uxbX6mo06lj/4Qay+/m1Ibr0ZCjWo4gi73nSPGivFTPSl6igqY9OojEyhVB1hiWx4FFX/oS5TPT4nSfs8WfiLFeQLReSKJQ7Bpf1NnX/LZ+MEcaeFXreNXreLTrvJ79nSzzYmKgJCNV6NJYP2ba/No750DM31RcRtcj/DnldDt9agUUduyw7UnvUMTP7RH2B4YlzpTs/arAwiSCcqhdRkY11yK0JxJOGU1hwzQEygWC7iuh/chv98ymuhGk2gUgpS3DMAW5HYe5bk2pSs11UukmgzeGfcQwtCI4MS+9ZQFhXcje4v7Es+3kCiixTSGLBOPTJlVShBazWvl+56FBpXfv5E7AAnwABsltHI/T6pxnc/Fkp3aV0Zm0R/Vt9AyCUvl2ISAyDX8RiA+y7FRd0rNO0YINrSUCzFTMQC7PPVCustQ6yeMWt0ukjaMSbf/uc477H3RqHVZeKfTTRWLORPIQwhE1x4O1/OePSYwFS1xL+tfPyjWH7ZqxFfRxJ/DKpMibUJdK/NJyhWRlGd2Ijq6AYUa6OI8nkkcZeJke6jUCyjXBtFeXgMQ2MTqI2Oo1gdRqFaQb5YRo5sAFx9xxpsCVaSncHeZtxNdC/uIU56ilSPTrOJuN1Ae30J9ZV5tNZX0Gk1oOMeIwWON1ARep0mGkvHsL5wCK21eZ4rlS8yKtDNdWjMI7fnDEy/9MVq7Ld/i+C9jshdmg9lw9ysaeGz8JlxZAC1PooxMg7SznGCQrmIH3//Nnz3t1+FiFQRsglYXZ4N1B4F0KRnVQK5FqTaKSCapO/U5vz+WcLOGv2OQ2EZI6L5m4gQYLFmpbchtUYlunDrO+qh1ynqhf0vwDrVCfjJdoCfxADc2s1h9AFXqYld5yh6+jZWNlSCdXAppFamiztYuO+4akpFEBDfEXSfxJZPxLz3ATG0mMkMb7jiccLQpE6Vwb9uUFm9P6UC2M+9LnSzi8l3PQ/nPOYiFJttzm6bTUxhDjq1t91Yl5a/hLu2ZfFRTFJQoVfKY+3aq7Hyoheh99kvc7S8Ko2YBUHGOBWhPDyKoekdqI1vRq5QRNLr8IvmsVQZQm1sGsOTmzA6vQGVkVEUSiUD5W0aqXmFBefvlB0MoXst6//Oqm2jmSnWnPL6k24X3XYb7cYq6otzWJ0/jObaIqMEtjsUSryAm6vzWDt6BzMELn5CMSOcAUmqQV2VH30ppl/7Gj12+l5o8hj09SewKMCBYhPFyDKcmAAhgTGtsYekUJKgUCriiv+4Glc99bWmDiFlG/LkWyYQ6hsdxyZAm0CsqaWWDhWVdunwvm9hDfD729NJYeOMgo7tSYOhPyibWJZlHNIYyFmUPWhdTBYP/BOWv/r7JwMB2LvaPo6J3Teq0a0zZlUKTJSp+uom1zROyHDWrMU/daUBD8Y9eX+81JGOhyLcM8vCsHA7rm5cqsf7IOkv4RVV4l1tY+qtf4qzn3R/FKyln4ifwnlpS+m21stu4vndCrLZfb0eomoJ670YS294HVovexXQaEIVNrBhzUj8BOWxGYxt2oPyyDTfZ9JrQycxiuUahic2YnzzDoxMbUSxOmSs+qb4V5gNKz1l2nVAJoa4uFCojQgkpkpMwBm6kp75zRmgzG0aqZr0emg3VrC+eAwrcwfRWl3g43LFMk9Za3UO67P7UV86YhADB4+RHWAO0egoRl/xYkw/+4+5SnHcbCPK54QaLbvyeAbgVxfd4aQCTqXdegmSchHf/vA3cfuz34ZonAqWWYHjoga9oJJ9JqVdKBOtKpeMoIL0e0mowYjpD+5TBdz3Au73/SqKr6SEUka9SH2f6lsQK6UKydKh/8DC5Q87kRoBP4kBGCNCafceNbLnWgxNl1kBc5V/+hhANDBuP7x3vmZBuD5/NLT3TldRTRuIPAJIlQ5zrphUcGr/5Mvb5uAKO+2eGWRCe+337OqbX8PY6/4U5/3+L6NkYf8RrbFuHQKcs2IXr5kWa6ixa4rr1xFMJ6lYKWLl1psx/6xnI/mP/4CKNnKlXx13WSUoDo1jZNNuDE1uYdddr93kMdZGJjC++VSMzGxFuUaWfZLy5vmyu5/cgIrKbjjd1UJotrUYA5+PjHQqQGzcf8wM+L11CcZ2Hpg5mMUZ03fWJmAzF/jCcdxDe3UZq3N3YW3+ACMFUg+IKTXXFrBy6BY0l48BZCjMkZuUXIizqDz217H1rW9Ffstm7j2QIztGRmtOJ96YuSb/BQUNTZGbkIqfxglapaL6+hs+pWdf8UFOImLGxWqAk/6m67MBr2aetFxDGdUgrY8L1cCpoXJQYmWa5e1iOwSRp1RN24BEnDodeZSONBUsO+j/fp2KtUtvKEps6fD1ev5L58vAzZ+RAVgOkj/r3mpy2xWoTsVIYtfrWejUwrfuXX/SuSs4ruPw2WF5N97xPAFhwgOKON7wxUSmJlX8LCbYyGlrXbW6mA/yiRIkR5cx/MKn4Pzn/wYb/JJcDseY+A2xxw5WC3zoVUk7JwSHOdOvlMfihz6E1T97LjC/DFWcsXC7y0QzunkPRjbuZlhNbjhKnhme2oKJLadiaGITBwMZuKe5GGeOu/cYrOHampGhjlSFQrGAPBGck+Ck33da6HY61lAY29ukSr95o9NHeQbbjAaIEXS76HU66MVd28DCpR6Y+TLIQCNHxK0itOurWDq2DytH96PXbiBfqvCY1ub2YfHgzYjb5AkgxkW9DY4gt20bxt/zbkw//GHItbsG6QgmkDLpWMZGfwnc0+reqIBtNKZEq6RUwBf/+O164UNfRjQ9avunZrwCwj2oLTNw8QLHi9OxCyGtf6cU/KzRIKPf+/WXsfYPXLGhM1CqBZxTRWRkYR9KQKJyubxeOXJAz/7wTGBu/ScxgJ+QC3CDGVeuNIzIGljcBDjx5olcRrM4gh1kZJFvhMIsJzFbR33Q4VlMlmbIfg+W5NnNP8AQaRXcfKGohypESI4to/J7j8JZz/8NlNpdhv1zicaaNfg5nT+4brmcDodtc5w7LTAy9BWL0CrB7F/9JZqvfSPnv6nSRiv1E1Qnt2Bs62koVkcsoTYwND6DDTvPxPD0Fl6oca+LuNtGnoKDCDK76jdUdovdh0UUcnneZ31xFmuLs2gszWNl/gjWl46h1VhDu7GGDhkNeSFZdY0s/JFCsVhCsVRBqTrMhsahic0YmdiEytgGFCpDbA+IiXm0m8YbQXkIhI/IlkAeChpHsYzpnWdhdMNOLB66DWuz+5mRDM/sRmlkA5YO3oT6/EHDrIqbER9YwPylv4bo9a/H5J/9MdChBhnG29KvSQeiouKjNP+H6ZoAVR3S5V6s7vcPT8flt9yJ7jW3A6NDthiLOQ8xYBPI5Uy12j5/l2psZbYLw5YS3zLYtBqaLf2VXbvuu7Be0yIr0JL81Y3K4rSUGUCGNKeJwv5j9J8qMFSzDOAkJANFumbzrbkY2SDp6otuisAe2WMtpLWKyMDjXjBMZDCSOP0+I9XlMVnJfzy+xzETAfq7kF8v+Ul0kF1xfhnRr1yC097wTAx3e4iVwqJWWKGCl7ZiD5/OPyBbskuAE6fvd5aWMfd7v4vOZz4LVdhsLPHdFqJ8CWPb92Jow052M/ZaddRGJjGx40yMTG9hlx1XA7b+cXaf2XOTW65ULiOne6yLH9l3M5YO3IyFY4ewsjSPbqPOzMXQuYvsE64wz3gNw6uTeYcIxkWi0fUKRVSGRjE8sQkTm3ZhcutejExvR6E2xkTfbdVZKjk1kJlXr8sIhBgBoZalQ7egvjzLLskNp5yP1aFxLB24CUm3BVUcg447mH3On6J+2+3Y/JY3oEAgk5KNKHgoHdHJLkQf2UfARQOHFAc/YyLRelOthAe848/x1Uc8H6rV5lyCkHrrCN0yARWYgZGs9pmJzj9+CfkwXbkJGwLr9Q6mi7Xo0KskXmuDkvYirzoMWNV9ACN1bU8YZrzkWlIUjEl+Ub/jz4oA7Bb3aH7lSDMjCoaWcItBL08PPkj0UJTaEo+oSRfAgQz1xYAMLqFhZGbUpQDxWfsss+FzIHwbFUfHrzeA7dtw+lv/DybzCbodjVVEWLS58ma92PQMf2mbsOGkVy/mevhrd96Jpcc9AfFVV0MVt9lefj0Ua2OY2HEuysMTJqhGRdiw82xM7TgdyBfZxUcRRUQ4BqIaJkBGv0Ihj8b8URy46locvvVqzB29C+31Vcu8Clx9KKJoQpcXTCMioxhF8NHv5KJjvkBxXZS0RHjZ3piNLiR+T0TTWF9DY2UBx267mm0Vo+MbsWH7mZjZcyFGNmxnWwS5Amm8ruoyxSEkPWOwnNl9IdYWDmHxwI3othoYnt6JQmUYi/uuRae5BpWvALnNqL/1jTh49DAm3/9+DFfKJnDIBTQZndq8I9UjWFv4t/3EBBRQbXWwY/dGnPb3v49b/vANUKwyyTDM0FcRPmYjqJ4hk08NAKcBsfpW5H5tOdRgV7/vBB5QsV+imaXoV6EtzJKmWFkBIquC9Ekgp37ngbbL7r7b7cQYQIoC3YWDzp7Sde0unhYdE/BEK8wH7jhH0+7hCDbiv7J6UND9A5c1n8ST8rMnXSwSprkIK9GSyYfEUsAP1bfNY/Pbno1NM8Oc4NKgVF42qXAWdjiTCegS0cx2TNSlp1ZC/YYbMf/oRwN3HoQqbjXxBEnMbr3xbWcwrO611pkJbNh9HoYnNyKm6/c61m+fQ8J94yIUa0PMEOfvvAF3XfcdHLnjerTWVrnIJhN8uWokHCUBlYegyyNAnnIFOHiT34NCgglF0P2yO5Gge6hzwH3oOB+gC0WBQCSliTmRZ8J6WVeW57Ay+0XcdtXljAq2nXEJNu6+CLnSCNrtdeMost4HUnHo9MNTm1GqjWB+/49RXz7Gqs7GMy/Bwr4foz5/2HQiLm1D82MfwbGlZeQ/8TEUhoagOt3QmdhMuoECFg26oCGKw9wP4FTqWtTu4KIn/BKWvnkd5j74RagNE6GhKrU455qEdBZbRciTnShCY+MJJHyXC9aTpZfAQXCl0K2PP5Gnkt5qW9nIhRn7de5vNQjeuydSRwimXwqKxwfYPzUDUNQkKty68c0Gwg/C3VXYdRjA+ZSl3iSMffIS8hzuOm5y7KwOPsKVE+/XKbKcNPhNjQeAR8c43iXPxIwp9bE1jL36WTjt4lOh2x30cjnM9Ujq2OQgzn+QYb5Bd+R1RSG9tRLWr74GS7/2a8DBeajSDBMVGZ7Gt5+J0Y27mBFQ3P3oxlMwfco5HAbc65Bf3RiuTKBOzLH8tFBnb7oSt1/1FcweuM3cDwX3VCg9ljpzUBfgKaA2BUQV4//OA7qUB2plYGQEoDZd5SIS9ghYWMuMIAYoXr/RAupr0Gt1fk/3rjrkeuxRAQDDDBqrUJzkYCL5Fo7sx8LB2zFy1Vex9cyLsWHPvZCvDKPTWjdBTnbq2XZRKGHjnguxdPROLB+5nQ2OU7vOZ8P12tE7OfFJlbeh+5XLceBRj8bMpz+F4dFRaMcExJryq4jTiCkyDVhTwEEqLqIiFOMEF73sKbj8ez9GcmgOqFXt+iQmb5KmYOMPAmLvM833L6SgbwvJ5P/xQi/YDdy6Ti/MwHScRd9pvuaYQO/BIyEZiKd3/9ld3x1IHP8nbydqA3DT429cwi9vsJD940U/tuNuqTiBfk3Fm0RSUEqY1x308q4tZzsIFldnNzBIIO3qSxn+iClQu+v5JRQeeQl2/p9Lke/2kJDRL6YmHFZvc9eyyMYyQ6vNUDO+HsP+9euuw8qll0IdWQaKU+zbJ6Ke3HEOhjdsM75+gvx7LsDIzC5o3WPLPMN97ppDQS5lFAolLOy7Abde8XnMH7rDBLhQoA8ReKmGZHwjUB0ForKJvBuvIdo6A7VjGsmmKWBqHBiiYhEmVJaCb6gsGY2TJ4h7i8Wm/Xe7A3RsO/BGA2plHWphEZibhV5ahF5bAwiux2PMDNBcN14DaKyuzuOGb3wY+6/5Ck656FGYOfXeiFWPIwlpzNQsmr0OGpjYvBvFyjCjAcoxGN92JnshVokpkFmxtBXxt7+Jhcc/CdFnP4kKoRhGFbJ3XyA4x8JJ/Z21qsBkL8bYxDD2vuL3ccOTX4GoQs+JC1hbRuJsAS74zFr55dqyqHbweh1gjBNbQAEZ25QbtKfeDKPxGohZy8FDZStNeXVDVhKS8eVOfxEx6XdHgnf/s40kKp/9RDV56kdQLHdVkpgwYA+Zgv4UJlOePROsI+MFsmGX8oHIEcqiIsKP7bmStLd4nkSGKckxhfvEuVJSIb+JifGvDmPvl1+NzdsngE6CJaWYAZCE8bp+eG9YiXH7KUWJKrUimvv3Y+lBvwzsn4MqEfF32Mo+sfNcVMc2Iu62OIJvw+4L2cJOEJ9dhAz5DXQmq3tzeQ63f//zOHDLDw26KFbYL4/qMPTkFujKJKDzABXPPHsXcN4eYNdmI+1oiVAnYSJmyprrWAInBkBMiuwTXMiD7jvm76ioB6submHytGjzXbMBLC0CR49CHz0K1Mn4FwOdOlR9CaCMQI4roBLnHUxvOx277/0o1Ca3o9tu2JZXLuCI6hvk2WU4x3aAOgrlKlaP3sbGQZoHdkW2DyH//zwWWz7+EZQ4RdlIe14K1hwgadStGiLxPQSQyKhaKuLrf/o2LH3gi4imx+wSswgLIm/AZaaKlnbpYjduPQ74DhmJn2IKRLSZI5yNyvV4cETNwsTFbqQskGI/XwREeAi8LYB6BUR69VhPz95+DnDzzT8pIegES4KZsFKPLhzTGWRoT4VGHm+TXLEPX1koZI0sPlsvyqgZIl9d2ALcBKXGIkMq+Say3VhtRZ+1JqZe88fYuH2Kc/rrUYSFWBMA4pb3RuOTFdzYNmKhTsI+/tbyEpYf9wRg/1Go0gaW9FGuiKk9F6AyNMG+8VJtFDN7LmKdOO61jOSmeyLGRNl9xTIOXfsN3PSdz6DTaUJVqlBEWOUa9MxOoEC18/PAzhmoB54Lfa8zgUmSyoSDm8BygxuMRtwtmLL5NJJOD4qkPH3XS2zQIGHzgAIUd/C1dfmIadCLkIIzEI6MA6MTULtOBebngAN3IlkgVaMM9JrQa/OIKCmoXMbcwVuxeOTN2HX+w7HtvIcjSTqIu00bLh4ZlaBUYSY4v+86zjAc3bSHiX/xrpvNtJY2o/epj2HuWc/B9Dv/EaVWR5Sfl+7osHro564CjlDVYQqKSoDTX/Bb+P5XrjSl2Us2m9K5/mCJyxVCsQVUUys7pRbIr4OXy49D2ro8VM+gAAtNg65vr5sio1DHSpoBU8va3bcwEJvcClrXrZNoAxCT4FCGV3OkEc83ihA2Q8kx+7wHolaAM/DZyZQknbpzeYpgOOj72Ut7R6o+sktwUdbpY1NIY2kF+Udcgi2/9UtQ7S436ZizBnHa10b5maXi+JJBMObbKEJPx1j9zd+GvvJHDGOZ+KMcpnafj+rIFBN/eWSKXWHkJqP8ekroMbedcA5/p7GKq7/yrzhy65WU/cNSX0clYPNpCtUNmq34Z+0ALr0f9Bm7jZW/3oSaW0VEhJxo9ChajwidEIBzA7Zsm3GuBWiNVQz9O0CXUIEtAkI5ClwdzIQ/u2KnjJr4/JZpTkxCjYwiWl4EjhyEnpuFKlQMElhbQFQsc5DQbVd8GktHbsXpD3wSitUxri3g5o7cm6T/T+86Dwt3kUFzkUOfaTqX9t8IRfUPi5vQeNdbsLz3NEz/+bNN+i9FDGZQtG9Txs1UYDorKWC008XMpgnsfN5v4Y7nvJVTiY03OwT2GGnrXIOZRZ0hgXSfFem5cl8NortBNTBdNGA2hsVzEv+1A/3Hs0n4TlEpD53zAt79doJuwDT3C62Y/DcBjjudvC9Kz/pKA40Hl4xMxXQPxvnmXRlrUaTRU7/QoeS7FBOwRgnrRrZIxiXJ2AF3OtC1IbXpb39HD9kqetSko012IhsM7MtmuwfgwQohhARJqYSlZz8bvS99AaqwhWEwjXvqlAtQHTXEXx2dxtTOc03qLvnOqakDjYHcZbURLBy4CTd+8zI01lcQkXGPko+mtgLTu0AtGKLTtis84UFan78Xmnp/L64xPCf1oUddg1tthvrFQh47KyVsG61hOFL43sE5HG12UYzyyNWqyBeLJr6fiJ5UA4L/1L+g3UHcbCFeb0B3OxwFaNLMzU0bdyoxhxg6bhlmUB0CTjsTanoG+sBd0Cs5Nk6CsgLry4gqI1g4eDOu/ORrccYvPwWT289Es77M16coRp6HCJjecQ7m7/oxhw6PbNiJbnMN60fvMFmFhY1Y/avnoXD2GRh9+EM9E0hJxeya1cA8uQVzCsVOD6c9+cE48JGvo3vlTVDDFSEtVHqN+uOdZySouWnh4hjQINhvBY/4LpBKIHDpVkyVAhRL3BxrVQM3/ymbgTzAZkLm8pl+UD8vA3DZtqkbdZLUk3/QC7K6gb/7AWbB7EilGPdhxRYIpVwu6XFIW4g/z/HQgDu9g/7zDVSf/7uYOmMrQ/9mFGGZMuFM4WVr888qeFYVodQI8vW//5/Re9vbmPhNlZ4uJnaeh9rERib+ysgkpnaeY/za3DfAxVNpFCs1HLz+m7jpO5+GLlDBz6pZJ7vPByozUJUIigj/EQ+gAGCo+TpyVpLHlFHX7qIaaVwwVMT9d0zhvuND2FIuIFF5tBXwyO0b8fKbFtAhVyJ5GEgt6FBvSSJkmxNhkxkoPoGi/XS7jbhRR2d1Bd3lZfTq62wjMHqQ9Z7QX2ISbcpvGAJOPQNYOAYc3A+tp6BKVeiVOUTlYbS6LVzzpXfh9Ac8ERtPvxit+pJpckqZhjYXYXzLXsT7r0W7voLxradzxmNj4SgUZRr2ylh66jNQ+d43Udq6zZQlY08EMeD0WnOCkJwVZMOZ0QlqpQJO+Ysn4ObffLkNIXHrxkl+yQiOgwJ8u7T00vZe8rTIFqqsHJ9kDsF457UHcS2PbMOV7NhdT0drS/A2AbnvSWEAZFPlITj1yMm8wBEH1v1PsS87j5ZgjgORvCrgpHyfIVDI9kx8vzQE+e+dnu+ktdX5veHERYet1xGdthPbnv0YCiVFl/R+nw3sogXsg9OaVViXbxdRfD9Z/K/6EZrP/hOo3JSRGL0eRjbu4SAZiuenaj3TO8616ak0BlNNl9SDfKmM277/Gey75mtQ5SFERFi1EajtZ1DPF+CcHcAzHg1smgaW6zoiXb6XIGY4H+O8ahGP3zKJS6aGMVWlOACyXSZo9OjVQz1RGCsXMFYo4GijTWEIiDtUuqzns/04eIwkPbnt4oTTbClYqDAyjvLwOLChxwygNXcMzdmj6JH1n3G3CZ9lwyR5Bei40XGoag2KmMDiItSUZQI870Xc8I2PcBrx9vMfxmXGTLCRNR6qHMa3noGF/dci7rRZNTjW66K1ughVHEZ85ADmnvksbPrCp9iD57LnnAdACkdTXFVhSWuMRgrlThe7f+V8HHn4vbH25R9AjQ2xG9boDg47Cncz/yOK2siC1PJLaQcQ5zHnSAu8NPMI9i5f9juzgjP5RhkbQThPullsds+7336Cr3BDOJO33hsrqYODgSyPk8En1ILwrcDQ/NGqBr6nXwbliNPbUhahqqx/WmmmIMNzUz37WP+zFmni6I0eRp/zeIyNVhja0tIm6J/uPm2d/84CbcdBELbdaGL1mc+CqneBXJn1/srEJpZgbOgqVjC5/Wyjt9omHMawoFj/vfGbH8W+q7/GkJ/UBkxvhj7lAmhdgHryw4AXPxV6dBSYXUY+jpG0e4hXGrhfMY8Pn7kRl99nB/5ozwxmaiU0uj2sdbpM/GTcp8ZtvJFerzV6WiHuJGwQ7HUpr8swAM4I7FlEwOoAJep0ETeaiNfrSJot5FQOtalNGNt1JoY37UQ+X0BCBj9CBWRz4MIiPSjK9CPC2rYTausWaEI6I1PQlWGe06g6hDuv/CLu/MFnkS/UTCoyhxKbBCHKfiQmQNWLiMFM7DyL7SSkDqniRrS+/Hks/uM7gFLBMCz7TIKPx9w02R/oRXbROWvCoHDWHX/2WNOPkBiODtGfBtHYxhsW4fggMS9X01DDXikQv2/omVnfkir8os0aBYVa4HtNCLQgWqyZxC9BHaLwiWVGJ8wBTihYwJFDaOwhbsS5UzxZDLhrGzqZ/s4b0exXdwdbMhN6N2G94QhB+OIhG5+qoXC93kR0r3Mw+YQHcIHJbi6HVVPML7WgDAKQhk2SmAliKubxkpdCX/l9oGB8/flSFRPb9vKRROATW89CVCixhDNDMd4LIqCbvnUZDt94BaLaGBOe2nIa9MYzoSplqL/+XegnPBRYXEdUp+Qdje5KE7t0gveevhFfuc9OPGbzOFa1wuFWD52uRt4GmJjapIqZAF2T/OJUgKTXpoIxXcQW+nPzTWIEVo0g+M92AWYExlvA31nbQK/eYJsD2TImKJhpejsjlqS1brwU7EXgzCCg1TRoYNt249arTUAPTzITjGqjuOu6r+GOH3wK+aJ1WVoioihCKmQ6tvk0fk+/j205LUDm3CTPefvGm6EqNj7AGaV5KTh5bJKU6NMa5TlQ7kW7i033PQO1X70PkpU1k63lo0CtwdOvLZEZ6tdcmrZS9f+lRRzHs4VlRHnfktcDl7O5LxeEllnpvt6BMwl49UKfRAZAJRlcwExaJ3J/XY61LMUsdfiBw3EPVcYFuK8GMQz/3qkb4jt3QvEA3Xcq6/JzRT66GuN/+liMlfKcU76caHL9D7YoeZ+/LeNFEvdb30L7jW+Cyk+bMlhRgYNacjaOf2zTaRzvT3HxjD8o8i9JWLrd/J+fxpFbf4RoaBSaquoQ5J/YDcxMAi9/BvS5u4Fjy8iRnk/6br2L52wYxfcv3I6nbCXCT3C0Qym9Bv7SsBKtjIePK/s4ItBoxRqdZgcJqQB0LvL7d9yrB8Wvrv2csH2AdWz6rhuzykHcRHXJRhAjoRp8nS5qI9MsrctDUxzRSEyMmYr1FpAdAZQOvGMnly1HZRQgJkDzVx7Gweu+jgPXfRWFMjWatUZZqi/Q6/J31fHNXAuhNrUVQxu287VVroJkdQlzz30edSkIqMyVObMqQXhuhjCXbEkxQgF7/vBR7G5FLNaC7/OQWT/CT5+Ouc8QvFMzxbHZLhDym8FkFtSatN0sa//L2tiCCmDqaZy4DhCdmA3AnN5DG1dZN8XgJO4Og/e/ORjlLZfCXCGV+OMFWgzaUpcTxO+LeQoZLlUBUiBX68jd90xMP+reLP3bKoe1XkgUdZNs2IfJ3TK97jWQz6HTamHtOX8BFVOyTYkX9fCGU1Adp0CfNi9equQTx9RMw0olJv4S7rjyyzh8E0l+CnPtANv3IhnezNF7eNnTkUyMAfOrXAugt97CTq3xpbM24Y17NyCXz+Fwt4cO5+tbbx29p8KZXOGH+BoxApPMQ7ROffa4HRj50dsxVMsQPbsJOfqPfP6GGRDRUwgwBRGRxNRt+o4CiWJGA8woyGNAdQKaddb/R6Z3Ymx6F6MiYgTsRnTlkYgxqBzU1u2coKbLw6wSsFCojmL/VV/CsVu+xwRvZt78R6XGKCW5PDTO1YfGtuxFvjoETXEThWm0v/RZ1D/4IUr2UdzHgA/OrnwTEEezTyVIGsQkOl1svPhM1B50PvRKwxJukPzG7RnKpIXlLQSINQAHFHuc9eqXeUDIQRBKHVcYsf36T5/XB/w4Q7i3tQ2gu7uH0z8LArAgQFgtXYST53W2WWFfHTPxR+oz4Rv5hZDsHkW449JS3V85c6mUJUgQst/ZGQK7QO33H40y5fzHCdf0c9WeKedUls7zp2AxS7Xp81h7y1uQ/OgHFvr3UBgaw/DMNk5xpYq9Q5PbWZL5XCOGtlUcve1KHPjxt9hQRunAavteYHQ7ol2boF/0e5zIo1broLo+veUGHlkt4jvnbcNDJ2s43I1RZ0mpjEue/nIJcs2MgF/2PiKVQzFXwFAhj2uPrmJ2ucXx8UzsROgUEUilzCn/ngibmpeSPs8MwNgaiPgpR4B6GzJToBcjBec6NASftBooloYxvmE3ClGR58DEFJBtgIKRCAFpqE1bTAMTiwR4SktV3H7Fp7F6dB8KpSGPBBzaJOZiKg3nMLZ5d3gQ0TCWX/BS3Z5b0KpgG4TYtSPlsLX88Dwt27qH5CHf8NRf9Xq/lNwuLLyv844nfrGWLNV696hfc5lNCkkv35w6cJz9M197fuMJ0C7NVJ6NIKsT3E5UBeCcTC+8Uz8Fa5kIzw5ttVJQSdoIUph+gK4kJsjW1Qo1BsO+vp12GFAqPDI1DusVSEj3P3s3pi+9F+u77SiHFte+C7kMKf3OXFJR4Y18ucShvu1XvQ4qmuS6NMS1hzee4rO6xjadagJ87LFUECNfKGNt7i7cceXlbO0nyYjpLdDjO4Dtm6Fe+HsmFHityWHH8WoLz5oZw6fO24qhQoQjBPfpaoLQycDVU8bQReMnFWy4kAelbt2yUMe/XH8Yr/zGbXjvDw8Ydx8TPjEAB/dJ8hPMJ+I3KgERPyMDRgcODbSgKcbA5QrwMaRKGIZAjCNuNti7MTK1E8XiECMEo3rZCWWpqhBt2YKIqvdWR6HI8EnZjrk8bvvex9FZX2ZCN0ZBEyOQK1YwPLmFowgpiIqqIrNBsDCK+NDtmH/dGxFTDofN7xC40i4fo9nTe6rd2KZcim6MnQ+5AMVzdkHXW2YR8RglQ5BLLSN8RNEYU83XBlv1re0M8csvBXIghu57RaZa6And3pFC5nSDyuCduAnwRBkA9Y5OZUcF3T6MRqQ9piyd6XDdcGOSyIWuJRmKPCxt6nBKV4D3A1iL/503VzudsGAbo7/9MFSrBTZ6rVuictIihdzcJLPmo5FQ4N0r/x5YmgPyVZaEVaqawzn9LQxPbkWedVqXvKLZ3Uf5/rd+79NUXdAQxtgGpTeeCjVSBv7yN5GQ5F9rMKTurTTxiu3jeNsZM5iPNVatOZ+MejHBe2vdbiYajR6QjyKMFfKYXW/jvdcexp9+/Xb8zXf24bLrD+Pqu5a4M0+OCJUy/piIKdXX5gQw0fegG13oBhE7EbV9scQPLyZ6YkTtjn85BkBESZWCSE0YHtmMcmnYFANl+rAGWEYCCti42aQkD00ChQqifBGt+gLuuOKToloP8TNiBj1UhzegWB3l1OXRjbsRUcQh2RGiCTTf+S50b7sTikN8XYUnIf3ZG2AkP9kAqEQOGTiHK0VsetyDmNGR/cCvDx93Hz73L0a3MJzF3rz3cSp+pWaEXJ8xW/AC6VmTVO5Uev+9qBDkxxUEVtCmT6YRMM7bzBpx0T4BKXzyMoHPE7SzAcgJFXYB18dOwK27swakGLN9EKHJhJD4nlvbg4gAtm3C6OPuz378ThShTgTmeY9RJK1AcDl+iqVSuYjG1Veh94F/gYqmTR2/YgnDG3awDaBYGUF1YgtiKuRBbj7L6Civ/84ffZH93+zSyheBnedqFMrAHz8eargMUJx6rBEvrOJV28bxgt3T2N+J0bYLt2OZVCexxE/wWytUchFuWWrh735wAM/86m14z9WHcdvsOnLtDkaUQo2eMBFpy8b2WxTASMC+qNR5QiXOGea3mfAZJdBn6vrr3INW6jPsJ6bAhj/DAEjiJ+QxoO+6bdRqUygWKqaIKBcONIyP1QzyCszMcIESNTZjrEvFKhYP34TZ276LHHsGSDCbfHyC5aRS0fOlVOOhacqmJM5X4RiDlde9no3heUIYTFBOMQ1hClxTSZvy7fTMadv82PtzrQBSh5wFnROVpPfIV3tNrbyw5okgbVCUcQRI9dQcbwLIsyjXRVhmtgFGbEcrDu6HXBchXI9nQD85KoArLX78Mzv6CjXV3CHSWi/UATU4nDI8OnferEfBqgMSUvl7z8B9/9cyAZI6ay2UL70ElQ1kgOuhRTDa8yQ+hl2t/HhcZR8ba96LgPU3vRkg/zcV10gSDG/Yyc03WA3YcIpPMDGRWtTBpoy5O6/lFyX1aFJGd54BHStET/5V6D3bgAVj8IvX2njhrmk8d88G7G+ThT9BjwhewH6KUWjGGtVCAautHl535SE84yu342O3zKPRiTFUzKNMi5Fce61gxY+S2OQKkHuva6U+XaPtJHhQB0xmoDH0cTgw/aWGKFQP0O7rCZ8YAb0nN6d9xT1K/OmiUp5AjmpJeqOaYep8rUoNanLCJDWRPYDccCWKiPwaWiuzLOU5XJsARNxl+0l1bIYNrFQtOUdRhtwbYRz1D30Ya7fdbqr/+DptQfbYv7xyKMx7jWK12l3Uds6g9qALoNcaPgEogNWMf94Re2bN+/Vn12iqWK0AuGGRSsOeIPSUfi10aiEoeT1meZG9sdQ1TroK4GR7lsukrKCOKAWucXHPHg7Iy8nzBaYg+WvfIGwugY/8H8RCvdtGuv7sLfDCK2P0SQ9iQ1CMCE1yI0s7oTyNL0KVIKoW0b7pJsSX/Tsi0v0pd788xPop+cArIxtQGp5g6O+KeeTyBXSbq9h/zVdNPj5V2Nl8CnR+CNEl5wC/dAHU7BKXW+itNvG0TaN43t5NTPxMqwz3jT+fiJ/ceURC1XwBn751Dk+5/GZ86KZZ1t9HKKyT4Lm16Cty27HBr2uNfMQIYiiCEhwJSFZ9Y9mnFzU94flxiUQuIEj87gifVQHLAML3hvjJYs9BRWTj6PVQK48ix1DIGdBs6CqpB2MTUNTXjyR+ZcS0AOv2cPC6ryKKCsEoy48gxtDYZuRyReQKZVQnN3K4NfI16LUl1N/7fk7pd3EWIVxM2JWta5QKuhIfLgCY/n8usRchY6XwBnhaSktjb3Q73hqVi8iv0Uw/wsxyDQKyL244Y0PLKiECTbsbTC/jkxUJyDNzHOuCnCBn1DDRXSHTSUCXPpeFVC0GGU8ElhMRgGm7g/AWCBUiZeEl7rleR+HC0zB24W5e1CQNSLJmwzM9CnFj53sBmm/5R6CxCp0rcnWf2sQM67Jkha9ObDZSzC2cxNTy23/dN7nDLjOF6ij05DaojVNQT/gVTk/l1lprDVxcLeIlp27EkaYJ6iHoT69OrNGONdaIoFWEVjvGi75yM/76a7fhyHILVbpHCu4hSU7uPZL6NlzYRPQZaz1xEPLju4w+59z068Zl+zHcJ9hv04C5PBlJeNOpiN6ThGd00HMviwJ4v46pfBS3OdWZysiXFDULcS3HbfEVku4UlzxFFYyUCRQid2GphqXDt2Bh/9XI5cocO8FTSgiGCpSOTvE1h6a2MgNmRqLG0Prnj6JzjCovUeET91C914lFi2EoGk1qXMJZXgk2PfBs5HdtYdtHkPzBDuBjSGQ0aWq9y/UrbAcZrxWv25Sdy9oPJK0OqhuQgtODhLwrduMFn92DssVOFgKg4vhZ44MvnSwMIn5KMjDc/i6nxEOqAR1SshMr0ncy/COrJw2CQPah0E+tHqqPuoQTYogomp6HBAgQOH+oO5cjy//hQ+h85N+h1ARD01y5hur4DMesU4Vciunn+nd8cIJCsYTV2X2YvYOgf82cdOupbPVWj38YF7agqrUkLadzEf7+9M2oxxqtbsxtsInoOz3zd72TIK9y2L/cxh989np86oYjqEQKJbIDtGJOXyZpT/n/zn9PnC0izY3ydxoddJfW0JpbROvoLJpHjqFx+CgaR4+hMTuL9vwsOouL6K6tISYbAAfI2E5rHDVILwfz6b2R9qa+ITEYYgBtU+WIwpktQyDGQBV/KPOhSIVL+BhbAstFDFK344kJGzI8aZ5gLofDN36b59Y8PqsK9LooDU2YXoH5MhtfmYiKNfSO3InVT36K/VXMyMTasSvN40v6lXwWSaeH0fEhTP7SeUCd6jIIHVwWikkR2HG2lA3KfRcQszPgBdoIa73vtFkzgDScS7VEaAruJHb4snPJyagHYGLneRAWl3Gag5e60mfvNSkx+H5rvr8Pkc+cMfAPSBUONoOQYS2QgO/NJRiQ49xk1JocQ/lX783JjZ1IET+wk2aZl2diNv+P3pPkywHNj30MauEoQIU9Y5JA29jIRV1xqqObfNMMd1OEeg/c+J/mEy3ImW1AYQjq3mdB790JLK5wgwyS0C86cxumC3kst7sokC3AWq5pHJ04QblQwNVHV/Hir92MI+tt1Ep5JB1bRYgSklyYKFUk4gWluHV5d71u4vlJT6dgGZbqQppZYlRktORyXabcObkjqfcAlSSneAQnrblYqpPkjA7oeNMwJJzXuMVcxxoTQKVRiPIGOXCikW1kQruQu3BiHFhfMzYaQktUnnD5CBYPXIupXRei21qzRS8SVg2KtXE0VmZRHd2I+sIRrqhEgL7+Lx9C7fefiiIZGcWKcaXinCAhOm+ZysKg0rmTD7kAx/7pM95VGVRWikngzgdBGrO9Sqw9n9/u3HfussdRDwZ8yoLd7OZs6e68xl0t9vSJK74p7QmXBDsxBEC9qflvBnJ7i0l4GfobINUzEjZ8OUAlcCGRjpgzvvzAJwRnlh1Tsucn3bDeQHTBHlR3b+J4+BYH01i/v1ClPAO1Q8lToc52B50PfMgEklLCSqGEKpXySmKu5lusEBS1hlKyVBdKWD52O5ap0CXp/pTYMrUNaqQG/Mp9odfWmWjj5ToeMzmMB43XcIykdDdGsxujRZK/l2C9HaOQy+FHh5bx3M9ch7mVJuf3M/FTHQAbyMM6PdULIDpcXkd732E0DxxCe2EJvQbl9ndNtSH7mDgxyyaccBVge7PcOINyHDot9sm3lufQXJnlXgV8PBE9S3hqG05MhSS9RQSMBqgsuEUGVOMQCWKqlcCMgzojmcpSrhCrWS02t3JmgzGg1sZshaYCjt32fXafmipCoWJPZXjaZlJWubowGy+jYfS+fwXiq69BVKZqqM4W4KCcfzy8USRll+YSwPB9TofaPMleEN+g0IUmp/CnXSU2XiBlxMuuYa/2ugoyGTe3NPpl0+fTVvQ0ah6EQgStGzMHfT6xqsAnHgkoB+UbdoSr+r+WE/X5Kt1LJAa5B5raOWXHyHyf+c1HjFm4Jr0A3kXjzt3tofzQe6MQEb2QHihTkNOnd/38eMGXc+h+7z+hr7oGiEYY7lJJrxwl+FADz2HKA5B95EwvwcNUQ59gLcHcDVtNLNX9LwRGa1DrTfbNT+VzeOqWCcw32uj2DOG3ugnaHUP8dNIfH13Diz57PZrNLipUSqtFxj16GUMfWwd7Cr2lOup3HELzwBHEXK9P6J2sx5NdwBr5GIpT6W/7sqXKWZJzpWDTHITTdCkdd30JrfV5xN2G3Y9UANL5iQkQAjAMwOQ60F8qvklqC0ClKYgxdKkoqmnGZa8tAm4IPQ2PAtUKdK5kKiFRbMDKLJYOXMfNU4xxj+Y45kKp1L2IvquMTpvnzRmDDbQ+8UmTrC5UUq/OiSVJRsAmjajTQ23TBGrn7zFVkX2ifRAwwZwwSH8X5Xmdu09Y70MkiljnmXp/qXialK08szIzTGzw5qHKyWQAPRenJC8SdA9pvXecLfhTBsQ0O156PMJ3YCa79Uv4tKUg6G9uMZsmHLHC6Jga+eXzeX/q8NO17qJs0JfkZRQ8QqTQ/MQnrMWZyndFnA1HEo7aYOWLNUNQ1l9Nhr/VuQNYObbfdMQtVQHaf2wIuM/ZwOo6S1O93sCTtk5hKBdhrd0z+j4RfzdBs6s54OfoWhsv//z1WFlroxKR5kFET5Kf9P4eom6CiI49OofWocOcuutt32Rgs4Y8Z8hilEJGM2YCtvinjX1XKVhvIT2XAyfDZoJer4VWYxndjukEZMJ8SeKTkYHUB3NchATN+jrW5xextrCC9fkFNFeXOGKSkpbyrjOzk6CWMqloaEQVjGmtVMfMb7kCFg782KgdlrBcNycKt6b3paFJRKWaZRBV1D9/OeIWNSfNh2r/onyYS1Ogz232sCTsDRh/wHmmLqJY33Klpo3MsiSs21HawjJtxXx1IbdPGjWHFec+i4Q6uXkEJ3NqhDfuZ9hOuCagSYcJ8MWoQekCXH00K1T01CaNG6kvxST2VWgJ+zg7QNrXn3nvuDT3pm8hd+apqOzdyg+Zwj64VF6GafsrWFdjrlhEd30d8Rf+AwpmkeXKw5ygQkRQrk2a6rXMJli2cSjrsTuvgU46iAhlTG81euR9zjE57Cvr7HvfUSvj4okhHKu32Q3opAxntZE/K4rw+i/fiIOz6xiqUO59zLECppgnwekISb2B5twC+9yNCYYq+DrCEqqRJHSW7iJD0hOiIXhTetqgGPOb+c5Jw16nwUMoUiNRa0twRTUp5blVb+DsPZvxgAtPx/T4GI4eOYpv/fBq3LR/jouKUiUv047cVmTiDp5UTj0GxsYQDc0jWemZsGhd5JZijaXDqFFmYKfBQT6mV8Iw11VkFDA8hfr8XVC5Gno33oD69T/G8L0uhKoTpDf9C0RTIWEn1+hRpyVqK3bfs3CgVrXMhmwIbnG4lZpZ6/xlv5iyV8nQhEwmSce8BLuRpAHXEDQsSBn5YhrtmP3D924fwXBOHgMQgMoWUAytle30BCuFv9+UKdBDMH9XoW24jxdI12U3dltZA94hD4kc7oYJ8GQroNXWuXvvRVTKs7unnUQsBWQhST8MNzZa9OUCOt+8Esltt3HYKUk9In6yAZAUZSnkdFhiNrkCGmtLWD62zxQAIWPU0BgwOgx1wRnQq3Xjd2zHeMTOTeiQvt+NURB4hxjTULmEj19xJ665bQ5D1SKnA5MlnffiYrbkOlxF4+icIfCcjZijE3hftvWJO6Odi4Zh/dV855i6MfI5Q6CrmmTsA8ZGoFNdkxLdQytuoUTuUOO9R7fd4g49b37+7+IPH/8QlCje326NehOXffn7eN6bP4y51UXkiuQRsKYlAaXJDUhuQbW2Bl0ZQbS+hCTpYunQjahNbPHPhmMt2AZQQ6e+gtLwOOoLB6ByJSSdWTS+crlhAMcL3LG/9IgJKIV8kmBk71aUtm1E++AxqIq1VThpLwguhWz7JJtQff0uQt9N0UjaMBnsivY5+aUYzHps/JM0JMpiSq3CAJOT6QZM3ZHbxOTKSwkml903FFdwsM9hGreX+2z1eAeh5OWzhhfOAz+OPuUhkkLl3qcztyMtlIpl+F1lMQlxCa5eQ6aDr3zVdMWhltbUi2N4giUQuQHJFWUKSZBUIn90hOVjd5ruPnRiyvUn+jnrNIDaelFr8UYLWyolnDZUxVyjzdF79XaP/64TnE8UrrxzHp/7/u2okrS05b05qIc69cQxuosraBw+FuA6Beqwld9CexvMw9LeWejZ/2/bfhEBcbqw9fN795yTyCT1Dfw34awuq8+iA4ss2h2q999Fp9NGLonx0dc9F3/65F9V+VyETruLdruDdquDUqmofu/XfxmffuNzMExkR/uT0dGXzLYPg5J8xka5rLiqjkNTk5RcDitHb2VDJLUvN082sdmVNX5PTUaoejLZZMiu3/nqd0xGpGsk4qjJywWOTEZiyYRiQvKjNZTO2mnmWMJrsYLNegwuuUHNcVIvD+uErUB6tLLVr6XgcsKrT02QLm/Rz9Da/sLeJzUXgHGpGIQYtAc9IoBCDDTlGUlNVmaSRcWddHiEwObZCbLfm7JuWV5vqZv04NFhlM/fY8J5ieCdEuj2TfMhq4ZFhma++S22/rOLrFjhhh2c/kupqxaNGBRAlXi6WD5ymzH+kbORWnVRn77TdwFUTYdXXIx7TQyj2elhpdVjoqfXWrvLhr/lZhcf/9bN7I/PU/ALETLdA3UZJlpfWUXz6JFgwHOVfdjA5wpyGF2fC2/YRBxjwXdGQEP8iqQ+uwCJ2Eke2tbcbKQLgUCuWo6pOmCfMY0NCXsLCrqLT7zpL/GYB12o2uSZ4P4USlGSUqmYZ2ndanfU/S44HS/6wyciXltDRKgli9jY6FoybkFCF6Uqo51OY5nVgIgkvI9Jo85JVfYGUOVg8sTwvaCG3rXXAYSOBArxdjkpiLXiYDCaHnpilfN2m/kbpLem7HZWnAkkGt5aGe3QllyOKboJ6z4cJVyQkpodyhYaRTiXCI0TToYT3U4QAVCAiyAqRy2CU6aL9AQ+5FFMdlJlr/UUlxvwyUdmiQn1bzIpoG4GrBinxJbc9g2o7Jjh0tkUDBc0LXmcPC01+Sigc/AuxNfcRMG3TDyU7GNadwFRocpS3yUaUfOP1voi6ktHzL2VKcR1GIqKeY4NMwOI2zGGiwVsL5dwbL2FhiX+5XYPiw2Tl/CDmw9j/+EFlHPk7iM3XxcRGf4oKajRRPPoMSOxabFTWK5r6c1S3jEBS8D8m9X7nTrga/07qW8kvgmFdYRoqz/xsba6D+8XbAtRpJFQnwPVwyff9Je49IEXqG63hxIhJb8eyHxIExxzxmIcJ+ppj3u42rRrOxs9nWphqiRHJkOQGMM0FQyJoSlZik+Ux8rcPlO/kBu3mseWy5e45iKdp1AZMbycmMSxY+jeeCPnHJl8gvBsmRcGMkWLHCmWEKpUhYkYtk9My9YEEItPrEUTUSmpLxuxKggktcrlz0KgCk4VSCcwlrCFMXnVgDVrm85yMlWAQa0N+ohODixYMDKvtNQ/7vn6NA5phEx1T5RHZq5Ppt4u8nt2IFcpsOQj+M+qh+AT7qG5hcUEkge6P77OpP2SNNKU7TdsDHD5EtsBjOuLFqUp871GQSnk7iJioei/XAFqzy5z0maXo812UKhqQtF9XS7iudrpsPSvd7tYXG/himv3o0TPjgx7nLdv8/U7PTTn5gIBE/EzEVtLv0cE9jtn4GOXnWMSVtKT5Hd57JA2ArufR3OiOIZXDxI25FFwUSlS+MT/+1I84gEXqg7ZKcivzqptkGCufTJtVHZtw9QELjpjN3SryZWCQ/Umk/7LevHoBHSe6tsXoKIit0pfXziIbptCqnN22ZrKyqQG0Ebpwpx+TSnEuo3Wj35knq/taCyRM5O2vWbCpcLIigGM7NoENVozSMkeE94MQAVSAHk/Y9gnELUonBskYuo3MzbHMsL67UMjfVm5LgDtZ9tOPBmoTx8RxOyMeVmpapUef8/9XHBwg7EMn/Cuj754AYc+sgFIYj/SFU/fzm8pK5Xq5blyP/Jc4RE53RTo/ehqw/hIKkURQ06S+hSAwoUrLDKhF0UEri0cMg0r6I4oW62YA3ZsNcU1KFin1cJEqYgV1vkN7Gf9n3LrE40bbz2ChdlFFGk5Or89/Y1jtBfnEbeaBopz113Kc7USOkX4jviNJFdSyjMxcy8wUyHPJcBYQnRWeWf19xWenEeAYL9KELc6qBVL+MRb/waPuP/5inR9KhlhwEEgaDe5rD+Lxz5WNbkUvjuvyLrkcZAaMDZmiJm8DVSpub6ETmOJ4wPcWbm+IqsJOS4kSkVYDVPLoXHVdWi7FWjlkG2867fExvTQbJJNobZxAvmZSZPk5BfGAEydAQHpL8OiDcFs0jjYjwJcTob73VsXMrSeQg0ClYSGNa6mwIkrASfYGCRSptinKQNmrupmtZ+kUzDGEJXiIlsSEXiKzUwwLwLpJxFqgsNDGU4rP6ZaLbEvOY/Sni2GoK2R3A1DTrKDbPyvbX2dXHed6YFA7r98mavTEBSmIKDgajELrtdcRXN1zkixQhGK2nmNVKBHR4AmVfU1UX1L602O4a9EEVvNeVFS9+FOgptvuAuKMgYVJRLlLAJVHNjTXl60DS3lg7YEl4Ez6TgTY813/nMPux2k98kkQcJzqLUrAislf06h12xiqFhQ//7W5+Oh9zmbDX15gv3enyorPdv5tAycivDSV0eOHTPXcRVvmMFyAzbDxKgl2MwGJLMLHOADRUlNLbTW5lGb2M45BsYWEbMakCOmW6AswRKS1qoxBN50C7tbqcy4gekmMNm0c3SyW3HYNRkCK6SSDJVR2jyF7k23AxVSLQyCMcJHSGNXE3MQrYhFFcLl7U/O6yLXrSBjeRL/nXBn9zEid5iosWHs6idu2z/x3oC84Ojs5Cc1Pm9HNn2M0NO4rUvr+yVm0YNwynoKzlgNfSslt5uARN6lKiCaYARsdCuXEe3caKAe9aPrO5+9lGQEuRx6lA9/+z4oFBhGk/+fstEIThMzMN1u7a3mIjRXlzn1lxFAaQighp/TU+Z3SpFNNKrUPmy5pecby5ioldWm0SpqFBevIswdncfisXmOwecUXIOlucBma2UBiIkxmK65ptIQucPC7HuDlGUETPQpOOkI3sb/M3KJg7vVufm8bz9L/ECvsY6RchH//rYX4sH3ORsE+w3xO63MVbBxURH2GfluNhGOHJ3HVTfewcVV2E1JDJOQADEBpzYQ0pgYNYlCbPk31YKbq8cy68zMXUR2GQroKQ+hS52KUURyaB+ShQXkpjeagijSJuW1Wc3/069Ud4GUi9KWSaxT+TRiSLYiTErHluop/5Qu9uFyFgIqDlFI4iyiD2Z2/Yf3oeVIJlrWC1lBM4EpuaT5E9IMTrQ7sEXfbmC2T5+fiGz/UpNU4b5N9UcRxg6T3S7uKjUXduJ5cUqRLdCHnxyZoul4O3kAuhx6m2yeMhLd9RQWpgZGoT6z0RIPMYC5I9B3HYOiygFJzNl+3Ls+H3HZbyPBrLswKqBTX+bYeC4vRfH/VAhjZsaU0iI9PtYoU8PMDuX6a8yvNPXqShsVlahaoYC5A3NcLCTiFD7Kl6eouohdinG7FcJSueSutW5bIueW1q4clofhoaSVuS3R2NPHTti0XMcYnCvQvufGaCRlI4Veq4XRUo6IX/2yk/xsyXfMPqyAQPxujZiGpaVihH/8109hcW4R+Q3UFl08MtbfI273xeOjOgEFUgHKxg4QaW4e6jIEvX+cn1uOXYCF6jCwQMUbC0jmV9A+chTFzRuhW06NDAE5jv4j+1UMExFY3EzFScRK91GuAvHK7sI2T8Gvab92hTD06Mx+aVuM9W+ys3BW/3eJSFIoCpd5EIG27fNJYQCuLHhkA/9E+KOrTiIs/35sGWL2jQ39rNhbckzF4RtHUU5iZNsvp9hManZTFmrfJ410YzLqjNQQU5KLfRAufmqAqcF8UwDU4SPA0hKgyNffZWszw1gyANoWX2ah0Hki7mdnBk4IKc/SS01NmtLalKpLcDQqcpw/ZQfm4wRxj4yAMagG8OrSGrFZY4CyNgl6detr9pzCG8JFR7JM1/m8XVKSiU/gXDw7z+6zkf7G6OcMfR72u47JnIKbGOJv1DFeLeHT7/hbPODCM5j4C5TJKJitoUkfi2Y3YiIRenHCgUGf+o9v4/Xv+xhyExOG+H2Mrg224bpeBUOoNH+VPPR6wbYf76HTWEHcWYeiakz2vswzKaLTSIxqRucjRkJ5CkeOQOF8bxcQelEq2LRn74JmsLRlxu7i1qNbxG7NyUadlnEOLEzjpkCsb08QQmhKhuGuGwhHrEzX/jsZcKxwqRpaONnpwAY4ZwfIvfVShO3Sd53uN2hibHGh4/krHSSXv6aMh+690GndgTIpiC3pMdTEMFfzIQhuAo/8Sf1CkAPiJUxx97PHgG4dyFWBJGL9nzYORnFpym6kSRftxqohfFrEBGvLRaBSNQE8PWqq0UOX+vkRc6YWXx2Td5+nOMA4RlxfN3n09qwUHkxE2OtSRXvrmfDTb6PIhA/ZLF9+Vr77sWSKrp1U0O1DKLDX/flk1uVHag8hoUYdkzVD/JdceIZqEfFbvdo/KZkeK+w/tGDJ9VosFfDJL38Tv/Xcv0evXDXGOq4za54lByXR0SRQSYUiBksMYHgEmF+xhlXqFdBBr9NCyaZhm/vJMVOmc7Eq4NdKF+rI0VS0p6B/80eZ8XP6NamHxABmxkzNQjvf3MvSu6zlWhceKX9OF1Nir2oTmMySd9F8YarCeERBXWkrCJZBoYbYkF/JRMQ1zJBlOvTPxQBkRSCX0CCes+WOZnKFSiAnOiv4vT40iLCzm+xD4L4yHM/AU2n8SgMEJiSLAHL5CPlulxtkmB/T9kh3EN+PSxKaO2pkg32geWp5zd1sTY8/M/6IU2jJ9ddprxuJRuoBLdhywVyHUkw5v1+htbiISqXCxE0BJ5TYQ+fqNVpIOk1EbKk3Li4OjOs0oROj+zsJHRavM7iZXo1eiju4792mAbWZWzV6vyE6kRCUcfkZyb+GyVoZn3v3S3Hf8/aqTscSvzCiGTljGVGmFS5JfiL+yz77VfzOX70G3VJFR/mcKYbC57FGTSYOMr7a4ouR4iq/emwMWh1ktYplX9xDt9NEaWjKx4XoKGYjIB/ObkPDLOic8bFjYdlZVCnXinaGPTuVrB6OjQBUYdjFAjjhmhKoA+xWcvlmYvE98We9YA7VuShDsdz9svSHsHEkXNUf65iAYaJG9p5sGwAZ8T0ctw8q5Ra0xRDsRISpkjAqeAxSnNhD+0GbnAHBQdzDTFN86p5N3bkYanQoTKZUqVL3J9+a+4yPzYcS1dxzvWBUVe5LLzcKcGlz3jofaCUY1R7kWvxUj8+65OrH5tDqdVWpVEVpaAzFfJHVgPrcHDQRu4+mIghOJowWIsqSI0eEJVTO2efQYzcXpjejV6csWpBFTnwgtpMSzmbidH5TE8zGACRsOe+tr2F6uITPvufvcJ9zT0PHWvvDZaRv2zwnmexGzVaKxYL+0Kcux1P+6lVIqhSum2elgAJ+UpFjzBAikz/Bcx0B1OxjdMzcHxcQMYy726ZuPgGxMVqx5+KowCjHwUL0XPT8fAZNS13dBorZBeuwZG6kyuqH6xxsKvqK+3QZZBhkbReLPws2/RrsN/x520TGfUejM6VTsiHu6c08a0sLfGpWZnDyGAC1xfY6nuil7i2g7kYFBJQYX+wj1qlAT3Jf/09ggzIZIw1C0szGMwP3oBLD0Tn/21TZcTqrP8aKIMunDO+kL5aWjSTmuv625yE3/DSMwMx1YhgDd9qlFmDWmEXXJ182t80m/zxlJFIRDVMws9VcRHtp2eQSxBqaC27Ymvn88LiUNlv+4+Y6YlIj6LxUjIPOXawiN0Rx8Kb2omN8oQaCu0+Jfe1v1gsQat/bCECXzptTiBvrmBmr4dPveIm6z7mn6U6nq6glmYenMnTDS8agDvbiRJPO/8FPfAFP/avXAMNjTPw88xb+B9qPLJCJoMmjQC9CB+QzrNlIQGKoXCKcGpzWUwjcsC5rg4hIDYhMhCZ9t0wuwbCU+kcaeZuFYX0U85VHlI9MFKbLCnQv14jTVv/pF3QSpjspLZZySuYFhuJz/VL7Bndl1rKSIqJUP0O3vDmSZZCo+1mNgKSdZXhPn/4jPstt0FeygWOIkEj/lejAXcveeH8/dHeqULXV/0u+eDsvAwM32JBhdRizLs0vK9aox4jHuN+Mp8BU/HUVZ1h6cAsw0TuRNvIEEHFygg7lAJgkGH6wxFB6FFBjKu3ItFqXhRcRBF9ewI4tk3jypQ/EhafvZJrYd3QBn/7Wtfj6FTdxRV3aj+0bbvac5E/rhXaRZj0mIgKQXX1G8m8cH8Ln/+mVuOCMXZTUo6iGojEtWuKXD1I8NT6TNfh98BNfxFP/6tWcEEWtu5jkiLCd4c8M1qa+k+qUgyqWoSkUl9p30/f0ntFBiOmPu62UoZnul0ujEQNhtyAtaZrnCAkxCztO7/hI5ado8UcFBlCg9Gs6h5g6KZME/klNQCqAJ3OAJBn7cBwPCI10AnH7qJSM7UJq1W4Ukh8cB+P+vEZAkwzkdA3iB8y1BJEGjphhCicYVBwOFS5GuaM8n4gFCBOQLkriGSwZ4+xZ3RH+vYnjNrhFVG7mczbroho6LSwRUuoKWxJqJYlHte6IcOWDJF2Uu+paPdt1zuWIOoqpsoUxfFYe1fZTnvjra+t46m88FG/489/COHkyxPbc371UveMj/6H/7LUfQDfKMWQnJuAkmpkOEdjkid9BSWvg4qGaMbHOX1/D5ukxfO6dL8X5Z+xS7Q65+nKIM8/C+1AsNfGdaK3I01IsFvCej3wGf/Ci10HVRgzx0+RSIVYmVLtW6GXj/9lqT8yhWIAi4i9QKjUxBfObfzicxUd2GeP6MyTrEtMtOnUIjD63KStz0BpLq46aBZL9TIFH9JKI2wqGcAJrf8jUAwhTZP9VgyV7AKriWWUYqyGFtLXcGwYtw/K2NDk2g1RPdhxAIE7n8nFjcHC/H5ynpbaD8gGsSupOp+J5z6qLVJMNETzxpziCHY/sX2glBC0iv5fVpUVkQh+fcl+wz1Bcj/+4slQh+s6VzWL92S8+S2g9l01HK8xa43l/gwoowceU4DJESGpKIdKorzXxhIfeG+972TP4ch0OTaUmd2aZUtjtHz3pV9SGyWH9m3/xRvQKRa6AY5KTwuw6/TaNmUMCjllDJPkjlvxbJkfw+fe8DOfu3QVn8EvDXDvXXqVzhEch1okm4n/Xhz6NP3zxGxCNjvN8JMQ8SX7w2rQLlD0lVmpbdYDRAUl8Cp9mBGBUAqN6pXVtKobimZJbZ3aMkpC4n8GAR+vJJgnGP2/R4DHK9SWQEwdgcaKxge7WH2/Wp7DaW1HvG954qgg7hOA4R1oDJCYvJ2dboaOzfQvlyQU9mTV+QkjgBGMGRW8k/+54DMZhHLH4MkNJIQXLr+Q5vdFDggn3wP2HYEHlbwRzCQs9XMa09g5fZyMYgyIi9Df5+6AHxJvTHjNnobck/S3hG+4jUnWpvn7cZtsBF+OgHPykg3azjolqDq/7s8fzmYgQc1GkWCDmIhTzOUYIFIX3G79yX/WB1/wZImrCEfc4Pdck+8jkH5EtyFDf5gJoU8ST7G291RVsGq/hC//0Sib+drujSPKnUrt9tFs6iMXE5GuW/G//l0/iD//2jciNUulua9CzBj8mTCvtuSUYIQJH25YBKIr8oxReYgIcGWjyC0LYspnbYPQUz8mFRws1jI2JmaWZIW2kK0lmF4koM+fRqXjOfplLYZGhC7+rVAmcoBJBWWKM1jjikXVKoMpcAT6H7V0n+gb/NNsJNgbJ+/an2UAfp7+kb1BsEiZlK5qg3+ghdvYPPfAHWZstyxTCcSF+QnHQjU04tcDIeSr6R+Iuz/lmHLgul4oYl2RhrIPSEU6CmCmlYpmuGIevT0gEGVOFXhdy6yLxTIELNv81mnjYxedg+6Yplvzcl9VDxAAtSV9vd3r4zUdcrN7/yj9G0miYmgFsGHTII/j4XZKP/0z5DVRZqL6OTZPD+Px7Xolz9u5UnXZHUZBPuMHUcjM2OS4Vbn5LEq0LhTze8O5/w7NeTMRPufw541LjMF8H8Umfp8/WwMcuU6MasEAtFqCpdgBV9CWDI8Nw6670woDmQS5ZUkXJPiMWnSseSk+cYgncshlMl5AEaECe9Y64g2T9PtckdpCtK7VGsmpwtuNP6pReLXY/GAwslYnslcQal2NPCeeTrQJISS6Qvm+q4Scsy4OkbSADU8RkOpWgT+X3XDhL7VKnHcRQ7F4tGy4rIGx2biRPcc9KVZ3ebS3trCWqVA6AGR7FBgQl0cTwU9Veqphri3E4CMlpuUEPdzqsu39eut02zt29kb8jV1qUz1F+dxgoT4O5I6680+nidx79QHS7Xfz+i96CqGyj4Wy7c28Ac5YPGw7MsH9tDZunRvCF97/KwP6Uqy88K+Mrd+/DM+bwWyL+d/0b/uIVb0NuegPr8vxo6S8zACPNUwiAGQBHmBsGwDkbFDhFHgIeHPnircfSwV5zfcoGNBgk9GFgNYyCqcjLwh4AHh2iGnlK7J3b1u/Ztaydaui8AbZgimS2glLD2hxo8Hb7eeIIapJUJaUe75dd1lNmR64HweC0+jOgKe4voDFIltCyHNUR10BCzw48hFgGT6IwaoS6iX7Sgq4ktIMsfJKBQe7y1PgxddmU8ihYj10ILmhmZEQMhGrbG4npG4AImmT/M0umUFmXOuS6PUI3VxeUIy3/9N5cg38nKzfZFNx6EhNNZbTM+ILBk6B6t9fD0379Icx8fv8Fb0KOymu7ykephWOuQYFRvfVVbJ4cwRfeF4ifUEXfc/TT5YJl3TwR8Rfwund8CM/7+7chN7XBRPBZnd9BffprMkmNi5RsMlwLkB0rZAi0+j95NEpkMDSGOFUp2crFMnORdrUBVsIA5kqam2l0xJsAVcMABixUvyZy/mGa83WpqapzuzrCT9mu/CJOqRupuIZUCC+NhKMf7JHWWzRIWHqjqqh94QwV9kzZ+wituD2t2EsJL8bP7QaME7syskR9nMngXUwvuxSnlPED/FcUjfDncOp9mpG4s8tknrQ+GCSp+cVKv7W6f9he0vOHfo3J2OpMjV9MTPpYdl5c3u1nSkcHnkuxAXlmAuQCcxV0OALQbaJpiGMQ7iGHphFWTcjlceuBOTtdSnGRDXssWdnlvDsLP7nBOp0env64h6LdbuNZf/s25IeM+9M12HSaO/nzydq/aWIEn3/fq3Hu6Vbnd7BfMm8vBcUccUcdTcSvX/3WD+AFr3kH8tMbkDDEt0RP0+T1fbsGSNI7yE+EzwzAwH8T9DMETWoAAR46FxUOrTdsD4FgzIsKlIlpkROPyxpeXZIOBwHZ+xgfFUvUzBY/RR+5qlOCmN63GqZFOnVtksvWnMIZALPqbjb2RaZFu3TrDGL2J0irmcaNK9QSyxDkUk0dlRKG9s1x7VU/sxHQYVxB5J6IxbMQm49Nl4PxYljuLRmHkJQpHScQdyDioFi4GZFLl38nib1kfMHse0/xFHtOOVdiNeQ2UMMJl4BDOrpJGaG/KYhGqav5InetNQRuuuPoZtM34mQG4O6Jm13aWH1Rg5+r7dJv1Sou/841mF9c4dTgrlUb/LryCo2Ej3SrClSP7/88+VK89q+fjt7SKgfFGFejafzBBr+1FWydGsUXP/APOO+MU8jIyAa//skwE2Lm10ovDqbSKOTz+u/e9D684NVvN7Cfw2+JsI0+TtCf9H8f8ecs/OTes8E+7OOnF+n6tQpUrWKs//wdMYscsLYaGpfyTeZQLNe8ADLDpQrPRt1xDUrc8ips2Gg1Jjd+uaLCvfIKtB97VLmZsjc5TdTB8L5pSf91cyQBqVy/TkCl1prYWQgwN8+mnqANEU6NVBgmfQjUiRP8z8YAcoqasmV0Q+vHPt4AVJbYM73TUqG8aUguLbthJm0cPH8lmVHQ0YK7xI07BywsmXSvATobXzlTj8jtFW2krDAqymEmOuk2zSi47ZXtfmxLy1J0ILWs5hNRZVsi6FbLBP4Q8TICIKMX1dJysNbGBHCBTkOgpMeShD581yG86LXv5QhEhvxCFUh5nsXACQVYm4D6y2c8Di9/3lPRnV9Azi4S6vPXW13G1qlhI/n37iS0YIjfrjIfLCURmluQ9kU6/8vf9E/429e9G/kNG5GQtd/q+S6vn+P8ozxH/XHylCN0+p6I2xE6vcgdODSEhIx/1LaJPAAk/el8K+smHoDn3Oj/XAnYZTy6abBM2XQpsiXNkEe0eXMQHaH2q9RAeXNLilWApVVREixbYUquy8wal2qnFM8eqguEkJZUoqCHEQy+96V7FikGEdSSlB4atn5oexLag2fZV+ChKSYmsuv8bpKziT5p7l5S18vqRJn3fgL7SVY+UCe5aRElc0voUssnkihyvmSshLQlEiOhWJPpDdyiyljTyXhvdCpum+2SSPh/qhZURLFMWYPWyEd/SQUgJiDHRUExzjVnM+44JZcr+xrLPZ2/MDaEd/3b5/DG935cl4pFhv4m0EjENIh15aLaaSM7RafbVS9+9pPx/D9+Errzs8gXC+jVG9i+eQZf/ODrydqProX93gJC3gyvQApVzDEAIqlCHi954/v0S974PuQ3bkTs9HwmeBfC6xhCji3/RuLnDaTOO1+/fRUsQ5iktmAloEIuwDx0iQoydkwDVRoTvdcxJ2RRN6aEiqO4GBEGVUZiU1KWQWH0GMtQ27Z5H3/YO9jYlUNPnH1p+ffRRYs4bN9I7wVx7jbJOAQlZ6IixYIy37n6GFnNOa04Z0jACk3PWOw5M2Qiz2tAYkJ6Ub/E+7lqAoq3Hnan1H5rwEjNkITZFr54jhYYhpkccSV/tYA2gktGcmNXC9ASpOOybjJocS2tIlmxdgDPaV3cdpB8IdJBmfJ5MzPcqcbABxOCavzSTp9PTw2XpbZNNJjIKXhnbcUH//CNUqkw2t+6BjlXQHTr9fUWEo3C6Aj+/JXvxDv/5TO6SKGpxAR8f7wwLWEqfUIAowHyCrz6+X+Av3j276Jz5+3YMTPMsP+sU3ew5KesPAd/fK1Fp2/KRWWHTurIC/7hXfrlb34/8tMzxPZM0VPS15nYLROgCD37ndH3bWSflezk7jO+/oI24b8F6KkJ7ppEWXj8e6UIvbIGrFIqtG1IGscolke4NHi6piOpZ1QijAx4tL9BVNH4ECIqyEIhFl5iCxXSLSEYQ6ALT+/eNet/S3uo3FpJCy7z1jan8Oe1HyTcyCJNYT9wM+6RtV/zIgQ5pTlnxKZHKp67nORIwBTiCkTnJ8KGdmbrwBlUJKC/ZwRpCBWs2gFDHc8LYIW0X5nSBiB1eH5stACpF9+RBeht1HveGMySFEMKj8CoXbb77tQ0sGMjMDsHqBqn63LiisundzfIEqjHi9NMha1ERIt2aQEYmwz7kv6aL0KzRDOGv1B6K8QLuJJShaEK/uhv3sTJwc/4nceQj57dd8EqLSSMmGN7jypJEv26F/4Rzjt1Oy46/0ycccpWdDs9U8bLRXVKN2GYXMvIzPkJ9v/V379Nv/Zt/4rCpk3cW9GH3VoiNwZfC//ZBWhSfhVLdOv7Z/hPrzwfrnWiVbmi1PQ4UKBx2HwLqgY0uwjq386djXl+eihRSzYuSSzWIhlHiTnrGN3mmhkDUf2GrchNTiLuGvdtWErmnjxiVeGWuVTqQXreTlqnE368O1S6po2RJfjzJZFKoSKowy0H6c3xnDYUpkrZAFLIOgVZBWNytOUaopw8G4CbCoH3hR6UyYFK3W7YMrXUUj8J94rnlnKnwG3MQhcSYJBNwA2VFifB8LuoMozlzDatyd+N0Ejcd9x1pphH/rRTubAETSgVonDJOgw1CfxbfY0ZQG0MKlc046YFSdF9ywueQRqVJG/VCtfcwrWaTic48RhsrjdZ85/5wtfhgx/7ki6WjDoQtmw15BAiZ7LiEtXt9dTvPuFSnHnqTjb4ReRLtPYXGXGZfSQmZJokfw5//ep36Ne+/UPIb9rIFXS54EReSH4b4MPBO2TRLxhJr0nlqZSE1He2AMsMSCLPTABjNc4B4CCgovUW3HqXQUmUZm0JpzRMdQCsx96iJYL/xi6ToNemlGzKOmwid+oe5Iar3DDFk4oP8rOflLXrsx6gOHK7e2jOlCJLoQw5O0KApYKIHPFLWJx2x7uYAk/cWfqw5xQOzsBHsg9IXksgjcCASieDATg3oKsqK8cb4H3qloUkCihZQu/+OQoSPAtl3M265KBQcy2lqw7AZQaFUGmfBN3bDtnePYYJcAiOKKbrTuIMghTaSvvmzqVyUqYIakyhu922KeBBBO4ZjVmE1KeeilKyFKdFS16DtVWg3bbps9aISf0C/Ep0hb9cHLlRA9xtMOSnqL+hGp72l6/GRz71FV0sFrnKjp+a7CPzZhITPUi/k2GQGAH5+f20O2OutX7JReZ1/nwef/H3b9evedsHWedPqDCJTewxEJ8s/ZaoqZQXh/ISEVMZL4rqK0CTb58+U5GNUok/o5BTKOSVIpff1g1QRUIKOURkH6gUWX3Stx2igmhAc43HQo1YyiNTSOJWivBo7o13tYu4Q2oafdtF+YJzuMaf15pcZ2CzRr2tU7tQB8qlWF5D7+i8YWBuh9RMHw9VC908dVygyBAMNDiGyH/voa9I53YELk/r2+pJ3SDrNfjJ2wliBWNj8tLseHdgOXMfGacUfDfzA0SwaRVhn9YA94toQx5GINUJ95UJRmFzDz3YG+8MY/fps/b8Xhq6MtthoRTOO8dYk6mZSK+LboN0euKH1HXWXJDjBigOv1BCZXjSttoi3zX1AmhDL81ZlGgJfogaWBhvQMg5SWuUKWDIUXUFhsVPee4r8akvfEuXSpYJuDItPhosqATBTUphw4rVAjnnwaDl9NV0IlUhn8dz/+4t+g3v/DfkN22muDrj23fWfA7yyVtDn7XoO6bAbr4cVNVKfyZ+iwAsGuAiH5MTKOzYjGJecRouW/wrZeDIIrBch07Ik9JkeF8emjLNWGlePSEpxIzGgG5zHTqmpizGV1+7z72dkheWWzCoO70J5D6lipcUkLR+YA7JsUV2UQY173gU6zLv0vAiFXHq1pxDe5KCB3ql5LFmIfJhPhoyCA5Ogc4gNxEPc7KagzovgLghacqwN+4gTT+TzMD4oNz0/ywRQbgbeSHhXmH9sX8sEhE4g2GpgPjmfYipMw8Xm0gz7DQGsAyA3G8toHrWOYimNlIbHD419agjXTem2n2+wYA9Vseojs8EonTVa48dMvEATtyQW4zSZGW4qaNBh258cofRq7muZJRHr1DAbz77JfjCV79PTECn1AFuEpmuKSWZpKlsEwpaBstxWIyGT2mC/fo5f/cW/aZ3fIir9zLsZ8Oeg/Cm6hEZ+6huP9fuZyu/8d8zoyBCrpS5xRqXR6POzEVSE8jVV+Dnl9+9BYXJEZODXykiIq8AeQB+dKupxkYSnYYbd1Cb2GRq/olFQMRGaIyIq722YNYItWWfnMbQeeczCGPEkn2+ol1YZMUOzWTz+juB9bpNIhKeA4uWpGBP27XCmLIJuqk1mt2ySDgLsqWuL3R+K3oEwg6qQxjOSVEBjjfq7A1YLisxVep3eacZuOLEDguhNHPwpbDdks0iidQ4HBII9eV5LMUC4gOz6Nw1a4yC1HTCdqTxmxUN4TSKm0rkt25G7sIzkGCN7QAdkjIcDGQy+QwwMkRIdoPK2EauTMvMiXoEkKqwQOW+2jZGnmICelDjUyYQxvro+cX0GfF/0JFWTKncZtOcn+YhX0A7ivDEP3oxvv7dH6FYLOquiJIzzo1QOjoRJaxcubG+9Wbvl8emSG3P6+e84m1487svQ2HzFmMYdcE8xLxcyTPr3jN/jcRnaz8TcQGaIhGJ4MtW8lcLUPSieH+KG6qVUD19B9kEdVQq6FyxoKl4KyiA6oc3QJEvliois2chj6GJraYnoi8/R1pWD902ZVT20F5fZq8E0ER0/vnQmzcjbtrKQNIeZ1x7/mNE0219PZ2rbgkNS2wGYzoJyy23TFKbV6GyaDSTP5M5Jqiykg76Duhb80HLSKe0iS4dJ+wFOEEGkAu2cqnHeF+s5FL01uffpeCRvAlTFMKO1S5yjySyMCprVuzTLdwDMo9GzjURDZbrKr72dkOqzkg6ANu4S/NZbMx/7qEPpaVhJH+LGnw2Gc7HttqMXwpxD6XKKGpjG00sPwUONRaB5jrU4qzRkd1lqMdgbQwRtf2yMDOYRlymnV0enkqN6pIvVbCugcc980X4zx9cixIxARe4Iiv2eHUpDctcSXTJ9Fw7LwoK+pOXvYWJP79xA2Kux2ckvgnuMRV3WCUhA5+P5rO+fnLzcVjvsIH9BUv8bNxj3d8wBwWU925HZcMYryyKXWBbCpUA+/FdwOwqdHcNaJEa0EN1dAbl0WmjevF4jeTocNHUmOsxUvtwU8Krg/JDHsyQnqMoLS2ktE7BACO6dC5CqxujddUNJhDJVfPtWyFOXcwgzkGIXurlng7k73IU6avJc4R4k+ySl+e39OTJ/2Q3ByWHte8GxEpJqhNNKDwh7snDdNeE0qEDSzCp/WUhB3GC1NTcXRSh63Tjy1+GyaK8Xuqse8WPuaUGN3ow7fvCuQWCCAyYrOhA+aEPBvJVU80n6aK1ZtxE5HMOvQrNnFDTjpENO23orgJa60ZVOLzP+PyJGRF0pu8mZqzEcmRKc0PGBhfJZofBgpkHSD23dY/Sb8slLLa7+LWnPg/f++F1KJdKutezFUxSgVb2SKkOCLuBR7fkbSjk8eyXvBlvfe9lKGycMcTvdH7qVsRhvbZYBxM6vcJnE71XhBodMWG9JPmHipb4TWQfwX8izGiogtGLztD5XKTzJSrrbZhL3ImRfPlKE024aiA9uVTHNu5GrmBKuxN8J+MoS+1Oy7QPX18ydRCIQRRrqD7i4ezS455/vv5hkCuJ1orYe2yfXVzIo33nYSQ33WkqSMlUgz7CvBshlOECIY7ACUWxLq0aYZWL8LxSp5A2L+dqFzYe0VnPxQK4tXKSbABuL67I6AclDUfHt4wGQj1e+Kq5Z0uywg3Wf57UATbuQHBB6Tpxks25XGgFlPJIvn8DdDtWJjJNshi/n6cbY3OJ0GskqJx9HtTZZwG6zkyhtTpvFmGvw6XAfKFIUN36NirjMyhUqrYeIFUEjqGXl9glyBLRdeylTLWxafYWGEibjleTmIC4FiMDm4hCSUfUqWih2cVjnvZXuPKam1AqFVWPSpC5ZxIS+ax2FubF7GJhv3X1keR/2/s/gfymGSYMo99baz6H9VK5rgD5fSgvS/kCN/HEcAV6qAxNrryaJX56URwAWflJ+nc7GDlrB6obRo1Lj+ICSOiOVqF/vA+45lbouA6skfVfI1esYmR6J9tdpH2D1TD2/2u0VmZtkFYdhXPORfXss9FZj73+L4u/8L1z3A5NqVEO6H6b378RWFwzzC5r/fccdMA6lt9LVODWpH+MwgNmg6/CmhfFVo5ncGSylsLRibqwkv31aPHyQzpZbsDMvZuPWU6YnZlB32Z17yCt04tfXjQlq/3X6awqt2/6bMb4lUBVClrfsh+9Ww8iofQ612FacM/sqOlzt9NFoVRE7TGPgEaDiaDXXOeXkUAUeebmwXSYpdbhjALIRkBMoFk3nWwO7guFlbjvXAI9SSggb3IH7GI19QdNvwHfOCPzHw2bmEChUsHcehuPftpf48c330lMgJmDG5IpJe5XarhLGquVomTwe/ZL3qzf+v5P6AL7+U1xTWfwozZbjFwKtloPEYhnAmTcKxkkM1SFHqsBZTIA2tJeDhkUI/bxU4BqaWYcU/c6k1Ib2V6QkFeGQFoxp9Tnfshl1NTSESPRu22ey0J1jPV8qTYSAiP/P/Vj7NSXrOGuheHfeDR7Hqi3I6tNDgUIIO1oNkfBkJFBAu2vXeV/cFmsd+fs8m9T79OrKDDy/mXs1WlGvwP1g9T6T6mpAwbjbRUu+o5n+6R5AYwbUOYyD+RUggOmmKE7Rt6b28trB/1E6PbpM2f0+wf9XxeRZrmEeZFlemlR975xJU85a0sFZ0yS/tvQapPgJhEDraORxz0eKA2za4+kd2vlGENPCjxxgSmmBkOkqaDE6ObTkCtUTGhwcw2KMgQXZoHZwwA1GCGCJ3cWEdPGbUYloBBaLvBJvimunhkER+r+rc0kUqQOoDBUxdHVOh71tBfqm2/bT3X5jDrAtEL6jtRTjQyhe4tUpCmr75kveC3e9k8fQ2FmBlRPyBG+KeZhdHoTyGMlPhXuZH3eGPxYNaCipdND0FXrxyf/P6GAooH9psKPgf+bH3QRirWqMXraeo16tKbiH9yC5Hs3QhU0sEYl2Y1aOLr5dMRURckJHQ6gjNFprbPa1Vw6ZOF/DyiNoPRrjzXd1dh7ItVC0zOJPSp2JnO0VEoFNOfX0Pre9ZyV6JGt1an9Ypb2KXu2gNqyKoNUc936kqXqJGOW4fFCRZBLO6XCCQDex3wkdzmZNoBYdgazGCzLtDI3LzP/JPoxhRQFlgsjzmxBsspiCmkvg4DJWcIXZZV4yIU8ki99z6d0k2BzBBbilJxG5ioek37ZRXT2uYge+kuAXmGXV3tt1lSgodZdpIfyYqHKQz1FbiiKCiQmoKklGI16hdphx8CdN1NUjoV6FoJMbkI8udEWFTWlxQxMNaq/q+rj7s/MhdXHqQZBAhSHath/bA6P/L3n4/Z9hxkJ9Kgtmiu57WMCOLiI+RWFFD/r5W/Fez7yBeS3bDYZD5yzb639NsCHDH0cyMNEbxkBB/dQB6QCookhqA1k9Ctw4A/H8ZcpCChn4H2JXpTLEGPmgtMwsWszNUxV5DKkDkE6n1dJrBG/+4smdmKJOjJF0N0Whqd3cfAPu/rE2iP1i+L/qRhrk+A/e1hWUf7lB6F41pn8zLK1/hwNOkHEPBQG/rf/8zoTLUrVlFK6fH+IpKfVYH0Ka3PQJpe5388ZJLLMw101U+FPQn+hjUhU4AyNttErhbym2xP9nCpAcIL6BeXcDoII3Q17/iMMFqlBZ4ss2kWdPVdmAn29hWxXoDBzQiczst4VjFC1KjS5lyjCjIxYtteEHJd3ZSaGFdBbgtQkuWpPewoRuOkERL3q1xeYAFkSSaZOkYa9Dia2n4lcqcpQWxEKIIPgygLUXTdZ2GySadj2sHknRwhSj0CZU8FIgCzvFIttCmkqrmtA9gAXdx/lTP+94RruPDKLRz7lL3Hw0DGUyiX+3kF9em6EGOi4fCGHZ7zwDXj7Bz+NwuZNiDl7z/n3zYut+Vym2/r56XOZIvmKrO+TsSzaNA5MDRvXH0l+epVyUCWrAtD7SgGJjjGyfSN23ets7pREwTd0n3FPq974ENof+hqSa/exBR8ri/zsKOlnYsf5xl7COVOmnTknOjHTjRiJJZ26XYM9jPz+U/hxsz3UOzpSmXK+CQ/dslaGAbQ+/W1b9dfqhinbVjbaxn00Zc5kx2GzXi1y6EtZ90pp+kRSaLslP8gVKMFBn91BIAgXi3eC24kHAkmp6+81XcyQDTHHNWakOIMfsGxcIY9I1TkTkD9AEdcnT2KhzIOz4+WqPiSxFteQfPkHdv4S9gY4iWv5UWAC9qSxyqG1HmPi0l9D4azzgZh80wQbD7F9Iem22RbAqN0yRNJXC+URTGzdy5FsfKKVo+wl0HfdAVWvc06AmSNiMBHi7aeaqjqMQI2k57shJMC5Nq4+tLUPuIIb9kXEnR8exq37D+JRz/gb3HbnIYoT4HBeepGrrVjIIx/l8Iy/fi3e+6F/R2HDlAF3pMOzj9++GN6b8F4v9RkFFKGoBDm1W9s2BT0xxKm7mvz3FYsArNGPYgDoM/VCrU6P4rwH3gtFDlYyPbmTuKfIaNi94SDa7/wMUIqh5g6YEuKdFsa2nIXy6AxiSpzyz9sEYXU7ddb/GwsHTA5CsobCGWdh6NJHol2nqkpcHzkIWB8f4zrAUgchipUponlkEe2v/4g9Fya4K7VavTx2aoFZOyabNLX0/fuQcRjklMQMbmlayZYx2rqRBqO2WM4S/qdGKEuIubV7UlQAYQRMzcyA7BE3GKe0Sv0k2F/EOdLfiXAf/y5VllroZuEgMTte17UMKPMdn5WiAj/7bZO2T64majsv1TyHzNjmQVFlZlzddg+qWkHtT57BlmYivm5zBc0lImqFTmPV+9INQyGPQAujW05ndYBaVaPXhqovcyKNvuU6czHXEIOuVa4h3nm6x1W+Jp1gZPyWbATsl+d0O0MANiEnJiYwOYVr9x3BxU/8U7zqbR/CTbfdxdWF9h04isu+8A088KnPw3s/8w3kt2wFRSFwGC9JdGpqagmedHxXpdfo/sZ9SepBNDUGbJkEKMmHdHgifB/ma2L6yTUYlYps4CvUirjXA+6FSq3MtQ7I6NajOaZahmQjfdVHoFsdqNWjQHOF0VOhNIypXedx3r+pshws5i2aQ52guXwUPUJWLMqbGPqTZyOu1dBtpeG/S6pyyNOEVhu3cQyg8fnvA4eOMXqR0lXmmgSUKtfwIPXVxBy4owJBZwhIIgb5S8qiePeacagNIFB0QDonvJ1YOrB3KKRht2/D5TiizdTLjiLVPdootv4eUxLYJ0iafWQqZJ8hzN6wb1XCYzGFOb0L0JdYNqm3GKog+cG16PzoJkT3Pt2k7XKdLBfb5ewbLqaOC+ApMog1GgnyT3oy8Jo3Un8uIKqgPrcf5ZFJNkr12i0UKN3XVg9m916+iKndF+LwtV8BchXo9RVE1Qkki7NQxATOuABoNwwBE1caHUe88zTk9t9iW7y5RiNpPmA4WugCpNi5njAxEPrN1yqYb7Xxwje+Hy//53/H+HAVjXYbK1QdqVBAYYokv5NAtpAHWfxtg04Ui5oTdhzDJQYwXIEiKz8F69i0XjbwFSITD2CaGvNvxBjIPkLuxfvd70KMj41gtUPETEyqB9L5e8MjWPu7f0Hv+zdClWJg0ahUFDux4cxL2O/PSVfcGdU8U0JaXVKlkKA+e5fNqViHOu0s1H7nt9FqxIgpF8Gaeu3m5aJbimTmIAHQ6iVof+QrRt2xqc/O+BckurO5uO/ThB8adzgdVQStySvfrY0go9B7Yh6gH2Q3f6xQ0X+K7sAnqgL086zU2IQSnLJwikGmvwgC3X32vzDf9hPBE5ydRMdsvKssPaiACxyjsFchY1G9gfhDl/NngqeUhx4CerxhRxhwqe2VQrvZRXlsDKPP/VNovc6LuddeRWPpiInTZ6u0mVIbVstpw0PT2zG+7QyGtVxPYOEuE4Jy161Qh+4AqJQYjY3LhfWgxzcgOeVMY9ji5CXbJUfkBrAB0boLWQ91hMuSPM+wPlepoLBhA1qJxpGFZax0Ei7kkR8bNwEwFLRTKBn3nivZRf0MSVXKk6s0QkTx/JNj0BvHgekR6GrJGvYI4hPUN1V9dN6k97ItoJBn4i+Wcrj//e6FmakptLjGXsRMp9ezxP+xb6Dzz1+BGsoBNA90j502Rmb2YGTTbnTZw+JgmXmOrcYKqwjNhUOIW/Te+P4n//rPkRseQpvqB/Dyk6A4oEfTq4DeJugVcmh9+xro710FDBnPTEqCe6NyEF4BVAfonjI1ptqEZ9BrlnZ82LytD5naLyCetCSVBsEMfUnbEY+U8NVJ8wLIMjthNKlhpHSd7P2k9aXwt9/o5yz9jgJd0EpQqMTxgvj7dCaHENx1iIMTlx8eQvKZb0IfWUJC1umCMfLxcOX5nIeSo8bIq5rj4JKppz0NubPOAeIVJs76/F0cokpW7h5FB3KATSjqQfno07svQmlkEglJNLJ0Lx4ylXFv+CHU8jxQGTLZiySFKXBofAp691lMnDx+Tru1xinXaEO85+hCG7TjKvCQPYHsAtQyLKpU2JjHPYyiAjS5QIjBENynl0vppfdUnntkBNH0JLBxCpgcMcY9JnwK7KEQ4JyV/LbAB90L+fqLOdbRh4ZKePDF98KGqQnUOx3EFJarNZrdHjq1Gta+dz3WXvIBYKgEfXS/LYXeRqE8jI1772/bexuYa6ogGb8/nZuCf+pzd9L4le4tq/y5F2H8t56MToPUC1O82jLxzFK0JTy4z6jx/Xff8xnTtNXp4X45ppR6+8c2M8kswZQtLiOA+tajh8GSaaTClDxd+VD7TFBSyoiYQSNZWjqJuQBccjXctKDuEME7sDTw4L/2lJ6T9tkrpLHFHjhIBRAvJ+3TPNcHzhjOTm/Ikn3wKBKCfrwHdZDpMx76EwdmrdBux+gMDWHo5X8LralCUI6Jep2MV7kC66fGaBXcpGS5JoLbePrFNgeBSoY3ocjdRdL8uu9Bra1Ak1GQLsTqQA96eBT61HOB6rDpMUh9723dfWYArq+es94Tk6C/VHasVOFOu4q8DaWqeRWJCVAwT4krD6NaRVStQg0PIZoYg5qeRLRhCpiehJ4cNck8FMHnCnVYPZ8RAMF//kvqgYIinz+VXui0sXHLFB58/3tjZGQIjW4HXaXR0gka7Q4atTKWbt6HxT9/pwnJnrsDam3JoJm4h02nP4A9J2SM4+BTu64oBbjdWOby6+vH9jGyMuuujcm/eymSchntjuvXkCpdExaQbWtAyT9xuYD2dfuRfOkKqLEhA/9TVvssFaeJLo1HheQdoIP7saQkdz+m76+3MZhwpEALBkQpuPg6JysS0G8+WTw70HSes2RGaW4QIvcy+6e6n2YYnmfnYuIyQRQp3cczcJnFZZmAUxlo92oZ8Qc+j2S1yZIxKlsPUGZzjIM2LvevItRXuhj5jcej9KuXQscLUPkyGvMH0Vqd5XFxVlrKAEnFgVooDU9i5vRLuMQVL1JKLabFT41Cf/RNUzykVDZ3xH7tBLpShd57LvSGTdyxhjlInnLpXRVexwiMIZC/d9Z8tt5Toc0qUBsCasPA8CgwOmpqHY6PA+P0dxR6fASg6jks3cljYqL5iPAd8bNrj6C+rebDDT6ozgDp+8TkdBdnnL4LD7jX+VyEdK3XQQeG+NfbHawNVbB020HM/9E/IlmtQ63PQS8cMV7a1jpm9lyM2uQWTuyRFnd299WX2B3YXptHk1SufAm6N4/qY38dQ495NFZXu/xs0rLSPUPtOyvRtHDdP3I4vv3jwNq6UX8k9dqjTZyGEC5iHfYb96TO7nQFsS598dkQFBQS3yx1pIzbKUtbH5YJl3TXc+d2UaBedp20kmCDrRHuYqmZH8TlaEIz+pD9nOWcKeOLnNRsRmE2vctzReHc97DOGQMN1qeYgOSGW5F89KsqR751iwI8unEPyBr23aVJUHR6mqt9Tb7+tcDwMJSmtOAEq4duZKs1FaggY5XJTHP6HQUMNTGy8VRWBzQtciqmsToHUDwBVQ264qvA6hJ0mRp6kHU+xzaKhAqanHI6sOdM6FKJ4TFH6ZE0d5DfxgOYary2tx4zABO+y+9t0U0jxQ2E53Rdl6NP3xGxk1W/4vanij7Gn89IwMJ9YgQUSclltCi2f7iCB9zvXjjrzL1oxD3U4x46ClhPYqy1u1gdqmHh+n2Yf/obkcyuQtVnoY8dMITcXMPEtnMxvv08W9HH9WKgNul5ZgidTp3Tr9cO3WSeZ9KEGp3U4699jW73gA5XZw+hKi4zXlp2VJ5CKsjbUkD7+jsQX/ZlqPGa7RRi+wt6ondLe4DE94xgAH0Fw1EfnYRzyNJ24pX6mMqeSTsc/DVcM7MBHYNMZtlJZAD2Lvo0nIy0D8E/Mi85OAdYt5Ftk0Q2WgoxSG9CuLwdQ3hI6d8GfSFsBz6azl60Wkby9k8Aq03OD1DFmEJwrQFJnsbVxLd9PhBhbbWD0llnYuTFL0QSL7Jxsddex9rRO7hoRbexxu4s40q0qhNbseuY2HkOxrefwwYvluLLR6DWZrkZRfK9r0HNHYWqDpmLM2EbmwJmNgHnXAS1Zbv1o8fQ1IyE3HdUlYd1f7IDWIs+STYmeBu6y+pA2aACSuW1lXpYTaAIOO7NVySJr020nyV8YhgulJeZAKkhCeIO9RGMsPesU3D/B9wL4zPjWO200CYXHTQzgdVuF8vDVSx8+xosPf11SBaWoRrHoI+SyhRBN1cxvuVsbDz9/uh1172l3cy/8fk3G8vcDmztyG3otVZNO7FkGdN//wrk9uzBylqXMxeJAZiX6afH2X5kQyCHDD1FKf3feBmwVjfz6x901Cd5pQaRCg4akJXnbIDyUPk2rOiADHznrAytBqEoobHYwctSwawMddnDTpoXIHvRIIn7C3+6Mlv2NoMVU5hTLPcTBOp/9FZRe64sxHKZhVL3kb+nVA6RwSXcOKmgi6Eqkh/fju4/fwlJntxnVD1Isnlx3+5+7KQTEyDYOf3c56D4gAdAd+dZmhE8Jf80Geyaq/OhIYhdXLSRa2vDnoswtuU06FbdwHgKFV49wt2B9A+/A3XHTaZ2IIXi0pi5hHfCwUx6914oYgTEEJiXJabQCUl7LsjpGnFYoyAjgTIUW/gtnGemQMzBlOpSFL7LNfk5xFdBogLr36dXQpFzcRf5vMK2U7fifg+6F3afvhsdKr7cbjPkb2qNeqeLVQCrQ1XMf+g/sPJ/3grdbEOtHwOO7jPemMY6xjedii3nPAQ9KuvlyprZaj20AOqEkBSVVTiM1uJRKGoL1j2K6qN+HUPP+kMD/WklxrZBKL9smTbniWGVih5Dwrp/53s3IvnYV6DGh4PrT6wbWZLb5wNYinSiLTCGNIEYU3m69JpXm1NtvZ1t6m7ijB1xpRB1gP/Zw8L+fl918noDmsBJwcVC2Gt6MKGunPWk6zCJIXfZCXjz2XWszdgMMnpZ0DbsQ/ODkTnV8nv32fhpfZdVF4NPD3+ogs7/+zHkn/BgqKkhHlDUIgljTmLCAczNurG7arnNrkZRF7HxPe/CXZfcH1im3PQKVg7dglyJGlhU0VxfRGVkWvJmPm+328LUnos4P2D50E1QVCmYc9o11MQ2JNf9CGp1GTj3XkC5bGoLsjuQ82ChR4ahJs8BGmvQs0eBRbIl9Mw8EuFz1p7dn6P8rP7OqoNL5TVEber1m3JezEhY0pu/lKXneylQPdORCma2b8DM9hnUhmscJr3SanobS7cXoxnHaAxXUF+pY/HlH0Lnsu9A1QrQc3dAU/wDoZX6Eia2nIkt51IvQyr6QbprCPihmWquLXCkfmdtCasHb2Ijq+6uINqyA5PvfhvWKbYq7jcLGQEhSIui/oqEBKjqb4LuK/4JiuM/SuwH5t579uVFmhAwHozyWk83ZjFZfA7euriUrCHL0aTU94WNL8Us5F+5g6QLq6KqTLcsK6CtdnLC9QBOsDuw6x0n7z6jAjjCshORqtprj3Vl+sKtSmwTJkgyNA9wPGdLzVpmP3HPrluvhfSmpr/jOiSRY4K9Wu+7C/pNHwFe/QccsZerUnXIUCTU21jceUkN4MK4OayvdTC59wxsfetbcPDJv8MSKul2sHTXdZjYdSFPU6e5ihJFA7peByylKIS4h+nT7sOtsxb3XwdF2YPNVei5O6Emt0Lvvx1YmmcmoLZsM0VVaMys8xuJhrERqKlxXtB6ZQVYXoZuNEIvQtqRDYdGmlOxTgrxZfWAoH3R6fWW+Hl9U5ci48IkF2FptIKhjeMY3TSJ0alRFMl12oux2jaVkVkTpbodvRjtYh7NagWr37kea6+6DMnNB4GKAojJ1VeNnaK+gpnd98b03vujTQjISmrukMRRwmT0W2ZUkHQaWNl/LfdaVNS6J6+x9QPvAzZtQnOJvC3W7Wc5gDmPfVKumyXH+JD0L6L3oa9AX/49cB8CvscotcbSdiQH8VPSJ6RoZyG6v7ZcvUJiewEkVIHjkqhY432XcCm//RmEAaGcEO3/VO3BpT0lQ9fSOCeJPozclaju43au6Kq4Uoj6E0pUSin3oZSGz/O/WZYrH6iTLFQfRoQIk1uQ/M1jQ4jf8zHkCAVcdKoJSyXJaGt6+mfnmIGdf0pOIcJZXe5g8rd+G5UfXInmG98IVdiIuL2O5QM3YHznBZwsRH72QmWY49dd/T86HRUQmTzlQuRLNczd9gO2PEfkEz96GzA6bSzR3/0asGMncOF9gIlxbjjC/e/I/cbrVHPtfYxuAk7ZauoftJrQ63Wg0TYGRnYjGsMiG/GI/xFB2Wq4XCeR7GDVIqKRMgqjVZQnh1CZGkFttIYiJf6wBI3RbrZsh2ZT6jzu9hAXcuiOVFA/uoTVd30UnY980z6GBvS+/SaNmuwYnTa2nv0QjO+4AM3GkjcQ0XwQkdKsNNdX0LEdfpb2Xcu5Fmx36B7D2FvejsJDHoylZXID5k3XNif1Uq57+8DYNqoRU3DU7Ap6L3u3aUEmEacKEaW8/gbU/vBLzy1ob48S7EMKxWzgW5Ye3YJy6m5fq1d5XFbVtvfqeYNZmN7G7ofEvu2TxQDEeCTns7q5rBckIxPNZKYCK+VTEnMZLCgpQd93jC39Zd5TSZeA+VOKUGgGGQaftlqYgB3bxrrZRvzSf0Lu31/Fo82VAWWrT5sW3maBhERI20KcPNFRhPnVHmZe+xocveF6tL50OVRhBt36AnsGyOBHRSvo6vli1fb4s3YBYq29Nsa2n8VuwqM3fBPdVh1RsQq9dBiKJOTEFmDfncDcLNRZ50Pv3QOMUG8BSj+OTUaga7tNwUQUPESRbZsnjXRn1YukHYUK28VOu5JKQFV6KgXkqiXkqSpvuYhcuYBcscCnJDWpmWh0GxTG66CrMbTxq5BHb6yK7vwamh/9Dpr/8jXoA7MAzd/8IUYwbOxrN1AslrHzfo9FcWwrGuvz3E7dPkZ+0WeS/K3mmqJgqOX912qO9aciq91DGH72n2H62X+EuZUOejov4vzdWrLvfQAfzYVFW8QA/u59wB37TSMSkmfZwrDKeYqORzcS7sul6YzZ/ZmxqTBhf5o0cg3q4XGkdpYWxMEBEYS2dxaInHAk4Am2Bov7R+T0omymsNslO+5UTLuQ7hgEj+Ru6YlJ92sxoN49E8lFTQvuNL8J3NmF60YGBYwPQ//Hd6Hf90XgGY9Ar9FBVM2Bq29m4IkFH1zjj3ERwfxEo1UoYOpf/wWHH/RgJD++GaowhdbyYSxHea4NQPEBupogTz0EObY9FFehCMLy6BS2XXQp5m75PtZm9/PCV+114Ngd0MNTPI7kh1dA3XQ91BmnAWedBTU9YSQ61xIgiafY6m0IwHYw5vRc6rlHjIICd6h4J6kFJoRYuVh+bhSqEHd6iHsxevQb18y3ZMFdyxLuOKSrZfQKETpHltD58FVof/w7wO1H2I2oklXofQdNMxmanPoSxjfvxrbzHo5ekkNzzRG/LcxgDWetxiraLVNhaXn/tejW56zR7xCGnvBkbHjLG7C83kM3sc/O9XOQkXT8lflk6j1oxJUiep//PpJ3XwZMUlAVpf3a3pmRQIp+bQ2yizsSHaDbuzLrwnPlScMb+/rXsAh/cV8OBA/BYCYWcRb9S28a/0O1AE6sHsBPgAkPygPf6KFw1m+qyV3/pgqVrtYJNwoL4ZNuMl3ctEtgcb876SlRwICGC0LK+kmS+/qp7dNB/B9fEltaGXynHVLeaU4oFZALRTGUNv3kNXSnC1WuIPrq24Hdm9gtFzVIeomnI3U/l/7rvHVIUBstonTgVhx8yK8i2X8YKk+ZgE2UxzZjZMvZPC/FCkHqYRMJx+LK3QDVXc0hny9g9ejtmLv9KlNshCL4aMETfB+dgSqPGqMZBfecugfqvDOB3duhy0WoXo8iDkxKLUF68tWznm+gPyMAK/kNarBMIB8hx8E9poOySTeO7CMl9UCxesBhwPUW4luOovPV69C9/EfQ+48YZhI3gIVjXAKNPRvtBgqFAraecQlGtp6DJrlGCc6zW9MyUdZGcmg1V9FsrLKKtLLvOnTXF01Pwc5h5H/5V7D3C5/BkipiuWFtOtZ678N7bbNXvywKCrmC5nyFZH4R7Qc8HfroInt+wlq1BSFcsVPvKXJ/TXEWnUIGUtewDChVrt6vxLDO0xLEEq6LdJVH6H40kI6KE3YB+dceS8jSiLxIr8919Nyt5wB33Ooz5H5OFcA59wSxiRt0LCiVkhuMEoE7ZiCUyJ8wNfHcVLiJkJPgPmfQgxhVSgWy10npVo45OZ5Cui9XpIzZH64XFqD/4s2IPvlqM2NVQJGeaTqCCzbtOomaU5IAJijeWG6jcMqp2PKFz+Dwrzwc8WFyD46htXyELzu6+Qy0KXU40abRpe0t6B875RP0EkYMtfEtmL/jKqwco3Lm9HsRauEAdGEOinrkkeS/9sfQP74J2DgBtXsH9GmnQG2dAShfn8pzEWFaV6Fnpm7slqHxH8rO4yxKcieSVyDyVX9zVIhjfhXJNbcjuXY/4u/fgeT2o2xbiHJEkHXoo0c4vJmlL7dRT7Bhx1nYfMb90UMR66uLpjMRJ++IllcaaNQX0KGWa0T8d17NqhNL/s5RVC/5JWz85GVYUGWsNUgno0rNaYe7IyQvcmgZ5U3pFqLh3nPfBn37XcDMtA35dfq6IK5BRjVv8HHv5WpziGGQBA2Cb+CvQcKljgh/5fUzANiiHZkmb25H2NuMMeCEbQA/Yacn5IDLYhTOepKa3PVhFMtkzaHMFHuoNbVaIvf9+PzJM3n5KYCfJWTvyMsMLQCorPmhX//PfmeJ1LkPbRtu83KUbZCBsTQDenYB0Wuei+i5T1RodbUiyLlqW8ZLpmKHx3drpyGXV6CUm/GpIpLrr8aBSx8DHDgGVZiA7jZQrE1iZNu5iHJF5AsllKhDkDNC2Xugajm8rKM8cvkCuqtzWNx/PZbn9psZzZeNR4NqAlRHoMtDlo9Tsf2KCe/dPA3s3gq1cwag/P2RKnStxCWvOKTXdOg1xjW6Z2pnTtCYVD0qyrneBA7PQ99xGLh1Fvr2I9BHF4Bmy7gPdQuqvsCVewl1kIeCSnhRIdSJTTux6cwHoTi6BWtLs+xZoRh+c3smSSpH408S1NcW0KPqvq01rNx1I2dUOslfuORB2Pa5T6E7NIr5lY6po2Ar/fgnbR+EY+b0Pipb+8RQAfoNn0HyDx8HhhIkZFMh5uaqRHHDkSgYDSLx3iIBF3TmiTYVgyJeKSoSIkiu0RQ1p0S/fe+YUlbwOZWbrR5iXcs17j+bkk/rc12DAO685SchgBNkAGc8SU2e8mEUq5YBHG8iAgNwRrbwpOSNpQN3vAyUcyXZuh2qUwu8FJFc2SISbxRIIYJM/gAzAmIAVi3QPZa+nCNGUplSZz/zJoVLzmCvQNTJ64SzfV3RD39bBsAIbSifM0xgdLKI3C034iAVqbzlNqjCBi6JnSuPYmzrGciXqgyHy0MTiKhluIFw1idN2X6mPXehUEIhX0Fr5SgWD1yL5blDSCifIF9imM7qC4UFl6kuXwUoDHHtAdfwihf9UIXz+SneXw1T5d4KowO2zFMxTa7M0QbWGsB6G6oR831rkuqkUhDDoPlaX4JemYUiKc+htzGrSmSAHN+wFZv23ge1qVOwXq+j3VgxUt9BZOv+5H5+3RYaa0s8lZ21eawcuI6bqZDbkQJ9yg97JDZ/7MNYL4+gsU7BPkbyU3yG9yjZJijGoWPnrWAqM8bDBWiqMPz010NNTSFXHIFuzaF34MccDakoItJGlSomeJN6rSQj8PDeCbpMAk6fpJcxKGJdenVXMgNxjCTilF6QamSegv0pr4FLJTZ/HQPo6blbzgH23XyyGMAT1eQpH0Gx1qUizm5yUiqApAJ/ct+pZADXG3TpDCpwzM8vIjeBoqrKgJBnySD4vZ08u2wGMAGyC9B7MqZRgZkW1NatqvAfb0Y8PgTVTaAaSgfXoL0+d5W0eDRSJkNXaUVIIJfEGJ0oAofuwrHH/Tq6P/gRVGETMwFSF0Y2nY7K2AwTYbE6zKmwskFEuA+zMKjKcLFQRNxcw8qRW7F4+Ga01lfM880XTYkwjho0kX8JRf5FBRNAE9mUX6o8TIudjuF7th2FOGKwaK5L5cx7RppzwRRyyVHYMklfQkr0PXU9Qg+l6jCmt52BDbsvgqpuQLPe4LgHt/CZqYmmozT39HuL7AG6i+b8AdSP7bNEGEPHcxh+8m9j4r3vRRMlNOodkwHJjyk0+bTz7dcKMYCI7B0U7UfEf/1BJI9+AVdeVrVRRNURoJdDdU8F61f+EPFyG9FolSsUGWbLNddsToCrCShySnz9hQHBQoOiUt0f6R7vs+6lgWv22fs17t2Lpg1dShNJrWVGB+ROiVCf6+nZm0+IAZyYDYAqKISrCptc4HCGE0v9xpXzFtzROJD950CcQj+VSMHPpQ8tTBsLg+UzPVw/R87SHhZg/54uGswa5GgiqT7crfvQe9YbkPu3lyIhd1uFZkFpDiYP5wxRS9aozfVESXPIRVhY6KA2sx1bL/8KDv/BM9D+6MehoinefeXgDeg2ljG0YRc6zTq3HqfutxQY5B68mSrrKaD2V+069x4cP+UiTJ1yLhfFXDpyO1bnDqLdaJgFRUTbadpSWi7xhAx+RcMMPK92EsgtHjufVK+A8xhiH2hjOh5TY44ciuUyRmZ2Ymr7WRjbfBZiVcb6+iq6C8eMUY/CfJPY+/VN7ZYIvW7TlvOi3optdpF2VmYVlyLrUfbAmh5/6Usx/pKXYLkeo9kyTVdco6TQ0kzklpjKbYxm6NGQx0PPrUH/3muA1Tr0KDVo6aI3P4uR+12A8z74HNSvugHXPuXv0D06y7UNfbi2IDzjlXbRiUH3zJqjAkqVWNPEGKTBqV3nbn0LZpCqquGejSMJx0izBkKLglImC2d344A3nPB2okZAQYpSrw7czwzm+Fd2c+SjtdgSKril56QS94uJMq0JxFD68FYgTHsNk3jkOLAdpf/O3YS18vp+gYr1WkwO6eRTX1XqhZuAf/hDxI0OcrVIqXXFFqbU5Jt7J/e1tzPy8ybD4FoHqjKGTR/5GNbPfgnmX/oqqLgMVRhCY/EudBpLGN64F6WhUTRX51CsjJrSYvb+TG1EpyLluBU5hcmSDluszWDLWadgq+qhU1/A2vxB1BcPY311jhkCw3u+cbq5pol5kPMh0JpHQb4aExEABTANoTa5CcPjGzAytQ21ye3Q+SrarRbmF5Y5joGEDun5hkisYVNr5NiSnqBdX0azucKrp706j9XDN7NHICpUkHRnOS5/F7Uhf8ITcWy5i27MFhUj+YUtzJV/C8oceTsomMnmQ5CU/N2XKlx3PRn9KA8YyfI6quefhp2vfwZyvRjT9zsTF3/u9bjqd16EtWtvRTQ1aW0LdF5q2U7p1eb6YaUMWtdiHQkVP/CRAFekPcAsw0BDfafOLCyHWNOSv9+16K4UhO6JbSeuAkyc8hFVqna1swFYTtbXsyzFFKxk9XcrLid1I6EWDFKjPPGmjrf/pHCU+N02hQjz4aCSQxceNgUpyDo+qQQUV29LdC+sqfxbngf9B4+GrrfNol5VHIynhDfAmiDsMGyEIlXWYg9cglI+wvRoDuuXfwGHn/YHSA4dgspvMB1vKTlxYjOGpndwEBD5yYulKtsJHFJyCSXu3k3SCbkxNXL5PIrFKqhnINkMyJZB3XI6jRV023W06qv86rXW0O1QTf0ul+rmp0fwN1dAoVhArpBHoTiM6sgEvwqVcRTKo2x4pDj/drOOTrvJVY9pI2lP+rnLQ5fdI02UYAed9honQFFQD1VPai1RUxWzl07mVfm+98em976T4hr0wmIbXSJ80vV9WV9R0DMlMEzcg85rLljF+QZPeyX0Bz6pMDVNcRQabYXiBedgyzv+AqMTQ6jEsXFyjBYRL63i2me+AnNf+E9E01M+otBUXhLqgFdrpY3K2A5S48naBfpkk1z/A9RWB/Z8lrsLLJIGP6lWiDUtZp258dpsT8/edBJtAJVznoDR7R+lOADDc4PuY9o1yezD4BnwN2705RRzGKwXDRpOerLScxv2z9gLzcJJlVvKGgLpr9X/vSHFvqe/xACMiqJ0o43cB14OPPYSYL2jIwJOq8ba3KfqyVs3wXnGsh9pNg5OTpWg9t2Jw3/2J2h9+nMARqEKVehuE7niEIY27ERpaJKNUfliGcXKMHIEk5008EwtMGDHuKgWHxXcJGlcLFVQKJVRKJaRKxSYWHNkpnBuULOZkulm5RjIzXbBGN1ul+MQuqR6UAUeNlBSbIA1molpJNjLhcw5ZyzhYygEmtAAxfPX5/ehvnAEOk4QUZ2D7iIz4MnnPhszL3+FWipV9OKSKRpq+HHw9Xvp54zKLuo0b0J9CfpjpAD9rNdCv/3DwASpWBpYbyJ31tmYet9LMTQ9hlI3RqlSQDmvUEKCYiWPnXmNK57/Nlz9jo8josIoTjawvSR4BAyxi3UtioWkiH6QkEstTssgbQxJCo0OWPUcLOZyHbIGwKwHwHynVZSL9PqJM4ATUwE0o6kBkMjpIZL8Au9K7erKLokfUh1/7PvslHjQ51QID5/S9Qnc0V4lGxRf7TisTxQSlXUZ8to4WcZyHPRqHjbVunvGy5EffjX0Q89HvN6BGslBrZm4IvnUUiqZMeTzcyF1hDL6Zuc6GN60C1s/9VksvuedWHrRS6FnqbDoNEvmlYM3MuyuTm4DRqgldg+RyqFQqhjPAesYwZ3JvRms8ZcIKMfuMM019Ej6k7vNR75TcwHuO+gKYJgFS/q6aYfu5sIscmJctE+O9HoPwKXl2kQKOqNit9VCp11nfZ+QTXt1DvW5Ozg3QuWpHVgPSfcYihdcgA2vfz0KD34wDq/GurVE7ddzSCiMkWsouKw20VxTuP9p+NoSvyLi/8u3QL/9X4DxDeaeVhuITtuB8tueDz1cQ5u8G1TcpNNDDjSXESZUjIKK8LA3/Clq2zbgO3/7DiiunVi09g+rBnE1HGNg7KvM4Y2FbnGK5ZYNfPOqsxSA7nSBDpyqHNazjEcQ0l6AgyCFxNg4t/2k9QUgSnFvU3cpmJCAwp6C05MQMuwCqJNdfX1ml5DanrwznGFAL7TsFftNCbyIrOS0RTbZlWYXevADCwhI16GiGdSH/rf+GvjuDVBDRe4SpEfIPO/GJ9UPR5/2fi2oIHtTrE0tgcWlLsae8Ydq+/e/q6pP+z2lk2Wl42XTgLS1jtWDN2L5ruu5xkC3uYZWfQXNtUV0uNiI7ZUnXK22JVS4YU4fziEqUMxBkV2NhAIolFcau+kclNlI8D8qFDg+gQyN7Lu3ZbZcnQf3NBjVMCMxuQyt9WXU1xa5eEenvozGwl1YuvMqvgf6XakCdG8Wekhj8uUvw+5vfwvRgx+Mo/MdNDsJW+E5SJOBjKzsI+bTfSLipxh/GjsR/1+9Bcnr3w2MTJrnTLEJu7Yh/+a/gRofQbfe0D2ldDdJdDeJdSdJ9FhJoVyI0EqAlVYH937ub+Ih736RiWlo2V4EjAKdmij8715ldIhSVPTN+uizvnzfbzD7MvtI9GjIJIMOHHKQcyNjiv0Y6R8OYf2J2wmGAp/7JDW5/cOqWOZQYAMbHcGY0wSJ7E7rIgTdx0yihf3OMC/JVORbaU/I1Hl1SpNUvqVBwdsNJNNy0s2dwqkAjkp1Wi1wLkIuuqEUGk2o4WFEl70K+j57gTWyCRQ05Qwk1O1CXNJBc2rpIwoiWks5dabRUElP1YbLmKgA3a9djsOv+Hs0v/p1rkwS5caNeSHpsOQvjW5EZYR82hUj6fOks5eYsHPc0svFyKefgpmNTIp2n+plPCB+XznP/gmQ3mu+IgZE7bnIc9GLKXeAUnfXuXV6e+WoKe1FXgdu60UGyxJGf/Nx2Pg3L1L6jDP0/HqCRpOq8eZty6/guvVOIS/ubeIOzSDh1UhzOrYayUM/7y1IXvduYJhyInIcx6BO3QX12r+G2jiFXNxDvlrQpVweJe6KFGH7dAWTFXpqGkNKgXqRUuXm4WoJt3/tSnz5aa9AZ7mBaKSKhPorcnl2Fy8g+k8KNdi/9/MqvQdpOklJ/rAQjZE7zfFST4HRqGc4IjDIPSPDoJ0N4CQHApXPelI0vuvDKJhIwHQMgGzC6YI05GQIYJmaMAvv3VMfZDRxRkZxJjlHbmqMTjVI55LGE3HHtkVVKEZidSvP7R0ktu/JYEY2AYqfrzeVqpR09JHXQD3gLOj1Dteuw5pGImowpIK6HJO00M1GxZpEHp0oqrw1MVlAJUmw+ImP6pXXvRGt7/8AQBFRNMJWaXJX0XUK1SEUh6dQHppERD0FVI6NhkbCF/j506J1PQM8CxBz76MPhWrl0RxHIlp5xHCcDI3x/9fee8BLVlT7wv/d6eQwZzIMechBkgRBySJBuSJJoogZuSoqiqAIKphFJAiICOoFVFQk5wwKSGbISUAYmHhi5/p+a1WtqlXVPTg+xvv0fbPhzOnTvbt2hZUj+8zZeEgI36iznt8kNWNiMSojb6A2PmztE5QsxEbCYSYCPXvthkmf/zy63/EODFeRLRqu2NzjZsYp+TqMVcOAB0wGL4psta45PgPqJXDk99A48wJkVHCF5jwyhmytNYBTvsT9CkHdnKh7UUfeFIoFlHJ5zJrRh8k9RRSMAeV6dVJBY5BNwCCrN9DT1YG5j7+Aqw/8Kkafegm5oUlces1Ki7b7ksQDBLhWTUOcy9XaxDywuTNX3D+SZjWz0owtxQlXAzDqSKyIgFBRAgCbC7DRMiAATgLo3PDA3KRVfs0EoNnMWzE6UEOx9AcKlnCZlFNHlFIpMrIu4Va6UovXdRLEVgQgjj5Um9TiKdDmVh9F5cZVRIACWfzfzjBIiTHkYiOx+oJvAO/ZHIZcfYR4Y5lNHvJSXYJgQu2jCmW2wxfpmGSg6p9UQH+lhpEr/oDXfnoOqjdQbj0h1CTrxycPhWky8hc7e7nOQL6rnysQ5fMdXHHHqjV51t255Tg3Fs3bUtvOqGX3SoqW6tdWB6fio4ToFKFHBIBKgVEEHxn16hPD7NprUFcjIqD0PCaaY9Rsi8XxSfu8F1M/8XFkb98KixtU77TGLcHI/sDP4m11bbt8wpY7T/ev1cos8pPXokk1C0kX/9jXYX75e6B/itP5R4DZayA7+YswgxT0Q4ldVCvRFj/Nlwpm1qwhDPZ2otBsoovLHVJ2BQ1tso4sQwcR5XoDHZ0lLHxtPq4/6AQsuOsx5KZOCrECLnw4GAUTSaBNYFzkLpcAL62wehhPchHa6UH6x8O1kl6ZAGQ5jMyrmXkvbgQ8+eQyIwDZ4Cq/ZhWg6bIBhft7Tu3KMvsc6CX5JLURRBtVBPPFZKVbfcfftYCqJSmtS/lPgjig5csW8TfJrhJk54+0RKA8BDS18gSbRvKnfzkzB+4EM1oznEVXofoCTk+j8fzWK7LPhx4CQ0hLYCmXuW8THaU8Jg3mUGgC9TtvxaJfnI/hK25E4/VXrCiOfuauHIpLOQxZkSWAAhkKO/s4apCMWfliJ+vyZNW2P1aMF/esBzpfSsquj7LyyIVHRTkpxp9sElT0lCr0kvgvVIxXRAFCIGtoHl0brIfBA/ZD//77oTB7NuY3gMWLatQF2Fr49XG4Kr5cYs3HOWg+Ya38lNPCHYL7O4DXF8Icchxw0+1A/yAM3TA8jNw6s2GO/zwnQXHQElU05nZlRPiA6avPMAMD1H25aZsWU48TahWZZ9rCbSGocSkRBJIEil0lVEbLuOWIb+GVP9yK3HRq/+bg0ncQ1p2DEs+AqLy+tXsiFXu1OIFFH/TXmobuMCrktEREQOL0mk4FWGYEQFSAjfbPBle+GMUOYgeuEZxaaBI6KeJ4MHhoxFO2gRQxolklSB2Fx2qJ3tmnEymD7NNWCuMNSywnrVzZNgHUepbaXG8McuHCXJqbSszWgJGJLHf8x5E77mAOFc5qNWTNPMwoNbQIhEBUGW8O0WsXwYjreriKPaaBrmIOAwMFUJUyihsYvfE6zL/09yjfcQ+wYJ6bZxdy6LH9AfhvF4nGXoGCVQu4aQj9LvjYdhsUJEjvjHxUuadBiG/FfSvKu41zHYspss6AOD0RgiJKq6+Orp3eicF99kbPttuj2t2FRRVwubRG06lw3oCsbSSBacnhhP4NbB5gVx8HFg2VkD34HJofPAZ4gqolUSalYYMf1l8b2Zc+A9NDtQPqnEzE7cp4jBwG1lrR9A32OuTPbEkEIQA5ZPS7RE1MmQgYJgK5BuVg2Katd3zpLDxz5u+QTRqw0ggfoqQRx+7AYCtoYxvwjU4EBxJU83DtIFpLkRHwWgIQsip1ghBNgdKB51XNG3/bCHhiWakA6x+QDa52EQqOAPgFykbQvarkdmIDsL/0b8Wc00lEBpI234+NAPYVS/RyX/rdZAOFyqqgGu9fjTZTSwXOUOjEbxuAQ5FAjhrPW4TcQXshd9rRzKlIJchRvNQ42fCSbEZvjIt0mlZmwdtpJQI6v87uDvRTng/x2hdfQO3u27D4htsx+pd7UH/mBaA8otw/FAyUVzHuomvL8xRB9u8L+DlCy1GBdfdDeo2NKqTkms4NN0LftlticPsdkdt0c1QHB0DZAaOLm6iWqfEJEZ8cxySw91cTAG2k9Z4fBT50C7NiY6P7BvLApbfDfPwkmEWLuLUbjzc8imzj9WA+83Fb3ZjLpLnCJtw4OUPPurPQOTRAyG+o8nshl5kiDCfKEvJ3UVIkkTEiADlHAFgdMJwGTcSYvCMPnvpb3H/82ch6u7h9m6g9OlYgqimgmZzKLvT4kQYFeVVRvWeBNMCjr6irviO/xYXCB0jJQPNrZt7fNlwaArCUocBC+oSziz9TDlZHScWieGxZjrm6z+7T9+gxBGDaqRJaOAg3h5wB/4Q2xNZ/MUQSiu87Qnx+NlFbm3VnD9mOYO1rroT1tCE0L7oCeOoF5M4+FthwVTRGasj1WEAE1c/0Bx8IkQ5u9p+49ZPQwg4El7JKWXETo4b7dnZMXhW9B6yKyQccimlj46i88CzKDz2AiQcfRP2RJ9B47nnUX5uLxjAF49SSYAVb6CICQP5x/fj4fQqEKaE4uRfFmTNRWHttFDZeHx2bbILejd6GbIWVUKPaqQ1gbBSoz6UyXXZ/KNyJ10a0koeWNu9hyz33lzdE6yfJhwJ8CF57SDgHzFfOgfnuz7mIadbbZ8ekjj5bbQbz0Q9ZGKpQdiSJSfTcBnLFHDrWWRW5wV5u7Z7PZxnhSMOYrEGhUFbIAsk01rRm02w4DIHjKKxUQNNqVqrY+LP7ojhtEPcc9QObQNTVyUFNIVXA4pg1aMfSsPSIDGHpck8s9Ya+otr4pzICU0EgwgUdiOfDRP8Bc/d3L40K8Lb9ssGVLrEqAFdZj10ikfidWu+Vbuf/toUgE964hAi+NhlXioGG+uqJsS8I3JHuHXMhbZFNvutdg2JgUa5BJxGwd0BeEwAuGgF6epD/1qeBI3antHmgUkVGW0bVrlwfymDw8XtiBWExnvj6bsFCbJlIqEgk6cOFjiJ6unPc26Po+H9peBGa8+aj/PIrqL7yEmpvzEX9jTdQGx6GGR4DxsosEnOh0GKBxefcYD96hiajNGUKmpOnITdrJXSuMAO5ocmodneDNH1azgQlB47XuXBJmJ9NQPEBPH72msSJuysQaBvEZLk9de5hFx99Z3IRuSdfgfnM99G89jagf8BWEiIsHh5Gtv27YA4/BKDMSpLGXDlzflKpgNLGqyFPdf9dth97XVgpdAWRKA2bCGlmy2bYNockBRiqhYKOXMbvk0TAGQH1WtbZ1WlevPGvuP2Ib6C6aBS5/j7rIWiXRajtAyIFaHXAMRu7K0qSVWBoS46rN1IDIRNcXfFKGQFH5y21G3BpqwLHImvk89RipNbM1cEzTGuxU4igQlz5u21HIyU2Rm+LoVETkpQIpNYCd59mgPJVLakwosuB2oiw8AWyZLvD56uJjApq9PfCVCpofPKbyN3+AHDKkcDMQTSHa8h6M+SIbZabyLgefeCErnqkGAzc8ynKz22UJEO6wESRVui9WqWJRWUKW6aGmhS1l0OxNIjSjEEUVl7Ddv1mS0E4QZm18H3Zgpz7u0IFfOvWnkbZvw3KySfkFlXPcz0rqXBItJJk4411CO5NWJKhaQIEFmmcBkx3ibIugV9ch+ZXToOZNxfgsueUhkzhvcPIdt8JZv8DbOESHtvFBlcbyFFB041Xh6FKz1VbGi3o1m79JkOd9q1BEpZlCmzxIc7v+kEwWlL3JUNTMygUCmZiooKVd9oM77nyR7jp4K9i9OmX2U1oKy4LUIs67Bcew2Sk/6vIFiXdRzAaEKUVplsqamu1cllXBW6YkM2oxUcPwWqykQqgKJb/gn2/tVhi4MZZW/XBy4zx+FFP9mQvkrGzdk56f38716XMSWr3yaBO/NNdWelG1kNLwNQONC++CtmfH0bupE8i22975v7NcSp6kUNWIW+BWw/TZ/EVtZZ48vTR5b1HjgxXEMPukgVYjjgs11FmT4QYOIMqxEU6JHzYb01ATpGSfI1bjorLedxQ9ib7y/IWK7O5vmoW8dw+u8mysuQi2WwXZGK9dE/TtiqfVAKem4vm184FfnMVTG+HrW5EXgeaIxUx2f8IYKd3WTes17sNMpIEqKrx5mvB9HcBlZq1A7gSYDrFjLVl8q4wQc08IgW4CCBLAUdWu85QKhRQnqhgaP3VsOfVp+GmQ07A63c8xG3USR2w2ZNuQ51qGBhEG8DUaKNsLxqiNGzGxXDxJve6w1vmrcE85UnEEC16a909DYvUE02APCUPfqFKRI9KuEWLVoipNQUfrBHTyGQajhgrK627IeinWrXRhh8dHCI/rugGfZn8x2/MR+OwE9A86FvAs3+3SStkoS42AKrsrcO1Rf0RiJStU4UwvE7tHRI2ApGMbeSqbkq4MTcJIoQkzwDp5WTRJt08jybX1SOR3f7w36ynk0haANV7YS2PPm9Sfz6DOo1ftz8uUdKOzy25LA4K8vMlThc/6YCErOdTUE+hiSbZVyYV+TXO/BPMTkfC/PYqGKreSy3NuPdCgYuoYLVZyPZ5r+16REY/t2dMyHq6kG25Nrc0J+S3SG7QYOnE7iH3CeT4hiZqjSaq1CWoafiHqjpTg1H6qVCRo0aGasPY1/J+06Cez6MyUUXnjCHsftn3sNL7t0dz7jxHJJ0xzqmM1qCs06sFrpT7jglFaniO0U3uzeQnxTP9hWA7WMbNQTmNzHEE33w0Bt6Ym2sI1m+3n1NYT6Lve7lNj5+EBLc8M55XmGvSi81KLmGXxP/khnJitjMXackgyRWg3xQJSD5nrspjre8kS5OxKJvcC/P7a9Dc6ZPIvvUrZKOjMINFNCi9oKMJ023I3hbXqVcH7HdDKslyaaxkyVaCEHblK53xDxMJ97uZOSJBBMNG4kkZACEeDMMNuT+YPvzY4jKUwqItzVQFbu3BsWTAFn3n0ye8pvx9Iob9ReCa+2DecxTM574LMzpiaxjK+ijEmXRlqoD82GMwp56BrH/AEgCKOCRD6UA3Mird1tPNeotE4fH8aK94HTYeg36RJkHrJo2tSj+G6rfY0u4Vh/Rlek2V9U2GapN+DDfaop96IYcKSRidJbz7Vydh3f8+gO0tXkTWlabSPAIpDCJRqHK+ft8UlxfVNsqNcYzdEYZsSXiGpc8FWMregOz1drgYShunxrhYKxHxXKVxefBQECNfUzXalXE+fCWJA4h3TvcH1NxUwvC1aCCPscJoUDEc20ouH2Ys2rP4rERcdlZsFif5HtJNHdFwxsNs8gBMdRyNE89EdvE1yD57ILIDdgEGisB4EyZHAEMVdTNkZG2TVFihuOnu6W3mSkSx6CoZZx5J/Wdp3Tn5FYtZUUy6PENre36Lw/6H41cCLf1DOEyWN+L2dMZ9VK0YMHfOAU67BOaq2+1zpV0Xr4ducNZz8cz0DwK332kJwpEfRbPeRK6/iNxmq6BJhKJKxkCSvshrI+V7QrsMp9nz29xKwdkv4BZE5yydhH1nNTlj7snouijx/MgwSPMzeOf3jkTPytNw33Fnc9t57snoOkKHEvmOCCjjYCyNWjzReRg62S1RWNveE0EI39yxLAjANDd6zvb59EEbWKLOzwUaI109IElYSCsyajdZ9Iz2S1dA7Chryt29fiXAqlWOtHOhxhLpZWTJbIT8Ws8T1YPvcKnEmvi5GAEO86SCo5TTP60D5pXXYD51MrKzf4/sEx8A3r8DzKROS7DJh04RKWylcq53jYxeMmmjN6lEQOaMXh9N9J2IIPhTizyULTYU/TibRR2fjDSIETrC2pDl+mzXJ6Qgbk+33f04srP/gOYVN9uCooNU1ZhKAorL0xlIfVq2cLwGzEAfcOstfE/u+M8it9agVWeqTc5BsIZTKcnE47hYY5c67MZriMEzM8hxizibXUj5XlxaTGwYfGNYG9NTt3YuWkQdoipVbHbUvuieNoQ7PvVdmDqFGpKOQ8RMIlsFZjSTTDc77YkZPgv7HFSC1u8neQRLeS2lF4C0xBgg9MFHU2Z50VlDo2+F4pBtMwhF3/FVhtRGqXEC5dTiUuxr5o+5+45DXNk4H8OQLkae4zZSRypKddZoPd4qZw/YGZPCsyTByD2XWY6Vt6k0tyE/8lPPwXzqFGRn/A7ZYXsgR4RgJYo7p1L7dSBPkYk5JgaEHOxCTF2nXkxUxyJlDVyGWYTLevG+6qzeWccvHZEULhiIoHgCWgDBvs9RSrQNxO0NQLH7XYAhKefa+4Dz/gjceC8MNRYd6ONGHexP560LdhjZ6tDyieaaBznzDRkGb74Z2VAXmucdz2N7a50rXOqAQh2gNWJrtmsROrPxCxTw45qL+h4jjmHzLATxRdMyGRp5SzRKWQ6NiSrW2X8n5Ad7ceuhJ8BUq0Cpw7lq3UM5hNgFjjlYiSz+bp2RFKDJg1S4EnRrL68GPFpKG8A/1RtQ9tP7z+VqYdIiwjohux0T9yKMGsTje/K+om4skot45gpipJthb9WqiBqt7VxCrjuTnzbTalF0/dS03UIRRHYTEit0hIU74tikGYbGnh6gD7Z55pdOReO0i4E9tkN2wE7INlkH6EdGcQRZpW6Yo3v5NPjuvCAQ7UFckKUFCiLYCNblUHJMBZ84Qs1uPq06cBluKzaztE7BOy5tl1qLm96cVZJeWgBcfTfMRdcB9z1igZ/i9XvIAOKIlfQI1PYfv+dOfBb2QWdPFs7Jk9D8zdUWSc8/nmsCYoziGpxHlRBe5CVxhLhu9fQfz82o8Cfn8SWa5e3B/FgnBwpTZULu3LN0sYRD9WJyWDRewUq7boltfnEC7jjoRKBA5y+SY5BUvRTA0qFEErnZisgSqVmOhvn4niUwAE8UbFjTMrIBuIuKrculkN+Di+LqgdO3i8KLpYXAz6REkixIHpD2Vgv99MKIKUa3vhPNO/WOOHWhrWqiVtmWigjXt3K3e99JQG6+4UepCs4CTMBEVWjQ3Q0zvBg4+xLgl38ENl03y967A3Lv3gpm9swMLgkwI38/6Z4UYy+4QcZG9zs1nPiGpl7LV6xDi2OeqCvC4MbyxFQ6aLmIVzbsURlNQoquAnLUr4Tunz8Oc/tjxlx6M8wt98G8/BoyqsZDDU1dGrbFAxUeK3q4T6kNAUJhb+kO+k0G1jqyaZPYY8Bh+eceb+uZUoUPUgXoAQ0b5+ePmMHI7gbLZ5wyYQI+sgpgoacu1asEQr3tzXAikbfDuehFGqqYz2NsvII1dtsa8445GI9//VzkpgypdagfZzOJ0+cDAof90BiV6GRt3pJz9Yl5y1AFUGKvHKDMVx4VEoRCZF5AlljzCXpNe+FAH4Dj/C7YxNMQ3SMg/W40slFUOFWC3T+tekEYQy/FSzbyff23Azwd1C8WYJfhaItwWvUk+OKdRb1YQja1wwaW3DsH5s5HkP3g18g2XRfYcWNk226CbK2VkQ1SNLu1DzQrTe5ZwOOxJ4D0WRdEJGIlERm7iV4YCfsj0onbB53b4gDdAy9zO9sSnbsLdTgbFzHel+YDDz4Fc9N9aN76IPDs36j7h6EIQ0wddFsgJ+8SZughrk6kSInSK9FNV52hI56ePRMRaNoQ7EuuQlaZQPbzE61YPUGJWoSVzljBRVlEE1AqTkYifMYh/VI0LVOBUd5Tx+K/NRAKyBRdCIOEEHNZh5wjAvUGVv/U3nj24mtRfWm+DRtm7i5Skohuqs+kYIRrcCIbEMzUioE4fLAJQTEoWuuVJHot2/bgjpAF5FH8IeEq7n2NNLzY7E2Cd2KRNBTQ0b0BFG2MKOU/onUpqVGAL8yvLdGIFqRKmKt7/McKOnw8gZBBAkSnCkhSjiMYlNvuRV0uvOEMYX3dhg+xNoHmTXcB196CbKAPZvaqyDbbANhqA2TrrAysOhNmEiX+uBmT/YC8CCwlUMQh4b+1rHtfhlcTJe069E6wgY8OCLnBqO0e3KQeI3I3lev/+wKY518BHngSuOtBNOn36wsMudyy7k5gkLg9glXf914Qq74tmiEw4dULn4uRHqGOa1HCKK1z+hDMH6635c/O/zrHNWQTnKzDkVGWEEojRy9w2q0wNl5A15e0MZ/Ka0CuUyU4Mk11DU5tjIatUMSD5wzq1QaK/d1Y8cDd8PwJ53MCEcdLiE1DzV/Dc1C+wuUDwBy8+3NyzDBSOh1TdBm4Sx0ItJRuQC4P6dQRLX5YY10soqRIEyNpiB/3NC5BZq0m6BbfuhmIUMIlzzzYAVIpIVCUoPYGQIxtiam1XYtq6uHi83afhXlKJpgbRKQBrmop5NHNj5mf6JfWYU1FRDPSmWm8egNmzjMwDz6B7Lzfw3R3AStPA9ZbBVh3DWDNVZFbeQZAovFgb4a+Esf/pCjVTmaS1ciKmDRUgfzoOPDyYjRfeQN47mXgiRdg5jyP+tMvIpu3yMYLUyB9F0Xt9Sti5vDMJ0/JQ1z1HI6DT6pHi9hLn3krd2qAET1avUUS0/SpMJdeiyYVPTn7BBt/Q5IRFy20GMvylgRQKG+oUcM3pDSXs/RLsRKRAiSQWSqWk1RAyM9lxUgdkhIGAGbuvAWe/95FXG7M10LQ8CJ44+FrSe7ydkruEk5TKhiLOLHMJIC8i7j3LgiRdAVsdFadE9k9VxQy5jiOcvmFgqAaL+U9ZwyKGqlrBIy/JxvghRF1q+fy0U6G8Psg5id7lkY4yjP1a7XPvt23+NNbJJTAdYM+qIK2NJdQMqgNPShw7Xs2HNIbBPjPE1I+B1O/kTlgo7uLm2xg8hDM9MlZNnUAGOo3hlJiBzphqD02tQyjKDsCSpISqKbBxDgMVc4dphbfi2FeWwjMmw9DSD5vMcziUZgqJfyS2bvIRVJNTxfQ75qqMDbIlohOG8qQeWu3arEl91g0CPtkhSOp/a0VW8I2EdKlfJkzCzQatF6YS65k/1x27vE2QahsezMEupEUlTPO5k9GQwFj+kfcqALTKkqU3uGoQof8HE9AuUgUSJWjZigUIQl0zpqCwrQ+1F9f6HsRxgSvlTGlFv4orsXth2VsDr4EdJTtxr25xOSf/1MjYIJ1Icc7VZ3DubWTCNwSfPy5+sSuTC0yepx7rZE4JR6JriAjtUNY2emIgCiuo4hBcEUJsKr7gyEk2QgBcDVuoh4EhVyHisoVNXuI20eRmE0DU7mrHur626NYGNXeGoaZv9BgztOZaTSMrWQj6yGEkBoOJOhagKHyX6Jviv2C248X89wmHH2dQK7bnbsKknGGtJDr7rZfGWukgae2jdhPdeddJf1FTFAHLrnioLwca/FX9c5tRd/pk4HfXcGJWrnzTmAdvzlONf14naTxW38gh+A6riC0Wrp3sW1AEQN3hrqnDEsDbio24NCwIMS9kThIqIF6dxdyVMDh1fnxspxHKFYl1evEdZ6gXRSvkWKW94rhX9EaLELq9rek84nf1j0ElKvPZfQ5ATCIPErSUN+O857lSqIS1VCxqJXKfS1O1UCV4yrH7ok6JiBadhC0dehQ0ClUAJFfmxgH7Xd1QmBEzDwRU+Kvn6uyy0ikWamErCOJcvZ756QJcSlFwUt664KnwkcWurx+u/60+Ys2AMcwrQ1+EsVp91ZlFfo9TCYi0p/fXiJeiqhLQpbTvVgSmDEFuPQarmaUO/8kNDrzQJlcJSTxuDhgMQ7CgoevwiN7SZzdb43fFU4Atf0fRSUQb4jdM5sw5WoqkqRDRDp8W4X3JpJrYOXJBsitwefraaLWVaPAIEkGWqYEgKxVgqJOZEs5cRrDL1Q/WYpQOWs/1TnT8ijtBtSEQU9H+uUpqUKIgMsAa/ladJiaQKn5i3NYfcVLKlrFiahAvNeK3ql9UZzfCxmhqw+L9K6hR0w4ZW4ORLVE6FmxYHSIMbdmBDGnihfAftOuJd6r1FYTSJ9rbSQ7kUpaIYjdnYmIsUE65JFYLxWXpKhn4qpy++6fEx2Om5aWiKSgqcCK21sKSKCbKU5gxhSYy26E+XCeJQHOzZho2HprZKnjAB1HeH3EgApL8YZS97kQvFxmqLyhreGqJBZ7jL6nTD7LoVmZQGO87Mo/awYjN+vCOnKLNCRVRFmpiEquStRLgQPBLaEYyyoUmJrbuRNt4YuaO/lZaxFYi3IB07VLL5pqKmm0iOkWgMU2YAHMAYSmjkmIcgAmAUItmiuVwM9fESwZw4uv7Z4hiNFWDAkEUVyCfu5kEJO5BEi0UxLXoh0rjnmSBQvBDOHHfj6OI+nQYS+UOPbn6ymmASt82ZoDLRWdFHHzdN4HR0qHQFmfW6N7cJCP0tBYzTRi9SngQAgu83UAKAhIpizRw6QKkU3gsuu4G1HuZyeh2VUEynUpA8RkMfOc0o2hu1C5s7C7SUTaqoB5Mb9zgRIbNszqgXgeKWaxmMP4/MVovDHf5kG4FmwxXBNP1TUmElBK4UtVTrZTDQwgqqrsgXOZJgPVo8gCIb5R4ofjxDHHboMg0cEr4uBxUZfl0ituh7h6QqmaICKpdAl2m+5CMCPkFmNbC0WW0NegLrSVSORZkZ0xofqesLkR2F5oJSG5TaIcLWwr8xjb2bT0oGRBHyUmgqjiHvxPq9rSEmlmIbotIY9t0KkhS7H5SDpIH5DovJKZmRp69B6pMfnMJJZfY7umz/JA+R51eJ46hOYfb+B6AbkLTobpKsGMUfkwiiPQwrsdyNr3Y6Sz1n4Vl6JBx4OK/SYZB+uuNOGiR54hWwyyoUF3Rk56iU8nQRV5T8GlxoPoC+FvIa72K67iMpbuyi2VBNBwccwiXKS3iciXTMyL1ELR+CYdjaI4n+c8bcQaTeX4T2lBHW4PS44nGBWkdJS9PRKLaOoH1J+oOYWb/V++q44G/iA6x8+yyBZyDK1fXEQ/37pMAM6Pk1abdd2ZfMNO9znXrrclsYPVXTrdhu+wThzVMpD3XZPQqOClmmPyniCyvd/5+dVYoUqu7Elwjfr9amm2mfIONa4POJOx9FpsMVK/JtL3p0+BueIWNA/5CjLybZInhaQGvj0LHX+chdKSGUH2QFbJ0NdoNDNGcq4vQMNn/Jv+rlHNAEofRoZRaovypzuCVNESw5BKp1GEVgz7WsJWFZxjaVURy9bI+GVREERMyUsyHOn3Yg6qMFQtNPlREnogBvF+RAYUQXhNOERsU18KHMNXTQ3GHk9lZV16eEVMojFjII1gVEsk7QiiSUX18IhweKoFlSIK4dFiSQ8NWQjxOFVL1yhgvdMhske40NTCIzLH4QdCIe9F5a39+8l8WLJXUWv+3rBHtiS42jcvijj3Ib+nRP4E6VuIQiRNKCIg0YV+rq71ErlOCUNnTIW5+lY0Dz4GuWaZ3alUcNXoiG0hxG5wa6S0z/EQRB4TIgT+h0qLcciBrZ/cbKLaVcDwI6+gfM0dyKhEnLJJxQxSGIO4egOCq0nFl4wTqWma28nWmWVMACRkzT+rlYNHkf8auVoWo7m+cinx61S2SA6cX5s2KTzpd9o81s9Le4LFkpiKWWqenvLKe2lTBkXslO8+ltwEce1riSALWybIlgC+ILxwbb0XDiFt1J4iGkI4XHadTbZxlYCZYFBQiubOaXUjkRjC34EQBILABT29NKHnK+ttQ8c9EdOERPY5ZB++6fn7ZzqJSX1upSrZAyUBiSRw3V1oHvQl5BoTVhLIgkuO18OE0wO6hC4GDYXrB9oflgCkEwPbCmwFokonMEaJXQuHgWIhYRWtyCp7FVfZbgHq+EtMO/T+abwTI8XSXUtbEiwm0fqglYgbovqU/tIiUjs/u+sMExH9lsMPj4ufKaJ87DL0hyX6kEgBqaTgwMXPU8/VLGHO8sqP71abuGOCdOF+tKqrjis+Im8RDOtwVvFQrtwhtJYMHNJHaoRHLsXZqWlIGwnAI3BCbEQNESkjcMdALDwH94QhBXJN0GTHU2kmxYnAcQOcxRASCIzbD0V8PDHQ6gd3SS4EInDTXWh+8AvIUWBTT9Hq5gVXvl32XWwfFtGodZOqGGWfY/OvMq4L0Mgy1OpN1KaUMH7t/ahffDkFYLlagQm8RfjdxqbiVV6NQwG+7C+xiiYqAr90SSFLmQuwdATA9YmOEDRVBSLumnJ8LTQE6SHAhhKH2qgHllBobqAAKyEKYWPEdRaeH6SN0Acvnq6WTkRkD+u0TDvYHry2yH/bCjFxleJI2HHTVVKHzC9llfyl8MyYqLVBNE04PBeUq53OHAA5Ek+diK4jPK1UFiOWPEMs4z7xyMKKAouIo8ZuQ8259NVWFdRjJtKAlhC8/aSNhENh1VTKfNoUNG+9F42DvohcdRxZH7VMczG8vhlLssdCBDQBINWLgxMzNOoN1PpLmHhhHqqf/4E1Miq7jLhPhca38BUtWUdSZMJE/ebpfRSJ1MGgBxZSTN4yAQglweJptI9iI4NbMOza96PEHqU66MSI2A2YIIQTdyLZIEJ65e5Sz6Wx4yJfuteAisRjl1u6yUn9QPVukCT0PJXrzc85HtKhc7zMMJv4cvtlo9WCJBEGytpyWtnxcKMcr3C0JBtGEwytfplkDxJ3lH+sl/o1YXYAGhHC6KnxXrmPvXU9wpBYpWpVD9xvH2WoDZiEpNaeEWo35m15carke/s9VhKYGIHpp7plrtKzU6ksz7PGTY7u83kMqvZpRqnDDTT6S9yEpX7wccDfXuZCJ5zyoWwUskWxmSqFO9k/2/OhPVMUfqkZZIxb7u9lKAEo4DftKJQmyKYNkWonXqsNiIpSeBFIjy/ERn3Ni+nBhZduZoCjRO+PbhVpIdwfpq/nrQw0ImnwenVn4TYH2sagyXNz6kqrOhDui6QeVb2wdYVOXPfpzlrclDGCSC2IqDmR9TbEHNsa9pZgVm6Hi/ZLrfurX2kGoRA7zpST3Y8BP5y7es/frlUVG2lo069bqzdz+bEpk2Hu/Cua+38euZFhYIjUgabtHELSADUaoXwCsRFwdXTX+psSDTnZsYHmlBKaCxeifsCXYR56FBh0or8ENjmiYeevICwyXqf7qiGxjRCgpGN/d0wgqfkZlm1ZcOnI2w5aNWK0+C5T5Eh+pHIqaVSaK0fcQ1dYdSm0yTg2ukPtlCJSGiW8D92P4UlOiJf3BEjW5krK+rnqdSdWXM7Nl3mIeKaIh0KAIHnEQB3Gj8kDd9JtqV3a+n39b9DUrJ9fc+EWzcNxYI/47aAkCihQ+5BKYGrwmJiKRyYQeT/faBmaoLlziQQRZV8JoohHumArie0j5Ca0VZtJEhhC856H0Hjfp5E9+QKyFUoAhQ7nKEjHnR17FNwmujPmCMHuAjCzBNz5MBp7/jfMXx9CNnmQ6zkEW4bsdtzuwy61TR/KNjKhtz+3HJSSNiO7qsjcVMN4GZYEs0gkCrnmNA7BFJeNDGztKEbL+1LsQYsOjstybIYSKX2TBHd/xPXcb43gseOgtViWWCK1xBSl70akw+2xPlKHaOqv4LJMRWG95naILIgZfOT2Xec3X1ITFP/aRZ3JvmipLN0br0okUlMAoqC3iqjigTlZj+fiiSXf2y3aPirebr9PemyVDi6eGj0//ZafvKzd7RyJ8F5TpX8cV6f36nVklDn55LNo7v4J5L54BHDgHsDMbjQpkI4SIDmd2OVsULtCl9iH51+D+ebFaF5wmd3zSf1ch8HGITiJzM+O3Q1RKVpbflTDpvOc+K2wsGcFMD9OTCzUJsQ5RPStZVIQREKBUyoeJmznoesCBKCJ9X/3jt8DjbzqgH22h3MTeKBN1+MO08/HhmVK5ddWYpq8r6UUeYyinyF9OCBMoOEihSTz8eK1u6I/XPssZRX3XEwIjUeksNc6yivMVUG+7E1KWwTBPe4qg2ii7FiipQiYJuwyk7gHXZzN6QhoPH5yZpow+DiJQCC0ZGLflFqAyR77o9NWGk1d3CxdiLU057SXDWjlbFnxEDSbyAb6uZBn48s/5FoL2Z7vAnbaArlVVuTajSwxUCHTVxZxA1hzw1+AG+4EXpsHTOqzrlYS+73EEVn3U7HO/VLl7SMATGp5MHzo80okWNnXaPk0EY7eW1YSgGn4klwRtZYZxpy1JdNM6bsRIGiuaqHMQpJfv15Voo8yXqiqqxE1DYCSrCNIFmlWoc7WSqWECKDbyGSeUyuUiQiPHcMuXbuGIpGhDdIogpu4OgP3aMNWY0tTm62I59di9ZY5a4+NcHN9Nrb3t/tTJLKEWy2RD8Xoy79dWHOAEX3Y8dqkEUcYJY7lCAQlIL+v16iJfebUNirbPrUE88rfYX54PnDGr9EY6rcdibnD84Rt/kp1E+ii2gqTqYmJg4e2rlD3B7VNylQdA53tpntM0Ix4OMm9o8IxdkOibfHrVdKAMA/v4VqWBUGon5ISPXwxQ7fwkOrYAsLJpXmKooJqQbJIXz8v2ii1A7pWmhCbFqU2xWFNGLR4L/pdkpvdMghvRpvy4FoM1bAryT2JBNVOj/aPMonYL8ZRR3wcUMcgFkszPldMXKGRC6qdJBUTRC2YxVtqVbLIa+K9PWLkUhPxdDOST6OyHF7W8DDcBulbJJ6EwEfESt7TGx+IgG0ITh2cpGaj2096TenEVKSVGnwQRx+r2G7KIkVQp9VJAzYbkT7nFuHOO6Cf6/Yz4vAWtqK0ek96FXZH9MEXkXWfRO7y9NKEnijzsqwKrPUSFn+1uKKSebQI3KICa4DRP8J94yKJspkcNSdx45pre8VR7AI6eEMIoHIR6j1rYbiK47UTqdM16G7L7QQGn4rrCJWMrYFByFebs/QVcgT5I/E+/tuOEbta9ZLsyzhJ6M0AKOWkLXOjDj/yLIV0Ee6nKpi3R7iVtZXMEkRO1+zvaRdiqAiC32O9Jik26uBMelfwFYqN2q+7TsR0UTEU+vFGBIeMrmKQEFYhyfYs5Gyl2Cpfjv6r/fJzFSLkCCl7DTRs893hS1oC0Nxf0wYO9aQSJcuMAOQjq4uUA9ME2uJkXMFFJc+2oVxiEXY0a6JixSmfAuuQrKvHunJUEcsg2iVSgTZXx/JS2Ci2EyjtM9JnE53KAW3rWLLGFODcDGUv0nkk3DTm4qnmrf7y4rVCkkjCTkuwx3ONaZn6TANO9Flb6hcHVilViAlcSGpUX1NioVuTEOpQrSD2IHjpRxHiyPDIYNTmXP3mJ1KEMlgHQSTkMECi/xj0bIqxNXaLaG/v44Qr7v1o4wPkHuqzGBSOsKGuHmUgSRRrREVWlTQkRVF4dY0EvpNzboG/Fpe6Znx0UXXYZd4YRBA7IHUQxLVoG4vJQbIOdZVC5xeiuFXMXn0Gujo6WfLiNlDNGm/2068s4m6uPnssQn6nVwUSm4jxEXS76Yh7RlQMB16ROiJjaSqst8FCvEfZ5ByC+06AVhuFBFN0rTxt6JFREvKZ5pRLkoknvvHcfdEMP98o2iaZsAIof6/60BHfGGmbDNBR+DcjSsaVdlkP9cljbg/8bQ7wPU4HVSXMLkhzpmVO4Uyic4vGU+NEcKBhR/1Nrj9+lrIV2KbmTBgaFCtAKcb5klOF6nYBfQNAgYqEOwR27eAi+HGzaSxeaAuWFDqdu7hqPy52Ieu1pd1Ct0pnn+CMJb2GBBil+1MELMxJm8uOAHAGiaPG3ocpoZaBLAT4USKLpujOQCEkMZfP0BgZxbt33TL74wUnuQ2wbICyrXq6uvC5487Caaf9D4qTB23DBvHpculXrRLo+Vp9tNUDGcv+eitbRfEEicLXFCyRiqLUEX0KUaRhsOxqTiY8nvO3I7uA21HPpaQnQiOo+9r37p+drs0BjkMSL+W0LFUhf7pgH9Og/uV4hDrqwyO2gy8hRb3ixijZsmTUBp1qaHHhAwFgeYR21QUWqUm5fZRW9QKX1eHdrYZQ9doTn9QYmUg6mVMjPb4GgsNVgifGceghe2L7rTfERKXKZ0Z1hifKVZxy+u8xf/5iGzjE81JVrhj07b2NsWHsu/d22GPnrbg1OakR9XoNXZ0duOnOOfjVpbcg1+FaoqvAIY5vUTausG2h+pOcS5AZ1YYtGwmA5WYnokWkJuZALQxXAaUK+LAHyC0VgFods1aciq6uEqrVGnEQupgOkHdl1VmDQG0MaPYGjtJS9FREy+Rg3YlK1RspfGndlkq0SiVfhxAenyOPQTBu+CKaEpyj7/UBOzKOto8o+T2ymrtBfPksd58YdT0TT1QKEXu9VpKGBQd4bI3H1yJ4Ki0EIi7AZmPgmjDlEXzhqP3xnp22RJ1KllNfA2q13czwxW/+HE899TfkO0u2L4EnAmrPPIcMex7AJ4bdQCAV8rew/YA44SOvr7aNseArHLL9m9+zv0jsb5bHsPpq03Huj49BqdSKLm/MH8a3v/UzFKZOtupAFK9g95H2IJdr4OtfPBzrrbe6/jrftf22m5rf/ukWVBoNfqbUZIzt0U6NUQQv2IrkjGzJeQsJ1L5kmREAqnPSgiUK0Vpk5LiNt+daNsbZR0G5yLpGzeorbFxxelLDZVKVK9Qm11E2BSRuO/zzQj1NJ94nu2z1aPtZ1E9QRGmWzjVG6ZcaUJ0Im97qw3VVsRL9JbGZaC7r7A9eJ+XHSCirDOtEbh0BGMt73tpv152OpxFb677KjaTVAK1za2bqxH4Gs0YN3V0FfObj72finV5XXHMznnrgYWSdU5SuqoKatBdM7ZEvRpp8yLsUcToVcNSif/kTj4dxexbyOfSj9WGGdG+CRTNexgfeuyMj//j4BArUgpwIQ7OJQqGAAz+wM350xq9RJfWAOFakm4fXuVInqrU6Gs0matUa8oU8jWFojInxCnLsjlTMQV+RFJRIO55+hQPPsnzTcNWTZUUAMiLv8lofTGJddoCmYS9w2RDv7Uss8C/WpSxa6eQ0d8KkZ3qE4zDMkHijvQ7BTR7q2GkRPhDRgGAilIdzUh1YVSSdVg/iYox+gzxyBaNWoNKBaDOVaVGbYh8q6Z0BkQOsin1DPz4QRSmQGVkyIunCr1YZ2WLksepZCMDxYzliTSK9b9aZLzJCkLZZq1EdHPKKNdFBdpwGEW03lqijOqxZ2wVE9dBz1sZklh4ERISQBDYgkYt+ixUc2v1zZbgESSJR1fhveSLqpFN6nxhTsbcTB+y9E99Drb8IcXN5Rl6WfDbcYA3suP0WuPrKu1AY6EeD9yhhDD5Ck9ILcmjmcqQWZFkuZ+jvHDE9hxM031Ac1IRELeV6DSTS2Qn0IfoXyzIOwFZiDAfkdlSy/4w+S2VV10RAx/YHz7SLC2dFOhCPoC7YQq7ezNxssOHJV7wRgKDqLFzb3uZv0LtUnIE/pUNj6y6VdHLuG94zu6Ee2UQtyTLkXcdaAmhu68RTMyye5VjEolr6yjXkuD8/JxHLZBy2eXBMOXFQ140uX0CeSkdzbX5rcPK2C6nyI1NNxP68W0jT1NnoxqXBCq48GL3fqHvVQ9QKq4akcGEJIH3POluavJcszeRo7+zeN4lQ04+cqttvNo5ztqy1lNPfxUIBBX5tuH1Bk2r2k/ed8vK5VJl9rlXJbFINXTmystP91hLsztbORdRPehbdR/vBe8YGR/t8OTeSMK3AZNekVQ0dk2FUdGOUfUciO0mhoyN4x/abYuON1uSUX4Y9XrrtKyA9Fw7bfzdcffkdMIb2yOnsXu1wsMDn4biAVw+CYcMaTaWxjBabpZOUYw5ii9MnqE0cgkDLNheAoENTzCBiBpoTgmJiK2wiEulhxEhma2L7MBWFOxn3xibLKXkG6lXUFy5yG8xVHOyAhSLyvT1oNOqoL1rAnxf7e1B0YludXIz5AnL9fXzQ3A9OrBmZ5LZbYG3UaqgvWmQn2FVCF1WTNU2UJypoVCn4wwAdncj1dPN3bYwUGQNrqA8vckS76DC/yrnhuYEhXl1j4TwONsn19aGzs4RyZQL1cQo4B9DXzyJlndcqBUKlBHbIicgXcqhXa6iPLAbqVaCjmHVSm7B8B8rlimmOj1HTPKC3D4VCEXUKVpFcCu2WcxtNyGTQQGN4vo12K5ZQ6O5gwlSpZKhPTACNCtDdhUJHp7Pw01lUPAJYoKZ5Wso6sng+6qMvoE7IS5hIwTXFPBq0PzWiNgTQhNg2Ai/rH6CK29ZKTn0UO7sAUgvJ+0x6N3+lwOuol8tojo9aICmVUOwq8lyrlQbq5SpQqwDFTh6TEFaIi0hgtoyZxztoCdUSBEusmXA0ajhkv92YsNUaZJ+Ka2OSNEDXbrtsiVVmz8CLf3sduc6OxI7qiBkTzxgntbLjk9mkThmHMjtbVRr8pogAg5s0VHWMNkjJy4wANKKSd4G9q4kleohnPomVMnBeFb7o7TMkCcSlkq3+W+PWVP09nTj08EPQ010id6EhblIqFXHfQ0/j2qtuR7G7hH0P3hP/tce7sNaaK6Ozo4QFC0fw2BPP4fJr7sLl19+FZq2OXFe3awrJfdSorgtyhTwai4fRM9iN3ffbCbvuuCXWXWc1TB7o4XuHR8bxyqvz8OBjT+OaG/6Me/86h4RC5Pt60KxNoJQ1sP8he2KF6ZPQdJ6KQt7giadexB8uv42Xs8fu22C/vXfB+uuugd6eTgwPT2DOk8/hymvvxOXX/xnlxaPID/SiySYXlmecZGVFQjr++vzF6Js6gD332Dnb7h0bY+01V8X0aYO82fMWjOKRR5/G1dfdjhtuvx/lRaPITxpCk8pkO4nF1+NnpMqjQQSjWcMWm62NPXbZGpttsh5mrTANpVIeixaP4+ln/4Zbb78XV1x3O15/eT5yg4PIyLrfrEV2y9CMG3jve7bD5MF+dHQPsFv30qvuwEvPPIedd9oCm2y0LipV6/4qdZTw8msL8Nsr70Z9wVxs885NcfAH98Laa66CRq2Bzx5/Gh577BmUertRHVsMVJtYafZK2GHrHbD129fD7NVXxtQpAygWidBX8Mqrr2POE8/i5jsfwU13PcbEtdDXg0ZdcWVqra7VG7q8UdrapGiHiNBMX3UFvHe3bekOEtgjg2guyxkqnVar1dHf34t93rs9fvCdnyPX1WXVAObmUsTTSkxBLRTNJ1V94vgKX5cwlea118KPqPDln7iW0g2YI7u8e5YXoFREoL7XTlA+sxZ4R7XcIn0oqLiBPCC5rqt60CadWNM0hhdi+513xE9+cEzL9B585Ek8++QzOP/sk7DtO96mP+JBtt5yA3zksPfhltv+aj766W/gmWdfQb5/0Bka6TANGvPnYe/99sDJX/0Y1l5z5cj+qq+999oOXzvmQ7j51r/ilB9fhJtv+ytQL2ODTWfjwrO/Js/031u0eAT33rMPTv76f+MQyjRLrrdvtjYOO3A3PPLos/jSST/F1VfcjvzggFMxrEjOUn2tgmalgo8esQ+O+cwB2ezVV2h7VO/can186iP/hTmPP48Tv/cL/OY3NyDf18uW6LDnVh1pzF+AjTffECd/7aN49w6bexFXX1tvsR4O/eB7stdem4fTzvmd+e4Pz2NYzvq7fYCxNcpZglqv13HAfnvwj1zFQobvfudx/OyMr2HlWdNbnvGXjfbFoUcdiROO/Uj0/uYbr4nHHngc1flvYPU1V8axn/8IPrDX9pg02JdCnL0fa2d77fFOHPt54J77nsCXTvypueWGu5AfmqpsqCJsauOnqulAEEtSwugY9jp4d0ybNimrVKps/IsuBvDQL+fQD+6Jn5zzW9SIeLOUEVLAneLnPRiWa9so12gZSURnlFvhGGzo06ART8dIcCHYZV0PgCqfCeVSVEwenxqUPI/XhhWtY1k9yxM20Y08YZHFBIAlPSiXLzJiVCoVVMpVTEyUmQJ3d3bij5f8kJG/Wqvxe2Sgod/VahXVSg2VcgXbv2uz7PrLz8Kaa66KxvgE9zshHbexeDE+8fH9cOmF32Tkp+/R+NVKNSOLbRivBgIGunbZaUvc9KdT8ckP7wVMTKC7u4eAPyNXJs+vUkWdv1fHn37zY0Z+mkOtWmcxn96387NjkjHpqt98Dx86bE805s1Fjm0eNZZOUC2jWK/ggrO/hnN+fDQI+WmsKj2j7uZWr/N8y+UKzTlbb93VcMnPT8RXj/0QGovmMZGzZ9FgO0lj3ms4YL8dccc1P8FuO2/Be0zzsOut8W8an/aO5jhjxhSc/LVPZJdd9H0M9hTRqFYiCYCRxr7OaM9JZRofL7P9o0GqEInjjQYT3YnxCVJX+O/X5y3ECV8+jJGf/qb36btE/+r0jFoZHz78A7j3ll/iI4e9l5Cf943Wy3Os1viM6Jk8b7fnW2y+Dq7/ww9wyMG7sepFMSe2f3pigzLkvnTqiFNJSW3IdRRw0H672iU56YFtIoU8q2qCPYSMtN8bbTgb2227CczYGGku3tPFYcfO+6WtnRHqmyb5vVUlILlDfkLtAC14B+Os+iypm7DsCoKILhpkek+lwqSIQknPNZ3IomYtxTM0MVAOz0gsigiNNcDlcrmMforFPEqu6urs1VfC+uutwQDAbhppCMLGMjJGZSgU8gxcq64yEz8/63iUUGPdtjGyAG/f+m348fe/mDVq9YyAnjgh2Q9oLqT/Wc5obQc0Fr03QXYFAJ0UGNasMA4UsgwF+m6R9FLS34Epkydhk43X4bmRyB10cUvH+VmFAiMfGZfOO+Mr2GmXLdBYON8WpGnWYUYX4NzTv0Kc2AM8ASFH4qngKyoYwu+X8kwUaC0nHfdRfO7ow9BYPMLEjsasL1yMvffdDb/++TezrhI9266FRGm2cjtjGq8/b/eOkGJivIw9dtsOZ55+AiaG5yVNVMjgRgoG2y0yNniSnpzLoU4diE3dGlh5f0ookFhAafQDvdmhB+6Z0XzpIpXO7jEwOrwIa220Fs4783gMDfajPF5homo9oz63whBToN3MF4vsNcplOd5Pet55Z34VW227CRojI8x1pbgM2y+MuKKDS5qebUbHscmm62HrLTbI6Ex4La748rx5C/HYY09bfz2pFiFMOfvwQXuy3YARmQyU0oTUwaKikd5WFq4Y8aUtueBM7LJNmam2YfjPs2VJABJZXYdi6gmpCVsqoIbQbijlurTRVn5r0vspKCg8KGySb7dJhjhjfask59HBEBIUOZaAZFVR/zKUOovEIbJtt34bdtl5YzQWzwPKC/GpD+/FxKTBAJCzVmhD9rAiIURGwEJjdnSUUMwXmft2dXXg3vsewRmnn0OqelavjvkqAZz9SauifpX1etZoNFiHJAQkJCOEYpIiUYJkLyiQwco+/yc/PBZ9kycxTJJd4qMfOwCHHrQnyhPljN1GrjttntZJSEtGz1zedHZ2GBq7SYUpeNgc6o0Gvnvikdjs7WuhvngBGmMjWH32CtnPTj+esYyeSRZ702xmpmkyInKWABb5hy3e9GaTCQQfwA033G69F0odtC5Ce2xEAEm/7+6h6hnUTbzEnFbidHzDYKtLs1RA79GzaL50boTUFBj30rOP4aFHnrLRurRHZEQF7WPR8HNKJXR0dvBcKZ7eIo7hNZG0Rfvzja98xLXoo8hEkjwdETCOS6u4FJ5YrYyD99uV10GEiWHMEahrrr0RJ3/7VN4H8gzQxXsBYPddt8XKa66I+vi4rSQWW7w9LNvMPs0TbUNR0ah90lKk0wfk95iQStaOBdsPli4OYCn7AhTKwaekyy/5qalXFjhjk4G62buD9cLiRbZkrnn/s7eOuJfEarKsQO4lAoqOkqXQc57CwuExAgxX19FFovE52y/utfsOQH0UXYN92HzT9fiBxKEZuZiA5XDWTy/AllvtiW13PhT7H3Q0zv3ZJXj9jQXoIqs7gOO+/kNUJygAJGdIXJeMMkvU6G27V80Gc+aMLPIP3P8I7v/rQ6jWG1mxVMyaZOOg8pJcxRbMtdddezV8cL/3oLFwAabPmo4Tjv04P8+6uewekKSzYMFifOW47+MdO+xn3rHjfjj+q6di0aJhQ8BPe0rAT4BLROGYTx+YoV7NzPhI9uXPHYRJg73s8cjnydROXukcE6hCIZ+9+tobePDBOXj5b39nRHYSlSkUC/jkUSfh52efj56+PiK4gXA7ICYCBaPKCwAANt1JREFU9uqrc/HonKfwyKNP4vEnnsXjjz+ODBXn3gtSofdOsEvNMBKXikXeXxqns6OAicVzccSnTmS1gKQDQvaJSgWXX34Djj/hVPOJT38NRx51PC75zeUsevvxs8wUqEcXgHdssQHWmL0Sq33c/tvBUyblyTwnMqhXKhicMYQPvG8HO0fuHhTg8fIrrjdXXnUtFtM+F4seikklIWPg3nvsAIyNWIITcUkfxchRrgIjdvfEFW2/oyXqIDHHKn+ICGxjSDTUp4QA8i0bAX9rfzWbi1l/8vm24Yr8ylK8wCF5yPwOCJ4tQWDwK4zD0V0EqD0ofRDC1YmEMVfI5/CNb52Gs8+5GIvGy5jWX8Qpp3wV+++3J1NxEUnlueuuO5uTMAYGhjB1ylDYX+/TBm686U7c85dbDdCDPyOP3/zPr3HiSivia1/7AsrVJq6/7s8oDAyhvngY+RyJruzl8xMkICaRldSJG266E8d8+Qd49MmnYRplrL/WbHzvO8djl1224flbM7MDDgD7ve9dOOfHP8Qeu+6JFVaYxqK/ECciVMPDY9jj/R8zf77tDqDQy661u2/7C2667W5ce8X56OnpZLGa5kDj7/6ed2LFNVZHrVLDvu9/N3NY90Dvsy5Xq+aYY0/BxZfegMXjDfR15LDf+3fCD777FUbKjx15Is498xLkB2ZwXIDjaN4S3KQ4DeTxjZNPw0/Pvhhdg0OoV0dRHS+js7uD7DORd1jOk2wixMmfevpFXPLbK/Hyq3NRKnbgulvuR65nJv56x5/x+S9/D2f8+Hj85tKr8a1v/QgPk1RgOq1LsTGGM08/D/d/5Qv4zrc+zy47p4YYsgd0d3dinVWn4elHngC6Onw6sHHAJ7UoWV0ZG8POu++MlWZNR43UiALlNFhp8NXXXsctdz2GxQsmcO11t2G//fZEo+akBsdYDvng7uaMn/4KjbpNZrOIH9eSFMIncE6wSz/sYo6k5yQ5yjHfQCCUEcZJFS5ytAzUXOWSN3cL/CMvgP1yfeJ1a8lBMaqx1FbLkJBO28LaCwDJNELyixTHlKVIPrTbHP6X3Fg+/NaX02S9t9rkw/n5L36Hrx1/EtA9A1mhhOdfeA1HfOLL2HKLjbHqqrOsOqC0mIH+PhS6ujBeabLhKL0IQf7nV2fgziM/jHvufRAPPfYsHnzwYTz+8EP4+Ec/CeRWRIErwJZ5XraQS1OFHmYszhLXfOjhJ7DX3keA3Nc5yh7LCnjowTnmvz5weHbXHZfhbRuty4YrAloRJzd+2zroG5qCd22zadgncgM2miDOc8EvL8Wfb7sGvUMz0SCLb0ZBO0XcfdttuPKqG3HA/u9FrVqx+nujgd6+Xmyw9nR0FrswONiPWrVqg4bcWknc/czRJ+OM004H+ldDVujE/LE6zjrjQjzzwlxMmdyHiy78A4qTZ/C47KJsMfzaza3UGsbUgArxIXQi66Cy20IZ3Tl7J49h5L/gl5eaT//3CRgddoU46by6u4FCB3L9Rfz0/Etwyx33YM79jwJYCKCb/f3clIuy8aod+PHpF2Sf+/QhmDFzmjtvspFYxJs+dYD1c1sxRwxu7vI99zhUDx866H30rm3nSWfbrCOfL+GKK2/C6y+9miHXbS763dVMAKyob6UEUkM3fds62GabTXHLjfehQC5dmoeoF/55bepTeg+ZMqr7Jj9vos5rf5WxOTQwjWHguREsxbV0BACLXoWZOR8GM616Id19dCim0IRg7wyRaHQplcRRzODOUA8Mxh03AbGCWout/XqoeCPPvfCiPyGXn4p8RyfqjRpKk6ZhbOEi3H7HPZYAEDekwA2ntnZ0FNHVUcDw3Dcw5/FnzaxZxNWaGemkdNG/pXwOO2y3Ff/Qg0aHx/DUMy/g6uvvMOdf+Ec8O2cOioOTuENMs17jDEZxcvrSTPk8zjznIowvrpjS5KlsDCRVoTR5BsbnvYzTz/wVzv3pt6z4XMjZDjUAJk+ehPXXWxtrzV7NzofDRXN8YATc791zR7xz27uskZIi6Hzl2iaGhgZNvVYnY2kEPGutOg19/f3uDKjVdYaaQ/6nnnrenHvB75AfWI076VC8P0f2TV0J19/wZw46yk+ajAYFH7lOt3JkPiLbibgcVcgSkA3lbVApbpdPpoGADIv07CefftF8/MjjUak0URoa8gmIJFHwSVPr7YUL8dTTz2G3978Xm2+6JlZeYRqmTJlsDbSuAAt5Onp6KcZDcvk5wpQnReqDgLQEABneBglGo4y9May/4ZrYcbvNGXE50Edx2t9e+ifkc1Vk3f245da78dJLr2GllWaIYdJGfOaKOPTAvXDLdfc4e0RwMbbtfGe3zUYCilSivOuSBxFvXmIX8NXy3HMaDYpkK2PZBQItHkaz8RqazZlO2fOJnFHabSQYOP9/pONLzLU2CWpPgQ6OsL85zNXdG0cP2HuIYZJRbu7rC9Bk/wttJIltFLxosHDRsN9lW7k48zo0UXU0R/GTMy/Eu3fZhkVa5tqOC3P/tzKnfzIykSi56abr88+nP/FBHHvc93HW2ZcywoirVDi1+GtJ733g4SdNVuxmiUDmzWHJxUE8/NhTTJwIQcQQRABFksPKs4ZYV6eLrOtcnNURqFVXmQXQzxKueq1uSJpoGkPGPX5v6tQpmDJ5ipyOtaq4vb//oSdRHTFZYVLB8DTJMkF172sG+d5e3jUWayW+PxxfUAQdNyCjIUcL+tiDzLUTC+yKS5y74//9H69FZaSK4tBkziuwZ+9CvknKGy/jE0cdgS99/sNYlQp1/oOL1RuXPxCpl8JwfE4AHG6SWpVDc2IMB31gJzbwEqGm/aPo0mKphAceeBTXX30TafvAiMkWjYzggl/+Bsd/5b/5nnyejLuMTtn737uDOWH2ikwgKMXXhg7rqEm/DfYXR1dG7n6bhqy9ZKmULPa2KCU4IxmA9v1VO/oJOeDE5luUAOwgpll7OTPNTYLTPugdcTso9dUg6yj3RaT0t34lESwt7Ap3C5Z/edfbTvwEbJy4Pde690/LmHIXJ3TQvd0duOKyy3HcV9fGt75xNJd9Yx+9M1iJNZ0lA0LOOsXeN7OBgX5z5uknoVxr4vxzzkEhv7Y/QdbDONLQeiLK46Ou0GPBuUmdiyiXN7V6IyMCQAYrO+cQyknhwiqJ2jE0mx5CLksCvNACKxBTEUklAUX2q6tU4uhIv8vyXWpnPTyaZdz7wEf1eIAkQuDt/Q5hQsCWI67M9dUxRpRa18ezsiABHmUI0PXCCy9YSztxQF87hQLEG6RP4OyzT8IRh+3F99L+c2Q0Ead600pNZHx1zTiIcEbg5rtxSBRO+DNTAigZRHunDmGfvXdpYUREUPr7+/DLX53JcEPITmrBjGlTrP3JrZ1gjYyBg4P92Qfet6M59ftnI9c51Bqd52mPKLLqo9S4FxELMbKnFaH8Z8Zwcl2TwlQB3CLVTd6KBOAGqdeeceK/mkpKylKdJBFdvErjIv4kdVeSXuJKUPYtBTl+o2K7ovfXh2IhUl5JQ6N9lqPETmKyIbKFvsk4+dtn4pFHH8MXjv4Y3rHVpiyaykXJLKLP21gEshvUWF8/8fgj8etfXYhqtRzFnkueANknps+YjMcefEr7v6yXoD6OyUODKHWW2FdOkge1mxL+unDhKAfi8B/NJhkIWBJock15g1JnZxvrSusldoVXXn0VHRRn77ikxDfQNWP6EEyOpB1txZXPKYGJuFzeamLUWltcVx70wkxoj8hcxJIRESmfd2F/uYLdfuJ1Triy5baEitDc6vNewaeP/hgjP4UPc0yHi1YkD8eSgt2CCqA1UG2Ii91wOQoGGx7FLvvsjDXXWJnPWs7PEvMm1lhjFf5Jr3qNDNCSBhzg9oP77Zb95AyKmiQiLZJ9Wx3AlgHkaan9bNH8VXMRj1fq6O0hZKjXyJV0L5byWgoC4HoDNGv3s2ch30k748R77YJQM/OqSQhOUNPXi4/G4EW7gCPNXeyHqcfSvk8AQQanUDVVj+uDjv1k5RBspt+EnWuuBORquPyPl+Dyy2/Blttsg3dtsxHevtkGWGvNVbDKSitgcNKA3QYO8HD+61odK600E5tssjEmJkb9tGTu7GMHsMeuO+OGK29GRz5DnXXnPCN7pbE42+3d29hxOZDHfpdESTJMPvXca5i/YLFaLVCt11if/dl5F+GsM8+1RkBDRraMcxI43qDYg5HhhahVRtE3OMWUOmzY7n133oHjvvoZtzl8jPZ7zSa23HwjTJm1kpm3YAKlbkrpJbUkzymxpkou1U5UJuoo9HSiWR7nCMUgiSn/tpOu8pTkk8sj5zIdvfQgpxcJjeJaDhkFxJG7hwZw5EcP4L85/oEkhEYT+WIBb7yxAOec+0v89aEnsWDhGCrlUXQWYX5x3o+zVdjoK5mLbo42ECBmVnLx6zoO3f899m9Xl8Lbu5zkYVU1l6Dl1CAysmq4Y69LvYFNN1mPDdB33fEA8n0DyLkMQ7psQpKfi/BBf6WRfD5exOpXQepuuStXQJ3DOR+2793aXAYE4Ld2q2qVv/Lgxc5iUgpYyXrialPIq4wT+vJrVjEFEj6sRQdPV9nWkOwSi4LO8KGMNUxq/K4G4yQBJBm3ZLuskY7qvY3gyKMOw2abrItjvvIj/OW2W7O/3HY3GQpMdx8wc/oM7Lfv+/DVYz/FyEeHz357x41WW3VlPPr40/GBsX3CBtp89MP74PKrb8RNV98IFLpZH27UKnjXLrvjo4fvz8hGAM5qAa0nn8dLL72KZ558BnMefwY77vAOG5HmtD9CqG22eTs+e/TxmBidAxRJr6fNqAJE1FDB+/c7ECvOHMLpZ10EVBcDpUmcIfrQo8/zHK2BzhJlev70GVPw3W8ejQ8f/mVUJ0aIxVqzRgb84PtfxK47vRP/9cHP45k5TyDfS356uz69Zjn3FcgKX5+HRrloa8mIoS40x4u65NijC3UaODx7vI61Nl8fq682yxnZrPTGsXX1Bg45/Au49so/AflJtixZdRFnBpKUJmMGw5iI9EoqdOCbI4QdH8PsdVfBzjtuZa32zn7AbcRchKVVGVQ0rFstG/5AFdBIpbTfIwmCgp8OPXgf3HXrAwR3thFQpLrqwjAaL1ygsuPwHLjk9sYn2TtmK/YC901ODzLN6vPAU8/JsSyLQCA3yIvPol551pFrCcSLb/Khvy7dNtTUiHUDh6zarCGLsF8OJTJtWKro1qIqqBJjLUYeJ7q6NF+x6kshS61c5AodVNgtW2+DNbPvn/IFHH7o3rj7tl9lhx2+Hwan9QCmnI0vLGfPPvEMfnDaL7Fg4TCJwa5qYQjQKFN+QHKYOpqrp6cru+w3Z+Lk7xyPHXfcGttt93Z885RjcMXvz0Jff6/LTHQEzY1zw413ZI3qPPzpqpvtPnABTguwJHauv95auOXmP2GrbbZDsYvM5BPktsCqa66F7/3oO/j9JT/BT049EQ//9QrsuMuOQHWE02xvv/tBzJ+/0IrzUqSCjV0NHH7IXtnll5+V7bXXDthyqw2wz7674Oorf4qjjzoM66+3Om64/ExsuOFsNBaNo1noo8QX3nCH2ByURXM94kP74t177IbpQyXMXmMGq1MkC2l7DCVyi5ZribgqY8ZqXIbBSUMciCSGYMd9ObrymWcJxvNAV7fn4F//2jHZiivO4LwA37fCjWprJgh8xOI/ymXst/d70Nvbzck8kjZBP3TexUKBog5NqaNkOjpKplQq0WsbGVosuvUJeNtgKCLSe++1M2ausRrqZY5LCLgSwaxfV4RPsVCgqzErNIp0YlAhB2LU99iSwPuGqLG36AUwdrDfVk2zeneu2VjHcOdEV6pFAhYS3T0ZoEW/sUQidlMp20YQJ51o6GqduQFDXblcvmCorqCIkKJhBkLABMDR0yCJcBGLPCH5ME755ufRSRFmE+Vs9uqr4Bc//x6eefpF3PfAI3jpldfQUezAjjtshRVXsP5ltti7iEUKY50z5yl0UlNJEe80nSFPQq2B7q4OHHvMx/nHX00q3NHguQiAigh53i//aNCxAm699QH8+S8PY6stN2J7gHBe4opbbL4hbr/1Yjz04BN4+e9zMTTYhw03XNv5+WsMhBtusBauu/qi7LQzzjfHnXg6XnvuGfzyf/6Izx51OOqmhszVLuCklloNe+62Pf34erR0UUARjbXKyjNx7VXnY99Dvoy7bv9r5tUT5pa2IAcZSGetON1QMNLLL8/F1KmT8MnPfhPn/+L3KBCndheRDiHhlrCLDccpi/kMCxbMYw8PW9ctV+akGYoG/OkZ38i+8Y0f4Y1FDbPijMnZwQfticMO3pv3jp7j080juhw8ViIh1Gs1dA324sB9rPjvKvUwIabPn37mBfzlLw+io6uL94Hv4bp99oipNsOsWdOx3bu24LFYBXPG36lTh7Dnrlvj3DN/jVxnv0v4EXVYqj556I8U5fBe8LSl3R2Crm1XQ9W1Ua/dYD97vR0qvsWy4I3q1WhUPowCsxw1yVinUwqB+x3SL1utVi5iyNY7CU1OPX8goCjY3u5iDnCIxuKyL0qhEd5VUVFEw5kbg6syyzAyvBjv/+BeeN/uOzLXIEpPmV0k4s9ecxX+0ZczCrFbrdFsEjfA/Q/OwdNPPIa3b7U5552T8GIJhAVuQgiOeUcuq9XrjFRsxSaRnnz7FInLYcqGENB0dnXigl/9Cffc/QSKQzNQHR7Bl752Gm684mwvgtrqPZmNcMyQbbbZ+qAf2VpCZJt4k8fY2ETW09OF3XfbMfvq1083ua4OfPs7Z5p9994zW3FFikuos8gtXgv624Yx57iGHXPrPNU2KHJa7MwZU3DhuV/HGmtsjXvvexDv2HoTbxPhiwLa6nXm6tOnTeYoyGlTBw0FpgknJyIWQl9pfHu+pKJJAlnWWcLTTz6FF158BWuvvbqLu3eelXoDO+/4Tv6ZGB/PuihgiBSfcoXXzCqEAwsvNQqcKhjKcbnvxXjnrltzMhnlbLCRkY6DAq4KRXPiyWfj1+dfCBSHnEFTpEs3UKOKwZlDePz+KzGNug0TjND6nAx7wD674txzfsF5Jt5wlibIefgVA0D8I7ZY30NSpTK7mBOiDYWsNjFiauWbl1b/d8e1NNdv7WDV124w1bG/A2R1igvpi8wUUzclcvkbRZcPIp/To6jqgA2EScUlRxYEgKzA44DBUUFrQVYtoUVqUBtlFyyAQB6TCbPyitMJO/m4GlTY0YnbkgYsKbvEfcXy3ag3jMQKnPDNs2CaJeTynWnqhrXcu+QYtiep07Ux+I4oGVKVGyDkf+ChJ3D0F77NFYfYiNjXg9uuv9V89vPfZGSi4BTyNFiko0QoY9N2KR2WDFVcNs0GPBGB6OnpMi///XXzgQO+ZMbKOeR7pmLu3DKOOPKbhhCcrOnVqkV02hvOpmPdlyqW5bMCxdM2TVZvVFn8pWX95MyLSBE3F196rQkuVVdL39W9kCQtBhvKNmyWaa5OGPRhq1YZFP2fzeGcvIF8sYSJBWX84uJr+BZK4vI5BEz86GxqbJyk15QaTpJBoViQuD4tR6uSyhr6c6xGHH7I+3lYTkoidZ2iEwtFlmCuvuk+FIbW4ICv/ORpyE+egvzQEApDk7nYSmn6TCx6bQGuvPoWG0vgjHTcS6DRxLbveDs222JzVIfn+2hU636lNuQaL9oZ9sL7LW0j3AvHzsh4lJl6+S7g2Zes6/7N3X9+C5bmpqAGLF5kahOXu3bJKrZTZC0XpOtkee2vtAgfdHSb/OLEeEelCUCaNcofpxp0dv5FjsaicEoCbFF2mAMTxzXCSQtF8m9bvT90aXVWWvp2nTLN6nxAknU3MLQiTjvtfBx73A9Rrxt0dHXy81gPjzbcVjJmAMll6Oiisk9NfOLIE3DFZbcApRkwVA+fYzFtYAd5o8l+Qc+6976HeR75guXatDbiCFzL0M2/o4OyCx/D+/Y+CguGK1Quh0U/qm2YH5qCM864yHzqv09hZOV0Y8etpdqNS+HN6G/LvQro5IzFR7HzrkdwZR0qRVYzeRSmTMe119yF9+/7OUOFPug+Mp413XyIwxIxtEmAltBSuC4RjM984fs49QcXojC0Gv589xx8+/sXGs5wzBc4f5/z8V1WnrNn2GSsPJUZI90Ypl5rmEajSR1nHYy7fHwl+1GeESHcj8+4FDfefC/PkWBG5khSGs2PMv7oG11dnXj6mRdx330PM55TIBQ9o0mZWC7wS3NcOsfGRBlrbrQmdt15K543n0eNin1Sghdw2RU3YcFLf+f2YCS7UaQ3SRb2d4YGnw+pKx34zWU3uoGthEJMghgHhW1/7EP7ImtWmagKbDeavL/sTLBxBBZeQzHYtHpjWgFIveQwxApR2kuU634Z1wOQxKDa6CVcp59ImLL8RvXmxBOg0wIlOEWJfjbMM4/h0TH+m0RqouAEUF1dnXzXGwsppDnPjRlHxjhv3XSUSqZYIqOMNcDQwY1N1FxEniv77QjP/PkUFQl093SR8cYiTy5nyHBHpgPTOYBvn3wG3r7tfvjpub/Fq3Pns5HHppvadFtr8OlgDwCJwVdffze2e/eHcfaZ/4PiAEXqERIXfbgrAz8v2ca5f+6Lp5ijPnMcFsxfzFyKxqOxrAGpgLlz5+Pk751ndtj9k3j59VHOF7A9UCw3bGYl5KeuiLPO+YPZZuePmUv/cKMZGZ1ARyelwpJRqsgEhuoK8OtiAY8/8Ry+eNxpeNfun8KTz76KfH+fzcYjolSvozDYh6uuvBlbbncwzr/wCk4uou/xmikrr4P2wBq/6Pr9ZTdim+0OxGk/ugD5yZPRyIrI9U/GsSechS986VQzd+4CU+rsMDQf+qHz6aJsTAAjI+R56XXZgmSL7ODnSFDSopGKTeoRiM4oHiIHUyhiol7Hf+3/OfOTsy4xxPEpW9DO0RrhaA9o7ddcf6fZaZcDzd1/eZgJDUlTnD3YYYMFFo9RURKydwSDIsbGsOVGa3CRERrTGvUKbA+i61e/vQ5ZR5ePTbGB/yJdEjwTs8iQ9U3CbXfcj8fmPGfnx/tX5IhCesz2222Frv7JPvaCJClSN+le+pwKp5AUR3Ee1gMgCmuQoEWYCXYs72lj8R/VsbmoLbrMvnlrnHjxJtdSGQqSewtZ94b3oX/FDW0ZVA6980aKNx/eTdx3CLKVVTtKebzz7euiq7NkuX+jbuvO5TLcfu9TIGAnlxsV23jnZuuip6dE6bR8AJ2UpfXGIvzl/idt5pbnIgamVsVAbxHbbrUJSwjkG27WqxkVy3z2b6+bx554gcflLL6xEXKyY+oqK2HrzTfApm9bAyutMBVd5BNvIps/f5F5+LGncfc9j+DxJ15iASvfTaHEdTRGx/CObTfKbr/+Au9eFIpOVuTtdzkIt95wWbbSWltg9/fsjE02Xs8Qx5o3f0H2wENP4rqb/mxef2EuMDCAHHFiV+AibJ99wca/kRGgQm6r1bHN1pvibRuuhRnTp/JnIyMjePbFV3HvPY/gznsewsTCCWBgkIt6kLHRutLEfWuBsVGuZpS1s9raK+GdW6xnNn3bWlhh1gosQSyYvwCPPf48brnjfjxw3yPML/L9A67OglXVWA1ZMB/TVpqCnbbfipOYpk6ZxAaxF19+FQ8++BRuvvNRTEw0sPEma2Lt1WZYY5qrvVeuNXDdHY+yJ4VtJ85OI4SUm5FSePBYGettvA7es+PmWH+dVTFABULKFTzz7IvZDbc8YO64+0EuCLrK6rOw2cbrIssV+Lwpbq3WyHDDXY9jolx23YFdPch6DX09eeywzebo7OhiuKvXKygWcnh94QRu/fMcV/VHjMqBybF0692D5DCqZLNnzzKbbLA6EwUbqVjjsZ5/6Q3c/9Cz2GijdbP1116RVCICfBsRWSji0Sdfwpwn/oZcR6cqhyfEphWP7L54xkqiRMmM/P1MjD98pDPY/0sIAIDtCsCtdRRXOyLrW/lnKHZTWR2y3rTieYv+7/R3XyLc5WFz+H4TGB31pb9sJQyK7mlydduMe6+Roaru7nOVdyVrjAw3Pb1tFkfAUwMmytaQyCdHKWoNkuMB4lAuS4v0WC6gUa7Y+/n5okdSrWr2+oK+l+umisA0DAEY0BgexlbbbJDdecOF9nC8AZqDesz2ux6GO265F42ObqBMHgufmGIpZ1enKXQSobGdd4PhSoxC7m7S752nlMRXUBEUor9cZdeWEOOLet13dXLAjCVIqvmp2xlvgcmRMmHIF24wYSUxFHvsmLWyrWbb0YF8T7cLwZW07DAmG93K4wBVSZK10Vxor0kq6+1HRpLaBFXtdU0rJbmL9rZvgLmf9hGJ68um7jriNzYGm1KZ4yxBLk5IVYBLnVzxmR87MQ6QW5bmQXtF86dn0By4gpLz8Yl7uF4jFuz2kN+whuVCEVlvf4geVMVP/NHoSj1MBCbIJ2zjEmQsMhyWOnksXn+1mpH9xBqFXIZrZxeybtrfJFhOEQCx/1mjv8plyHIGYwubZvHf3w688JAFCN/4cFk3B2XRIiM1wJQXHJ8rda9iDOd5hkoYHrxCHIAsSeiCzyZwkQwcyjtIWWoqP9B1Z7W2AIkhpRLbg3bjvLvP7qWN/rTflkALdp+Ucshz+GvUiZazrzgJw6kNrK+SPk4lqagUTbAeuTHd36x70g+CRZh0OFmnxNBzKfgG7zCl1jbQhUJXH7Ju4cJMWNiDTtFtXF3KClMqvU55g10cScPZdnI9fcj1uexG11NA5skWCykELBF2cRCLv0gPZSTupgaVvZGROsusdV30bWuR1p2LHDEi0b6jBzmKmmL4EynIPtOWEqdndPKYzvLnz9oWCgl9EILbmM7HBhFRuDBV3M319KjCK7aoBz2L4YTgqLMbOWYG9myE4TBxlfk7H6Ah+1OxA7lJxHldjUrfuNRFMPq24QFm/fyiZh00vx5k3b3eSSCJOlyQtVlHrruIXE+H7oHM47Mzy2fH6vP3m6ERR9PIBkyjaGqjV1jk57ydpUb+/wMC4GMCRlEdOBW18qlZsZMzv7SDLxxwqGGe8h4xANqNt1Vz+HX8MEu93QYwD2ZsdwZHvxlBdNQ0WkIurUtIOwRd/XVfQCHU1WNDUGItlgOXFUZpzIxg5May3MW+FbK17GVbS9MI1hJsdXubIy+A7/RKRTJb+jCyjcE1JvHl7FwhSbfVfo+dt0V8Mq5IWQigcmPa95zJ2IayeaO5iPlWygh7Z3VikXTsaLaps5OS9DYrImbxXMRphVYS7CX76StJuwAYvpdEZpuLEFf0ZTNyEJU5LTfYgAKzkWxEm14scRzGETCZa1SBzhHkIJnYeYczClWqnCeXCli3EFlrxXd5FCSVejei++33WU4vgryk76UQAyYUWTaxqInq6PfsZ3P+SYn+nycA4hLMUF98nin3HpUVZq7OlMh2sYgPJu1g4rdROp1KIlCgcDobzGdsitvIU2j7XfutgMieM/k6BVbUC+JTyJ92j/D7LznlPoWZL/usePNF/1OioSMC/AyqCeKIgCSu+ChElxZrEd1xKFtE0zcw9Z7CFmIfZBJbZV3Cph2HchVqpdCKl4V8DLlLUWbEDcmk/oraBrqxJfpSR8/wI0MvQktE5GPF2fywSf07H43nISHWc33FG0E2EbGVLOlcbXafKXbAtKlB0ep+tvNQfSuhpEa/Ve3a3QscIiIiPgzeH05ioFN76/fUMwhZeyDSSvwN83dNbMJ63SKzrJEZUzSVsd+g/tyd/6zu/xYIAM8jD7wxaqp9x6PSdxE6e6zMKat1kxd+7NUVeVO1hfZlQaR7b8vDXNqqHzoY+YTzp3XSw7nFfwepQyVk+JstN7RRrdIkU4ngCRGwDTbERmCLchBc2nRV62akoQqU5SZAwV+18Qk64MMiKt0f712MSDER0IAmgOSJpPs7AnKPDwK0vrNDwD0vdUmqrCUwfrcjbiiBe+mZJdxNAFpK4gYW0Ipg/I9TDf0wFuFJYvLrcWK9hM9ao56DML8gIQqebIaANMVVM1lIcsoR549AxTV69Z7isE6P0N7OEMudWrpLg9IiqUz2W221pjEWSqk09cJxTIx81X7kcnb+FwgAXVbvrz13iSl3fjxX6treZFnN5nTq22KjhoCcwISFWVXnzFM3pUq4wwklxMKhRsK43ml1tPKX38AEMbI2HNC7LyNKLQgnnWqdpZ4/y6FO/nr29UcNJHjIKrXDcpGJdjYuucMBtcQ/RgkrGto99qoKMFoPF/bloTK+P22SqlUYz/XVXgVdQhHVhEOH/YwYvCIs7g5WuV1nuZSoRvH6MTEK52Nbifm3tfEsvKkaZgRO7RmMeq6ty+AM+Ag1DNTmJJAT8R2/YZZAutQdTyxEKgiwZZlKGCCWkdTvBJD1nsZkkm0rjaxeLTXLi38EPP/U/yn3bx37n7tctNGsDbO+6fegZ1oBhtPHWsYMSwocWW9E6IisOHZUN11th0caNSbjZijgoQ/OI7YOXAwDJCqDuFUVV9PExotx2gVkqxZ3FAs4ZO+dMHVqvwsWsW6uBYtGceHvbuaGJZYIEKd3QVAS+NzG6i8cwMOOQhQpOWb3TktA0QL9a4/q0lvOS68pZ9e/0z9DMk2Ee8mtLhIqTIO/49S8VA2QmSrEC2bM+AqrkLviMtktEgXr20Hkjj5TUkumlRk3GdnP2OoTCGYs42iRX6RdfVdiu/NAKW8mFMjbPWJCHe4wVDIib4bnPomxv28OfGICOLGN3vGvJwDwlKc0+8tZ76xTUOquwtRc5obmEgFw9IJjvSemnMEeEMQhybCzG+GNBYE2aLGt3WElNRYDYiiGG0b3ImTENaWzjBAD8QOTCjQybL9E9QXoBRUMJeTunxSkFmcMDMYuIQK2eE0r+ZQJBzUiJYuS+aaJnlxCMrxxpQXwYmnAEpYII1unEYnA8pYS91sg3lP0MG5KwB2BCfgTSrgF7qsy+lx0pnSd01Al98bW9rAAv38mMVBHxt8YhpIe8W6cwNrkHW3jtLaidueor+Q0eftFfQlwaK8m9cNsNscXFTHy2k5oPHfTW+H+eu5v4fv75jhKsHPdq7PeWbuQYuB8Qequ0J5aDtkLli6eXYhBCp9x6dCYEvvo+rbLiglBq109cguqIJnwrVAyW3zzIYgpWMsDJthuQAFA5cnWvy9GPpILZMxACCxMhYpBkczpn+3+DjXggrFKs2VHTMRSjxbilgQZxUWpWssymSUAb5ShlhIR/1c7ihQIVyJRhHw4T7oifqzHijvh6DVGd8XjOMkn+o4JkmM8b/0qhNaHahih9V3LN1K4dM8UKUPXF9B2C/t/1OEnPBRZPWvWS82RV3+E8mNHv1Xkj/fpLasCk1fIOmfem/XPnGlrbdnsbwHUYJ1Wlm5+IwYOb4PzhhExErpvCPV1REUbqDUX8LRfy5gJoHh+oFinGAoDK3LbJJKHCv7wxr1o1PgKaxDiJwY5ma3LRHKIkgJuTDDjqjp6RREHlNZo0mzSM+GUurbuR/xshbCyh25fAv1IESUdsw2IRQ8LcNAOoYVQt46T7H10Xqk0FIl8MdSbdm/qtaWV+wLkWLOQtP+Sr8SnYglOTOLSuYrE434la/N1LCgDrGhG596G8Yd2AfZtOI9cK9D9LxMAhOijWbvm+mdeY7om19GsCctzDESOUIp6KNdXNB3X9k0hS5ioQk6GxWAZbqX1KQfTLwL99oglnFTzGQlGUfMM/RGDpVc9XD0mnk0gItGCnBsvEAUfROVhN0ZLeaTWUP3uKTdshDShiEzbPYkkA/1KiEbyHV2OPSbAKUlSD0zhPwU+T3gT4hPLFhHtaiHAodBwPL6apAijXvIx8Xa0IISfd2yRkn0LKlNKRN9sXE0ENDwrIurH5dec325G31iAsb9vBrz2YmC8b+36J5KB3vRq2DDhl69tjs//HCojhSxXJIplEUV3OvbBE0F88peOrW4VHj1x4G8nCXtQiKaBwL7VMorHR1eKwBMobugXAZZK4lV8muUbL85r5A6qDo+pXHuS7uolAQ/lwplVBqV/Qy3DA328Zr1PmkhoLuKR31LjwBTFkyl+cj+S+u2amaZCW0r7ohAjXZLbj2v3K/S9UytwB9uOncn8/Lb4M3Kioa9EJfcLIbVja7NGwKuw5wKLnuAmUmkcPKVpp6iEMT7rO22TXIXMwjw8IMud6l5dOt9WigHGFxqURw6zyM/Vft4y8i9LAkBhwnUmAvXnTjUTb3zf1CZKyAqUb+c+T1KF2myK3r1ACZVY5A4+aOvtwCVwkVggDKO6M48Rg6fkUF9PJSp0GvFbxfWSH23h9zmcApbyOqy+ldDpvxWX8M9tU2C1TdCK4LpmTVJrzv7b2ko64kItl5IQhKCpPbABXu7OdgKYlpg0QqvPU0LS8vg0StMTpUBwecc0EfYShd7LNs+QvUltIv60A3PwdgCpYuf3QX0vkS7CX9aA2UKgVU1M/o+rtOQMysMFU17wUTSevtIy2rem9/+LCIDkCuybR+WpL5rRv5+F2ngJuQJ1k9DsR8uS7keqxsrmChRFJNL9CjXp2wKKCtjw7cmXKN8JbocD8FN1uMtH7sZxlZdUMJL92wObR0Spa+RyuNu95zmhQt5ke6K1t3RaVpitME0v0W5frMMGGG9VIdxyQoCTH0TGD0gQcCiQCwm7jQZzQ1ikVDJPqhfL3ggytxgmk0PWUqD7J+RYJHf67Q2RorHBP2uTyepgTx+N5llaIvQikXYTB3T2MJ3AooWN1OCnRJSM6s+PFszEos+g9szPfTLeMryWMQGgmZNh4oQcqk9/yoy9fgHq1Y4sX3KpdWoTlDFNiEBQCaQVmBvSU11y+9CGuc99rXc1bixmKK9yIDR6tinS80sn8grH9tRdW+aFGGhZQw3kA5yiUNqEm7dwiEAEWzZWEYVYRZHW1uE9v14VcRktWgNdJIk5uPOzauPaU/sWzcV1s4mDuMJlYT1w01jCCbMO7wek0upKOqZ9Pz6H+Nzi5QUZTMVyiGol+xhCV6M9lzBxfh2dk967mFiFOQUpIixd7ZUm+pIAUR0rmtH5x6L6+Gn/CuSPlvgvGJd+miit+5Nc/4xPo9BZM82aKIEiPzmOKK4pbQkRjhqEMB/8kvYk1HnT0fcVdw/Kbogw8yxQxXTL9MXXHbHUdripXTdK/0+4eNCH7QTbWsfT53i7QiIFRSG/9p+Yn8ZxkNFv3RJLI7had+s8RJIIwVAexVqWEVQUWWnKMltiCPReIokz8OfiZQd1q4OdVMJLhYu2+6LJrp5n61r8DisXXeu8fR3IcL9+Lt+SUHyllliBiBPjKIIsj/JIZsYWHY3qnB/9q5D/XyEByOVOiiSBx49qDr/8FVRHirl8kUquuibxTmTSB+j3J7EJSEFE4TptQ2YDkge66sRVxZn4K1GpsiUswBdd0JOLdWZP26MsPj/hWKpwgkCC0mENEtCo9kGrR/Y+zb2TdesZRQZJRY41kUylFn+Pkgj0JMWfExnoUz03GLAiSU/tn+/8GUkQ0Qghlj6aQCy9WcSU6n8hKMvDlV+CbtCpty08IUg86dkmWxLteXqAvpuaMlA6UAi3KCKhpCq+h5G/nlF1n/HF42Zs8T7/auSXaf0rLwkUaqA4+/Bc19BP0TlUMjBVmLpNn9OpnPLLc2X7VmuEXDiyiBjwR/rvIJotIWIo4ZtBRGzP3RLSL5yqJfIt+Yr7TswVk1sjThUz34hiJNQjEXiSQdtwcyeYRQp0BO3tgDuZoEtH8aO7KshRxOESgmwkuCrmomECKTPXUkpMj+wILdw7Pfo2nNpPDD4VVElc+ovqM/3bS6/xM/3oPlpU72M7KiLj8qZQbn8J44teNeXRD6L+xK3/auRXU/9XX24hhdXeiVLvz7OuqbPZOIgmnR3VuG6Zkk/ucFzEwq7O9lLI5KP1RLtMDzoFbOE2inNrTHuT0NoYVhJ3WguRWiL4tYrbiiPE98VXy2fRGzLmm41gv+SDbrS6EgG8RS8fqpxQnSAK2xRqnR3tCXO0hyIiB27pp6+CGVunGhAlxDEl+QgSD+zGjaSoZCgxEIc4gqyFwMagohFV7Z/MQ7tcPUEV8SORHhMZxo1v0+sblTzG5t9oJkY+Arz4wv8G8rvZ/q9dLliodwo6V/x21jn5CHT0kzukmpmmkwYS7hVxLf1RGy63BLWAVYBIaXbA67E4CYttcwWjtMoVT5+ZXIHBakIkeeR6DfK5CjXVc/dzUIJ3W4EmlYr0ZFLulsj2al/sO4mrMSUqPkBKcWJFRFpJkNhFQtPOeHqpDT7OW/Dr9THyCcsVPPYRi3rd6Xm4ltMRCGWKqGl7kBBETQQULCUJmPrwfHEYZdgN5+K4GnKNDKZkyoubZnz4JFTnnOQ+/KfKev2nEABEC8uvvnfWNXgquoZWQi5vm8lnTVUsXZ9qksjiACQAoCCxlFYORRu1yK1QKIwjlweqhNtFpyufyedxFp/lZm2Qy5+7AE8bgiZZYG2kDq9CpLqBWkbIb5cIyiDKyu1LzrMLrr3ArUUY0kArEpJCRk940qwAWYJ7zyFGhGAtmBN7V9QXFTzoM1qCpNNG2tNSjV1iWg8iwuDkSqWzIHEEITDAqH+G9s54IuoQP0PJ1KvAxKIHTHn4c6g/c6s/pmUU5PPvSADkmYQ5DWDqDHRO/UbW2X8ESgMEG7Y4O4yr0Eh3x51d/OsW2VxLAorbp0AcccM3F5NVeaL4fSWGhrTiWCqwlF9JhVFWmxZRAotqy9mjGIG2QrK/r4UbRXsTvqv5ezq2ILVG4lSAsB+o/Wm5WZ6i6bYvvpFRQTMtHdhhYkQN46ZrTs8tpYat6/V3KSIQu3Xb3OgWHRJ+5EN1Y2rcgc5TCcZQNwYXDOMOPlSUtrx4pFkd+wHKc75D7SWXRWLPfwoBaJUGCqtvi47er2al3nej2EsbT7JiHYYkAsGchFK3Ykr0F2+3r0IjbyqpLh2iBbc0hxNXZXiWjlXxnN93uBHAV9wG7esPRCK4w+DwURskbYuZ6QL1trQiQ7u1JtsT+Le3v4Rb2809VqOU6O7e0+pFK/eOZbPkJFvmqkfynh+HjO2JfUrkVMmSLNljO6g7I1f1Se9DW2IQAqsC8fNGUUJ6SgctcYXiysgY6mO/MvWxH6H6wpPu6/9rIv+/EwFIpAHqDDJ7l6zQ8zkUundFqTvnSjXXHOm29/rGoP+Ak0fW6NgGEFIO9f1LmmIQoMN3UsSUx4gQrj/QUoRGDq0XJj7kJUgdKZNrfZbe1uSlkkRjNWhJWxgkkze5KYjkvAwVqxF9K9R7tFso0lKbdYbKpq0Ej/fJFY5NJHatvvh5+ySoeBNiqTCREJaEESp4IVLVvCTgJUDxAdNEc5kxBVOvAJXRxaY2fjHKo6cCLz5hb2au/5Yz+v6TCYBcYhqzG1FYbYus0PNBFDr3zkrdKxuqAW+DfSgriprE0Y22h7S1QSeuIB0QpBFYrvikfRVfkwjIwewcQVzy7WRQO48g8isOr/mhtsKnl/ueqBHxnN8USt0/mjoohEpD6dpdCWLFaw7kRq8qnY8/CU9tZE/0vursy2TP30Q0i42rgZjGNQ5SVaJVHdBLDVcbaUmvT0cwyRv2dutXtSJngdkNVX+ujwPV8SdMY+J/UB7/FfC35xXim/9NXf/fnQDIJTXA3cYMDKK4wruzQn5P5EvboNCxelbo4hZPVs/iFdjC8G02MxEWWy2+CkE0gAuXkNvtb1v2uz3i5mycnGM3ISY9EA5BgVR8DJ+2itHiIUqf511YLWpAsK4ZKk8sREj0WZ1FF+saan9irq/VID1vs6R1tKwhGNzczGKSIWeQWvwjnaO9wNKeEKUEyfF2Fbgg3N4IDLSUr1YP0xMNQ7sNIVuVWwFVhGrUYOoTdTRqc0y9cjtq1avQqN4EvFj+d0P8f1cCIFcO2JcqnWq9qBtYdUOUOrZCvrB5VuhYP8vlVzFZfhD5IvXZtkialKtqdfWFA47hKmtPOnzYcvhuzFyFgCQZevo7AmT8WoiLLnIRyod74E9dbZEur0XjRFRfUqqtErDs7dq04n1oSqQVWiWdgCRgSw/TTqJIcib8fFUshxTTVPst4r+ft1Lv0hTo6Io698i8Q0n5iEhylw73PPV+ppOb/P7Lfeo1/6YcFOq6Qh2E6oTxc2Gaz6FRe9w0aveh0rwHeOrRGMnZp9+WSf3fvv5dCUASSUhXi4U0B0yegWLPDGSdKyPLz0SW9SMz1DqGWr2U+PRs155gALbtcgRbchyMxAcTjAS2fxwb9ChMKUfxy8b28LH9rJrU6INvcH2wkv5RzPQyamdtI7ys49lF1HAdBxfjq7NO5G+6TxLwqXsJdd+LgkYJAt366W9eD3XNcPVRbOVRqrTGzYrJjuXaGFlTpl+lbf2b42dQi2aSpKiXGD2fKjqJQi7d6bkhYHDimfBsv3Zqp0Ox7IJdzFpzzvDfZOO/bQVNzfPotS3wzxvqnmHf5w1WFET5hrVs1mSl0DUtIX07b8iTRP3yFHfm9fnvCw76vXf38EBCTWWZ0uzctiXOmqNU5xWmPg/1+qtomFeAhS8BCxe3gi7n7OP/to7//9LlgJGoKf382xOv5df/f66cRXiGSwmd/I+4/mMm2uby3NaqC6+3Wcu0f0B56Tv/6J7l11u/9Nn8J+73621g61atV/0Hrmn5tfxafi2/ll/Lr+XX8mv5tfxafi2/ll/Lr+XX8mv5tfxafi2/ll/Lr+XX8mv5tfxafi2/ll/Lr+XX8mv5tfxafuH/lev/A9+xNNjU2yKWAAAAAElFTkSuQmCC'))
    app.setWindowIcon(QIcon(icon_pixmap))

    window = MainWindow()

    window.showMaximized()

    sys.exit(
        app.exec()
    )


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        index = sys.argv.index("--self-test")
        sys.exit(packaged_self_test(sys.argv[index + 1]))
    main()

