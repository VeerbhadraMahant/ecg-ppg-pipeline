const fs = require('fs');
const path = require('path');
const D = require('docx');
const REFS = require('./refs.js');
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, AlignmentType,
  HeadingLevel, BorderStyle, ShadingType, Footer, PageNumber, PageBreak, LevelFormat, TabStopType
} = D;

const FIG = path.join(__dirname, 'figures');
const FONT = 'Times New Roman';
const TW = 9026; // text width in DXA (A4, 1 inch margins)

// ---------- citation bookkeeping ----------
const order = [];
function c(...keys) {
  const nums = keys.map(k => {
    if (!REFS[k]) throw new Error('unknown ref ' + k);
    let i = order.indexOf(k);
    if (i < 0) { order.push(k); i = order.length - 1; }
    return i + 1;
  });
  nums.sort((a, b) => a - b);
  return '[' + nums.join('], [') + ']';
}

// ---------- text helpers ----------
function runs(text, opt = {}) {
  // *italic* markup only
  const out = [];
  const parts = text.split(/(\*[^*]+\*)/g).filter(s => s !== '');
  for (const p of parts) {
    if (p.startsWith('*') && p.endsWith('*') && p.length > 2)
      out.push(new TextRun({ text: p.slice(1, -1), italics: true, font: FONT, size: opt.size || 28, bold: opt.bold }));
    else out.push(new TextRun({ text: p, font: FONT, size: opt.size || 28, bold: opt.bold, italics: opt.italics }));
  }
  return out;
}
const body = (text, opt = {}) => new Paragraph({
  children: runs(text, opt), alignment: opt.align || AlignmentType.JUSTIFIED,
  spacing: { after: opt.after ?? 120, line: 276 }, indent: opt.indent,
  keepNext: opt.keepNext,
});
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT, size: 32, bold: true })], spacing: { before: 320, after: 160 }, keepNext: true });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT, size: 28, bold: true })], spacing: { before: 220, after: 100 }, keepNext: true });
const h3 = (t) => new Paragraph({ children: [new TextRun({ text: t, font: FONT, size: 28, bold: true, italics: true })], spacing: { before: 140, after: 80 }, keepNext: true });
const center = (t, opt = {}) => new Paragraph({ children: runs(t, opt), alignment: AlignmentType.CENTER, spacing: { after: opt.after ?? 120 } });
const bullets = (items) => items.map(t => new Paragraph({
  numbering: { reference: 'bul', level: 0 }, children: runs(t), alignment: AlignmentType.JUSTIFIED, spacing: { after: 70, line: 276 } }));
const numbered = (items, ref = 'num') => items.map(t => new Paragraph({
  numbering: { reference: ref, level: 0 }, children: runs(t), alignment: AlignmentType.JUSTIFIED, spacing: { after: 70, line: 276 } }));

// ---------- figures and tables ----------
let figN = 0, tabN = 0;
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function fig(file, widthIn, caption) {
  figN += 1;
  const f = path.join(FIG, file);
  const { w, h } = pngSize(f);
  const wpx = Math.round(Math.min(widthIn, 6.2) * 96), hpx = Math.round(wpx * h / w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: 'png', data: fs.readFileSync(f), transformation: { width: wpx, height: hpx },
        altText: { title: `Figure ${figN}`, description: caption, name: file } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: `Figure ${figN}: `, bold: true, font: FONT, size: 24 }), ...runs(caption, { size: 24 })] }),
  ];
}
function eq(file, widthIn, number) {
  const f = path.join(FIG, file);
  const { w, h } = pngSize(f);
  const wpx = Math.round(Math.min(widthIn, 6.2) * 96), hpx = Math.round(wpx * h / w);
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 120 },
    children: [new ImageRun({ type: 'png', data: fs.readFileSync(f), transformation: { width: wpx, height: hpx },
      altText: { title: `Equation ${number}`, description: `Equation ${number}`, name: file } }),
      new TextRun({ text: number === '' ? '' : `    (${number})`, font: FONT, size: 24 })] });
}
const border = { style: BorderStyle.SINGLE, size: 4, color: '000000' };
const borders = { top: border, bottom: border, left: border, right: border };
function cell(text, w, o = {}) {
  return new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    shading: undefined,
    margins: { top: 50, bottom: 50, left: 80, right: 80 },
    children: String(text).split('\n').map(line => new Paragraph({
      alignment: o.align || AlignmentType.LEFT, spacing: { after: 0 },
      children: runs(line, { size: o.size || 20, bold: o.head || o.bold }) })),
  });
}
function table(caption, widths, header, rows, o = {}) {
  tabN += 1;
  const sum = widths.reduce((a, b) => a + b, 0);
  if (sum !== TW) widths = widths.map(x => Math.round(x * TW / sum));
  const diff = TW - widths.reduce((a, b) => a + b, 0); widths[0] += diff;
  const trs = [new TableRow({ tableHeader: true, children: header.map((t, i) => cell(t, widths[i], { head: true, size: o.size })) })];
  rows.forEach(r => {
    const bold = r._bold; const cells = r.cells || r;
    trs.push(new TableRow({ cantSplit: true, children: cells.map((t, i) => cell(t, widths[i], {
      size: o.size, bold, fill: r._fill, align: (o.numFrom !== undefined && i >= o.numFrom) ? AlignmentType.CENTER : AlignmentType.LEFT })) }));
  });
  return [
    new Paragraph({ spacing: { before: 160, after: 80 }, keepNext: true, alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: `Table ${tabN}: `, bold: true, font: FONT, size: 24 }), ...runs(caption, { size: 24 })] }),
    new Table({ width: { size: TW, type: WidthType.DXA }, columnWidths: widths, rows: trs }),
    new Paragraph({ spacing: { after: 160 }, children: [] }),
  ];
}
const T = JSON.parse(fs.readFileSync(path.join(FIG, 'table_rows.json')));
const f3 = x => x.toFixed(3);
const pm = (a) => `${a[0].toFixed(3)} ± ${a[1].toFixed(3)}`;
const metricRows = (rows) => rows.map(r => [r.label, f3(r.acc[0]), pm(r.f1), pm(r.auc), f3(r.sens[0]), f3(r.spec[0])]);

// =====================================================================
const S = [];  // all body children in order
const P = (...a) => S.push(...a.flat());

// ---------- title page ----------
P(new Paragraph({ spacing: { before: 900 }, children: [] }));
P(center('Mini-Project Report', { size: 32, bold: true, after: 200 }));
P(center('TY B.Tech Artificial Intelligence and Machine Learning', { size: 28, after: 500 }));
P(center('Verifying ICU Arrhythmia Alarms from ECG and Pulse Signals: Cross-Attention Fusion Compared with Feature-Based and Deep Learning Baselines', { size: 40, bold: true, after: 600 }));
P(center('Submitted by', { size: 28, after: 120 }));
['Veerbhadra Mahant', 'Ved Jadhav', 'Sharvil Patil'].forEach(n => P(center(n, { size: 30, bold: true, after: 40 })));
P(center('', { after: 200 }));
P(center('Under the guidance of', { size: 28, after: 100 }));
P(center('Dr. Roshani Raut', { size: 30, bold: true, after: 40 }));
P(center('Department of Information Technology', { size: 28, after: 400 }));
P(center('Pimpri Chinchwad College of Engineering, Pune', { size: 30, bold: true, after: 60 }));
P(center('Pune, India', { size: 28, after: 300 }));
P(center('veerbhadra.mahant24@pccoepune.org  |  ved.jadhav24@pccoepune.org', { size: 24, after: 40 }));
P(center('sharvil.patil24@pccoepune.org  |  roshani.raut@pccoepune.org', { size: 24, after: 300 }));
P(center('October 2026', { size: 28 }));
P(new Paragraph({ children: [new PageBreak()] }));

// ---------- abstract ----------
P(h1('Abstract'));
P(body('False alarms from ICU bedside monitors wear down staff attention, and an alarm that gets ignored can be a real one. The project asks whether a second signal, the pulse waveform (a photoplethysmogram, PPG, or arterial pressure where PPG is missing), can verify an arrhythmia alarm that the ECG has triggered, and whether cross-attention is the right way to combine the two. Two open PhysioNet datasets are used: the 2015 Computing in Cardiology Challenge training set (750 records, of which 592 form the PPG cohort) and VTaC (4,542 usable ventricular tachycardia alarms). Each input is the 10 s of ECG and pulse that end at the alarm onset, filtered and z-normalised. The cross-attention network is compared with ECG-only, PPG-only and concatenation models of equal size, other neural variants, six classical learners on 266 engineered features, an attention CNN-GRU recipe and ensembles, all under record-wise cross-validation with an inner validation set. After a validation leak in the first version of the code was fixed, cross-attention reached F1 0.699 and AUC 0.841 on Challenge 2015, no better than concatenation (F1 0.693). The attention CNN-GRU recipe reached F1 0.828 on Challenge 2015 and F1 0.858 (accuracy 0.913, AUC 0.966) on the VTaC test split. Counterfactual tests show that the networks use what each signal looks like, not the timing between ECG and pulse. The main limitations are small data, no prospective validation and weak transfer from Challenge 2015 to VTaC. Larger pretraining corpora and leave-one-site-out testing are the sensible next steps.', { after: 120 }));
P(new Paragraph({ children: [new PageBreak()] }));

// ---------- 1 Introduction ----------
P(h1('1. Introduction'));
P(h2('1.1 Background'));
P(body(`A bedside monitor in an intensive care unit (ICU) watches the electrocardiogram (ECG) around the clock and raises an alarm whenever the rhythm looks dangerous. Many of those alarms are wrong. In one review of 5,386 critical ECG arrhythmia alarms from 447 adult patient records, 42.7% were false ${c('abou')}. Staff who hear false alarms all shift tend to respond more slowly, a pattern known as alarm fatigue ${c('drew', 'cvach')}, and the real alarm that arrives late is the outcome everyone is trying to avoid.`));
P(body(`False alarms are commonly caused by ECG artefacts and low-voltage signals on a single channel ${c('abou')}. That makes checking the alarm awkward, because a loose electrode or a moving patient corrupts the very signal a second look would rely on. Ventricular tachycardia (VT) is the hardest case. VT alarms are among the most frequent life-threatening alarms, and false ones have been the most difficult to detect reliably ${c('vtac')}.`));
P(h2('1.2 Importance of AI, ML and deep learning'));
P(body(`Rule-based alarm logic and hand-built features both depend on a beat detector, and detectors fail on exactly the noisy segments that matter. A deep network can read the raw waveform and learn what to look at, and an attention layer can learn which part of one signal should be compared with which part of another ${c('vaswani')}. Published attention-based networks for this task report sensitivity above 90% on the 2015 Challenge data ${c('mousavi')}, which is the argument for deep learning here.`));
P(body('That argument needs a fair test, and the measurements in this report only partly support it. With a few hundred labelled records, carefully engineered features were competitive with plain deep networks on one dataset and ahead of them on the other. Sections 7 and 8 give the numbers.'));
P(h2('1.3 Problem definition'));
P(body('The input is one alarm event: the 10 seconds of ECG and pulse signal (PPG, or arterial blood pressure when PPG is not recorded) that end at the moment the monitor raised the alarm, sampled at 250 Hz, so each signal has 2,500 samples. The output is a probability that the alarm is true. The task is binary classification, true alarm versus false alarm, and the intended application is decision support: ranking or suppressing alarms that are very likely false while keeping sensitivity to real events high. The system is a research prototype and is not a medical device.'));
P(h2('1.4 Objectives'));
P(...numbered([
  'Prepare leak-free, decision-time windows from two open ICU alarm datasets and define record-wise evaluation splits.',
  'Implement a cross-attention fusion network and compare it with ECG-only, PPG-only and concatenation baselines of equal parameter count.',
  'Benchmark it against other neural architectures, classical models on engineered features, a stronger deep training recipe and ensembles, using the same splits.',
  'Test with perturbation experiments whether the fusion models rely on the timing between ECG and pulse, and measure robustness to noise, sensor loss and a change of dataset.',
  'Report every result with corrected statistics, analyse misclassifications and state the limitations.'], 'obj'));
P(h2('1.5 Scope'));
P(body('Included: retrospective experiments on the PhysioNet/CinC Challenge 2015 training set and the VTaC database, an AF-screening side task on MIMIC PERform AF, noise-robustness tests on MIT-BIH Noise Stress Test recordings, and a comparison with an open PPG foundation model. Excluded: prospective or bedside testing, any regulatory claim, data from outside US ICUs, demographic or skin-tone analysis of optical signals, and the other alarm types in VTaC (the database only contains VT alarms). The 500-record Challenge 2015 test set was never released, so no result here uses it.'));

// ---------- 2 Literature survey ----------
P(h1('2. Literature Survey'));
P(body(`The idea of using a second physiological signal to question an ECG alarm is not new. Aboukhalil and colleagues built a suppression algorithm from the shape and timing of the arterial blood pressure waveform and tested it on alarms from the MIMIC II database. It cut the share of false critical ECG alarms from 42.7% to 17.2% where both signals were available, but it also suppressed 9.4% of true VT alarms, which the authors flagged as the reason for further refinement ${c('abou')}.`));
P(body(`The 2015 PhysioNet/Computing in Cardiology Challenge turned the problem into a shared benchmark. Organisers released 750 five-minute ICU segments for training, kept 500 back for testing, and scored entries with a penalty that weights a suppressed true alarm five times more heavily than an accepted false one; 38 teams submitted 215 entries ${c('cinc', 'cliff16')}. The released training set is the starting point for most later work, including this project.`));
P(body(`Deep learning entered through that dataset. Mousavi, Fotoohinasab and Afghah combined convolutional layers, an attention mechanism and LSTM units, trained in two steps (pre-training, then fine-tuning), and reported 93.88% sensitivity and 92.05% specificity using three signals, and 90.71% sensitivity, 88.30% specificity and an AUC of 89.51 for VT alarms with a single ECG lead ${c('mousavi')}. Zhou and co-authors added a similarity loss from pair-wise comparisons of waveform segments to the usual classification loss and fed in learned embeddings from a rule-based method for each alarm type; they report better performance than the Challenge’s winning entries ${c('zhou')}. We re-implemented the first of these as the starting point for our strongest recipe, working from the published description, so our version is not the authors’ code.`));
P(body(`Data have grown since. VTaC offers more than 5,000 VT alarm recordings from bedside monitors, each labelled by at least two independent annotators, together with benchmark results for conventional, deep, semi-supervised and generative methods ${c('vtac')}. On the signal side, a benchmark of 15 PPG beat detectors found that eight performed well at rest in hospital and wearable data, but accuracy dropped in atrial fibrillation, in neonates and during exercise ${c('charlton')}. PaPaGei, an open PPG foundation model pretrained on about 57,000 hours of data, improved classification metrics by 6.3% and regression metrics by 2.9% in at least 14 of 20 tasks ${c('papagei')}.`));
P(...table('Comparison of the key studies reviewed. Results are quoted from the cited sources; "not stated" means the abstract does not give a number.',
  [1500, 1850, 1850, 2000, 1850],
  ['Study', 'Data', 'Method', 'Reported result', 'Limitation or relevance'],
  [
    [`Aboukhalil et al., 2008 ${c('abou')}`, 'MIMIC II: 5,386 critical ECG alarms, 447 adult records', 'Suppression rules from ABP morphology and timing', 'False alarms 42.7% to 17.2%; 59.7% of false alarms suppressed; true alarms lost: 0% except VT (9.4%)', 'Needs ABP; some true VT alarms suppressed; prospective evaluation recommended'],
    [`Clifford et al., 2015 and 2016 ${c('cinc', 'cliff16')}`, 'Challenge 2015: 750 training and 500 hidden test segments, 5 min each', 'Open challenge; 5x penalty for suppressed true alarms', '38 teams, 215 entries; best score not stated', 'Only the 750 training records are public; defines the data and the score used here'],
    [`Mousavi et al., 2020 ${c('mousavi')}`, 'Challenge 2015', 'CNN, attention, LSTM; pre-training then fine-tuning', 'Sens. 93.88%, spec. 92.05% (3 signals); VT, one ECG lead: sens. 90.71%, spec. 88.30%, AUC 89.51', 'Protocol not reproduced, so not directly comparable with ours'],
    [`Zhou et al., 2022 ${c('zhou')}`, 'Challenge 2015', 'CNN with classification plus similarity loss; rule-based alarm-type embeddings', 'Better than the winning Challenge entries (numbers not stated)', 'Needs alarm-type knowledge; figures not stated'],
    [`Lehman et al., 2023 ${c('vtac')}`, 'VTaC: over 5,000 VT alarms, at least 2 annotators', 'Benchmark of conventional, deep, semi-supervised and generative methods', 'Dataset and benchmark paper; results in the paper', 'VT alarms only; no detailed clinical information'],
    [`Pillai et al., 2025 ${c('papagei')}`, '20 tasks, 10 datasets; pretrained on about 57,000 h of PPG', 'Open PPG foundation model (PaPaGei)', '+6.3% classification, +2.9% regression in at least 14 tasks', 'Listed tasks do not include ICU alarm verification, so transfer had to be tested'],
    [`Charlton et al., 2022 ${c('charlton')}`, '8 datasets, 15 PPG beat detectors', 'Benchmark framework for beat detection', 'Eight detectors F1 of 90% or more at rest; poorer in AF, neonates, exercise', 'Source of the MIMIC PERform AF data used in the AF side task'],
  ], { size: 20 }));
P(body(`Taken together, the reviewed work leaves a practical question open. Whether numbers from different papers can be lined up depends on splits and validation details that the abstracts do not give. A fair comparison needs the same splits, the same parameter budget, a validation set separate from the test fold, and strong non-deep baselines next to the networks. The project addresses that gap: a leakage-audited comparison on two datasets, and a direct test of whether attention between ECG and pulse is doing what its design assumes.`));

// ---------- 3 Dataset ----------
P(h1('3. Dataset Used'));
P(body(`Two labelled datasets carry the main experiments. Table ${tabN + 1} summarises all data sources, and the sub-sections give the details. The source pages were last checked on 14 October 2026.`));
P(...table('Datasets and external resources.', [1700, 1250, 1700, 1500, 2876],
  ['Dataset', 'Samples', 'Features', 'Classes / target', 'Source and citation'],
  [
    ['PhysioNet/CinC Challenge 2015, PPG cohort', '592 records (724 in the recovered cohort)', '2 channels x 2,500 samples (raw); 266 engineered features', 'True alarm (223), false alarm (369)', `${c('cinc', 'cincdata')}; physionet.org/content/challenge-2015/1.0.0/`],
    ['VTaC v1.1', '4,542 events kept of 5,037', '2 channels x 2,500 samples (raw); 266 engineered features', 'True VT alarm (28.0%), false', `${c('vtac', 'vtacdata')}; physionet.org/content/vtac/`],
    ['MIMIC PERform AF', '35 subjects; 2,100 windows used', 'ECG and PPG, 125 Hz', 'AF (19 subjects), non-AF (16)', `${c('perform', 'charlton')}; zenodo.org/records/15906524`],
    ['MIT-BIH Noise Stress Test', 'noise recordings', 'baseline wander, electrode motion, muscle artefact', 'not applicable (noise source)', `${c('nstdb')}; physionet.org/content/nstdb/1.0.0/`],
  ], { size: 20 }));
P(h2('3.1 PhysioNet/CinC Challenge 2015'));
P(body(`The training set has 750 five-minute ICU records at 250 Hz, each holding a bedside-monitor alarm that fires at the 300 s mark and an expert true or false label for one of five alarm types ${c('cinc', 'cliff16')}. Recordings were read with the WFDB tools from PhysioNet ${c('physionet')}. Of the 750 records, 627 have a PPG channel, 343 have arterial blood pressure (ABP) and all have a second ECG lead. The original PPG cohort keeps the 592 records with a usable PPG window, of which 37.7% are true alarms. A recovered cohort of 724 records (39.6% true) uses ABP as the pulse channel for 132 records without PPG and drops 26 records whose window is too short or nearly flat.`));
P(...table('Alarm types in the 592-record PPG cohort.', [3400, 1800, 1900, 1926],
  ['Alarm type', 'Records', 'True alarms', 'False alarms'],
  [['Asystole', '93', '14', '79'], ['Extreme bradycardia', '73', '39', '34'], ['Extreme tachycardia', '111', '106', '5'], ['Ventricular flutter / fibrillation', '45', '5', '40'], ['Ventricular tachycardia', '270', '59', '211'], { cells: ['Total', '592', '223', '369'], _bold: true }], { numFrom: 1 }));
P(body('The table shows how uneven the data are. Almost every tachycardia alarm is true, almost every flutter/fibrillation alarm is false, and asystole and flutter/fibrillation have too few true examples for per-type conclusions. VT, the largest group, is mixed.'));
P(h2('3.2 VTaC'));
P(body(`VTaC contains 5,037 annotated VT alarm events from bedside monitors of three manufacturers in US hospitals, with at least two expert annotators per event and the alarm at the 300 s mark of each recording ${c('vtac', 'vtacdata')}. The database licence is CC BY-SA 4.0. Version 1.1 also releases the individual annotator votes. The host throttles every connection, so only the window from 60 s before to 30 s after each alarm was downloaded with parallel range requests, about 1 GB instead of 18 GB. After removing 292 events with no pulse channel and 203 low-quality windows, 4,542 events from 2,138 patients remained, 28.0% of them true alarms. The official patient-disjoint split gives 3,662 training, 453 validation and 427 test events, with 120 true alarms in the test split. Hospital and device identifiers are not released.`));
P(h2('3.3 Data quality, imbalance and splits'));
P(body('Quality control was simple and applied to each window on its own samples: windows that were too short, almost flat or clipped, or with a quality score below 0.05, were dropped, and the counts above say how many. Each Challenge 2015 record yields one window, so splitting by record and by window is the same thing. Duplicate detection across records was not performed beyond record identifiers. Class imbalance (about 38% true alarms in Challenge 2015 and 28% in VTaC) was handled in the loss function, not by resampling, because resampling can place copies of one patient on both sides of a split.'));
P(body('Challenge 2015 was evaluated with record-wise 5-fold cross-validation repeated for 10 seeds, which gives folds of about 119 test records. In each fold, 15% of the remaining records form an inner validation set (about 71 records), leaving about 402 for fitting. VTaC uses the official split. The test fold or test split was used once per model to produce the numbers in Section 7. No augmentation or resampling was applied to validation or test data.'));
P(h2('3.4 Auxiliary data'));
P(body(`MIMIC PERform AF holds 20-minute recordings of 35 critically ill adults at 125 Hz, 19 with atrial fibrillation and 16 without ${c('perform', 'charlton')}. Sixty evenly spaced 10 s windows per subject (2,100 in total) were used for a side task. Real noise for the stress tests came from the MIT-BIH Noise Stress Test database ${c('nstdb')}. Each dataset’s licence allows research use, and redistribution of derived windows should follow the stricter of the PhysioNet terms and CC BY-SA 4.0. Patient overlap between the corpora cannot be checked from the released identifiers.`));

// ---------- 4 Methodology ----------
P(h1('4. Proposed Methodology'));
P(h2('4.1 Overall approach'));
P(body('The project did not start from a finished design. It started from one hypothesis, that cross-attention between ECG and pulse would verify alarms better than simpler fusion, and then tried to break that hypothesis with fairer tests. A cross-attention network was built first, with three same-size relatives (ECG-only, PPG-only, concatenation) so any difference could be traced to the fusion step. Stronger rivals were added next: other fusion layers, other network families, feature-based learners and a better training recipe. The last stage perturbed the inputs to see what the networks actually use.'));
P(h2('4.2 Workflow'));
P(...fig('workflow.png', 6.1, 'Workflow from problem definition to final evaluation. Splits are made by record before any model is fitted, and the outer test data are touched once.'));
P(h2('4.3 Preprocessing'));
P(body('Only these steps were applied. Signals were resampled to 250 Hz, then band-pass filtered with a zero-phase fourth-order Butterworth filter (ECG 0.5 to 40 Hz, pulse 0.5 to 8 Hz). Each 10 s window was z-normalised on its own samples, which removes absolute amplitude differences caused by gain and placement. The window is aligned to end at the alarm onset. An earlier version centred the window on the alarm, which let the model see samples after it; that was corrected, and the 30 s after the alarm are now used only in a separate time-to-verdict analysis. For the classical models, 266 features were computed per window and imputed, clipped and standardised using statistics from the fitting data of each fold only.'));
P(h2('4.4 Model selection'));
P(body(`Cross-attention was chosen first because the project hypothesis was about alignment between two signals. Its comparators are the ladder variants, built to about 261k parameters each by scaling channel widths, so a gap between them is not a capacity effect. The wider set covers four other fusion layers (bidirectional cross-attention, joint transformer, bottleneck tokens, quality-gated cross-attention) and six early-fusion families: ResNet1D ${c('he')}, InceptionTime ${c('fawaz')}, a bidirectional GRU, a temporal convolutional network (TCN) ${c('bai')}, a patch transformer and a small state-space model. That makes 14 neural architectures. Classical learners (logistic regression, RBF-SVM, random forest, extra trees, XGBoost ${c('chen')} and an MLP) were run on engineered features, with a Pan-Tompkins-style beat detector ${c('pan')} behind the timing features, because the feature route is the obvious non-deep alternative and was likely to be strong with only a few hundred records.`));
P(body('The strongest recipe came from a grid declared before any run: six architectures, augmentation on or off, two losses and two learning rates, 48 configurations in all, scored only on inner-validation AUC. An attention CNN-GRU, following the design of Mousavi et al. as far as the published description allows, won the grid and was then combined with weight averaging, test-time augmentation and a three-member ensemble.'));
P(h2('4.5 Implementation details'));
P(body(`All code is Python 3. Models use PyTorch ${c('torch')}; classical models and metrics use scikit-learn ${c('sklearn')} and XGBoost; signal processing uses SciPy, PyWavelets and pandas. Experiments ran on one workstation with an RTX PRO 4000 Blackwell GPU (24 GB) and 20 CPU cores. A single cross-attention fold trains in about 5 s on average, one recipe fold (three members) in about 21 s on Challenge 2015 and 45 s on VTaC. The repository, listed in Annexure A, holds the code, configuration files and every result file used in this report.`));

// ---------- 5 Architecture ----------
P(h1('5. Model Architecture'));
P(h2('5.1 Cross-attention fusion network (the proposed model)'));
P(body('Each signal goes through its own one-dimensional CNN, with separate weights because ECG and pulse look and fail so differently. The encoders keep the time axis instead of pooling it away, since attention needs a sequence to compare. ECG positions act as queries over the pulse positions, which supply keys and values, and the fused sequence is averaged over time before a small classification head.'));
P(...fig('arch_ladder.png', 6.4, 'Architecture of the cross-attention fusion network. Shapes are (time steps x channels). The two encoders do not share weights.'));
P(...table('Layer-by-layer description of the cross-attention fusion network (width multiplier 1.0; 261,505 trainable parameters).', [2700, 1700, 4626],
  ['Layer', 'Output shape', 'Role'],
  [
    ['Input (ECG and PPG/ABP)', '2 x (2500)', 'Two 10 s windows at 250 Hz, z-normalised'],
    ['Conv1d(1 to 32, k = 15), BN, ReLU, MaxPool 2, Dropout 0.3', '(1250 x 32)', 'Local waveform features, such as QRS shape or pulse upstroke'],
    ['Conv1d(32 to 64, k = 9), BN, ReLU, MaxPool 2, Dropout 0.3', '(625 x 64)', 'Longer-range patterns'],
    ['Conv1d(64 to 128, k = 7), BN, ReLU, MaxPool 2, Dropout 0.3', '(312 x 128)', 'Final sequence of 312 feature vectors per signal'],
    ['Linear(128 to 128) + sinusoidal positional encoding', '(312 x 128)', 'Gives attention a sense of time order'],
    ['Multi-head attention, 4 heads, dimension 128; ECG queries, PPG keys and values', '(312 x 128)', 'Each ECG step collects the pulse steps it finds relevant'],
    ['Residual connection + LayerNorm', '(312 x 128)', 'Stabilises the fused sequence'],
    ['Mean over time', '(128)', 'Summary vector for the whole window'],
    ['Linear(128 to 64), ReLU, Dropout 0.3, Linear(64 to 1), sigmoid', '(1)', 'Probability that the alarm is true'],
  ], { size: 20 }));
P(body(`The ECG-only and PPG-only models use one encoder and a mean-pooled head, with channel widths scaled by 1.79 to reach the same parameter count. The concatenation model keeps both encoders but replaces attention with a linear projection of each pooled signal followed by concatenation. Details are in the configuration files of the repository. The sigmoid output is a probability, and the loss and optimiser are described in Section 6.`));
P(h2('5.2 Attention CNN-GRU recipe (the strongest deep model)'));
P(body(`This model replaces cross-attention with a recurrent branch per signal. A three-block CNN shrinks 2,500 samples to 39 time steps, a bidirectional GRU reads them, and additive attention pools the sequence into one context vector. The two context vectors are concatenated and classified. Three members with different initialisations are trained and their probabilities averaged.`));
P(...fig('arch_recipe.png', 6.4, 'Architecture of the attention CNN-GRU recipe. Per-signal branches do not share weights; three members are averaged at test time.'));
P(...table('Layer-by-layer description of one member of the recipe (width multiplier 1.287; 258,857 trainable parameters).', [2900, 1500, 4626],
  ['Layer (per signal branch)', 'Output shape', 'Role'],
  [
    ['Conv1d(1 to 21, k = 15), BN, ReLU, MaxPool 4', '(625 x 21)', 'Waveform features, coarse time step'],
    ['Conv1d(21 to 41, k = 7), BN, ReLU, MaxPool 4', '(156 x 41)', 'Intermediate features'],
    ['Conv1d(41 to 82, k = 5), BN, ReLU, MaxPool 4', '(39 x 82)', 'Sequence of 39 steps'],
    ['Bidirectional GRU, 82 units per direction, Dropout 0.3', '(39 x 164)', 'Temporal context in both directions'],
    ['Additive attention pooling (dimension 82)', '(164)', 'Weighted summary of the window'],
    ['Concatenate ECG and PPG branches', '(328)', 'Joint representation'],
    ['Linear(328 to 64), ReLU, Dropout 0.3, Linear(64 to 1)', '(1)', 'Logit; sigmoid gives the probability'],
  ], { size: 20 }));
P(body('Each branch has 118,868 parameters and the head has 21,121, which adds up to the reported 258,857. Training uses augmentation (a joint time shift of up to one second, amplitude scaling, noise and random dropout of one modality), focal loss, AdamW with a one-cycle schedule, an exponential moving average of the weights and, at test time, averaging over five small time shifts.'));
P(h2('5.3 Transfer learning with a PPG foundation model'));
P(body(`A side experiment tested whether an open PPG foundation model could replace the scratch PPG encoder. The pretrained network was PaPaGei-S (a ResNet1D with a mixture-of-experts, 512-dimensional embeddings) ${c('papagei')}, with weights from Zenodo record 13983110 and the authors’ code (BSD-3-Clause-Clear licence; the weights record states no licence). Its input is 10 s at 125 Hz, so windows were decimated from 250 Hz. Three setups were run: a frozen encoder with a linear probe, a frozen encoder feeding a scratch ECG CNN and the same fusion layers, and fine-tuning of only the last residual block and final batch normalisation at a learning rate of 1e-4 while all other layers stayed frozen. Low-rank adapters were not used.`));

// ---------- 6 Math ----------
P(h1('6. Mathematical Model'));
P(body('Only the mathematics that the code uses is given here.'));
P(h2('6.1 Activation and output'));
P(body('Every convolution is followed by batch normalisation and the rectified linear unit'));
P(eq('eq_relu.png', 1.6, 1));
P(body('The final logit z is turned into the probability that the alarm is true by the sigmoid function, and a decision threshold of 0.5 gives the class labels reported in Section 7.'));
P(eq('eq_sigmoid.png', 1.7, 2));
P(h2('6.2 Cross-attention'));
P(body('Let E and P be the ECG and PPG feature sequences after the linear projection and positional encoding, each of shape (T\' x d). With learned matrices W_Q, W_K and W_V, the attention output is'));
P(eq('eq_attn.png', 4.2, 3));
P(body('where d_k is the key dimension (128 divided by the 4 heads). Row i of the softmax gives the weights with which ECG step i reads the pulse steps. If the network really tracked the pulse transit time, these weights would peak at a lag matching the physiological delay; Section 8 shows that they do not.'));
P(h2('6.3 Recurrent layer and attention pooling'));
P(body('The recipe uses a gated recurrent unit with update gate z_t, reset gate r_t and candidate state h~_t:'));
P(eq('eq_gru1.png', 4.6, ''));
P(eq('eq_gru2.png', 5.2, 4));
P(body('Here sigma is the sigmoid, the circled dot is element-wise multiplication, and W, U and b are learned. Running the GRU forwards and backwards over the 39 steps gives states h_t of size 164, which additive attention pools into one context vector c:'));
P(eq('eq_addattn.png', 5.0, 5));
P(h2('6.4 Loss function'));
P(body(`Both model families are trained with a focal loss ${c('lin')}, a weighted binary cross-entropy that multiplies each example by (1 minus p_t) raised to a power gamma, with gamma = 2. Here p_t is the probability the model gives to the correct class and w is a positive-class weight for the imbalance:`));
P(eq('eq_focal.png', 6.0, 6));
P(body('Well-classified examples (p_t near 1) are down-weighted, so training concentrates on the hard ones, which suits a task where most alarms are easy and the rare true alarms matter most. In the recipe grid, a binary cross-entropy with label smoothing of 0.05 was tested as the alternative.'));
P(h2('6.5 Optimisers and weight averaging'));
P(body(`The baseline ladder uses Adam. The recipe uses AdamW ${c('adamw')}, which applies weight decay lambda directly to the weights instead of through the gradient, with a one-cycle learning-rate schedule ${c('smith')} (15% warm-up, then cosine decay). With gradient g_t, moment decays beta_1 and beta_2, and bias-corrected moments, one step is`));
P(eq('eq_adamw.png', 6.4, 7));
P(body('The recipe also keeps an exponential moving average of the weights (horizon of about three epochs), and the averaged weights at their best inner-validation epoch are used for prediction:'));
P(eq('eq_ema.png', 2.6, 8));
P(h2('6.6 Regularisation'));
P(body('Dropout (rate 0.3) is applied after each convolution block and in the head; weight decay is 1e-4 for the ladder and 1e-2 for the recipe; early stopping on inner-validation data ends training when the validation score stops improving. Time-shift, scaling, noise and modality-dropout augmentation act as further regularisers in the recipe.'));

// ---------- 7 Results ----------
P(h1('7. Results and Experiments'));
P(h2('7.1 Experimental setup'));
P(body('All experiments ran on the workstation described in Section 4.5. For Challenge 2015 each seed gives five folds with about 402 fitting, 71 inner-validation and 119 test records; seeds 0 to 9 vary the fold assignment and the initialisation (50 test evaluations per model). For VTaC the official split gives 3,662 training, 453 validation and 427 test events (120 true alarms), and seeds 0 to 4 vary the initialisation (10 seeds for the two single-signal models). The recipe grid used seeds 100 to 102 on the Challenge 2015 PPG cohort and never produced test predictions. The configuration file sets a global seed of 42. Every model sees the identical splits, so comparisons are paired.'));
P(h2('7.2 Hyperparameter details'));
P(body('Most settings were fixed in advance and not tuned. The table states which ones were searched and which were simply set.'));
P(...table('Hyperparameters: values searched and final values.', [1900, 3500, 3626],
  ['Parameter', 'Values tested', 'Final value'],
  [
    ['Learning rate', 'Recipe grid: 3e-4 and 1e-3. Ladder: not tuned', '1e-3 (ladder and recipe)'],
    ['Batch size', 'Not tuned', '16 (ladder); 32 (recipe, Challenge 2015); 64 (recipe, VTaC)'],
    ['Epochs', 'Not tuned; early stopping', 'Ladder: up to 60, patience 10. Recipe: up to 30, patience 8 (Challenge 2015); up to 20, patience 6 (VTaC)'],
    ['Optimiser', 'Not compared', 'Adam (ladder); AdamW with one-cycle schedule (recipe)'],
    ['Dropout and weight decay', 'Not tuned', 'Dropout 0.3; weight decay 1e-4 (ladder), 1e-2 (recipe)'],
    ['Loss', 'Recipe grid: focal loss and BCE with label smoothing 0.05', 'Focal loss, gamma = 2'],
    ['Augmentation', 'Recipe grid: off and on', 'On'],
    ['Architecture', 'Recipe grid: 6 architectures', 'Attention CNN-GRU (mean inner-validation AUC 0.936 for the winning configuration)'],
    ['Ensemble size', 'Ablation: 1, 2 and 3 members', '3'],
    ['Classical models', '30 random configurations per fold and algorithm, chosen on inner validation', 'Chosen separately in each fold'],
  ], { size: 20 }));
P(body('Every ladder variant used the same settings; none was tuned separately. The recipe grid was scored only on inner-validation AUC, and the test folds were not used to make any of these choices, with the one caveat discussed in Section 8.'));
P(h2('7.3 Performance metrics'));
P(body(`The positive class is "true alarm". Accuracy, precision, recall (sensitivity) and F1 follow the standard definitions, where TP, TN, FP and FN are true positives, true negatives, false positives and false negatives. Specificity is TN/(TN+FP), and AUC is the area under the ROC curve.`));
P(eq('eq_metrics.png', 6.2, 9));
P(body(`Because the task is binary, the headline F1 is the F1 of the true-alarm class at a threshold of 0.5. In the class-wise reports below, macro averages take the unweighted mean of the two classes and weighted averages weight each class by its support. The PhysioNet Challenge score, which counts a missed true alarm as five false alarms ${c('cinc')}, is`));
P(eq('eq_score.png', 2.6, 10));
P(body(`Model differences are tested with a paired record-level bootstrap with 5,000 resamples ${c('efron')} and with the Nadeau-Bengio corrected resampled t-test ${c('nadeau')}, with Holm correction ${c('holm')} over a comparison family listed in the configuration file before the final runs. A difference is called robust only when both tests agree.`));
P(eq('eq_nb.png', 2.0, 11));
P(h2('7.4 Tables and graphs'));
P(h3('Main comparison'));
P(body(`Tables ${tabN + 1} and ${tabN + 2} compare the models on the held-out folds and the held-out test split. Values are means over folds and seeds, with the standard deviation across them for F1 and AUC; these standard deviations are not confidence intervals, since folds share training data.`));
const cincRows = T.cinc.map(r => r.label);
P(...table('Challenge 2015, PPG cohort (592 records; 5-fold cross-validation, 10 seeds, 50 test folds).', [3300, 900, 1350, 1350, 1050, 1076],
  ['Model', 'Acc.', 'F1', 'AUC', 'Sens.', 'Spec.'],
  metricRows(T.cinc).map(r => (r[0].startsWith('Attention CNN-GRU') ? { cells: r, _bold: true } : r)), { size: 20, numFrom: 1 }));
P(...table('VTaC, official split (427 test events, 120 true alarms; 5 seeds, 10 for the single-signal models).', [3300, 900, 1350, 1350, 1050, 1076],
  ['Model', 'Acc.', 'F1', 'AUC', 'Sens.', 'Spec.'],
  metricRows(T.vtac).map(r => (r[0].startsWith('Attention CNN-GRU') ? { cells: r, _bold: true } : r)), { size: 20, numFrom: 1 }));
P(body('Ensembles of ten models reached accuracy 0.876, F1 0.827 and AUC 0.948 on Challenge 2015 (10 seeds), and 0.910, 0.839 and 0.955 on VTaC (5 seeds). Adding the recipe to the ensemble gave accuracy 0.878, F1 0.830 and AUC 0.950 on Challenge 2015 (5 seeds). The full leaderboard with bootstrap intervals is in the repository file docs/LEADERBOARD.md.'));
P(...fig('leaderboard.png', 6.4, 'F1 of the compared models on Challenge 2015 (left) and VTaC (right), threshold 0.5. The originally proposed cross-attention model (red) is near the bottom on both.'));
P(h3('Confusion matrices and class-wise reports'));
P(...fig('confusion.png', 5.0, 'Confusion matrices, as counts per test pass (Challenge 2015: pooled over 50 test folds and divided by 10 seeds, 592 records; VTaC: mean of 5 seeds, 427 events).'));
P(body('Confusion matrices were built from the true-positive, true-negative, false-positive and false-negative counts stored for every fold. The class-wise numbers below come from the pooled counts, so they differ in the third decimal from the fold-averaged values in the previous tables.'));
P(...table('Class-wise classification report from the pooled confusion counts (support is per seed).', [2600, 1300, 1000, 1000, 1000, 1126],
  ['Model and dataset', 'Class', 'Precision', 'Recall', 'F1', 'Support'],
  [
    ['Cross-attention, Challenge 2015', 'True alarm', '0.673', '0.731', '0.701', '223'],
    ['', 'False alarm', '0.829', '0.785', '0.806', '369'],
    { cells: ['', 'Macro avg. / weighted avg. F1', '0.751 / 0.770', '0.758 / 0.765', '0.753 / 0.766', '592'], _fill: 'F2F2F2' },
    ['CNN-GRU recipe, Challenge 2015', 'True alarm', '0.830', '0.826', '0.828', '223'],
    ['', 'False alarm', '0.895', '0.898', '0.896', '369'],
    { cells: ['', 'Macro avg. / weighted avg. F1', '0.863 / 0.871', '0.862 / 0.871', '0.862 / 0.871', '592'], _fill: 'F2F2F2' },
    ['Cross-attention, VTaC', 'True alarm', '0.689', '0.813', '0.746', '120'],
    ['', 'False alarm', '0.922', '0.857', '0.888', '307'],
    { cells: ['', 'Macro avg. / weighted avg. F1', '0.805 / 0.856', '0.835 / 0.844', '0.817 / 0.848', '427'], _fill: 'F2F2F2' },
    ['CNN-GRU recipe, VTaC', 'True alarm', '0.793', '0.933', '0.858', '120'],
    ['', 'False alarm', '0.972', '0.905', '0.937', '307'],
    { cells: ['', 'Macro avg. / weighted avg. F1', '0.883 / 0.922', '0.919 / 0.913', '0.897 / 0.915', '427'], _fill: 'F2F2F2' },
  ], { size: 20, numFrom: 2 }));
P(h3('Training behaviour'));
P(body('Per-epoch training and validation curves were not archived by the training scripts: each run stored the epoch of the best validation score, the validation AUC at that epoch and the final test metrics, and the model checkpoints were not kept. Plots of accuracy and loss against epoch can therefore not be shown without re-running the training, and none has been drawn here. What the stored records do show is where early stopping ended training.'));
P(...fig('best_epoch.png', 6.0, 'Epoch at which each ensemble member of the CNN-GRU recipe reached its best inner-validation AUC (EMA weights). Dashed line: epoch cap.'));
P(body('On Challenge 2015 the best epoch is spread widely between 4 and 29 with a median region around 15, so most members stopped well before the 30-epoch cap. On VTaC all 15 members peaked between epoch 14 and 19 of a 20-epoch cap, which suggests those runs were still improving slowly when the cap arrived and might gain a little from more epochs.'));
P(h3('Effect of the validation leak'));
P(...fig('leakage.png', 4.2, 'F1 on Challenge 2015 before and after the validation leak was removed. "Original" scores selected the best epoch on the fold that was then reported.'));
P(body('The first version of the training script chose the best epoch and the stopping point using the same fold it then reported. Fixing that, with a separate inner validation set and 10 instead of 3 seeds, lowered every F1: PPG-only 0.660 to 0.602, ECG-only 0.736 to 0.642, concatenation 0.750 to 0.693 and cross-attention 0.774 to 0.699. The cross-attention lead over concatenation shrank from +0.024 to +0.006. The original numbers are archived in the repository and are not used in any other table.'));
P(h3('Representative predictions'));
P(body('Two persistent errors of the cross-attention model are shown below, with the ECG (top) and PPG (bottom) windows, the detected beats and the model outputs. The text above each figure is produced by the error-analysis script; the note under the plot is an automatic heuristic, not a clinician annotation.'));
P(...fig('../../runs/challenge2015_ppg/error_taxonomy/figures/case_FN_cross_attention_1_v597l.png', 6.0, 'A missed true alarm (false negative). Record v597l is a true ventricular tachycardia alarm. The cross-attention model gives it a true-alarm probability of 0.24, and the gradient-boosting and ResNet1D models also miss it (0.21 and 0.37). The heuristic note says the PPG looks non-periodic and the beat detectors disagree (ECG 78 bpm, PPG 119 bpm).'));
P(...fig('../../runs/challenge2015_ppg/error_taxonomy/figures/case_FP_cross_attention_1_v160s.png', 6.0, 'A false alarm accepted as true (false positive). Record v160s is a false ventricular tachycardia alarm that cross-attention scores 0.72, and the gradient-boosting and ResNet1D models also score it above 0.5 (0.78 and 0.69). The ECG shows a fast, regular run of 29 detected beats (182 bpm) and the PPG has an irregular shape (157 bpm).'));
P(body('Both examples make the same point from opposite sides. In the first, the ECG alone looks like a calm rhythm and the noisy pulse is misleading; in the second, a plausible-looking fast ECG run is enough to convince the model. Neither case contains a clean disagreement between the two signals of the kind the original design hoped to exploit.'));

// ---------- 8 Discussion ----------
P(h1('8. Results Discussion'));
P(body(`The first thing the numbers say is that the proposed model did not win. On Challenge 2015 the cross-attention network scores F1 0.699, below every feature-based model and below ResNet1D, TCN and InceptionTime in Table 7. Only concatenation and the two single-signal models score lower. Against concatenation, which has the same encoders and no attention, it gains 0.006 F1, with a 95% bootstrap interval from -0.015 to +0.028 ${c('efron')}. Four attention variants (bidirectional, joint transformer, gated, and the plain one) cannot be told apart. Fusion itself does help: on VTaC, ECG-only reaches F1 0.709 and PPG-only 0.566 against 0.744 for concatenation. It is the attention part that adds nothing measurable.`));
P(body('The strongest results came from other routes. XGBoost on engineered features reached F1 0.818 and AUC 0.939 on Challenge 2015, and the attention CNN-GRU recipe reached F1 0.828 there and 0.858 (AUC 0.966) on VTaC. The ten-model ensemble’s F1 lead over the best classical model on Challenge 2015 (+0.008, interval -0.007 to +0.022) is inside the noise, while its lead over cross-attention (+0.128) is not. On VTaC the top three models are within the bootstrap interval of each other, but all three are well above cross-attention (+0.11 F1 for the recipe).'));
P(body(`The confusion matrices show where the errors sit. On VTaC the recipe misses 8 of 120 true alarms per seed on average (recall 0.933) and raises about 29 false positives among 307 false alarms (specificity 0.905); cross-attention misses about 22 true alarms and raises 44 false ones. On Challenge 2015 the recipe misses about 39 of 223 true alarms, against 60 for cross-attention. The per-type breakdown of cross-attention explains part of this. Tachycardia has an F1 of 0.92 but a specificity of only 0.42, because 106 of its 111 alarms are true and the model rarely says no; VT has an F1 of 0.49; asystole (14 true alarms) and flutter/fibrillation (5) are too small to judge. In the error analysis, VT dominates the persistent misses (18 of 29 for the gradient-boosting model, although VT is 26% of true alarms).`));
P(body('Overfitting is hard to read without training curves, so only indirect evidence is available. For the recipe, the mean inner-validation AUC of the chosen members was 0.931 on Challenge 2015 and 0.965 on VTaC, against test AUCs of 0.932 and 0.966, so there is no large validation-to-test gap. The validation numbers are maxima over epochs and therefore slightly optimistic, and a member is compared here with an ensemble, so this is a consistency check and not proof. For the ladder models the evidence points the other way: their test scores are modest and the validation sets are small, so we cannot separate underfitting from a weak use of the second signal. Fifteen to twenty-five true alarms per Challenge 2015 validation set also make epoch selection noisy.'));
P(body('The comparison is fair on splits, seeds and parameter count, but not on training. The ladder variants use Adam, F1-based early stopping and a batch of 16, while the recipe uses augmentation, AdamW, weight averaging and ensembling. The ablation shows how much that matters: removing weight averaging costs 0.008 F1, a single member with raw weights and no test-time augmentation loses 0.035, and ensembling three members instead of one adds about 0.012. Part of the gap between the recipe and the baseline CNNs is therefore the trainer and not the architecture. A shuffled-label control on one fold gave an AUC of 0.57, which is chance level, and the split check confirmed that fitting, validation and test records were disjoint, so no leakage was found. The recipe configuration was selected on inner-validation scores from Challenge 2015 folds whose records are test records in other seeds, so a small optimistic bias is possible on that dataset. The VTaC result, where the same recipe was applied without re-tuning, is the cleaner check.'));
P(body(`Why does the fusion network not use timing? Replacing the pulse with another patient’s pulse of the same label changed AUC by only 0.002 for cross-attention and 0.015 for concatenation, even though that pulse has no timing relation to the ECG. Shifting the pulse by up to two seconds changed AUC by at most 0.005 for the CNN fusion models. An opposite-label pulse, in contrast, dropped cross-attention AUC from 0.842 to 0.535. The network reads the pulse as a patient-independent cue (is there a regular, well-formed pulse or not), not as an answer to "did this beat produce a pulse". The attention weights agree: the lag they imply has a Spearman correlation of -0.039 with the measured pulse transit time (permutation p = 0.39), and only 5.3% of the attention mass falls in the physiological lag range, against 5.6% for uniform attention. The pulse-transit-time argument that motivated the design is not supported by these data. Occlusion-based explanations were faithful in deletion and insertion tests while attention weights were barely better than a random ordering, so attention maps should not be presented as explanations.`));
P(body('Robustness tells a similar story. With real noise on the ECG and synthetic motion on the PPG at 0 dB signal-to-noise ratio, ResNet1D held its AUC at 0.844 (0.907 when clean), while cross-attention fell from 0.856 to 0.696 and concatenation from 0.833 to 0.720. Zeroing the PPG at test time cost cross-attention 0.12 AUC and ResNet1D 0.27. Transfer between datasets is lopsided: models trained on Challenge 2015 scored F1 of only 0.56 to 0.60 on VTaC when they used the ECG (PPG-only was near chance at 0.432), but models trained on VTaC reached F1 0.773 and AUC 0.927 on the VT alarms of Challenge 2015 with ResNet1D. More data helped more than a cleverer model, and numbers from a single dataset overstate what a model would do elsewhere.'));
P(body('Strengths of the work are the controlled protocol, a leak that was found and corrected, and perturbation tests that probe what the networks use. Its limits are real. The datasets are small and the fold-to-fold F1 standard deviation is about 0.05. The engineered features and, on Challenge 2015, the alarm type itself carry the alarm definitions, which is built-in knowledge and not learned physiology; the alarm-type ablation was not run. The reproduction of the published attention CNN-RNN is ours and had only a two-seed sanity run outside the recipe. There was no prospective validation, no demographic analysis of optical PPG, and a safety layer (calibration and a certified suppression threshold) was run only for the concatenation model, where validation-chosen thresholds aimed at 95 to 100% sensitivity realised about 92 to 97% on test folds, so sensitivity targets should not be assumed to transfer.'));

// ---------- 9 Conclusion ----------
P(h1('9. Conclusion and Future Scope'));
P(body('The project set out to show that cross-attention between ECG and pulse verifies ICU arrhythmia alarms better than simpler models. After a validation leak was removed it did not. The cross-attention network reached F1 0.699 and AUC 0.841 on Challenge 2015 and F1 0.746 on VTaC, matching naive concatenation and trailing feature-based models, ResNet1D and the attention CNN-GRU recipe, which reached F1 0.828 on Challenge 2015 and F1 0.858 with accuracy 0.913 and AUC 0.966 on the VTaC test split. The objectives were met in the sense that all five were carried out and reported, but the central hypothesis was not confirmed.'));
P(body('The main contribution is the evidence behind that result: matched-size comparisons, paired statistics, counterfactual tests showing that the networks use signal content and not ECG-to-pulse timing, and an explicit record of the leak that had inflated the first numbers by 0.075 F1. The principal limitations are small datasets, no prospective validation, weak transfer from Challenge 2015 to VTaC and training curves that were not archived.'));
P(body('Realistic next steps are to pretrain on larger unlabelled pulse and ECG corpora (PulseDB, VitalDB and MIMIC-III were not accessible), to test leave-one-site-out and leave-one-device-out, to try a window that keeps the pre-alarm 10 s and adds post-alarm seconds, to rerun key models while logging per-epoch curves, to run the alarm-type ablation, and eventually to evaluate prospectively with clinician review. Any deployment would need the model to assist, not replace, clinical judgement.'));

// ---------- 10 References (placeholder, filled after) ----------
const REF_INDEX = S.length;

// ---------- Annexures builder (after refs index so cites are numbered in order) ----------
const AN = [];
const A = (...a) => AN.push(...a.flat());
A(new Paragraph({ children: [new PageBreak()] }));
A(h1('Annexure A: Code Repository'));
A(body('Repository: https://github.com/VeerbhadraMahant/ecg-ppg-pipeline'));
A(body('The README describes the pipeline and the project layout. Python dependencies are listed in requirements.txt (numpy, scipy, pandas, scikit-learn, matplotlib, torch, wfdb, pyyaml, tqdm, xgboost, lightgbm, PyWavelets and the Jupyter tools, among others). The steps below reproduce the main results; the datasets are downloaded separately from PhysioNet and Zenodo and are not stored in the repository.'));
A(...bullets([
  'Create an environment and install the packages: python -m venv .venv, activate it, then pip install -r requirements.txt.',
  'Download the data: python scripts/download_data.py --dataset challenge2015 and python scripts/download_vtac.py --workers 12.',
  'Preprocess: python -m src.data.preprocess, then python scripts/build_vtac_windows.py.',
  'Match parameter counts: python scripts/match_params.py.',
  'Train the baseline ladder with 10 seeds: python -m src.train --variant all --run-name challenge2015_ppg; for VTaC use --protocol official with the VTaC windows file.',
  'Classical models: python -m src.classical --dataset cinc --run-name classical_cinc.',
  'Evaluate and tabulate: python -m src.evaluate --run-name challenge2015_ppg and python scripts/final_leaderboard.py.',
  'Unit tests: python -m pytest tests.',
  'Regenerate the figures of this report from the stored results: python report/make_report_figures.py; rebuild this document with node report/build_report.js.']));
A(h1('Annexure B: Additional Results'));
A(h2('B.1 Paired comparisons'));
A(...fig('forest.png', 4.6, 'Paired F1 differences (challenger minus baseline) with 95% bootstrap intervals on Challenge 2015, from runs/challenge2015_ppg/significance.csv.'));
A(...table('Paired F1 differences on Challenge 2015 (50 seed-by-fold pairs); Holm-adjusted p-values.', [1900, 1900, 900, 2100, 1100, 1126],
  ['Baseline', 'Challenger', 'dF1', '95% CI', 'p (N-B)', 'p (boot.)'],
  [
    ['Concatenation', 'Cross-attention', '+0.006', '[-0.015, +0.028]', '1.00', '1.00'],
    ['ECG-only', 'Cross-attention', '+0.058', '[+0.028, +0.088]', '1.00', '0.003'],
    ['PPG-only', 'Cross-attention', '+0.094', '[+0.058, +0.129]', '0.077', '< 0.001'],
    ['Cross-attention', 'Bidirectional', '-0.008', '[-0.023, +0.008]', '1.00', '1.00'],
    ['Cross-attention', 'Joint transformer', '-0.006', '[-0.025, +0.013]', '1.00', '1.00'],
    ['Cross-attention', 'Gated', '-0.002', '[-0.015, +0.012]', '1.00', '1.00'],
    ['Cross-attention', 'Bottleneck', '-0.047', '[-0.070, -0.026]', '1.00', '< 0.001'],
    ['ECG-only', 'ResNet1D', '+0.118', '[+0.089, +0.148]', '0.052', '< 0.001'],
    ['ResNet1D', 'Cross-attention', '-0.060', '[-0.087, -0.035]', '0.95', '< 0.001'],
    ['TCN', 'Cross-attention', '-0.052', '[-0.080, -0.023]', '1.00', '< 0.001'],
    ['Gradient boosting (v1)', 'Cross-attention', '-0.099', '[-0.133, -0.066]', '0.049', '< 0.001'],
  ], { size: 20, numFrom: 2 }));
A(body('Only the comparison against gradient boosting satisfies the rule that both tests must agree after correction; the others are described as suggestive or not different.'));
A(h2('B.2 Perturbation, noise and transfer experiments'));
A(...fig('counterfactual.png', 4.6, 'Test AUC when the PPG is perturbed and the ECG is intact (seed-0 checkpoints, five test folds).'));
A(...fig('noise.png', 4.6, 'AUC versus injected noise level, mean over noise types (real Noise Stress Test noise on the ECG, synthetic motion artefact on the PPG).'));
A(...fig('transfer.png', 4.6, 'Cross-dataset transfer measured by F1.'));
A(...fig('attention_vs_ptt.png', 5.6, 'Attention-implied lag versus measured pulse transit time (left) and the two lag distributions (right).'));
A(h2('B.3 Other results'));
A(body(`AF screening on MIMIC PERform AF (35 subjects, subject-wise cross-validation, low power): a gradient-boosting model on RR-irregularity features reached subject-level AUC 1.00, significantly better than the best neural model (0.93); fusion beat a PPG-only network by +0.28 subject AUC; distilling a PPG-only student from the fusion teacher gave no benefit. Foundation model: a frozen PaPaGei linear probe reached F1 0.58 on Challenge 2015, and the best PaPaGei fusion model was about 0.04 F1 below scratch cross-attention ${c('papagei')}. Label reliability: 7.7% of VTaC events have split annotator votes, and accuracy on them is much lower (cross-attention 0.704 against 0.854 on unanimous events). Safety layer, concatenation model: a 10% miss tolerance at 90% confidence could be certified in 42 of 50 folds, suppressing 22.5% of false alarms while missing 2.9% of true alarms, with calibration reducing the expected calibration error from 0.167 to 0.121 (temperature scaling ${c('guo')}, exact Clopper-Pearson bound ${c('clopper')}).`));
A(h1('Annexure C: Implementation Evidence'));
A(body('No end-user application was built. The project includes a static results dashboard (site/index.html) and a replay page (site/replay.html) that display the stored leaderboards. The screenshot below shows the dashboard header and the Challenge 2015 leaderboard. One caveat: the dashboard labels the best VTaC model "attn CNN-RNN + features", but the recipe code only uses the hand-crafted features for the "_feat" architectures, and the winning architecture is the plain attention CNN-GRU without them, which is also consistent with its parameter count (258,857).'));
A(...fig('dashboard.png', 5.4, 'Screenshot of the results dashboard (site/index.html).'));
A(h1('Annexure D: Team Contributions'));
A(body('The contribution of each member is to be written by the team before submission; it has been left blank here because it cannot be derived from the repository.'));
A(...table('Team contributions.', [2600, 6426], ['Member', 'Contribution (to be completed by the team)'],
  [['Veerbhadra Mahant', ''], ['Ved Jadhav', ''], ['Sharvil Patil', '']]));

// ---------- references ----------
const refParas = [h1('10. References')];
order.forEach((k, i) => refParas.push(new Paragraph({
  spacing: { after: 90, line: 276 }, alignment: AlignmentType.LEFT, indent: { left: 480, hanging: 480 },
  children: [new TextRun({ text: `[${i + 1}]\t`, font: FONT, size: 24 }), new TextRun({ text: REFS[k], font: FONT, size: 24 })],
  tabStops: [{ type: TabStopType.LEFT, position: 480 }] })));
const missing = Object.keys(REFS).filter(k => !order.includes(k));
if (missing.length) console.log('uncited refs (not listed):', missing.join(', '));

// ---------- assemble ----------
const children = [...S, ...refParas, ...AN];
const doc = new Document({
  creator: 'Mini-project team', title: 'ICU alarm verification mini-project report',
  styles: {
    default: { document: { run: { font: FONT, size: 28 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 32, bold: true, font: FONT }, paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 28, bold: true, font: FONT }, paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: 'bul', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    { reference: 'num', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    { reference: 'obj', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '(%1)', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 420 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 24 })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then(buf => { fs.writeFileSync(path.join(__dirname, 'ICU_Alarm_Verification_Report.docx'), buf); console.log('written', buf.length, 'refs', order.length, 'figs', figN, 'tables', tabN); });
