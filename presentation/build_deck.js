// Builds presentation/ReDeEP_presentation.pptx from the result files in data/.
//   node presentation/build_deck.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const DATA = path.join(__dirname, "..", "data");
const read = f => JSON.parse(fs.readFileSync(path.join(DATA, f), "utf8"));
const tryRead = f => (fs.existsSync(path.join(DATA, f)) ? read(f) : null);

const THEME = {
  name: "ReDeEP Teal",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B1B1B", lt1: "FFFFFF", dk2: "0F3D3E", lt2: "E6EFEE",
    accent1: "F2A541", accent2: "2E7D7A", accent3: "8C9796", accent4: "C8553D",
    accent5: "6BAF92", accent6: "C9D1D0", hlink: "2E7D7A", folHlink: "6B6B6B",
  },
};
const HEX = THEME.colors;

// ---------------------------------------------------------------- data
const backbones = [
  ["TinyLlama-1.1B", "results_tinyllama_1.1b.json"],
  ["Llama-3.2-1B", "results_llama3.2_1b.json"],
  ["Llama-3.2-3B (4-bit)", "results_llama3.2_3b_4bit.json"],
  ["Llama-2-7B (4-bit)", "results_llama2_7b_4bit.json"],
].map(([name, f]) => ({ name, r: read(f) }));
const l2 = backbones[3].r;
const robT = read("robustness_tinyllama_1.1b.json").conditions;
const rob7 = read("robustness_llama2_7b_4bit.json").conditions;
const numT = read("robustness_numeric_subset_tinyllama_1.1b.json").response_numbers_contradicted;
const num7 = read("robustness_numeric_subset_llama2_7b_4bit.json").response_numbers_contradicted;
const SPLIT = "same_test_split_as_llama2_7b_4bit";
const judges = [
  ["GPT: gpt-oss-120b", "results_judge_groq_openai_gpt-oss-120b.json"],
  ["Gemini 3.5 Flash-Lite", "results_judge_gemini_gemini-3.5-flash-lite.json"],
  ["Llama-3.2-3B judge", "results_judge_llama_unsloth_Llama-3.2-3B-Instruct.json"],
].map(([name, f]) => ({ name, r: tryRead(f) })).filter(j => j.r && j.r[SPLIT]);
const f3 = v => v.toFixed(3);

// ---------------------------------------------------------------- deck
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.author = "Gourilakshmi S, Adithya K";
pres.title = "ReDeEP: reproduction and robustness study";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
if (process.env.MAXSLIDE) { const orig = pres.addSlide.bind(pres); let n = 0; const dummy = { addText(){return dummy;}, addShape(){return dummy;}, addChart(){return dummy;}, addNotes(){return dummy;}, addImage(){return dummy;} }; pres.addSlide = o => (++n <= +process.env.MAXSLIDE ? orig(o) : dummy); }
if (process.env.NOCHARTS) { const orig = pres.addSlide.bind(pres); pres.addSlide = o => { const sl = orig(o); sl.addChart = () => sl; return sl; }; }

pres.defineSlideMaster({
  title: "Title dark",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.45, w: 8.8, h: 1.5, fontSize: 38, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 3.15, w: 8.8, h: 1.2, fontSize: 18, color: C.background2, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "Content",
  background: { color: C.background1 },
  margin: [0.5, 0.5, 0.5, 0.5],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9, h: 0.75, fontSize: 24, bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "ReDeEP reproduction and robustness study", options: { x: 0.5, y: 5.2, w: 6, h: 0.3, fontSize: 10, color: C.accent3, margin: 0 } } },
  ],
  slideNumber: { x: 9.0, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: C.accent3, align: "right" },
});

const T = (s, text, o) => s.addText(text, Object.assign({ isTextBox: true, margin: 0, fontSize: 15, color: C.text1, valign: "top" }, o));
const card = (s, x, y, w, h, fill) => s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill || C.background2 }, line: { type: "none" } });
const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", titleFontFace: "+mn-lt", catAxisLabelColor: "4A4A4A", valAxisLabelColor: "4A4A4A", dataLabelColor: "1B1B1B", catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, dataLabelFontSize: 11, legendFontSize: 12 };

// 1 Title ------------------------------------------------------------
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "Title dark", sectionTitle: "Introduction" });
s.addText("ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via Mechanistic Interpretability", { placeholder: "title", fontSize: 28 });
s.addText("A reproduction and robustness study  |  IOE-414 Deep Learning, IIIT Kottayam\nGourilakshmi S (2023BCS0021) & Adithya K (2023BCS0051)\ngithub.com/ad1thyaaa/ReDeEP-Reproduction-Robustness", { placeholder: "body", fontSize: 16 });
s.addNotes("This project reproduces ReDeEP, a method that detects hallucinations in retrieval-augmented generation by reading a model's internals, and extends it with a robustness test and a comparison against GPT and Gemini judges.");

// 2 Problem ------------------------------------------------------------
s = pres.addSlide({ masterName: "Content", sectionTitle: "Introduction" });
s.addText("RAG models still make things up, even with the answer in front of them", { placeholder: "title" });
card(s, 0.5, 1.3, 4.3, 3.6);
T(s, "Retrieved passage", { x: 0.75, y: 1.5, w: 3.8, h: 0.35, bold: true, color: C.text2 });
T(s, "“The bridge opened in 1932 and carries 160,000 vehicles a day.”", { x: 0.75, y: 1.9, w: 3.8, h: 0.9, italic: true });
T(s, "Model answer", { x: 0.75, y: 2.9, w: 3.8, h: 0.35, bold: true, color: C.text2 });
T(s, "“The bridge opened in 1932 and carries about 200,000 vehicles a day.”", { x: 0.75, y: 3.3, w: 3.8, h: 0.9, italic: true });
T(s, "Fluent and confident, but unsupported.", { x: 0.75, y: 4.3, w: 3.8, h: 0.4, color: C.accent4, bold: true });
T(s, [
  { text: "Output-only signals miss it", options: { bold: true, breakLine: true } },
  { text: "Perplexity and entropy measure confidence, not faithfulness to the passage.", options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "ReDeEP looks inside instead", options: { bold: true, breakLine: true } },
  { text: "It asks whether the model used the passage (attention heads) or its own memory (FFN layers)." },
], { x: 5.2, y: 1.4, w: 4.3, h: 3.4 });
s.addNotes("Example is illustrative. The core problem: a RAG answer can be confident and fluent yet unsupported by its context, so confidence-based detectors are weak.");

// 3 How ReDeEP works ----------------------------------------------------
pres.addSection({ title: "Method" });
s = pres.addSlide({ masterName: "Content", sectionTitle: "Method" });
s.addText("ReDeEP scores two internal signals and combines them", { placeholder: "title" });
const box = (x, y, w, h, head, body, fill, headColor) => {
  card(s, x, y, w, h, fill);
  T(s, head, { x: x + 0.15, y: y + 0.12, w: w - 0.3, h: 0.35, bold: true, fontSize: 15, color: headColor || C.text2 });
  T(s, body, { x: x + 0.15, y: y + 0.5, w: w - 0.3, h: h - 0.6, fontSize: 13 });
};
box(0.5, 1.75, 2.1, 1.6, "Prompt + response", "One forward pass through a LLaMA model");
box(3.15, 1.25, 2.9, 1.25, "Copying heads → ECS", "Does the response build on what the heads read in the passage?");
box(3.15, 2.85, 2.9, 1.25, "Knowledge FFNs → PKS", "How much do FFN layers inject the model's own memory?");
box(6.6, 1.75, 2.9, 1.6, "Hallucination score", "Logistic regression: high PKS and low ECS = likely hallucination", C.accent1, C.text1);
// a negative h makes PowerPoint refuse the file: keep w/h positive, flip for upward arrows
const arrow = (x1, y1, x2, y2) => s.addShape(pres.shapes.LINE, { x: x1, y: Math.min(y1, y2), w: x2 - x1, h: Math.abs(y2 - y1), flipV: y2 < y1, line: { color: HEX.accent3, width: 1.5, endArrowType: "triangle" } });
arrow(2.6, 2.35, 3.15, 1.9); arrow(2.6, 2.75, 3.15, 3.45); arrow(6.05, 1.9, 6.6, 2.35); arrow(6.05, 3.45, 6.6, 2.75);
T(s, "ECS: cosine similarity between a response token and the passage tokens a head attends to most.  PKS: Jensen-Shannon divergence of LogitLens predictions before vs. after each FFN.", { x: 0.5, y: 4.35, w: 9, h: 0.7, fontSize: 12, color: C.accent3 });
s.addNotes("Top 8 copying heads are chosen from the OV circuit; top 5 knowledge FFN layers by correlation with the label on the training fold. Baselines: perplexity and length-normalised entropy from the same forward pass.");

// 3b What the scores look like ------------------------------------------
const SS = read("score_summary_llama2_7b.json");
s = pres.addSlide({ masterName: "Content", sectionTitle: "Method" });
s.addText("What ECS, PKS and the final score look like on real answers", { placeholder: "title" });
const sh = o => Object.assign({ bold: true, color: "FFFFFF", fill: { color: HEX.dk2 }, align: "center", fontSize: 13 }, o || {});
const sc = (t, o) => ({ text: t, options: Object.assign({ align: "center", fontSize: 13 }, o || {}) });
s.addTable([
  [{ text: "Llama-2-7B, 415 answers", options: sh({ align: "left" }) }, { text: "Truthful (avg)", options: sh() }, { text: "Hallucinated (avg)", options: sh() }, { text: "AUC alone", options: sh() }],
  [sc("ECS  (uses passage, higher = good)", { align: "left", bold: true }), sc(SS.ecs_t.toFixed(3)), sc(SS.ecs_h.toFixed(3), { color: C.accent4, bold: true }), sc(SS.ecs_auc.toFixed(3))],
  [sc("PKS  (memory injected, higher = bad)", { align: "left", bold: true }), sc(SS.pks_t.toFixed(3)), sc(SS.pks_h.toFixed(3), { color: C.accent4, bold: true }), sc(SS.pks_auc.toFixed(3))],
  [sc("ReDeEP score  (0 = truthful, 1 = halluc.)", { align: "left", bold: true }), sc(SS.score_t.toFixed(2)), sc(SS.score_h.toFixed(2), { color: C.accent4, bold: true }), sc(SS.comb_auc.toFixed(3), { bold: true })],
], { x: 0.5, y: 1.25, w: 9.0, colW: [3.9, 1.6, 1.8, 1.7], fontFace: "Calibri", border: { type: "solid", pt: 0.5, color: "D0D7D6" }, fill: { color: "FFFFFF" }, rowH: 0.42 });
const ex = (x, fill, head, q, vals, note) => {
  card(s, x, 3.15, 4.4, 1.85, fill);
  T(s, head, { x: x + 0.15, y: 3.25, w: 4.1, h: 0.3, bold: true, fontSize: 13, color: C.text2 });
  T(s, q, { x: x + 0.15, y: 3.57, w: 4.1, h: 0.55, fontSize: 11.5, italic: true });
  T(s, vals, { x: x + 0.15, y: 4.12, w: 4.1, h: 0.3, fontSize: 13, bold: true, color: C.accent2 });
  T(s, note, { x: x + 0.15, y: 4.45, w: 4.1, h: 0.5, fontSize: 11 });
};
ex(0.5, C.background2, "Truthful answer (QA)", "“Meaning of the name Teresa”: lists the meanings given in the passages.",
   "ECS 0.371   PKS 0.059   → score 0.17", "High passage use, little memory injection: low score.");
ex(5.1, C.background2, "Hallucinated answer (QA)", "“Average temperature in Tallahassee”: invents a calculation “79°F – 56°F = 23°F” not in the passage.",
   "ECS 0.297   PKS 0.101   → score 0.65", "Less passage use, more memory injection: high score.");
s.addNotes("Averages and examples use heads/layers fitted on all 415 answers, for illustration; the AUC column for the final score is out-of-fold (5-fold CV). ECS is a cosine similarity (range about 0.1-0.5 here); PKS is a Jensen-Shannon divergence averaged over the 5 knowledge FFN layers.");

// 3c How AUC works --------------------------------------------------------
s = pres.addSlide({ masterName: "Content", sectionTitle: "Method" });
s.addText("How we measure a detector: AUC (area under the ROC curve)", { placeholder: "title" });
s.addImage({ path: path.join(__dirname, "..", "report_docx", "figures", "roc.png"), x: 0.5, y: 1.2, w: 4.1, h: 3.33 });
T(s, "ROC curves, Llama-2-7B and smaller models: ReDeEP (colour) above perplexity / entropy (grey).", { x: 0.5, y: 4.6, w: 4.1, h: 0.45, fontSize: 10.5, color: C.accent3 });
card(s, 4.9, 1.25, 4.6, 3.75);
T(s, [
  { text: "AUC = chance a random hallucinated answer gets a higher score than a random truthful one", options: { bold: true, breakLine: true } },
  { text: " ", options: { breakLine: true, fontSize: 6 } },
  { text: "1.0 = perfect    0.5 = coin flip", options: { breakLine: true } },
  { text: `Ours (Llama-2-7B): ${f3(l2.cv.AUC[0])} → right about ${Math.round(100 * l2.cv.AUC[0])}% of the time`, options: { breakLine: true } },
  { text: " ", options: { breakLine: true, fontSize: 6 } },
  { text: "Toy example", options: { bold: true, breakLine: true } },
  { text: "Hallucinated scores 0.9, 0.6, 0.3; truthful 0.7, 0.4, 0.2. Of the 9 (H, T) pairs the hallucinated one is higher in 7 → AUC = 7/9 = 0.78.", options: { breakLine: true } },
  { text: " ", options: { breakLine: true, fontSize: 6 } },
  { text: "Why AUC: needs no cut-off threshold, so it compares detectors fairly; it is the paper's main metric.", options: {} },
], { x: 5.1, y: 1.4, w: 4.25, h: 3.5, fontSize: 13 });
s.addNotes("The ROC curve plots the share of hallucinations caught against the share of truthful answers wrongly flagged as the threshold slides from 1 to 0; AUC is the area under it. We report mean AUC over 5-fold cross-validation.");

// 4 Setup ----------------------------------------------------------------
s = pres.addSlide({ masterName: "Content", sectionTitle: "Method" });
s.addText("Four LLaMA models, one 6 GB laptop GPU, the paper's test set", { placeholder: "title" });
const stats = [["415", "RAGTruth test responses\n(212 hallucinated)"], ["4", "LLaMA backbones,\n1.1B to 7B"], ["6 GB", "RTX 4050 VRAM;\n3B and 7B in 4-bit"], ["5-fold", "stratified\ncross-validation"]];
stats.forEach(([big, small], i) => {
  const x = 0.5 + i * 2.3;
  card(s, x, 1.3, 2.05, 1.9);
  T(s, big, { x: x + 0.15, y: 1.45, w: 1.75, h: 0.8, fontSize: 36, bold: true, color: C.accent2, fontFace: THEME.headFontFace });
  T(s, small, { x: x + 0.15, y: 2.3, w: 1.75, h: 0.8, fontSize: 13 });
});
T(s, [
  { text: "Backbones: ", options: { bold: true } },
  { text: "TinyLlama-1.1B-Chat, Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Llama-2-7B-chat (the paper's model).", options: { breakLine: true } },
  { text: "Responses: ", options: { bold: true } },
  { text: "all generated by llama-2-7b-chat, as in the paper. Copying heads and FFN layers are re-selected inside each training fold." },
], { x: 0.5, y: 3.5, w: 9, h: 1.4, fontSize: 14 });
s.addNotes("Engineering: forward hooks keep only top-k attention positions per layer instead of full attention maps, which lets 7B fit in 6 GB. Verified identical to the original implementation to within 6e-8.");

// 5 Headline reproduction ----------------------------------------------
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "Content", sectionTitle: "Results" });
s.addText("The reproduction matches the paper on its own model", { placeholder: "title" });
card(s, 0.5, 1.3, 4.2, 3.4, C.text2);
T(s, f3(l2.cv.AUC[0]), { x: 0.8, y: 1.6, w: 3.6, h: 1.3, fontSize: 66, bold: true, color: C.accent1, fontFace: THEME.headFontFace });
T(s, "Our AUC, Llama-2-7B-chat\n(4-bit, 5-fold CV)", { x: 0.8, y: 3.0, w: 3.6, h: 0.9, fontSize: 15, color: C.background1 });
card(s, 5.1, 1.3, 4.4, 3.4);
T(s, "0.7325", { x: 5.4, y: 1.6, w: 3.8, h: 1.3, fontSize: 66, bold: true, color: C.accent3, fontFace: THEME.headFontFace });
T(s, "Paper's AUC, Llama-2-7B-chat\n(Sun et al., ICLR 2025)", { x: 5.4, y: 3.0, w: 3.8, h: 0.9, fontSize: 15 });
T(s, `Difference ${f3(l2.cv.AUC[0] - 0.7325)}, well inside the fold-to-fold spread of ±${f3(l2.cv.AUC[1])}.`, { x: 0.5, y: 4.8, w: 9, h: 0.35, fontSize: 13, color: C.accent3 });
s.addNotes(`PCC ${f3(l2.cv.PCC[0])} vs paper 0.3979. Achieved on a laptop GPU with 4-bit weights.`);

// 6 Across backbones ----------------------------------------------------
s = pres.addSlide({ masterName: "Content", sectionTitle: "Results" });
s.addText("ReDeEP beats both baselines on every backbone", { placeholder: "title" });
s.addChart(pres.charts.BAR, [
  { name: "ReDeEP", labels: backbones.map(b => b.name), values: backbones.map(b => +b.r.cv.AUC[0].toFixed(3)) },
  { name: "Perplexity", labels: backbones.map(b => b.name), values: backbones.map(b => +b.r.cv.PPL_AUC[0].toFixed(3)) },
  { name: "LN-Entropy", labels: backbones.map(b => b.name), values: backbones.map(b => +b.r.cv.LNEnt_AUC[0].toFixed(3)) },
], Object.assign({
  x: 0.5, y: 1.15, w: 6.4, h: 3.9, barDir: "col", barGapWidthPct: 60,
  chartColors: [HEX.accent1, HEX.accent3, HEX.accent6],
  valAxisMinVal: 0.5, valAxisMaxVal: 0.8, valAxisMajorUnit: 0.05, valAxisLabelFormatCode: "0.00",
  valGridLine: { color: "E3E3E3", size: 0.5 }, catGridLine: { style: "none" },
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00",
  showLegend: true, legendPos: "t", showTitle: false,
  valAxisTitle: "AUC (0.50 = chance)", showValAxisTitle: true, valAxisTitleFontSize: 11, valAxisTitleColor: "4A4A4A", valAxisTitleFontFace: "+mn-lt",
}, chartText));
card(s, 7.2, 1.3, 2.3, 3.6);
T(s, [
  { text: "+0.06 to +0.11", options: { bold: true, fontSize: 22, color: C.accent2, breakLine: true } },
  { text: "AUC over the baselines", options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "Size barely matters", options: { bold: true, breakLine: true } },
  { text: "1.1B TinyLlama scores highest (0.771); 3B and 7B are within noise." },
], { x: 7.4, y: 1.5, w: 1.95, h: 3.3, fontSize: 13 });
s.addNotes("Fold standard deviations: ReDeEP 0.03-0.05, baselines 0.06-0.08. Differences between backbones under ~0.05 are within noise.");

// 7 Extension design ----------------------------------------------------
pres.addSection({ title: "Extension" });
s = pres.addSlide({ masterName: "Content", sectionTitle: "Extension" });
s.addText("Extension: keep the answer, corrupt the passage", { placeholder: "title" });
[["Accurate", "The real RAGTruth passage. The control.", C.background2],
 ["Irrelevant passage", "A passage from a different source on the same task type. Nothing in the answer is supported.", C.background2],
 ["Numbers corrupted", "Every number in the real passage scaled by 1.3–3×. Numbers the answer copied are now contradicted.", C.background2]]
  .forEach(([h, b, fill], i) => {
    const x = 0.5 + i * 3.1;
    card(s, x, 1.3, 2.8, 2.6, fill);
    s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: 1.5, w: 0.5, h: 0.5, fill: { color: i === 2 ? C.accent1 : C.accent2 }, line: { type: "none" } });
    T(s, String(i + 1), { x: x + 0.2, y: 1.5, w: 0.5, h: 0.5, align: "center", valign: "middle", bold: true, color: C.background1, fontSize: 16 });
    T(s, h, { x: x + 0.2, y: 2.15, w: 2.4, h: 0.4, bold: true, color: C.text2, fontSize: 16 });
    T(s, b, { x: x + 0.2, y: 2.6, w: 2.4, h: 1.2, fontSize: 13 });
  });
T(s, "A faithful detector should flag both corrupted conditions. 150 held-out responses per backbone; detector refit on the other 265.", { x: 0.5, y: 4.2, w: 9, h: 0.7, fontSize: 14 });
s.addNotes("Tested on TinyLlama-1.1B and Llama-2-7B. Paired Wilcoxon signed-rank test on the per-response score change.");

// 8 Extension result ----------------------------------------------------
s = pres.addSlide({ masterName: "Content", sectionTitle: "Extension" });
s.addText("It flags off-topic passages but misses contradicted numbers", { placeholder: "title" });
s.addChart(pres.charts.BAR, [
  { name: "Irrelevant passage", labels: ["TinyLlama-1.1B", "Llama-2-7B"], values: [robT.irrelevant.score_shift_vs_accurate, rob7.irrelevant.score_shift_vs_accurate].map(v => +v.toFixed(3)) },
  { name: "Numbers corrupted", labels: ["TinyLlama-1.1B", "Llama-2-7B"], values: [robT.contradictory.score_shift_vs_accurate, rob7.contradictory.score_shift_vs_accurate].map(v => +v.toFixed(3)) },
], Object.assign({
  x: 0.5, y: 1.15, w: 5.0, h: 3.9, barDir: "bar", barGapWidthPct: 50,
  chartColors: [HEX.accent3, HEX.accent1],
  valAxisMinVal: -0.05, valAxisMaxVal: 0.35, catAxisOrientation: "maxMin", catAxisLabelPos: "low", valAxisMajorUnit: 0.05, valAxisLabelFormatCode: "+0.00;-0.00;0",
  valGridLine: { color: "E3E3E3", size: 0.5 }, catGridLine: { style: "none" },
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "+0.000;-0.000;0",
  showLegend: true, legendPos: "t",
  valAxisTitle: "Change in hallucination score vs. accurate passage", showValAxisTitle: true, valAxisTitleFontSize: 11, valAxisTitleColor: "4A4A4A", valAxisTitleFontFace: "+mn-lt",
}, chartText));
card(s, 5.7, 1.3, 3.8, 3.6);
T(s, "Llama-2-7B, average signals", { x: 5.85, y: 1.42, w: 3.5, h: 0.3, bold: true, fontSize: 13, color: C.text2 });
const rh = o => Object.assign({ bold: true, color: "FFFFFF", fill: { color: HEX.dk2 }, align: "center", fontSize: 11 }, o || {});
const rc = (t, o) => ({ text: t, options: Object.assign({ align: "center", fontSize: 11 }, o || {}) });
s.addTable([
  [{ text: "Passage", options: rh({ align: "left" }) }, { text: "ECS", options: rh() }, { text: "PKS", options: rh() }, { text: "Score", options: rh() }],
  [rc("Accurate", { align: "left" }), rc(f3(rob7.accurate.E_A_mean)), rc(f3(rob7.accurate.P_F_mean)), rc(rob7.accurate.score_mean.toFixed(2))],
  [rc("Irrelevant", { align: "left" }), rc(f3(rob7.irrelevant.E_A_mean)), rc(f3(rob7.irrelevant.P_F_mean), { bold: true, color: C.accent4 }), rc(rob7.irrelevant.score_mean.toFixed(2), { bold: true, color: C.accent4 })],
  [rc("Numbers changed", { align: "left" }), rc(f3(rob7.contradictory.E_A_mean), { bold: true, color: C.accent2 }), rc(f3(rob7.contradictory.P_F_mean)), rc(rob7.contradictory.score_mean.toFixed(2))],
], { x: 5.85, y: 1.8, w: 3.5, colW: [1.55, 0.65, 0.65, 0.65], fontFace: "Calibri", border: { type: "solid", pt: 0.5, color: "C9D1D0" }, fill: { color: "FFFFFF" }, rowH: 0.3 });
T(s, [
  { text: "Irrelevant: ", options: { bold: true } },
  { text: "PKS rises (memory takes over), so the score jumps. Correct.", options: { breakLine: true } },
  { text: "Numbers changed: ", options: { bold: true } },
  { text: `ECS even rises and the score stays flat: it rose in only ${Math.round(100 * num7.frac_score_increased)}% of the ${num7.n} contradicted answers.` },
], { x: 5.85, y: 3.25, w: 3.5, h: 1.6, fontSize: 12 });
s.addNotes(`Llama-2-7B: score ${f3(rob7.contradictory.score_shift_vs_accurate)} (p=${rob7.contradictory.score_shift_wilcoxon_p.toExponential(1)}), ECS rises ${f3(rob7.accurate.E_A_mean)} to ${f3(rob7.contradictory.E_A_mean)}. TinyLlama: +${f3(robT.contradictory.score_shift_vs_accurate)}, p=${robT.contradictory.score_shift_wilcoxon_p.toFixed(2)}.`);

// 9 Interpretation ------------------------------------------------------
s = pres.addSlide({ masterName: "Content", sectionTitle: "Extension" });
s.addText("ReDeEP checks whether an answer is about the passage, not whether it agrees", { placeholder: "title" });
card(s, 0.5, 1.3, 4.3, 2.0);
T(s, "What ECS measures", { x: 0.75, y: 1.45, w: 3.8, h: 0.4, bold: true, color: C.text2, fontSize: 16 });
T(s, "Cosine similarity between hidden states: is the answer built from the same content the copying heads read?", { x: 0.75, y: 1.9, w: 3.8, h: 1.6, fontSize: 14 });
card(s, 5.2, 1.3, 4.3, 2.0);
T(s, "What it cannot see", { x: 5.45, y: 1.45, w: 3.8, h: 0.4, bold: true, color: C.text2, fontSize: 16 });
T(s, "A changed number likely barely moves the hidden state, so a contradicted fact still looks “grounded”. In Llama-2-7B, ECS even rose.", { x: 5.45, y: 1.9, w: 3.8, h: 1.6, fontSize: 14 });
card(s, 0.5, 3.65, 9, 1.0, C.text2);
T(s, "Implication: pair ReDeEP with a fact-level check (numbers, entities) before trusting it on knowledge conflicts.", { x: 0.75, y: 3.77, w: 8.5, h: 0.75, fontSize: 15, bold: true, color: C.background1, valign: "middle" });
s.addNotes("The paper scopes ReDeEP to RAG hallucination with accurate context; this test shows what happens just outside that scope.");

// 10 Judges ---------------------------------------------------------------
pres.addSection({ title: "Comparison" });
s = pres.addSlide({ masterName: "Content", sectionTitle: "Comparison" });
const l2single = l2.single_split;
const jrows = [{ name: "ReDeEP on Llama-2-7B", auc: l2single.AUC }].concat(judges.map(j => ({ name: j.name, auc: j.r[SPLIT].AUC })), [{ name: "Perplexity", auc: l2single.PPL_AUC }]);
const best = jrows.slice(1).reduce((a, b) => (b.auc > a.auc ? b : a));
s.addText(best.auc > l2single.AUC ? "Large API judges beat ReDeEP on accuracy; ReDeEP wins on cost" : "Reading the model's internals beats asking a model to judge", { placeholder: "title" });
s.addChart(pres.charts.BAR, [{ name: "AUC", labels: jrows.map(r => r.name), values: jrows.map(r => +r.auc.toFixed(3)) }], Object.assign({
  x: 0.5, y: 1.15, w: 6.1, h: 3.9, barDir: "bar", barGapWidthPct: 45,
  chartColors: [HEX.accent1].concat(jrows.slice(1).map(() => HEX.accent3)), invertedColors: [HEX.accent3],
  valAxisMinVal: 0.4, valAxisMaxVal: 0.9, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: "0.0",
  valGridLine: { color: "E3E3E3", size: 0.5 }, catGridLine: { style: "none" }, catAxisOrientation: "maxMin",
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", showLegend: false,
  valAxisTitle: `AUC on the same ${l2.n - Math.round(l2.n * 0.7)} held-out test responses`, showValAxisTitle: true, valAxisTitleFontSize: 11, valAxisTitleColor: "4A4A4A", valAxisTitleFontFace: "+mn-lt",
}, chartText));
card(s, 6.9, 1.3, 2.6, 3.6);
T(s, [
  { text: "Black-box judges ", options: { bold: true } },
  { text: "read the passage and answer, and state a hallucination probability (0–100).", options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "ReDeEP ", options: { bold: true } },
  { text: "needs no prompt and no API, only one forward pass of an open model." },
], { x: 7.1, y: 1.5, w: 2.25, h: 3.3, fontSize: 13 });
s.addNotes("GPT-family judge: OpenAI's open-weights gpt-oss-120b via Groq's free tier. Gemini: 3.5 Flash-Lite via the free tier. All judges and ReDeEP are scored on the same held-out split.");

// ---- official code & data: load results ------------------------------
const ev = f => tryRead(`eval_chunk_${f}.json`);
const tok = f => tryRead(`results_tok_${f}.json`);
const BIGM = [["Llama-2-7B", "llama2_7b"], ["Llama-2-13B", "llama2_13b"], ["Llama-3-8B", "llama3_8b"]];
const nc = e => (e ? e.nested_cv.AUC_mean.toFixed(3) : "–");
const tk = (t, ds) => (t === "llama2_7b" && ds === "ragtruth" ? l2.cv.AUC[0].toFixed(3) : (tok(`${t}_${ds}`) ? tok(`${t}_${ds}`).cv.AUC[0].toFixed(3) : "–"));
function aarfStats(tag) {
  const g = path.join(DATA, `aarf_${tag}.jsonl`), j = path.join(DATA, `aarf_judge_${tag}.jsonl`);
  if (!fs.existsSync(g)) return null;
  const rows = fs.readFileSync(g, "utf8").trim().split("\n").map(JSON.parse);
  let steps = 0, fire = 0;
  rows.forEach(r => (r.gate_trace || []).forEach(t => { steps++; if (t[2] > 0.6) fire++; }));
  const by = {};
  if (fs.existsSync(j)) fs.readFileSync(j, "utf8").trim().split("\n").filter(Boolean).map(JSON.parse)
    .forEach(x => { (by[x.id] = by[x.id] || {})[x.order] = x.aarf_won; });
  const vs = Object.values(by), both = vs.filter(v => v.aarf_first != null && v.baseline_first != null);
  return { n: rows.length, changed: rows.filter(r => r.baseline !== r.aarf).length, fire: steps ? 100 * fire / steps : 0,
           offA: vs.filter(v => v.aarf_first === true).length, offB: vs.filter(v => v.aarf_first === false).length,
           bothA: both.filter(v => v.aarf_first && v.baseline_first).length,
           bothB: both.filter(v => !v.aarf_first && !v.baseline_first).length };
}

// 11 Official code: results table ---------------------------------------
pres.addSection({ title: "Official code" });
s = pres.addSlide({ masterName: "Content", sectionTitle: "Official code" });
s.addText("With the authors' code and data, chunk level matches and 13B is strongest", { placeholder: "title" });
const hdr = o => Object.assign({ bold: true, color: "FFFFFF", fill: { color: HEX.dk2 }, align: "center", fontSize: 13 }, o || {});
const cellO = (t, o) => ({ text: t, options: Object.assign({ align: "center", fontSize: 13 }, o || {}) });
const trows = [[{ text: "Backbone", options: hdr({ align: "left" }) }, { text: "RAGTruth token", options: hdr() },
  { text: "RAGTruth chunk", options: hdr() }, { text: "Dolly token", options: hdr() }, { text: "Dolly chunk", options: hdr() }]];
BIGM.forEach(([n, t]) => trows.push([cellO(n, { align: "left", bold: true }), cellO(tk(t, "ragtruth")), cellO(nc(ev(`${t}_ragtruth`))),
  cellO(tk(t, "dolly")), cellO(nc(ev(`${t}_dolly`)))]));
s.addTable(trows, { x: 0.5, y: 1.3, w: 6.2, colW: [1.6, 1.15, 1.15, 1.15, 1.15], fontFace: "Calibri", border: { type: "solid", pt: 0.5, color: "D0D7D6" },
  fill: { color: "FFFFFF" }, rowH: 0.45 });
T(s, "AUC, leakage-free 5-fold CV. 13B and 8B ran in 4-bit on a free Kaggle T4 GPU, each on its own responses.", { x: 0.5, y: 3.35, w: 6.2, h: 0.6, fontSize: 12, color: C.accent3 });
const offLog = ev("official_llama2_7B_response_chunk");
card(s, 7.0, 1.3, 2.5, 3.6);
T(s, [
  { text: "Chunk level reproduced", options: { bold: true, breakLine: true } },
  { text: `Ours ${nc(ev("llama2_7b_ragtruth"))} vs. authors' own scores ${nc(offLog)}`, options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "Mixed on Llama-3-8B", options: { bold: true, breakLine: true } },
  { text: "Small margin on RAGTruth; LN-Entropy wins on Dolly (AC)." },
], { x: 7.2, y: 1.5, w: 2.15, h: 3.3, fontSize: 13 });
s.addNotes("Chunk-level ECS uses BGE embeddings of the most-attended prompt chunk, as in the official code; our features agree with the authors' logged values head by head in 148/160 spot checks.");

// 12 Official code: leakage audit ---------------------------------------
s = pres.addSlide({ masterName: "Content", sectionTitle: "Official code" });
s.addText("The official evaluation picks features on the test set, inflating small-data results", { placeholder: "title" });
const aud = [["Authors' logs, RAGTruth 7B", offLog], ["Ours, RAGTruth 7B", ev("llama2_7b_ragtruth")], ["Ours, Dolly 7B", ev("llama2_7b_dolly")],
             ["Ours, Dolly 13B", ev("llama2_13b_dolly")], ["Ours, Dolly 8B", ev("llama3_8b_dolly")]].filter(a => a[1]);
s.addChart(pres.charts.BAR, [
  { name: "Official protocol (selected on test)", labels: aud.map(a => a[0]), values: aud.map(a => +a[1].official_protocol.AUC.toFixed(3)) },
  { name: "Leakage-free nested CV", labels: aud.map(a => a[0]), values: aud.map(a => +a[1].nested_cv.AUC_mean.toFixed(3)) },
], Object.assign({
  x: 0.5, y: 1.15, w: 6.2, h: 3.9, barDir: "col", barGapWidthPct: 60, chartColors: [HEX.accent3, HEX.accent1],
  valAxisMinVal: 0.6, valAxisMaxVal: 0.85, valAxisMajorUnit: 0.05, valAxisLabelFormatCode: "0.00",
  valGridLine: { color: "E3E3E3", size: 0.5 }, catGridLine: { style: "none" },
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", showLegend: true, legendPos: "t",
}, chartText));
const dl = aud.filter(a => a[0].startsWith("Ours, Dolly")).map(a => a[1].official_protocol.AUC - a[1].nested_cv.AUC_mean);
const dleak = dl.length ? dl.reduce((x, y) => x + y, 0) / dl.length : 0;
card(s, 7.0, 1.3, 2.5, 3.6);
T(s, [
  { text: `+${(offLog.official_protocol.AUC - offLog.nested_cv.AUC_mean).toFixed(3)} on RAGTruth`, options: { bold: true, fontSize: 18, color: C.accent2, breakLine: true } },
  { text: "small: the paper's claims hold", options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: `+${dleak.toFixed(2)} on Dolly (AC)`, options: { bold: true, fontSize: 18, color: C.accent4, breakLine: true } },
  { text: "same on all 3 models; 99 responses, so picking the best of 32 heads on test data fits noise" },
], { x: 7.2, y: 1.5, w: 2.15, h: 3.3, fontSize: 13 });
s.addNotes(`We first reproduced the authors' exact stored result (AUC ${offLog.official_protocol.AUC.toFixed(4)}) from their own logs, then moved feature and hyperparameter selection into training folds.`);

// 13 AARF ---------------------------------------------------------------
s = pres.addSlide({ masterName: "Content", sectionTitle: "Official code" });
s.addText("AARF: the gate is always on or almost never on, and its win is mostly position bias", { placeholder: "title" });
const arows = [[{ text: "Backbone", options: hdr({ align: "left" }) }, { text: "Answers changed", options: hdr() }, { text: "Gate fires", options: hdr() },
  { text: "Judge, official order", options: hdr() }, { text: "Judge, both orders", options: hdr() }]];
BIGM.forEach(([n, t]) => { const a = aarfStats(t); if (a) arows.push([cellO(n, { align: "left", bold: true }), cellO(`${a.changed}/${a.n}`),
  cellO(`${a.fire.toFixed(0)}%`), cellO(`${a.offA} : ${a.offB}`), cellO(`${a.bothA} : ${a.bothB}`)]); });
s.addTable(arows, { x: 0.5, y: 1.3, w: 6.2, colW: [1.5, 1.15, 1.0, 1.3, 1.25], fontFace: "Calibri", border: { type: "solid", pt: 0.5, color: "D0D7D6" },
  fill: { color: "FFFFFF" }, rowH: 0.5 });
T(s, "Judge columns: AARF wins : baseline wins (Gemini 3.5 Flash-Lite). Both orders = verdict must survive swapping the two answers.", { x: 0.5, y: 3.5, w: 6.2, h: 0.7, fontSize: 12, color: C.accent3 });
card(s, 7.0, 1.3, 2.5, 3.6);
T(s, [
  { text: "Gate miscalibrated", options: { bold: true, breakLine: true } },
  { text: "Released ranges make the 0.6 threshold always or never crossed.", options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "Position bias", options: { bold: true, breakLine: true } },
  { text: "Swapping the two answers flipped the judge's verdict in 39% (7B) and 47% (13B) of pairs." },
], { x: 7.2, y: 1.5, w: 2.15, h: 3.3, fontSize: 13 });
s.addNotes("AARF: amplify copying heads x1.2 and damp knowledge FFNs x0.8 when the token hallucination score exceeds 0.6. Authors judged with GPT-4o in a fixed order (AARF first).");

// 14 Limitations --------------------------------------------------------
s = pres.addSlide({ masterName: "Content", sectionTitle: "Official code" });
s.addText("Limitations to keep in mind", { placeholder: "title" });
[["4-bit weights", "3B, 7B, 8B and 13B ran quantised (laptop or free Kaggle GPU); 7B still matched the paper."],
 ["Small Dolly (AC)", "99 responses per model: fold-to-fold spread near 0.08-0.10 AUC."],
 ["Numbers only", "The contradiction test corrupts numbers; entity swaps and negations are untested."],
 ["Free-tier judges", "GPT judge is open-weights gpt-oss-120b; Gemini is Flash-Lite instead of GPT-4o."]]
  .forEach(([h, b], i) => {
    const x = 0.5 + (i % 2) * 4.6, y = 1.3 + Math.floor(i / 2) * 1.8;
    card(s, x, y, 4.4, 1.6);
    T(s, h, { x: x + 0.2, y: y + 0.15, w: 4.0, h: 0.4, bold: true, color: C.text2, fontSize: 16 });
    T(s, b, { x: x + 0.2, y: y + 0.6, w: 4.0, h: 0.9, fontSize: 14 });
  });
s.addNotes("Also: copying heads scored on 500 sampled vocabulary tokens; 415 responses give fold std of 0.03-0.05 AUC.");

// 15 Conclusion ---------------------------------------------------------
s = pres.addSlide({ masterName: "Title dark", sectionTitle: "Official code" });
s.addText("ReDeEP works, and now we know where it stops working", { placeholder: "title", fontSize: 32 });
s.addText(`Reproduced: AUC ${f3(l2.cv.AUC[0])} vs. 0.7325; chunk level matches the authors exactly
Strongest on Llama-2-13B; mixed on Llama-3-8B
Official evaluation leaks test labels: small on RAGTruth, large on Dolly (AC)
Blind to contradicted numbers; AARF's gain does not survive a fair judge`, { placeholder: "body", fontSize: 16 });
s.addNotes("Future work: entity and negation corruptions; combine ECS with a numeric consistency check; run at full precision on a larger GPU; score responses generated by each backbone.");

// 16 Links / thank you -------------------------------------------------
s = pres.addSlide({ masterName: "Title dark", sectionTitle: "Official code" });
s.addText("Thank you", { placeholder: "title" });
s.addText([
  { text: "Code, results and report: github.com/ad1thyaaa/ReDeEP-Reproduction-Robustness", options: { breakLine: true } },
  { text: "Base paper: Sun et al., ReDeEP, ICLR 2025 (doi.org/10.48550/arXiv.2410.11414)", options: { breakLine: true } },
  { text: "Official code and data: github.com/Jeryi-Sun/ReDEeP-ICLR  |  RAGTruth: github.com/ParticleMedia/RAGTruth" },
], { placeholder: "body", fontSize: 15 });
s.addNotes("Questions. All numbers in the report and slides are generated from the result files in the repository.");

(async () => {
  const out = process.env.OUT || path.join(__dirname, "ReDeEP_presentation.pptx");
  await pres.writeFile({ fileName: out });
  if (!process.env.NOTHEME) await applyTheme(out, THEME);
  console.log("wrote", out, "judges:", judges.map(j => j.name).join(", ") || "none yet");
})();
