const pptxgen = require("pptxgenjs");

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "OpenAI Codex";
pptx.company = "NC State FDS-Emerge";
pptx.subject = "FinQA project intro";
pptx.title = "FinQA Project Update";
pptx.lang = "en-US";
pptx.theme = {
  headFontFace: "Aptos Display",
  bodyFontFace: "Aptos",
  lang: "en-US",
};

const colors = {
  navy: "19324D",
  slate: "40566B",
  teal: "0E7490",
  gold: "C98A2E",
  bg: "F7F5F1",
  white: "FFFFFF",
  text: "1F2937",
  soft: "E8EEF3",
  green: "3A7D44",
};

function addTitle(slide, title, subtitle = "") {
  slide.addText(title, {
    x: 0.6, y: 0.35, w: 9.5, h: 0.5,
    fontFace: "Aptos Display", fontSize: 24, bold: true, color: colors.navy,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.6, y: 0.88, w: 11.5, h: 0.35,
      fontFace: "Aptos", fontSize: 10.5, color: colors.slate,
    });
  }
  slide.addShape(pptx.ShapeType.line, {
    x: 0.6, y: 1.22, w: 11.5, h: 0,
    line: { color: colors.gold, pt: 1.2 }
  });
}

function addBullets(slide, items, x, y, w, h, opts = {}) {
  const runs = [];
  items.forEach((t) => runs.push({ text: t, options: { bullet: { indent: 12 } } }));
  slide.addText(runs, {
    x, y, w, h,
    fontFace: "Aptos",
    fontSize: opts.fontSize || 16,
    color: opts.color || colors.text,
    breakLine: true,
    paraSpaceAfterPt: 12,
    valign: "top",
    margin: 0.06,
  });
}

function addFlowBox(slide, text, x, y, w, h, fill, fontColor = colors.text) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h,
    rectRadius: 0.06,
    fill: { color: fill },
    line: { color: fill, pt: 1 },
  });
  slide.addText(text, {
    x: x + 0.08, y: y + 0.11, w: w - 0.16, h: h - 0.18,
    align: "center", valign: "mid",
    fontFace: "Aptos", fontSize: 14, bold: true, color: fontColor,
  });
}

function addArrow(slide, x, y, w) {
  slide.addShape(pptx.ShapeType.chevron, {
    x, y, w, h: 0.32,
    fill: { color: colors.gold },
    line: { color: colors.gold, pt: 0.5 },
  });
}

// Slide 1
{
  const slide = pptx.addSlide();
  slide.background = { color: colors.bg };
  slide.addShape(pptx.ShapeType.rect, {
    x: 0, y: 0, w: 13.33, h: 7.5,
    fill: { color: colors.bg }, line: { color: colors.bg, pt: 0 }
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 0.7, y: 1.2, w: 0.25, h: 4.2,
    fill: { color: colors.teal }, line: { color: colors.teal, pt: 0 }
  });
  slide.addText("Financial Question Answering\nUsing FinQA", {
    x: 1.1, y: 1.3, w: 7.8, h: 1.2,
    fontFace: "Aptos Display", fontSize: 29, bold: true, color: colors.navy,
    breakLine: true,
  });
  slide.addText("Project update", {
    x: 1.1, y: 2.7, w: 3.5, h: 0.3, fontSize: 14, color: colors.gold, bold: true
  });
  slide.addText("Lasya Banka\nFDS-Emerge Summer Program", {
    x: 1.1, y: 3.15, w: 4.8, h: 0.8, fontSize: 18, color: colors.text, breakLine: true
  });
  slide.addText("Focus: retrieval-augmented finance QA with an evidence and audit perspective", {
    x: 1.1, y: 4.25, w: 6.2, h: 0.7, fontSize: 18, color: colors.slate, breakLine: true
  });
  slide.addNotes(`
[Speaker Notes]
This is a short introduction to my project idea. I am working on a financial question-answering project using the FinQA dataset. My focus is on retrieval-augmented generation, or RAG, with an evidence and audit perspective.
`);
}

// Slide 2
{
  const slide = pptx.addSlide();
  slide.background = { color: colors.white };
  addTitle(slide, "Dataset, API, And Framework Details");
  addBullets(slide, [
    "Dataset chosen: FinQA",
    "Source: official GitHub repository and associated paper",
    "Public splits available locally: Train 6,251 | Dev 883 | Test 1,147",
    "Each example includes text before the table, the financial table, text after the table, the question, the answer, and evidence fields."
  ], 0.8, 1.55, 6.1, 4.8);
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 7.25, y: 1.6, w: 5.0, h: 4.8,
    fill: { color: colors.soft }, line: { color: colors.soft, pt: 0.5 }
  });
  slide.addText("Technical stack", {
    x: 7.55, y: 1.9, w: 2.8, h: 0.3, fontSize: 18, bold: true, color: colors.navy
  });
  slide.addText("API used right now:\nNo paid external API currently\n\nFramework:\nHaystack\n\nEmbedding model:\nsentence-transformers/all-MiniLM-L6-v2\n\nOther tools:\nPython, pandas, GitHub dataset files", {
    x: 7.55, y: 2.35, w: 4.15, h: 3.5, fontSize: 16, color: colors.text, breakLine: true
  });
  slide.addNotes(`
[Speaker Notes]
I chose FinQA because it is fully public and finance-specific. It gives me both text and table data, which makes it a strong dataset for this problem.

For the technical setup, I am currently not using a paid external API. The main framework I am planning to use is Haystack for retrieval, and the embedding model is sentence-transformers/all-MiniLM-L6-v2. So this is an open-source and local pipeline at this stage.
`);
}

// Slide 3
{
  const slide = pptx.addSlide();
  slide.background = { color: colors.bg };
  addTitle(slide, "What Is The Project?", "RAG-based financial question answering");
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 0.8, y: 1.65, w: 4.2, h: 3.55,
    fill: { color: colors.white }, line: { color: colors.gold, pt: 1 }
  });
  slide.addText("RAG definition", {
    x: 1.1, y: 1.95, w: 2.3, h: 0.3, fontSize: 18, bold: true, color: colors.navy
  });
  slide.addText("Retrieval-Augmented Generation means the system first retrieves relevant information from source documents and then uses that retrieved information to help generate the answer.", {
    x: 1.1, y: 2.35, w: 3.45, h: 2.2, fontSize: 17, color: colors.text, breakLine: true
  });
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 5.45, y: 1.65, w: 6.75, h: 3.55,
    fill: { color: colors.navy }, line: { color: colors.navy, pt: 1 }
  });
  slide.addText("Project definition", {
    x: 5.8, y: 1.95, w: 3.8, h: 0.3, fontSize: 18, bold: true, color: colors.white
  });
  slide.addText("The project focuses on evidence-grounded financial question answering.\nFor each question, the system retrieves the most relevant text and table evidence from the source report associated with that example.\nThe retrieved evidence is then used to support answer generation, followed by a check on whether the answer is truly supported by the retrieved source material.", {
    x: 5.8, y: 2.32, w: 5.95, h: 2.55, fontSize: 16.2, color: colors.white, breakLine: true
  });
  slide.addNotes(`
[Speaker Notes]
This slide explains the core idea. RAG stands for Retrieval-Augmented Generation. Instead of making the model answer from memory alone, the system first retrieves relevant evidence from the source material.

My project is a finance version of that idea. In simple terms, someone asks a question about a financial document, the system searches the document, finds the most relevant parts, then answers using those parts, and later we can check whether the answer is actually supported by the evidence.
`);
}

// Slide 4
{
  const slide = pptx.addSlide();
  slide.background = { color: colors.white };
  addTitle(slide, "Questions Of Interest");
  addBullets(slide, [
    "Can a simple baseline retriever find the correct text or table evidence for a financial question?",
    "Does using table content help more than using text alone?",
    "How often does the retrieved evidence align with the dataset’s gold evidence fields?",
    "If I later add answer generation, can I label answers as supported, weakly supported, or unsupported?",
    "Can this project connect an audit mindset to LLM-based financial QA?"
  ], 0.9, 1.65, 11.1, 4.8, { fontSize: 18 });
  slide.addNotes(`
[Speaker Notes]
These are the main questions I want the project to answer. The first stage is retrieval: can a simple baseline find the correct evidence? Then I want to study whether tables are especially important in finance QA, whether the retrieved evidence matches the dataset’s gold evidence, and whether I can later support-check generated answers.
`);
}

// Slide 5
{
  const slide = pptx.addSlide();
  slide.background = { color: colors.bg };
  addTitle(slide, "Ways To Visualize The Dataset", "Plus the planned project flow");
  slide.addText("Example visualization", {
    x: 0.8, y: 1.55, w: 2.8, h: 0.3, fontSize: 18, bold: true, color: colors.navy
  });
  slide.addImage({
    path: "/Users/lasya/finance_audit_rag/outputs/finqa_split_bar_chart.svg",
    x: 0.75, y: 1.95, w: 5.6, h: 4.15
  });

  slide.addText("Project flow chart", {
    x: 6.9, y: 1.55, w: 2.6, h: 0.3, fontSize: 18, bold: true, color: colors.navy
  });
  addFlowBox(slide, "FinQA dataset", 6.95, 2.05, 1.65, 0.7, colors.soft);
  addArrow(slide, 8.72, 2.23, 0.45);
  addFlowBox(slide, "Split into text + table units", 9.2, 2.05, 2.1, 0.7, colors.soft);
  addArrow(slide, 11.42, 2.23, 0.45);
  addFlowBox(slide, "Embed and retrieve evidence", 6.95, 3.15, 2.2, 0.7, colors.teal, colors.white);
  addArrow(slide, 9.28, 3.33, 0.45);
  addFlowBox(slide, "Generate answer", 9.78, 3.15, 1.8, 0.7, colors.soft);
  addArrow(slide, 6.95, 4.42, 0.45);
  addFlowBox(slide, "Check supportability", 7.55, 4.25, 2.0, 0.7, colors.gold, colors.white);
  addArrow(slide, 9.68, 4.42, 0.45);
  addFlowBox(slide, "Output: answer + evidence", 10.28, 4.25, 2.15, 0.7, colors.green, colors.white);
  slide.addNotes(`
[Speaker Notes]
This slide shows two things. First, it includes an actual visualization of the FinQA split sizes: train, dev, and test. This gives a quick sense of dataset scale and class balance at the split level.

Second, it shows the project flow. I start from FinQA, break each example into searchable units, embed and retrieve evidence, later generate an answer, and then ideally check whether the answer is supported before producing the final output.
`);
}

// Slide 6
{
  const slide = pptx.addSlide();
  slide.background = { color: colors.white };
  slide.addShape(pptx.ShapeType.rect, {
    x: 0, y: 0, w: 13.33, h: 7.5,
    fill: { color: colors.white }, line: { color: colors.white, pt: 0 }
  });
  slide.addText("Thank You!", {
    x: 3.2, y: 2.0, w: 6.8, h: 0.8,
    align: "center", fontFace: "Aptos Display", fontSize: 28, bold: true, color: colors.navy
  });
  slide.addText("Questions or feedback are welcome.", {
    x: 3.4, y: 3.0, w: 6.4, h: 0.4,
    align: "center", fontSize: 18, color: colors.slate
  });
  slide.addText("FinQA + Haystack + Sentence Transformers", {
    x: 3.0, y: 4.05, w: 7.2, h: 0.4,
    align: "center", fontSize: 16, color: colors.gold, bold: true
  });
  slide.addNotes(`
[Speaker Notes]
Thank the audience and invite feedback, especially on scope, baseline design, and what would make the project most useful or realistic for the summer.
`);
}

pptx.writeFile({ fileName: "/Users/lasya/finance_audit_rag/FinQA_Project_Intro_Lasya.pptx" });
