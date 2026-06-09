import 'dotenv/config';
import fs from 'fs';

// ==========================================
// CONFIGURATION & INITIALIZATION
// ==========================================
const MOCK_TODAY = "2026-05-20"; // Wednesday anchor date
const DATASET_PATH = './synthetic_event_extraction_dataset.json';

// Safely load the dataset
let testSuite = [];
try {
  testSuite = JSON.parse(fs.readFileSync(DATASET_PATH, 'utf-8'));
} catch (err) {
  console.error(`💥 Failed to load dataset from ${DATASET_PATH}:`, err.message);
  process.exit(1);
}

// ==========================================
// HEURISTIC PREPROCESSING
// ==========================================
function appendDatesToDayNames(text, anchorDateStr) {
  const daysOfWeekMap = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday'];
  
  const [year, month, day] = anchorDateStr.split('-').map(Number);
  const anchorDate = new Date(year, month - 1, day);
  const anchorDayIndex = anchorDate.getDay();

  const dayRegex = /\b(next\s+)?(sunday|monday|tuesday|wednesday|thursday|friday|saturday)\b/gi;

  return text.replace(dayRegex, (match, nextModifier, dayName) => {
    const targetDayIndex = daysOfWeekMap.indexOf(dayName.toLowerCase());
    let daysDiff = targetDayIndex - anchorDayIndex;
    
    if (nextModifier) {
      daysDiff += 7; 
    } else {
      if (daysDiff <= 0) daysDiff += 7; 
    }

    const resolvedDate = new Date(anchorDate);
    resolvedDate.setDate(anchorDate.getDate() + daysDiff);

    const yyyy = resolvedDate.getFullYear();
    const mm = String(resolvedDate.getMonth() + 1).padStart(2, '0');
    const dd = String(resolvedDate.getDate()).padStart(2, '0');
    
    return `${match} (${yyyy}-${mm}-${dd})`;
  });
}

// ==========================================
// EXTRACTION ENGINE: OLLAMA (STRICT PARSING)
// ==========================================
async function extractWithOllama(text, today) {
  const daysOfWeekMap = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
  const [year, month, day] = today.split('-').map(Number);
  const dayNameToday = daysOfWeekMap[new Date(year, month - 1, day).getDay()];

  // Updated system prompt: Short, aggressive constraints optimized for 0.8B architectures
  const systemPrompt = `You are a strict data-extraction utility.
OUTPUT FORMAT: A single JSON array with one object inside.
CRITICAL LAWS:
1. NEVER output markdown blocks like \`\`\`json or \`\`\`.
2. Do NOT write any introduction or conclusion text. Start at [ and stop at ].
3. If no end time is present, calculate it exactly 1 hour after start_time.
4. Schema: [{"title": string, "date": "YYYY-MM-DD", "start_time": "HH:mm", "end_time": "HH:mm", "repeats": string|null}]`;

  const response = await fetch('http://localhost:11434/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'qwen-strict-parser', // 👈 Pointing to your custom Modelfile variant
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: text }
      ],
      format: 'json',  
      stream: false,  
      options: { 
        temperature: 0.0 // Locks the output to zero variation
      }
    })
  });

  if (!response.ok) throw new Error(`Ollama API error: ${response.statusText}`);

  const data = await response.json();
  let content = data.message?.content ? data.message.content.trim() : "";
  console.log(`[Ollama Raw Response]:`, content);

  if (!content) throw new Error('Ollama model returned no content');

  // SELF-HEALING REGEX: Strips markdown symbols if any slip past the template layer
  if (content.startsWith('```')) {
    content = content.replace(/^```json\s*/i, '').replace(/```$/, '').trim();
  }

  const parsedData = JSON.parse(content);
  const rawEvent = Array.isArray(parsedData) ? parsedData[0] : parsedData;
  
  return {
    id: rawEvent.id || null,
    summary: rawEvent.title || rawEvent.summary || null,
    date: rawEvent.date || null,
    start_time: rawEvent.start_time || null,
    end_time: rawEvent.end_time || null,
    repeats: rawEvent.repeats || null
  };
}

// ==========================================
// AUTOMATED TEST EXECUTION LOOP
// ==========================================
async function runTestSuite() {
  console.log(`🚀 Running accuracy tests against OLLAMA (qwen-strict-parser)...`);
  
  const normalizedSuite = testSuite.map((item, index) => {
    if (item.steps) return item;
    return {
      description: item.name || item.description || `Flat Prompt #${index + 1}`,
      steps: [{
        input: item.input,
        output: Array.isArray(item.output) ? item.output : [item.output]
      }]
    };
  });

  console.log(`Loaded ${normalizedSuite.length} test sequences.\n`);
  
  let totalPromptsTested = 0;
  let totalPromptsPassed = 0;

  for (let i = 0; i < normalizedSuite.length; i++) {
    const sequence = normalizedSuite[i];
    console.log(`------------------------------------------------------------`);
    console.log(`▶️ Running Test Case #${i + 1}: ${sequence.description}`);
    console.log(`------------------------------------------------------------`);
    
    for (const step of sequence.steps) {
      totalPromptsTested++;
      
      const preprocessedInput = appendDatesToDayNames(step.input, MOCK_TODAY);
      console.log(`\n[Input (Raw)]: "${step.input}"`);
      if (preprocessedInput !== step.input) {
        console.log(`[Input (Heuristic Added)]: "${preprocessedInput}"`);
      }
      
      try {
        const actual = await extractWithOllama(preprocessedInput, MOCK_TODAY);
        const expected = step.output[0];

        const expectedSummary = expected.summary || expected.title || "";
        const expectedBegin = expected.date || expected.begin_time || "";
        
        const expStart = expected.start_time || expected.begin_time || "00:00";
        const expEnd = expected.end_time || "00:00";
        const actStart = actual.start_time || "00:00";
        const actEnd = actual.end_time || "00:00";

        const summaryPass = actual.summary ? actual.summary.toLowerCase().trim() === expectedSummary.toLowerCase().trim() : false;
        const beginPass = actual.date ? actual.date.startsWith(expectedBegin) : false;
        const endPass = actual.end_time ? actual.end_time.startsWith(expEnd) : false;
        const repeatsPass = actual.repeats === expected.repeats;

        if (summaryPass && beginPass && endPass) {
          console.log(`✅ Step Passed!`);
          totalPromptsPassed++;
        } else {
          console.log(`❌ Step Failed! Reasons:`, { summaryPass, beginPass, endPass, repeatsPass });
          console.log(`   Expected:`, JSON.stringify(expected));
          console.log(`   Actual:  `, JSON.stringify(actual));
        }

      } catch (error) {
        console.error(`💥 Execution error on step: ${error.message}`);
      }
    }
  }

  const accuracy = totalPromptsTested > 0 ? ((totalPromptsPassed / totalPromptsTested) * 100).toFixed(2) : 0;
  console.log(`\n============================================================`);
  console.log(`📊 Final Results: ${totalPromptsPassed}/${totalPromptsTested} steps passed (${accuracy}% Accuracy)`);
  console.log(`============================================================`);
}

runTestSuite();