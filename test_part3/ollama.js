import fs from 'fs';

// ==========================================
// CONFIGURATION & INITIALIZATION
// ==========================================

const MOCK_TODAY = "2026-05-20"; // Wednesday anchor date
// const DATASET_PATH = './synthetic_event_extraction_dataset.json';
const DATASET_PATH = './max.json';

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

/**
 * Finds days of the week (including "next <day>") in a string 
 * and appends their calculated calendar dates.
 * Ex: "next monday" -> "next monday (2026-06-01)"
 */
function appendDatesToDayNames(text, anchorDateStr) {
  const daysOfWeekMap = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday'];
  
  const [year, month, day] = anchorDateStr.split('-').map(Number);
  const anchorDate = new Date(year, month - 1, day);
  const anchorDayIndex = anchorDate.getDay();

  // Updated Regex to catch:
  // 1. "day after tomorrow" or "dat"
  // 2. "tomorrow" or "tmrw"
  // 3. "next [day]" or just "[day]"
  const dayRegex = /\b(?:(day\s+after\s+tomorrow|dat)|(tomorrow|tmrw)|(?:(next\s+)?(sunday|monday|tuesday|wednesday|thursday|friday|saturday)))\b/gi;

  return text.replace(dayRegex, (match, datMatch, tomorrowMatch, nextModifier, dayName) => {
    let daysDiff = 0;

    // Handle "day after tomorrow" / "dat"
    if (datMatch) {
      daysDiff = 2;
    }
    // Handle "tomorrow" / "tmrw"
    else if (tomorrowMatch) {
      daysDiff = 1;
    } 
    // Handle standard day names
    else if (dayName) {
      const targetDayIndex = daysOfWeekMap.indexOf(dayName.toLowerCase());
      daysDiff = targetDayIndex - anchorDayIndex;
      
      if (nextModifier) {
        // If "next" is stated, look for the occurrence in the NEXT calendar week
        daysDiff += 7; 
      } else {
        // Passed or today -> means next week's occurrence
        if (daysDiff <= 0) daysDiff += 7; 
      }
    }

    // Calculate the target calendar date
    const resolvedDate = new Date(anchorDate);
    resolvedDate.setDate(anchorDate.getDate() + daysDiff);

    const yyyy = resolvedDate.getFullYear();
    const mm = String(resolvedDate.getMonth() + 1).padStart(2, '0');
    const dd = String(resolvedDate.getDate()).padStart(2, '0');
    
    return `${match} (${yyyy}-${mm}-${dd})`;
  });
}

// ==========================================
// EXTRACTION ENGINE: OLLAMA (QWEN)
// ==========================================
async function extractWithOllama(text, today) {
  // FIX 1: Define dayNameToday to prevent ReferenceError
  const daysOfWeekMap = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
  const [year, month, day] = today.split('-').map(Number);
  const dayNameToday = daysOfWeekMap[new Date(year, month - 1, day).getDay()];

  const systemPrompt = `Extract event details from the user's text. if event already exists, use the id mapping.
    
    IF no end time is provided, assume the event ends 1 hour after the start time. 
    If the input text contains an explicit date in parentheses, e.g., (YYYY-MM-DD), you MUST use this date for the "date" field, otherwise today's date is ${today}. 
    You MUST respond with a list containing a single JSON object containing ONLY these keys: 
    "id" (number), "title", "date" (Strict format: YYYY-MM-DD), "start_time" (Strict format: HH:mm 24-hour clock), "end_time" (Strict format: HH:mm 24-hour clock), "repeats" ("daily", "weekdays", "weekly", or null)
    Output as json, with only one event. DO NOT CHOOSE TIMES OUTSIDE FREE SLOTS.
    
    Match phrasing to hours: morning (08:00-11:30), midday/lunch (11:30-13:30), afternoon (13:00-17:30), 
    evening/end-of-day (18:00-24:00). Never default to midnight (00:00) unless requested. "repeats" is null unless explicitly recurring. `;

  const response = await fetch('http://localhost:11434/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'qwen3.5:2b',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: text }
      ],
      format: 'json',
      stream: false,  
      think: false,
      options: { temperature: 0.1, num_predict: 250 }
    })
  });

  if (!response.ok) throw new Error(`Ollama API error: ${response.statusText}`);

  const data = await response.json();
  const content = data.message?.content;
  console.log(`[Ollama Raw Response]:`, content);

  if (!content) throw new Error('Ollama model returned no content');
  const cleanContent = content
    .replace(/^```json\s*/i, '')  // Remove leading ```json
    .replace(/^```\s*/, '')       // Remove generic leading ```
    .replace(/\s*```$/, '')       // Remove trailing ```
    .trim();                      // Clear surrounding whitespace

  const parsedData = JSON.parse(cleanContent);
  const rawEvent = Array.isArray(parsedData) ? parsedData[0] : parsedData;
  
  const allDayRegex = /\ball[- ]?day\b/i;
  const isAllDay = allDayRegex.test(text); 
  if (rawEvent == undefined || !rawEvent.start_time) {
    console.log("FAILED");
    return {};
  }
  if (rawEvent.start_time === rawEvent.end_time) {
    // Split hours and minutes
    let [hours, minutes] = rawEvent.end_time.split(':').map(Number);
    
    // Add 60 minutes (1 hour)
    hours = (hours + 1) % 24; 
    
    // Format back to hh:mm string with padding
    rawEvent.end_time = `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`;
  }

  return {
    id: rawEvent.id || null,
    title: rawEvent.title || null,
    date: rawEvent.date || null,
    // If regex matches, forcefully override both times
    start_time: isAllDay ? "00:00" : (rawEvent.start_time || null),
    end_time: isAllDay ? "23:59" : (rawEvent.end_time || null), 
    repeats: rawEvent.repeats || null
  };
}

// ==========================================
// AUTOMATED TEST EXECUTION LOOP
// ==========================================
async function runTestSuite() {
  console.log(`🚀 Running accuracy tests against OLLAMA (qwen3:0.6b)...`);
  
  // Normalize dataset to ensure compatibility across both structural formats
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
      
      // Inject date preprocessor heuristic
      const preprocessedInput = appendDatesToDayNames(step.input, MOCK_TODAY);
      if (preprocessedInput !== step.input) {
        console.log(`[Input (Heuristic Added)]: "${preprocessedInput}"`);
      } else {
        console.log(`\n[Input (Raw)]: "${step.input}"`);

      }
      const actual = await extractWithOllama(preprocessedInput, MOCK_TODAY);

      
      // try {
        // const actual = await extractWithOllama(preprocessedInput, MOCK_TODAY);
      //   const expected = step.output[0];

      //   // ==========================================
      //   // CORRECTED EVALUATION SECTION
      //   // ==========================================

      //   // Handle property name fallbacks across schemas during evaluation
      //   const expectedSummary = expected.title || expected.title || "";
      //   const expectedDate = expected.date || expected.begin_time || ""; // Unified date fallback

      //   const expStart = expected.start_time || "00:00";
      //   const expEnd = expected.end_time || "00:00";
      //   const actStart = actual.start_time || "00:00";
      //   const actEnd = actual.end_time || "00:00";

      //   // Calculate durations
      //   const expectedDuration = (new Date(`1970-01-01T${expEnd}:00Z`) - new Date(`1970-01-01T${expStart}:00Z`)) / (1000 * 60); 
      //   const actualDuration = (new Date(`1970-01-01T${actEnd}:00Z`) - new Date(`1970-01-01T${actStart}:00Z`)) / (1000 * 60);

      //   // Corrected assertions mapping to your extraction engine properties
      //   const summaryPass = true;
      //   const beginPass = actual.date ? actual.date.startsWith(expectedDate) : false; // Changed from actual.begin_time to actual.date
      //   const endPass = actual.end_time ? actual.end_time.startsWith(expEnd) : false;
      //   const repeatsPass = actual.repeats === expected.repeats;
      //   console.log(actual.repeats, expected.repeats);
      //   if (summaryPass && beginPass && endPass && repeatsPass) {
      //     console.log(`✅ Step Passed!`);
      //     totalPromptsPassed++;
      //   } else {
      //     console.log(`❌ Step Failed! Reasons:`, { summaryPass, beginPass, endPass, repeatsPass });
      //     console.log(`   Expected:`, JSON.stringify(expected));
      //     console.log(`   Actual:  `, JSON.stringify(actual));
      //   }

      // } catch (error) {
      //   console.error(`💥 Execution error on step: ${error.message}`);
      // }
    }
  }

  const accuracy = totalPromptsTested > 0 ? ((totalPromptsPassed / totalPromptsTested) * 100).toFixed(2) : 0;
  console.log(`\n============================================================`);
  console.log(`📊 Final Results: ${totalPromptsPassed}/${totalPromptsTested} steps passed (${accuracy}% Accuracy)`);
  console.log(`============================================================`);
}

runTestSuite();
